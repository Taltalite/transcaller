#!/usr/bin/env python
"""
Bonito-Compatible Dataset Generation Pipeline (Spawn Mode).
Aligned with bonito/io.py logic:
1. Z-Score Normalization (Critical for training).
2. Outlier filtering (typical_indices).
3. Random shuffling.
4. Correct dtypes (float16, uint8).
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
import multiprocessing
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pod5
import pysam
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Global Constants
# --------------------------------------------------------------------------------------
BASE_TO_INT = {"A": 1, "C": 2, "G": 3, "T": 4}
PAD_VAL = 0
COMPLEMENT = str.maketrans("ACGT", "TGCA")

REFSEQ_TO_UCSC = {
    "NC_000001.11": "chr1",  "NC_000002.12": "chr2",  "NC_000003.12": "chr3",
    "NC_000004.12": "chr4",  "NC_000005.10": "chr5",  "NC_000006.12": "chr6",
    "NC_000007.14": "chr7",  "NC_000008.11": "chr8",  "NC_000009.12": "chr9",
    "NC_000010.11": "chr10", "NC_000011.10": "chr11", "NC_000012.12": "chr12",
    "NC_000013.11": "chr13", "NC_000014.9":  "chr14", "NC_000015.10": "chr15",
    "NC_000016.10": "chr16", "NC_000017.11": "chr17", "NC_000018.10": "chr18",
    "NC_000019.10": "chr19", "NC_000020.11": "chr20", "NC_000021.9":  "chr21",
    "NC_000022.11": "chr22", "NC_000023.11": "chrX",  "NC_000024.10": "chrY",
    "NC_012920.1":  "chrM",
}

def complement_base(base: str, is_reverse: bool) -> str:
    base = base.upper()
    if not is_reverse:
        return base
    return base.translate(COMPLEMENT)

# Ref: bonito/io.py
def typical_indices(x, n=2.5):
    """
    Filter out sequences that are too long or too short compared to the distribution.
    """
    mu, sd = np.mean(x), np.std(x)
    idx, = np.where((mu - n*sd < x) & (x < mu + n*sd))
    return idx

# --------------------------------------------------------------------------------------
# Data Structures
# --------------------------------------------------------------------------------------
@dataclass
class TaskData:
    read_id: str
    read_length: int
    reference_name: str
    is_reverse: bool
    ts_tag: int
    mv_tag: Sequence[int]
    read_to_ref_pairs: List[Tuple[int, int]]
    signal_len_arg: int
    stride_arg: int
    max_label_len: int

@dataclass
class Sample:
    signal: np.ndarray 
    label: np.ndarray  
    label_len: int

# --------------------------------------------------------------------------------------
# Worker State & LRU Cache
# --------------------------------------------------------------------------------------
global_fasta_handle = None
pod5_lookup: Dict[str, Tuple[str, int, int]] = {}
pod5_reader_cache: OrderedDict = None
MAX_OPEN_POD5_PER_WORKER = 32

def worker_init(fasta_path: str, lookup: Dict[str, Tuple[str, int, int]]):
    global global_fasta_handle
    global pod5_lookup
    global pod5_reader_cache
    os.environ["TMPDIR"] = "/tmp"
    try:
        global_fasta_handle = pysam.FastaFile(fasta_path)
        pod5_lookup = lookup
        pod5_reader_cache = OrderedDict()
    except Exception as e:
        print(f"[Worker Error] Init failed: {e}")
        raise e

def get_pod5_reader(path: str) -> pod5.Reader:
    global pod5_reader_cache
    if path in pod5_reader_cache:
        pod5_reader_cache.move_to_end(path)
        return pod5_reader_cache[path]
    
    if len(pod5_reader_cache) >= MAX_OPEN_POD5_PER_WORKER:
        _, reader_to_close = pod5_reader_cache.popitem(last=False)
        reader_to_close.close()
    
    try:
        new_reader = pod5.Reader(path)
        pod5_reader_cache[path] = new_reader
        return new_reader
    except Exception as e:
        time.sleep(0.1)
        new_reader = pod5.Reader(path)
        pod5_reader_cache[path] = new_reader
        return new_reader

# --------------------------------------------------------------------------------------
# Processing Logic
# --------------------------------------------------------------------------------------
def fetch_calibrated_signal(read_id: str) -> Tuple[np.ndarray, pod5.Read]:
    if read_id not in pod5_lookup:
        raise KeyError("Read ID missing")
    pod5_path, batch_idx, row_idx = pod5_lookup[read_id]
    reader = get_pod5_reader(pod5_path)
    batch = reader.get_batch(batch_idx)
    pod5_read = batch.get_read(row_idx)
    
    cal = pod5_read.calibration
    raw = pod5_read.signal
    pa_signal = (raw.astype(np.float32) + cal.offset) * cal.scale
    return pa_signal, pod5_read

def build_label_lookup(read_to_ref_pairs, reference_name, is_reverse):
    label_lookup = {}
    for read_pos, ref_pos in read_to_ref_pairs:
        try:
            base = global_fasta_handle.fetch(reference_name, ref_pos, ref_pos + 1)
        except Exception:
            continue
        if not base: continue
        base = complement_base(base, is_reverse)
        label_int = BASE_TO_INT.get(base)
        if label_int:
            label_lookup[read_pos] = label_int
    return label_lookup

def process_task(task: TaskData) -> Tuple[List[Sample], Dict[str, int]]:
    stats = {"valid_samples": 0}
    samples = []

    try:
        pa_signal, _ = fetch_calibrated_signal(task.read_id)
    except Exception as e:
        return samples, stats

    if pa_signal.shape[0] < task.signal_len_arg:
        return samples, stats

    # 1. 解析 Moves
    mv = np.asarray(task.mv_tag, dtype=np.int64)
    if mv.size <= 1: return samples, stats
    stride = mv[0]
    moves = mv[1:]
    base_indices = np.flatnonzero(moves)
    
    if base_indices.size > task.read_length:
        base_indices = base_indices[: task.read_length]
    
    ts_offset = int(task.ts_tag) if task.ts_tag is not None else 0
    base_positions = ts_offset + (base_indices * stride) + (stride // 2)
    
    # 2. 染色体名称修正
    fixed_ref_name = REFSEQ_TO_UCSC.get(task.reference_name, task.reference_name)
    label_lookup = build_label_lookup(task.read_to_ref_pairs, fixed_ref_name, task.is_reverse)
    if not label_lookup: return samples, stats

    valid_positions = []
    valid_labels = []
    
    for read_pos_idx in range(min(task.read_length, len(base_positions))):
        label = label_lookup.get(read_pos_idx)
        if label:
            valid_positions.append(base_positions[read_pos_idx])
            valid_labels.append(label)

    if not valid_positions: return samples, stats

    base_pos_arr = np.asarray(valid_positions)
    label_arr = np.asarray(valid_labels, dtype=np.int16)

    # 3. 滑动窗口切割
    chunk_len = task.signal_len_arg
    chunk_stride = task.stride_arg
    max_label = task.max_label_len

    for win_start in range(0, pa_signal.shape[0] - chunk_len + 1, chunk_stride):
        win_end = win_start + chunk_len
        
        left = np.searchsorted(base_pos_arr, win_start, side="left")
        right = np.searchsorted(base_pos_arr, win_end, side="left")

        if left >= right: continue
        label_seq = label_arr[left:right]
        
        if label_seq.size > max_label: continue
        if label_seq.size < 5: continue

        signal_window = pa_signal[win_start:win_end]

        # [CRITICAL STEP] Z-Score Normalization
        # (x - mean) / std. 
        # 为了数值稳定性，加一个小 epsilon
        sig_mean = np.mean(signal_window)
        sig_std = np.std(signal_window)
        if sig_std < 1e-5: continue # Skip flat signals
        
        normalized_signal = (signal_window - sig_mean) / sig_std

        # [ALIGNMENT] Bonito uses float16 for signals
        normalized_signal = normalized_signal.astype(np.float16)

        # Pad Labels
        padded_label = np.full((max_label,), PAD_VAL, dtype=np.uint8) # Bonito uses uint8 for labels
        padded_label[: label_seq.size] = label_seq.astype(np.uint8)

        samples.append(Sample(
            signal=normalized_signal, 
            label=padded_label,
            label_len=int(label_seq.size),
        ))
        stats["valid_samples"] += 1

    return samples, stats

# --------------------------------------------------------------------------------------
# IO & Merging
# --------------------------------------------------------------------------------------
def build_pod5_lookup(pod5_dir: str) -> Dict[str, Tuple[str, int, int]]:
    lookup = {}
    pod5_paths = []
    for root, dirs, files in os.walk(pod5_dir):
        for f in files:
            if f.endswith(".pod5"):
                pod5_paths.append(os.path.join(root, f))
                
    for pod5_path in tqdm(pod5_paths, desc="Indexing POD5"):
        try:
            with pod5.Reader(pod5_path) as reader:
                for batch_idx in range(reader.batch_count):
                    batch = reader.get_batch(batch_idx)
                    for row_idx in range(batch.num_reads):
                        read = batch.get_read(row_idx)
                        lookup[str(read.read_id)] = (pod5_path, batch_idx, row_idx)
        except Exception:
            pass
    return lookup

def flush_temp_chunk(chunk_id, chunk, temp_dir, manifest_list):
    if not chunk: return chunk_id
    
    signals = np.stack([s.signal for s in chunk], axis=0) # float16
    labels = np.stack([s.label for s in chunk], axis=0)   # uint8
    lengths = np.asarray([s.label_len for s in chunk], dtype=np.uint16) # Bonito uses uint16 for lengths

    sig_path = os.path.join(temp_dir, f"temp_sig_{chunk_id}.npy")
    lbl_path = os.path.join(temp_dir, f"temp_lbl_{chunk_id}.npy")
    len_path = os.path.join(temp_dir, f"temp_len_{chunk_id}.npy")

    np.save(sig_path, signals)
    np.save(lbl_path, labels)
    np.save(len_path, lengths)

    manifest_list.append({
        "signals": sig_path, "labels": lbl_path, "lengths": len_path,
        "num_samples": int(signals.shape[0])
    })
    return chunk_id + 1

def merge_chunks_to_final(output_dir, chunk_manifest, signal_len, max_label_len):
    if not chunk_manifest:
        print("No samples generated.")
        return

    print(f"[Merge] Loading {len(chunk_manifest)} batches into memory for filtering/shuffling...")

    # 1. Load EVERYTHING into memory first (necessary for global shuffling and filtering)
    # 200k samples * 4000 * 2 bytes (float16) ~ 1.6 GB RAM. This is safe.
    all_signals = []
    all_labels = []
    all_lengths = []

    for chunk_info in tqdm(chunk_manifest, desc="Loading Temps"):
        all_signals.append(np.load(chunk_info["signals"]))
        all_labels.append(np.load(chunk_info["labels"]))
        all_lengths.append(np.load(chunk_info["lengths"]))
        
        # Cleanup immediately
        os.remove(chunk_info["signals"])
        os.remove(chunk_info["labels"])
        os.remove(chunk_info["lengths"])

    # Concatenate
    signals = np.concatenate(all_signals, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    lengths = np.concatenate(all_lengths, axis=0)

    print(f"[Merge] Initial Pool: {signals.shape[0]} samples.")

    # 2. [ALIGNMENT] Filter outliers using typical_indices
    # Bonito removes reads with weird label lengths
    indices = typical_indices(lengths)
    print(f"[Merge] Retaining {len(indices)} samples after outlier filtering.")

    # 3. [ALIGNMENT] Random Permutation (Shuffling)
    # This is crucial so batches are not correlated
    indices = np.random.permutation(indices)

    # Apply selection
    signals = signals[indices]
    labels = labels[indices]
    lengths = lengths[indices]
    
    total_samples = signals.shape[0]

    # 4. Write Final Files (Memmap)
    print("[Merge] Writing final dataset...")
    final_sig_path = os.path.join(output_dir, "chunks.npy")
    final_lbl_path = os.path.join(output_dir, "references.npy")
    final_len_path = os.path.join(output_dir, "reference_lengths.npy")

    # Bonito Dtypes: float16 for signals, uint8 for labels, uint16 for lengths
    fp_sig = np.lib.format.open_memmap(final_sig_path, mode='w+', dtype=np.float16, shape=(total_samples, signal_len))
    fp_lbl = np.lib.format.open_memmap(final_lbl_path, mode='w+', dtype=np.uint8, shape=(total_samples, max_label_len))
    fp_len = np.lib.format.open_memmap(final_len_path, mode='w+', dtype=np.uint16, shape=(total_samples,))

    # Write in large blocks to save IO
    block_size = 10000
    for i in tqdm(range(0, total_samples, block_size), desc="Writing Disk"):
        end = min(i + block_size, total_samples)
        fp_sig[i:end] = signals[i:end]
        fp_lbl[i:end] = labels[i:end]
        fp_len[i:end] = lengths[i:end]

    del fp_sig, fp_lbl, fp_len
    print(f"Dataset ready at: {output_dir}")
    print(f"Final stats: {total_samples} chunks. Signal Mean ~0, Std ~1.")

def get_total_reads_from_index(bam_path):
    try:
        lines = pysam.idxstats(bam_path).splitlines()
        return sum(int(x.split('\t')[2]) for x in lines)
    except:
        return 0

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["TMPDIR"] = "/tmp" 
    
    parser = argparse.ArgumentParser(description="Bonito Data Prep (Aligned)")
    parser.add_argument("--bam_file", required=True)
    parser.add_argument("--pod5_dir", required=True)
    parser.add_argument("--reference_fasta", required=True)
    parser.add_argument("--output_dir", required=True)
    
    parser.add_argument("--chunk_len", type=int, default=4000)
    parser.add_argument("--stride", type=int, default=2000)
    parser.add_argument("--max_chunks", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=16)
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    print("[1/5] Building POD5 index...")
    lookup = build_pod5_lookup(args.pod5_dir)
    print(f"      Found {len(lookup)} reads.")
    
    print("[2/5] Scanning BAM for valid training chunks...")
    bam_file = pysam.AlignmentFile(args.bam_file, "rb")
    total_est = get_total_reads_from_index(args.bam_file)

    chunk_manifest = []
    chunk_buffer = []
    chunk_id = 0
    total_valid_samples = 0
    MAX_LABEL_LEN = 600 

    with ProcessPoolExecutor(max_workers=args.workers, initializer=worker_init, initargs=(args.reference_fasta, lookup)) as executor:
        futures = set()
        
        for read in tqdm(bam_file, total=total_est, unit="read", desc="Dispatching"):
            if args.max_chunks > 0 and total_valid_samples >= args.max_chunks:
                break

            if read.is_unmapped or not read.has_tag("mv") or read.query_name not in lookup:
                continue

            ts_tag = read.get_tag("ts") if read.has_tag("ts") else 0
            mv_tag = read.get_tag("mv")
            
            aligned_pairs = [(rp, ref_pos) for rp, ref_pos in read.get_aligned_pairs(matches_only=False) 
                             if rp is not None and ref_pos is not None]
            
            if not aligned_pairs: continue

            task = TaskData(
                read_id=read.query_name,
                read_length=read.query_length,
                reference_name=read.reference_name,
                is_reverse=read.is_reverse,
                ts_tag=ts_tag,
                mv_tag=mv_tag,
                read_to_ref_pairs=aligned_pairs,
                signal_len_arg=args.chunk_len,
                stride_arg=args.stride,
                max_label_len=MAX_LABEL_LEN 
            )

            futures.add(executor.submit(process_task, task))

            if len(futures) >= args.workers * 10: 
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                futures.difference_update(done)
                for future in done:
                    samples, stats = future.result()
                    chunk_buffer.extend(samples)
                    total_valid_samples += stats["valid_samples"]
                
                if len(chunk_buffer) >= 5000:
                    chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)
                    chunk_buffer = []

        for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing Workers"):
            samples, stats = future.result()
            chunk_buffer.extend(samples)
            total_valid_samples += stats["valid_samples"]
            if len(chunk_buffer) >= 5000:
                chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)
                chunk_buffer = []

    if chunk_buffer:
        chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)

    print("[3/5] Merging to final .npy files (with filtering & shuffling)...")
    merge_chunks_to_final(args.output_dir, chunk_manifest, args.chunk_len, MAX_LABEL_LEN)

    try: os.rmdir(temp_dir)
    except: pass