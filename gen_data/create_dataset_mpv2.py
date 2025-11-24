#!/usr/bin/env python
"""Dataset generation pipeline for Dorado + Transcaller (Import-Time TMPDIR Fix).

Changes:
1. [ULTIMATE FIX] Parses --output_dir and sets TMPDIR *BEFORE* importing pod5.
   This ensures the underlying C++ Arrow library sees the new path during initialization.
2. Includes all v2 improvements:
   - Merging chunks to final .npy
   - MAX_LABEL_LEN = 300 / MIN_LABEL_LEN = 40
   - Signal Clipping (-5, 5)
   - --forward_only debug mode
"""

from __future__ import annotations

import sys
import os

# ==============================================================================
# [CRITICAL] Pre-import Environment Setup
# 必须在导入 pod5/numpy 之前设置 TMPDIR，否则 C++ 扩展可能已经锁定了 /tmp
# ==============================================================================
def _setup_tmpdir_pre_import():
    output_dir = None
    # 简单的参数预解析，找到 output_dir
    args = sys.argv
    for i, arg in enumerate(args):
        if arg == "--output_dir" and i + 1 < len(args):
            output_dir = args[i+1]
            break
    
    if output_dir:
        # 创建一个专属的临时目录在你的大硬盘上
        system_tmp = os.path.join(output_dir, "system_tmp_cache")
        try:
            os.makedirs(system_tmp, exist_ok=True)
            # 强制覆盖环境变量
            os.environ["TMPDIR"] = system_tmp
            # 限制 Arrow 线程数，防止文件句柄耗尽
            os.environ["ARROW_IO_THREADS"] = "2" 
            print(f"[System] 成功将 TMPDIR 重定向至: {system_tmp}")
            print(f"[System] 该操作在 import pod5 之前执行，确保生效。")
        except Exception as e:
            print(f"[Warning] 尝试设置 TMPDIR 失败: {e}")
    else:
        print("[Warning] 未在参数中找到 --output_dir，将使用系统默认 TMPDIR (可能导致空间不足)。")

_setup_tmpdir_pre_import()

# ==============================================================================
# Imports (Must be AFTER the setup above)
# ==============================================================================
import argparse
import json
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

# 延迟导入，确保它们看到新的环境变量
import numpy as np
import pod5
import pysam
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------
SIGNAL_LENGTH = 2048
WINDOW_STRIDE = 1024
MAX_LABEL_LEN = 300  
MIN_LABEL_LEN = 40   
CHUNK_SIZE = 2048 

BASE_TO_INT = {"A": 1, "C": 2, "G": 3, "T": 4}
PAD_VAL = 0
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def complement_base(base: str, is_reverse: bool) -> str:
    base = base.upper()
    if not is_reverse:
        return base
    return base.translate(COMPLEMENT)


def get_total_reads_from_index(bam_path: str) -> int:
    try:
        stats_str = pysam.idxstats(bam_path)
        total_count = 0
        for line in stats_str.splitlines():
            parts = line.split('\t')
            if len(parts) >= 3:
                total_count += int(parts[2])
        return total_count
    except Exception as e:
        print(f"[Warning] Could not read BAM index stats: {e}")
        return 0


@dataclass
class TaskData:
    read_id: str
    read_length: int
    reference_name: str
    is_reverse: bool
    ts_tag: int
    mv_tag: Sequence[int]
    read_to_ref_pairs: List[Tuple[int, int]]


@dataclass
class Sample:
    signal: np.ndarray
    label: np.ndarray
    label_len: int


# --------------------------------------------------------------------------------------
# Worker state
# --------------------------------------------------------------------------------------
global_fasta_handle = None
pod5_lookup: Dict[str, Tuple[str, int, int]] = {}
pod5_reader_cache: Dict[str, pod5.Reader] = {}

def worker_init(fasta_path: str, lookup: Dict[str, Tuple[str, int, int]]):
    # 虽然我们在主进程设置了 TMPDIR，但为了保险，在子进程初始化时再次确认
    # 注意：此时 output_dir 对应的 tmp 已经在 os.environ 里了（因为 fork 继承）
    global global_fasta_handle
    global pod5_lookup
    global_fasta_handle = pysam.FastaFile(fasta_path)
    pod5_lookup = lookup


# --------------------------------------------------------------------------------------
# Processing Logic
# --------------------------------------------------------------------------------------

def fetch_signal(read_id: str) -> Tuple[np.ndarray, pod5.Read]:
    if read_id not in pod5_lookup:
        raise KeyError("Read ID missing in POD5 index")
    pod5_path, batch_idx, row_idx = pod5_lookup[read_id]
    reader = pod5_reader_cache.get(pod5_path)
    if reader is None:
        reader = pod5.Reader(pod5_path)
        pod5_reader_cache[pod5_path] = reader
    batch = reader.get_batch(batch_idx)
    pod5_read = batch.get_read(row_idx)
    return pod5_read.signal.astype(np.float32), pod5_read


def build_label_lookup(read_to_ref_pairs, reference_name, is_reverse):
    label_lookup = {}
    for read_pos, ref_pos in read_to_ref_pairs:
        base = global_fasta_handle.fetch(reference_name, ref_pos, ref_pos + 1)
        if not base: continue
        base = complement_base(base, is_reverse)
        label_int = BASE_TO_INT.get(base)
        if label_int:
            label_lookup[read_pos] = label_int
    return label_lookup


def normalize_signal(signal_window):
    median = np.median(signal_window)
    mad = np.median(np.abs(signal_window - median))
    if mad == 0: raise ValueError("MAD is zero")
    norm = (signal_window - median) / mad
    # [v2 Feature] Signal Clipping
    return np.clip(norm, -5.0, 5.0)


def process_task(task: TaskData) -> Tuple[List[Sample], Dict[str, int]]:
    stats = {"valid_samples": 0, "skipped_short": 0, "skipped_long": 0}
    samples = []

    try:
        raw_signal, _ = fetch_signal(task.read_id)
    except KeyError:
        return samples, stats

    if raw_signal.shape[0] < SIGNAL_LENGTH:
        return samples, stats

    mv = np.asarray(task.mv_tag, dtype=np.int64)
    if mv.size <= 1: return samples, stats
    
    stride = mv[0]
    moves = mv[1:]
    base_indices = np.flatnonzero(moves)
    
    if base_indices.size < task.read_length: return samples, stats
    base_indices = base_indices[: task.read_length]
    
    ts_offset = int(task.ts_tag) if task.ts_tag is not None else 0
    base_positions = ts_offset + (base_indices * stride)

    label_lookup = build_label_lookup(task.read_to_ref_pairs, task.reference_name, task.is_reverse)
    if not label_lookup: return samples, stats

    valid_positions = []
    valid_labels = []
    for read_pos in range(task.read_length):
        label = label_lookup.get(read_pos)
        if label:
            valid_positions.append(base_positions[read_pos])
            valid_labels.append(label)

    if not valid_positions: return samples, stats

    base_pos_arr = np.asarray(valid_positions)
    label_arr = np.asarray(valid_labels, dtype=np.int16)
    search_positions = base_pos_arr

    for win_start in range(0, raw_signal.shape[0] - SIGNAL_LENGTH + 1, WINDOW_STRIDE):
        win_end = win_start + SIGNAL_LENGTH
        left = np.searchsorted(search_positions, win_start, side="left")
        right = np.searchsorted(search_positions, win_end, side="left")

        if left >= right: continue

        label_seq = label_arr[left:right]
        
        # [v2 Feature] Length Filtering
        if label_seq.size < MIN_LABEL_LEN:
            stats["skipped_short"] += 1
            continue
        if label_seq.size > MAX_LABEL_LEN:
            stats["skipped_long"] += 1
            continue

        signal_window = raw_signal[win_start:win_end]
        try:
            norm_signal = normalize_signal(signal_window)
        except ValueError: continue

        padded_label = np.full((MAX_LABEL_LEN,), PAD_VAL, dtype=np.int16)
        padded_label[: label_seq.size] = label_seq

        samples.append(Sample(
            signal=norm_signal.reshape(1, SIGNAL_LENGTH).astype(np.float32),
            label=padded_label,
            label_len=int(label_seq.size),
        ))
        stats["valid_samples"] += 1

    return samples, stats


# --------------------------------------------------------------------------------------
# POD5 Lookup
# --------------------------------------------------------------------------------------
def build_pod5_lookup(pod5_dir: str) -> Dict[str, Tuple[str, int, int]]:
    lookup = {}
    pod5_paths = [os.path.join(pod5_dir, f) for f in os.listdir(pod5_dir) if f.endswith(".pod5")]
    for pod5_path in tqdm(pod5_paths, desc="Indexing POD5"):
        with pod5.Reader(pod5_path) as reader:
            for batch_idx in range(reader.batch_count):
                batch = reader.get_batch(batch_idx)
                for row_idx in range(batch.num_reads):
                    read = batch.get_read(row_idx)
                    lookup[str(read.read_id)] = (pod5_path, batch_idx, row_idx)
    return lookup


# --------------------------------------------------------------------------------------
# Temporary Chunk Writer
# --------------------------------------------------------------------------------------
def flush_temp_chunk(
    chunk_id: int,
    chunk: List[Sample],
    temp_dir: str,
    manifest_list: List[Dict],
) -> int:
    if not chunk:
        return chunk_id

    signals = np.stack([sample.signal for sample in chunk], axis=0)
    labels = np.stack([sample.label for sample in chunk], axis=0)
    lengths = np.asarray([sample.label_len for sample in chunk], dtype=np.int16)

    sig_path = os.path.join(temp_dir, f"temp_sig_{chunk_id}.npy")
    lbl_path = os.path.join(temp_dir, f"temp_lbl_{chunk_id}.npy")
    len_path = os.path.join(temp_dir, f"temp_len_{chunk_id}.npy")

    np.save(sig_path, signals)
    np.save(lbl_path, labels)
    np.save(len_path, lengths)

    manifest_list.append({
        "signals": sig_path,
        "labels": lbl_path,
        "lengths": len_path,
        "num_samples": int(signals.shape[0])
    })
    return chunk_id + 1


# --------------------------------------------------------------------------------------
# Merge Function
# --------------------------------------------------------------------------------------
def merge_chunks_to_final(
    output_dir: str, 
    chunk_manifest: List[Dict]
) -> None:
    if not chunk_manifest:
        print("No samples generated. Skipping merge.")
        return

    total_samples = sum(item["num_samples"] for item in chunk_manifest)
    print(f"[Merge] Merging {len(chunk_manifest)} chunks ({total_samples} samples total)...")

    final_sig_path = os.path.join(output_dir, "signals.npy")
    final_lbl_path = os.path.join(output_dir, "labels.npy")
    final_len_path = os.path.join(output_dir, "lengths.npy")

    fp_sig = np.lib.format.open_memmap(
        final_sig_path, mode='w+', dtype=np.float32, shape=(total_samples, 1, SIGNAL_LENGTH)
    )
    fp_lbl = np.lib.format.open_memmap(
        final_lbl_path, mode='w+', dtype=np.int16, shape=(total_samples, MAX_LABEL_LEN)
    )
    fp_len = np.lib.format.open_memmap(
        final_len_path, mode='w+', dtype=np.int16, shape=(total_samples,)
    )

    current_idx = 0
    for chunk_info in tqdm(chunk_manifest, desc="Merging Chunks"):
        n = chunk_info["num_samples"]
        
        c_sig = np.load(chunk_info["signals"])
        c_lbl = np.load(chunk_info["labels"])
        c_len = np.load(chunk_info["lengths"])

        fp_sig[current_idx : current_idx + n] = c_sig
        fp_lbl[current_idx : current_idx + n] = c_lbl
        fp_len[current_idx : current_idx + n] = c_len

        current_idx += n

        os.remove(chunk_info["signals"])
        os.remove(chunk_info["labels"])
        os.remove(chunk_info["lengths"])

    fp_sig.flush()
    fp_lbl.flush()
    fp_len.flush()

    print(f"[Merge] Successfully created:\n  - {final_sig_path}\n  - {final_lbl_path}\n  - {final_len_path}")


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    print("[1/5] Building POD5 index…")
    lookup = build_pod5_lookup(args.pod5_dir)
    
    print("[2/5] Opening BAM file…")
    total_reads_est = get_total_reads_from_index(args.bam_file)
    if total_reads_est > 0:
        print(f"Estimated reads: {total_reads_est}")
    
    bam_file = pysam.AlignmentFile(args.bam_file, "rb")

    print("[3/5] Initializing workers…")
    max_workers = args.workers
    max_queue_size = max_workers * 8
    
    chunk_manifest: List[Dict] = []
    chunk_buffer: List[Sample] = []
    chunk_id = 0
    total_stats = {"valid_samples": 0, "skipped_short": 0, "skipped_long": 0}
    
    start_time = time.time()

    if args.forward_only:
        print("⚠️ WARNING: Running in FORWARD-ONLY mode. Reverse reads will be skipped!")

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init,
        initargs=(args.reference_fasta, lookup),
    ) as executor:
        futures = set()
        iterator = tqdm(bam_file, desc="Scanning BAM", total=total_reads_est, unit="read", dynamic_ncols=True)

        for read in iterator:
            if read.is_unmapped or not read.has_tag("mv"): continue
            if read.query_name not in lookup: continue
            
            if args.forward_only and read.is_reverse:
                continue

            ts_tag = read.get_tag("ts") if read.has_tag("ts") else 0
            mv_tag = read.get_tag("mv")
            aligned_pairs = [(rp, ref_pos) for rp, ref_pos in read.get_aligned_pairs(matches_only=False) if rp is not None and ref_pos is not None]
            
            if not aligned_pairs: continue

            task = TaskData(
                read_id=read.query_name,
                read_length=read.query_length,
                reference_name=read.reference_name,
                is_reverse=read.is_reverse,
                ts_tag=ts_tag,
                mv_tag=mv_tag,
                read_to_ref_pairs=aligned_pairs,
            )

            futures.add(executor.submit(process_task, task))

            while len(futures) >= max_queue_size:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                futures.difference_update(done)
                for future in done:
                    samples, stats = future.result()
                    chunk_buffer.extend(samples)
                    total_stats["valid_samples"] += stats["valid_samples"]
                    total_stats["skipped_short"] += stats["skipped_short"]
                    total_stats["skipped_long"] += stats["skipped_long"]
                if len(chunk_buffer) >= CHUNK_SIZE:
                    chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)
                    chunk_buffer = []

        print("[4/5] Waiting for remaining workers…")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing"):
            samples, stats = future.result()
            chunk_buffer.extend(samples)
            total_stats["valid_samples"] += stats["valid_samples"]
            total_stats["skipped_short"] += stats["skipped_short"]
            total_stats["skipped_long"] += stats["skipped_long"]
            if len(chunk_buffer) >= CHUNK_SIZE:
                chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)
                chunk_buffer = []

    if chunk_buffer:
        chunk_id = flush_temp_chunk(chunk_id, chunk_buffer, temp_dir, chunk_manifest)

    bam_file.close()

    print(f"Stats: Valid={total_stats['valid_samples']}, "
          f"Skipped(Short<40)={total_stats['skipped_short']}, "
          f"Skipped(Long>300)={total_stats['skipped_long']}")

    print("[5/5] Merging chunks into final dataset…")
    merge_chunks_to_final(args.output_dir, chunk_manifest)

    # Cleanup
    # 注意：我们在脚本开头设置的 system_tmp_cache 也会被自动清理吗？
    # 最好在这里显式清理一下，或者留着也没关系，它在 output_dir 里
    try:
        os.rmdir(temp_dir)
    except OSError:
        pass

    manifest_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(manifest_path, "w") as f:
        json.dump({
            "signal_length": SIGNAL_LENGTH,
            "num_samples": total_stats["valid_samples"],
            "files": ["signals.npy", "labels.npy", "lengths.npy"],
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Done. Output located in: {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate merged NumPy dataset")
    parser.add_argument("--bam_file", required=True)
    parser.add_argument("--pod5_dir", required=True)
    parser.add_argument("--reference_fasta", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--forward_only", action="store_true", 
                        help="Debug mode: Only process reads mapped to forward strand.")
    args = parser.parse_args()
    main(args)