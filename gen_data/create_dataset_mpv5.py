#!/usr/bin/env python
"""
Generate Bonito-compatible CTC training data from a BAM + POD5 + reference FASTA.
Supports both 'quantile' and 'pa' (pico-ampere) normalization strategies.
"""
from __future__ import annotations

import argparse
import bisect
import multiprocessing
import os
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pod5
import pysam
from tqdm import tqdm
import shutil

BASE_TO_INT = {"A": 1, "C": 2, "G": 3, "T": 4}
COMPLEMENT = str.maketrans("ACGT", "TGCA")
MAX_OPEN_POD5_PER_WORKER = 32
EPS = 1e-5


def complement_base(base: str, is_reverse: bool) -> str:
    base = base.upper()
    if not is_reverse:
        return base
    return base.translate(COMPLEMENT)


def normalize_ref_name(name: str) -> str:
    name = name.lower()
    if name.startswith("chr"):
        name = name[3:]
    return name


def build_reference_mapping(
    bam_header: pysam.AlignmentHeader,
    fasta: pysam.FastaFile,
) -> Dict[str, str]:
    fasta_names = list(fasta.references)
    fasta_lengths = dict(zip(fasta.references, fasta.lengths))
    normalized = {}
    for name in fasta_names:
        key = normalize_ref_name(name)
        if key not in normalized:
            normalized[key] = name
        else:
            normalized[key] = None

    mapping = {}
    for bam_name in bam_header.references:
        if bam_name in fasta_names:
            mapping[bam_name] = bam_name
            continue

        norm_key = normalize_ref_name(bam_name)
        if norm_key in normalized and normalized[norm_key] is not None:
            mapping[bam_name] = normalized[norm_key]
            continue

        bam_len = bam_header.get_reference_length(bam_name)
        length_matches = [
            name for name, length in fasta_lengths.items() if length == bam_len
        ]
        if len(length_matches) == 1:
            mapping[bam_name] = length_matches[0]

    return mapping


def typical_indices(x: np.ndarray, n: float = 2.5) -> np.ndarray:
    if x.size == 0:
        return np.array([], dtype=np.int64)
    mu, sd = np.mean(x), np.std(x)
    idx, = np.where((mu - n*sd < x) & (x < mu + n*sd))
    return idx


def compute_norm_params(signal: np.ndarray, strategy: str, 
                        pa_mean: float, pa_std: float) -> Tuple[float, float]:
    """
    Calculate shift and scale.
    If strategy is 'pa', use fixed mean/std from config.
    If strategy is 'quantile', calculate from signal distribution.
    """
    if strategy == "pa":
        # [standardisation] standardise = 1 logic in Bonito
        return pa_mean, pa_std
    else:
        # Default Bonito quantile logic
        qa, qb = np.quantile(signal, [0.2, 0.9])
        shift = 0.51 * (qa + qb)
        scale = 0.53 * (qb - qa)
        if scale < EPS:
            scale = 1.0
        return shift, scale


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
    max_label_len: Optional[int]
    clip_value: float
    # Normalization params
    norm_strategy: str
    pa_mean: float
    pa_std: float


@dataclass
class Sample:
    signal: np.ndarray
    label: np.ndarray
    label_len: int


class WorkerState:
    fasta: Optional[pysam.FastaFile] = None
    pod5_lookup: Dict[str, Tuple[str, int, int]] = {}
    pod5_reader_cache: OrderedDict = OrderedDict()
    reference_mapping: Dict[str, str] = {}


def worker_init(
    fasta_path: str,
    lookup: Dict[str, Tuple[str, int, int]],
    reference_mapping: Dict[str, str],
) -> None:
    os.environ["TMPDIR"] = "/tmp"
    WorkerState.fasta = pysam.FastaFile(fasta_path)
    WorkerState.pod5_lookup = lookup
    WorkerState.reference_mapping = reference_mapping
    WorkerState.pod5_reader_cache = OrderedDict()


def get_pod5_reader(path: str) -> pod5.Reader:
    cache = WorkerState.pod5_reader_cache
    if path in cache:
        cache.move_to_end(path)
        return cache[path]

    if len(cache) >= MAX_OPEN_POD5_PER_WORKER:
        _, reader_to_close = cache.popitem(last=False)
        reader_to_close.close()

    try:
        new_reader = pod5.Reader(path)
    except Exception:
        time.sleep(0.1)
        new_reader = pod5.Reader(path)
    cache[path] = new_reader
    return new_reader


def fetch_calibrated_signal(read_id: str) -> Tuple[np.ndarray, pod5.Read]:
    if read_id not in WorkerState.pod5_lookup:
        raise KeyError("Read ID missing from POD5 index")
    pod5_path, batch_idx, row_idx = WorkerState.pod5_lookup[read_id]
    reader = get_pod5_reader(pod5_path)
    batch = reader.get_batch(batch_idx)
    pod5_read = batch.get_read(row_idx)
    cal = pod5_read.calibration
    raw = pod5_read.signal
    pa_signal = (raw.astype(np.float32) + cal.offset) * cal.scale
    return pa_signal, pod5_read


def build_label_lookup(
    read_to_ref_pairs: Iterable[Tuple[int, int]],
    reference_name: str,
    is_reverse: bool,
) -> Dict[int, int]:
    fasta = WorkerState.fasta
    label_lookup: Dict[int, int] = {}
    for read_pos, ref_pos in read_to_ref_pairs:
        try:
            base = fasta.fetch(reference_name, ref_pos, ref_pos + 1)
        except Exception:
            continue
        if not base:
            continue
        base = complement_base(base, is_reverse)
        label_int = BASE_TO_INT.get(base)
        if label_int:
            label_lookup[read_pos] = label_int
    return label_lookup


def process_task(task: TaskData) -> Tuple[List[Sample], Dict[str, int]]:
    stats = {"valid_samples": 0}
    samples: List[Sample] = []

    try:
        pa_signal, _ = fetch_calibrated_signal(task.read_id)
    except Exception:
        return samples, stats

    if pa_signal.shape[0] < task.signal_len_arg:
        return samples, stats

    # --- Compute Norm Params (PA or Quantile) ---
    global_shift, global_scale = compute_norm_params(
        pa_signal, 
        strategy=task.norm_strategy,
        pa_mean=task.pa_mean,
        pa_std=task.pa_std
    )
    # --------------------------------------------

    mv = np.asarray(task.mv_tag, dtype=np.int64)
    if mv.size <= 1:
        return samples, stats
    stride = int(mv[0])
    moves = mv[1:]
    base_indices = np.flatnonzero(moves)

    if base_indices.size > task.read_length:
        base_indices = base_indices[: task.read_length]

    ts_offset = int(task.ts_tag) if task.ts_tag is not None else 0
    base_positions = ts_offset + (base_indices * stride) + (stride // 2)

    reference_name = WorkerState.reference_mapping.get(task.reference_name)
    if not reference_name:
        return samples, stats

    label_lookup = build_label_lookup(task.read_to_ref_pairs, reference_name, task.is_reverse)
    if not label_lookup:
        return samples, stats

    valid_positions = []
    valid_labels = []
    for read_pos_idx in range(min(task.read_length, len(base_positions))):
        label = label_lookup.get(read_pos_idx)
        if label:
            valid_positions.append(base_positions[read_pos_idx])
            valid_labels.append(label)

    if not valid_positions:
        return samples, stats

    base_pos_arr = np.asarray(valid_positions)
    label_arr = np.asarray(valid_labels, dtype=np.int16)

    chunk_len = task.signal_len_arg
    chunk_stride = task.stride_arg

    for win_start in range(0, pa_signal.shape[0] - chunk_len + 1, chunk_stride):
        win_end = win_start + chunk_len

        left = np.searchsorted(base_pos_arr, win_start, side="left")
        right = np.searchsorted(base_pos_arr, win_end, side="left")

        if left >= right:
            continue
        label_seq = label_arr[left:right]
        if label_seq.size < 5:
            continue
        if task.max_label_len is not None and label_seq.size > task.max_label_len:
            continue

        signal_window = pa_signal[win_start:win_end]
        
        # Apply Normalization
        normalized_signal = (signal_window - global_shift) / global_scale
        
        normalized_signal = np.clip(normalized_signal, -task.clip_value, task.clip_value)
        normalized_signal = normalized_signal.astype(np.float16)

        samples.append(
            Sample(
                signal=normalized_signal,
                label=label_seq.astype(np.uint8),
                label_len=int(label_seq.size),
            )
        )
        stats["valid_samples"] += 1

    return samples, stats


def build_pod5_lookup(pod5_dir: str) -> Dict[str, Tuple[str, int, int]]:
    lookup: Dict[str, Tuple[str, int, int]] = {}
    pod5_paths = []
    for root, _, files in os.walk(pod5_dir):
        for name in files:
            if name.endswith(".pod5"):
                pod5_paths.append(os.path.join(root, name))

    for pod5_path in tqdm(pod5_paths, desc="Indexing POD5"):
        with pod5.Reader(pod5_path) as reader:
            for batch_idx in range(reader.batch_count):
                batch = reader.get_batch(batch_idx)
                for row_idx in range(batch.num_reads):
                    read = batch.get_read(row_idx)
                    lookup[str(read.read_id)] = (pod5_path, batch_idx, row_idx)

    return lookup


def flush_temp_chunk(
    chunk_id: int,
    chunk: List[Sample],
    temp_dir: str,
    manifest_list: List[Dict[str, object]],
) -> int:
    if not chunk:
        return chunk_id

    signals = np.stack([s.signal for s in chunk], axis=0)
    lengths = np.asarray([s.label_len for s in chunk], dtype=np.uint16)

    offsets = np.zeros(len(chunk) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(lengths, dtype=np.int64)
    labels_flat = np.concatenate([s.label for s in chunk], axis=0)

    sig_path = os.path.join(temp_dir, f"temp_sig_{chunk_id}.npy")
    lbl_path = os.path.join(temp_dir, f"temp_lbl_{chunk_id}.npy")
    off_path = os.path.join(temp_dir, f"temp_off_{chunk_id}.npy")
    len_path = os.path.join(temp_dir, f"temp_len_{chunk_id}.npy")

    np.save(sig_path, signals)
    np.save(lbl_path, labels_flat)
    np.save(off_path, offsets)
    np.save(len_path, lengths)

    manifest_list.append(
        {
            "signals": sig_path,
            "labels": lbl_path,
            "offsets": off_path,
            "lengths": len_path,
            "num_samples": int(signals.shape[0]),
        }
    )
    return chunk_id + 1


def build_chunk_ranges(chunk_manifest: List[Dict[str, object]]) -> List[int]:
    starts = [0]
    total = 0
    for chunk_info in chunk_manifest:
        total += int(chunk_info["num_samples"])
        starts.append(total)
    return starts


def find_chunk_index(starts: List[int], index: int) -> int:
    return bisect.bisect_right(starts, index) - 1


def merge_chunks_to_final(
    output_dir: str,
    chunk_manifest: List[Dict[str, object]],
    signal_len: int,
    max_label_len: Optional[int],
) -> None:
    if not chunk_manifest:
        print("No samples generated.")
        return

    lengths_list = []
    for chunk_info in tqdm(chunk_manifest, desc="Loading lengths"):
        lengths_list.append(np.load(chunk_info["lengths"]))
    lengths = np.concatenate(lengths_list, axis=0)

    indices = typical_indices(lengths)
    if indices.size == 0:
        print("No samples remained after outlier filtering.")
        return

    if max_label_len is None:
        max_label_len = int(lengths[indices].max())

    indices = np.random.permutation(indices)
    total_samples = int(indices.size)

    final_sig_path = os.path.join(output_dir, "chunks.npy")
    final_lbl_path = os.path.join(output_dir, "references.npy")
    final_len_path = os.path.join(output_dir, "reference_lengths.npy")

    fp_sig = np.lib.format.open_memmap(
        final_sig_path, mode="w+", dtype=np.float16, shape=(total_samples, signal_len)
    )
    fp_lbl = np.lib.format.open_memmap(
        final_lbl_path, mode="w+", dtype=np.uint8, shape=(total_samples, max_label_len)
    )
    fp_len = np.lib.format.open_memmap(
        final_len_path, mode="w+", dtype=np.uint16, shape=(total_samples,)
    )

    starts = build_chunk_ranges(chunk_manifest)
    chunk_cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def load_chunk_data(idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if idx in chunk_cache:
            return chunk_cache[idx]
        chunk_info = chunk_manifest[idx]
        signals = np.load(chunk_info["signals"], mmap_mode="r")
        labels = np.load(chunk_info["labels"], mmap_mode="r")
        offsets = np.load(chunk_info["offsets"], mmap_mode="r")
        lengths_chunk = np.load(chunk_info["lengths"], mmap_mode="r")
        chunk_cache[idx] = (signals, labels, offsets, lengths_chunk)
        return chunk_cache[idx]

    block_size = 2048
    for out_start in tqdm(range(0, total_samples, block_size), desc="Writing dataset"):
        out_end = min(out_start + block_size, total_samples)
        block_indices = indices[out_start:out_end]
        block_signals = np.empty((out_end - out_start, signal_len), dtype=np.float16)
        block_labels = np.zeros((out_end - out_start, max_label_len), dtype=np.uint8)
        block_lengths = lengths[block_indices].astype(np.uint16)

        for pos, global_index in enumerate(block_indices):
            chunk_idx = find_chunk_index(starts, int(global_index))
            local_index = int(global_index - starts[chunk_idx])
            signals, labels, offsets, lengths_chunk = load_chunk_data(chunk_idx)
            block_signals[pos] = signals[local_index]
            label_len = int(lengths_chunk[local_index])
            label_start = int(offsets[local_index])
            block_labels[pos, :label_len] = labels[label_start:label_start + label_len]

        fp_sig[out_start:out_end] = block_signals
        fp_lbl[out_start:out_end] = block_labels
        fp_len[out_start:out_end] = block_lengths

    del fp_sig, fp_lbl, fp_len

    for chunk_info in chunk_manifest:
        os.remove(chunk_info["signals"])
        os.remove(chunk_info["labels"])
        os.remove(chunk_info["offsets"])
        os.remove(chunk_info["lengths"])

    print(f"Dataset ready at: {output_dir}")
    print(f"Final stats: {total_samples} chunks.")


def get_total_reads_from_index(bam_path: str) -> int:
    try:
        lines = pysam.idxstats(bam_path).splitlines()
        return sum(int(x.split("\t")[2]) for x in lines)
    except Exception:
        return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bonito CTC dataset generation from BAM/POD5")
    parser.add_argument("--bam-file", required=True)
    parser.add_argument("--pod5-dir", required=True)
    parser.add_argument("--reference-fasta", required=True)
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--chunk-len", type=int, default=12000, help="Matches model config chunksize")
    parser.add_argument("--overlap", type=int, default=600, help="Matches model config overlap")
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--max-label-len", type=int, default=None)
    parser.add_argument("--clip-value", type=float, default=5.0)
    parser.add_argument("--max-chunks", type=int, default=-1)
    parser.add_argument("--workers", type=int, default=16)

    # NEW ARGS FOR PA STRATEGY
    parser.add_argument("--norm-strategy", choices=["quantile", "pa"], default="quantile",
                        help="Normalization strategy: 'quantile' (default) or 'pa' (fixed mean/std)")
    parser.add_argument("--pa-mean", type=float, default=0.0, help="Fixed mean for PA strategy")
    parser.add_argument("--pa-std", type=float, default=1.0, help="Fixed std for PA strategy")

    return parser.parse_args()


def main() -> None:
    multiprocessing.set_start_method("spawn", force=True)
    os.environ["TMPDIR"] = "/tmp"

    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    temp_dir = os.path.join(args.output_dir, "temp_chunks")
    os.makedirs(temp_dir, exist_ok=True)

    if args.stride is None:
        stride = args.chunk_len - args.overlap
    else:
        stride = args.stride
    if stride <= 0:
        raise ValueError("Stride must be positive. Check --chunk-len/--overlap/--stride.")
    
    print(f"Configuration:")
    print(f"  Chunk Size : {args.chunk_len}")
    print(f"  Overlap    : {args.overlap}")
    print(f"  Strategy   : {args.norm_strategy}")
    if args.norm_strategy == "pa":
        print(f"  PA Mean    : {args.pa_mean}")
        print(f"  PA Std     : {args.pa_std}")

    print("[1/5] Building POD5 index...")
    lookup = build_pod5_lookup(args.pod5_dir)
    print(f"      Found {len(lookup)} reads.")

    print("[2/5] Preparing reference mapping...")
    bam_file = pysam.AlignmentFile(args.bam_file, "rb")
    fasta = pysam.FastaFile(args.reference_fasta)
    reference_mapping = build_reference_mapping(bam_file.header, fasta)
    fasta.close()

    print("[3/5] Scanning BAM for valid training chunks...")
    if not os.path.exists(args.bam_file + ".bai") and not os.path.exists(args.bam_file + ".csi"):
        print("Warning: BAM index not found. Progress bar total will be inaccurate.")
        return 0
    
    total_est = get_total_reads_from_index(args.bam_file)

    chunk_manifest: List[Dict[str, object]] = []
    chunk_buffer: List[Sample] = []
    chunk_id = 0
    total_valid_samples = 0

    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=worker_init,
        initargs=(args.reference_fasta, lookup, reference_mapping),
    ) as executor:
        futures = set()

        for read in tqdm(bam_file, total=total_est, unit="read", desc="Dispatching"):
            if args.max_chunks > 0 and total_valid_samples >= args.max_chunks:
                break

            if read.is_unmapped or not read.has_tag("mv") or read.query_name not in lookup:
                continue

            ts_tag = read.get_tag("ts") if read.has_tag("ts") else 0
            mv_tag = read.get_tag("mv")

            aligned_pairs = [
                (rp, ref_pos)
                for rp, ref_pos in read.get_aligned_pairs(matches_only=False)
                if rp is not None and ref_pos is not None
            ]
            if not aligned_pairs:
                continue

            task = TaskData(
                read_id=read.query_name,
                read_length=read.query_length,
                reference_name=read.reference_name,
                is_reverse=read.is_reverse,
                ts_tag=ts_tag,
                mv_tag=mv_tag,
                read_to_ref_pairs=aligned_pairs,
                signal_len_arg=args.chunk_len,
                stride_arg=stride,
                max_label_len=args.max_label_len,
                clip_value=args.clip_value,
                # New normalization args
                norm_strategy=args.norm_strategy,
                pa_mean=args.pa_mean,
                pa_std=args.pa_std,
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

    print("[4/5] Merging to final .npy files (with Bonito outlier filtering & shuffling)...")
    merge_chunks_to_final(args.output_dir, chunk_manifest, args.chunk_len, args.max_label_len)

    print("Cleaning up temporary files...")
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        print(f"Warning: Failed to clean up temp dir: {e}")


if __name__ == "__main__":
    main()