#!/usr/bin/env python
"""Dataset generation pipeline for Dorado + Transcaller.

This script consumes a BAM file that contains Dorado basecalls produced with
``--emit-moves`` and produces a NumPy dataset (chunked ``.npy`` files).  Each
sample contains a normalized 2048-sample current window and the aligned base
labels that overlap that window.

Compared to the previous implementation, this version:
* uses ``AlignedSegment.get_aligned_pairs`` to derive the reference base for
  each base in the read, so insertions/deletions are handled correctly;
* keeps the labels in read orientation (reverse-complementing when needed);
* stores the output as NumPy chunks instead of HDF5.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pod5
import pysam
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Configuration defaults (can be overridden via CLI)
# --------------------------------------------------------------------------------------
SIGNAL_LENGTH = 2048
WINDOW_STRIDE = 1024
MAX_LABEL_LEN = 200
CHUNK_SIZE = 2048

BASE_TO_INT = {"A": 1, "C": 2, "G": 3, "T": 4}
PAD_VAL = 0
COMPLEMENT = str.maketrans("ACGT", "TGCA")


def complement_base(base: str, is_reverse: bool) -> str:
    base = base.upper()
    if not is_reverse:
        return base
    return base.translate(COMPLEMENT)


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
    global global_fasta_handle
    global pod5_lookup
    global_fasta_handle = pysam.FastaFile(fasta_path)
    pod5_lookup = lookup


# --------------------------------------------------------------------------------------
# Core per-read processing
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
    if pod5_read is None:
        raise KeyError("Read not found in POD5 file")

    return pod5_read.signal.astype(np.float32), pod5_read


def build_label_lookup(
    read_to_ref_pairs: Sequence[Tuple[int, int]],
    reference_name: str,
    is_reverse: bool,
) -> Dict[int, int]:
    label_lookup: Dict[int, int] = {}
    for read_pos, ref_pos in read_to_ref_pairs:
        base = global_fasta_handle.fetch(reference_name, ref_pos, ref_pos + 1)
        if not base:
            continue
        base = complement_base(base, is_reverse)
        label_int = BASE_TO_INT.get(base)
        if label_int is None:
            continue
        label_lookup[read_pos] = label_int
    return label_lookup


def normalize_signal(signal_window: np.ndarray) -> np.ndarray:
    median = np.median(signal_window)
    mad = np.median(np.abs(signal_window - median))
    if mad == 0:
        raise ValueError("MAD is zero")
    return (signal_window - median) / mad


def process_task(task: TaskData) -> Tuple[List[Sample], Dict[str, int]]:
    stats = {
        "id_not_in_pod5_index": 0,
        "read_not_found": 0,
        "missing_mv": 0,
        "missing_signal": 0,
        "mv_length_mismatch": 0,
        "signal_too_short": 0,
        "mad_is_zero": 0,
        "window_no_bases": 0,
        "window_label_too_long": 0,
        "valid_samples": 0,
    }

    samples: List[Sample] = []

    try:
        raw_signal, _ = fetch_signal(task.read_id)
    except KeyError:
        stats["id_not_in_pod5_index"] += 1
        return samples, stats

    if raw_signal.shape[0] < SIGNAL_LENGTH:
        stats["signal_too_short"] += 1
        return samples, stats

    mv = np.asarray(task.mv_tag, dtype=np.int64)
    if mv.size <= 1:
        stats["missing_mv"] += 1
        return samples, stats

    stride = mv[0]
    moves = mv[1:]
    base_indices = np.flatnonzero(moves)
    if base_indices.size < task.read_length:
        stats["mv_length_mismatch"] += 1
        return samples, stats

    base_indices = base_indices[: task.read_length]
    ts_offset = int(task.ts_tag) if task.ts_tag is not None else 0
    base_positions = ts_offset + (base_indices * stride)

    label_lookup = build_label_lookup(
        task.read_to_ref_pairs, task.reference_name, task.is_reverse
    )

    if not label_lookup:
        stats["window_no_bases"] += 1
        return samples, stats

    valid_positions = []
    valid_labels = []
    for read_pos in range(task.read_length):
        label = label_lookup.get(read_pos)
        if label is None:
            continue
        valid_positions.append(base_positions[read_pos])
        valid_labels.append(label)

    if not valid_positions:
        stats["window_no_bases"] += 1
        return samples, stats

    base_pos_arr = np.asarray(valid_positions)
    label_arr = np.asarray(valid_labels, dtype=np.int16)

    search_positions = base_pos_arr

    for win_start in range(0, raw_signal.shape[0] - SIGNAL_LENGTH + 1, WINDOW_STRIDE):
        win_end = win_start + SIGNAL_LENGTH
        left = np.searchsorted(search_positions, win_start, side="left")
        right = np.searchsorted(search_positions, win_end, side="left")

        if left >= right:
            stats["window_no_bases"] += 1
            continue

        label_seq = label_arr[left:right]
        if label_seq.size > MAX_LABEL_LEN:
            stats["window_label_too_long"] += 1
            continue

        signal_window = raw_signal[win_start:win_end]
        try:
            norm_signal = normalize_signal(signal_window)
        except ValueError:
            stats["mad_is_zero"] += 1
            continue

        padded_label = np.full((MAX_LABEL_LEN,), PAD_VAL, dtype=np.int16)
        padded_label[: label_seq.size] = label_seq

        samples.append(
            Sample(
                signal=norm_signal.reshape(1, SIGNAL_LENGTH).astype(np.float32),
                label=padded_label,
                label_len=int(label_seq.size),
            )
        )
        stats["valid_samples"] += 1

    return samples, stats


# --------------------------------------------------------------------------------------
# POD5 lookup helper
# --------------------------------------------------------------------------------------

def build_pod5_lookup(pod5_dir: str) -> Dict[str, Tuple[str, int, int]]:
    lookup: Dict[str, Tuple[str, int, int]] = {}
    pod5_paths = [
        os.path.join(pod5_dir, f)
        for f in os.listdir(pod5_dir)
        if f.endswith(".pod5")
    ]
    for pod5_path in tqdm(pod5_paths, desc="Indexing POD5"):
        with pod5.Reader(pod5_path) as reader:
            for batch_idx in range(reader.batch_count):
                batch = reader.get_batch(batch_idx)
                for row_idx in range(batch.num_reads):
                    read = batch.get_read(row_idx)
                    lookup[str(read.read_id)] = (pod5_path, batch_idx, row_idx)
    return lookup


# --------------------------------------------------------------------------------------
# Chunk writer
# --------------------------------------------------------------------------------------

def flush_chunk(
    chunk_id: int,
    chunk: List[Sample],
    output_dir: str,
    manifest: List[Dict[str, str]],
) -> int:
    if not chunk:
        return chunk_id

    signals = np.stack([sample.signal for sample in chunk], axis=0)
    labels = np.stack([sample.label for sample in chunk], axis=0)
    lengths = np.asarray([sample.label_len for sample in chunk], dtype=np.int16)

    signal_path = os.path.join(output_dir, f"signals_chunk_{chunk_id:05d}.npy")
    label_path = os.path.join(output_dir, f"labels_chunk_{chunk_id:05d}.npy")
    length_path = os.path.join(output_dir, f"lengths_chunk_{chunk_id:05d}.npy")

    np.save(signal_path, signals)
    np.save(label_path, labels)
    np.save(length_path, lengths)

    manifest.append(
        {
            "signals": os.path.basename(signal_path),
            "labels": os.path.basename(label_path),
            "lengths": os.path.basename(length_path),
            "num_samples": int(signals.shape[0]),
        }
    )

    return chunk_id + 1


# --------------------------------------------------------------------------------------
# Main driver
# --------------------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] Building POD5 index…")
    lookup = build_pod5_lookup(args.pod5_dir)
    print(f"Indexed {len(lookup)} reads from POD5 files.")

    print("[2/5] Opening BAM file…")
    bam_file = pysam.AlignmentFile(args.bam_file, "rb")

    print("[3/5] Initializing workers…")
    max_workers = args.workers
    max_queue_size = max_workers * 8

    total_stats = {
        "bam_reads": 0,
        "tasks_submitted": 0,
        "valid_samples": 0,
    }

    manifest: List[Dict[str, str]] = []
    chunk_buffer: List[Sample] = []
    chunk_id = 0
    start_time = time.time()

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=worker_init,
        initargs=(args.reference_fasta, lookup),
    ) as executor:
        futures = set()
        iterator = tqdm(bam_file, desc="Scanning BAM")

        for read in iterator:
            total_stats["bam_reads"] += 1

            if read.is_unmapped or not read.has_tag("mv"):
                continue
            if read.query_name not in lookup:
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
            )

            futures.add(executor.submit(process_task, task))
            total_stats["tasks_submitted"] += 1

            while len(futures) >= max_queue_size:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                futures.difference_update(done)
                for future in done:
                    samples, stats = future.result()
                    chunk_buffer.extend(samples)
                    total_stats["valid_samples"] += stats["valid_samples"]
                if len(chunk_buffer) >= CHUNK_SIZE:
                    chunk_id = flush_chunk(chunk_id, chunk_buffer, args.output_dir, manifest)
                    chunk_buffer = []

        print("[4/5] Waiting for remaining workers…")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Finishing"):
            samples, stats = future.result()
            chunk_buffer.extend(samples)
            total_stats["valid_samples"] += stats["valid_samples"]
            if len(chunk_buffer) >= CHUNK_SIZE:
                chunk_id = flush_chunk(chunk_id, chunk_buffer, args.output_dir, manifest)
                chunk_buffer = []

    if chunk_buffer:
        chunk_id = flush_chunk(chunk_id, chunk_buffer, args.output_dir, manifest)

    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "signal_length": SIGNAL_LENGTH,
                "max_label_len": MAX_LABEL_LEN,
                "chunk_size": CHUNK_SIZE,
                "num_chunks": len(manifest),
                "chunks": manifest,
                "stats": total_stats,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
            f,
            indent=2,
        )

    bam_file.close()
    elapsed = time.time() - start_time
    print("[5/5] Done.")
    print(f"Total valid samples: {total_stats['valid_samples']}")
    print(f"Chunks written: {len(manifest)}")
    print(f"Manifest saved to: {manifest_path}")
    print(f"Elapsed time: {elapsed / 60:.2f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate NumPy dataset from Dorado BAM + POD5")
    parser.add_argument("--bam_file", required=True, help="Sorted/Indexed BAM with Dorado mv/ts tags")
    parser.add_argument("--pod5_dir", required=True, help="Directory with POD5 files")
    parser.add_argument("--reference_fasta", required=True, help="Reference FASTA used for alignment")
    parser.add_argument("--output_dir", required=True, help="Destination directory for NumPy chunks")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    args = parser.parse_args()

    main(args)
