#!/usr/bin/env python3
"""
Fixed generation of Bonito-compatible --save-ctc data.
Fixes:
1. Replaced bonito.RejectCounter with collections.Counter (fixes KeyError/AttributeError).
2. Added explicit UUID -> String conversion for read IDs (fixes mismatch).
3. Kept global aligner optimization.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, NamedTuple

# Adjust path to find bonito
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mappy as mp
import numpy as np
import pysam
from tqdm import tqdm

from bonito.io import typical_indices
from bonito.pod5 import get_reads as get_pod5_reads
from bonito.util import mean_qscore_from_qstring

# Global config and aligner
_GLOBAL_ALIGNER: Optional[mp.Aligner] = None
_THREAD_CONFIG: Dict[str, Any] = {}
BASE_TO_INT = {"A": 1, "C": 2, "G": 3, "T": 4}

class AlnInfo(NamedTuple):
    """Lightweight structure to pass alignment data to workers."""
    query_sequence: str
    qual: Optional[str]
    cigartuples: List
    reference_name: str
    reference_start: int
    is_reverse: bool
    mapping_quality: int
    mv_tag: Any
    ts_tag: int

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replicate bonito --save-ctc (Fixed Version).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--bam", required=True, help="Aligned Dorado BAM")
    parser.add_argument("--pod5", required=True, help="Directory containing source pod5 reads")
    parser.add_argument("--reference", required=True, help="Reference fasta")
    parser.add_argument("--output", default=".", help="Output directory")
    parser.add_argument("--recursive", action="store_true", help="Recurse through pod5")
    parser.add_argument("--chunksize", type=int, default=4000)
    parser.add_argument("--overlap", type=int, default=400)
    parser.add_argument("--min-qscore", type=float, default=5.0)
    parser.add_argument("--min-coverage", type=float, default=0.90)
    parser.add_argument("--min-accuracy", type=float, default=0.90)
    parser.add_argument("--min-mapq", type=int, default=20)
    parser.add_argument("--mm2-preset", default="lr:hq", help="Mappy preset")
    parser.add_argument("--rna", action="store_true", help="Expect RNA orientation")
    parser.add_argument("--limit", type=int, default=0, help="Limit alignments processed")
    parser.add_argument("--max-reads", type=int, default=0, help="Max reads producing chunks")
    default_workers = max(1, min(8, os.cpu_count() or 1))
    parser.add_argument("--workers", type=int, default=default_workers, help="Worker threads")
    return parser.parse_args()

def load_alignments(bam_path: str, min_mapq: int) -> Dict[str, AlnInfo]:
    alignments: Dict[str, AlnInfo] = {}
    print(f"[Info] Scanning BAM: {bam_path} ...")
    
    with pysam.AlignmentFile(bam_path, "rb") as bam:
        for aln in tqdm(bam.fetch(until_eof=True), desc="Parsing BAM", unit="aln"):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            if aln.mapping_quality < min_mapq:
                continue
            
            try:
                mv = aln.get_tag("mv")
            except KeyError:
                continue 
            
            ts = 0
            if aln.has_tag("ts"):
                ts = aln.get_tag("ts")

            info = AlnInfo(
                query_sequence=aln.query_sequence,
                qual=aln.qual,
                cigartuples=aln.cigartuples,
                reference_name=aln.reference_name,
                reference_start=aln.reference_start,
                is_reverse=aln.is_reverse,
                mapping_quality=aln.mapping_quality,
                mv_tag=mv,
                ts_tag=ts
            )
            # Ensure key is string
            alignments[str(aln.query_name)] = info
            
    if not alignments:
        raise RuntimeError("No primary alignments passed the specified filters.")
    return alignments

def decode_move_table(tag_value) -> tuple[int, np.ndarray]:
    if isinstance(tag_value, str):
        values = [int(x) for x in tag_value.split(",") if x]
    elif isinstance(tag_value, (list, tuple, np.ndarray)):
        values = [int(x) for x in tag_value]
    else:
        values = list(tag_value)
    
    stride = values[0]
    moves = np.asarray(values[1:], dtype=np.int32)
    return stride, moves

def chunk_iterator(signal: np.ndarray, chunksize: int, overlap: int) -> Iterable[tuple[int, np.ndarray]]:
    if len(signal) < chunksize:
        return
    step = chunksize - overlap
    if step <= 0: step = 1 
    
    _, offset = divmod(len(signal) - chunksize, step)
    for start in range(offset, len(signal) - chunksize + 1, step):
        yield start, signal[start : start + chunksize]

def signal_to_query_span(
    chunk_start: int, chunk_len: int, trimmed_samples: int,
    dorado_trim: int, stride: int, cum_moves: np.ndarray, query_length: int,
) -> Optional[tuple[int, int]]:
    delta = trimmed_samples - dorado_trim
    dorado_start = max(0, chunk_start + delta)
    dorado_end = max(0, chunk_start + chunk_len + delta)
    
    num_steps = len(cum_moves)
    step_start = min(num_steps, max(0, dorado_start // stride))
    step_end = min(num_steps, math.ceil(dorado_end / stride))
    
    if step_end <= step_start:
        return None
        
    base_start = int(cum_moves[step_start - 1]) if step_start > 0 else 0
    base_end = int(cum_moves[step_end - 1]) if step_end > 0 else 0
    
    base_start = min(query_length, base_start)
    base_end = min(query_length, max(base_start + 1, base_end))
    
    if base_end <= base_start:
        return None
    return base_start, base_end

def slice_alignment(
    aln_info: AlnInfo, q_start: int, q_end: int, aligner: mp.Aligner
) -> Optional[tuple[str, float, float]]:
    cigars = aln_info.cigartuples
    if not cigars: return None
    
    query_seq = aln_info.query_sequence
    chunk_len = q_end - q_start
    if chunk_len <= 0: return None

    ref_pos = aln_info.reference_start
    q_pos = 0
    aligned_q = 0
    matches = 0
    blen = 0
    ref_start = None
    ref_end = None

    for op, length in cigars:
        if op in (0, 7, 8): # Match/Mismatch
            block_q_start = q_pos
            block_q_end = q_pos + length
            overlap_start = max(q_start, block_q_start)
            overlap_end = min(q_end, block_q_end)

            if overlap_start < overlap_end:
                offset = overlap_start - block_q_start
                ref_block_start = ref_pos + offset
                ref_block_end = ref_block_start + (overlap_end - overlap_start)

                if ref_start is None: ref_start = ref_block_start
                ref_end = ref_block_end
                
                try:
                    ref_slice = aligner.seq(aln_info.reference_name, ref_block_start, ref_block_end)
                    q_slice = query_seq[overlap_start:overlap_end]
                    matches += sum(1 for a, b in zip(q_slice.upper(), ref_slice.upper()) if a == b)
                except Exception:
                    pass
                
                aligned_q += overlap_end - overlap_start
                blen += overlap_end - overlap_start
            
            ref_pos += length
            q_pos = block_q_end
        
        elif op == 1: # INS
            q_pos += length
        elif op == 2: # DEL
            if q_start <= q_pos < q_end:
                if ref_start is None: ref_start = ref_pos
                ref_end = ref_pos + length
                blen += length
            ref_pos += length
        elif op == 3: # SKIP
            ref_pos += length
        elif op == 4: # SOFT CLIP
            q_pos += length
        
        if q_pos >= q_end:
            break

    if ref_start is None or ref_end is None or blen == 0:
        return None
    
    coverage = aligned_q / chunk_len
    accuracy = matches / blen
    
    ref_seq = aligner.seq(aln_info.reference_name, ref_start, ref_end)
    if aln_info.is_reverse:
        ref_seq = mp.revcomp(ref_seq)
        
    return ref_seq, coverage, accuracy

def process_read_task(task: tuple[str, np.ndarray, AlnInfo]):
    read_id, signal, aln = task
    aligner = _GLOBAL_ALIGNER
    cfg = _THREAD_CONFIG
    
    local_chunks = []
    local_targets = []
    local_lengths = []
    # Use standard Counter
    rejects = Counter()

    try:
        stride, moves = decode_move_table(aln.mv_tag)
    except Exception:
        rejects["invalid_mv"] += 1
        return local_chunks, local_targets, local_lengths, rejects, 0

    dorado_trim = aln.ts_tag if aln.ts_tag >= 0 else 0
    if dorado_trim >= len(signal):
        rejects["trim_out_of_range"] += 1
        return local_chunks, local_targets, local_lengths, rejects, 0
    
    if dorado_trim:
        signal = signal[dorado_trim:]
    
    cum_moves = np.cumsum(moves, dtype=np.int64)
    qstring = aln.qual
    
    c_chunksize = cfg["chunksize"]
    c_overlap = cfg["overlap"]

    for chunk_start, chunk_signal in chunk_iterator(signal, c_chunksize, c_overlap):
        span = signal_to_query_span(
            chunk_start, len(chunk_signal), dorado_trim, dorado_trim,
            stride, cum_moves, len(aln.query_sequence)
        )
        if span is None:
            rejects["no_query_span"] += 1
            continue
            
        q_start, q_end = span
        q_subseq = aln.query_sequence[q_start:q_end]
        
        if not q_subseq:
            rejects["empty_seq"] += 1
            continue
            
        if cfg["min_qscore"] > 0:
            q_subqual = qstring[q_start:q_end] if qstring and qstring != "*" else None
            mean_q = mean_qscore_from_qstring(q_subqual) if q_subqual else aln.mapping_quality
            if mean_q < cfg["min_qscore"]:
                rejects["low_qscore"] += 1
                continue
        
        alignment_slice = slice_alignment(aln, q_start, q_end, aligner)
        if alignment_slice is None:
            rejects["no_alignment_slice"] += 1
            continue
            
        ref_seq, coverage, accuracy = alignment_slice
        
        if coverage < cfg["min_coverage"]:
            rejects["low_coverage"] += 1
            continue
        if accuracy < cfg["min_accuracy"]:
            rejects["low_accuracy"] += 1
            continue
        if "N" in ref_seq:
            rejects["N_in_sequence"] += 1
            continue

        try:
            encoded = [BASE_TO_INT[base] for base in ref_seq]
        except KeyError:
            rejects["non_acgt_reference"] += 1
            continue
            
        if cfg["rna"]:
            encoded = encoded[::-1]
            
        local_chunks.append(chunk_signal.astype(np.float16))
        local_targets.append(encoded)
        local_lengths.append(len(encoded))

    contributed = 1 if local_chunks else 0
    return local_chunks, local_targets, local_lengths, rejects, contributed

def format_timespan(seconds: float) -> str:
    if seconds <= 0: return "0s"
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    if h: return f"{int(h)}h{int(m):02d}m"
    return f"{int(m)}m{s:04.1f}s"

def main() -> None:
    args = parse_args()
    
    alignments = load_alignments(args.bam, args.min_mapq)
    print(f"[Info] Loaded {len(alignments)} suitable alignments.", flush=True)

    print(f"[Info] Loading reference index: {args.reference} ...", flush=True)
    global _GLOBAL_ALIGNER
    _GLOBAL_ALIGNER = mp.Aligner(args.reference, preset=args.mm2_preset)
    if not _GLOBAL_ALIGNER:
        raise RuntimeError("Failed to load minimap2 index")
    print("[Info] Reference index ready.", flush=True)

    global _THREAD_CONFIG
    _THREAD_CONFIG = vars(args)
    
    read_ids = set(alignments.keys())
    print(f"[Info] Initializing Pod5 reader for {len(read_ids)} reads...", flush=True)
    
    reads = get_pod5_reads(
        args.pod5,
        read_ids=read_ids, # bonito should accept set of strings
        skip=False, 
        n_proc=min(4, args.workers),
        recursive=args.recursive,
        do_trim=False 
    )

    chunks: List[np.ndarray] = []
    targets: List[List[int]] = []
    lengths: List[int] = []
    
    # FIX: Use standard Counter
    reject_counter = Counter()
    
    total_reads = len(read_ids)
    if args.limit: total_reads = min(total_reads, args.limit)
    
    pbar_reads = tqdm(total=total_reads, desc="Reads", unit="read", position=0)
    pbar_chunks = tqdm(desc="Chunks", unit="chunk", position=1)
    
    start_time = time.time()
    contributing_reads = 0
    processed_count = 0
    stop_submitting = False
    pending = set()
    max_workers = args.workers
    max_pending = max_workers * 2 

    executor = ThreadPoolExecutor(max_workers=max_workers)

    def harvest(return_when):
        nonlocal contributing_reads
        if not pending: return
        done, _ = wait(pending, return_when=return_when)
        for fut in done:
            pending.remove(fut)
            try:
                res = fut.result()
                c_list, t_list, l_list, r_counts, contrib = res
                
                if c_list:
                    chunks.extend(c_list)
                    targets.extend(t_list)
                    lengths.extend(l_list)
                    pbar_chunks.update(len(c_list))
                
                reject_counter.update(r_counts)
                if contrib: contributing_reads += 1
                pbar_reads.update(1)
                
            except Exception as e:
                tqdm.write(f"[Error] Worker failed: {e}")
                pbar_reads.update(1)

    try:
        for read in reads:
            if stop_submitting: break
            
            # FIX: Force string conversion for lookup
            read_id_str = str(read.read_id)
            
            aln_info = alignments.pop(read_id_str, None)
            if aln_info is None:
                # Now safe to increment
                reject_counter["id_mismatch_pod5_bam"] += 1
                continue
            
            task = (read_id_str, read.signal, aln_info)
            pending.add(executor.submit(process_read_task, task))
            
            processed_count += 1
            if args.limit and processed_count >= args.limit:
                stop_submitting = True
            
            if len(pending) >= max_pending:
                harvest(FIRST_COMPLETED)
                
            if args.max_reads and contributing_reads >= args.max_reads:
                stop_submitting = True
                
        while pending:
            harvest(ALL_COMPLETED)
            
    except KeyboardInterrupt:
        tqdm.write("\n[Warn] Interrupted by user. Saving what we have...")
    finally:
        executor.shutdown(wait=False)
        pbar_reads.close()
        pbar_chunks.close()
        
        print("\n" + "="*40)
        print("PROCESSING SUMMARY")
        print("="*40)
        print(f"Total Processed: {processed_count}")
        print(f"Contributing Reads: {contributing_reads}")
        print(f"Total Chunks: {len(chunks)}")
        print("-" * 20)
        print("REJECTION REASONS:")
        if not reject_counter:
            print("  (No rejections recorded)")
        # Counter.most_common() exists in standard library
        for reason, count in reject_counter.most_common():
            print(f"  {reason}: {count}")
        print("="*40)

    if not chunks:
        print("[Error] No chunks were generated. See rejection reasons above.")
        sys.exit(1)

    print(f"[Info] Saving {len(chunks)} chunks to {args.output}...", flush=True)
    os.makedirs(args.output, exist_ok=True)
    
    chunk_array = np.asarray(chunks, dtype=np.float16)
    length_array = np.asarray(lengths, dtype=np.uint16)
    
    keep = typical_indices(length_array)
    rng = np.random.default_rng()
    rng.shuffle(keep)
    
    chunk_array = chunk_array[keep]
    length_array = length_array[keep]
    targets_kept = [targets[i] for i in keep]
    
    max_len = int(length_array.max()) if len(length_array) > 0 else 0
    ref_array = np.zeros((len(targets_kept), max_len), dtype=np.uint8)
    for idx, row in enumerate(targets_kept):
        ref_array[idx, : len(row)] = row

    np.save(os.path.join(args.output, "chunks.npy"), chunk_array)
    np.save(os.path.join(args.output, "references.npy"), ref_array)
    np.save(os.path.join(args.output, "reference_lengths.npy"), length_array)
    
    print(f"[Success] Done in {format_timespan(time.time() - start_time)}.")

if __name__ == "__main__":
    main()