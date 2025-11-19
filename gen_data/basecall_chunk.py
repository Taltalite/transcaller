#!/usr/bin/env python3
"""
Make CTC training chunks from:
 - raw reads (fast5/pod5) via bonito.Reader
 - basecalls per-read (FASTA or TSV)
 - reference fasta (for mappy alignment)

Features:
 - align basecalls -> reference with mappy, use CIGAR to map read-base -> ref-pos when possible
 - map chunk sample-range -> read-base-range via proportional mapping across aligned region,
   then map read-base-range -> reference interval (precise via CIGAR map when available)
 - two-pass mode to compute total_chunks and max_target_len, then allocate np.memmap arrays
 - optional multiprocessing to speed chunk extraction
"""
import os
import sys
import argparse
import numpy as np
from functools import partial
from tqdm import tqdm

# optional deps
try:
    import mappy as mp
except Exception:
    mp = None

try:
    # bonito.Reader is convenient for reading fast5/pod5; if not available you must provide own reader
    sys.path.append('/home/lijy/workspace/')
    from bonito.reader import Reader
except Exception:
    Reader = None

BASE_TO_INT = {'A':1,'C':2,'G':3,'T':4,'a':1,'c':2,'g':3,'t':4,'N':0,'n':0}

def load_basecalls(path):
    mapping = {}
    with open(path, 'r') as fh:
        first = fh.readline()
        fh.seek(0)
        if first.startswith('>'):
            rid = None
            seqs = []
            for line in fh:
                line = line.strip()
                if not line: continue
                if line.startswith('>'):
                    if rid is not None:
                        mapping[rid] = ''.join(seqs)
                    rid = line[1:].split()[0]
                    seqs = []
                else:
                    seqs.append(line)
            if rid is not None:
                mapping[rid] = ''.join(seqs)
        else:
            for line in fh:
                line = line.strip()
                if not line: continue
                parts = line.split()
                if len(parts) >= 2:
                    mapping[parts[0]] = parts[1]
    return mapping

def seq_to_int_array(seq):
    return np.array([BASE_TO_INT.get(ch, 0) for ch in seq], dtype=np.uint8)

def sliding_count(length, chunksize, overlap):
    step = chunksize - overlap
    if step <= 0:
        return 0
    n = (max(0, length - chunksize) + step) // step + 1 if length>0 else 0
    return n

def build_read_to_ref_map_from_cigar(cigar_ops, q_st, r_st, strand):
    """
    cigar_ops: list of (op_len, op_char) like [(10,'M'), (1,'I'), (5,'D'), ...]
    returns dict read_index -> ref_index (or None if read base maps to an insertion)
    read indices are absolute wrt read (0-based)
    """
    read_to_ref = {}
    read_pos = q_st
    ref_pos = r_st
    for l,op in cigar_ops:
        if op in ('M','=', 'X'):
            for i in range(l):
                read_to_ref[read_pos] = ref_pos
                read_pos += 1
                ref_pos += 1
        elif op == 'I':  # insertion to ref: consumes read only
            for i in range(l):
                read_to_ref[read_pos] = None
                read_pos += 1
        elif op == 'D':  # deletion from ref: consumes ref only
            ref_pos += l
        elif op == 'S' or op == 'H':  # soft/hard clipping: consumes read (S) or none (H)
            if op == 'S':
                read_pos += l
            # H consumes nothing in sequence present (ignore)
        else:
            # fallback: advance conservatively
            read_pos += l
            ref_pos += l
    # if alignment was on reverse strand, ref positions are still coordinates on reference;
    # caller will handle strand (we will fetch reference subsequence via aligner.seq which takes coords)
    return read_to_ref

def parse_cigar_str(cigar_str):
    # parse e.g. "10M1I5M" -> [(10,'M'), (1,'I'), (5,'M')]
    ops = []
    num = ''
    for ch in cigar_str:
        if ch.isdigit():
            num += ch
        else:
            ops.append((int(num), ch))
            num = ''
    return ops

def map_chunk_to_ref_interval(s, e, sig_len, read_len, align_hit):
    """
    align_hit: object from mappy with attributes: ctg, r_st, r_en, q_st, q_en, strand, mapq, cigar_str (if present)
    returns: (ctg, ref_start, ref_end) or None if failed
    Strategy:
      - compute read-base interval corresponding to [s,e) by proportional mapping across aligned query region [q_st,q_en)
      - if CIGAR available, build read->ref map and get min/max ref positions for bases in that read interval
      - else fallback to proportional mapping between q_st..q_en -> r_st..r_en
    """
    q_st = align_hit.q_st
    q_en = align_hit.q_en
    r_st = align_hit.r_st
    r_en = align_hit.r_en
    if q_en <= q_st or r_en <= r_st:
        return None
    # aligned read bases length
    aligned_read_len = q_en - q_st
    # relative base indices (within aligned region)
    rel_bstart = int(np.floor(((s / sig_len) * aligned_read_len)))
    rel_bend = int(np.ceil(((e / sig_len) * aligned_read_len)))
    rel_bstart = max(0, min(rel_bstart, aligned_read_len))
    rel_bend = max(0, min(rel_bend, aligned_read_len))
    if rel_bend <= rel_bstart:
        return None
    read_bstart = q_st + rel_bstart
    read_bend = q_st + rel_bend

    # try CIGAR mapping if present
    if getattr(align_hit, 'cigar_str', None):
        cigar_ops = parse_cigar_str(align_hit.cigar_str)
        rmap = build_read_to_ref_map_from_cigar(cigar_ops, q_st, r_st, align_hit.strand)
        ref_positions = [rmap.get(i) for i in range(read_bstart, read_bend) if rmap.get(i) is not None]
        if len(ref_positions) == 0:
            # fallback to proportional mapping
            pass
        else:
            ref_start = min(ref_positions)
            ref_end = max(ref_positions) + 1  # make end exclusive
            return (align_hit.ctg, ref_start, ref_end)
    # fallback proportional mapping
    frac_start = (read_bstart - q_st) / max(1, (q_en - q_st))
    frac_end = (read_bend - q_st) / max(1, (q_en - q_st))
    ref_start = int(round(r_st + frac_start * (r_en - r_st)))
    ref_end = int(round(r_st + frac_end * (r_en - r_st)))
    if ref_end <= ref_start:
        ref_end = ref_start + 1
    return (align_hit.ctg, ref_start, ref_end)

def process_read_for_counts(read_meta, basecall_seq, aligner, chunksize, overlap):
    """
    read_meta: object with .signal (numpy array) and .name or read_id
    returns: list of (per-chunk target_length)
    """
    sig = np.asarray(read_meta.signal, dtype=np.float32)
    Ls = len(sig)
    if Ls == 0:
        return []
    Lb = len(basecall_seq)
    # align basecall -> reference
    hits = list(aligner.map(basecall_seq))
    if len(hits) == 0:
        return []
    hit = hits[0]
    # count chunks
    n = sliding_count(Ls, chunksize, overlap)
    target_lengths = []
    for s_idx in range(n):
        s = s_idx * (chunksize - overlap)
        e = min(s + chunksize, Ls)
        mapped = map_chunk_to_ref_interval(s, e, Ls, Lb, hit)
        if mapped is None:
            target_lengths.append(0)
        else:
            ctg, rs, re = mapped
            target_lengths.append(max(0, re - rs))
    return target_lengths

def extract_chunks_from_read(read_meta, basecall_seq, aligner, chunksize, overlap, ref_fasta):
    """
    returns list of tuples: (chunk_signal np.float16 length chunksize,
                             target_uint8_array,
                             target_len)
    """
    sig = np.asarray(read_meta.signal, dtype=np.float32)
    Ls = len(sig)
    if Ls == 0:
        return []
    Lb = len(basecall_seq)
    hits = list(aligner.map(basecall_seq))
    if len(hits) == 0:
        return []
    hit = hits[0]
    results = []
    step = chunksize - overlap
    i = 0
    while i < Ls:
        s = i
        e = min(i + chunksize, Ls)
        chunk = sig[s:e].astype(np.float32)
        if len(chunk) < chunksize:
            pad = np.zeros(chunksize - len(chunk), dtype=chunk.dtype)
            chunk = np.concatenate([chunk, pad])
        mapped = map_chunk_to_ref_interval(s, e, Ls, Lb, hit)
        if mapped is None:
            # skip chunk with no mapping
            i += step
            continue
        ctg, rs, re = mapped
        # safe bounds: ensure rs<re
        if re <= rs:
            i += step
            continue
        # fetch reference sequence from aligner (aligner.seq)
        try:
            refseq = aligner.seq(ctg, rs, re)
        except Exception:
            # fallback: if aligner doesn't implement seq, skip
            i += step
            continue
        target_arr = seq_to_int_array(refseq)
        results.append((chunk.astype(np.float16), target_arr, len(target_arr)))
        i += step
    return results

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reads-dir", required=True, help="reads directory (fast5/pod5) readable by bonito.Reader")
    p.add_argument("--basecalls", required=True, help="FASTA or TSV with per-read basecalls (id\\tseq)")
    p.add_argument("--reference", required=True, help="reference fasta for minimap2/mappy")
    p.add_argument("--chunksize", type=int, default=2048)
    p.add_argument("--overlap", type=int, default=400)
    p.add_argument("--output-dir", default=".")
    p.add_argument("--shuffle", action="store_true")
    p.add_argument("--num-workers", type=int, default=0, help="use multiprocessing for extraction (0 = no parallelism)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    if mp is None:
        print("Error: mappy (minimap2 Python binding) is required. pip install mappy", file=sys.stderr)
        sys.exit(1)
    if Reader is None:
        print("Error: bonito.reader.Reader is required for reading fast5/pod5. Ensure bonito is on PYTHONPATH.", file=sys.stderr)
        sys.exit(1)

    basecalls = load_basecalls(args.basecalls)
    aligner = mp.Aligner(args.reference)
    if not aligner:
        print("Error: failed to load reference index with mappy", file=sys.stderr)
        sys.exit(1)

    reader = Reader(args.reads_dir if args.reads_dir else args.reads_dir, recursive=True)

    # FIRST PASS: compute total chunks and max target length
    total_chunks = 0
    max_target_len = 0
    reads = list(reader.get_reads(args.reads_dir, n_proc=1, recursive=True))
    if args.verbose:
        print(f"Scanned {len(reads)} reads for chunk accounting")
    for read_meta in tqdm(reads, desc="counting"):
        rid = getattr(read_meta, 'read_id', None) or getattr(read_meta, 'name', None) or getattr(read_meta, 'id', None)
        if rid is None or rid not in basecalls:
            continue
        basecall_seq = basecalls[rid]
        t_lengths = process_read_for_counts(read_meta, basecall_seq, aligner, args.chunksize, args.overlap)
        # only keep positive lengths
        t_lengths = [l for l in t_lengths if l>0]
        total_chunks += len(t_lengths)
        if t_lengths:
            max_target_len = max(max_target_len, max(t_lengths))

    if total_chunks == 0:
        print("No chunks produced. Exiting.", file=sys.stderr)
        sys.exit(0)

    os.makedirs(args.output_dir, exist_ok=True)
    # allocate memmap arrays
    chunks_mm_path = os.path.join(args.output_dir, "chunks.npy")
    refs_mm_path = os.path.join(args.output_dir, "references.npy")
    lengths_path = os.path.join(args.output_dir, "reference_lengths.npy")

    # allocate chunks memmap (N x chunksize)
    chunks_mm = np.memmap(chunks_mm_path, mode='w+', dtype=np.float16, shape=(total_chunks, args.chunksize))
    refs_mm = np.memmap(refs_mm_path, mode='w+', dtype=np.uint8, shape=(total_chunks, max_target_len))
    lengths_mm = np.memmap(lengths_path, mode='w+', dtype=np.uint16, shape=(total_chunks,))

    # SECOND PASS: fill memmaps
    idx = 0
    if args.num_workers <= 1:
        it = tqdm(reads, desc="extracting")
        for read_meta in it:
            rid = getattr(read_meta, 'read_id', None) or getattr(read_meta, 'name', None) or getattr(read_meta, 'id', None)
            if rid is None or rid not in basecalls:
                continue
            basecall_seq = basecalls[rid]
            entries = extract_chunks_from_read(read_meta, basecall_seq, aligner, args.chunksize, args.overlap, args.reference)
            for chunk, target_arr, tlen in entries:
                if tlen == 0:
                    continue
                chunks_mm[idx, :] = chunk
                refs_mm[idx, :tlen] = target_arr
                lengths_mm[idx] = tlen
                idx += 1
    else:
        # parallel extraction: each worker processes a subset of reads and returns results
        from concurrent.futures import ProcessPoolExecutor, as_completed
        # prepare args per read for mapping
        jobs = []
        with ProcessPoolExecutor(max_workers=args.num_workers) as ex:
            futures = {}
            for read_meta in reads:
                rid = getattr(read_meta, 'read_id', None) or getattr(read_meta, 'name', None) or getattr(read_meta, 'id', None)
                if rid is None or rid not in basecalls:
                    continue
                basecall_seq = basecalls[rid]
                # we can't send read_meta object across easily; instead re-open reader in worker by read path or
                # pass a minimal descriptor; for simplicity we serialize signal and id (may be large). If too slow,
                # use single-process mode or implement worker-side Reader opening.
                futures[ex.submit(extract_chunks_from_read, read_meta, basecall_seq, mp.Aligner(args.reference), args.chunksize, args.overlap, args.reference)] = rid
            for fut in tqdm(as_completed(futures), total=len(futures), desc="gathering"):
                entries = fut.result()
                for chunk, target_arr, tlen in entries:
                    if tlen == 0:
                        continue
                    chunks_mm[idx, :] = chunk
                    refs_mm[idx, :tlen] = target_arr
                    lengths_mm[idx] = tlen
                    idx += 1

    # final shrink if idx < total_chunks
    if idx < total_chunks:
        # flush and rewrite with actual size
        chunks_mm.flush(); refs_mm.flush(); lengths_mm.flush()
        # load into normal np arrays and rewrite exact-sized files (cheap if not huge)
        chunks = np.array(chunks_mm[:idx], dtype=np.float16)
        refs = np.array(refs_mm[:idx], dtype=np.uint8)
        lengths = np.array(lengths_mm[:idx], dtype=np.uint16)
        np.save(chunks_mm_path, chunks)
        np.save(refs_mm_path, refs)
        np.save(lengths_path, lengths)
    else:
        # flush memmaps to disk as .npy: memmap already created as raw binary .npy-like content but not with header.
        # simplest: read back and save via np.save to produce standard .npy files
        chunks = np.array(chunks_mm, dtype=np.float16)
        refs = np.array(refs_mm, dtype=np.uint8)
        lengths = np.array(lengths_mm, dtype=np.uint16)
        np.save(chunks_mm_path, chunks)
        np.save(refs_mm_path, refs)
        np.save(lengths_path, lengths)

    print("Wrote chunks.npy, references.npy, reference_lengths.npy")
    if args.shuffle:
        # load and shuffle (on-disk shuffle could be implemented, here load into memory for simplicity)
        data = np.load(os.path.join(args.output_dir, "chunks.npy"))
        refs = np.load(os.path.join(args.output_dir, "references.npy"))
        lens = np.load(os.path.join(args.output_dir, "reference_lengths.npy"))
        perm = np.random.permutation(len(data))
        np.save(os.path.join(args.output_dir, "chunks.npy"), data[perm])
        np.save(os.path.join(args.output_dir, "references.npy"), refs[perm])
        np.save(os.path.join(args.output_dir, "reference_lengths.npy"), lens[perm])
        print("Shuffled output")

if __name__ == "__main__":
    main()