import os
import h5py
import pysam
import pod5
import numpy as np
import argparse
from tqdm import tqdm

# --- Key Adjustable Parameters ---
SIGNAL_LENGTH = 4096 
WINDOW_STRIDE = 512
# --------------------

BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

def parse_moves(mv_tag, stride):
    moves = np.array(mv_tag)
    if stride > 0: moves[1:] = moves[1:] * stride
    base_signal_starts = np.cumsum(moves)
    return base_signal_starts

def main(args):
    # --- Statistics dictionary for robust reporting ---
    stats = {
        "bam_reads_processed": 0, "id_not_in_pod5_index": 0, "read_not_found_in_pod5_file": 0,
        "missing_tags": 0, "signal_too_short": 0, "total_windows_processed": 0,
        "window_mad_is_zero": 0, "window_no_bases": 0, "window_label_invalid": 0,
        "valid_samples_created": 0,
    }

    print("Step 1: Building index of read_ids to POD5 file paths...")
    pod5_files = [os.path.join(args.pod5_dir, f) for f in os.listdir(args.pod5_dir) if f.endswith('.pod5')]
    read_id_to_pod5_path = {}
    for pod5_path in tqdm(pod5_files, desc="Indexing POD5 files"):
        with pod5.Reader(pod5_path) as reader:
            for read in reader.reads():
                read_id_to_pod5_path[str(read.read_id)] = pod5_path
    print(f"Indexed {len(read_id_to_pod5_path)} unique reads from POD5 files.")

    print("Steps 2 & 3: Extracting data and assembling HDF5 dataset...")
    bam_file = pysam.AlignmentFile(args.bam_file, "rb")
    ref_fasta = pysam.FastaFile(args.reference_fasta)
    
    bam_file_size = bam_file.mapped

    with h5py.File(args.output_hdf5, 'w') as hf:
        # ... (HDF5 dataset creation remains the same)
        event_ds = hf.create_dataset('event', (0, SIGNAL_LENGTH), maxshape=(None, SIGNAL_LENGTH), dtype=np.float32)
        label_ds = hf.create_dataset('label', (0, 200), maxshape=(None, 200), dtype=np.int32)
        label_len_ds = hf.create_dataset('label_len', (0,), maxshape=(None,), dtype=np.int32)

        for read in tqdm(bam_file.fetch(), total=bam_file_size, desc="Processing BAM records"):
            stats["bam_reads_processed"] += 1
            read_id_str = str(read.query_name)
            
            if read_id_str not in read_id_to_pod5_path:
                stats["id_not_in_pod5_index"] += 1
                continue

            try:
                stride = read.get_tag('ts')
                mv_tag = read.get_tag('mv')
                ground_truth_label_str = ref_fasta.fetch(read.reference_name, read.reference_start, read.reference_end).upper()
                base_signal_starts = parse_moves(mv_tag, stride)
                
                pod5_path = read_id_to_pod5_path[read_id_str]
                
                # --- THIS BLOCK IS THE FINAL CORRECTION ---
                # All operations on a read must happen while its file is open.
                with pod5.Reader(pod5_path) as reader:
                    pod5_read = None
                    for r in reader.reads():
                        if str(r.read_id) == read_id_str:
                            pod5_read = r
                            break
                    
                    if pod5_read is None:
                        stats["read_not_found_in_pod5_file"] += 1
                        continue # continue the outer BAM loop

                    # Access signal and do ALL processing INSIDE this 'with' block
                    raw_signal = pod5_read.signal

                    if len(raw_signal) < SIGNAL_LENGTH:
                        stats["signal_too_short"] += 1
                        continue # continue the outer BAM loop

                    # The entire windowing logic is now safely inside the file handle context
                    for win_start in range(0, len(raw_signal) - SIGNAL_LENGTH, WINDOW_STRIDE):
                        stats["total_windows_processed"] += 1
                        win_end = win_start + SIGNAL_LENGTH
                        signal_window = raw_signal[win_start:win_end]
                        median = np.median(signal_window)
                        mad = np.median(np.abs(signal_window - median))
                        if mad == 0:
                            stats["window_mad_is_zero"] += 1
                            continue
                        normalized_signal = (signal_window - median) / mad
                        first_base_idx = np.searchsorted(base_signal_starts, win_start, side='right')
                        last_base_idx = np.searchsorted(base_signal_starts, win_end, side='left')
                        if first_base_idx >= last_base_idx:
                            stats["window_no_bases"] += 1
                            continue
                        label_str_window = ground_truth_label_str[first_base_idx:last_base_idx]
                        label_int_window = [BASE_TO_INT[b] for b in label_str_window if b in BASE_TO_INT]
                        if not label_int_window or len(label_int_window) > 200:
                            stats["window_label_invalid"] += 1
                            continue
                        
                        # Save valid sample
                        current_size = event_ds.shape[0]
                        event_ds.resize(current_size + 1, axis=0)
                        label_ds.resize(current_size + 1, axis=0)
                        label_len_ds.resize(current_size + 1, axis=0)
                        padded_label = np.full((200,), -1, dtype=np.int32)
                        padded_label[:len(label_int_window)] = label_int_window
                        event_ds[current_size, :] = normalized_signal
                        label_ds[current_size, :] = padded_label
                        label_len_ds[current_size] = len(label_int_window)
                        stats["valid_samples_created"] += 1

            except KeyError:
                stats["missing_tags"] += 1
                continue
    
    bam_file.close()
    ref_fasta.close()
    print("\n--- PROCESSING FINISHED ---")
    print(f"Final valid samples created: {stats['valid_samples_created']}")
    print("\n--- DETAILED STATISTICS REPORT ---")
    for key, value in stats.items():
        print(f"{key:<30}: {value}")
    print("----------------------------------\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a Melchior training dataset from Dorado BAM and POD5 files.")
    parser.add_argument("--bam_file", type=str, required=True)
    parser.add_argument("--pod5_dir", type=str, required=True)
    parser.add_argument("--reference_fasta", type=str, required=True)
    parser.add_argument("--output_hdf5", type=str, required=True)
    args = parser.parse_args()
    main(args)