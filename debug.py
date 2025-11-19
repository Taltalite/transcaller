import os
import h5py
import pysam
import pod5
import numpy as np
import argparse
from tqdm import tqdm

# --- PASTE THE READ ID FROM STEP 1 HERE ---
DEBUG_READ_ID = "11d02d7b-5b2c-460a-ba74-e0fd4fb9341f" 

# --- Configuration (same as before) ---
SIGNAL_LENGTH = 4096
WINDOW_STRIDE = 512
BASE_TO_INT = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

# --- UNCHANGED FUNCTIONS ---
def parse_moves(mv_tag, stride):
    moves = np.array(mv_tag)
    if stride > 0:
        moves[1:] = moves[1:] * stride
    base_signal_starts = np.cumsum(moves)
    return base_signal_starts

def main():
    # Hardcode paths for easy debugging
    args = {
        "bam_file": "./HG002_aligned_reads.sorted.bam",
        "pod5_dir": "/data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5",
        "reference_fasta": "/home/lijy/windows_ssd/HG002/ref/GCF_000001405.26_GRCh38_genomic.fna",
    }

    print(f"--- STARTING DEBUG MODE FOR READ ID: {DEBUG_READ_ID} ---")

    # --- Indexing (same as before) ---
    print("\nStep 2: Building index of read_ids to POD5 file paths...")
    pod5_files = [os.path.join(args["pod5_dir"], f) for f in os.listdir(args["pod5_dir"]) if f.endswith('.pod5')]
    read_id_to_pod5_path = {}
    for pod5_path in pod5_files:
        with pod5.Reader(pod5_path) as reader:
            for read in reader.reads():
                read_id_to_pod5_path[read.read_id] = pod5_path
    print(f"Indexed {len(read_id_to_pod5_path)} reads.")

    # --- Main Processing Loop (with debug prints) ---
    print("\nStep 3: Attempting to process the debug read...")
    bam_file = pysam.AlignmentFile(args["bam_file"], "rb")
    ref_fasta = pysam.FastaFile(args["reference_fasta"])

    found_in_bam = False
    for read in bam_file.fetch():
        if read.query_name == DEBUG_READ_ID:
            found_in_bam = True
            print(f"\n[SUCCESS] Found read '{DEBUG_READ_ID}' in the BAM file.")

            # --- TRACE THE LOGIC ---
            if read.is_unmapped or read.is_secondary or read.is_supplementary:
                print("[FAIL] Read is unmapped or secondary/supplementary. Skipping.")
                break

            if read.query_name not in read_id_to_pod5_path:
                print("[FAIL] Read ID found in BAM, but not found in the POD5 file index. Check your --pod5_dir.")
                break
            else:
                print(f"[SUCCESS] Read ID found in POD5 index. Path: {read_id_to_pod5_path[read.query_name]}")

            try:
                mv_tag = read.get_tag('mv')
                stride = read.get_tag('st')
                print(f"[SUCCESS] Found 'mv' and 'st' tags. Stride is {stride}.")

                base_signal_starts = parse_moves(mv_tag, stride)
                pod5_path = read_id_to_pod5_path[read.query_name]
                with pod5.Reader(pod5_path) as reader:
                    raw_signal = reader.get_read(read.query_name).signal
                print(f"[INFO] Raw signal length is {len(raw_signal)} samples.")

                if len(raw_signal) < SIGNAL_LENGTH:
                    print(f"[FAIL] Raw signal length ({len(raw_signal)}) is less than required SIGNAL_LENGTH ({SIGNAL_LENGTH}). No windows can be created.")
                    break

                window_created_count = 0
                for win_start in range(0, len(raw_signal) - SIGNAL_LENGTH, WINDOW_STRIDE):
                    win_end = win_start + SIGNAL_LENGTH
                    
                    first_base_idx = np.searchsorted(base_signal_starts, win_start, side='right')
                    last_base_idx = np.searchsorted(base_signal_starts, win_end, side='left')
                    
                    print(f"  -> Processing window [{win_start}, {win_end}]: Found bases from index {first_base_idx} to {last_base_idx}.")

                    if first_base_idx >= last_base_idx:
                        print("     [FAIL] No complete bases fall within this window. Skipping window.")
                        continue
                    
                    ground_truth_label_str = ref_fasta.fetch(read.reference_name, read.reference_start, read.reference_end).upper()
                    label_str_window = ground_truth_label_str[first_base_idx:last_base_idx]
                    label_int_window = [BASE_TO_INT[b] for b in label_str_window if b in BASE_TO_INT]
                    
                    print(f"     [INFO] Label sequence for window: '{label_str_window}' (Length: {len(label_int_window)})")

                    if not label_int_window:
                        print("     [FAIL] Label sequence is empty after filtering for A,C,G,T. Skipping window.")
                        continue
                    if len(label_int_window) > 200:
                        print("     [FAIL] Label sequence is too long (> 200). Skipping window.")
                        continue
                    
                    print("     [SUCCESS] This window is a valid sample and would be saved.")
                    window_created_count += 1
                
                if window_created_count == 0:
                    print("\n[CONCLUSION] Read was processed, but no valid windows were generated.")
                else:
                    print(f"\n[CONCLUSION] Read was processed successfully, generating {window_created_count} valid samples.")

            except (KeyError, ValueError) as e:
                print(f"[FAIL] An error occurred. Does this read have 'mv' and 'st' tags? Error: {e}")

            break # Stop after processing the debug read
            
    if not found_in_bam:
        print(f"[FAIL] Could not find read '{DEBUG_READ_ID}' in the BAM file at all.")

    bam_file.close()
    ref_fasta.close()

if __name__ == "__main__":
    main()