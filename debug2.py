import os
import pysam
import pod5
from tqdm import tqdm

# --- CONFIGURE YOUR PATHS HERE ---
BAM_FILE = './HG002_aligned_reads.sorted.bam'
POD5_DIR = '/data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5'
# ---------------------------------

def run_diagnostics_v3():
    print("--- Starting Corrected Dataset Diagnostics (v3) ---")

    # --- Check 1: Read ID Overlap with Type Correction ---
    print("\n[Check 1] Analyzing overlap between BAM and POD5 read IDs with type correction...")
    
    pod5_read_ids = set()
    try:
        pod5_files = [os.path.join(POD5_DIR, f) for f in os.listdir(POD5_DIR) if f.endswith('.pod5')]
        for pod5_path in tqdm(pod5_files, desc="Indexing POD5 reads"):
            with pod5.Reader(pod5_path) as reader:
                for r in reader.reads():
                    read_id = r.read_id
                    # Ensure the ID is a string
                    if isinstance(read_id, bytes):
                        read_id = read_id.decode('utf-8')
                    pod5_read_ids.add(read_id)
    except FileNotFoundError:
        print(f"  [ERROR] POD5 directory not found at: {POD5_DIR}")
        return

    # Let's check the type of a sample pod5 ID
    if pod5_read_ids:
        sample_pod5_id = next(iter(pod5_read_ids))
        print(f"Sample POD5 read ID type: {type(sample_pod5_id)}")

    # Get all read IDs from the BAM file
    bam_read_ids = set()
    try:
        with pysam.AlignmentFile(BAM_FILE, "rb") as bam:
            # Use total=bam.count() if it works, otherwise remove it
            for read in tqdm(bam.fetch(), desc="Indexing BAM reads"):
                query_name = read.query_name
                # Ensure the ID is a string
                if isinstance(query_name, bytes):
                    query_name = query_name.decode('utf-8')
                bam_read_ids.add(query_name)
    except FileNotFoundError:
        print(f"  [ERROR] BAM file not found at: {BAM_FILE}")
        return

    # Let's check the type of a sample bam ID
    if bam_read_ids:
        sample_bam_id = next(iter(bam_read_ids))
        print(f"Sample BAM read ID type: {type(sample_bam_id)}")
    
    # Find the intersection
    common_read_ids = bam_read_ids.intersection(pod5_read_ids)

    print(f"\n--- Report for Check 1 (Corrected) ---")
    print(f"Total reads found in POD5 directory: {len(pod5_read_ids)}")
    print(f"Total primary reads in BAM file: {len(bam_read_ids)}")
    print(f"Number of reads in common: {len(common_read_ids)}")

    if common_read_ids:
         print("\n[SUCCESS] Overlap found! The data type mismatch was the issue.")
    else:
        print("\n[FAILURE] Still 0 reads in common. This implies a very unusual data mismatch problem.")

if __name__ == "__main__":
    run_diagnostics_v3()