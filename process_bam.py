import os
import h5py
import pysam
import numpy as np
from tqdm import tqdm
from ont_fast5_api.fast5_interface import get_fast5_file

def index_fast5_files(fast5_dir: str) -> dict:
    """
    Scans a directory of FAST5 files and creates a map of read_id to file_path.

    Args:
        fast5_dir: The directory containing FAST5 files.

    Returns:
        A dictionary mapping each read_id to its FAST5 file path.
    """
    print(f"Indexing FAST5 files in {fast5_dir}...")
    read_id_to_path = {}
    for root, _, files in os.walk(fast5_dir):
        for filename in files:
            if filename.endswith(".fast5"):
                fast5_path = os.path.join(root, filename)
                try:
                    with get_fast5_file(fast5_path, mode="r") as f5:
                        for read in f5.get_reads():
                            read_id_to_path[read.read_id] = fast5_path
                except Exception as e:
                    print(f"Warning: Could not process file {fast5_path}. Error: {e}")
    print(f"Found and indexed {len(read_id_to_path)} reads.")
    return read_id_to_path

def reverse_complement(seq: str) -> str:
    """Computes the reverse complement of a DNA sequence."""
    complement_map = str.maketrans("ATCGN", "TAGCN")
    return seq.upper().translate(complement_map)[::-1]

def create_training_dataset_generator(bam_path: str, fast5_dir: str):
    """
    Creates a generator that yields training pairs of (raw_signal, ground_truth_sequence).

    This function reads an aligned BAM file, finds the corresponding raw signal
    in the FAST5 files, and pairs it with the ground-truth reference sequence.

    Args:
        bam_path: Path to the sorted and indexed BAM file.
        fast5_dir: Path to the directory containing FAST5 files.

    Yields:
        A tuple of (raw_signal, ground_truth_sequence) for each primary alignment.
        - raw_signal: NumPy array of int16 raw signal values.
        - ground_truth_sequence: The corresponding reference sequence as a string.
    """
    # 1. Index all FAST5 files for quick lookup
    fast5_index = index_fast5_files(fast5_dir)

    # 2. Open the BAM file
    with pysam.AlignmentFile(bam_path, "rb") as bam_file:
        print("Processing BAM file to generate dataset...")
        for record in tqdm(bam_file, desc="Reads Processed"):
            # 3. Filter for good quality, primary alignments
            if (
                record.is_unmapped
                or record.is_secondary
                or record.is_supplementary
            ):
                continue

            read_id = record.query_name
            
            # 4. Find the corresponding FAST5 file
            if read_id not in fast5_index:
                # print(f"Warning: Read ID {read_id} from BAM not found in FAST5 index. Skipping.")
                continue

            try:
                # 5. Extract raw signal from the FAST5 file
                fast5_path = fast5_index[read_id]
                with get_fast5_file(fast5_path, mode="r") as f5:
                    read = f5.get_read(read_id)
                    raw_signal = read.get_raw_data()

                # 6. Get the ground-truth sequence from the reference
                # This is the actual sequence on the reference genome that the read mapped to.
                ground_truth_sequence = record.get_reference_sequence()

                # 7. IMPORTANT: Handle strand direction
                # The signal is always read 5' to 3' relative to the strand in the pore.
                # If the alignment was to the reverse strand, we must reverse-complement
                # the reference sequence to match the signal's direction.
                if record.is_reverse:
                    ground_truth_sequence = reverse_complement(ground_truth_sequence)
                
                # Ensure signal and sequence are not empty
                if raw_signal is not None and ground_truth_sequence:
                    yield (raw_signal, ground_truth_sequence)

            except Exception as e:
                print(f"Warning: Failed to process read {read_id}. Error: {e}")


# --- Example Usage ---
if __name__ == "__main__":
    # Define paths to your data
    BAM_FILE_PATH = "basecalls.sorted.bam" # Your sorted and indexed BAM from Bonito
    FAST5_DIR_PATH = "/path/to/your/fast5/data"

    # Create the dataset generator
    dataset_generator = create_training_dataset_generator(BAM_FILE_PATH, FAST5_DIR_PATH)

    # Now you can use this generator to feed data into your model's training loop
    # Let's just inspect the first 5 samples
    print("\n--- Inspecting first 5 data samples ---")
    for i, (signal, sequence) in enumerate(dataset_generator):
        if i >= 5:
            break
        print(f"\nSample {i+1}:")
        print(f"  - Signal shape: {signal.shape}, Signal dtype: {signal.dtype}")
        print(f"  - Signal snippet: {signal[:15]}...")
        print(f"  - Ground truth sequence (len={len(sequence)}): {sequence[:50]}...")