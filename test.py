import pod5, os
pod5_dir = '/data/biolab-nvme-pool1/zhaoxy/R10_data/giab_lsk_2023.05_hg002/sample/pod5'
# Get the first pod5 file in the directory
first_pod5_file = os.path.join(pod5_dir, os.listdir(pod5_dir)[0])
print(f"--- First 5 Read IDs from POD5 file: {first_pod5_file} ---")
with pod5.Reader(first_pod5_file) as reader:
    for i, read in enumerate(reader.reads()):
        if i >= 5: break
        print(read.read_id)