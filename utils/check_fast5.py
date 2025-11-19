from ont_fast5_api.fast5_interface import get_fast5_file

def print_all_raw_data(fast5_filepath):
    with get_fast5_file(fast5_filepath, mode="r") as f5:
        for read in f5.get_reads():
            raw_data = read.get_raw_data()
            print(read.read_id, raw_data, raw_data.shape)
            tracking = read.get_tracking_id()    # 字典，包含 flow cell / device 信息
            channel = read.get_channel_info()    # 字典，包含 digitisation / range / offset / channel_number 等
            print("tracking_id attrs:", tracking)
            print("channel_id attrs:", channel)
            break
            # print all the information in the read
        
print_all_raw_data('/data/biolab-backup-hdd1/zhuai-hdd/20230413_293T_total_RNA_drs/fast5/FAW44582_c3dba87c_68.fast5')