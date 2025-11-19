import os
import h5py
import pysam
import pod5
import numpy as np
import argparse
import atexit
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
import time

# --- [配置参数] ---
SIGNAL_LENGTH = 2048 
WINDOW_STRIDE = 1024
HDF5_WRITE_CHUNK_SIZE = 1024 
MAX_LABEL_LEN = 200

# --- [关键修改 1: 碱基映射更新] ---
# N 和 n 映射为 0，通常作为 Padding 或 Blank
BASE_TO_INT = {
    'A': 1, 'C': 2, 'G': 3, 'T': 4,
    'a': 1, 'c': 2, 'g': 3, 't': 4,
    'N': 0, 'n': 0
}
PAD_VAL = 0  # 对应 BASE_TO_INT 中的 N/Blank

# --- 全局变量 (用于多进程继承) ---
global_fasta_handle = None
global_pod5_lookup = None
global_pod5_reader_cache = {}


def _close_cached_pod5_readers():
    for reader in global_pod5_reader_cache.values():
        try:
            reader.close()
        except Exception:
            pass


atexit.register(_close_cached_pod5_readers)

def worker_init(fasta_path):
    """初始化工作进程的 FASTA 句柄"""
    global global_fasta_handle
    global_fasta_handle = pysam.FastaFile(fasta_path)

def process_task(task_data):
    """
    处理单个 Read 的核心函数。
    修正了 Stride 解析和 TS 起始点对齐逻辑。
    """
    # 1. 解包任务数据 (注意：ts_tag 对应 template_start_offset)
    read_id_str, ref_name, ref_start, ref_end, ts_tag, mv_tag = task_data
    
    worker_stats = {
        "id_not_in_pod5_index": 0, "read_not_found_in_pod5_file": 0,
        "missing_tags": 0, "signal_too_short": 0, "total_windows_processed": 0,
        "window_mad_is_zero": 0, "window_no_bases": 0, "window_label_invalid": 0,
        "dbg_label_is_empty": 0,
        "dbg_label_is_too_long": 0,
        "valid_samples_created": 0,
    }
    samples_list = [] 

    global global_fasta_handle
    global global_pod5_lookup
    global global_pod5_reader_cache
    
    try:
        if read_id_str not in global_pod5_lookup:
            worker_stats["id_not_in_pod5_index"] += 1
            return samples_list, worker_stats

        pod5_path, batch_idx, row_idx = global_pod5_lookup[read_id_str]
        
        # 2. 获取参考序列
        # 注意：BAM 中的序列可能包含 soft clipping，这里最好直接用 BAM query sequence 
        # 如果你坚持用 FASTA 参考序列，请确保 ref_start/end 与信号是完全对齐的。
        # **修正建议**：训练 basecaller 通常使用 BAM 中的 query_sequence (因为它是实际测到的序列)，
        # 但既然你传了 ref 坐标，这里保留你原本的逻辑读取 FASTA。
        ground_truth_label_str = global_fasta_handle.fetch(ref_name, ref_start, ref_end).upper()

        # 🚀 ==========================================================
        # 🚀 [关键修改 2 & 3]：Move Table 解析与坐标计算
        # ==========================================================
        
        # A. 解析 MV 标签
        raw_mv = np.array(mv_tag, dtype=np.int64)
        
        # 自动获取 Stride (根据你的发现，mv[0] 是 6)
        stride = raw_mv[0] 
        
        # 获取实际的 moves (0/1 序列)
        moves = raw_mv[1:]
        
        # B. 找到所有发生碱基转换的时间步 (Frame Indices)
        # np.flatnonzero(moves) 返回的是 moves 数组中值为 1 的索引位置
        # 例如: moves=[1, 0, 1] -> indices=[0, 2]
        base_frame_indices = np.flatnonzero(moves)
        
        # C. 结合 TS 标签计算绝对采样点坐标
        # ts_tag: Read 在原始信号中的绝对起始点
        # 公式: 绝对坐标 = TS + (Frame_Index * Stride)
        if ts_tag is None:
            # 如果没有 ts 标签，回退到 0 (但在你的数据中应该都有)
            ts_offset = 0
        else:
            ts_offset = ts_tag

        base_signal_starts_absolute = ts_offset + (base_frame_indices * stride)

        # 完整性检查：确保计算出的碱基数量与序列长度大致匹配
        # len(base_signal_starts_absolute) 应该等于 (或非常接近) len(ground_truth_label_str)
        # 如果你是从 FASTA 获取的序列，可能会有 Indel 导致的长度差异，这里不做强行 Assert，但请留意。

        # 🚀 ==========================================================

        # 3. 打开 POD5 读取信号
        reader = global_pod5_reader_cache.get(pod5_path)
        if reader is None:
            reader = pod5.Reader(pod5_path)
            global_pod5_reader_cache[pod5_path] = reader

        batch = reader.get_batch(batch_idx)
        pod5_read = batch.get_read(row_idx)

        if pod5_read is None:
            worker_stats["read_not_found_in_pod5_file"] += 1
            return samples_list, worker_stats

        raw_signal = pod5_read.signal

        if len(raw_signal) < SIGNAL_LENGTH:
            worker_stats["signal_too_short"] += 1
            return samples_list, worker_stats

        # 4. 滑动窗口处理
        # 这里的逻辑是：我们在 raw_signal 上滑动，切出一段信号
        # 然后查看 base_signal_starts_absolute 中有哪些点落在这个窗口内
            
        total_bases = len(base_signal_starts_absolute)
        left_idx = 0
        right_idx = 0

        for win_start in range(0, len(raw_signal) - SIGNAL_LENGTH, WINDOW_STRIDE):
            worker_stats["total_windows_processed"] += 1
            win_end = win_start + SIGNAL_LENGTH

            signal_window = raw_signal[win_start:win_end]

            # 归一化
            median = np.median(signal_window)
            mad = np.median(np.abs(signal_window - median))

            if mad == 0:
                worker_stats["window_mad_is_zero"] += 1
                continue

            normalized_signal = (signal_window - median) / mad

            # 5. 标签对齐 (Label Alignment)
            # 增量移动指针，避免对 searchsorted 的重复调用
            while left_idx < total_bases and base_signal_starts_absolute[left_idx] <= win_start:
                left_idx += 1
            while right_idx < total_bases and base_signal_starts_absolute[right_idx] < win_end:
                right_idx += 1

            first_base_idx = left_idx
            last_base_idx = right_idx

            if first_base_idx >= last_base_idx:
                worker_stats["window_no_bases"] += 1
                continue

            # 切片获取对应的碱基序列
            # 注意：如果 ref_seq 长度与 mv 推导出的 bases 数量不一致，这里可能会越界，加个保护
            current_ref_len = len(ground_truth_label_str)
            safe_last = min(last_base_idx, current_ref_len)

            if first_base_idx >= safe_last:
                continue

            label_str_window = ground_truth_label_str[first_base_idx:safe_last]

            # 转换字符到整数
            label_int_window = [BASE_TO_INT[b] for b in label_str_window if b in BASE_TO_INT]

            if not label_int_window:
                worker_stats["dbg_label_is_empty"] += 1
                worker_stats["window_label_invalid"] += 1
                continue

            if len(label_int_window) > MAX_LABEL_LEN:
                worker_stats["dbg_label_is_too_long"] += 1
                worker_stats["window_label_invalid"] += 1
                continue

            # 6. Padding (使用 0 填充)
            padded_label = np.full((MAX_LABEL_LEN,), PAD_VAL, dtype=np.int32)
            padded_label[:len(label_int_window)] = label_int_window

            normalized_signal = normalized_signal.reshape(1, SIGNAL_LENGTH)

            samples_list.append((normalized_signal, padded_label, len(label_int_window)))
            worker_stats["valid_samples_created"] += 1

    except KeyError:
        worker_stats["missing_tags"] += 1
    except Exception as e:
        # 捕获其他潜在错误防止进程崩溃
        # print(f"Error processing {read_id_str}: {e}") 
        pass
    
    return samples_list, worker_stats


def write_chunk_to_hdf5(datasets, chunk):
    if not chunk:
        return
    event_ds, label_ds, label_len_ds = datasets
    current_size = event_ds.shape[0]
    new_size = current_size + len(chunk)
    
    event_ds.resize(new_size, axis=0)
    label_ds.resize(new_size, axis=0)
    label_len_ds.resize(new_size, axis=0)
    
    # 预分配 numpy 数组以加速写入
    chunk_len = len(chunk)
    signals = np.zeros((chunk_len, 1, SIGNAL_LENGTH), dtype=np.float32)
    labels = np.zeros((chunk_len, MAX_LABEL_LEN), dtype=np.int32)
    lengths = np.zeros((chunk_len,), dtype=np.int32)

    for i, (sig, lab, length) in enumerate(chunk):
        signals[i] = sig
        labels[i] = lab
        lengths[i] = length

    event_ds[current_size:new_size] = signals
    label_ds[current_size:new_size] = labels
    label_len_ds[current_size:new_size] = lengths


def consume_completed_futures(completed_futures, futures_set, total_stats, results_chunk, hdf5_datasets):
    if not completed_futures:
        return results_chunk

    for future in completed_futures:
        futures_set.remove(future)
        samples_list, worker_stats = future.result()

        for key, value in worker_stats.items():
            total_stats[key] += value

        if samples_list:
            results_chunk.extend(samples_list)

    if len(results_chunk) >= HDF5_WRITE_CHUNK_SIZE:
        write_chunk_to_hdf5(hdf5_datasets, results_chunk)
        return []

    return results_chunk


def main(args):
    global global_pod5_lookup
    
    total_stats = {
        "bam_reads_processed": 0, "id_not_in_pod5_index": 0, "read_not_found_in_pod5_file": 0,
        "missing_tags": 0, "signal_too_short": 0, "total_windows_processed": 0,
        "window_mad_is_zero": 0, "window_no_bases": 0, "window_label_invalid": 0,
        "dbg_label_is_empty": 0,
        "dbg_label_is_too_long": 0,
        "valid_samples_created": 0, "tasks_submitted": 0
    }
    
    start_time = time.time()

    print("Step 1: Building detailed (path, batch, row) index from POD5 files...")
    pod5_files = [os.path.join(args.pod5_dir, f) for f in os.listdir(args.pod5_dir) if f.endswith('.pod5')]
    
    global_pod5_lookup = {} 
    
    for pod5_path in tqdm(pod5_files, desc="Indexing POD5 files"):
        with pod5.Reader(pod5_path) as reader:
            for batch_idx in range(reader.batch_count):
                batch = reader.get_batch(batch_idx)
                for row_idx in range(batch.num_reads): 
                    read_record = batch.get_read(row_idx) 
                    read_id_str = str(read_record.read_id) 
                    global_pod5_lookup[read_id_str] = (pod5_path, batch_idx, row_idx)
                    
    print(f"Indexed {len(global_pod5_lookup)} unique reads from POD5 files.")

    print("Step 2: Setting up HDF5 file and process pool...")
    bam_file = pysam.AlignmentFile(args.bam_file, "rb")
    # 有些 BAM 没有 mapped 属性，或者非常大，用 try-except 更稳健
    try:
        bam_file_size = bam_file.mapped if bam_file.mapped > 0 else 100000
    except:
        bam_file_size = 100000 # Dummy value for tqdm
    
    max_workers = args.workers
    MAX_QUEUE_SIZE = max_workers * 10
    
    print(f"Using {max_workers} worker processes.")

    with h5py.File(args.output_hdf5, 'w') as hf:
        event_ds = hf.create_dataset('event', (0, 1, SIGNAL_LENGTH), maxshape=(None, 1, SIGNAL_LENGTH), dtype=np.float32)
        label_ds = hf.create_dataset('label', (0, MAX_LABEL_LEN), maxshape=(None, MAX_LABEL_LEN), dtype=np.int32)
        label_len_ds = hf.create_dataset('label_len', (0,), maxshape=(None,), dtype=np.int32)
        
        hdf5_datasets = (event_ds, label_ds, label_len_ds)
        results_chunk = []

        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=worker_init,
            initargs=(args.reference_fasta,)
        ) as executor:

            futures = set()
            print("Step 3 & 4: Submitting tasks and consuming results...")
            
            # 迭代 BAM 文件
            for read in tqdm(bam_file, desc="Processing Reads"):
                total_stats["bam_reads_processed"] += 1
                
                try:
                    # 检查是否有必要的标签
                    if not read.has_tag('mv') or not read.has_tag('ts'):
                        total_stats["missing_tags"] += 1
                        continue

                    task_data = (
                        read.query_name,
                        read.reference_name,
                        read.reference_start,
                        read.reference_end,
                        read.get_tag('ts'), # 传递 TS 标签
                        read.get_tag('mv')  # 传递 MV 标签
                    )
                    
                    # 移除了 stride 参数，因为现在从 mv[0] 自动获取
                    futures.add(executor.submit(process_task, task_data))
                    total_stats["tasks_submitted"] += 1
                    
                except Exception as e:
                    # print(f"Skipping read due to error: {e}")
                    continue
                
                # 消费者逻辑
                while len(futures) >= MAX_QUEUE_SIZE:
                    done_futures, _ = wait(futures, timeout=0, return_when=FIRST_COMPLETED)
                    if not done_futures:
                        done_futures, _ = wait(futures, return_when=FIRST_COMPLETED)

                    results_chunk = consume_completed_futures(
                        done_futures,
                        futures,
                        total_stats,
                        results_chunk,
                        hdf5_datasets,
                    )

                # 抢先消费已经完成的任务，避免在主循环末尾堆积
                if futures:
                    done_futures, _ = wait(futures, timeout=0, return_when=FIRST_COMPLETED)
                    results_chunk = consume_completed_futures(
                        done_futures,
                        futures,
                        total_stats,
                        results_chunk,
                        hdf5_datasets,
                    )

            # Step 5: 处理剩余任务
            print("Step 5: Consuming remaining tasks...")
            remaining_futures = list(futures)
            futures.clear()
            for future in tqdm(as_completed(remaining_futures), total=len(remaining_futures)):
                samples_list, worker_stats = future.result()
                for key, value in worker_stats.items():
                    total_stats[key] += value
                
                if samples_list:
                    results_chunk.extend(samples_list)
                
                if len(results_chunk) >= HDF5_WRITE_CHUNK_SIZE:
                    write_chunk_to_hdf5(hdf5_datasets, results_chunk)
                    results_chunk = [] 

            if results_chunk:
                write_chunk_to_hdf5(hdf5_datasets, results_chunk)

    bam_file.close()
    end_time = time.time()
    print("\n--- PROCESSING FINISHED ---")
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Final valid samples created: {total_stats['valid_samples_created']}")
    print("\n--- DETAILED STATISTICS REPORT ---")
    for key, value in total_stats.items():
        print(f"{key:<30}: {value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bam_file", type=str, required=True)
    parser.add_argument("--pod5_dir", type=str, required=True)
    parser.add_argument("--reference_fasta", type=str, required=True)
    parser.add_argument("--output_hdf5", type=str, required=True)
    parser.add_argument("--workers", type=int, default=8)
    # 移除了 basecaller-stride 参数，因为代码现在会自动识别
    args = parser.parse_args()
    
    main(args)