import torch
import numpy as np
import argparse
import os
import sys
from bonito.util import load_model, load_symbol

# 模拟 Bonito 的 Read 对象
class MockRead:
    def __init__(self, signal, idx):
        self.read_id = f"verification_sample_{idx}"
        # 确保是 float32，避免 float16 在某些操作中报错
        self.signal = signal.astype(np.float32) 
        self.run_id = "test_run"
        self.filename = "chunks.npy"
        self.mux = 1
        self.channel = 1
        self.start = 0.0
        self.duration = len(signal) / 5000.0 # 假设 5khz
        self.template_start = 0.0
        self.template_duration = self.duration
        self.num_samples = len(signal)
        # 兼容性字段
        self.trimmed_samples = 0

def decode_label(int_arr, alphabet):
    """
    解码 Label: 0=Pad, 1=A, 2=C... (假设 custom dataset 是 1-based)
    Alphabet: ['N', 'A', 'C', 'G', 'T']
    """
    return "".join([alphabet[i] for i in int_arr if i != 0])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_dir", required=True, help="Path to your v5 dataset")
    parser.add_argument("--model_path", required=True, help="Path to bonito model directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--samples", type=int, default=10, help="Number of chunks to verify")
    args = parser.parse_args()

    # 1. Load Data
    print(f"> Loading Dataset from {args.custom_dir}")
    try:
        chunks = np.load(os.path.join(args.custom_dir, "chunks.npy"), mmap_mode='r')
        refs = np.load(os.path.join(args.custom_dir, "references.npy"), mmap_mode='r')
        lens = np.load(os.path.join(args.custom_dir, "reference_lengths.npy"), mmap_mode='r')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load Model & Config
    print(f"> Loading Model from {args.model_path}")
    device = torch.device(args.device)
    try:
        # 加载模型对象
        model = load_model(args.model_path, device, half=True)
        # 动态加载 basecall 函数 (这是最稳健的方法)
        basecall_func = load_symbol(args.model_path, "basecall")
        
        # 获取配置参数
        config = model.config["basecaller"]
        chunksize = config.get("chunksize", 4000)
        overlap = config.get("overlap", 500)
        batchsize = config.get("batchsize", 32)
        
        # 获取 Alphabet
        alphabet = getattr(model, 'alphabet', ['N', 'A', 'C', 'G', 'T'])
        print(f"> Model loaded. Chunksize: {chunksize}, Overlap: {overlap}")
        print(f"> Alphabet: {alphabet}")

    except Exception as e:
        print(f"Error loading model or basecall function: {e}")
        import traceback
        traceback.print_exc()
        return

    # 3. Select Samples
    total_chunks = chunks.shape[0]
    indices = np.random.choice(total_chunks, args.samples, replace=False)
    
    # 构造 Mock Reads 列表
    mock_reads = []
    for idx in indices:
        # 注意：这里我们传入整个 chunk。
        # 如果模型 chunksize == dataset chunk_len，bonito 会把它当作一个 chunk 处理。
        mock_reads.append(MockRead(chunks[idx], idx))

    print("\n" + "="*100)
    print(f"{'IDX':<6} | {'STATUS':<12} | {'SEQ COMPARISON (Top: Model Pred, Bottom: Your Label)'}")
    print("="*100)

    # 4. Run Basecalling (Inference)
    # 这是一个生成器，我们直接迭代它
    with torch.no_grad():
        # 调用 bonito 的 basecall 接口
        # 注意：这里的 reads 已经是 MockRead 对象，里面包含了 signal
        # 我们不需要传 reader，直接传 list
        results_iter = basecall_func(
            model, 
            mock_reads, 
            batchsize=batchsize, 
            chunksize=chunksize, 
            overlap=overlap
        )
        
        # 收集结果并建立索引映射
        results_map = {}
        for res_read, res_dict in results_iter:
            # 提取原始 idx
            original_idx = int(res_read.read_id.split('_')[-1])
            results_map[original_idx] = res_dict['sequence']

    # 5. Compare Results
    for idx in indices:
        if idx not in results_map:
            print(f"{idx:<6} | {'ERROR':<12} | Basecaller returned no result for this chunk.")
            continue
            
        pred_seq = results_map[idx]
        
        # Decode Ground Truth
        ref_len = lens[idx]
        label_seq = decode_label(refs[idx][:ref_len], alphabet)

        # Compare First 50 bases
        check_len = min(50, len(pred_seq), len(label_seq))
        if check_len == 0:
            print(f"{idx:<6} | {'EMPTY':<12} | Pred or Label is empty.")
            continue

        match = "MATCH" if pred_seq[:check_len] == label_seq[:check_len] else "MISMATCH"
        
        # Diagnosis Logic
        rc_map = str.maketrans("ACGT", "TGCA")
        label_rc = label_seq.translate(rc_map)[::-1]
        label_rev = label_seq[::-1]
        
        hint = ""
        if match == "MISMATCH":
            if pred_seq[:check_len] == label_rc[:check_len]:
                match = "REV_COMP"
                hint = "Label is Reverse Complemented (Did you forget to flip reverse strand?)"
            elif pred_seq[:check_len] == label_rev[:check_len]:
                match = "REVERSED"
                hint = "Label is Reversed but NOT Complemented"
            elif pred_seq[:check_len] in label_seq:
                match = "SHIFT(L)"
                hint = "Prediction is substring of Label (Soft-clip/Trim offset issue)"
            elif label_seq[:check_len] in pred_seq:
                match = "SHIFT(P)"
                hint = "Label is substring of Prediction"
            else:
                # 检查中间是否匹配（Indel 导致的严重错位）
                mid = len(pred_seq) // 2
                if mid > 20 and pred_seq[mid:mid+20] in label_seq:
                     match = "INDEL/GAP"
                     hint = "Start mismatched, but middle matched (Indel/Gap issue)"

        print(f"{idx:<6} | {match:<12} | Pred:  {pred_seq[:80]}...")
        print(f"{'':<6} | {'':<12} | Label: {label_seq[:80]}...")
        if hint:
             print(f"{'':<6} | {'':<12} | [Diagnosis] {hint}")
        elif match == "MISMATCH":
             print(f"{'':<6} | {'':<12} | [Debug] Label RC would be: {label_rc[:80]}...")
        
        print("-" * 100)

if __name__ == "__main__":
    main()