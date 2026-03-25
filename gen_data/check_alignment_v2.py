import torch
import numpy as np
import argparse
import os
import sys
from bonito.util import load_model, load_symbol
from difflib import SequenceMatcher

# 模拟 Bonito 的 Read 对象
class MockRead:
    def __init__(self, signal, idx, mean, std):
        self.read_id = f"sample_{idx}"
        
        # === 核心魔法：反向归一化 ===
        # 数据集里的 signal 是已经 (x - mean) / std 过的
        # Basecall 流程会再次执行 (x - mean) / std
        # 所以我们需要先还原成 Raw pA： raw = signal * std + mean
        raw_signal = signal * std + mean
        
        self.signal = raw_signal.astype(np.float32) 
        self.run_id = "validation"
        self.filename = "chunks.npy"
        self.mux = 1
        self.channel = 1
        self.start = 0.0
        self.duration = len(signal) / 5000.0
        self.template_start = 0.0
        self.template_duration = self.duration
        self.num_samples = len(signal)
        self.trimmed_samples = 0
        self.calibration_scale = 1.0 # 模拟相关属性
        self.calibration_offset = 0.0

def decode_label(int_arr, alphabet):
    return "".join([alphabet[i] for i in int_arr if i != 0])

def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--custom_dir", required=True, help="Path to V8 dataset")
    parser.add_argument("--model_path", required=True, help="Path to bonito model directory")
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()

    # 1. Load Dataset
    print(f"> Loading Dataset from {args.custom_dir}")
    try:
        chunks = np.load(os.path.join(args.custom_dir, "chunks.npy"), mmap_mode='r')
        refs = np.load(os.path.join(args.custom_dir, "references.npy"), mmap_mode='r')
        lens = np.load(os.path.join(args.custom_dir, "reference_lengths.npy"), mmap_mode='r')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # 2. Load Model & Extract Config
    print(f"> Loading Model from {args.model_path}")
    device = torch.device(args.device)
    try:
        model = load_model(args.model_path, device, half=True)
        basecall_func = load_symbol(args.model_path, "basecall")
        
        # 获取 Normalization 参数用于反转
        norm_conf = model.config.get("standardisation", {})
        pa_mean = norm_conf.get("mean", 0.0)
        pa_std = norm_conf.get("stdev", 1.0)
        
        # 获取 Alphabet
        alphabet = getattr(model, 'alphabet', ['N', 'A', 'C', 'G', 'T'])
        
        print(f"> PA Norm Params found: Mean={pa_mean}, Std={pa_std}")
        
        # 强制覆盖 chunksize，确保不会把我们的 chunk 再切碎
        chunksize = chunks.shape[1] # 应该是 12000

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 3. Prepare Mock Reads (With Inverse Norm)
    total_chunks = chunks.shape[0]
    indices = np.random.choice(total_chunks, args.samples, replace=False)
    
    mock_reads = []
    for idx in indices:
        # 传入 mean/std 进行反向归一化
        mock_reads.append(MockRead(chunks[idx], idx, pa_mean, pa_std))

    print("\n" + "="*100)
    print(f"{'IDX':<8} | {'SIM%':<6} | {'RC_SIM%':<7} | {'STATUS':<12}")
    print("="*100)

    # 4. Run Basecalling
    with torch.no_grad():
        # overlap=0 避免拼接逻辑干扰
        results_iter = basecall_func(
            model, 
            mock_reads, 
            batchsize=16, 
            chunksize=chunksize, 
            overlap=0 
        )
        
        results_map = {}
        for res_read, res_dict in results_iter:
            original_idx = int(res_read.read_id.split('_')[-1])
            results_map[original_idx] = res_dict['sequence']

    # 5. Compare
    for idx in indices:
        if idx not in results_map:
            print(f"{idx:<8} | ERROR  | No result from basecaller")
            continue
            
        pred_seq = results_map[idx]
        ref_len = lens[idx]
        label_seq = decode_label(refs[idx][:ref_len], alphabet)
        
        # Calc Similarity
        sim_fwd = similarity(pred_seq, label_seq) * 100
        
        # Check Reverse Complement
        rc_map = str.maketrans("ACGT", "TGCA")
        label_rc = label_seq.translate(rc_map)[::-1]
        sim_rc = similarity(pred_seq, label_rc) * 100
        
        status = "OK" if sim_fwd > 85 else "BAD"
        if sim_rc > sim_fwd + 20: status = "REV_COMP?"
        elif sim_fwd < 50 and sim_rc < 50: status = "SHIFT/WRONG"

        print(f"{idx:<8} | {sim_fwd:5.1f} | {sim_rc:6.1f}  | {status:<12}")
        
        if status != "OK":
             print(f"  Pred:  {pred_seq[:60]}...")
             print(f"  Label: {label_seq[:60]}...")
             if status == "REV_COMP?":
                 print(f"  RC:    {label_rc[:60]}...")
        else:
             print(f"  Match: {pred_seq[:30]}... == {label_seq[:30]}...")

        print("-" * 100)

if __name__ == "__main__":
    main()