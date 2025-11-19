import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
import h5py
import os
import argparse
from tqdm import tqdm
import numpy as np
import random
import editdistance  # 用于计算 Levenshtein 距离
import matplotlib.pyplot as plt

# --- 关键导入 ---
# 确保 'model.py' (或您保存它的地方) 可以在 Python 路径中被找到
import sys
try:
    sys.path.append('/home/lijy/workspace/') # 替换为您的项目路径
    from my_basecaller.model.transcaller_light import TranscallerLight
except ImportError:
    print("="*80)
    print("错误: 无法导入 'TranscallerLight'。")
    print("请确保您的模型代码在 Python 路径中。")
    print("="*80)
    exit(1)

# ==========================================================================================
# 步骤 1: HDF5 数据集类 (与 train.py 相同)
# ==========================================================================================

class BasecallerHDF5Dataset(Dataset):
    """
    用于读取 HDF5 格式的 Basecaller 数据集的自定义 Dataset。
    """
    def __init__(self, h5_file_path):
        super().__init__()
        self.h5_file_path = h5_file_path
        
        try:
            with h5py.File(self.h5_file_path, 'r') as f:
                self.dataset_len = f['event'].shape[0]
        except Exception as e:
            print(f"打开或验证 HDF5 文件 {h5_file_path} 失败: {e}")
            raise
            
        self.h5_file = None
        self.pid = None 

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if self.pid != os.getpid():
            if self.h5_file:
                self.h5_file.close() 
            self.h5_file = h5py.File(self.h5_file_path, 'r')
            self.pid = os.getpid()
            
        # 仅在评估时，我们才需要所有数据
        event = self.h5_file['event'][idx] 
        label = self.h5_file['label'][idx] 
        label_len = self.h5_file['label_len'][idx]
        
        event_tensor = torch.from_numpy(event).float()
        label_tensor = torch.from_numpy(label).long()
        label_len_tensor = torch.tensor(label_len).long()
        
        return event_tensor, label_tensor, label_len_tensor

# ==========================================================================================
# 步骤 2: CTC 贪婪解码器
# ==========================================================================================

def greedy_decode(log_probs, blank_id=4):
    """
    执行 CTC 贪婪解码 (Best Path Decoding)。

    Args:
        log_probs (Tensor): 模型的输出 (T, B, C)
        blank_id (int): 空白标签的索引

    Returns:
        list[list[int]]: 解码后的 batch，(B, Seq)
    """
    
    # 1. 找到每个时间步概率最高的 token
    # (T, B, C) -> (T, B)
    best_path = torch.argmax(log_probs, dim=-1)
    
    decoded_batch = []
    batch_size = best_path.shape[1]
    
    # 2. 遍历 batch 中的每个样本
    for i in range(batch_size):
        seq = best_path[:, i]
        
        # 3. 折叠重复的 token
        # [0, 0, 1, 1, 1, 0, 2, 2] -> [0, 1, 0, 2]
        collapsed_seq = []
        last_token = -1
        for token in seq:
            if token.item() != last_token:
                collapsed_seq.append(token.item())
                last_token = token.item()
                
        # 4. 移除 blank token
        # [0, 1, 4, 0, 2, 4] -> [0, 1, 0, 2]
        final_seq = [t for t in collapsed_seq if t != blank_id]
        decoded_batch.append(final_seq)
        
    return decoded_batch

# ==========================================================================================
# 步骤 3: 评估函数
# ==========================================================================================

def evaluate_accuracy(model, data_loader, device, output_len, blank_id):
    """
    在测试集上计算 Read Accuracy 和 Base Accuracy。
    """
    model.eval()
    
    total_reads = 0
    correct_reads = 0
    total_edits = 0
    total_base_len = 0
    
    progress_bar = tqdm(data_loader, desc='[评估中]', leave=True)
    
    with torch.no_grad():
        for events, labels, label_lengths in progress_bar:
            events = events.to(device, non_blocking=True)
            # labels 和 label_lengths 保留在 CPU 上，因为解码和比较在 CPU 上进行
            
            # 1. 前向传播
            log_probs = model(events) # (T, B, C)
            
            # 2. CTC 贪婪解码
            # (T, B, C) -> list[list[int]] (长度为 B)
            decoded_batch = greedy_decode(log_probs, blank_id)
            
            # 3. 逐个样本比较
            for i in range(len(decoded_batch)):
                true_label_ids = labels[i][:label_lengths[i]].tolist()
                pred_label_ids = decoded_batch[i]
                
                # 3a. 计算 Read Accuracy
                if true_label_ids == pred_label_ids:
                    correct_reads += 1
                
                # 3b. 计算 Base Accuracy (使用 Levenshtein 距离)
                edits = editdistance.eval(true_label_ids, pred_label_ids)
                total_edits += edits
                total_base_len += len(true_label_ids)
                
                total_reads += 1
    
    read_accuracy = (correct_reads / total_reads) * 100
    # Base Accuracy = 1.0 - (编辑距离 / 真实长度)
    base_accuracy = (1.0 - (total_edits / total_base_len)) * 100
    
    return read_accuracy, base_accuracy

# ==========================================================================================
# 步骤 4: 可视化函数
# ==========================================================================================

def visualize_one_sample(model, data_loader, device, token_map, blank_id, output_path):
    """
    运行一个 batch，并可视化第一个样本。
    """
    model.eval()
    
    print("\n" + "="*80)
    print("生成单一样本可视化...")
    
    with torch.no_grad():
        # 1. 获取一个 batch
        try:
            events, labels, label_lengths = next(iter(data_loader))
        except StopIteration:
            print("错误: data_loader 为空。")
            return
            
        events = events.to(device)
        
        # 2. 运行模型
        log_probs = model(events) # (T, B, C)
        
        # 3. 选择第一个样本进行分析
        i = 0 
        sample_log_probs = log_probs[:, i, :] # (T, C)
        
        # 4. 解码第一个样本
        pred_label_ids = greedy_decode(log_probs, blank_id)[i]
        
        # 5. 获取真实标签
        true_label_ids = labels[i][:label_lengths[i]].tolist()
        
        # 6. 将 ID 转换为字符串
        # token_map = {0: 'A', 1: 'C', 2: 'G', 3: 'T', 4: '<B>'}
        true_str = "".join([token_map.get(t, '?') for t in true_label_ids])
        pred_str = "".join([token_map.get(t, '?') for t in pred_label_ids])
        
        # 7. (输出 1) 打印文本对比
        print(f"样本 {i} 结果:")
        print(f"  真实 (True): {true_str}")
        print(f"  预测 (Pred): {pred_str}")
        
        # 8. (输出 2) 绘制概率图
        # (T, C) -> (C, T)
        probs = torch.exp(sample_log_probs).cpu().numpy().T
        
        plt.figure(figsize=(20, 6))
        plt.imshow(probs, aspect='auto', interpolation='nearest', cmap='viridis')
        
        # 设置 Y 轴标签
        plt.yticks(ticks=range(len(token_map)), labels=token_map.values())
        
        plt.xlabel("模型输出时间步 (Timestep)")
        plt.ylabel("碱基")
        plt.title("Basecalling 概率图 (单一样本)")
        plt.colorbar(label="概率")
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"可视化图像已保存至: {output_path}")
        print("="*80)

# ==========================================================================================
# 步骤 5: 主函数
# ==========================================================================================

def main(args):
    
    # --- 1. 设置环境 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 定义 token 映射
    # 确保这与您的数据预处理一致！
    TOKEN_MAP = {0: 'A', 1: 'C', 2: 'G', 3: 'T', args.blank_id: '<B>'}

    # --- 2. 准备数据集 ---
    print("加载测试数据集中...")
    test_dataset_full = BasecallerHDF5Dataset(args.data_file)
    
    if args.num_samples > 0:
        num_samples = min(len(test_dataset_full), args.num_samples)
        print(f"使用 {num_samples} 个随机样本进行测试...")
        # 随机抽取子集
        indices = torch.randperm(len(test_dataset_full))[:num_samples]
        test_dataset = Subset(test_dataset_full, indices)
    else:
        print(f"使用完整测试集: {len(test_dataset_full)} 个样本")
        test_dataset = test_dataset_full
        
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False, # 评估时不需要打乱
        num_workers=args.num_workers,
        pin_memory=True
    )

    # --- 3. 初始化模型 ---
    print("初始化模型...")
    model = TranscallerLight(
        input_length=args.input_len,
        output_length=args.output_len,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depth=args.depth,
        num_heads=args.num_heads,
        mlp_ratio=args.mlp_ratio, # 确保添加了 mlp_ratio
        drop_path_rate=0.0 # 评估时关闭 drop_path
    ).to(device)
    
    # --- 4. 加载 Checkpoint ---
    if not os.path.exists(args.checkpoint):
        print(f"错误: 找不到 Checkpoint 文件: {args.checkpoint}")
        return
        
    print(f"加载 Checkpoint: {args.checkpoint}")
    try:
        # train.py 保存的是 state_dict
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    except RuntimeError as e:
        print(f"错误: 加载 state_dict 失败。")
        print("这通常意味着您的模型架构参数 (embed_dim, depth, heads) 与 checkpoint 不匹配。")
        print(f"Pytorch 错误: {e}")
        return
        
    print(f"模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- 5. 运行可视化 (如果需要) ---
    if args.visualize:
        visualize_one_sample(
            model, 
            test_loader, 
            device, 
            TOKEN_MAP, 
            args.blank_id, 
            args.vis_output
        )

    # --- 6. 运行完整评估 ---
    print("开始计算准确性...")
    read_acc, base_acc = evaluate_accuracy(
        model, 
        test_loader, 
        device, 
        args.output_len, 
        args.blank_id
    )
    
    print("\n" + "="*80)
    print("评估完成!")
    print(f"  Read Accuracy (序列准确率): {read_acc:.2f} %")
    print(f"  Base Accuracy (碱基准确率): {base_acc:.2f} %")
    print("="*80)

# ==========================================================================================
# 步骤 6: Argparse 命令行参数
# ==========================================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="评估 TranscallerLight Basecaller 模型")
    
    # --- 关键路径 ---
    parser.add_argument('--data-file', type=str, required=True,
                        help="HDF5 *测试* 数据集文件路径")
    parser.add_argument('--checkpoint', type=str, default="./checkpoints/model_best.pth",
                        help="要评估的模型 .pth 文件路径")
    
    # --- (Utils) 数据集控制 ---
    parser.add_argument('--num-samples', type=int, default=-1,
                        help="要使用的测试样本数量。-1 表示使用全部。 (默认: -1)")
    
    # --- (Utils) 可视化 ---
    parser.add_argument('--visualize', action='store_true',
                        help="生成一个样本的可视化图表")
    parser.add_argument('--vis-output', type=str, default="basecalling_visualization.png",
                        help="可视化图表的输出文件名")

    # --- 评估超参数 ---
    parser.add_argument('--batch-size', type=int, default=128,
                        help="评估时的批量大小 (默认: 128)")
    parser.add_argument('--num-workers', type=int, default=8,
                        help="DataLoader 使用的进程数 (默认: 8)")

    # --- 模型架构参数 (必须与您训练的模型 *完全* 匹配!) ---
    # 
    # 🚀 注意: 我已将默认值修改为 'TranscallerLight' 的推荐值。
    # 如果您训练时使用了其他值, 请在此处明确指定。
    #
    parser.add_argument('--input-len', type=int, default=2048,
                        help="输入信号序列长度 (默认: 2048)")
    parser.add_argument('--output-len', type=int, default=420,
                        help="模型输出序列长度 (默认: 420)")
    parser.add_argument('--num-classes', type=int, default=5,
                        help="类别数 (A,C,G,T,blank) (默认: 5)")
    parser.add_argument('--blank-id', type=int, default=4,
                        help="CTCLoss 的空白标签 ID (默认: 4)")
    
    parser.add_argument('--embed-dim', type=int, default=384,
                        help="Transformer 嵌入维度 (默认: 384)")
    parser.add_argument('--depth', type=int, default=6,
                        help="Transformer 层数 (默认: 6)")
    parser.add_argument('--num-heads', type=int, default=4,
                        help="Transformer 注意力头数 (默认: 4)")
    parser.add_argument('--mlp-ratio', type=float, default=2.0,
                        help="MLP 隐藏层比例 (默认: 2.0)")
    
    args = parser.parse_args()
    
    # 打印所有配置
    print("="*80)
    print("评估配置:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")
    print("="*80)
    
    main(args)