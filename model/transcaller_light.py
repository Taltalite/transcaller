import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchinfo import summary
import warnings
from functools import partial
import collections.abc
from itertools import repeat

# ==========================================================================================
# 辅助函数和类 (与您提供的代码相同)
# ==========================================================================================

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return tuple(x)
        return tuple(repeat(x, n))
    return parse

to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    with torch.no_grad():
        return _trunc_normal_(tensor, mean, std, a, b)

def _trunc_normal_(tensor, mean, std, a, b):

    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    l = norm_cdf((a - mean) / std)
    u = norm_cdf((b - mean) / std)
    tensor.uniform_(2 * l - 1, 2 * u - 1)
    tensor.erfinv_()
    tensor.mul_(std * math.sqrt(2.))
    tensor.add_(mean)
    tensor.clamp_(min=a, max=b)
    return tensor

def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)
        linear_layer = partial(nn.Conv2d, kernel_size=1) if use_conv else nn.Linear
        self.fc1 = linear_layer(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.norm = norm_layer(hidden_features) if norm_layer is not None else nn.Identity()
        self.fc2 = linear_layer(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim 必须能被 num_heads 整除"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) 
        x = F.scaled_dot_product_attention(
            q, k, v, 
            dropout_p=self.attn_drop.p if self.training else 0.0 # 注意：仅在训练时启用dropout
        )
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., 
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, 
            attn_drop=attn_drop, proj_drop=drop
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim, 
            hidden_features=mlp_hidden_dim, 
            act_layer=act_layer, 
            drop=drop
        )
    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Head(nn.Module):
    def __init__(self, in_features, seq_len, out_seq_len, num_classes):
        super().__init__()
        # self.adaptive_pool = nn.AdaptiveAvgPool1d(out_seq_len)
        self.out_proj = nn.Linear(in_features, num_classes)
    def forward(self, x):
        # x = x.transpose(1, 2)
        # x = self.adaptive_pool(x)
        # x = x.transpose(1, 2)
        x = self.out_proj(x)
        return x.permute(1, 0, 2)

# ==========================================================================================
# 🚀 改进后的轻量化模块
# ==========================================================================================

class FastEmbedLight(nn.Module):
    """
    一个基于 1D 卷积的轻量化嵌入模块，增加了下采样功能。
    
    参数:
        in_chans (int): 输入通道数 (通常是 1)
        in_dim (int): 第一个卷积层的输出通道数 (中间维度)
        embed_dim (int): 最终嵌入维度 (Transformer 的隐藏维度 C)
        downsample_ratio (int): 下采样率 (目前硬编码为 4x)
    """
    def __init__(self, in_chans=1, in_dim=128, embed_dim=384, downsample_ratio=4):
        super().__init__()
        
        if downsample_ratio != 4:
            # 您也可以通过循环和参数来使其更灵活
            warnings.warn("这个 FastEmbedLight 版本目前硬编码为 4x 下采样")
            
        self.downsample_ratio = downsample_ratio
        
        # 定义一个包含两个 1D 卷积块的序列
        self.conv_down = nn.Sequential(
            # --- 第 1 块 ---
            # (k=7, s=2, p=3) -> 序列长度减半 (例如 2048 -> 1024)
            nn.Conv1d(in_chans, in_dim, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(in_dim, eps=1e-4),
            nn.GELU(approximate='tanh'),

            # --- 第 2 块 ---
            # (k=3, s=2, p=1) -> 序列长度再次减半 (例如 1024 -> 512)
            nn.Conv1d(in_dim, embed_dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim, eps=1e-4),
            nn.GELU(approximate='tanh')
        )

    def forward(self, x):
        """
        前向传播。
        输入 x 形状: (B, C_in, N_in)  -> (B, 1, 2048)
        """
        
        # 1. 通过卷积块
        # x 形状: (B, C_in, N_in) -> (B, embed_dim, N_in / 4)
        # 例如: (B, 1, 2048) -> (B, 384, 512)
        x = self.conv_down(x)
        
        # 2. 调整维度顺序
        # x 形状: (B, embed_dim, N_out) -> (B, N_out, embed_dim)
        # 例如: (B, 384, 512) -> (B, 512, 384)
        x = x.transpose(1, 2)
        
        return x


class TranscallerLight(nn.Module):
    """
    轻量化版本的 Transcaller 模型。
    
    主要变化:
    1. 使用 FastEmbedLight 进行 4x 下采样。
    2. Positional Embedding 长度适应下采样后的序列。
    3. 默认参数 (embed_dim, depth, num_heads, mlp_ratio) 已被缩小。
    """
    def __init__(self, in_chans=1, 
                 embed_dim=384,  # <-- 缩小
                 depth=6,        # <-- 缩小
                 num_heads=4,    # <-- 缩小
                 mlp_ratio=2.0,  # <-- 缩小
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 input_length=2048, output_length=420, num_classes=5,
                 downsample_ratio=4): # <-- 新增参数
        
        super().__init__()
        
        # 1. 使用轻量化 Stem (带下采样)
        self.stem = FastEmbedLight(
            in_chans=in_chans, 
            in_dim=128, 
            embed_dim=embed_dim, 
            downsample_ratio=downsample_ratio
        )
        
        # 2. 计算下采样后的序列长度
        # self.transformer_seq_len = input_length // downsample_ratio
        
        # 第 1 层 Conv1d: (k=7, s=2, p=3) [cite: 17]
        l1_out = math.floor((input_length + 2 * 3 - 7) / 2) + 1

        # 第 2 层 Conv1d: (k=3, s=2, p=1) [cite: 18]
        l2_out = math.floor((l1_out + 2 * 1 - 3) / 2) + 1

        self.transformer_seq_len = l2_out # (1998 -> 999 -> 500)
        
        # 3. Positional Embedding 适应新的序列长度
        self.pos_embed = nn.Parameter(torch.zeros(1, self.transformer_seq_len, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)

        # 4. 随机深度 (Stochastic Depth) 衰减
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # 5. Transformer 编码器堆栈 (使用缩小的 depth, dim, heads, mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=nn.LayerNorm
            )
            for i in range(depth) # 使用缩小的 depth
        ])
        
        # 6. 最终的归一化层
        self.norm = nn.LayerNorm(embed_dim)
        
        # 7. 输出头 (seq_len 参数传入下采样后的长度)
        self.head = Head(embed_dim, self.transformer_seq_len, output_length, num_classes=num_classes)
        
    def forward(self, x):
        """
        完整的前向传播。
        输入 x 形状: (B, 1, 2048)
        """
        
        # 1. 嵌入与下采样
        # x 形状: (B, 1, 2048) -> (B, 512, embed_dim)
        x = self.stem(x)
        
        # 2. 添加位置编码
        # (B, 512, embed_dim) + (1, 512, embed_dim)
        x = x + self.pos_embed
        
        # 3. 通过所有 Transformer 块
        # x 形状保持不变: (B, 512, embed_dim)
        for block in self.blocks:
            x = block(x)
            
        # 4. 最终归一化
        # x 形状: (B, 512, embed_dim)
        x = self.norm(x)
        
        # 5. 通过输出头
        # x 形状: (B, 512, embed_dim) -> (420, B, 5)
        x = self.head(x)
        
        # 6. 计算对数概率
        # (CTCLoss 期望对数概率作为输入)
        # 输出形状: (420, B, 5)
        return F.log_softmax(x, dim=-1)

# ==========================================================================================
# 主函数：实例化和测试 (轻量化版本)
# ==========================================================================================
if __name__ == '__main__':
    # --- 1. 配置和参数 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 模型参数 (使用了示例中的参数)
    BATCH_SIZE = 4 
    
    # 🚀 轻量化超参数 
    EMBED_DIM = 384    # 嵌入维度 (原: 768)
    DEPTH = 6          # Transformer 层数 (原: 12)
    NUM_HEADS = 4      # 注意力头数 (原: 6)
    MLP_RATIO = 2.0    # MLP 比例 (原: 4.0)
    
    # 信号参数 (保持不变)
    SEQ_LEN = 2048     # 原始输入序列长度
    OUTPUT_LEN = 420   # 输出序列长度
    NUM_CLASSES = 5    # 分类数 {A, C, G, T, <blank>}

    # --- 2. 实例化 (轻量化) 模型 ---
    model = TranscallerLight(
        input_length=SEQ_LEN,
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        output_length=OUTPUT_LEN,
        num_classes=NUM_CLASSES,
        downsample_ratio=4  # 明确指定下采样率
    ).to(device)

    # --- 3. 使用 torchinfo 打印架构摘要 ---
    print("\n" + "="*80)
    print("轻量化模型架构摘要 (TranscallerLight)")
    print("="*80)
    
    # 定义 torchinfo 需要的输入大小
    # (batch_size, in_channels, sequence_length)
    input_size = (BATCH_SIZE, 1, SEQ_LEN)
    
    # 打印摘要
    summary(model, input_size=input_size, device=device,
            col_names=["input_size", "output_size", "num_params", "mult_adds"])
    
    # --- 4. (可选) 测试一次前向传播 ---
    print("\n" + "="*80)
    print("测试一次前向传播...")
    print(f"创建随机输入: {input_size}")
    dummy_input = torch.randn(input_size).to(device)
    
    with torch.no_grad(): # 关闭梯度计算
        output = model(dummy_input)
        
    print(f"模型成功执行！")
    print(f"最终输出形状: {output.shape} (应为: {OUTPUT_LEN, BATCH_SIZE, NUM_CLASSES})")
    print("="*80)