import os

# --- 根治并行冲突问题的代码 ---
# 必须放在 import torch 和 numpy 等库之前
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from typing import Type # --- MODIFIED: Added for type hinting ---

# ==========================================================================================
# --- Self-implemented replacements for TIMM components (Your code is great here) ---
# ==========================================================================================

class Mlp(nn.Module):
    """A simple MLP block."""
    def __init__(self, in_features: int, hidden_features: int = None, out_features: int = None, act_layer: Type[nn.Module] = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x)
        return x

def drop_path(x: torch.Tensor, drop_prob: float = 0., training: bool = False) -> torch.Tensor:
    """Stochastic depth implementation."""
    if drop_prob == 0. or not training: return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor

class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) module."""
    def __init__(self, drop_prob: float = 0.):
        super().__init__(); self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)

class FastEmbed(nn.Module):
    def __init__(self, in_chans: int = 1, in_dim: int = 512, embed_dim: int = 1024):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv1d(in_chans, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(in_dim, eps=1e-4), nn.GELU(approximate='tanh'),
            nn.Conv1d(in_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim, eps=1e-4), nn.GELU(approximate='tanh')
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_down(x).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p if self.training else 0.0)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4., qkv_bias: bool = False, drop: float = 0., attn_drop: float = 0., drop_path: float = 0., norm_layer: Type[nn.Module] = nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Head(nn.Module):
    def __init__(self, in_features: int, out_seq_len: int, num_classes: int):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(out_seq_len)
        self.out_proj = nn.Linear(in_features, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x has shape (B, N, C)
        x = self.adaptive_pool(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(x) # Shape is now [Batch, out_seq_len, num_classes]
        
        # --- MODIFIED ---
        # Removed x.permute(1, 0, 2) to change output shape from
        # (SeqLen, Batch, Classes) to the desired (Batch, SeqLen, Classes).
        # The previous format is standard for PyTorch's CTCLoss.
        return x

class Melchior(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 1024, depth: int = 20, num_heads: int = 8,
                 mlp_ratio: float = 4., qkv_bias: bool = False, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1, seq_len: int = 4096,
                 output_length: int = 420, num_classes: int = 5):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.stem = FastEmbed(in_chans=in_chans, in_dim=512, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.seq_len, self.embed_dim))
        nn.init.normal_(self.pos_embed, std=.02)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                             drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i])
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = Head(embed_dim, output_length, num_classes=num_classes)
    
    def _interpolate_pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == self.seq_len: return self.pos_embed
        pos_embed_transposed = self.pos_embed.transpose(1, 2)
        interpolated_pos_embed = F.interpolate(
            pos_embed_transposed, size=x.shape[1], mode='linear', align_corners=False)
        return interpolated_pos_embed.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = x + self._interpolate_pos_embed(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        # Note: The log_softmax is now applied to a [Batch, Seq, Class] tensor
        return F.log_softmax(x, dim=-1)
    
    
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- MODIFIED: Changed batch size to 32 as requested ---
    BATCH_SIZE = 32 
    EMBED_DIM = 768
    DEPTH = 12
    NUM_HEADS = 6
    TRAINED_SEQ_LEN = 4096
    OUTPUT_LEN = 420      # This corresponds to 'X' in your request [32, X, 5]
    NUM_CLASSES = 5

    model = Melchior(
        embed_dim=EMBED_DIM, depth=DEPTH, num_heads=NUM_HEADS,
        seq_len=TRAINED_SEQ_LEN,
        output_length=OUTPUT_LEN,
        num_classes=NUM_CLASSES
    ).to(device)

    print("\n" + "="*80)
    print(f"Summary with Input Shape [{BATCH_SIZE}, 1, {TRAINED_SEQ_LEN}]")
    print("="*80)
    input_size = (BATCH_SIZE, 1, TRAINED_SEQ_LEN)
    summary(model, input_size=input_size, device=str(device), depth=5,
            col_names=["input_size", "output_size", "num_params"])