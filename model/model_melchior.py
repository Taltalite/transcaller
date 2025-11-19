import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- Core Dependencies ---
# Ensure you have these packages installed:
# pip install torch timm torchinfo
from timm.models.layers import DropPath, trunc_normal_, Mlp
from torchinfo import summary

class FastEmbed(nn.Module):
    def __init__(self, in_chans=1, in_dim=512, embed_dim=1024):
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv1d(in_chans, in_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(in_dim, eps=1e-4), nn.GELU(approximate='tanh'),
            nn.Conv1d(in_dim, embed_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(embed_dim, eps=1e-4), nn.GELU(approximate='tanh')
        )
    def forward(self, x):
        return self.conv_down(x).transpose(1, 2)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop.p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.mixer = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Head(nn.Module):
    def __init__(self, in_features, seq_len, out_seq_len, num_classes):
        super().__init__()
        self.adaptive_pool = nn.AdaptiveAvgPool1d(out_seq_len)
        self.out_proj = nn.Linear(in_features, num_classes)
    def forward(self, x):
        x = self.adaptive_pool(x.transpose(1, 2)).transpose(1, 2)
        x = self.out_proj(x)
        return x.permute(1, 0, 2)

class Melchior(nn.Module):
    def __init__(self, in_chans=1, embed_dim=1024, depth=20, num_heads=8, mlp_ratio=4.,
                 qkv_bias=False, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 output_length=420, num_classes=5):
        super().__init__()
        self.stem = FastEmbed(in_chans=in_chans, in_dim=512, embed_dim=embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, 4096, embed_dim))
        trunc_normal_(self.pos_embed, std=.02)
        
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        # --- CHANGED: Replaced Mamba/Transformer alternation with pure Transformer blocks ---
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
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.head = Head(embed_dim, 4096, output_length, num_classes=num_classes)
        
    def forward(self, x):
        x = self.stem(x) + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        x = self.head(x)
        return F.log_softmax(x, dim=-1)

# ==========================================================================================
# Main function to instantiate and summarize the model
# ==========================================================================================
if __name__ == '__main__':
    # --- 1. Configuration and Parameters ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Model parameters (reduced for faster example run)
    BATCH_SIZE = 4 
    EMBED_DIM = 768
    DEPTH = 12
    NUM_HEADS = 6
    SEQ_LEN = 4096
    OUTPUT_LEN = 420
    NUM_CLASSES = 5 # e.g., {A, C, G, T, <blank>}

    # --- 2. Instantiate the Model ---
    model = Melchior(
        embed_dim=EMBED_DIM,
        depth=DEPTH,
        num_heads=NUM_HEADS,
        output_length=OUTPUT_LEN,
        num_classes=NUM_CLASSES
    ).to(device)

    # --- 3. Use torchinfo to print the architecture summary ---
    print("\n" + "="*80)
    print("Model Architecture Summary (Pure Transformer Version)")
    print("="*80)
    # batch_size, in_channels, sequence_length
    input_size = (BATCH_SIZE, 1, SEQ_LEN)
    summary(model, input_size=input_size, device=device)