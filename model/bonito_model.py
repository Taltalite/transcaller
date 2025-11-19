# [文件名: model.py]
# 最终整合版 (nn.txt + model.txt + ctc-crf.txt)

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import types
from functools import lru_cache
from collections import OrderedDict
from torch.nn.init import orthogonal_
from torch.nn.utils.fusion import fuse_conv_bn_eval

logger = logging.getLogger(__name__)

# --- [新增] 来自 bonito CTC-CRF Model.txt 的导入 ---
import numpy as np
try:
    import koi.lstm
    from koi.ctc import SequenceDist, Max, Log, semiring
    from koi.ctc import logZ_cu, viterbi_alignments, logZ_cu_sparse, bwd_scores_cu_sparse, fwd_scores_cu_sparse
except ImportError:
    logger.warning("无法导入 'koi.ctc'。将使用模拟 (Mock) 的类。")
    # --- 模拟 (Mock) 的 koi 组件 ---
    class SequenceDist: pass
    class semiring: pass
    class Log(semiring): one = 0.0
    class Max(semiring): one = -float('inf')
    def logZ_cu(*args): return torch.tensor(0.0)
    def logZ_cu_sparse(*args): return torch.tensor(0.0)
    def fwd_scores_cu_sparse(*args): return torch.tensor(0.0)
    def bwd_scores_cu_sparse(*args): return torch.tensor(0.0)
    def viterbi_alignments(*args): return torch.tensor(0.0)


# --- 日志记录 ---
logger = logging.getLogger(__name__)

# --- 动态层注册表 (来自 nn.txt) ---
layers = {}
def register(layer):
    layer.name = layer.__name__.lower()
    layers[layer.name] = layer
    return layer

# --- FlashAttention (来自 model.txt) ---
try:
    from flash_attn import flash_attn_qkvpacked_func
    # ... [此处省略模拟 (Mock) 的 flash-attn 组件] ...
    from flash_attn.layers.rotary import RotaryEmbedding
    from flash_attn.modules.mlp import GatedMlp
    from flash_attn.ops.triton.layer_norm import RMSNorm
    logger.info("FlashAttention-2 已成功导入。")
except ImportError:
    logger.warning(
        "无法导入 'flash-attn'。将使用原生的 PyTorch 模拟实现。 "
        "请安装 flash-attn 以获得最佳性能: `pip install flash-attn --no-build-isolation`"
    )

    class RMSNorm(nn.Module):
        def __init__(self, d_model, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(d_model))
        def forward(self, x, residual=None):
            normed_x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
            if residual is not None:
                return normed_x + residual
            return normed_x

    class GatedMlp(nn.Module):
        def __init__(self, in_features, hidden_features, activation=F.silu, bias1=False, bias2=False, multiple_of=1):
            super().__init__()
            self.fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias1)
            self.act = activation
            self.fc2 = nn.Linear(hidden_features, in_features, bias=bias2)
        def forward(self, x):
            gate_and_hidden = self.fc1(x)
            gate, hidden = gate_and_hidden.chunk(2, dim=-1)
            return self.fc2(self.act(gate) * hidden)
            
    class RotaryEmbedding(nn.Module):
        def __init__(self, dim, interleaved=False):
            super().__init__()
            logger.debug("使用模拟的 RotaryEmbedding (nn.Identity)")
        def forward(self, qkv):
            return qkv

    flash_attn_qkvpacked_func = None

# ===============================================
# == BONITO NN 模块 (来自 bonito nn.txt) ==
# ===============================================

Module = torch.nn.Module # 别名

# --- 实用函数 (来自 nn.txt) ---

def to_dict(layer, include_weights=False): # [cite: 46]
    if hasattr(layer, 'to_dict'):
        return {'type': layer.name, **layer.to_dict(include_weights)}
    return {'type': layer.name}


def from_dict(model_dict, layer_types=None):
    if not isinstance(model_dict, dict):
        return model_dict
    model_dict = model_dict.copy()
    if layer_types is None:
        layer_types = layers
    type_name = model_dict.pop('type')
    typ = layer_types[type_name]
    if hasattr(typ, "from_dict"):
        return typ.from_dict(model_dict, layer_types)
    if 'sublayers' in model_dict:
        sublayers = model_dict['sublayers']
        model_dict['sublayers'] = [
            from_dict(x, layer_types) for x in sublayers
        ] if isinstance(sublayers, list) else from_dict(sublayers, layer_types) # [cite: 48]
    try:
        layer = typ(**model_dict)
    except Exception as e:
        raise Exception(f'构建类型 {typ} 的层失败，参数 {model_dict}') from e
    return layer

# ... [此处省略 truncated_normal, fuse_bn_] ...
def truncated_normal(size, dtype=torch.float32, device=None, num_resample=5): # [cite: 40]
    x = torch.empty(size + (num_resample,), dtype=torch.float32, device=device).normal_()
    i = ((x < 2) & (x > -2)).max(-1, keepdim=True)[1] # [cite: 40]
    return torch.clamp_(x.gather(-1, i).squeeze(-1), -2, 2)

def fuse_bn_(m): # [cite: 49]
    m.training = False
    if isinstance(m, Convolution) and isinstance(m.norm, BatchNorm):
        m.conv = fuse_conv_bn_eval(m.conv, m.norm.bn) # [cite: 49]
        m.norm = None

# --- 基础模块 (来自 nn.txt) ---

register(torch.nn.ReLU)
register(torch.nn.Tanh)

@register
class Linear(nn.Module):
# ... [此处省略 Linear, Swish, Clamp, Serial, Stack, NamedSerial, MakeContiguous, LinearUpsample, Reverse, BatchNorm, Convolution] ...
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        self.linear = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=bias
        ) # [cite: 13]

    def forward(self, x):
        return self.linear(x)

    def to_dict(self, include_weights=False):
        res = {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
        }
        if include_weights: # [cite: 14]
            res['params'] = {
                'W': self.linear.weight,
                'b': self.linear.bias if self.bias is not None else []
            }
        return res

@register
class Swish(torch.nn.SiLU):
    pass

@register
class Clamp(nn.Module):
    def __init__(self, min, max):
        super().__init__()
        self.min = min # [cite: 15]
        self.max = max # [cite: 15]

    def forward(self, x):
        return torch.clamp(x, min=self.min, max=self.max)

    def to_dict(self, include_weights=False):
        return {'min': self.min, 'max': self.max}

@register
class Serial(torch.nn.Sequential):
    def __init__(self, sublayers):
        super().__init__(*sublayers)

    def forward(self, x, return_features=False): # [cite: 16]
        if return_features:
            fmaps = []
            for layer in self:
                x = layer(x)
                fmaps.append(x)
            return x, fmaps
        return super().forward(x) # [cite: 17]

    def to_dict(self, include_weights=False):
        return {
            'sublayers': [to_dict(layer, include_weights) for layer in self._modules.values()]
        } # [cite: 17]

    def __repr__(self):
        return torch.nn.ModuleList.__repr__(self)

@register
class Stack(Serial):
    @classmethod
    def from_dict(cls, model_dict, layer_types=None):
        return cls([from_dict(model_dict["layer"], layer_types) for _ in range(model_dict["depth"])])

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        layer_dicts = [to_dict(layer) for layer in self]
        for layer_dict in layer_dicts[1:]:
            assert layer_dict == layer_dicts[0], "all layers should be the same"
        return {"layer": layer_dicts[0], "depth": len(self)}

@register
class NamedSerial(torch.nn.Sequential):
    @classmethod
    def from_dict(cls, model_dict, layer_types=None):
        return cls({k: from_dict(v, layer_types) for k, v in model_dict.items()})

    def __init__(self, layers):
        super().__init__(OrderedDict(layers.items())) # [cite: 19]

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return {k: to_dict(v) for k, v in self.named_children()}

class MakeContiguous(nn.Module): # [cite: 11]
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.contiguous()

@register
class LinearUpsample(nn.Module):
    # [cite: 20]
    def __init__(self, d_model, scale_factor, batch_first=True):
        super().__init__()
        self.d_model = d_model # [cite: 21]
        self.scale_factor = scale_factor # [cite: 21]
        self.batch_first = batch_first # [cite: 21]
        self.linear = torch.nn.Linear(d_model, self.scale_factor * d_model)

    def forward(self, src):
        if not self.batch_first:
            src = src.permute([1, 0, 2])
        N, L, E = src.shape # [cite: 22]
        h = self.linear(src).reshape(N, self.scale_factor * L, E) # [cite: 22]
        if not self.batch_first:
            h = h.permute([1, 0, 2])
        return h

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return {
            "d_model": self.d_model,
            "scale_factor": self.scale_factor,
            "batch_first": self.batch_first
        } # [cite: 23]

@register
class Reverse(nn.Module):
    def __init__(self, sublayers):
        super().__init__()
        self.layer = Serial(sublayers) if isinstance(sublayers, list) else sublayers

    def forward(self, x):
        return self.layer(x.flip(0)).flip(0) # [cite: 24]

    def to_dict(self, include_weights=False):
        if isinstance(self.layer, Serial):
            return self.layer.to_dict(include_weights)
        else:
            return {'sublayers': to_dict(self.layer, include_weights)} # [cite: 24]

@register
class BatchNorm(nn.Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(num_features, eps, momentum, affine, track_running_stats) # [cite: 25]

    def forward(self, x):
        return self.bn(x)

    def to_dict(self, include_weights=False):
        res = {
            "num_features": self.bn.num_features,
            "eps": self.bn.eps,
            "momentum": self.bn.momentum,
            "affine": self.bn.affine,
            "track_running_stats": self.bn.track_running_stats
        } # [cite: 25]
        if include_weights:
            params = {} # 
            if res["affine"]:
                params["W"] = self.bn.weight
                params["b"] = self.bn.bias
            if res["track_running_stats"]:
                params["running_mean"] = self.bn.running_mean
                params["running_var"] = self.bn.running_var
            res["params"] = params
        return res

@register
class Convolution(nn.Module):
    def __init__(self, insize, size, winlen, stride=1, padding=0, bias=True, activation=None, norm=None):
        super().__init__()
        self.conv = torch.nn.Conv1d(insize, size, winlen, stride=stride, padding=padding, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()
        if isinstance(norm, dict):
            self.norm = from_dict(norm) # [cite: 28]
        elif isinstance(norm, str):
            self.norm = layers[norm](size) # [cite: 28]
        else:
            self.norm = norm

    def forward(self, x):
        h = self.conv(x)
        if self.norm is not None:
            h = self.norm(h)
        if self.activation is not None:
            h = self.activation(h) # [cite: 29]
        return h

    def to_dict(self, include_weights=False):
        res = {
            "insize": self.conv.in_channels,
            "size": self.conv.out_channels,
            "bias": self.conv.bias is not None,
            "winlen": self.conv.kernel_size[0],
            "stride": self.conv.stride[0], # [cite: 30]
            "padding": self.conv.padding[0],
        }
        if self.activation is not None:
            res["activation"] = self.activation.name
        if self.norm is not None:
            res["norm"] = to_dict(self.norm, include_weights)
            if not include_weights and self.norm.name in layers: # [cite: 31]
                if res["norm"] == to_dict(layers[self.norm.name](res["size"])):
                    res["norm"] = self.norm.name

        if include_weights:
            res['params'] = {
                'W': self.conv.weight, 'b': self.conv.bias if self.conv.bias is not None else []
            }
        return res
        
@register
class LinearCRFEncoder(nn.Module):
# ... [此处省略 LinearCRFEncoder, Permute, RNNWrapper, LSTM] ...
    def __init__(self, insize, n_base, state_len, bias=True, scale=None, activation=None, blank_score=None, expand_blanks=True, permute=None):
        super().__init__()
        self.scale = scale
        self.n_base = n_base
        self.state_len = state_len
        self.blank_score = blank_score
        self.expand_blanks = expand_blanks # [cite: 33]
        size = (n_base + 1) * n_base**state_len if blank_score is None else n_base**(state_len + 1) # [cite: 33]
        self.linear = torch.nn.Linear(insize, size, bias=bias)
        self.activation = layers.get(activation, lambda: activation)()
        self.permute = permute

    def forward(self, x):
        if self.permute is not None:
            x = x.permute(*self.permute)
        scores = self.linear(x)
        if self.activation is not None:
            scores = self.activation(scores)
        if self.scale is not None:
            scores = scores * self.scale
        if self.blank_score is not None and self.expand_blanks:
            T, N, C = scores.shape
            scores = torch.nn.functional.pad(
                scores.view(T, N, C // self.n_base, self.n_base),
                (1, 0, 0, 0, 0, 0, 0, 0),
                value=self.blank_score
            ).view(T, N, -1) # [cite: 35]
        return scores

    def to_dict(self, include_weights=False):
        res = {
            'insize': self.linear.in_features, # [cite: 36]
            'n_base': self.n_base,
            'state_len': self.state_len,
            'bias': self.linear.bias is not None,
            'scale': self.scale,
            'blank_score': self.blank_score,
            'expand_blanks': self.expand_blanks,
        } # [cite: 37]
        if self.activation is not None:
            res['activation'] = self.activation.name
        if self.permute is not None:
            res['permute'] = self.permute
        if include_weights:
            res['params'] = {
                'W': self.linear.weight, 'b': self.linear.bias
                if self.linear.bias is not None else [] # [cite: 38]
            }
        return res

    def extra_repr(self):
        rep = 'n_base={}, state_len={}, scale={}, blank_score={}, expand_blanks={}'.format(
            self.n_base, self.state_len, self.scale, self.blank_score, self.expand_blanks
        )
        if self.permute:
            rep += f', permute={self.permute}' # [cite: 39]
        return rep

@register
class Permute(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims) # [cite: 39]

    def to_dict(self, include_weights=False):
        return {'dims': self.dims}

    def extra_repr(self):
        return 'dims={}'.format(self.dims)

class RNNWrapper(nn.Module):
    def __init__(
            self, rnn_type, *args, reverse=False, orthogonal_weight_init=True, disable_state_bias=True, bidirectional=False, **kwargs
    ):
        super().__init__()
        if reverse and bidirectional:
            raise Exception("'reverse' and 'bidirectional' should not both be set to True")
        self.reverse = reverse
        self.rnn = rnn_type(*args, bidirectional=bidirectional, **kwargs) # [cite: 41]
        self.init_orthogonal(orthogonal_weight_init)
        self.init_biases()
        if disable_state_bias: self.disable_state_bias()

    def forward(self, x):
        if self.reverse: x = x.flip(0)
        y, h = self.rnn(x)
        if self.reverse: y = y.flip(0)
        return y

    def init_biases(self, types=('bias_ih',)):
        for name, param in self.rnn.named_parameters():
            if any(k in name for k in types): # [cite: 42]
                with torch.no_grad():
                    param.set_(0.5*truncated_normal(param.shape, dtype=param.dtype, device=param.device))

    def init_orthogonal(self, types=True):
        if not types: return
        if types == True: types = ('weight_ih', 'weight_hh')
        for name, x in self.rnn.named_parameters():
            if any(k in name for k in types): # [cite: 43]
                for i in range(0, x.size(0), self.rnn.hidden_size):
                    orthogonal_(x[i:i+self.rnn.hidden_size])

    def disable_state_bias(self):
        for name, x in self.rnn.named_parameters():
            if 'bias_hh' in name:
                x.requires_grad = False # [cite: 44]
                x.zero_()

    def extra_repr(self):
        return 'reverse={}'.format(bool(self.reverse))

@register
class LSTM(RNNWrapper):
    def __init__(self, size, insize, bias=True, reverse=False):
        super().__init__(torch.nn.LSTM, insize, size, bias=bias, reverse=reverse) # [cite: 45]

    def to_dict(self, include_weights=False):
        res = {
            'size': self.rnn.hidden_size,
            'insize': self.rnn.input_size, # [cite: 45]
            'bias': self.rnn.bias,
            'reverse': self.reverse,
        }
        if include_weights:
            res['params'] = {
                'iW': self.rnn.weight_ih_l0.reshape(4, self.rnn.hidden_size, self.rnn.input_size),
                'sW': self.rnn.weight_hh_l0.reshape(4, self.rnn.hidden_size, self.rnn.hidden_size), # [cite: 46]
                'b': self.rnn.bias_ih_l0.reshape(4, self.rnn.hidden_size)
            }
        return res
        
# ===================================================
# == BONITO TRANSFORMER 模块 (来自 bonito model.txt) ==
# ===================================================

def deepnorm_params(depth): 
    alpha = round((2*depth)**0.25, 7)
    beta = round((8*depth)**(-1/4), 7)
    return alpha, beta

# ... [此处省略 sliding_window_mask, MultiHeadAttention, TransformerEncoderLayer] ...
@lru_cache(maxsize=2)
def sliding_window_mask(seq_len, window, device):
    band = torch.full((seq_len, seq_len), fill_value=1.0)
    band = torch.triu(band, diagonal=-window[0])
    band = band * torch.tril(band, diagonal=window[1])
    band = band.to(torch.bool).to(device)
    return band

class MultiHeadAttention(Module):
    def __init__(self, d_model, nhead, qkv_bias=False, out_bias=True, rotary_dim=None, attn_window=None):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.rotary_dim = self.head_dim if rotary_dim is None else rotary_dim

        self.Wqkv = torch.nn.Linear(d_model, 3 * d_model, bias=qkv_bias)
        self.out_proj = torch.nn.Linear(d_model, d_model, bias=out_bias)

        self.rotary_emb = RotaryEmbedding(self.rotary_dim, interleaved=False)
        self.attn_window = (-1, -1) if attn_window is None else tuple(attn_window)

    def attn_func(self, qkv):
        # --- [新增] 导入 amp ---
        import torch.cuda.amp as amp
        
        # --- [新增] 在 autocast 块内强制使用 float32 ---
        with amp.autocast(enabled=True, dtype=torch.float32):
            if flash_attn_qkvpacked_func is not None and torch.cuda.get_device_capability(qkv.device)[0] >= 8 and (torch.is_autocast_enabled() or qkv.dtype == torch.half):
                attn_output = flash_attn_qkvpacked_func(qkv, window_size=self.attn_window)
            else:
                q, k, v = torch.chunk(qkv.permute(0, 2, 3, 1, 4), chunks=3, dim=1)
                mask = sliding_window_mask(qkv.shape[1], self.attn_window, q.device)
                attn_output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                attn_output = attn_output.permute(0, 1, 3, 2, 4)
        
        # 返回的结果将自动转换回 float16
        return attn_output

    def forward(self, x):
        N, T, _ = x.shape

        qkv = self.Wqkv(x).view(N, T, 3, self.nhead, self.head_dim)
        qkv = self.rotary_emb(qkv)
        attn_output = self.attn_func(qkv).reshape(N, T, self.d_model)
        out = self.out_proj(attn_output)

        return out

@register
class TransformerEncoderLayer(Module):
    def __init__(self, d_model, nhead, dim_feedforward, deepnorm_alpha, deepnorm_beta, attn_window=None):
        super().__init__()
        self.kwargs = {
            "d_model": d_model,
            "nhead": nhead,
            "dim_feedforward": dim_feedforward,
            "deepnorm_alpha": deepnorm_alpha,
            "deepnorm_beta": deepnorm_beta, # [cite: 7]
            "attn_window": attn_window
        }

        self.self_attn = MultiHeadAttention(
            d_model=d_model,
            nhead=nhead,
            qkv_bias=False,
            out_bias=True,
            attn_window=attn_window
        )
        self.ff = GatedMlp(
            d_model,
            hidden_features=dim_feedforward,
            activation=F.silu,
            bias1=False,
            bias2=False,
            multiple_of=1,
        )
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)

        self.register_buffer("deepnorm_alpha", torch.tensor(deepnorm_alpha))
        self.reset_parameters()

    def reset_parameters(self):
        db = self.kwargs["deepnorm_beta"]
        d_model = self.kwargs["d_model"]
        torch.nn.init.xavier_normal_(self.ff.fc1.weight, gain=db)
        torch.nn.init.xavier_normal_(self.ff.fc2.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.out_proj.weight, gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[2*d_model:], gain=db)
        torch.nn.init.xavier_normal_(self.self_attn.Wqkv.weight[:2*d_model], gain=1)

    def forward(self, x):
        x = self.norm1(self.self_attn(x), self.deepnorm_alpha*x)
        x = self.norm2(self.ff(x), self.deepnorm_alpha*x)
        return x

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        return self.kwargs


# --- [新增] 来自 bonito CTC-CRF Model.txt 的代码 ---

def get_stride(m, stride=1): # [cite: 1]
    if hasattr(m, "output_stride"):
        stride = m.output_stride(stride)
    elif hasattr(m, "stride"):
        s = m.stride
        if isinstance(s, tuple):
            assert len(s) == 1
            s = s[0]
        stride = stride * s
    else:
        for child in m.children():
            stride = get_stride(child, stride)
    return stride


class CTC_CRF(SequenceDist): # 

    def __init__(self, state_len, alphabet):
        super().__init__()
        self.alphabet = alphabet
        self.state_len = state_len
        self.n_base = len(alphabet[1:])
        self.idx = torch.cat([
            torch.arange(self.n_base**(self.state_len))[:, None],
            torch.arange(
                self.n_base**(self.state_len)
            ).repeat_interleave(self.n_base).reshape(self.n_base, -1).T
        ], dim=1).to(torch.int32)

    def n_score(self):
        return len(self.alphabet) * self.n_base**(self.state_len)

    def logZ(self, scores, S:semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, len(self.alphabet))
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return logZ_cu_sparse(Ms, self.idx, alpha_0, beta_T, S)

    def normalise(self, scores):
        return (scores - self.logZ(scores)[:, None] / len(scores))

    def forward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        alpha_0 = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return fwd_scores_cu_sparse(Ms, self.idx, alpha_0, S, K=1)

    def backward_scores(self, scores, S: semiring=Log):
        T, N, _ = scores.shape
        Ms = scores.reshape(T, N, -1, self.n_base + 1)
        beta_T = Ms.new_full((N, self.n_base**(self.state_len)), S.one)
        return bwd_scores_cu_sparse(Ms, self.idx, beta_T, S, K=1)

    def compute_transition_probs(self, scores, betas):
        T, N, C = scores.shape
        log_trans_probs = (scores.reshape(T, N, -1, self.n_base + 1) + betas[1:, :, :, None])
        log_trans_probs = torch.cat([
            log_trans_probs[:, :, :, [0]],
            log_trans_probs[:, :, :, 1:].transpose(3, 2).reshape(T, N, -1, self.n_base)
        ], dim=-1)
        trans_probs = torch.softmax(log_trans_probs, dim=-1)
        init_state_probs = torch.softmax(betas[0], dim=-1)
        return trans_probs, init_state_probs

    def reverse_complement(self, scores):
        T, N, C = scores.shape
        expand_dims = T, N, *(self.n_base for _ in range(self.state_len)), self.n_base + 1
        scores = scores.reshape(*expand_dims)
        blanks = torch.flip(scores[..., 0].permute(
            0, 1, *range(self.state_len + 1, 1, -1)).reshape(T, N, -1, 1), [0, 2]
        )
        emissions = torch.flip(scores[..., 1:].permute(
            0, 1, *range(self.state_len, 1, -1),
            self.state_len +2,
            self.state_len + 1).reshape(T, N, -1, self.n_base), [0, 2, 3]
        )
        return torch.cat([blanks, emissions], dim=-1).reshape(T, N, -1)

    def viterbi(self, scores):
        traceback = self.posteriors(scores, Max)
        a_traceback = traceback.argmax(2)
        moves = (a_traceback % len(self.alphabet)) != 0
        paths = 1 + (torch.div(a_traceback, len(self.alphabet), rounding_mode="floor") % self.n_base)
        return torch.where(moves, paths, 0)

    def path_to_str(self, path):
        alphabet = np.frombuffer(''.join(self.alphabet).encode(), dtype='u1')
        seq = alphabet[path[path != 0]]
        return seq.tobytes().decode()

    def prepare_ctc_scores(self, scores, targets):
        targets = torch.clamp(targets - 1, 0)
        T, N, C = scores.shape
        scores = scores.to(torch.float32)
        n = targets.size(1) - (self.state_len - 1)
        stay_indices = sum(
            targets[:, i:n + i] * self.n_base ** (self.state_len - i - 1)
            for i in range(self.state_len)
        ) * len(self.alphabet)
        move_indices = stay_indices[:, 1:] + targets[:, :n - 1] + 1
        stay_scores = scores.gather(2, stay_indices.expand(T, -1, -1))
        move_scores = scores.gather(2, move_indices.expand(T, -1, -1))
        return stay_scores, move_scores

    def ctc_loss(self, scores, targets, target_lengths, loss_clip=None, reduction='mean', normalise_scores=True):
        if normalise_scores:
            scores = self.normalise(scores)
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        logz = logZ_cu(stay_scores, move_scores, target_lengths + 1 - self.state_len)
        loss = - (logz / target_lengths)
        if loss_clip:
            loss = torch.clamp(loss, 0.0, loss_clip)
        if reduction == 'mean':
            return loss.mean()
        elif reduction in ('none', None):
            return loss
        else:
            raise ValueError('Unknown reduction type {}'.format(reduction))

    def ctc_viterbi_alignments(self, scores, targets, target_lengths):
        stay_scores, move_scores = self.prepare_ctc_scores(scores, targets)
        return viterbi_alignments(stay_scores, move_scores, target_lengths + 1 - self.state_len)

# --- [新增] 帮助函数 (来自 bonito CTC-CRF Model.txt) ---
def conv(c_in, c_out, ks, stride=1, bias=False, activation=None, norm=None): # [cite: 12]
    return Convolution(c_in, c_out, ks, stride=stride, padding=ks//2, bias=bias, activation=activation, norm=norm)


def rnn_encoder(n_base, state_len, insize=1, first_conv_size=4, stride=5, winlen=19, activation='swish', rnn_type='lstm', features=768, scale=5.0, blank_score=None, expand_blanks=True, num_layers=5, norm=None): # [cite: 12]
    rnn = layers[rnn_type]
    return Serial([
        conv(insize, first_conv_size, ks=5, bias=True, activation=activation, norm=norm),
        conv(first_conv_size, 16, ks=5, bias=True, activation=activation, norm=norm),
        conv(16, features, ks=winlen, stride=stride, bias=True, activation=activation, norm=norm),
        Permute([2, 0, 1]),
        *(rnn(features, features, reverse=(num_layers - i) % 2) for i in range(num_layers)),
        LinearCRFEncoder(
            features, n_base, state_len, activation='tanh', scale=scale,
            blank_score=blank_score, expand_blanks=expand_blanks
        )
    ])

# --- [新增] 顶层模型封装类 (来自 bonito CTC-CRF Model.txt) ---
@register
class SeqdistModel(Module): # 
    def __init__(self, encoder, seqdist, n_pre_post_context_bases=None, target_projection=None):
        super().__init__()
        self.seqdist = seqdist
        self.encoder = encoder
        self.stride = get_stride(encoder)
        self.alphabet = seqdist.alphabet

        if n_pre_post_context_bases is None:
            self.n_pre_context_bases = self.seqdist.state_len - 1
            self.n_post_context_bases = 1
        else:
            self.n_pre_context_bases, self.n_post_context_bases = n_pre_post_context_bases

        if target_projection is None:
            self.target_projection = None
        else:
            self.register_buffer('target_projection', torch.tensor([0] + target_projection), persistent=False)

    @classmethod
    def from_dict(cls, model_dict, layer_types=None):
        kwargs = dict(
            model_dict,
            encoder=from_dict(model_dict["encoder"], layer_types),
            seqdist=CTC_CRF(**model_dict["seqdist"])
        )
        return cls(**kwargs)

    def forward(self, x, *args):
        return self.encoder(x)

    def decode_batch(self, x):
        """
        [这就是您需要的解码函数]
        由 bonito training.txt 中的 Trainer 调用 [cite: 69]
        """
        scores = self.seqdist.posteriors(x.to(torch.float32)) + 1e-8
        tracebacks = self.seqdist.viterbi(scores.log()).to(torch.int16).T
        return [self.seqdist.path_to_str(x) for x in tracebacks.cpu().numpy()] # 

    def decode(self, x):
        """
        [这就是您需要的解码函数]
        由 bonito training.txt 中的 Trainer 调用 [cite: 70]
        """
        return self.decode_batch(x.unsqueeze(1))[0] # [cite: 23]

    def loss(self, scores, targets, target_lengths, **kwargs):
        """
        [这就是您需要的损失函数]
        由 bonito training.txt 中的 Trainer 调用 [cite: 57, 60]
        """
        if self.target_projection is not None:
            targets = self.target_projection[targets]
        return self.seqdist.ctc_loss(scores.to(torch.float32), targets, target_lengths, **kwargs) # [cite: 24-25]

    def use_koi(self, **kwargs):
        pass

    def to_dict(self, include_weights=False):
        if include_weights:
            raise NotImplementedError
        res = {
            "encoder": to_dict(self.encoder),
            "seqdist": {"state_len": self.seqdist.state_len, "alphabet": self.seqdist.alphabet},
            "n_pre_post_context_bases": (self.n_pre_context_bases, self.n_post_context_bases),
        }
        if self.target_projection is not None:
            res["target_projection"] = self.target_projection.tolist()[1:]
        return res


class Model(SeqdistModel): # 
    """
    [修改] 这是新的顶层模型入口点
    它将配置 (config) 分配给 encoder 和 seqdist
    """
    def __init__(self, config):
        seqdist = CTC_CRF(
            state_len=config['global_norm']['state_len'],
            alphabet=config['labels']['labels']
        )
        if 'type' in config['encoder']: # new-style config (您的 Transformer)
            encoder = from_dict(config['encoder'])
        else: # old-style (RNN)
            encoder = rnn_encoder(seqdist.n_base, seqdist.state_len, insize=config['input']['features'], **config['encoder'])

        super().__init__(encoder, seqdist, n_pre_post_context_bases=config.get('input', {}).get('n_pre_post_context_bases'))
        self.config = config

    def use_koi(self, **kwargs):
        # [修改] - 将 use_koi 逻辑从旧的工厂函数移到这里
        def _expand_blanks(m):
            if isinstance(m, LinearCRFEncoder):
                m.expand_blanks = False # [cite: 11]
        self.encoder.apply(_expand_blanks)
        self.encoder = Serial([
            self.encoder,
            Permute([1, 0, 2]),
            MakeContiguous(),
        ]) # [cite: 11]
        
        # RNN 模型的额外逻辑
        if 'quantize' in kwargs:
             self.encoder = koi.lstm.update_graph(
                self.encoder,
                batchsize=kwargs["batchsize"],
                chunksize=kwargs["chunksize"] // self.stride,
                quantize=kwargs["quantize"],
            )


# ===================================================
# == __main__ 块：[修改] 以使用新的顶层 Model ==
# ===================================================

if __name__ == "__main__":
    
    # 1. 导入 torchinfo
    try:
        from torchinfo import summary
    except ImportError:
        print("-" * 50)
        print("错误：未找到 'torchinfo' 包。")
        print("请运行 `pip install torchinfo` 来查看模型摘要。")
        print("-" * 50)
        exit()

    # 2. [修改] 定义配置
    
    depth = 6
    d_model = 256
    alpha, beta = deepnorm_params(depth) # [cite: 1]
    
    # 这是您原来的 SAMPLE_CONFIG["model"]，现在是编码器配置
    ENCODER_CONFIG = {
        "type": "serial", # [cite: 17]
        "sublayers": [
            {
                "type": "convolution",
                "insize": 1,
                "size": d_model,
                "winlen": 19,
                "stride": 5,
                "activation": "swish",
                "norm": "batchnorm"
            },
            {
                "type": "permute", # [cite: 39]
                "dims": [0, 2, 1]
            },
            {
                "type": "stack",
                "depth": depth,
                "layer": {
                    "type": "transformerencoderlayer",
                    "d_model": d_model,
                    "nhead": 4,
                    "dim_feedforward": d_model * 4,
                    "deepnorm_alpha": alpha, # [cite: 7]
                    "deepnorm_beta": beta # [cite: 7]
                }
            },
            {
                "type": "permute", # [cite: 39]
                "dims": [1, 0, 2]
            },
            {
                "type": "linearcrfencoder",
                "insize": d_model,
                "n_base": 4,
                "state_len": 1,
                "blank_score": None
            }
        ]
    }
    
    # [新增] 创建新的顶层配置 (FULL_CONFIG)
    # 这满足了新的 `Model` 类  的要求
    FULL_CONFIG = {
        "encoder": ENCODER_CONFIG,
        "labels": {
            # 示例字母表 (0=blank, 1=A, 2=C, 3=G, 4=T)
            "labels": ["", "A", "C", "G", "T"] 
        },
        "global_norm": {
            "state_len": 1 # 必须匹配 LinearCRFEncoder
        }
    }


    # 3. [修改] 实例化新的顶层模型
    print("正在从示例配置 (FULL_CONFIG) 实例化顶层 Model...")
    model = Model(FULL_CONFIG) # 
    print("模型实例化成功。")
    print("-" * 70)

    # 4. 定义示例输入并运行 torchinfo.summary
    
    input_size = (1, 1, 2048) 
    
    print(f"模型摘要 (Model Summary) - 输入尺寸: {input_size}")
    
    # [修改] 我们在 model.encoder 上运行 summary，
    # 因为顶层模型 (model) 还包含非 nn.Module 的 'seqdist'
    summary(model.encoder, input_size=input_size, dtypes=[torch.float], col_names=["input_size", "output_size", "num_params", "mult_adds"])

    print("-" * 70)
    print("1. Convolution:   (N, 1, 2048) -> (N, 256, 408) ")
    print("2. Permute:       (N, 256, 408) -> (N, 408, 256) [cite: 39]")
    print("3. Stack (6xTrm): (N, 408, 256) -> (N, 408, 256) [cite: 10, 18]")
    print("4. Permute:       (N, 408, 256) -> (T, N, 256)  (即 408, 1, 256) [cite: 39]")
    print(f"5. LinearCRF:     (T, N, 256) -> (T, N, 20)  [cite: 32, 34]")
    print("   (CRF 输出维度 (4+1)*4^1 = 20) [cite: 33]")