"""
FALG-Mamba-MLF-PG: Multi-Level Frequency-Aware Token Fusion for lightweight
raw-waveform audio deepfake detection.

Design notes
------------
1) FreqAwareCompressionTK keeps the explicit [T, K] structure as [B,T,K,C].
   Levels are aligned ONLY along the time axis T, never over a flattened T*K axis.
2) Cross-level fusion is token-wise attention over levels, optionally biased by a
   learnable global level prior (``mlf_fusion='prior_attn'``).
3) Levels have very different time resolutions (e.g. 597 / 99 / 16 frames for a
   64600-sample input). ``mlf_align_level`` selects the common time grid and
   downsampling is done with anti-aliased average pooling, not point-sampling
   interpolation.
"""

import os
import sys
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mamba_simple import RMSNorm

_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _root_dir not in sys.path:
    sys.path.insert(0, _root_dir)

from myBlocks import My_Residual_block, My_SERes2Net_block, CONV
from oc_softmax import OCSoftmax


def _parse_prior(prior, n_level: int = 3) -> torch.Tensor:
    """Parse prior from str/list/tuple/tensor and normalize it."""
    if isinstance(prior, str):
        vals = [float(x.strip()) for x in prior.split(',') if x.strip()]
        prior_t = torch.tensor(vals, dtype=torch.float32)
    elif isinstance(prior, torch.Tensor):
        prior_t = prior.detach().float().clone()
    elif isinstance(prior, Iterable):
        prior_t = torch.tensor(list(prior), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported prior type: {type(prior)}")
    if prior_t.numel() != n_level:
        raise ValueError(f"Expected {n_level} prior values, got {prior_t.numel()}: {prior}")
    prior_t = prior_t.clamp_min(1e-6)
    prior_t = prior_t / prior_t.sum()
    return prior_t


# ─────────────────────────────────────────────────────────────────────────────
# Front-end: shared multi-level Sinc + SE-Res2Net encoder
# ─────────────────────────────────────────────────────────────────────────────
class MultiLevelSincEncoder(nn.Module):
    """Shared Sinc + SE-Res2Net encoder returning x2/x3/x4 feature maps.

    x2: [B,32,F,T]  shallow/mid: finer local artefacts
    x3: [B,64,F,T]  middle: medium-range artefacts
    x4: [B,64,F,T]  deep: high-receptive-field artefacts
    """
    def __init__(self, sinc_channels: int = 70, freq_pool: int = 3, preact: bool = True):
        super().__init__()
        self.sinc = CONV(out_channels=sinc_channels, kernel_size=128, in_channels=1)
        self.first_bn = nn.BatchNorm2d(1)
        self.selu = nn.SELU(inplace=True)
        self.freq_pool = freq_pool

        filts = [[1, 32], [32, 32], [32, 64], [64, 64]]
        self.block1 = My_Residual_block(
            filts[0], conv1=[2, 3, 1, 1, 1, 1], conv2=[2, 3, 0, 1, 1, 2],
            conv3=[1, 3, 0, 1, 1, 2], first=True, preact=preact)
        self.block2 = My_SERes2Net_block(
            filts[1], conv1=[2, 3, 1, 1, 1, 1], conv2=[3, 3, 1, 1, 1, 2],
            conv3=[1, 3, 0, 1, 1, 2], preact=preact)
        self.block3 = My_SERes2Net_block(
            filts[2], conv1=[2, 3, 1, 1, 1, 1], conv2=[3, 3, 1, 1, 1, 2],
            conv3=[1, 3, 0, 1, 1, 2], preact=preact)
        self.block4 = My_SERes2Net_block(
            filts[3], conv1=[2, 3, 1, 1, 1, 1], conv2=[3, 3, 1, 1, 1, 2],
            conv3=[1, 3, 0, 1, 1, 2], preact=preact)

    def forward(self, x: torch.Tensor):
        # x: [B, 1, T_wave]
        feat = torch.abs(self.sinc(x)).unsqueeze(1)       # [B,1,C_sinc,T]
        feat = F.max_pool2d(feat, (self.freq_pool, 3))
        feat = self.selu(self.first_bn(feat))

        x1 = self.block1(feat)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        return x2, x3, x4


# ─────────────────────────────────────────────────────────────────────────────
# Frequency-aware tokenization, preserving explicit [T,K]
# ─────────────────────────────────────────────────────────────────────────────
class FreqAwareCompressionTK(nn.Module):
    """K-query frequency soft pooling that returns [B,T,K,C].

    For each level l and time index t:
        a_{k,f,t} = softmax_f(q_k^T X_{:,f,t})
        z_{t,k}   = sum_f a_{k,f,t} X_{:,f,t}

    Keeping K explicit avoids corrupting frequency-token semantics during
    cross-level time alignment.
    """
    def __init__(self, channels: int, n_query: int = 4):
        super().__init__()
        self.n_query = n_query
        self.score = nn.Conv2d(channels, n_query, kernel_size=1, bias=False)
        self.temp = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,F,T] -> [B,T,K,C]
        weight = F.softmax(self.score(x) * self.temp.abs(), dim=2)  # [B,K,F,T]
        out = torch.einsum('bcft,bkft->btkc', x, weight)            # [B,T,K,C]
        return out


class FreqFlattenTK(nn.Module):
    """Removed in the public release: the frequency axis is never collapsed."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("freq_flatten is not part of the released model")


def _equal_band_widths(n_freq: int, n_band: int):
    raise NotImplementedError("fixed_band tokenizer is not part of the released model")


class FreqFixedBandTK(nn.Module):
    """Removed in the public release: only the learnable K-query tokenizer is kept."""
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("fixed_band tokenizer is not part of the released model")


# ─────────────────────────────────────────────────────────────────────────────
# Prior-guided cross-level token fusion
# ─────────────────────────────────────────────────────────────────────────────
class PriorGuidedCrossLevelFusion(nn.Module):
    """Token-wise cross-level attention with learnable global level prior.

    Inputs: list of S sequences, each [B,T_i,K,C].
    Output: fused sequence [B,T_ref*K,C].

    Time lengths are aligned before flattening; K frequency-token structure is
    preserved during interpolation.

    score_s(t,k) = token_score_s(t,k) + gamma * log prior_s
    alpha_s(t,k) = softmax_s(score_s(t,k))
    z(t,k)       = sum_s alpha_s(t,k) * value_s(t,k)
    """
    def __init__(self, d_model: int, n_level: int = 3, mode: str = 'prior_attn',
                 prior_init=(0.2, 0.3, 0.5), prior_strength: float = 0.5,
                 dropout: float = 0.05, value_norm: bool = True,
                 align_level: str = 'middle'):
        super().__init__()
        assert mode in ('prior_attn', 'attn', 'prior_only', 'mean')
        assert align_level in ('shallow', 'middle', 'deep')
        self.d_model = d_model
        self.n_level = n_level
        self.mode = mode
        self.align_level = align_level
        self.eps = 1e-6

        self.level_embed = nn.Parameter(torch.zeros(n_level, d_model))
        nn.init.trunc_normal_(self.level_embed, std=0.02)
        self.score_norm = nn.LayerNorm(d_model)
        self.value_norm = nn.LayerNorm(d_model) if value_norm else nn.Identity()
        self.score = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1, bias=False),
        )

        prior_t = _parse_prior(prior_init, n_level=n_level)
        self.level_prior_logit = nn.Parameter(torch.log(prior_t))

        # gamma = softplus(prior_strength_logit). The logit is initialised with the
        # INVERSE softplus of the requested strength so that get_prior_strength()
        # returns exactly `prior_strength` at init. (The old code stored the raw
        # value, so --mlf_prior_strength 0.0 silently produced gamma=softplus(0)=0.693.)
        self.uses_prior = mode in ('prior_attn', 'prior_only')
        strength = float(prior_strength)
        if strength < 0:
            raise ValueError(f'prior_strength must be >= 0, got {strength}')
        if strength == 0.0:
            # gamma is exactly 0: the prior term is switched off, not merely small.
            self.prior_strength_logit = nn.Parameter(torch.tensor(0.0), requires_grad=False)
            self.zero_prior_strength = True
        else:
            inv_softplus = float(torch.log(torch.expm1(torch.tensor(strength))))
            self.prior_strength_logit = nn.Parameter(torch.tensor(inv_softplus))
            self.zero_prior_strength = False

        # Levels prior only receives gradient in prior_* modes; keep it out of the
        # optimizer (and out of the parameter count) otherwise.
        if not self.uses_prior:
            self.level_prior_logit.requires_grad_(False)
            self.prior_strength_logit.requires_grad_(False)

        self.drop = nn.Dropout(dropout)

        self.last_prior = None
        self.last_alpha_mean = None

    @staticmethod
    def _align_time(z: torch.Tensor, target_t: int) -> torch.Tensor:
        """Align [B,T,K,C] onto a common time grid of length ``target_t``.

        Downsampling uses adaptive average pooling, which integrates over the
        source frames instead of point-sampling them. Linear interpolation would
        alias badly here: the shallow level can be ~37x longer than the deep one,
        so point-sampling would throw away almost all of the shallow evidence
        that multi-level fusion is supposed to exploit.
        """
        if z.size(1) == target_t:
            return z
        B, T, K, C = z.shape
        zt = z.permute(0, 2, 3, 1).reshape(B * K, C, T)  # [B*K,C,T]
        if target_t < T:
            zt = F.adaptive_avg_pool1d(zt, target_t)     # anti-aliased downsample
        else:
            zt = F.interpolate(zt, size=target_t, mode='linear', align_corners=False)
        return zt.reshape(B, K, C, target_t).permute(0, 3, 1, 2)  # [B,T_ref,K,C]

    def get_prior(self) -> torch.Tensor:
        return F.softmax(self.level_prior_logit, dim=0)

    def get_prior_strength(self) -> torch.Tensor:
        if self.zero_prior_strength:
            return torch.zeros((), device=self.prior_strength_logit.device)
        return F.softplus(self.prior_strength_logit)

    def forward(self, seqs: Sequence[torch.Tensor]):
        if len(seqs) != self.n_level:
            raise ValueError(f'Expected {self.n_level} levels, got {len(seqs)}')
        if self.align_level == 'shallow':
            target_t = seqs[0].size(1)
        elif self.align_level == 'deep':
            target_t = seqs[-1].size(1)
        else:  # 'middle'
            target_t = seqs[len(seqs) // 2].size(1)
        aligned = [self._align_time(z, target_t) for z in seqs]
        stack = torch.stack(aligned, dim=1)  # [B,S,T,K,C]

        prior = self.get_prior()             # [S]
        gamma = self.get_prior_strength()    # scalar >= 0

        if self.mode == 'mean':
            alpha = stack.new_full((*stack.shape[:4], 1), 1.0 / self.n_level)  # [B,S,T,K,1]
            value = self.value_norm(stack)
            fused_tk = value.mean(dim=1)
        elif self.mode == 'prior_only':
            alpha = prior.view(1, self.n_level, 1, 1, 1).expand(*stack.shape[:4], 1)
            value = self.value_norm(stack)
            fused_tk = (alpha * value).sum(dim=1)
        else:
            level_e = self.level_embed.view(1, self.n_level, 1, 1, self.d_model)
            h = self.score_norm(stack + level_e)
            logits = self.score(h)  # [B,S,T,K,1]
            if self.mode == 'prior_attn':
                prior_log = torch.log(prior + self.eps).view(1, self.n_level, 1, 1, 1)
                logits = logits + gamma * prior_log
            alpha = F.softmax(logits, dim=1)
            value = self.value_norm(stack)
            fused_tk = (alpha * value).sum(dim=1)  # [B,T,K,C]

        B, T, K, C = fused_tk.shape
        fused = fused_tk.reshape(B, T * K, C)

        # Store lightweight analysis stats. These are detached and safe to read.
        with torch.no_grad():
            self.last_prior = prior.detach()
            self.last_alpha_mean = alpha.detach().mean(dim=(0, 2, 3, 4))  # [S]
        return self.drop(fused), alpha, aligned


# ─────────────────────────────────────────────────────────────────────────────
# Mamba backend: same external behavior as lite model
# ─────────────────────────────────────────────────────────────────────────────
class Mamba2Block(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, headdim: int = 16, layer_idx=None):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self.mixer = Mamba2(d_model, d_state=d_state, headdim=headdim,
                            layer_idx=layer_idx, use_mem_eff_path=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mixer(self.norm(x.to(self.norm.weight.dtype)).contiguous())


class FALGMambaBlock(nn.Module):
    """Local (depthwise conv) + global (Mamba-2) branch with alternating scan."""
    def __init__(self, d_model: int, d_state: int = 16, headdim: int = 16,
                 reverse: bool = False, dropout: float = 0.1, layer_idx: int = 0):
        super().__init__()
        self.reverse = reverse
        self.norm = RMSNorm(d_model)
        self.local_conv = nn.Conv1d(d_model, d_model, 3, padding=1, groups=d_model, bias=False)
        self.local_act = nn.GELU()
        self.mamba = Mamba2Block(d_model, d_state=d_state, headdim=headdim, layer_idx=layer_idx)
        self.gate = nn.Parameter(torch.tensor(0.5))
        self.drop = nn.Dropout(dropout)

    def _global(self, xn: torch.Tensor) -> torch.Tensor:
        if self.reverse:
            return self.mamba(xn.flip(1)).flip(1)
        return self.mamba(xn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = self.norm(x)
        x_local = self.local_act(self.local_conv(xn.transpose(1, 2))).transpose(1, 2)
        x_global = self._global(xn)
        g = torch.sigmoid(self.gate)
        fused = g * x_local + (1.0 - g) * x_global
        return x + self.drop(fused)


class AttentiveStatPool(nn.Module):
    def __init__(self, d_model: int, n_head: int = 4):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.dh = d_model // n_head
        self.attn = nn.Linear(d_model, n_head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, dh = self.n_head, self.dh
        w = F.softmax(self.attn(x), dim=1)
        xh = x.view(B, L, H, dh)
        mean = torch.einsum('blh,blhd->bhd', w, xh)
        mean_sq = torch.einsum('blh,blhd->bhd', w, xh.pow(2))
        std = (mean_sq - mean.pow(2)).clamp(1e-8).sqrt()
        return torch.cat([mean.reshape(B, D), std.reshape(B, D)], dim=-1)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────
class Model(nn.Module):
    """Drop-in replacement model with prior-guided multi-level fusion."""
    def __init__(self, d_args):
        super().__init__()
        d_model = d_args.get('d_model', 64)
        n_layer = d_args.get('n_layer', 6)
        d_state = d_args.get('d_state', 16)
        dropout = d_args.get('dropout', 0.1)
        num_classes = d_args.get('num_classes', 2)
        n_query = d_args.get('n_query', 4)
        headdim = d_args.get('headdim', 16)
        pool_heads = d_args.get('pool_heads', 4)
        sinc_channels = d_args.get('sinc_channels', 70)
        freq_pool = d_args.get('freq_pool', 3)

        # Released model: score = raw classifier logits (logit_source='cls'); the
        # OC head is kept only as an auxiliary loss on the embedding.
        self.use_oc = d_args.get('use_oc', False)
        self.logit_source = 'cls'
        preact = d_args.get('preact_fix', True)

        self.level_names = ('X1_shallow', 'X2_middle', 'X3_deep')
        self.front = MultiLevelSincEncoder(sinc_channels=sinc_channels, freq_pool=freq_pool, preact=preact)

        level_channels = (32, 64, 64)
        self.level_proj = nn.ModuleList([
            nn.Conv2d(c, d_model, kernel_size=1, bias=True) if c != d_model else nn.Identity()
            for c in level_channels
        ])
        # Frequency-aware K-query tokenizer (the only released tokenizer).
        self.freq_compress = nn.ModuleList([
            FreqAwareCompressionTK(d_model, n_query=n_query) for _ in level_channels
        ])

        # Token-wise cross-level attention (attn); common time grid = middle level.
        self.cross_level = PriorGuidedCrossLevelFusion(
            d_model=d_model,
            n_level=len(level_channels),
            mode='attn',
            prior_init=d_args.get('mlf_prior_init', (0.2, 0.3, 0.5)),
            prior_strength=0.0,
            dropout=d_args.get('mlf_dropout', 0.05),
            value_norm=d_args.get('mlf_value_norm', True),
            align_level='middle',
        )

        # Local-global alternating scans (scan='alt').
        self.blocks = nn.ModuleList([
            FALGMambaBlock(d_model, d_state=d_state, headdim=headdim,
                           reverse=(i % 2 == 1), dropout=dropout,
                           use_gate=True, scan='alt', layer_idx=i)
            for i in range(n_layer)
        ])
        self.pool = AttentiveStatPool(d_model, n_head=pool_heads)
        self.embed = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.GELU())
        self.cls = nn.Linear(d_model, num_classes)

        if self.use_oc:
            self.oc_head = OCSoftmax(
                d_model,
                m_real=d_args.get('oc_m_real', 0.9),
                m_fake=d_args.get('oc_m_fake', 0.2),
                alpha=d_args.get('oc_alpha', 20.0),
            )

        self.last_level_alpha = None

    def _forward_tokens(self, x: torch.Tensor):
        feats = self.front(x.unsqueeze(1))
        seqs = []
        for feat, proj, comp in zip(feats, self.level_proj, self.freq_compress):
            feat = proj(feat)
            seqs.append(comp(feat))  # [B,T,K,C]

        fused, alpha, aligned = self.cross_level(seqs)
        self.last_level_alpha = alpha.detach()
        return fused, aligned, alpha

    def get_level_prior(self) -> torch.Tensor:
        return self.cross_level.get_prior()

    def get_prior_strength(self) -> torch.Tensor:
        return self.cross_level.get_prior_strength()

    def get_mlf_weight_info(self):
        prior = self.get_level_prior().detach().cpu()
        gamma = float(self.get_prior_strength().detach().cpu())
        dyn = None
        if self.cross_level.last_alpha_mean is not None:
            dyn = self.cross_level.last_alpha_mean.detach().cpu()
        return {
            'names': self.level_names,
            'prior': prior,
            'prior_strength': gamma,
            'dynamic_alpha_mean': dyn,
        }

    def forward(self, x, return_embedding=False, **kwargs):
        seq, _aligned_seqs, _alpha = self._forward_tokens(x)

        for blk in self.blocks:
            seq = blk(seq)
        emb = self.embed(self.pool(seq))

        # Score = raw, un-normalized classifier logits; the exported score is
        # logits[:, 1] - logits[:, 0] (larger = more bonafide).
        logits = self.cls(emb)

        if return_embedding:
            return logits, emb
        return logits
