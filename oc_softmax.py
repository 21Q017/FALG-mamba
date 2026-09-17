"""OC-Softmax head for audio anti-spoofing.

Labels follow the repository convention:
    spoof    -> 0
    bonafide -> 1

This implementation is self-contained so the overlay can run in a RawTFNet
project without relying on extra user-local files.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class OCSoftmax(nn.Module):
    """One-class softmax loss with a single bonafide center.

    The score is cosine similarity to a learnable center. During training,
    bonafide samples are pushed above m_real and spoof samples below m_fake.
    """

    def __init__(self, feat_dim: int, m_real: float = 0.9, m_fake: float = 0.2, alpha: float = 20.0):
        super().__init__()
        self.center = nn.Parameter(torch.randn(feat_dim))
        nn.init.normal_(self.center, mean=0.0, std=0.01)
        self.m_real = float(m_real)
        self.m_fake = float(m_fake)
        self.alpha = float(alpha)

    def score(self, emb: torch.Tensor) -> torch.Tensor:
        emb_n = F.normalize(emb, dim=-1)
        cen_n = F.normalize(self.center, dim=0)
        return torch.matmul(emb_n, cen_n)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        labels = labels.long().view(-1)
        s = self.score(emb)
        is_real = labels == 1
        # bonafide: penalize scores below m_real; spoof: penalize scores above m_fake.
        real_loss = F.softplus(self.alpha * (self.m_real - s[is_real])) if is_real.any() else s.new_zeros(())
        fake_loss = F.softplus(self.alpha * (s[~is_real] - self.m_fake)) if (~is_real).any() else s.new_zeros(())
        n_real = int(is_real.sum().item())
        n_fake = int((~is_real).sum().item())
        total = max(n_real + n_fake, 1)
        return (real_loss.sum() + fake_loss.sum()) / total
