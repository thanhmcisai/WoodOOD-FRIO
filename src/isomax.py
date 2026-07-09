"""IsoMax+ loss — PRIOR WORK, KHONG phai dong gop cua du an.

Trich dan BAT BUOC:
  Macedo, D., & Ludermir, T. (2021). "Enhanced Isotropy Maximization Loss:
  Seamless and High-Performance Out-of-Distribution Detection Simply Replacing
  the SoftMax Loss." (IsoMax+).
  Repo goc: https://github.com/dlmacedo/entropic-out-of-distribution-detection
  (reference_code/entropic-ood/losses/isomaxplus.py) — copy trung thanh phan loss.

Cach dung:
  head = IsoMaxPlusLossFirstPart(num_features, num_classes)   # thay nn.Linear cuoi
  logits = head(features)
  loss   = IsoMaxPlusLossSecondPart(entropic_scale=10.0)(logits, targets)  # thay CrossEntropyLoss
OOD score (suy luan): minimum distance = logits.max(1) (logit lon = gan prototype).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class IsoMaxPlusLossFirstPart(nn.Module):
    """Thay lop output nn.Linear() cua model (Macedo & Ludermir 2021)."""
    def __init__(self, num_features, num_classes, temperature=1.0):
        super().__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.temperature = temperature
        self.prototypes = nn.Parameter(torch.Tensor(num_classes, num_features))
        self.distance_scale = nn.Parameter(torch.Tensor(1))
        nn.init.normal_(self.prototypes, mean=0.0, std=1.0)
        nn.init.constant_(self.distance_scale, 1.0)

    def forward(self, features):
        distances = torch.abs(self.distance_scale) * torch.cdist(
            F.normalize(features), F.normalize(self.prototypes),
            p=2.0, compute_mode="donot_use_mm_for_euclid_dist")
        logits = -distances
        return logits / self.temperature   # temperature co the calibrate SAU train


class IsoMaxPlusLossSecondPart(nn.Module):
    """Thay nn.CrossEntropyLoss() (Macedo & Ludermir 2021)."""
    def __init__(self, entropic_scale=10.0):
        super().__init__()
        self.entropic_scale = entropic_scale

    def forward(self, logits, targets):
        # Probabilities va logarit tinh RIENG & tuan tu -> KHONG dung nn.CrossEntropyLoss.
        distances = -logits
        probs = nn.Softmax(dim=1)(-self.entropic_scale * distances)
        probs_at_targets = probs[range(distances.size(0)), targets]
        return -torch.log(probs_at_targets).mean()
