"""4 backbone (pretrained ImageNet) + head thay duoc: CE (nn.Linear) hoac IsoMax+.
Backbone: ResNet50, EfficientNet-B1, MobileNetV3-Large, ConvNeXt-Base (theo config).
IsoMax+ head = prior work (xem src/isomax.py, cite Macedo & Ludermir 2021)."""
import torch
import torch.nn as nn
import torchvision.models as tvm
from .isomax import IsoMaxPlusLossFirstPart


def build_backbone(name, pretrained=True):
    """Tra ve (feature_extractor, feat_dim): lop Linear cuoi -> Identity (xuat dac trung pooled)."""
    if name == 'resnet50':
        w = tvm.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.resnet50(weights=w); fd = m.fc.in_features; m.fc = nn.Identity()
    elif name == 'efficientnet_b1':
        w = tvm.EfficientNet_B1_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.efficientnet_b1(weights=w); fd = m.classifier[1].in_features; m.classifier[1] = nn.Identity()
    elif name == 'mobilenet_v3_large':
        w = tvm.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        m = tvm.mobilenet_v3_large(weights=w); fd = m.classifier[3].in_features; m.classifier[3] = nn.Identity()
    elif name == 'convnext_base':
        w = tvm.ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.convnext_base(weights=w); fd = m.classifier[2].in_features; m.classifier[2] = nn.Identity()
    else:
        raise ValueError(f'backbone khong ho tro: {name}')
    return m, fd


class WoodClassifier(nn.Module):
    """backbone -> features -> head. head=Linear (CE) | IsoMaxPlusLossFirstPart (isomax_plus).
    forward -> logits; forward_features -> dac trung (cho OOD/Mahalanobis sau nay)."""
    def __init__(self, backbone_name, num_classes, loss='ce', pretrained=True,
                 gradient_checkpointing=False, isomax_temperature=1.0):
        super().__init__()
        self.name = backbone_name
        self.loss_type = loss
        self.gc = bool(gradient_checkpointing)
        self.backbone, self.feat_dim = build_backbone(backbone_name, pretrained)
        if loss == 'ce':
            self.head = nn.Linear(self.feat_dim, num_classes)
        elif loss in ('isomax_plus', 'isomax'):
            self.head = IsoMaxPlusLossFirstPart(self.feat_dim, num_classes, temperature=isomax_temperature)
        else:
            raise ValueError(f'loss khong ho tro: {loss}')

    def forward_features(self, x):
        if self.gc and self.training:
            import torch.utils.checkpoint as cp
            return cp.checkpoint(self.backbone, x, use_reentrant=False)
        return self.backbone(x)

    def forward(self, x):
        return self.head(self.forward_features(x))


def build_model(cfg, backbone_name, num_classes, loss):
    bb = cfg['id_training']['backbones'][backbone_name]
    return WoodClassifier(
        backbone_name, num_classes, loss=loss,
        pretrained=cfg['id_training'].get('pretrained', True),
        gradient_checkpointing=bool(bb.get('gradient_checkpointing', False)),
        isomax_temperature=cfg['isomax'].get('temperature', 1.0),
    )
