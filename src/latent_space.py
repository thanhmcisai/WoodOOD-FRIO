"""Pha 2b — DREAM-OOD Buoc 1 (Du et al., NeurIPS 2023, "Dream the Impossible").
Hoc khong gian dac trung CO DIEU KIEN TEXT: prototype moi lop = embedding CLIP text-encoder
cua TEN KHOA HOC (co dinh). Anh encoder resnet34 + projection -> feat_dim (=768, khop CLIP ViT-L/14
= text-encoder SD1.5). Loss = cross-entropy cosine voi prototype text (vMF, eq.2), nhiet do tu config.
Xuat id_feat_{split}.npz de lay mau outlier taxonomy-guided (2c). CITE: DREAM-OOD; token_embed = CLIP.
"""
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from . import utils, data as D, sd_data as SD

CLIP_NAME = 'openai/clip-vit-large-patch14'   # ViT-L/14 -> text dim 768 (khop feature_dim & SD1.5)


def build_text_prototypes(cfg, id_classes, device):
    """Embedding CLIP-text cho ten khoa hoc moi lop ID -> (n_cls, 768) chuan hoa L2 (co dinh)."""
    from transformers import CLIPTextModelWithProjection, CLIPTokenizer
    tax = SD.load_taxonomy(cfg)
    prompts = [f"macroscopic wood cross-section of {tax[c][0]}" for c in id_classes]
    tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
    clip = CLIPTextModelWithProjection.from_pretrained(CLIP_NAME).to(device).eval()
    with torch.no_grad():
        ids = tok(prompts, padding=True, truncation=True, return_tensors='pt').to(device)
        te = clip(**ids).text_embeds.float()      # (n_cls, 768)
    te = F.normalize(te, dim=1)
    del clip; torch.cuda.empty_cache()
    return te   # (n_cls, 768)


class LatentEncoder(nn.Module):
    """resnet34 (pretrained ImageNet) -> Linear(512, feat_dim) -> L2-normalized feature."""
    def __init__(self, feat_dim=768, pretrained=True):
        super().__init__()
        w = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        m = models.resnet34(weights=w); self.bb_dim = m.fc.in_features; m.fc = nn.Identity()
        self.backbone = m
        self.proj = nn.Linear(self.bb_dim, feat_dim)

    def forward(self, x, normalize=True):
        z = self.proj(self.backbone(x))
        return F.normalize(z, dim=1) if normalize else z


@torch.no_grad()
def _proto_acc(enc, loader, proto, dev, dt):
    enc.eval(); c = t = 0
    for xb, yb in loader:
        xb = xb.to(dev)
        with torch.autocast('cuda', dtype=dt):
            z = enc(xb)
        pred = (z.float() @ proto.T).argmax(1).cpu()
        c += (pred == yb).sum().item(); t += len(yb)
    return c / max(t, 1)


def train_latent(cfg, split, seed, log=print):
    import time
    utils.set_seed(seed)
    ls = cfg['latent_space']; feat_dim = int(ls['feature_dim']); temp = float(ls['temperature'])
    max_ep = min(int(ls.get('epochs', 100)), int(ls.get('max_epochs', 40)))
    bs = int(ls.get('batch', 128)); lr = float(ls.get('lr', 1e-3))
    dev = utils.pick_device(cfg); dt = utils.amp_dtype(cfg)
    L = D.make_loaders(cfg, split, seed, batch_size=bs, roles=('train', 'val_id'))
    idc = L['id_classes']; n_cls = L['num_classes']
    proto = build_text_prototypes(cfg, idc, dev)
    enc = LatentEncoder(feat_dim=feat_dim).to(dev)
    opt = torch.optim.Adam(enc.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    scaler = torch.cuda.amp.GradScaler(enabled=(dt == torch.float16))
    tag = f'latent_{split}_seed{seed}'
    ck = utils.rp(cfg, cfg['paths']['checkpoints'], tag + '.pt')
    logf = utils.rp(cfg, cfg['paths']['logs'], 'latent.log')
    start = 0; best = -1.0
    if os.path.exists(ck):
        s = torch.load(ck, map_location=dev)
        enc.load_state_dict(s['enc']); opt.load_state_dict(s['opt']); sched.load_state_dict(s['sched'])
        start = s['epoch'] + 1; best = s['best']
        if s.get('done'):
            log(f'[{tag}] da DONE (proto_acc={best:.4f}) -> bo qua'); return {'tag': tag, 'ckpt': ck, 'best': best, 'skipped': True}
    for ep in range(start, max_ep):
        enc.train(); t0 = time.time(); run = 0.0
        for xb, yb in L['train']:
            xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dt):
                z = enc(xb); logits = (z @ proto.T) / temp; loss = F.cross_entropy(logits, yb)
            if scaler.is_enabled(): scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else: loss.backward(); opt.step()
            run += loss.item() * len(xb)
        sched.step()
        acc = _proto_acc(enc, L['val_id'], proto, dev, dt); best = max(best, acc)
        torch.save({'enc': enc.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': ep, 'best': best, 'done': False, 'feat_dim': feat_dim,
                    'id_classes': idc}, ck)
        utils.log(f'[{tag}] ep{ep+1}/{max_ep} loss={run/max(L["train_n"],1):.4f} val_proto_acc={acc:.4f} ({time.time()-t0:.0f}s)', logf)
    s = torch.load(ck, map_location=dev); s['done'] = True; torch.save(s, ck)
    return {'tag': tag, 'ckpt': ck, 'best': best, 'skipped': False, 'id_classes': idc}


@torch.no_grad()
def extract_id_feat(cfg, split, seed):
    """Trich dac trung ID (role=train) -> synth/embeddings/id_feat_{split}.npz (feats,labels,classes,proto)."""
    dev = utils.pick_device(cfg); dt = utils.amp_dtype(cfg)
    ls = cfg['latent_space']; feat_dim = int(ls['feature_dim'])
    ck = utils.rp(cfg, cfg['paths']['checkpoints'], f'latent_{split}_seed{seed}.pt')
    s = torch.load(ck, map_location=dev); idc = s['id_classes']
    enc = LatentEncoder(feat_dim=feat_dim).to(dev); enc.load_state_dict(s['enc']); enc.eval()
    proto = build_text_prototypes(cfg, idc, dev).cpu().numpy()
    L = D.make_loaders(cfg, split, seed, batch_size=256, roles=('train',))
    feats = []; labs = []
    for xb, yb in L['train']:
        with torch.autocast('cuda', dtype=dt):
            z = enc(xb.to(dev))
        feats.append(z.float().cpu().numpy()); labs.append(np.asarray(yb))
    feats = np.concatenate(feats); labs = np.concatenate(labs)
    out = utils.ensure_dir(utils.rp(cfg, cfg['paths']['synth'], 'embeddings'))
    p = os.path.join(out, f'id_feat_{split}.npz')
    np.savez_compressed(p, feats=feats.astype(np.float16), labels=labs.astype(np.int32),
                        classes=np.array(idc), proto=proto.astype(np.float16))
    return {'path': p, 'n': len(feats), 'dim': feats.shape[1], 'n_cls': len(idc)}
