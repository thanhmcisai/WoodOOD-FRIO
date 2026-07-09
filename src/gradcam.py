"""Pha 7 — Grad-CAM cho cap loai DE NHAM cung chi (giai phau go). CITE: Selvaraju et al. 2017."""
import os, glob
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
from . import utils, data as D, engine as E


def _same_genus_pair(cfg, split):
    idc = D.id_classes(D.read_manifest(cfg, split))
    p = utils.rp(cfg, 'taxonomy.csv')
    if os.path.exists(p):
        import unicodedata
        tax = pd.read_csv(p)
        cc = [c for c in tax.columns if 'class' in c.lower()][0]
        gc = [c for c in tax.columns if 'genus' in c.lower()][0]
        norm = lambda s: unicodedata.normalize('NFC', str(s))
        t = tax[[cc, gc]].copy(); t[cc] = t[cc].map(norm)
        idn = set(norm(c) for c in idc)
        t = t[t[cc].isin(idn)]
        for genus, sub in t.groupby(gc):
            cls = sub[cc].tolist()
            if len(cls) >= 2:
                # map lai ve ten thu muc thuc
                back = {norm(c): c for c in idc}
                return str(genus), [back[cls[0]], back[cls[1]]]
    return 'pair', idc[:2]


def gradcam_confusable(cfg, backbone='mobilenet_v3_large', split='40_10', n_per=3, log=print):
    dev = utils.pick_device(cfg)
    genus, classes = _same_genus_pair(cfg, split)
    log(f'[gradcam] genus={genus} classes={classes}')
    model, _ = E.load_best(cfg, backbone, 'ce', split, 0); model.eval()
    convs = [m for m in model.backbone.modules() if isinstance(m, nn.Conv2d)]
    target = convs[-1]
    store = {}
    target.register_forward_hook(lambda m, i, o: store.__setitem__('a', o))
    target.register_full_backward_hook(lambda m, gi, go: store.__setitem__('g', go[0].detach()))
    tf = D.build_transforms(cfg, train=False); droot = utils.data_root(cfg)
    fig, axs = plt.subplots(2, n_per, figsize=(3 * n_per, 6))
    for r, cls in enumerate(classes):
        imgs = sorted(glob.glob(os.path.join(droot, cls, '*')))[:n_per]
        for c, ip in enumerate(imgs):
            im = Image.open(ip).convert('RGB'); x = tf(im).unsqueeze(0).to(dev).requires_grad_(True)
            store.clear(); logits = model(x); k = int(logits.argmax(1))
            model.zero_grad(); logits[0, k].backward()
            a = store['a'].detach()[0]; g = store['g'][0]
            w = g.mean((1, 2)); cam = F.relu((w[:, None, None] * a).sum(0))
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            cam = F.interpolate(cam[None, None], size=(im.height, im.width), mode='bilinear')[0, 0].cpu().numpy()
            ax = axs[r, c] if n_per > 1 else axs[r]
            ax.imshow(im); ax.imshow(cam, cmap='jet', alpha=.45); ax.axis('off')
            if c == 0: ax.set_ylabel(cls, fontsize=8)
            ax.set_title(cls[:14], fontsize=7)
    fig.suptitle(f'Grad-CAM — cap de nham cung chi ({genus})')
    d = utils.rp(cfg, cfg['paths']['figures']); utils.ensure_dir(d)
    plt.tight_layout(); fig.savefig(os.path.join(d, 'gradcam_confusable.png'), dpi=300)
    fig.savefig(os.path.join(d, 'gradcam_confusable.pdf')); plt.close(fig)
    return 'gradcam_confusable'
