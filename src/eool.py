"""Pha 3 — EOIL: IsoMax+ classifier + energy-based OOD regularization voi outlier TONG HOP (Pha 2d).
L = L_isomaxplus(ID) + beta * L_energy. Energy Ec=-T*logsumexp(logits/T).
L_energy = mean(relu(Ec_id - m_in)^2) + mean(relu(m_out - Ec_ood)^2), margin THICH UNG theo gap.
ood_dir/otypes/max_per cho ablation (random sampling, matched count). CITE: Liu(2020),Du(2023),Macedo(2021)."""
import os, glob, itertools
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from . import utils, data as D, engine as E, isomax as IM

OOD_LOCAL = '/content/synth_local/images'


def ood_base(cfg):
    return OOD_LOCAL if os.path.isdir(OOD_LOCAL) else utils.rp(cfg, cfg['paths']['synth'], 'images')


class OODFolder(Dataset):
    """Anh OOD {base}/{split}/{ot}/*.png. max_per = gioi han so anh moi otype (matched-count ablation)."""
    def __init__(self, cfg, split, tf, otypes=('near', 'far'), base=None, max_per=None):
        b = base or ood_base(cfg); self.paths = []
        for ot in otypes:
            ps = sorted(glob.glob(os.path.join(b, split, ot, '*.png')))
            if max_per:
                ps = ps[:max_per]
            self.paths += ps
        self.tf = tf
    def __len__(self): return len(self.paths)
    def __getitem__(self, i): return self.tf(Image.open(self.paths[i]).convert('RGB'))


def energy(logits, T=1.0):
    return -T * torch.logsumexp(logits / T, dim=1)


@torch.no_grad()
def _mean_energy(model, loader, dev, dt, T, n_batch=8, ood=False):
    model.eval(); es = []
    for i, b in enumerate(loader):
        x = (b if ood else b[0]).to(dev)
        with torch.autocast('cuda', dtype=dt):
            lo = model(x)
        es.append(energy(lo.float(), T).cpu())
        if i + 1 >= n_batch: break
    return float(torch.cat(es).mean())


def train_eool(cfg, backbone, split, seed=0, log=print, ood_dir=None, otypes=('near', 'far'),
               ckpt_suffix='', max_per=None):
    dev = utils.pick_device(cfg); dt = utils.amp_dtype(cfg); ec = cfg['eool']
    T = float(ec.get('T', 1.0)); beta = float(ec.get('energy_weight_beta', ec.get('energy_weight', 1.0)))
    epochs = int(ec.get('epochs', 10)); lr = float(ec.get('lr', 1e-4)); gap = float(ec.get('margin_gap', 1.0))
    bs = int(cfg['id_training']['backbones'][backbone]['batch'])
    out = utils.rp(cfg, cfg['paths']['checkpoints'], f'eool{ckpt_suffix}_{backbone}_{split}_seed{seed}.pt')
    if os.path.exists(out) and not cfg.get('force'):
        log(f'[eool] {out} da co -> bo qua'); return out
    model, n_cls = E.load_best(cfg, backbone, 'isomax_plus', split, seed)
    model = model.to(dev)
    crit = IM.IsoMaxPlusLossSecondPart(entropic_scale=float(cfg['isomax']['entropic_scale']))
    Lid = D.make_loaders(cfg, split, seed, batch_size=bs, roles=('train',))
    idl = Lid['train']
    tf = D.build_transforms(cfg, train=True)
    ood_ds = OODFolder(cfg, split, tf, otypes=otypes, base=ood_dir, max_per=max_per)
    ool = DataLoader(ood_ds, batch_size=bs, shuffle=True,
                     num_workers=cfg['hardware']['num_workers'], pin_memory=True, drop_last=True)
    e_id0 = _mean_energy(model, idl, dev, dt, T); e_ood0 = _mean_energy(model, ool, dev, dt, T, ood=True)
    m_in = e_id0 - gap; m_out = e_ood0 + gap
    log(f'[eool{ckpt_suffix} {backbone} {split}] N_ood={len(ood_ds)} E_id={e_id0:.3f} E_ood={e_ood0:.3f} m=({m_in:.2f},{m_out:.2f})')
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(dt == torch.float16))
    model.train()
    for ep in range(epochs):
        oiter = itertools.cycle(ool); tc = te = 0.0; nb = 0
        for x, y in idl:
            xo = next(oiter); x, y, xo = x.to(dev), y.to(dev), xo.to(dev)
            opt.zero_grad()
            with torch.autocast('cuda', dtype=dt):
                lo_id = model(x); lo_od = model(xo)
                l_cls = crit(lo_id, y)
                l_en = (F.relu(energy(lo_id.float(), T) - m_in) ** 2).mean() + \
                       (F.relu(m_out - energy(lo_od.float(), T)) ** 2).mean()
                loss = l_cls + beta * l_en
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            tc += float(l_cls); te += float(l_en); nb += 1
        with torch.no_grad():
            ei = _mean_energy(model, idl, dev, dt, T); eo = _mean_energy(model, ool, dev, dt, T, ood=True)
        log(f'[eool{ckpt_suffix} {backbone} {split}] ep{ep+1}/{epochs} cls={tc/nb:.4f} en={te/nb:.4f} gap={eo-ei:.3f}')
        model.train()
    torch.save({'model': model.state_dict(), 'n_cls': n_cls, 'backbone': backbone,
                'm_in': m_in, 'm_out': m_out, 'T': T, 'beta': beta}, out)
    del model; torch.cuda.empty_cache()
    log(f'[eool] saved {out}')
    return out
