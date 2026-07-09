"""Pha 2e — danh gia chat luong anh OOD sinh (Q1 doi, bai cu thieu):
 - FID(anh sinh vs anh THAT) tung split.
 - Do DA DANG (mean pairwise cosine distance dac trung ImageNet resnet18).
 - SANITY outlier: model ID (Pha1) KHONG duoc tu tin cao tren anh OOD (conf near/far < real test_id).
Ghi results/synth_quality.csv + luoi anh mau near/far. CITE: DREAM-OOD (danh gia FID/diversity)."""
import os, glob, random
import numpy as np
import torch, torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
from . import utils, data as D, engine as E


def _gen_paths(cfg, split, otype):
    return sorted(glob.glob(os.path.join(utils.rp(cfg, cfg['paths']['synth'], 'images', split, otype), '*.png')))


def _real_paths(cfg, split, n, seed=0):
    """Anh THAT cua cac lop ID trong split (tu data_local)."""
    df = D.read_manifest(cfg, split); idc = set(D.id_classes(df))
    droot = utils.data_root(cfg); rng = random.Random(seed); out = []
    for c in idc:
        cd = os.path.join(droot, c)
        if os.path.isdir(cd):
            out += [os.path.join(cd, f) for f in os.listdir(cd) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    rng.shuffle(out); return out[:n]


@torch.no_grad()
def _diversity(paths, dev, n=400, seed=0):
    rng = random.Random(seed); ps = rng.sample(paths, min(n, len(paths)))
    m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1); m.fc = torch.nn.Identity(); m.eval().to(dev)
    tf = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),
                             transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    zs = []
    for i in range(0, len(ps), 64):
        x = torch.stack([tf(Image.open(p).convert('RGB')) for p in ps[i:i+64]]).to(dev)
        zs.append(F.normalize(m(x), dim=1).cpu())
    z = torch.cat(zs); sim = z @ z.T; n_ = len(z)
    mean_off = (sim.sum() - n_) / (n_ * (n_ - 1))
    return float(1 - mean_off)                                   # cao = da dang


@torch.no_grad()
def _fid(gen_paths, real_paths, dev, n=1000, seed=0):
    from torchmetrics.image.fid import FrechetInceptionDistance
    tf = transforms.Compose([transforms.Resize(299), transforms.CenterCrop(299), transforms.ToTensor()])
    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(dev)
    rng = random.Random(seed)
    for paths, real in [(rng.sample(real_paths, min(n, len(real_paths))), True),
                        (rng.sample(gen_paths, min(n, len(gen_paths))), False)]:
        for i in range(0, len(paths), 32):
            x = torch.stack([tf(Image.open(p).convert('RGB')) for p in paths[i:i+32]]).to(dev)
            fid.update(x, real=real)
    return float(fid.compute())


@torch.no_grad()
def _id_conf(cfg, split, paths, dev, dt, n=500, seed=0):
    """Mean max-softmax cua model ID (mobilenet ce) tren tap anh."""
    rng = random.Random(seed); ps = rng.sample(paths, min(n, len(paths)))
    model, _ = E.load_best(cfg, 'mobilenet_v3_large', 'ce', split, 0)
    tf = D.build_transforms(cfg, train=False); confs = []
    for i in range(0, len(ps), 64):
        x = torch.stack([tf(Image.open(p).convert('RGB')) for p in ps[i:i+64]]).to(dev)
        with torch.autocast('cuda', dtype=dt):
            p = F.softmax(model(x).float(), 1)
        confs.append(p.max(1).values.cpu())
    del model; torch.cuda.empty_cache()
    return float(torch.cat(confs).mean())


def evaluate_split(cfg, split, dev, dt):
    near = _gen_paths(cfg, split, 'near'); far = _gen_paths(cfg, split, 'far'); allg = near + far
    real = _real_paths(cfg, split, 3000)
    fid = _fid(allg, real, dev)
    div_gen = _diversity(allg, dev); div_real = _diversity(real, dev)
    c_near = _id_conf(cfg, split, near, dev, dt); c_far = _id_conf(cfg, split, far, dev, dt)
    # real test_id conf
    dfl = D.make_loaders(cfg, split, 0, batch_size=64, num_workers=4, roles=('test_id',))
    model, _ = E.load_best(cfg, 'mobilenet_v3_large', 'ce', split, 0); cr = []
    for xb, _ in dfl['test_id']:
        with torch.no_grad(), torch.autocast('cuda', dtype=dt):
            pr = F.softmax(model(xb.to(dev)).float(), 1)
        cr.append(pr.max(1).values.cpu())
    del model; torch.cuda.empty_cache()
    c_real = float(torch.cat(cr).mean())
    return {'scenario': split, 'fid': round(fid, 2), 'div_gen': round(div_gen, 4), 'div_real': round(div_real, 4),
            'conf_gen_near': round(c_near, 4), 'conf_gen_far': round(c_far, 4), 'conf_real_id': round(c_real, 4),
            'n_gen': len(allg), 'n_real': len(real)}
