"""Pha 2c — Lay mau outlier embedding TAXONOMY-GUIDED (bien the NPOS / DREAM-OOD Buoc 2).
DONG GOP MOI cua du an:
  - NEAR-OOD: noi suy embedding giua cac loai CUNG CHI (fallback cung HO) -> "loai la cung chi".
  - FAR-OOD : lay mau XA moi cum HO (nhieu manh + chon diem xa moi centroid) -> "loai khac ho".
Khong gian = dac trung 768-D da can text (2b). Xuat outlier_embed_{split}.npz (embed, otype).
CITE: NPOS (Tao et al., ICLR 2023), DREAM-OOD (Du et al., NeurIPS 2023).
"""
import os
import numpy as np
from . import utils, sd_data as SD


def _l2n(x, axis=-1):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def load_id_feat(cfg, split):
    p = utils.rp(cfg, cfg['paths']['synth'], 'embeddings', f'id_feat_{split}.npz')
    z = np.load(p, allow_pickle=True)
    return z['feats'].astype(np.float32), z['labels'].astype(int), list(z['classes']), z['proto'].astype(np.float32)


def class_centroids(feats, labels, n_cls):
    c = np.zeros((n_cls, feats.shape[1]), np.float32)
    for k in range(n_cls):
        m = feats[labels == k]
        c[k] = m.mean(0) if len(m) else 0.0
    return _l2n(c)


def _tax_groups(cfg, classes):
    """genus->[idx], family->[idx] tren cac lop ID (dùng ten lop trong split)."""
    tax = SD.load_taxonomy(cfg)                      # class -> (sci, genus)
    import csv
    fam = {}
    with open(utils.rp(cfg, cfg['paths']['taxonomy']), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            fam[r['class_name'].strip()] = r['family'].strip()
    genus = {}; family = {}
    for i, c in enumerate(classes):
        g = tax.get(c, (c, c))[1]; fa = fam.get(c, c)
        genus.setdefault(g, []).append(i); family.setdefault(fa, []).append(i)
    return genus, family


def sample_near(centroids, genus, family, n, rng, mag):
    """Noi suy giua 2 loai cung chi (fallback cung ho). Tra ve (n,768) L2-normalized."""
    pairs_g = [v for v in genus.values() if len(v) >= 2]
    pairs_f = [v for v in family.values() if len(v) >= 2]
    pools = pairs_g if pairs_g else pairs_f
    if not pools:                                    # khong co nhom -> noi suy 2 centroid gan nhau bat ky
        pools = [list(range(len(centroids)))]
    out = []
    for _ in range(n):
        grp = pools[rng.integers(len(pools))]
        a, b = rng.choice(grp, size=2, replace=False)
        al = rng.uniform(0.3, 0.7)
        e = al * centroids[a] + (1 - al) * centroids[b] + rng.normal(0, mag, centroids.shape[1])
        out.append(e)
    return _l2n(np.asarray(out, np.float32))


def sample_far(centroids, feats, n, rng, mag_far, n_cand_mult=8):
    """Nhieu manh quanh diem ID roi CHON diem xa moi centroid nhat (xa moi cum ho)."""
    dim = feats.shape[1]; cent = centroids
    n_cand = max(n * n_cand_mult, 2000)
    base = feats[rng.integers(0, len(feats), size=n_cand)]
    cand = _l2n(base + rng.normal(0, mag_far, (n_cand, dim)).astype(np.float32))
    # khoang cach cosine toi centroid GAN nhat (lon = xa moi cum)
    sim = cand @ cent.T                              # (n_cand, n_cls)
    d_near = 1.0 - sim.max(1)                         # xa centroid gan nhat
    idx = np.argsort(-d_near)[:n]                     # chon xa nhat
    return cand[idx]


def generate_outliers(cfg, split, seed=0):
    sp = cfg['sampling']; sy = cfg['synth']
    rng = np.random.default_rng(seed)
    feats, labels, classes, proto = load_id_feat(cfg, split)
    n_cls = len(classes)
    cent = class_centroids(feats, labels, n_cls)
    genus, family = _tax_groups(cfg, classes)
    n_total = int(sy['n_outlier_per_id_class']) * n_cls
    n_near = int(round(float(sp.get('near_frac', 0.5)) * n_total)); n_far = n_total - n_near
    mag = float(sp['gaussian_mag'])
    near = sample_near(cent, genus, family, n_near, rng, mag)
    far = sample_far(cent, feats, n_far, rng, mag_far=mag * 5.0)
    embed = np.concatenate([near, far], 0).astype(np.float16)
    otype = np.array(['near'] * len(near) + ['far'] * len(far))
    out = utils.ensure_dir(utils.rp(cfg, cfg['paths']['synth'], 'embeddings'))
    p = os.path.join(out, f'outlier_embed_{split}.npz')
    np.savez_compressed(p, embed=embed, otype=otype, n_cls=n_cls)
    # sanity: khoang cach trung binh toi centroid gan nhat (far phai > near)
    dn = 1.0 - (near @ cent.T).max(1).mean(); df = 1.0 - (far @ cent.T).max(1).mean()
    return {'path': p, 'n_near': len(near), 'n_far': len(far), 'dim': embed.shape[1],
            'near_meandist': round(float(dn), 4), 'far_meandist': round(float(df), 4),
            'n_genus_ge2': sum(len(v) >= 2 for v in genus.values()),
            'n_family_ge2': sum(len(v) >= 2 for v in family.values())}
