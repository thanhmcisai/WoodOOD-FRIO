#!/usr/bin/env python3
"""Pha 0 - Gom cum PSEUDO-SPECIMEN khi da mat metadata mau vat.

Y tuong: anh cung mot khoi go se giong nhau o (1) muc pixel [aHash] va/hoac
(2) muc cau truc giai phau [dac trung sau ConvNeXt-Base ImageNet pretrained].
Ta noi canh giua cac anh CUNG LOP neu Hamming(aHash) <= t1 HOAC cosine(feat) >= t2,
roi union-find -> moi thanh phan lien thong = 1 pseudo-specimen. make_splits.py se
chia train/val/test theo cum nay de giam ro ri (leakage).

QUAN TRONG:
  - Dung ConvNeXt-Base ImageNet pretrained, DONG BANG (KHONG fine-tune tren go). Neu dung
    chinh model dang train tren go de dinh nghia cum roi lai chia du lieu cho no -> vong lap
    phuong phap, reviewer se bat loi.
  - Chi noi canh TRONG CUNG MOT LOP (muc tieu: cung KHOI GO, khong phai cung LOAI).
  - Day la XAP XI specimen -> phai ghi ro limitation trong bai.

Chay:
  # buoc 1 (nang): trich dac trung + aHash (co cache), roi gom cum voi nguong mac dinh
  python cluster_pseudo_specimen.py --data_root data/Wood_ID_1-50 --out_dir results/pseudo \\
      --hamming_thresh 5 --cosine_thresh 0.90 --batch_size 64

  # buoc 2 (nhe): thu nguong khac -> KHONG trich lai (dung cache)
  python cluster_pseudo_specimen.py --data_root data/Wood_ID_1-50 --out_dir results/pseudo \\
      --cosine_thresh 0.85

Dau ra:
  pseudo_specimen_map.csv   image_path,class,pseudo_specimen_id   (-> nap vao make_splits.py)
  cluster_stats.csv         thong ke kich thuoc cum moi lop
  cosine_hist.png           histogram cosine noi-lop (chon nguong o "thung lung")
  summary.json
Phu thuoc: numpy, scipy, pillow, torch, torchvision (matplotlib/tqdm tuy chon).
"""
import argparse, os, json, csv
import numpy as np

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


# ---------------------------------------------------------------- aHash
def average_hash_pil(pil_img, hash_size=8):
    g = pil_img.convert('L').resize((hash_size, hash_size))
    a = np.asarray(g, dtype=np.float32)
    return (a > a.mean()).astype(np.uint8).flatten()


# ---------------------------------------------------------------- union-find
class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


# ---------------------------------------------------------------- feature extraction (lazy torch)
def pick_device(requested):
    import torch
    if requested == 'cpu':
        return torch.device('cpu')
    if torch.cuda.is_available():
        try:
            torch.zeros(1).cuda()
            return torch.device('cuda')
        except Exception as e:
            print(f'[!] CUDA loi ({e}) -> fallback CPU.')
    else:
        print('[i] Khong thay GPU -> chay CPU (cham hon nhung on).')
    return torch.device('cpu')


def build_convnext(device):
    """ConvNeXt-Base ImageNet pretrained; bo lop Linear cuoi -> tra dac trung 1024-D."""
    import torch, torch.nn as nn
    from torchvision import models
    try:
        model = models.convnext_base(weights=models.ConvNeXt_Base_Weights.IMAGENET1K_V1)
    except Exception as e:
        raise SystemExit(f'[LOI] Khong tai duoc trong so ConvNeXt-Base (can mang): {e}')
    model.classifier[-1] = nn.Identity()   # thay Linear cuoi -> lay feature truoc phan loai
    model.eval().to(device)
    return model


def _make_dataset(items, data_root, hash_size):
    import torch
    from torch.utils.data import Dataset
    from torchvision import transforms
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    class DS(Dataset):
        def __len__(self): return len(items)

        def __getitem__(self, i):
            cls, fn = items[i]
            fp = os.path.join(data_root, cls, fn)
            try:
                with Image.open(fp) as im:
                    im = im.convert('RGB')
                    x = tf(im)
                    ah = average_hash_pil(im, hash_size)
                return x, torch.from_numpy(ah), i, True
            except Exception:
                return (torch.zeros(3, 224, 224), torch.zeros(hash_size * hash_size, dtype=torch.uint8),
                        i, False)
    return DS()


def extract_features(data_root, device_str, batch_size, hash_size, num_workers,
                     cache_path, recompute):
    """Tra ve: keys(list 'class/fn'), class_ids(np int), classes(list), feats(np f16), hashes(np u8)."""
    if cache_path and os.path.exists(cache_path) and not recompute:
        print(f'[i] Nap cache dac trung: {cache_path}')
        z = np.load(cache_path, allow_pickle=True)
        return (list(z['keys']), z['class_ids'], list(z['classes']),
                z['feats'], z['hashes'])

    import torch
    from torch.utils.data import DataLoader
    device = pick_device(device_str)
    model = build_convnext(device)

    classes = sorted(d for d in os.listdir(data_root)
                     if os.path.isdir(os.path.join(data_root, d)))
    items, class_ids = [], []
    for ci, c in enumerate(classes):
        for fn in sorted(os.listdir(os.path.join(data_root, c))):
            if os.path.splitext(fn)[1].lower() in IMG_EXT:
                items.append((c, fn)); class_ids.append(ci)
    if not items:
        raise SystemExit(f'Khong thay anh trong {data_root}')
    print(f'[i] {len(items)} anh / {len(classes)} lop. Trich dac trung tren {device}...')

    ds = _make_dataset(items, data_root, hash_size)
    loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    try:
        from tqdm import tqdm
        loader = tqdm(loader, total=len(loader))
    except Exception:
        pass

    feats = np.zeros((len(items), 1024), dtype=np.float16)
    hashes = np.zeros((len(items), hash_size * hash_size), dtype=np.uint8)
    ok_mask = np.zeros(len(items), dtype=bool)
    with torch.no_grad():
        for x, ah, idx, ok in loader:
            try:
                f = model(x.to(device)).float().cpu().numpy()
            except RuntimeError as e:
                raise SystemExit(f'[LOI] {e}\n  -> Thu giam --batch_size hoac chay --device cpu.')
            idx = idx.numpy()
            feats[idx] = f.astype(np.float16)
            hashes[idx] = ah.numpy().astype(np.uint8)
            ok_mask[idx] = ok.numpy().astype(bool)

    n_bad = int((~ok_mask).sum())
    if n_bad:
        print(f'[!] {n_bad} anh loi/khong doc duoc -> bo qua.')
    keys = [f'{items[i][0]}/{items[i][1]}' for i in range(len(items)) if ok_mask[i]]
    class_ids = np.array([class_ids[i] for i in range(len(items)) if ok_mask[i]], dtype=np.int32)
    feats, hashes = feats[ok_mask], hashes[ok_mask]

    if cache_path:
        os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
        np.savez_compressed(cache_path, keys=np.array(keys), class_ids=class_ids,
                            classes=np.array(classes), feats=feats, hashes=hashes)
        print(f'[i] Da luu cache: {cache_path}')
    return keys, class_ids, classes, feats, hashes


# ---------------------------------------------------------------- clustering (thuan numpy/scipy)
def cluster_within_classes(keys, class_ids, classes, feats, hashes,
                           hamming_thresh, cosine_thresh, hist_cap=1_000_000):
    """Noi canh trong cung lop (Hamming<=t1 HOAC cosine>=t2), union-find -> pseudo_specimen_id."""
    from scipy.spatial.distance import pdist, squareform
    N = len(keys)
    cluster_of = [''] * N
    stats = []           # (class, n, n_clusters, largest, mean_size, singletons)
    hist_vals = []
    n_bits = hashes.shape[1]

    for ci, cname in enumerate(classes):
        idxs = np.where(class_ids == ci)[0]
        n = len(idxs)
        if n == 0:
            continue
        if n == 1:
            cluster_of[idxs[0]] = f'{cname}__ps0000'
            stats.append((cname, 1, 1, 1, 1.0, 1))
            continue

        H = hashes[idxs]
        F = feats[idxs].astype(np.float32)
        norm = np.linalg.norm(F, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        F = F / norm
        ham = squareform(pdist(H, metric='hamming') * n_bits)   # don vi bit
        cos = F @ F.T

        # thu thap cosine noi-lop cho histogram (tam giac tren)
        iu = np.triu_indices(n, k=1)
        if len(hist_vals) < hist_cap:
            hist_vals.append(cos[iu].astype(np.float32))

        uf = UnionFind(n)
        link = (ham <= hamming_thresh) | (cos >= cosine_thresh)
        ii, jj = np.where(np.triu(link, k=1))
        for a, b in zip(ii, jj):
            uf.union(int(a), int(b))

        roots = [uf.find(i) for i in range(n)]
        remap, cl_local = {}, []
        for r in roots:
            if r not in remap:
                remap[r] = len(remap)
            cl_local.append(remap[r])
        sizes = np.bincount(cl_local)
        for local_i, r_local in enumerate(cl_local):
            cluster_of[idxs[local_i]] = f'{cname}__ps{r_local:04d}'
        stats.append((cname, n, len(sizes), int(sizes.max()),
                      float(sizes.mean()), int((sizes == 1).sum())))

    hist = np.concatenate(hist_vals) if hist_vals else np.array([])
    return cluster_of, stats, hist


# ---------------------------------------------------------------- outputs
def write_outputs(out_dir, keys, cluster_of, stats, hist, hamming_thresh, cosine_thresh):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'pseudo_specimen_map.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(['image_path', 'class', 'pseudo_specimen_id'])
        for k, cl in zip(keys, cluster_of):
            w.writerow([k, k.split('/')[0], cl])
    with open(os.path.join(out_dir, 'cluster_stats.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class', 'n_images', 'n_pseudo_specimens', 'largest_cluster',
                    'mean_cluster_size', 'singletons'])
        for row in stats:
            w.writerow(row)

    n_img = len(keys)
    n_clu = len({c for c in cluster_of})
    summary = {
        'hamming_thresh': hamming_thresh, 'cosine_thresh': cosine_thresh,
        'n_images': n_img, 'n_pseudo_specimens_total': n_clu,
        'avg_images_per_pseudo_specimen': round(n_img / max(n_clu, 1), 2),
        'reduction_ratio': round(n_clu / max(n_img, 1), 3),
        'cosine_percentiles_within_class': (
            {p: round(float(np.percentile(hist, p)), 4) for p in [50, 75, 90, 95, 99]}
            if hist.size else None),
    }
    with open(os.path.join(out_dir, 'summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # histogram (tuy chon)
    if hist.size:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.figure(figsize=(7, 4))
            plt.hist(hist, bins=80)
            plt.axvline(cosine_thresh, color='r', ls='--', label=f'cosine_thresh={cosine_thresh}')
            plt.xlabel('Cosine similarity (within-class image pairs)')
            plt.ylabel('Count'); plt.legend(); plt.tight_layout()
            plt.savefig(os.path.join(out_dir, 'cosine_hist.png'), dpi=150)
            plt.close()
        except Exception as e:
            print(f'[i] Bo qua histogram (thieu matplotlib?): {e}')
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir', default='results/pseudo')
    ap.add_argument('--hamming_thresh', type=int, default=5,
                    help='Hamming(aHash) <= nguong => cung khoi (mac dinh 5)')
    ap.add_argument('--cosine_thresh', type=float, default=0.90,
                    help='cosine(feat) >= nguong => cung khoi. Chon o "thung lung" cosine_hist.png')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--hash_size', type=int, default=8)
    ap.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    ap.add_argument('--cache', default=None,
                    help='duong dan cache .npz (mac dinh <out_dir>/_features_cache.npz)')
    ap.add_argument('--recompute_features', action='store_true',
                    help='ep trich lai dac trung (bo qua cache)')
    a = ap.parse_args()

    cache = a.cache or os.path.join(a.out_dir, '_features_cache.npz')
    keys, class_ids, classes, feats, hashes = extract_features(
        a.data_root, a.device, a.batch_size, a.hash_size, a.num_workers,
        cache, a.recompute_features)

    cluster_of, stats, hist = cluster_within_classes(
        keys, class_ids, classes, feats, hashes, a.hamming_thresh, a.cosine_thresh)

    summary = write_outputs(a.out_dir, keys, cluster_of, stats, hist,
                            a.hamming_thresh, a.cosine_thresh)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f'\n[i] Da ghi {a.out_dir}/pseudo_specimen_map.csv. '
          f'Nap vao make_splits.py qua --pseudo_specimen_map.')
    print('[i] Xem cosine_hist.png + cluster_stats.csv de KIEM nguong: '
          'neu 1 lop go lai chi con vai cum lon -> nguong qua long (dang gop ca loai).')


if __name__ == '__main__':
    main()
