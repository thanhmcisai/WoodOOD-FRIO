#!/usr/bin/env python3
"""Pha 0 - Kiem toan dataset WoodOOD-50.

Dem anh/lop, do phan giai, file hong, va phat hien anh GAN TRUNG (average-hash)
de canh bao nguy co ro ri o cap mau vat (specimen-level leakage).

Cach chay:
    python audit_dataset.py --data_root /path/to/Wood_ID_1-50 --out_dir results/audit

Phu thuoc: numpy, pillow, scipy
"""
import argparse, os, json, sys, csv
from collections import defaultdict
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def ahash(path, hash_size=8):
    """Average hash -> mang bit uint8 do dai hash_size*hash_size."""
    with Image.open(path) as im:
        im = im.convert('L').resize((hash_size, hash_size), Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32)
    return (arr > arr.mean()).astype(np.uint8).flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_dir', default='results/audit')
    ap.add_argument('--hash_size', type=int, default=8)
    ap.add_argument('--dup_threshold', type=int, default=5,
                    help='Khoang cach Hamming <= nguong => coi la gan trung')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    classes = sorted(d for d in os.listdir(args.data_root)
                     if os.path.isdir(os.path.join(args.data_root, d)))
    if not classes:
        sys.exit(f'Khong thay thu muc lop nao trong {args.data_root}')

    rows, corrupt = [], []
    hashes = {}
    per_class = defaultdict(int)
    res_counter = defaultdict(int)
    exact = defaultdict(list)   # hash bytes -> ["class/file", ...] (phat hien trung y het toan cuc)

    for c in classes:
        cdir = os.path.join(args.data_root, c)
        hashes[c] = []
        for fn in sorted(os.listdir(cdir)):
            if os.path.splitext(fn)[1].lower() not in IMG_EXT:
                continue
            fp = os.path.join(cdir, fn)
            try:
                with Image.open(fp) as im:
                    w, h, mode = im.size[0], im.size[1], im.mode
                    im.verify()                       # phat hien file hong
                b = ahash(fp, args.hash_size)
                rows.append((c, fn, w, h, mode, 'ok'))
                per_class[c] += 1
                res_counter[f'{w}x{h}'] += 1
                hashes[c].append((fn, b))
                exact[b.tobytes()].append(f'{c}/{fn}')
            except Exception as e:
                corrupt.append((c, fn, str(e)))
                rows.append((c, fn, -1, -1, 'NA', f'CORRUPT: {e}'))

    # --- near-duplicate TRONG tung lop (nghi mau vat lap) ---
    from scipy.spatial.distance import pdist, squareform
    near_dups = []
    for c in classes:
        items = hashes[c]
        if len(items) < 2:
            continue
        H = np.stack([b for _, b in items])
        d = squareform(pdist(H, metric='hamming') * H.shape[1])  # ve don vi bit
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                if d[i, j] <= args.dup_threshold:
                    near_dups.append((c, items[i][0], items[j][0], int(round(d[i, j]))))

    # --- trung y het XUYEN lop (dang lo, kiem tra nhan) ---
    cross_exact = {k: v for k, v in exact.items()
                   if len({x.split('/')[0] for x in v}) > 1}

    total = sum(per_class.values())
    counts = {c: per_class[c] for c in classes}
    mn = min(counts, key=counts.get) if counts else None
    mx = max(counts, key=counts.get) if counts else None

    with open(os.path.join(args.out_dir, 'audit_per_image.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['class', 'file', 'width', 'height', 'mode', 'status'])
        w.writerows(rows)
    with open(os.path.join(args.out_dir, 'near_duplicates_within_class.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['class', 'file_a', 'file_b', 'hamming'])
        w.writerows(near_dups)

    summary = {
        'num_classes': len(classes),
        'total_images': total,
        'min_class': [mn, counts[mn]] if mn else None,
        'max_class': [mx, counts[mx]] if mx else None,
        'imbalance_ratio_max_over_min': round(counts[mx] / max(counts[mn], 1), 2) if mn else None,
        'num_corrupt': len(corrupt),
        'num_distinct_resolutions': len(res_counter),
        'top_resolutions': sorted(res_counter.items(), key=lambda x: -x[1])[:10],
        'near_duplicate_pairs_within_class': len(near_dups),
        'exact_cross_class_duplicate_groups': len(cross_exact),
        'per_class_counts': counts,
    }
    with open(os.path.join(args.out_dir, 'audit_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in summary.items() if k != 'per_class_counts'},
                     indent=2, ensure_ascii=False))
    if corrupt:
        print(f'\n[!] {len(corrupt)} file hong -> xem audit_per_image.csv')
    if near_dups:
        print(f'[!] {len(near_dups)} cap anh GAN TRUNG trong lop -> nghi cung mau vat. '
              f'CAN chia theo specimen o make_splits.py de tranh leakage.')
    if cross_exact:
        print(f'[!!] {len(cross_exact)} nhom anh TRUNG Y HET xuyen lop -> kiem tra nhan!')


if __name__ == '__main__':
    main()
