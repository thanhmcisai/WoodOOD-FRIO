#!/usr/bin/env python3
"""Pha 0 (v2) - Sinh split ID/OOD CO KIEM SOAT tu VN50_metadata.csv.

Dung cot da curate:
  - ood_role_suggestion: 'ID-core' -> luon ID; 'far-OOD...' -> pool far; 'near-OOD pool...' -> pool near
  - benchmark_flag chua 'UNRESOLVED-SPECIES' (Pterocarpus sp.) -> EP vao ID (khong lam OOD)
    (tat bang --allow_unresolved_ood neu that su muon)
Rang buoc: khi dua 1 loai near ra OOD, GIU >=1 loai cung chi trong ID (near-by-genus hop le).
Near/Far cuoi cung KIEM TRA LAI theo taxonomy: family khong co trong ID -> far; nguoc lai -> near.

Che do:
  - Metadata-only (khong --data_root): chi xuat class_assignment.csv (xem truoc phan lop).
  - Co --data_root: xuat them manifest.csv (chia anh ID cap specimen qua --specimen_regex).

    python make_splits.py --meta_csv VN50_metadata.csv --out_dir results/splits \\
        --scenarios 40/10,30/20 --seed 0 [--data_root data/Wood_ID_1-50 --specimen_regex '(?P<sid>.+?)_\\d+']
"""
import argparse, os, csv, re, json, random
from collections import defaultdict, OrderedDict

IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_meta(path):
    meta = OrderedDict()
    with open(path, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            cls = r['folder_name'].strip()
            role = (r.get('ood_role_suggestion') or '').strip().lower()
            flag = (r.get('benchmark_flag') or '').strip().upper()
            if role.startswith('far'):
                pool = 'far'
            elif role.startswith('near'):
                pool = 'near'
            else:
                pool = 'id_core'
            n = r.get('folder_image_count', '').strip()
            meta[cls] = {
                'genus': r['genus'].strip(),
                'family': (r.get('family_normalized') or r.get('family', '')).strip(),
                'pool': pool, 'flag': flag,
                'conf': (r.get('match_confidence') or '').strip().lower(),
                'n_img': int(n) if n.isdigit() else 0,
            }
    return meta


def pick_ood(meta, n_ood, seed, allow_low_confidence_ood):
    classes = list(meta)
    n_far_t = max(1, n_ood // 2)
    n_near_t = n_ood - n_far_t
    rng = random.Random(seed + n_ood)

    # Khoa ve ID: lop chua chac loai (UNRESOLVED) HOAC do tin cay dinh danh thap (med/low,
    # vd Pterocarpus erinaceus/indicus suy luan) -> tranh lam "OOD" nham (nguy co trung an).
    forced_id = set()
    if not allow_low_confidence_ood:
        forced_id = {c for c in classes
                     if 'UNRESOLVED-SPECIES' in meta[c]['flag']
                     or meta[c].get('conf') in ('med', 'medium', 'low')}
    far_pool = [c for c in classes if meta[c]['pool'] == 'far' and c not in forced_id]
    near_pool = [c for c in classes if meta[c]['pool'] == 'near' and c not in forced_id]
    rng.shuffle(far_pool); rng.shuffle(near_pool)

    by_genus = defaultdict(list)
    for c in classes:
        by_genus[meta[c]['genus']].append(c)
    taken = defaultdict(int)      # so loai da lay ra OOD moi chi

    ood = []
    # FAR truoc
    for c in far_pool:
        if sum(1 for x in ood if meta[x]['pool'] == 'far') >= n_far_t:
            break
        ood.append(c)

    def near_count():
        return sum(1 for x in ood if meta[x]['pool'] == 'near')

    # NEAR: giu >=1 loai cung chi trong ID
    for c in near_pool:
        if near_count() >= n_near_t:
            break
        g = meta[c]['genus']
        if taken[g] < len(by_genus[g]) - 1:     # con it nhat 1 loai o ID
            ood.append(c); taken[g] += 1

    # Bu neu con thieu (giu rang buoc near; far khong rang buoc)
    for c in near_pool:
        if len(ood) >= n_ood:
            break
        g = meta[c]['genus']
        if c not in ood and taken[g] < len(by_genus[g]) - 1:
            ood.append(c); taken[g] += 1
    for c in far_pool:
        if len(ood) >= n_ood:
            break
        if c not in ood:
            ood.append(c)

    ood = set(ood[:n_ood])
    id_set = [c for c in classes if c not in ood]
    return id_set, ood, sorted(forced_id)


def label_nf(ood, id_set, meta):
    id_f = {meta[c]['family'] for c in id_set}
    return {c: ('far' if meta[c]['family'] not in id_f else 'near') for c in ood}


def list_images(cdir):
    return [fn for fn in sorted(os.listdir(cdir))
            if os.path.splitext(fn)[1].lower() in IMG_EXT]


def specimen_of(fn, p):
    if p is None:
        return fn
    m = p.search(fn)
    return m.group('sid') if m else fn


def split_id_images(imgs, key_of, vf, tf, rng):
    groups = defaultdict(list)
    for fn in imgs:
        groups[key_of(fn)].append(fn)
    keys = list(groups); rng.shuffle(keys); n = len(keys)
    nt = max(1, round(tf * n)) if n > 1 else 0
    nv = max(1, round(vf * n)) if n - nt > 1 else 0
    tk, vk = set(keys[:nt]), set(keys[nt:nt + nv])
    out = {'train': [], 'val_id': [], 'test_id': []}
    for k in keys:
        role = 'test_id' if k in tk else ('val_id' if k in vk else 'train')
        out[role].extend(groups[k])
    return out


def load_pseudo_map(path):
    m = {}
    with open(path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            m[r['image_path'].strip()] = r['pseudo_specimen_id'].strip()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_csv', required=True)
    ap.add_argument('--out_dir', default='results/splits')
    ap.add_argument('--scenarios', default='40/10,30/20')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--data_root', default=None, help='neu co -> xuat manifest anh')
    ap.add_argument('--specimen_regex', default=None)
    ap.add_argument('--pseudo_specimen_map', default=None,
                    help='CSV tu cluster_pseudo_specimen.py (image_path,class,pseudo_specimen_id). '
                         'Uu tien hon --specimen_regex khi chia anh ID.')
    ap.add_argument('--val_frac', type=float, default=0.1)
    ap.add_argument('--test_id_frac', type=float, default=0.1)
    ap.add_argument('--allow_low_confidence_ood', action='store_true',
                    help='cho phep dua lop do tin cay thap (med/low) hoac UNRESOLVED vao OOD (mac dinh KHONG -> giu o ID)')
    a = ap.parse_args()

    meta = load_meta(a.meta_csv)
    pat = re.compile(a.specimen_regex) if a.specimen_regex else None
    pseudo = load_pseudo_map(a.pseudo_specimen_map) if a.pseudo_specimen_map else None
    if a.data_root:
        if pseudo is not None:
            print(f'[i] Chia anh ID theo PSEUDO-SPECIMEN ({len(pseudo)} anh trong map).')
        elif pat is not None:
            print('[i] Chia anh ID theo specimen_regex.')
        else:
            print('[!] Khong co pseudo_specimen_map / specimen_regex => chia CAP ANH (nguy co leakage). '
                  'Nen chay cluster_pseudo_specimen.py truoc.')
    os.makedirs(a.out_dir, exist_ok=True)

    summary = {}
    for scen in a.scenarios.split(','):
        n_id, n_ood = map(int, scen.split('/'))
        id_set, ood, forced = pick_ood(meta, n_ood, a.seed, a.allow_low_confidence_ood)
        nf = label_nf(ood, id_set, meta)

        sdir = os.path.join(a.out_dir, scen.replace('/', '_'))
        os.makedirs(sdir, exist_ok=True)
        with open(os.path.join(sdir, 'class_assignment.csv'), 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['class', 'role', 'ood_type', 'genus', 'family', 'pool', 'flag', 'n_img'])
            for c in id_set:
                w.writerow([c, 'ID', 'NA', meta[c]['genus'], meta[c]['family'],
                            meta[c]['pool'], meta[c]['flag'], meta[c]['n_img']])
            for c in sorted(ood):
                w.writerow([c, 'OOD', nf[c], meta[c]['genus'], meta[c]['family'],
                            meta[c]['pool'], meta[c]['flag'], meta[c]['n_img']])

        n_near = sum(1 for v in nf.values() if v == 'near')
        n_far = sum(1 for v in nf.values() if v == 'far')
        entry = {'n_id': len(id_set), 'n_ood': len(ood), 'ood_near': n_near, 'ood_far': n_far,
                 'forced_id_unresolved': forced,
                 'ood_classes': {c: nf[c] for c in sorted(ood)}}

        if a.data_root:
            rng = random.Random(a.seed + n_ood)
            manifest = []
            for c in id_set:
                cdir = os.path.join(a.data_root, c)
                if not os.path.isdir(cdir):
                    continue
                if pseudo is not None:
                    def key_of(fn, cls=c):
                        return pseudo.get(f'{cls}/{fn}', f'{cls}/{fn}')
                elif pat is not None:
                    def key_of(fn):
                        return specimen_of(fn, pat)
                else:
                    def key_of(fn):
                        return fn
                parts = split_id_images(list_images(cdir), key_of, a.val_frac, a.test_id_frac, rng)
                for role, fns in parts.items():
                    for fn in fns:
                        manifest.append([os.path.join(c, fn), c, role, 'NA'])
            for c in ood:
                cdir = os.path.join(a.data_root, c)
                if not os.path.isdir(cdir):
                    continue
                for fn in list_images(cdir):
                    manifest.append([os.path.join(c, fn), c, 'test_ood', nf[c]])
            with open(os.path.join(sdir, 'manifest.csv'), 'w', newline='', encoding='utf-8') as f:
                w = csv.writer(f)
                w.writerow(['image_path', 'class', 'role', 'ood_type'])
                w.writerows(manifest)
            roles = defaultdict(int)
            for _, _, r, _ in manifest:
                roles[r] += 1
            entry['image_roles'] = dict(roles)

        summary[scen] = entry
        tail = f" | anh: {entry.get('image_roles')}" if a.data_root else " | (metadata-only)"
        print(f"[{scen}] ID={len(id_set)} OOD={len(ood)} (near={n_near}, far={n_far}){tail}")
        if forced:
            print(f"    forced ID (do tin cay thap / chua chac loai): {forced}")

    with open(os.path.join(a.out_dir, 'splits_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    main()
