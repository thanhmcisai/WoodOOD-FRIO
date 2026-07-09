#!/usr/bin/env python3
"""Pha 0 - Sinh khung taxonomy.csv tu ten thu muc lop.

Dien san nhung gi suy duoc tu bai bao (softwood, mot so genus); phan con lai de
TODO cho ban hoan thien. taxonomy.csv la dau vao bat buoc cho make_splits.py va
cho ca phan sinh du lieu taxonomy-guided (Pha 2).

    python build_taxonomy_template.py --data_root /path/to/Wood_ID_1-50 --out_csv taxonomy.csv
"""
import argparse, os, csv, re, unicodedata

# 15 ho da neu trong bai (tu vung hop le de dien cot family)
FAMILIES = ['Cupressaceae', 'Ebenaceae', 'Euphorbiaceae', 'Fabaceae', 'Fagaceae',
            'Juglandaceae', 'Lauraceae', 'Magnoliaceae', 'Malvaceae', 'Meliaceae',
            'Moraceae', 'Oleaceae', 'Rubiaceae', 'Sapotaceae', 'Vochysiaceae']

# 4 loai softwood da neu trong bai
SOFTWOODS = {'calocedrus', 'callitris columellaris',
             'fokienia hodginsii', 'cunninghamia lanceolata'}

# Anh xa best-effort ten thuong goi (VN) -> ten khoa hoc, suy tu bai + chu thich Fig.
# BAN BO SUNG TIEP cho du 50 lop.
COMMON2SCI = {
    'go dousse': 'Afzelia africana', 'go pachy': 'Afzelia pachyloba',
    'go quanze': 'Afzelia quanzensis', 'go quanzensis': 'Afzelia quanzensis',
    'cam lai': 'Dalbergia oliveri', 'trac': 'Dalbergia cochinchinensis',
    'fokienia': 'Fokienia hodginsii', 'juglans': 'Juglans regia',
}


def norm(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r'[^a-z ]', ' ', s.lower()).strip()


def looks_binomial(s):
    parts = s.replace('_', ' ').split()
    return len(parts) >= 2 and parts[0][:1].isupper() and parts[0].isalpha()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_root', required=True)
    ap.add_argument('--out_csv', default='taxonomy.csv')
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(args.data_root)
                     if os.path.isdir(os.path.join(args.data_root, d)))

    rows = []
    for c in classes:
        key = norm(c)
        sci = ''
        if looks_binomial(c):
            sci = c.replace('_', ' ')
        elif key in COMMON2SCI:
            sci = COMMON2SCI[key]
        genus = sci.split()[0] if sci else 'TODO'
        wood = 'softwood' if norm(sci) in SOFTWOODS else 'TODO'
        rows.append([c, sci or 'TODO', genus, 'TODO', wood])

    with open(args.out_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['class_name', 'scientific_name', 'genus', 'family', 'wood_type'])
        w.writerows(rows)

    filled = sum(1 for r in rows if r[1] != 'TODO')
    print(f'Da ghi {args.out_csv}: {len(rows)} lop; suy duoc scientific_name cho {filled} lop.')
    print('HAY dien not cac o TODO (genus, family, wood_type).')
    print('  - genus: tu dau tien cua ten khoa hoc (vd Afzelia).')
    print('  - family: 1 trong', len(FAMILIES), 'ho:', ', '.join(FAMILIES))
    print('  - wood_type: softwood/hardwood. Softwood da biet:', ', '.join(sorted(SOFTWOODS)))


if __name__ == '__main__':
    main()
