#!/usr/bin/env python3
"""Sinh taxonomy.csv tu VN50_metadata.csv (da co san genus/family/science_name).
wood_type = softwood cho Cupressaceae, con lai hardwood.

    python build_taxonomy_from_metadata.py --meta_csv VN50_metadata.csv --out_csv taxonomy.csv
"""
import argparse, csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--meta_csv', required=True)
    ap.add_argument('--out_csv', default='taxonomy.csv')
    a = ap.parse_args()

    rows = []
    with open(a.meta_csv, newline='', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            cls = r['folder_name'].strip()
            genus = r['genus'].strip()
            fam = (r.get('family_normalized') or r.get('family', '')).strip()
            sci = (r.get('species_class') or r.get('science_name', '')).strip()
            wood = 'softwood' if fam.lower() == 'cupressaceae' else 'hardwood'
            rows.append([cls, sci, genus, fam, wood])

    with open(a.out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['class_name', 'scientific_name', 'genus', 'family', 'wood_type'])
        w.writerows(rows)

    soft = sum(1 for x in rows if x[4] == 'softwood')
    n_fam = len({x[3] for x in rows})
    n_gen = len({x[2] for x in rows})
    print(f'Ghi {a.out_csv}: {len(rows)} lop | {soft} softwood | {n_fam} ho | {n_gen} chi.')


if __name__ == '__main__':
    main()
