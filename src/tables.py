"""Pha 6 — sinh bang LaTeX TU results/ (KHONG hardcode so lieu). CLAUDE.md muc 5.
table1: phan loai ID; table2/3: OOD 40_10 / 30_20 (mean+-std tren backbone, near/far, FPR95/AUROC/AUPR)."""
import numpy as np
import pandas as pd
from . import utils


def _fmt_cell(m, s, best):
    if np.isnan(m):
        return '--'
    txt = f'{m:.1f}$\\pm${s:.1f}'
    return f'\\textbf{{{txt}}}' if best else txt


def ood_table(cfg, split):
    df = pd.read_csv(utils.rp(cfg, cfg['paths']['results'], f'ood_{split}.csv'))
    df['fam'] = df.method.str.split('/').str[0]; df['scorer'] = df.method.str.split('/').str[-1]
    rows = [('ce', 'MSP', 'MSP'), ('ce', 'MaxLogit', 'MaxLogit'), ('ce', 'Energy', 'Energy'),
            ('ce', 'KNN', 'KNN'), ('ce', 'Mahalanobis', 'Mahalanobis'),
            ('isomax_plus', 'IsoMax+', 'IsoMax+'),
            ('eool', 'MaxLogit', 'EOIL (ours)'), ('eool', 'Energy', 'EOIL-Energy (ours)')]
    metrics = [('fpr95', True), ('auroc', False), ('aupr', False)]
    agg = {}
    for fam, sc, _ in rows:
        for ot in ['near', 'far']:
            sub = df[(df.fam == fam) & (df.scorer == sc) & (df.ood_type == ot)]
            for mt, _lb in metrics:
                agg[(fam, sc, ot, mt)] = (sub[mt].mean(), sub[mt].std(ddof=0)) if len(sub) else (np.nan, np.nan)
    best = {}
    for ot in ['near', 'far']:
        for mt, lb in metrics:
            vals = [(agg[(f, s, ot, mt)][0], (f, s)) for f, s, _ in rows if not np.isnan(agg[(f, s, ot, mt)][0])]
            best[(ot, mt)] = (min if lb else max)(vals)[1] if vals else None
    lines = [r'\begin{tabular}{l' + 'c' * 6 + '}', r'\toprule',
             r' & \multicolumn{3}{c}{Near-OOD} & \multicolumn{3}{c}{Far-OOD} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}',
             r'Method & FPR95$\downarrow$ & AUROC$\uparrow$ & AUPR$\uparrow$ & FPR95$\downarrow$ & AUROC$\uparrow$ & AUPR$\uparrow$ \\',
             r'\midrule']
    for fam, sc, disp in rows:
        cells = []
        for ot in ['near', 'far']:
            for mt, lb in metrics:
                m, s = agg[(fam, sc, ot, mt)]
                cells.append(_fmt_cell(m, s, best[(ot, mt)] == (fam, sc)))
        lines.append(f'{disp} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def id_table(cfg, scenario='40_10'):
    df = pd.read_csv(utils.rp(cfg, cfg['paths']['results'], 'id_classification.csv'))
    df = df[df.scenario == scenario]
    mc = [c for c in ['acc', 'macro_f1', 'balanced_acc'] if c in df.columns]
    has_t = 'infer_gpu_ms' in df.columns
    lines = [r'\begin{tabular}{ll' + 'c' * (len(mc) + (1 if has_t else 0)) + '}', r'\toprule',
             'Backbone & Loss & ' + ' & '.join(c.replace('_', ' ').title() for c in mc)
             + (r' & GPU (ms)' if has_t else '') + r' \\', r'\midrule']
    for (bb, meth), r in df.groupby(['backbone', 'method']):
        cells = [f'{r[c].mean()*100:.1f}$\\pm${r[c].std(ddof=0)*100:.1f}' for c in mc]
        if has_t:
            cells.append(f'{r["infer_gpu_ms"].mean():.1f}')
        lines.append(f'{bb} & {meth} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    return '\n'.join(lines)


def write_all(cfg):
    import os
    out = utils.rp(cfg, cfg['paths']['tables']); utils.ensure_dir(out)
    res = {'table1_id_classification.tex': id_table(cfg),
           'table2_ood_40_10.tex': ood_table(cfg, '40_10'),
           'table3_ood_30_20.tex': ood_table(cfg, '30_20')}
    for fn, tex in res.items():
        with open(os.path.join(out, fn), 'w') as f:
            f.write(tex)
    return list(res.keys())
