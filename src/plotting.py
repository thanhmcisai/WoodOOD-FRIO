"""Pha 7 — sinh HINH tu results/ (KHONG hardcode). Style: PDF+PNG 300dpi, mau mu-mau."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score
from . import utils

plt.rcParams.update({'font.size': 11, 'axes.grid': True, 'grid.alpha': .3,
                     'figure.dpi': 110, 'savefig.bbox': 'tight'})
CB = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7', '#56B4E9', '#F0E442', '#999999']


def _save(fig, cfg, name):
    d = utils.rp(cfg, cfg['paths']['figures']); utils.ensure_dir(d)
    fig.savefig(os.path.join(d, name + '.png'), dpi=300); fig.savefig(os.path.join(d, name + '.pdf'))
    plt.close(fig)


def fig_ood_comparison(cfg, split):
    df = pd.read_csv(utils.rp(cfg, cfg['paths']['results'], f'ood_{split}.csv'))
    df['fam'] = df.method.str.split('/').str[0]; df['scorer'] = df.method.str.split('/').str[-1]
    rows = [('ce', 'MSP', 'MSP'), ('ce', 'Energy', 'Energy'), ('ce', 'Mahalanobis', 'Maha'),
            ('ce', 'KNN', 'KNN'), ('isomax_plus', 'IsoMax+', 'IsoMax+'), ('eool', 'MaxLogit', 'EOIL')]
    labels = [r[2] for r in rows]; x = np.arange(len(rows)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.2))
    for i, ot in enumerate(['near', 'far']):
        mu = [df[(df.fam == f) & (df.scorer == s) & (df.ood_type == ot)]['auroc'].mean() for f, s, _ in rows]
        sd = [df[(df.fam == f) & (df.scorer == s) & (df.ood_type == ot)]['auroc'].std(ddof=0) for f, s, _ in rows]
        ax.bar(x + (i - .5) * w, mu, w, yerr=sd, capsize=3, label=f'{ot}-OOD', color=CB[i])
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel('AUROC (%)'); ax.set_ylim(60, 100)
    ax.set_title(f'OOD detection — {split.replace("_", "/")}'); ax.legend()
    _save(fig, cfg, f'ood_comparison_{split}'); return 'ood_comparison_' + split


def fig_synth_quality(cfg):
    df = pd.read_csv(utils.rp(cfg, cfg['paths']['results'], 'synth_quality.csv'))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.8)); x = np.arange(len(df))
    a1.bar(x, df['fid'], color=CB[0]); a1.set_xticks(x); a1.set_xticklabels(df['scenario'])
    a1.set_ylabel('FID (gen vs real)'); a1.set_title('Chat luong anh sinh')
    w = .25
    for i, (c, lb) in enumerate([('conf_gen_near', 'near'), ('conf_gen_far', 'far'), ('conf_real_id', 'real ID')]):
        a2.bar(x + (i - 1) * w, df[c], w, label=lb, color=CB[i])
    a2.set_xticks(x); a2.set_xticklabels(df['scenario']); a2.set_ylabel('ID conf'); a2.set_title('Sanity outlier'); a2.legend(fontsize=8)
    _save(fig, cfg, 'synth_quality'); return 'synth_quality'


def fig_data_statistics(cfg):
    import glob
    droot = utils.data_root(cfg)
    counts = [len(glob.glob(os.path.join(d, '*'))) for d in sorted(glob.glob(os.path.join(droot, '*'))) if os.path.isdir(d)]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(np.arange(len(counts)), sorted(counts, reverse=True), color=CB[2])
    ax.set_xlabel('Lop (giam dan)'); ax.set_ylabel('So anh'); ax.set_title(f'Phan bo anh/lop (N={len(counts)})')
    _save(fig, cfg, 'data_statistics'); return 'data_statistics'


def fig_score_distribution(cfg, split):
    df = pd.read_csv(utils.rp(cfg, cfg['paths']['results'], f'scores_{split}.csv'))
    fig, ax = plt.subplots(figsize=(7, 4))
    groups = [('id', 'ID (in)', CB[2]), ('near', 'near-OOD', CB[1]), ('far', 'far-OOD', CB[3])]
    for key, lb, col in groups:
        s = df[df.ood_type == key]['score'] if key != 'id' else df[df.is_id == 1]['score']
        if len(s):
            ax.hist(s, bins=50, density=True, alpha=.55, label=lb, color=col)
    ax.set_xlabel('MaxLogit score (cao = in-distribution)'); ax.set_ylabel('Mat do')
    ax.set_title(f'Phan bo score in vs out — {split.replace("_", "/")}'); ax.legend()
    _save(fig, cfg, f'score_distribution_{split}'); return 'score_distribution_' + split


def fig_roc_pr(cfg, split):
    df = pd.read_csv(utils.rp(cfg, cfg['paths']['results'], f'scores_{split}.csv'))
    sid = df[df.is_id == 1]['score'].values
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 4))
    for i, ot in enumerate(['near', 'far']):
        so = df[df.ood_type == ot]['score'].values
        if not len(so): continue
        y = np.r_[np.ones(len(sid)), np.zeros(len(so))]; s = np.r_[sid, so]
        fpr, tpr, _ = roc_curve(y, s); pr, rc, _ = precision_recall_curve(y, s)
        au = roc_auc_score(y, s)
        a1.plot(fpr, tpr, color=CB[i], label=f'{ot} (AUROC {au*100:.1f})')
        a2.plot(rc, pr, color=CB[i], label=f'{ot}')
    a1.plot([0, 1], [0, 1], 'k--', alpha=.3); a1.set_xlabel('FPR'); a1.set_ylabel('TPR'); a1.set_title('ROC'); a1.legend()
    a2.set_xlabel('Recall'); a2.set_ylabel('Precision'); a2.set_title('PR'); a2.legend()
    fig.suptitle(f'ROC / PR — {split.replace("_", "/")}')
    _save(fig, cfg, f'roc_pr_{split}'); return 'roc_pr_' + split


def fig_ablation(cfg):
    p = utils.rp(cfg, cfg['paths']['results'], 'ablation_sampling.csv')
    if not os.path.exists(p): return None
    df = pd.read_csv(p); df = df[df.scorer == 'MaxLogit']
    samps = [s for s in ['random', 'taxonomy_matched', 'taxonomy'] if s in df.sampling.unique()]
    conds = [(sp, ot) for sp in ['40_10', '30_20'] for ot in ['near', 'far']]
    x = np.arange(len(conds)); w = .8 / len(samps)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, s in enumerate(samps):
        vals = [df[(df.sampling == s) & (df.split == sp) & (df.ood_type == ot)]['auroc'].mean() for sp, ot in conds]
        ax.bar(x + (i - (len(samps)-1)/2) * w, vals, w, label=s, color=CB[i])
    ax.set_xticks(x); ax.set_xticklabels([f'{sp}\n{ot}' for sp, ot in conds]); ax.set_ylabel('AUROC (%)')
    ax.set_ylim(85, 100); ax.set_title('Ablation: taxonomy-guided vs random sampling (MaxLogit)'); ax.legend()
    _save(fig, cfg, 'ablation_sampling'); return 'ablation_sampling'


def generate_all(cfg):
    figs = []
    for split in ['40_10', '30_20']:
        figs.append(fig_ood_comparison(cfg, split))
    figs.append(fig_synth_quality(cfg))
    for fn in [lambda: fig_data_statistics(cfg), lambda: fig_ablation(cfg)]:
        try: figs.append(fn())
        except Exception as e: print('skip:', e)
    for split in ['40_10', '30_20']:
        for fn in [fig_score_distribution, fig_roc_pr]:
            try: figs.append(fn(cfg, split))
            except Exception as e: print('skip', fn.__name__, e)
    return [f for f in figs if f]
