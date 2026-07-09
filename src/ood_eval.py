"""Pha 4/5 — danh gia OOD detection tren anh THAT (test_id vs test_ood near/far).
Scorers baseline: MSP, MaxLogit, Energy, KNN, Mahalanobis, IsoMax+. evaluate_eool: cham diem model EOIL.
Metric: FPR@95TPR, AUROC, AUPR (ID = positive). CITE: MSP(Hendrycks17), Energy(Liu20), KNN(Sun22), Maha(Lee18), IsoMax+(Macedo21)."""
import numpy as np
import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader
from . import utils, data as D, engine as E, models as M


@torch.no_grad()
def _extract(model, loader, dev, dt):
    model.eval(); Ls=[]; Fs=[]; Ys=[]
    for xb, yb in loader:
        xb = xb.to(dev, non_blocking=True)
        with torch.autocast('cuda', dtype=dt):
            z = model.forward_features(xb); lo = model.head(z)
        Ls.append(lo.float().cpu()); Fs.append(z.float().cpu()); Ys.append(yb.clone())
    return torch.cat(Ls).numpy(), torch.cat(Fs).numpy().astype(np.float32), torch.cat(Ys).numpy()


def _loader(cfg, split, role, ood_type=None, c2i=None, bs=128):
    df = D.read_manifest(cfg, split); sub = df[df.role == role]
    if ood_type is not None:
        sub = sub[sub.ood_type == ood_type]
    ds = D.WoodDataset(sub.reset_index(drop=True), utils.data_root(cfg), c2i or {}, D.build_transforms(cfg, False))
    return DataLoader(ds, batch_size=bs, num_workers=cfg['hardware']['num_workers'], pin_memory=True), len(ds)


def _post_scores(logits, feat, ref_feat, ref_lab, n_cls, is_isomax):
    lg = torch.tensor(logits); p = F.softmax(lg, 1).numpy()
    mx = logits.max(1, keepdims=True)
    out = {'MSP': p.max(1), 'MaxLogit': logits.max(1),
           'Energy': (np.log(np.exp(logits - mx).sum(1)) + mx[:, 0])}
    if is_isomax:
        out['IsoMax+'] = logits.max(1)
    fq = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)
    fr = ref_feat / (np.linalg.norm(ref_feat, axis=1, keepdims=True) + 1e-8)
    out['KNN'] = np.sort(fq @ fr.T, 1)[:, -50]
    mu = np.stack([ref_feat[ref_lab == c].mean(0) for c in range(n_cls)]).astype(np.float64)
    cen = (ref_feat - mu[ref_lab]).astype(np.float64)
    cov = np.cov(cen.T) + 1e-3 * np.eye(ref_feat.shape[1])
    w, V = np.linalg.eigh(cov)
    Winv = (V * (1.0 / np.sqrt(np.clip(w, 1e-8, None)))) @ V.T
    qw = feat.astype(np.float64) @ Winv; muw = mu @ Winv
    d2 = (qw**2).sum(1, keepdims=True) + (muw**2).sum(1)[None] - 2 * qw @ muw.T
    out['Mahalanobis'] = -d2.min(1)
    return out


def ood_metrics(s_id, s_ood):
    y = np.r_[np.ones(len(s_id)), np.zeros(len(s_ood))]; s = np.r_[s_id, s_ood]
    auroc = roc_auc_score(y, s); aupr = average_precision_score(y, s)
    thr = np.percentile(s_id, 5)
    return round(float((s_ood >= thr).mean()) * 100, 2), round(auroc * 100, 2), round(aupr * 100, 2)


def _score_all(cfg, split, model, n_cls, is_iso, backbone, method_prefix, seed, ref_n=4000, dump=None):
    dev = utils.pick_device(cfg); dt = utils.amp_dtype(cfg)
    id_classes = D.id_classes(D.read_manifest(cfg, split)); c2i = {c: i for i, c in enumerate(id_classes)}
    tr_ld, _ = _loader(cfg, split, 'train', c2i=c2i)
    _, tr_ft, tr_lab = _extract(model, tr_ld, dev, dt)
    idx = np.random.default_rng(0).permutation(len(tr_ft))[:ref_n]
    ref_ft, ref_lab = tr_ft[idx], tr_lab[idx]
    lid, fid_, _ = _extract(model, _loader(cfg, split, 'test_id', c2i=c2i)[0], dev, dt)
    sc_id = _post_scores(lid, fid_, ref_ft, ref_lab, n_cls, is_iso)
    rows = []; dumped = []
    for ot in ['near', 'far']:
        ld, ln = _loader(cfg, split, 'test_ood', ot, c2i=c2i)
        if ln == 0:
            continue
        lo, fo, _ = _extract(model, ld, dev, dt)
        sc_o = _post_scores(lo, fo, ref_ft, ref_lab, n_cls, is_iso)
        for m in sc_id:
            fpr, au, ap = ood_metrics(sc_id[m], sc_o[m])
            rows.append({'scenario': split, 'backbone': backbone, 'method': method_prefix + m,
                         'ood_type': ot, 'fpr95': fpr, 'auroc': au, 'aupr': ap, 'seed': seed})
        if dump is not None:
            for v in sc_o[dump]:
                dumped.append({'score': float(v), 'is_id': 0, 'ood_type': ot})
    if dump is not None:
        for v in sc_id[dump]:
            dumped.append({'score': float(v), 'is_id': 1, 'ood_type': 'id'})
        return rows, dumped
    return rows


def evaluate(cfg, split, backbone, loss, seed=0, ref_n=4000):
    model, n_cls = E.load_best(cfg, backbone, loss, split, seed)
    rows = _score_all(cfg, split, model, n_cls, loss == 'isomax_plus', backbone, loss + '/', seed, ref_n)
    del model; torch.cuda.empty_cache()
    return rows


def evaluate_eool(cfg, split, backbone, seed=0, ref_n=4000, ckpt_name=None, dump=None):
    dev = utils.pick_device(cfg)
    cn = ckpt_name or f'eool_{backbone}_{split}_seed{seed}.pt'
    ck = torch.load(utils.rp(cfg, cfg['paths']['checkpoints'], cn), map_location=dev)
    n_cls = int(ck['n_cls'])
    model = M.build_model(cfg, backbone, n_cls, 'isomax_plus').to(dev)
    model.load_state_dict(ck['model']); model.eval()
    res = _score_all(cfg, split, model, n_cls, True, backbone, 'eool/', seed, ref_n, dump=dump)
    del model; torch.cuda.empty_cache()
    return res
