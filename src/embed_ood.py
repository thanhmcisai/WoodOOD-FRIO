"""METHOD (sua nut that decode): OOD detection O EMBEDDING SPACE (encoder Pha 2b) voi outlier
TAXONOMY-GUIDED. Bypass hoan toan buoc decode->anh (noi tin hieu taxonomy bi mat).
Baselines khong dung outlier: MaxCos (max cosine toi prototype), KNN, Mahalanobis (embedding).
Proposed: Disc-taxonomy (discriminator ID-vs-outlier voi outlier taxonomy) vs Disc-random.
Metric FPR@95/AUROC/AUPR near/far. CITE: DREAM-OOD (outlier), Sun2022(KNN), Lee2018(Maha)."""
import numpy as np
import torch, torch.nn.functional as F
from sklearn.neural_network import MLPClassifier
from . import utils, data as D, latent_space as LS, ood_eval as OE


def load_encoder(cfg, split, seed=0):
    dev = utils.pick_device(cfg)
    ck = torch.load(utils.rp(cfg, cfg['paths']['checkpoints'], f'latent_{split}_seed{seed}.pt'), map_location=dev)
    enc = LS.LatentEncoder(feat_dim=int(cfg['latent_space']['feature_dim'])).to(dev)
    enc.load_state_dict(ck['enc']); enc.eval()
    return enc, ck['id_classes']


@torch.no_grad()
def _emb(enc, loader, dev, dt):
    o = []
    for xb, _ in loader:
        with torch.autocast('cuda', dtype=dt):
            o.append(enc(xb.to(dev)).float().cpu().numpy())
    return np.concatenate(o)


def _take(cfg, npz, n):
    z = np.load(utils.rp(cfg, cfg['paths']['synth'], 'embeddings', npz), allow_pickle=True)
    E = z['embed'].astype(np.float32); ot = np.array(z['otype'])
    return np.concatenate([E[ot == 'near'][:n], E[ot == 'far'][:n]])


def _maha_fns(ref, lab, n_cls):
    mu = np.stack([ref[lab == c].mean(0) for c in range(n_cls)]).astype(np.float64)
    cov = np.cov((ref - mu[lab]).astype(np.float64).T) + 1e-3 * np.eye(ref.shape[1])
    w, V = np.linalg.eigh(cov); Winv = (V * (1/np.sqrt(np.clip(w, 1e-8, None)))) @ V.T
    muw = mu @ Winv
    def score(e):
        qw = e.astype(np.float64) @ Winv
        d2 = (qw**2).sum(1, keepdims=True) + (muw**2).sum(1)[None] - 2*qw@muw.T
        return -d2.min(1)
    return score


def evaluate(cfg, split, n_out=250, disc_seeds=(0, 1, 2)):
    dev = utils.pick_device(cfg); dt = utils.amp_dtype(cfg)
    enc, idc = load_encoder(cfg, split)
    proto = LS.build_text_prototypes(cfg, idc, dev).cpu().numpy()               # (C,768) L2-norm
    z = np.load(utils.rp(cfg, cfg['paths']['synth'], 'embeddings', f'id_feat_{split}.npz'), allow_pickle=True)
    idf = z['feats'].astype(np.float32); idl = z['labels'].astype(int); n_cls = len(idc)
    rng = np.random.default_rng(0); idx = rng.permutation(len(idf))[:3000]
    idsub, idsub_l = idf[idx], idl[idx]
    e_id = _emb(enc, OE._loader(cfg, split, 'test_id')[0], dev, dt)
    e_nr = _emb(enc, OE._loader(cfg, split, 'test_ood', 'near')[0], dev, dt)
    e_fr = _emb(enc, OE._loader(cfg, split, 'test_ood', 'far')[0], dev, dt)
    def norm(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    rows = []
    def add(method, seed, sc):
        sid = sc(e_id)
        for ot, eo in [('near', e_nr), ('far', e_fr)]:
            fpr, au, ap = OE.ood_metrics(sid, sc(eo))
            rows.append({'scenario': split, 'method': method, 'ood_type': ot,
                         'fpr95': fpr, 'auroc': au, 'aupr': ap, 'seed': seed})
    # --- baselines (khong outlier) ---
    add('MaxCos', 0, lambda e: (norm(e) @ proto.T).max(1))
    frn = norm(idsub)
    add('KNN', 0, lambda e: np.sort(norm(e) @ frn.T, 1)[:, -50])
    mfn = _maha_fns(idsub, idsub_l, n_cls); add('Mahalanobis', 0, mfn)
    # --- proposed: discriminator ID vs outlier (taxonomy vs random) ---
    for samp, npz in [('Disc-random', f'outlier_embed_rand2_{split}.npz'),
                      ('Disc-taxonomy', f'outlier_embed_{split}.npz')]:
        out = _take(cfg, npz, n_out)
        X = np.concatenate([idsub, out]); y = np.r_[np.ones(len(idsub)), np.zeros(len(out))]
        for s in disc_seeds:
            clf = MLPClassifier((128,), max_iter=400, random_state=s).fit(X, y)
            add(samp, s, lambda e, c=clf: c.predict_proba(e)[:, 1])
    return rows
