"""Train loop ID classification — resumable, checkpoint, early-stop tren val macro-F1.
Checkpoint: resume (model+opt+sched+meta) local moi epoch + sync Drive dinh ky; best = model-only tren Drive.
AMP bf16 (L4); cosine annealing; Adam. Tham so tu config (khong hardcode). lr: uu tien per-backbone."""
import os, time, shutil
import numpy as np
import torch
import torch.nn as nn
from . import utils, data as D, models as M, metrics as Me
from .isomax import IsoMaxPlusLossSecondPart

LOCAL_CK = '/content/ck'


def _tag(backbone, loss, split, seed):
    return f'{backbone}_{loss}_{split}_seed{seed}'


def _make_criterion(cfg, loss):
    if loss == 'ce':
        return nn.CrossEntropyLoss()
    return IsoMaxPlusLossSecondPart(entropic_scale=cfg['isomax']['entropic_scale'])


def train_model(cfg, backbone, loss, split, seed, log=print):
    utils.set_seed(seed)
    dev = utils.pick_device(cfg); dt = utils.amp_dtype(cfg)
    it = cfg['id_training']
    bbc = it['backbones'][backbone]
    max_ep = int(it.get('max_epochs', it['epochs']))
    patience_lim = int(it.get('early_stop_patience', max_ep))
    sync_every = int(it.get('ckpt_sync_every', 5))
    bs = bbc['batch']
    lr = float(bbc.get('lr', it['lr']))            # uu tien lr per-backbone (vd ConvNeXt=2e-4)

    L = D.make_loaders(cfg, split, seed, batch_size=bs, roles=('train', 'val_id'))
    n_cls = L['num_classes']
    model = M.build_model(cfg, backbone, n_cls, loss).to(dev)
    crit = _make_criterion(cfg, loss)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    scaler = torch.cuda.amp.GradScaler(enabled=(dt == torch.float16))

    tag = _tag(backbone, loss, split, seed)
    utils.ensure_dir(LOCAL_CK); utils.ensure_dir(utils.rp(cfg, cfg['paths']['checkpoints']))
    ck_local = os.path.join(LOCAL_CK, tag + '.pt')
    ck_drive = utils.rp(cfg, cfg['paths']['checkpoints'], tag + '.pt')          # resume
    ck_best = utils.rp(cfg, cfg['paths']['checkpoints'], tag + '_best.pt')       # model-only
    logf = utils.rp(cfg, cfg['paths']['logs'], 'id_train.log')

    start_ep, best_f1, best_ep, patience = 0, -1.0, -1, 0
    src = ck_local if os.path.exists(ck_local) else (ck_drive if os.path.exists(ck_drive) else None)
    if src:
        ck = torch.load(src, map_location=dev)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched'])
        start_ep = ck['epoch'] + 1; best_f1 = ck['best_f1']; best_ep = ck['best_ep']; patience = ck['patience']
        if ck.get('done'):
            log(f'[{tag}] da DONE (best macro-F1={best_f1:.4f} @ep{best_ep}) -> bo qua')
            return {'tag': tag, 'best_f1': best_f1, 'best_ep': best_ep, 'epochs_run': ck['epoch'] + 1,
                    'best_ckpt': ck_best, 'num_classes': n_cls, 'resumed': True, 'skipped': True}
        log(f'[{tag}] RESUME tu epoch {start_ep} (best={best_f1:.4f})')

    def save_resume(ep, done=False):
        obj = {'model': model.state_dict(), 'opt': opt.state_dict(), 'sched': sched.state_dict(),
               'epoch': ep, 'best_f1': best_f1, 'best_ep': best_ep, 'patience': patience,
               'done': done, 'cfg_tag': tag}
        torch.save(obj, ck_local)                                   # nhanh (local) moi epoch
        if done or (ep % sync_every == 0):
            torch.save(obj, ck_drive)                               # sync Drive dinh ky/cuoi

    for ep in range(start_ep, max_ep):
        model.train(); t0 = time.time(); run = 0.0; nseen = 0
        for xb, yb in L['train']:
            xb = xb.to(dev, non_blocking=True); yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dt):
                logits = model(xb); lval = crit(logits, yb)
            if scaler.is_enabled():
                scaler.scale(lval).backward(); scaler.step(opt); scaler.update()
            else:
                lval.backward(); opt.step()
            run += lval.item() * len(xb); nseen += len(xb)
        sched.step()
        yt, yp = Me.predict(model, L['val_id'], dev, dt)
        vf1 = Me.classification_metrics(yt, yp)['macro_f1']
        improved = vf1 > best_f1
        if improved:
            best_f1, best_ep, patience = vf1, ep, 0
            torch.save({'model': model.state_dict(), 'num_classes': n_cls,
                        'backbone': backbone, 'loss': loss}, ck_best)
        else:
            patience += 1
        save_resume(ep, done=False)
        utils.log(f'[{tag}] ep{ep+1}/{max_ep} lr={lr:g} loss={run/max(nseen,1):.4f} val_f1={vf1:.4f} '
                  f'best={best_f1:.4f}@{best_ep+1} pat={patience} ({time.time()-t0:.0f}s)', logf)
        if patience >= patience_lim:
            utils.log(f'[{tag}] EARLY-STOP @ep{ep+1} (patience {patience_lim})', logf)
            break
    save_resume(ep, done=True)
    return {'tag': tag, 'best_f1': best_f1, 'best_ep': best_ep, 'epochs_run': ep + 1,
            'best_ckpt': ck_best, 'num_classes': n_cls, 'resumed': bool(src), 'skipped': False}


def load_best(cfg, backbone, loss, split, seed):
    """Nap best checkpoint -> model eval-ready + n_cls."""
    dev = utils.pick_device(cfg)
    ck_best = utils.rp(cfg, cfg['paths']['checkpoints'], _tag(backbone, loss, split, seed) + '_best.pt')
    ck = torch.load(ck_best, map_location=dev)
    model = M.build_model(cfg, backbone, ck['num_classes'], loss).to(dev)
    model.load_state_dict(ck['model']); model.eval()
    return model, ck['num_classes']
