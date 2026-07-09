"""Metric phan loai ID + do thoi gian suy luan. macro-F1/balanced-acc qua sklearn (tinh DUNG)."""
import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'macro_f1': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'balanced_acc': float(balanced_accuracy_score(y_true, y_pred)),
    }


@torch.no_grad()
def predict(model, loader, device, amp_dtype=None):
    """Tra ve (y_true, y_pred) tren loader (argmax logits — dung cho ca CE lan IsoMax+)."""
    model.eval()
    yt, yp = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        if amp_dtype is not None and device.type == 'cuda':
            with torch.autocast('cuda', dtype=amp_dtype):
                logits = model(xb)
        else:
            logits = model(xb)
        yp.append(logits.float().argmax(1).cpu().numpy())
        yt.append(np.asarray(yb))
    return np.concatenate(yt), np.concatenate(yp)


@torch.no_grad()
def inference_time_ms(model, device, image_size=224, n_warmup=10, n_iter=50):
    """ms/anh (batch=1) tren `device`. GPU: co torch.cuda.synchronize."""
    model.eval().to(device)
    x = torch.randn(1, 3, image_size, image_size, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iter):
        model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000.0
