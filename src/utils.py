"""Wood-OOD tien ich chung: config, seed, paths, device, timer.
Nguon chan ly tham so = configs/config.yaml (khong hardcode)."""
import os, random, time, json
import numpy as np
import yaml


def load_config(path='configs/config.yaml'):
    with open(path, encoding='utf-8') as f:
        return yaml.safe_load(f)


def set_seed(seed):
    """Seed torch/numpy/random + cudnn deterministic (tai lap)."""
    import torch
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def data_root(cfg):
    """Uu tien ban sao local (ext4 NFC) neu co, fallback Drive."""
    return cfg['paths'].get('data_local') or cfg['paths']['data']


def rp(cfg, *parts):
    """Duong dan tuyet doi tu paths.root."""
    return os.path.join(cfg['paths']['root'], *parts)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p


def amp_dtype(cfg):
    import torch
    m = str(cfg['hardware'].get('amp_dtype', 'bf16')).lower()
    if m == 'bf16' and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def pick_device(cfg):
    import torch
    return torch.device('cuda' if (cfg['hardware'].get('device', 'cuda') == 'cuda'
                                    and torch.cuda.is_available()) else 'cpu')


class Timer:
    def __enter__(self): self.t = time.time(); return self
    def __exit__(self, *a): self.dt = time.time() - self.t


def append_csv_row(path, row: dict, fieldnames=None):
    """Ghi 1 dong/seed vao CSV (tao header neu file chua co). Bang tu tinh mean+/-std."""
    import csv
    ensure_dir(os.path.dirname(path) or '.')
    exists = os.path.exists(path)
    fn = fieldnames or list(row.keys())
    with open(path, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fn)
        if not exists: w.writeheader()
        w.writerow(row)


def log(msg, logfile=None):
    line = f'[{time.strftime("%H:%M:%S")}] {msg}'
    print(line, flush=True)
    if logfile:
        ensure_dir(os.path.dirname(logfile) or '.')
        with open(logfile, 'a', encoding='utf-8') as f: f.write(line + '\n')
