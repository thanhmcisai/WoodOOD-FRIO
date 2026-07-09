"""Dataset/dataloader tu manifest (da chia specimen-aware o Pha 0).
manifest.csv: image_path, class, role, ood_type. role in {train,val_id,test_id,test_ood}.
Nhan phan loai = index cua lop ID (sort). image_path la tuong doi (NFC) -> ghep voi data_root."""
import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from . import utils

Image.MAX_IMAGE_PIXELS = None
ID_ROLES = ('train', 'val_id', 'test_id')


def read_manifest(cfg, split):
    return pd.read_csv(utils.rp(cfg, cfg['paths']['splits'], split, 'manifest.csv'), encoding='utf-8')


def id_classes(df):
    """Danh sach lop ID (sort on dinh) = lop xuat hien o train/val_id/test_id."""
    return sorted(df[df.role.isin(ID_ROLES)]['class'].unique().tolist())


def build_transforms(cfg, train):
    sz = cfg['data']['image_size']
    mean, std = cfg['data']['normalize_mean'], cfg['data']['normalize_std']
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(sz, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAdjustSharpness(2, p=0.5),
            transforms.RandomAutocontrast(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    return transforms.Compose([
        transforms.Resize(int(round(sz * 1.15))),
        transforms.CenterCrop(sz),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


class WoodDataset(Dataset):
    def __init__(self, df, data_root, class_to_idx, transform):
        self.paths = [os.path.join(data_root, p) for p in df['image_path'].tolist()]
        # nhan: -1 cho anh OOD (class khong nam trong class_to_idx)
        self.labels = [class_to_idx.get(c, -1) for c in df['class'].tolist()]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        with Image.open(self.paths[i]) as im:
            x = self.transform(im.convert('RGB'))
        return x, self.labels[i]


def make_loaders(cfg, split, seed, batch_size, num_workers=None, balanced=True, roles=('train', 'val_id', 'test_id')):
    """Tra ve dict: loaders theo role + id_classes + class_to_idx. Train dung WeightedRandomSampler (chong mat can bang)."""
    nw = cfg['hardware']['num_workers'] if num_workers is None else num_workers
    droot = utils.data_root(cfg)
    df = read_manifest(cfg, split)
    idc = id_classes(df)
    c2i = {c: i for i, c in enumerate(idc)}

    out = {'id_classes': idc, 'class_to_idx': c2i, 'num_classes': len(idc)}
    for role in roles:
        sub = df[df.role == role].reset_index(drop=True)
        is_train = (role == 'train')
        ds = WoodDataset(sub, droot, c2i, build_transforms(cfg, train=is_train))
        if is_train and balanced and len(ds):
            labels = np.array(ds.labels)
            cnt = np.bincount(labels, minlength=len(idc)).astype(np.float64)
            w = 1.0 / np.clip(cnt, 1, None)
            samp_w = torch.as_tensor(w[labels], dtype=torch.double)
            g = torch.Generator().manual_seed(seed)
            sampler = WeightedRandomSampler(samp_w, num_samples=len(ds), replacement=True, generator=g)
            loader = DataLoader(ds, batch_size=batch_size, sampler=sampler, num_workers=nw,
                                pin_memory=True, drop_last=True, persistent_workers=nw > 0)
        else:
            loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=nw,
                                pin_memory=True, persistent_workers=nw > 0)
        out[role] = loader
        out[role + '_n'] = len(ds)
    return out
