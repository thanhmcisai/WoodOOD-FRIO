"""Dataset cho LoRA fine-tune SD: anh go THAT + caption sinh tu prompts.yaml x taxonomy.csv.
Caption LoRA CHI dung condition.normal (mo ta anh that). Pixel chuan hoa [-1,1] (VAE SD)."""
import os, csv, random
import yaml
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from . import utils

Image.MAX_IMAGE_PIXELS = None
IMG_EXT = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}


def load_taxonomy(cfg):
    """class_name -> (scientific_name, genus)."""
    out = {}
    with open(utils.rp(cfg, cfg['paths']['taxonomy']), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            out[r['class_name'].strip()] = (r['scientific_name'].strip(), r['genus'].strip())
    return out


class WoodCaptionDataset(Dataset):
    def __init__(self, cfg, resolution=512, seed=0):
        self.cfg = cfg
        self.res = resolution
        self.tax = load_taxonomy(cfg)
        self.pb = yaml.safe_load(open(utils.rp(cfg, cfg['paths']['prompts']), encoding='utf-8'))
        droot = utils.data_root(cfg)
        self.items = []   # (abs_path, class_name)
        for c in sorted(os.listdir(droot)):
            cd = os.path.join(droot, c)
            if not os.path.isdir(cd) or c.startswith('.'):
                continue
            for fn in os.listdir(cd):
                if os.path.splitext(fn)[1].lower() in IMG_EXT:
                    self.items.append((os.path.join(cd, fn), c))
        self.rng = random.Random(seed)
        self.tf = transforms.Compose([
            transforms.Resize(resolution),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),   # -> [-1,1]
        ])

    def __len__(self):
        return len(self.items)

    def caption_for(self, class_name):
        sci, genus = self.tax.get(class_name, (class_name, class_name))
        pb = self.pb
        tmpl = self.rng.choice(pb['base_templates'])
        photo = self.rng.choice(pb['photographic'])
        surf = self.rng.choice(pb['surface'])
        cond = self.rng.choice(pb['condition']['normal'])          # LoRA: chi normal
        cap = tmpl.format(sci=sci, genus=genus, photo=photo, surface=surf, condition=cond)
        hint = pb.get('genus_hints', {}).get(genus)
        if hint and self.rng.random() < 0.5:
            cap = cap + ', ' + hint
        return cap

    def __getitem__(self, i):
        path, c = self.items[i]
        with Image.open(path) as im:
            px = self.tf(im.convert('RGB'))
        return {'pixel_values': px, 'caption': self.caption_for(c)}
