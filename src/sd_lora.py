"""Pha 2a — LoRA fine-tune Stable Diffusion 1.5 tren anh go (KHONG full FT).
diffusers + peft LoRA tren UNet (to_q/k/v/out). VAE + text-encoder DONG BANG (fp16).
Tham so tu config.sd_lora. Resumable: luu peft-state (.pt, resume) + diffusers lora (.safetensors, inference)."""
import os, math, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, StableDiffusionPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from . import utils, sd_data as SD


def _lora_params(unet):
    return [p for p in unet.parameters() if p.requires_grad]


def train_sd_lora(cfg, log=print):
    sd = cfg['sd_lora']; base = sd['base_model']; res = int(sd['resolution'])
    steps = int(sd['train_steps']); accum = int(sd.get('grad_accum', 1)); bs = int(sd.get('batch', 1))
    save_every = int(sd.get('save_every', 500))
    dev = utils.pick_device(cfg); wdtype = torch.float16
    out_dir = utils.ensure_dir(utils.rp(cfg, cfg['paths']['checkpoints'], 'sd_lora'))
    resume_pt = os.path.join(out_dir, 'lora_resume.pt')
    final_flag = os.path.join(out_dir, 'pytorch_lora_weights.safetensors')
    logf = utils.rp(cfg, cfg['paths']['logs'], 'sd_lora.log')

    # skip neu da xong
    if os.path.exists(resume_pt) and torch.load(resume_pt, map_location='cpu').get('done'):
        utils.log('[sd_lora] da DONE -> bo qua', logf)
        return {'lora_dir': out_dir, 'skipped': True}

    tok = CLIPTokenizer.from_pretrained(base, subfolder='tokenizer')
    text_encoder = CLIPTextModel.from_pretrained(base, subfolder='text_encoder').to(dev, dtype=wdtype)
    vae = AutoencoderKL.from_pretrained(base, subfolder='vae').to(dev, dtype=wdtype)
    unet = UNet2DConditionModel.from_pretrained(base, subfolder='unet').to(dev)  # base fp32
    sched = DDPMScheduler.from_pretrained(base, subfolder='scheduler')
    vae.requires_grad_(False); text_encoder.requires_grad_(False); unet.requires_grad_(False)

    lc = LoraConfig(r=int(sd['rank']), lora_alpha=int(sd['alpha']), init_lora_weights='gaussian',
                    target_modules=['to_k', 'to_q', 'to_v', 'to_out.0'])
    unet.add_adapter(lc)
    for p in unet.parameters():
        if p.requires_grad: p.data = p.data.float()
    if sd.get('gradient_checkpointing', True):
        unet.enable_gradient_checkpointing()

    # optimizer: 8bit adam neu co bitsandbytes
    params = _lora_params(unet)
    try:
        import bitsandbytes as bnb
        opt = bnb.optim.AdamW8bit(params, lr=float(sd['lr'])) if sd.get('use_8bit_adam', True) \
            else torch.optim.AdamW(params, lr=float(sd['lr']))
    except Exception as e:
        log(f'[sd_lora] bnb loi ({e}) -> AdamW thuong'); opt = torch.optim.AdamW(params, lr=float(sd['lr']))
    scaler = torch.cuda.amp.GradScaler()

    start_step = 0
    if os.path.exists(resume_pt):
        ck = torch.load(resume_pt, map_location='cpu')
        set_peft_model_state_dict(unet, {k: v.to(dev) for k, v in ck['lora'].items()})
        start_step = ck['step']; utils.log(f'[sd_lora] RESUME tu step {start_step}', logf)

    ds = SD.WoodCaptionDataset(cfg, resolution=res, seed=cfg['project']['seeds'][0])
    loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=cfg['hardware']['num_workers'],
                        pin_memory=True, drop_last=True, persistent_workers=True)
    utils.log(f'[sd_lora] {len(ds)} anh | steps={steps} accum={accum} bs={bs} res={res} '
              f'| #lora_params={sum(p.numel() for p in params)}', logf)

    def save(step, done=False):
        lora_sd = get_peft_model_state_dict(unet)
        torch.save({'lora': {k: v.detach().cpu() for k, v in lora_sd.items()}, 'step': step, 'done': done}, resume_pt)
        if done:
            StableDiffusionPipeline.save_lora_weights(out_dir, unet_lora_layers=lora_sd, safe_serialization=True)

    unet.train(); step = start_step; t0 = time.time(); micro = 0; opt.zero_grad(set_to_none=True); run = 0.0
    done = False
    while not done:
        for batch in loader:
            px = batch['pixel_values'].to(dev, dtype=wdtype)
            with torch.no_grad():
                lat = vae.encode(px).latent_dist.sample() * vae.config.scaling_factor
                ids = tok(list(batch['caption']), padding='max_length', truncation=True,
                          max_length=tok.model_max_length, return_tensors='pt').input_ids.to(dev)
                enc = text_encoder(ids)[0]
            noise = torch.randn_like(lat)
            ts = torch.randint(0, sched.config.num_train_timesteps, (lat.shape[0],), device=dev).long()
            noisy = sched.add_noise(lat, noise, ts)
            target = noise if sched.config.prediction_type == 'epsilon' else sched.get_velocity(lat, noise, ts)
            with torch.autocast('cuda', dtype=wdtype):
                pred = unet(noisy, ts, enc).sample
            loss = F.mse_loss(pred.float(), target.float()) / accum
            scaler.scale(loss).backward(); run += loss.item() * accum; micro += 1
            if micro % accum == 0:
                scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); step += 1
                if step % 50 == 0:
                    utils.log(f'[sd_lora] step {step}/{steps} loss={run/50:.4f} ({(time.time()-t0)/max(step-start_step,1):.1f}s/step)', logf); run = 0.0
                if step % save_every == 0:
                    save(step); utils.log(f'[sd_lora] ckpt @step{step}', logf)
                if step >= steps:
                    done = True; break
    save(step, done=True)
    utils.log(f'[sd_lora] XONG {step} steps -> {out_dir}', logf)
    return {'lora_dir': out_dir, 'skipped': False, 'steps': step}
