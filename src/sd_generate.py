"""Dung pipeline SD1.5 (+LoRA Pha 2a) va sinh anh. Dung cho sanity 2a & generation 2d.
safety_checker=None (tranh filter lam trang anh go). Scheduler DPMSolver (nhanh)."""
import os
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from . import utils


def load_pipeline(cfg, lora_dir=None, device=None):
    base = cfg['sd_lora']['base_model']
    pipe = StableDiffusionPipeline.from_pretrained(base, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    if lora_dir and os.path.exists(os.path.join(lora_dir, 'pytorch_lora_weights.safetensors')):
        pipe.load_lora_weights(lora_dir)
        pipe.fuse_lora()
        print(f'[gen] da nap+fuse LoRA tu {lora_dir}')
    else:
        print('[gen] KHONG nap LoRA (SD goc)')
    pipe.set_progress_bar_config(disable=True)
    pipe.to(device or utils.pick_device(cfg))
    return pipe


@torch.no_grad()
def generate(pipe, prompts, negative_prompt=None, steps=50, guidance=7.5, seed=0, res=512):
    if isinstance(prompts, str):
        prompts = [prompts]
    g = torch.Generator(device=pipe.device).manual_seed(int(seed))
    neg = [negative_prompt] * len(prompts) if negative_prompt else None
    out = pipe(prompts, negative_prompt=neg, num_inference_steps=int(steps),
               guidance_scale=float(guidance), height=int(res), width=int(res), generator=g)
    return out.images


def clean_caption(sci, pb):
    """Caption 'sach' (normal) cho sanity: mo ta anh that theo danh phap."""
    return (f"macroscopic cross-section of {sci} wood, 50x magnification, sharp focus, "
            f"freshly sanded 600-grit polished surface, clean, well-preserved")
