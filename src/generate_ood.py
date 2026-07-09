"""Pha 2d — decode outlier embedding -> anh OOD qua SD+LoRA (prompt_embeds). CLAUDE.md §7.
Diem tich hop kho: outlier embed (768, khong gian CLIP text-embed da can o 2b) duoc XAP XI thanh
conditioning bang TO HOP CO TRONG SO prompt_embeds cua cac loai ID (w=softmax(e·proto/tau)):
 - NEAR: w tap trung 1-2 loai cung chi -> pha SEQUENCE embeds -> "loai la cung chi".
 - FAR/RANDOM : w loang -> pha trung binh + tron STRESSOR embed.
GHI RO: XAP XI co nguyen tac (proto/e chi de TINH TRONG SO; conditioning cuoi la SEQUENCE embeds THAT).
Resume-able. emb_file/out_subroot cho phep tai su dung cho ablation (random sampling). CITE: DREAM-OOD."""
import os
import numpy as np
import torch
from . import utils, sd_data as SD, sd_generate as G


@torch.no_grad()
def _seq_embeds(pipe, captions, device):
    ids = pipe.tokenizer(captions, padding='max_length', max_length=77, truncation=True,
                         return_tensors='pt').input_ids.to(device)
    return pipe.text_encoder(ids)[0]


@torch.no_grad()
def decode_outliers(cfg, split, max_gen=None, tau=0.1, log=print, emb_file=None, out_subroot='images'):
    dev = utils.pick_device(cfg); sy = cfg['synth']
    steps = int(sy['gen_steps']); guid = float(sy['guidance_scale'])
    res = int(sy['gen_resolution']); rs = int(sy['resize_to']); gb = int(sy['gen_batch'])
    emb_file = emb_file or f'outlier_embed_{split}.npz'
    z = np.load(utils.rp(cfg, cfg['paths']['synth'], 'embeddings', emb_file), allow_pickle=True)
    embed = z['embed'].astype(np.float32); otype = np.array(z['otype'])
    if max_gen: embed = embed[:max_gen]; otype = otype[:max_gen]
    outdir = utils.rp(cfg, cfg['paths']['synth'], out_subroot, split)
    for t in set(otype.tolist()):
        utils.ensure_dir(os.path.join(outdir, str(t)))
    paths = [os.path.join(outdir, str(otype[i]), f'{split}_{i:05d}.png') for i in range(len(embed))]

    LORA = utils.rp(cfg, cfg['paths']['checkpoints'], 'sd_lora')
    pipe = None; tax = SD.load_taxonomy(cfg)
    idz = np.load(utils.rp(cfg, cfg['paths']['synth'], 'embeddings', f'id_feat_{split}.npz'), allow_pickle=True)
    classes = list(idz['classes'])
    proto = torch.tensor(idz['proto'].astype(np.float32), device=dev)
    sp = stress = neg1 = None; dt = torch.float16
    logf = utils.rp(cfg, cfg['paths']['logs'], 'gen_ood.log')
    n_done = 0
    for i0 in range(0, len(embed), gb):
        pj = paths[i0:i0+gb]
        if all(os.path.exists(p) for p in pj):
            n_done += len(pj); continue
        if pipe is None:
            pipe = G.load_pipeline(cfg, LORA)
            sp = _seq_embeds(pipe, [G.clean_caption(tax[c][0], None) for c in classes], dev); dt = sp.dtype
            stress = _seq_embeds(pipe, ["macroscopic wood cross-section, weathered fungal-stained surface, "
                                        "insect boreholes, atypical unknown species, 50x"], dev)[0]
            negp = __import__('yaml').safe_load(open(utils.rp(cfg, cfg['paths']['prompts'])))['negative_prompt']
            neg1 = _seq_embeds(pipe, [negp], dev)[0]
        eb = torch.tensor(embed[i0:i0+gb], device=dev); ob = otype[i0:i0+gb]
        w = torch.softmax((eb @ proto.T) / tau, dim=1).to(dt)
        cond = torch.einsum('bc,csd->bsd', w, sp)
        is_far = torch.tensor([t in ('far', 'random') for t in ob], device=dev, dtype=dt).view(-1, 1, 1)
        cond = (1 - 0.5*is_far) * cond + (0.5*is_far) * stress
        neg = neg1.unsqueeze(0).expand(len(eb), -1, -1)
        g = torch.Generator(device=dev).manual_seed(1000 + i0)
        imgs = pipe(prompt_embeds=cond, negative_prompt_embeds=neg, num_inference_steps=steps,
                    guidance_scale=guid, height=res, width=res, generator=g).images
        for j, im in enumerate(imgs):
            im.resize((rs, rs)).save(pj[j]); n_done += 1
        if (i0 // gb) % 10 == 0:
            utils.log(f'[gen_ood {split}/{out_subroot}] {n_done}/{len(embed)}', logf)
    utils.log(f'[gen_ood {split}/{out_subroot}] DONE {n_done}/{len(embed)}', logf)
    return {'split': split, 'n_generated': n_done, 'dir': outdir}
