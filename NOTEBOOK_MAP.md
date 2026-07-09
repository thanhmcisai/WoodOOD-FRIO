# NOTEBOOK_MAP.md — Sổ đăng ký cell (NGUỒN CHÂN LÝ về trạng thái notebook)

> Claude Code PHẢI cập nhật bảng này sau mỗi lần tạo/sửa/chạy cell (xem CLAUDE.md mục 2).
> Trước khi viết cell mới: đọc bảng này + `src/` để KHÔNG tạo trùng.
> status: TODO → RUNNING → DONE. Ghi output thực tế + thời gian chạy + ghi chú.

## Bảng cell

| Cell | Phase | Name | Input | Output | Status | Runtime | Ghi chú |
|------|-------|------|-------|--------|--------|---------|---------|
| 00 | setup | mount_setup_config | config.yaml | env ready | TODO | | mount Drive, cài deps (pin), seed |
| 01 | data | data_check | data/, splits/, taxonomy.csv | results/data_check.json | TODO | | đối chiếu lớp/ảnh khớp taxonomy |
| 02 | data | dataset_module | manifest.csv | dataloaders (src/data.py) | TODO | | specimen-aware, theo role |
| 03 | id | train_backbones | data | checkpoints/{bb}_{loss}.pt | TODO | | 4 backbone × {ce, isomax+}, pretrained |
| 04 | id | eval_id_classification | ckpt | results/id_classification.csv | TODO | | ACC/macroF1/bal-acc/inference, ≥3 seed |
| 05 | 2a | sd_lora_finetune | data/, prompts.yaml | checkpoints/sd_lora/ | TODO | | LoRA, KHÔNG full FT |
| 05a | 2a | sd_sanity_check | sd_lora | results/sd_sanity.json, figures/sd_sanity_grid.png | TODO | | sinh thử loài đã biết + FID sơ bộ |
| 06 | 2b | learn_latent_space | data/, token_embed | synth/embeddings/id_feat_*.npy | TODO | | DREAM-OOD Bước1 (cite) |
| 07 | 2c | sample_outliers | id_feat, taxonomy | synth/embeddings/outlier_embed_*.npy | TODO | | kNN, taxonomy-guided near/far |
| 08 | 2d | generate_ood_images | outlier_embed, sd_lora, prompts | synth/images/... | TODO | | embedding→prompt_embeds (mục 7) |
| 08a | 2e | eval_synth_quality | synth/, real | results/synth_quality.csv | TODO | | FID, LPIPS-diversity, sanity outlier |
| 09 | 3 | train_eool | data, synth | checkpoints/eool_*.pt | TODO | | IsoMax+ + β·L_ood, mọi bb×split×{near,far}, ≥3 seed |
| 10 | 4 | baselines_openood | ckpt id | results/ood_40_10.csv, results/ood_30_20.csv | TODO | | MSP,ODIN,Energy,Mahalanobis,ReAct,KNN,MaxLogit,ViM,GEN,SHE,IsoMax+ |
| 11 | 5 | aggregate_metrics | ckpts | results/ood_*.csv (mean±std, sig-test) | TODO | | gộp EOIL + baselines |
| 12 | 6 | ablations | tùy | results/ablation.csv, results/ablation_sampling.csv | TODO | | 4 biến thể + quét β + taxo vs random + có/không LoRA |
| 13 | 7 | qualitative | ckpt, scores | results/scores_*.csv, results/threshold_*.csv | TODO | | score dist, confusion ngưỡng, ROC/PR, Grad-CAM |
| 14 | 8 | export_onnx | ckpt | checkpoints/*.onnx | TODO | | tùy chọn |
| 15 | fig | make_figures | results/, synth/ | figures/*.pdf,*.png | TODO | | CHỈ đọc results/; no hardcode |
| 16 | tab | make_tables | results/ | tables/*.tex | TODO | | CHỉ đọc results/; no hardcode |

## src/ modules (điền khi tạo)

| Module | Chức năng | Trạng thái |
|--------|-----------|------------|
| utils.py | config loader, seed, paths, logging | TODO |
| data.py | Dataset/dataloader từ manifest (specimen-aware) | TODO |
| models.py | 4 backbone (pretrained) + head | TODO |
| isomax.py | IsoMax+ (Macêdo & Ludermir 2021 — cite) | TODO |
| latent_space.py | học không gian có điều kiện text (DREAM-OOD — cite) | TODO |
| sampling.py | kNN outlier sampling taxonomy-guided (NPOS — cite) | TODO |
| generate.py | mở rộng prompt + decode embedding→ảnh (diffusers, prompt_embeds) | TODO |
| eool.py | loss năng lượng + φ + train EOIL | TODO |
| baselines_openood.py | bọc model cho OpenOOD | TODO |
| metrics.py | FPR@95TPR, AUROC, AUPR, bootstrap sig-test | TODO |
| gradcam.py | Grad-CAM | TODO |
| plotting.py | mọi hàm vẽ (đọc results/, style cố định) | TODO |
| tables.py | build .tex từ CSV (mean±std) | TODO |
