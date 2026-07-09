# Pha 0 — Kiểm toán dữ liệu & sinh split (nền móng)

Ba script này chuẩn bị nền cho toàn bộ dự án: kiểm toán dataset, dựng bảng taxonomy,
và sinh split ID/OOD với **Near/Far theo taxonomy** (thay t-SNE) + **chia ảnh cấp specimen**
(chống leakage). Không cần GPU.

**Quyết định đã chốt (dùng xuyên suốt dự án):** pretrained ImageNet · baseline dùng OpenOOD · fine-tune SD bằng LoRA/DreamBooth.

## Cài đặt
```bash
pip install numpy scipy pillow        # đủ cho Pha 0
```

## Chạy theo đúng thứ tự

### 1) Kiểm toán dataset
```bash
python audit_dataset.py --data_root /path/to/Wood_ID_1-50 --out_dir results/audit --dup_threshold 5
```
Đầu ra trong `results/audit/`:
- `audit_summary.json` — số lớp, tổng ảnh, lớp min/max, tỉ lệ mất cân bằng, số file hỏng, số cặp ảnh gần trùng.
- `audit_per_image.csv` — từng ảnh (kích thước, mode, trạng thái/CORRUPT).
- `near_duplicates_within_class.csv` — các cặp ảnh gần trùng **trong cùng lớp**.

**Đọc kết quả:**
- File hỏng → xoá/sửa trước khi train.
- Nhiều cặp gần trùng trong lớp = dấu hiệu **nhiều ảnh cùng một khối gỗ (specimen)** → bắt buộc chia cấp specimen ở bước 3.
- `exact_cross_class_duplicate_groups > 0` = ảnh trùng y hệt ở hai lớp khác nhau → **kiểm tra nhãn ngay** (nghi gán sai).

### 2) Dựng khung bảng taxonomy
```bash
python build_taxonomy_template.py --data_root /path/to/Wood_ID_1-50 --out_csv taxonomy.csv
```
Script tự suy `scientific_name`/`genus` nếu tên thư mục là tên khoa học, và đánh dấu softwood đã biết.
**Bạn phải điền nốt các ô `TODO`** (đặc biệt `family` cho đủ 50 loài). Cột bắt buộc:
`class_name, scientific_name, genus, family, wood_type`.

> `taxonomy.csv` được dùng ở *cả* bước 3 (định nghĩa Near/Far) *lẫn* Pha 2 (sinh dữ liệu taxonomy-guided). Điền càng chuẩn càng tốt.

### 3) Sinh split ID/OOD
```bash
python make_splits.py --taxonomy_csv taxonomy.csv --data_root /path/to/Wood_ID_1-50 \
    --out_dir results/splits --seed 0 --scenarios 40/10,30/20 \
    --specimen_regex '(?P<sid>.+?)_\d+\.\w+$'
```
Đầu ra cho mỗi kịch bản (vd `results/splits/40_10/`):
- `manifest.csv` — `image_path, class, role, ood_type`. Role ∈ {train, val_id, test_id, test_ood}. `ood_type` ∈ {near, far, NA}.
- `class_assignment.csv` — mỗi lớp là ID hay OOD, near/far, genus, family.
- `results/splits/splits_summary.json` — tổng hợp số lớp/ảnh mỗi vai trò.

**Định nghĩa Near/Far** (theo tập ID cuối cùng):
- **FAR**: lớp OOD mà *họ* không xuất hiện trong ID.
- **NEAR**: lớp OOD mà *genus hoặc họ* có trong ID (khó hơn — vd loài khác cùng chi Afzelia/Dalbergia).

## ⚠️ Việc bạn PHẢI kiểm tra: `--specimen_regex`
Đây là điểm chống leakage quan trọng nhất. Regex cần một nhóm `(?P<sid>...)` để lấy **id mẫu vật** từ tên file, sao cho mọi ảnh của cùng một khối gỗ có chung `sid`.
- Nếu tên file là `Afzelia_afr_block03_img07.jpg` → dùng `'(?P<sid>.+?)_img\d+'`.
- Nếu **không** có quy luật đặt tên theo mẫu vật, tạm bỏ cờ này (script sẽ chia **cấp ảnh** và in cảnh báo) — nhưng phải nêu rõ hạn chế này trong bài, vì nó có thể làm kết quả cao ảo.

## Ngưỡng ID/OOD cho metric (nhắc lại giao thức)
Khi train/đánh giá ở các pha sau: dùng **train** để huấn luyện, **val_id** để chọn ngưỡng @95TPR
(95% mẫu val_id có score dưới ngưỡng), rồi báo cáo trên **test_id + test_ood**. OOD **chỉ** ở test.

## Tiếp theo
Sau khi (a) điền xong `taxonomy.csv` và (b) xác nhận split hợp lý → chuyển sang **Pha 1** (train 4 backbone + IsoMax+, tương thích OpenOOD) và **Pha 2a** (fine-tune SD bằng LoRA trên ảnh gỗ).
