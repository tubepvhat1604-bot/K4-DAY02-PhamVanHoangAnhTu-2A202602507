# Lab Day 2 (DeepWeeds) — Phạm Văn Hoàng Anh Tú, MSSV 2A202602507

**Cấu hình chung kết F01** gồm:
- backbone ConvNeXt-T (`convnext_tiny.in12k_ft_in1k`);
- công thức nền T00 cộng thêm TrivialAugment, huấn luyện 12 epoch;
- suy luận ở độ phân giải 288 kèm temperature scaling, với T khớp trên val.

Kết quả trên test, trung bình 3 seed (`eval.py score`):

| Cấu hình | macro-F1 | top-1 | ECE |
|---|---|---|---|
| F01 | **0,9789 ± 0,0012** | **98,29 ± 0,10 %** | 0,0047 |
| Mốc T00 + I00 | 0,9709 ± 0,0012 | 97,66 ± 0,16 % | 0,0099 |

- Độ trễ p95 ở batch 1 là 7,89 ms, đo trên Tesla T4 với AMP, không tính tiền xử lý.
- `eval.py grade` tự chấm phần I được 19/20.

Phân tích đầy đủ nằm trong [`report.md`](report.md). Mọi bảng số nằm trong [`results.xlsx`](results.xlsx).

## Nội dung thư mục

| Đường dẫn | Nội dung |
|---|---|
| `results.xlsx` | 7 sheet: Summary, Backbones, Training, Inference, Final, PerClass, Latency. Ô B5.1 sinh tự động từ log bằng `code/buoc5.py` |
| `report.md` | báo cáo theo GUIDE mục 6.3 |
| `curves/` | mỗi thí nghiệm huấn luyện (B01–B06, T00–T11, F01) có một ảnh `<exp_id>[_seed<k>]_<mota>.png`; ngoài ra có biểu đồ tổng hợp, ma trận nhầm lẫn và ảnh lỗi |
| `figures/` | EDA, ảnh sau augmentation, ước lượng thời gian Bước 0 |
| `predictions/` | dự đoán test của F01 và T00 (3 seed); thêm F01 chưa TS (`*_uncal_*`) và dự đoán val của F01 (`*_val.csv`) |
| `logs/` | `config.json`, `history.csv`, `done.json` của từng run; kết quả Bước 3 (`inference_val.csv`, `latency.csv`, `buoc3_choice.json`); kết quả `eval.py` (`logs/eval/`) |
| `code/` | khung `starter/` đã hoàn thiện cùng code của từng bước và các notebook |

**Code trong `code/`:**

| File | Vai trò |
|---|---|
| `dataset.py`, `model.py`, `losses.py`, `train.py` | khung starter đã hoàn thiện; `train.run(Config)` là hàm train dùng chung cho mọi cấu hình, có resume và tự bỏ qua run đã xong |
| `pipeline_checks.py` | EDA và các kiểm tra Bước 0 |
| `inference.py`, `benchmark.py` | TTA, temperature scaling, soup, gộp BN; đo độ trễ p50/p95/p99 |
| `buoc3.py`, `buoc4.py`, `buoc5.py` | Bước 3, Bước 4 (test một lần), Bước 5 |
| `test_code.py` | test tự viết |

## Môi trường và phiên bản

- **Bước 0–1:** Google Colab, GPU T4.
- **Bước 2–4:** Kaggle, GPU Tesla T4 (Colab miễn phí hết lượt GPU giữa chừng).
- **Bước 5:** Colab, CPU là đủ.
- **Phiên bản** (ghi trong `logs/*/config.json`): Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, numpy 2.1.3, pandas 2.2.3. Cài thêm bằng `pip install timm openpyxl`.
- **Dữ liệu:**
  - `images.zip` lấy từ Zenodo record 7939060, MD5 `b7b30f96d466fba86016aa5a26606e0f`, giải nén ra đĩa cục bộ;
  - 4 file CSV fold 0 lấy nguyên bản từ `github.com/AlexOlsen/DeepWeeds/labels`, không sửa.
- **Seed:** seed 0 cho mọi thí nghiệm; T00 và F01 dùng seed 0, 1, 2.

## Chạy lại

| Notebook | Mở | Việc |
|---|---|---|
| `code/lab_day2.ipynb` | [Open in Colab](https://colab.research.google.com/github/tubepvhat1604-bot/K4-DAY02-PhamVanHoangAnhTu-2A202602507/blob/main/submissions/2A202602507_pham_van_hoang_anh_tu/code/lab_day2.ipynb) | notebook chính, chạy từ trên xuống |
| `code/kaggle_buoc2.ipynb` | Kaggle: File → Import Notebook → GitHub | Bước 2: T00 (3 seed) và T01–T10 |
| `code/kaggle_buoc2b.ipynb` | như trên | T11 (tổ hợp) và F01 (3 seed) |
| `code/kaggle_buoc3.ipynb` | như trên, Add Input = output của notebook 2b | Bước 3 trên val |
| `code/kaggle_buoc4.ipynb` | như trên, Add Input = output của notebook 2 và 2b | Bước 4: test một lần mỗi seed, `eval.py score` và `grade` |

Các notebook Kaggle cần bật **GPU T4** và **Internet on**, rồi chạy bằng **Save Version → Save & Run All**. Mỗi notebook tự clone code từ repo, tự tải và kiểm tra dữ liệu, cuối cùng gói kết quả nhỏ (không có `.pt`) vào `kaggle_buoc*.zip`.

**Thứ tự chạy:**
1. Trong `lab_day2.ipynb`, chạy Ô 0.1–0.4 (mount Drive, nạp code, tải dữ liệu, cấu hình), rồi Bước 0 (B0.1–B0.6) và Bước 1 (B1.1–B1.2) trên Colab GPU.
2. Chạy lần lượt `kaggle_buoc2`, `kaggle_buoc2b`, `kaggle_buoc3`, `kaggle_buoc4`. Đưa các file `kaggle_buoc*.zip` vào `MyDrive/deepweeds_data/`.
3. Trong `lab_day2.ipynb`:
   - Ô B2.M gộp các zip vào thư mục bài nộp và `deepweeds_runs/`.
   - Ô B2.1 và B2.4 tự bỏ qua các run đã có `done.json`.
   - Ô B3.1 và B4.1 chỉ đọc lại kết quả.
   - Ô B4.2–B4.3 chấm lại bằng `eval.py` từ `predictions/`.
   - Ô B5.1–B5.3 sinh `results.xlsx`, biểu đồ, ảnh lỗi và tự kiểm tra bài nộp.
4. Chấm lại từ thư mục này:

   ```bash
   python eval.py score --pred "submissions/2A202602507_pham_van_hoang_anh_tu/predictions/F01_seed*_test.csv" \
       --test-csv labels/test_subset0.csv --labels labels/labels.csv --tag F01
   python eval.py grade \
       --final    "submissions/2A202602507_pham_van_hoang_anh_tu/predictions/F01_seed*_test.csv" \
       --baseline "submissions/2A202602507_pham_van_hoang_anh_tu/predictions/T00_seed*_test.csv" \
       --uncal    "submissions/2A202602507_pham_van_hoang_anh_tu/predictions/F01_uncal_seed*_test.csv" \
       --final-val "submissions/2A202602507_pham_van_hoang_anh_tu/predictions/F01_seed*_val.csv" \
       --latency-p95-ms 7.89 --latency-method proper \
       --test-csv labels/test_subset0.csv --val-csv labels/val_subset0.csv --labels labels/labels.csv
   ```

**Quy tắc đã tuân theo:**
- Mọi lựa chọn đều làm trên val.
- Test chạy đúng một lần mỗi seed, trong `buoc4.run_test`. Hàm này tự dừng nếu đã có file `*_test.csv`.
- Checkpoint (`best.pt`) nằm ngoài repo, trên Drive và trong output của Kaggle. Repo không chứa ảnh hay checkpoint.
