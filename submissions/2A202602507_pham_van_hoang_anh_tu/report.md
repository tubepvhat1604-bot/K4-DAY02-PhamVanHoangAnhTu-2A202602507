# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds

**Phạm Văn Hoàng Anh Tú — MSSV 2A202602507**

Mọi con số trong báo cáo lấy từ `results.xlsx`. File này được ô B5.1 của `code/lab_day2.ipynb` gom tự động từ log thật của từng lần chạy: `logs/<exp_id>_seed<k>/done.json`, `logs/inference_val.csv`, `logs/latency.csv` và `logs/eval/` (kết quả của `eval.py`). Không có số nào được gõ tay. Δ ghi bằng điểm macro-F1 (0,01 = 1 điểm). "Không phân biệt được" nghĩa là chênh lệch không vượt quá nhiễu giữa các seed.

## 1. Tóm tắt

- **Bài toán:** phân loại 9 lớp (8 loài cỏ dại và `Negative`) trên DeepWeeds, dùng fold 0 chia sẵn của tác giả.
- **Đã chạy:**
  - 6 backbone (Bước 1);
  - 12 cấu hình công thức huấn luyện trên 5 trục A, B, C, D, F, gồm một tổ hợp (Bước 2);
  - 13 cấu hình suy luận kèm đo độ trễ p50/p95/p99 (Bước 3);
  - chung kết 3 seed, mỗi seed chạy test đúng một lần (Bước 4).
- **Cấu hình tốt nhất (F01):**
  - backbone ConvNeXt-T (`convnext_tiny.in12k_ft_in1k`);
  - công thức nền T00 thêm TrivialAugment, 12 epoch;
  - suy luận ở độ phân giải 288 (I04_res288), thêm temperature scaling với T khớp trên val.
- **Kết quả test (3 seed, 3507 ảnh):**
  - top-1 **98,29 ± 0,10 %**, macro-F1 **0,9789 ± 0,0012**, ECE **0,0047 ± 0,0020**;
  - recall Chinee apple 96,8 % và Snake weed 94,6 %;
  - độ trễ p95 ở batch 1 là **7,89 ms** (Tesla T4, AMP).
- **So với mốc T00 + I00:** macro-F1 test của mốc là 0,9709 ± 0,0012, nên F01 hơn **+0,0080**, khoảng 6,7 lần std. Đây là chênh lệch vượt nhiễu.
- **Kết luận chính:**
  - Backbone (kèm trọng số tiền huấn luyện) quyết định phần lớn chất lượng. Chọn sai backbone có thể mất tới 24 điểm.
  - Trong công thức huấn luyện, chỉ trục khởi tạo tạo chênh lệch vượt xa nhiễu.
  - TrivialAugment và test ở độ phân giải 288 mỗi thứ góp thêm khoảng 0,4–0,6 điểm. Độ phân giải 288 gần như không tốn thêm độ trễ ở batch 1.
  - `eval.py grade` tự chấm phần I được **19/20**. Ý duy nhất thiếu là I2: được 4/5 vì Δ = 0,0080 nhỏ hơn ngưỡng 0,01.

## 2. Dữ liệu và thiết lập

**Dữ liệu.**
- Lấy `images.zip` từ Zenodo (record 7939060) và kiểm tra MD5 `b7b30f96d466fba86016aa5a26606e0f`.
- 4 file CSV fold 0 tải nguyên bản từ `github.com/AlexOlsen/DeepWeeds/labels`, không sửa.
- Ô B0.1 chạy các kiểm tra S1–S4 và in ra:
  - train 10 501 ảnh (59,97 %), val 3 501 (20,00 %), test 3 507 (20,03 %);
  - giao từng cặp theo `Filename` đều bằng 0;
  - hợp ba tập đủ 17 509 ảnh, không thiếu file nào.

**Phân bố lớp** (ô B0.2, `figures/eda_class_distribution.png`, ảnh mẫu ở `figures/eda_samples.png`):

| Lớp | train | val | test | tổng |
|---|---|---|---|---|
| Chinee apple | 675 | 225 | 226 | 1 126 |
| Lantana | 637 | 213 | 213 | 1 063 |
| Parkinsonia | 618 | 206 | 207 | 1 031 |
| Parthenium | 613 | 204 | 205 | 1 022 |
| Prickly acacia | 637 | 212 | 213 | 1 062 |
| Rubber vine | 605 | 202 | 202 | 1 009 |
| Siam weed | 644 | 215 | 215 | 1 074 |
| Snake weed | 609 | 203 | 204 | 1 016 |
| Negative | 5 463 | 1 821 | 1 822 | 9 106 |

- `Negative` chiếm 52 % dữ liệu. Mỗi loài cỏ chỉ có khoảng 5,8–6,4 %, nên dữ liệu mất cân bằng khoảng 9 : 1 giữa `Negative` và từng loài. Vì vậy chỉ số chính là macro-F1, không phải accuracy.
- Tổng số ảnh mỗi lớp khớp Table 1 của bài báo, trừ một điểm lệch nhỏ:
  - Ảnh `20170714-110407-3.jpg` có `Label = 0` (Chinee apple) trong cả 5 fold CSV, nhưng có `Label = 1` (Lantana) trong `labels.csv`.
  - Vì thế tổng theo fold là Chinee apple 1 126 và Lantana 1 063, lệch 1 ảnh so với `labels.csv`.
  - Ảnh này nằm trong tập train. Bài làm giữ nguyên CSV đúng quy tắc S1–S6 và chỉ ghi nhận ở đây.

**Kiểm tra pipeline** (ô B0.3–B0.5):
- Loss ban đầu của ResNet-50 là 2,1756, gần ln 9 = 2,1972 (lệch 0,022).
- Overfit 16 ảnh: loss giảm từ 2,182 xuống 0,0002 sau 60 bước, accuracy trên batch đạt 1,00.
- Ảnh sau augmentation đã giải chuẩn hoá nằm ở `figures/aug_basic.png` và `figures/aug_trivial.png`.
- Test tự viết trong `code/test_code.py` kiểm tra:
  - focal γ = 0 cho đúng CE;
  - label smoothing;
  - CutMix có λ theo diện tích thật và trộn cả nhãn;
  - weight decay không áp cho norm/bias;
  - EMA;
  - TTA lật và multi-crop;
  - gộp xác suất và gộp logit;
  - temperature scaling khôi phục được T đã biết;
  - model soup;
  - gộp BN chính xác tới 1e-5;
  - phân vị độ trễ.

  Test của repo (`tests/`) đều đạt.

**Công thức nền T00** (`code/train.py`, lớp `Config`):
- tinh chỉnh toàn bộ mạng, ảnh 224;
- augmentation cơ bản: RandomResizedCrop và lật ngang;
- AdamW với LR backbone 1e-4, LR head 1e-3, weight decay 0,05 (không áp cho norm/bias);
- warmup 1 epoch rồi cosine;
- batch 64, 12 epoch, AMP, CE;
- chọn checkpoint theo macro-F1 val tốt nhất.

**Phần cứng và phiên bản:**
- Bước 0–1 chạy trên Google Colab (GPU T4).
- Bước 2–4 chạy trên Kaggle (Tesla T4), vì Colab miễn phí hết lượt GPU giữa chừng.
- Phiên bản ghi trong `config.json` của mỗi run: Python 3.13.15, torch 2.11.0+cu128, torchvision 0.26.0+cu128, timm 1.0.29, numpy 2.1.3.
- Seed 0 cho Bước 1–3. T00 và F01 chạy thêm seed 1 và 2.

**Thay đổi so với gợi ý của GUIDE, do giới hạn ngân sách GPU:**
- Dùng 12 epoch.
- Bước 2 chỉ chạy trên 1 backbone (ConvNeXt-T).
- Ablation chạy 1 seed, còn 3 seed dành cho T00 và chung kết.

**Nhiễu nền tảng:** cùng cấu hình T00 seed 0, chạy trên Colab (B02) được macro-F1 val 0,9666, còn chạy trên Kaggle (T00 seed 0) được 0,9636. Do đó mọi run Bước 2–4 đều chạy trên Kaggle để so sánh cùng điều kiện. Ngược lại, trên cùng nền tảng kết quả lặp lại được: F01 seed 0 và T03 seed 0 có cùng cấu hình, và history của hai run trùng khớp (macro-F1 val đều 0,9742).

## 3. So sánh backbone (Bước 1)

Cùng công thức T00, cùng split, seed 0. Số liệu ở sheet `Backbones`, biểu đồ ở `curves/B0*_*.png` và `curves/summary_backbones.png`.

| exp_id | backbone (tag timm) | #tham số (M) | GMAC | macro-F1 val | top-1 val | train/epoch (s) | độ trễ b1 sơ bộ (ms) |
|---|---|---|---|---|---|---|---|
| B02 | convnext_tiny.in12k_ft_in1k | 27,8 | 4,46 | **0,9666** | 0,9740 | 53,3 | 8,1 |
| B06 | swin_tiny_patch4_window7_224.ms_in1k | 27,5 | 4,49 | 0,9485 | 0,9600 | 68,6 | 18,2 |
| B03 | deit_small_patch16_224.fb_in1k | 21,7 | 4,60 | 0,9470 | 0,9634 | 40,5 | 6,6 |
| B01 | resnet50.a1_in1k | 23,5 | 4,09 | 0,8145 | 0,8655 | 46,9 | 7,5 |
| B04 | efficientnet_b0.ra_in1k | 4,0 | 0,39 | 0,7758 | 0,8343 | 36,0 | 10,8 |
| B05 | mobilenetv3_large_100.ra_in1k | 4,2 | 0,22 | 0,7245 | 0,7983 | 31,8 | 8,5 |

Độ trễ ở bảng này là phép đo sơ bộ: 30 lần, có warmup và synchronize. Phép đo đầy đủ p50/p95/p99 nằm ở Bước 3.

**Nhận xét.**
- **Đo được:**
  - ConvNeXt-T cao nhất, hơn Swin-T 1,8 điểm.
  - Swin-T và DeiT-S chỉ cách nhau 0,15 điểm, nên với 1 seed hai mạng này không phân biệt được.
  - ResNet-50 thấp hơn ConvNeXt-T tới 15,2 điểm. EfficientNet-B0 và MobileNetV3 thấp hơn 19–24 điểm.
  - Cả ba CNN dùng BatchNorm (B01, B04, B05) đạt best epoch = 12 và đường val vẫn đang đi lên (`curves/B01_resnet50.png`, `B04`, `B05`). Hai transformer đạt đỉnh sớm hơn (DeiT ở epoch 11, Swin ở epoch 7).
- **Phỏng đoán (chưa kiểm chứng):**
  - Ba CNN này thiếu khớp (underfit) với LR backbone 1e-4 trong 12 epoch.
  - Bộ trọng số `resnet50.a1` được huấn luyện bằng BCE + LAMB với augmentation rất mạnh, nên có thể khó tinh chỉnh ở LR thấp.
  - ConvNeXt-T có thêm lợi thế được tiền huấn luyện trên ImageNet-12k.

  Như vậy khoảng cách 15 điểm giữa ResNet-50 và ConvNeXt-T rất có thể gồm cả phần do công thức và trọng số tiền huấn luyện, chứ không chỉ do kiến trúc (GUIDE câu hỏi 1). Muốn tách hai phần này cần chạy thêm ResNet-50 với LR cao hơn, việc này chưa làm được vì ngân sách GPU.
- **FLOPs không dự đoán được độ trễ:**
  - EfficientNet-B0 có số GMAC chỉ bằng khoảng 1/10 ResNet-50 nhưng chạy chậm hơn ở batch 1 (10,8 ms so với 7,5 ms).
  - Swin-T có GMAC gần bằng ConvNeXt-T nhưng chậm hơn khoảng 2,3 lần.

**Chọn ConvNeXt-T để đi tiếp,** vì ba lý do:
- macro-F1 val cao nhất;
- độ trễ 8,1 ms, chỉ chậm hơn DeiT-S 1,5 ms nhưng nhanh hơn Swin-T 2,3 lần;
- ngân sách GPU chỉ đủ cho một backbone ở Bước 2.

## 4. Công thức huấn luyện (Bước 2)

ConvNeXt-T, mỗi run chỉ khác T00 đúng **một** yếu tố. Ablation chạy seed 0, còn T00 chạy 3 seed để ước lượng nhiễu. Số liệu ở sheet `Training`, biểu đồ ở `curves/T*_*.png` và `curves/summary_training.png`. Dự đoán viết trước khi chạy nằm trong notebook (ô markdown đầu Bước 2).

**Mốc T00:** macro-F1 val **0,9688 ± 0,0046** (seed 0/1/2 lần lượt là 0,9636 / 0,9712 / 0,9717).

| exp_id | Trục | Khác T00 | macro-F1 val | Δ so với mean T00 | ECE val | F1 Chinee | F1 Snake |
|---|---|---|---|---|---|---|---|
| T01 | A | đóng băng backbone | 0,8450 | −12,39 | 0,031 | 0,809 | 0,769 |
| T02 | A | train từ đầu | 0,3181 | −65,07 | 0,032 | 0,203 | 0,329 |
| T03 | B | + TrivialAugment | **0,9742** | +0,53 | 0,011 | 0,959 | 0,940 |
| T04 | B | + CutMix | 0,9683 | −0,05 | 0,011 | 0,958 | 0,925 |
| T05 | B | + lật dọc và xoay 90° | 0,9697 | +0,09 | 0,009 | 0,945 | 0,944 |
| T06 | C | label smoothing 0,1 | 0,9668 | −0,20 | **0,087** | 0,919 | 0,922 |
| T07 | C | focal γ = 2 | 0,9670 | −0,18 | **0,041** | 0,923 | 0,923 |
| T08 | C | CE trọng số class-balanced | 0,9661 | −0,27 | 0,010 | 0,946 | 0,913 |
| T09 | D | sampler cân bằng lớp | 0,9709 | +0,21 | 0,011 | 0,953 | 0,936 |
| T10 | F | EMA 0,999 | 0,9655 | −0,33 | 0,011 | 0,928 | 0,919 |
| T11 | B + D | TrivialAugment + sampler | 0,9686 | −0,02 | 0,011 | 0,950 | 0,918 |

**Yếu tố có tác dụng rõ:**
- **Trục A (khởi tạo)** là trục duy nhất tạo chênh lệch vượt xa nhiễu: đóng băng backbone mất 12,4 điểm, train từ đầu mất 65 điểm.
- Với khoảng 10 nghìn ảnh, 12 epoch và LR 1e-4, train từ đầu gần như chưa học được gì. Loss train chỉ giảm từ 1,70 xuống 1,20 (`curves/T02_scratch.png`).
- Đóng băng backbone vẫn còn thiếu khớp: val F1 vẫn tăng tới epoch 12, loss train dừng ở 0,355. Đặc trưng ImageNet chưa đủ để tách các loài cỏ.

**Không phân biệt được:** mọi thay đổi ở trục B, C, D, F nằm trong khoảng ±1,2 std quanh mean T00. Với 1 seed, không kết luận được thay đổi nào tốt hơn T00. TrivialAugment (+0,53, khoảng 1,2 std) và sampler cân bằng (+0,21) là hai thay đổi có Δ dương lớn nhất.

**Tác dụng phụ đo được:**
- Label smoothing làm ECE tăng khoảng 7 lần (từ 0,013 lên 0,087), focal làm ECE tăng khoảng 3 lần, trong khi F1 không đổi. Đây là tác dụng phụ về hiệu chuẩn mà slide trang 56–57 nhắc tới. Label smoothing kéo xác suất lớp đúng xuống, làm mô hình thiếu tự tin.
- CutMix có loss train cao (0,477 ở epoch 12) vì nhãn đã bị trộn. Accuracy train lúc này không còn ý nghĩa, nên chỉ dựa vào val.

**Tổ hợp (tham lam theo trục):** ghép hai yếu tố có Δ dương lớn nhất thành T11. T11 chỉ đạt 0,9686, thấp hơn cả T03 (0,9742) lẫn T09 (0,9709). Như vậy hai hiệu ứng **không cộng dồn**, nhưng vì chỉ có 1 seed nên cũng không kết luận được là chúng triệt tiêu nhau.

**Chọn cấu hình cho chung kết:** quy tắc chọn được viết trong code trước khi chạy: lấy argmax macro-F1 val seed 0 trong ba ứng viên T03, T09, T11. Quy tắc này chọn **T03**, đặt tên là F01. F01 chạy 3 seed được val **0,9746 ± 0,0012**, so với T00 0,9688 ± 0,0046, tức +0,58 điểm. Cả 3 seed F01 đều cao hơn mọi seed T00.

**Bác bỏ một phỏng đoán ở Bước 1:**
- Bước 1 phỏng đoán "nghẽn ở CPU đọc ảnh", dựa trên việc MobileNetV3 train chỉ nhanh hơn ResNet-50 khoảng 1,5 lần.
- Ở Bước 2, T01 (đóng băng) chỉ mất 17,5 s mỗi epoch, so với khoảng 48 s khi tinh chỉnh toàn bộ.
- Vậy với ConvNeXt-T và ảnh đã nạp sẵn vào RAM, thời gian bị chi phối bởi lan truyền ngược trên GPU, không phải bởi khâu đọc ảnh.

**Biểu đồ training (hiện tượng đáng chú ý):**
- T00 seed 0 có val loss nhỏ nhất ở epoch 9 rồi tăng nhẹ (từ 0,106 lên 0,110), trong khi loss train vẫn giảm tới 0,034 ở epoch 12. Đây là dấu hiệu quá khớp nhẹ ở cuối lịch LR.
- F01 (TrivialAugment) có loss train cao hơn (0,07) nhưng val loss thấp hơn (0,079). Augmentation đã chính quy hoá đúng như mong đợi.

## 5. Phương pháp suy luận (Bước 3)

Làm trên **val**, không huấn luyện lại. Mô hình là F01 seed 0; các phương pháp một mô hình còn được chạy lại cho seed 1 và 2 để lấy trung bình 3 seed. Số liệu ở sheet `Inference` và `Latency`, biểu đồ đánh đổi ở `curves/buoc3_tradeoff.png`.

**Cách đo độ trễ** (`code/benchmark.py`):
- Tesla T4 trên Kaggle, torch 2.11.0+cu128, ảnh đầu vào ngẫu nhiên đã nằm sẵn trên GPU.
- **Không tính** tiền xử lý.
- Warmup 10 lần, gọi `torch.cuda.synchronize()` trước và sau mỗi lần đo, đo 100 lần, báo cáo p50/p95/p99.
- TTA với K view được đo như một batch gồm K ảnh.

| exp_id | K | ảnh vào | macro-F1 val (seed 0) | mean 3 seed ± std | ECE val | p50 / p95 / p99 b1 (ms) | chi phí so với I00 |
|---|---|---|---|---|---|---|---|
| I00 1 view (mốc) | 1 | 224 | 0,9742 | 0,9746 ± 0,0012 | 0,0113 | 7,39 / 7,88 / 8,29 | 1,00 |
| I01 lật ngang, gộp prob | 2 | 224 | 0,9741 | 0,9744 ± 0,0007 | 0,0090 | 7,34 / 8,98 / 9,01 | 0,99 |
| I01 lật ngang, gộp logit | 2 | 224 | 0,9741 | 0,9746 ± 0,0010 | 0,0091 | như trên | 0,99 |
| I02 5 crop, gộp prob | 5 | 5 × 224 từ 256 | 0,9737 | 0,9752 ± 0,0013 | 0,0074 | 10,16 / 12,22 / 12,23 | 1,37 |
| I02 5 crop, gộp logit | 5 | 5 × 224 từ 256 | 0,9737 | 0,9750 ± 0,0012 | 0,0092 | như trên | 1,37 |
| I04 độ phân giải 256 | 1 | 256 | 0,9737 | 0,9753 ± 0,0014 | 0,0104 | 7,43 / 8,10 / 8,93 | 1,00 |
| **I04 độ phân giải 288** | 1 | 288 | **0,9784** | **0,9785 ± 0,0008** | 0,0063 | 7,42 / 7,89 / 8,24 | 1,00 |
| I04 độ phân giải 320 | 1 | 320 | 0,9774 | 0,9767 ± 0,0015 | 0,0089 | 7,43 / 8,98 / 9,00 | 1,00 |
| I05 ensemble 3 seed | 3 mô hình | 224 | 0,9768 | — | 0,0074 | 22,85 / 25,35 / 36,60 | 3,09 |
| I06 soup 3 seed | 1 | 224 | 0,9733 | — | **0,0362** | 7,78 / 8,30 / 9,08 | 1,05 |
| I07 temperature scaling | 1 | 224 | 0,9742 | — | 0,0079 | 7,39 / 7,88 / 8,29 | 1,00 |
| I08 FP32 | 1 | 224 | 0,9742 | — | 0,0118 | 5,50 / 8,30 / 8,38 | 0,74 |
| I08 FP16 thuần | 1 | 224 | 0,9742 | — | 0,0116 | 5,31 / 5,63 / 6,02 | 0,72 |

**Đo được:**
- **TTA lật và 5 crop không giúp:** mọi chênh lệch đều ≤ 0,06 điểm, nhỏ hơn std giữa các seed. Gộp prob và gộp logit cũng không phân biệt được. Phỏng đoán: TrivialAugment lúc train đã làm mô hình gần như bất biến với các phép này. TTA còn tốn thêm 14–55 % độ trễ p95.
- **Độ phân giải 288 là cải thiện duy nhất ổn định trên cả 3 seed:** tăng +0,39 điểm trên trung bình 3 seed (0,9785 so với 0,9746), khoảng 3 lần std. Kết quả này khớp với hiệu ứng FixRes (slide trang 68): RandomResizedCrop lúc train làm vật thể trông to hơn lúc test, nên tăng độ phân giải test sẽ bù lại. Mức tăng không đơn điệu: ở 320 thì giảm lại còn 0,9767.
- **Ở batch 1, ảnh 288 không chậm hơn ảnh 224** (p50 7,42 so với 7,39 ms). Phỏng đoán: ở batch 1 trên T4, độ trễ bị chi phối bởi chi phí khởi chạy kernel chứ không phải FLOPs. Chưa đo ảnh 288 ở batch 32.
- **Ensemble 3 seed** tăng +0,26 điểm (chỉ đo trên một nhóm 3 mô hình), nhưng chậm gấp 3,1 lần. Nó vừa kém chính xác hơn vừa chậm hơn I04_res288.
- **Model soup** (trung bình trọng số 3 seed) không tăng F1 (−0,09) và làm ECE tệ hơn khoảng 3 lần. Phỏng đoán: 3 lần tinh chỉnh khác seed không nằm trong cùng một vùng lòng chảo của hàm mất mát, nên trung bình trọng số làm hỏng hiệu chuẩn.
- **Hiệu chuẩn:** temperature scaling khớp trên val cho T = 1,40 > 1, tức mô hình hơi quá tự tin. ECE giảm từ 0,0113 xuống 0,0079. Khi khớp T trên một nửa val và đo trên nửa còn lại (cross-fit), ECE là 0,0044. Accuracy không đổi, đúng như lý thuyết.
- **dtype:**
  - Ở batch 1, AMP **chậm hơn** FP32 (p50 7,39 so với 5,50 ms), đúng cảnh báo ở slide trang 73.
  - Ở batch 32, AMP nhanh hơn FP32 khoảng 2,7 lần (651 so với 242 ảnh/s), còn FP16 thuần đạt 807 ảnh/s.
  - FP16 thuần nhanh nhất ở batch 1 (p95 5,63 ms) mà F1 không đổi; logit lệch tối đa 0,036 so với FP32.
- **Gộp BN:** ConvNeXt dùng LayerNorm nên không có cặp nào để gộp. Phần minh hoạ trên ResNet-50 (`logs/bn_fusion_check.json`):
  - gộp được 53 cặp conv–BN;
  - logit lệch tối đa 2,2e-7 (FP32), tức gộp chính xác;
  - p50 giảm từ 5,68 xuống 4,70 ms, p95 giảm từ 8,38 xuống 4,85 ms.

**Ngoại tuyến hay thời gian thực:**
- Kết quả trên DeepWeeds ủng hộ kết luận của slide: TTA và ensemble chỉ hợp khi chạy ngoại tuyến, và ở đây chúng còn không chính xác hơn.
- Trên robot nên dùng những thứ không tốn thêm độ trễ: độ phân giải đã dò, temperature scaling, FP16 và gộp BN (khi kiến trúc có BN).

**Quy tắc chốt cho chung kết** (viết trước trong `code/buoc3.py`): lấy phương pháp một mô hình có macro-F1 val trung bình 3 seed cao nhất; nếu hoà thì chọn phương pháp ít view hơn; luôn cộng thêm temperature scaling. Quy tắc này chọn **I04_res288 + TS**.

## 6. Cấu hình tốt nhất và kết quả test (Bước 4)

**Mô tả để tái lập F01:**
- ConvNeXt-T `convnext_tiny.in12k_ft_in1k`, tinh chỉnh toàn bộ, ảnh train 224.
- RandomResizedCrop + lật ngang + TrivialAugmentWide.
- AdamW (LR backbone 1e-4, LR head 1e-3, wd 0,05, không áp cho norm/bias), warmup 1 epoch rồi cosine.
- Batch 64, 12 epoch, AMP, CE; chọn checkpoint theo macro-F1 val tốt nhất. Seed 0, 1, 2.
- Suy luận: resize ảnh về 288 cho 1 view, AMP, chia logit cho T khớp trên val của từng seed (T = 1,284 / 1,216 / 1,245).
- Code: `Config(aug="trivial")` trong `code/train.py`, `buoc4.run_test` trong `code/buoc4.py`. Notebook: `code/kaggle_buoc2b.ipynb` và `code/kaggle_buoc4.ipynb`.

**Test chạy đúng một lần mỗi seed:**
- `buoc4.run_test` tự dừng nếu thư mục `predictions/` đã có bất kỳ file `*_test.csv` nào.
- Mọi lựa chọn (backbone, công thức, phương pháp suy luận, T) đều đã chốt trên val trước khi chạy test.
- Mốc T00 + I00 dùng cùng 3 seed, suy luận 1 view ở 224 và không có TS.

**Kết quả test** (`eval.py score`, sheet `Final`):

| Cấu hình | macro-F1 val | macro-F1 test | top-1 test | balanced acc. test | ECE test |
|---|---|---|---|---|---|
| **F01** (mean ± std, 3 seed) | 0,9785 ± 0,0008 | **0,9789 ± 0,0012** | **0,9829 ± 0,0010** | 0,9738 ± 0,0018 | 0,0047 ± 0,0020 |
| F01 chưa TS | — | 0,9789 ± 0,0012 | 0,9829 ± 0,0010 | 0,9738 ± 0,0018 | 0,0081 ± 0,0010 |
| T00 + I00 (mốc) | 0,9688 ± 0,0046 | 0,9709 ± 0,0012 | 0,9766 ± 0,0016 | 0,9739 ± 0,0026 | 0,0099 ± 0,0006 |

- **Cải thiện so với mốc:** Δ macro-F1 test = +0,0080, trong khi std lớn hơn trong hai nhóm là 0,0012. Như vậy Δ ≈ 6,7 lần std, **vượt nhiễu**. Top-1 tăng +0,63 điểm.
- Balanced accuracy không đổi (0,9738 so với 0,9739), vì F01 đánh đổi recall giữa các lớp, xem mục F1 từng lớp bên dưới.
- **Val và test lệch rất ít:** macro-F1 val 0,9785, test 0,9789, chênh 0,0004. Lo ngại thiên lệch khi chọn res288 trong 8 phương pháp trên val không thể hiện trên test.
- **Temperature scaling trên test:** ECE giảm từ 0,0081 xuống 0,0047, accuracy giữ nguyên.
- **`eval.py grade`** (`logs/eval/grade.txt`):

  | Ý | Điểm | Chi tiết |
  |---|---|---|
  | I1 | 7/7 | top-1 98,29 % |
  | I2 | 4/5 | Δ = 0,0080 > s nhưng < 0,01 |
  | I3 | 4/4 | Chinee 96,8 %, Snake 94,6 %; mốc bài báo 88,5 % và 88,8 % |
  | I4a | 1/1 | ECE sau TS nhỏ hơn trước |
  | I4b | 1/1 | val và test chênh 0,0004 |
  | I5 | 2/2 | p95 7,89 ms |

  Tổng **19/20**. So với bài báo (ResNet-50 đạt 95,7 %, khoảng 100 epoch, 5 fold) thì số của bài này cao hơn, nhưng định nghĩa accuracy và điều kiện chạy khác nhau (README mục 2.3), nên đây chỉ là tham khảo.

**F1 từng lớp trên test** (sheet `PerClass`, mean ± std 3 seed):

| Lớp | F01 precision | F01 recall | F01 F1 | mốc F1 | mốc recall |
|---|---|---|---|---|---|
| Chinee apple | 0,975 | **0,968** | 0,971 ± 0,002 | 0,946 | 0,926 |
| Lantana | 0,997 | 0,970 | 0,983 | 0,977 | 0,991 |
| Parkinsonia | 0,976 | 0,994 | 0,985 | 0,978 | 0,982 |
| Parthenium | 1,000 | 0,967 | 0,983 | 0,983 | 0,974 |
| Prickly acacia | 0,968 | 0,958 | 0,963 | 0,956 | 0,973 |
| Rubber vine | 0,992 | 0,979 | 0,985 | 0,985 | 0,984 |
| Siam weed | 0,994 | 0,989 | 0,991 | 0,974 | 0,994 |
| Snake weed | 0,975 | **0,946** | 0,960 ± 0,003 | 0,955 | 0,961 |
| Negative | 0,982 | 0,994 | 0,988 | 0,984 | 0,980 |

**Ma trận nhầm lẫn** (`curves/confusion_test_F01.png`, `curves/confusion_test_T00.png`; cộng 3 seed, tức 10 521 lượt dự đoán):
- Chinee apple ↔ Snake weed (cặp khó nhất trong bài báo): F01 có 7 + 5 = 12 lượt nhầm, mốc có 15 + 7 = 22. Recall Chinee apple tăng từ 92,6 % lên 96,8 %.
- F01 **dịch lỗi sang phía `Negative`:**
  - Ảnh `Negative` bị đoán thành một loài cỏ giảm từ 108 xuống 35 lượt, nên precision của Siam weed, Lantana và Prickly acacia tăng.
  - Ngược lại, ảnh có cỏ bị đoán là `Negative` tăng từ 69 lên 102 lượt. Nhiều nhất là Snake weed → Negative (27 lượt, mốc chỉ 10) và Prickly acacia → Negative (20 lượt, mốc 6).
  - Vì vậy recall Snake weed giảm nhẹ từ 96,1 % xuống 94,6 %, và đây là lớp có F1 thấp nhất của F01.

**Phân tích lỗi bằng ảnh** (`curves/errors_F01_seed0.png`, ô B5.2): ảnh test F01 seed 0 bị đoán sai ở 4 cặp. Ở seed 0, Chinee → Snake có 1 ảnh, Snake → Chinee 2 ảnh, Snake → Negative 11 ảnh, Chinee → Negative 5 ảnh.
- **Giả thuyết (chưa kiểm chứng):** DeepWeeds chụp cỏ trong môi trường tự nhiên, nhiều ảnh chỉ có một phần nhỏ cây đích lẫn trong cỏ nền.
  - TrivialAugment (crop, đổi màu mạnh) cùng việc test ở 288 khiến mô hình đòi bằng chứng rõ hơn trước khi gọi tên một loài. Kết quả là các ảnh chỉ có ít lá Snake weed hoặc Prickly acacia bị đẩy về `Negative`.
  - Muốn kiểm chứng, có thể dời ngưỡng quyết định cho `Negative` trên val, hoặc đo recall theo tỉ lệ diện tích cây trong ảnh. Bài này chưa làm vì cần gắn nhãn thêm.

## 7. Kết luận và khuyến nghị

- **Cấu hình nào tốt nhất?** F01 = ConvNeXt-T + TrivialAugment + test ở 288 + temperature scaling. Kết quả test: macro-F1 0,9789 ± 0,0012, top-1 98,29 ± 0,10 %. F01 hơn mốc T00 + I00 **+0,0080 macro-F1**, khoảng 6,7 lần std, nên vượt nhiễu.
- **Yếu tố nào đóng góp nhiều nhất?**
  - **Backbone cùng trọng số tiền huấn luyện** đóng góp nhiều nhất: trên val ở cùng công thức, chênh 15–24 điểm (B02 so với B01, B04, B05), có thể lẫn cả phần do công thức như đã nói ở mục 3.
  - Trong **công thức huấn luyện**, khởi tạo pretrained và tinh chỉnh toàn bộ là bắt buộc (đóng băng mất 12 điểm, train từ đầu mất 65 điểm). Các tinh chỉnh còn lại chỉ thêm được tối đa khoảng +0,6 điểm (TrivialAugment, xác nhận qua 3 seed).
  - **Suy luận** thêm khoảng +0,4 điểm nhờ dò độ phân giải. TTA, ensemble và soup không giúp.
  - Lưu ý: các phần này đo trên val với những mốc khác nhau. Trên test chỉ đo tổng hiệu ứng (+0,8 điểm), chưa tách riêng phần huấn luyện và phần suy luận.
- **Triển khai trên robot với ngân sách 30–100 ms mỗi khung:**
  - Chọn F01 với 1 view ở 288. Trên T4, p95 là 7,9 ms (AMP), không tính tiền xử lý.
  - Nên chuyển sang FP16 thuần: p95 ở 224 là 5,6 ms mà F1 không đổi. Cần đo lại cho 288.
  - Giữ temperature scaling để xác suất đáng tin khi đặt ngưỡng phun thuốc.
  - Không dùng TTA hay ensemble: chậm gấp 1,4–3,1 lần mà không chính xác hơn.
  - Trên thiết bị nhúng (ví dụ Jetson), độ trễ sẽ lớn hơn T4 nhiều (bài báo đo ResNet-50 trên TX2 mất 53–180 ms), nên phải đo lại trên phần cứng thật trước khi chốt. ConvNeXt-T không có BN để gộp. Nếu thiết bị quá chậm, phương án dự phòng là một mạng nhỏ hơn được huấn luyện lại với LR cao hơn.

## 8. Hạn chế và việc tiếp theo

- **Số seed:** Bước 1 và các ablation Bước 2 chỉ có 1 seed. Nhiều kết luận "không phân biệt được" có thể đổi nếu chạy thêm seed. Chỉ T00 và F01 có 3 seed.
- **Một fold:** chỉ dùng fold 0. Bài báo dùng 5 fold.
- **Chia ngẫu nhiên, không theo địa điểm:** ảnh cùng một địa điểm và một buổi chụp có thể nằm ở cả train lẫn test, nên điểm test có thể lạc quan so với khi gặp địa điểm, mùa hoặc ánh sáng mới.
- **Rủi ro lệch phân phối:** chưa đánh giá trên ảnh bị nhiễu, mờ hay thiếu sáng. ECE thấp trên test không bảo đảm hiệu chuẩn tốt ngoài miền dữ liệu.
- **Ngân sách GPU:**
  - 12 epoch thay vì khoảng 100 epoch như bài báo;
  - Bước 2 chỉ trên một backbone;
  - chưa thử trục E (LR/optimizer) và trục G (độ phân giải, số epoch lúc train);
  - chưa kiểm chứng giả thuyết underfit của các CNN dùng BN.
- **Nền tảng:** Bước 1 chạy trên Colab, Bước 2–4 trên Kaggle. Cùng cấu hình có thể lệch khoảng 0,3 điểm giữa hai nơi (mục 2), nên bảng Backbones không so trực tiếp được với bảng Training ở mức 0,3 điểm.
- **Độ trễ:** đo trên T4, không tính tiền xử lý. Phép đo ở Bước 1 chỉ là sơ bộ (30 lần).
- **Việc tiếp theo:**
  - chạy thêm seed cho T03 và T09 để kiểm tra hai yếu tố này;
  - thử LR lớn hơn cho ResNet-50 và EfficientNet;
  - huấn luyện ở 256–288 (trục G);
  - đánh giá trên ảnh bị làm nhiễu;
  - thử 5 fold.

## 9. Phụ lục

- **Danh sách `exp_id`** (cấu hình đầy đủ trong `logs/<exp_id>_seed<k>/config.json`):

  | Nhóm | exp_id | Ảnh biểu đồ |
  |---|---|---|
  | Backbone | B01–B06 | `curves/B0*_*.png` |
  | Công thức huấn luyện | T00 (seed 0, 1, 2), T01–T11 | `curves/T*_*.png` |
  | Suy luận | I00–I08 | sheet `Inference` |
  | Chung kết | F01 (seed 0, 1, 2) | `curves/F01*_trivial.png` |

- **Notebook:**
  - `code/lab_day2.ipynb`: Colab, chạy từ trên xuống. Bước 0–1 chạy trực tiếp trên Colab. Bước 2–4 chỉ gộp kết quả từ Kaggle và đọc lại, không train lại.
  - `code/kaggle_buoc2.ipynb`: T00 và T01–T10.
  - `code/kaggle_buoc2b.ipynb`: T11 và F01.
  - `code/kaggle_buoc3.ipynb`: suy luận và độ trễ.
  - `code/kaggle_buoc4.ipynb`: chạy test và chấm điểm.
- **File dự đoán:**
  - `predictions/F01_seed{0,1,2}_test.csv`
  - `predictions/F01_uncal_seed{0,1,2}_test.csv`
  - `predictions/F01_seed{0,1,2}_val.csv`
  - `predictions/T00_seed{0,1,2}_test.csv`
