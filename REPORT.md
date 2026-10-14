Họ tên: Doan Phuong Linh
`dataset_version`: d10-21e4070301

> Mã: `lab/` (labels, vlm, reward, viz, report_table) · test của tôi: `lab_tests/` · notebook: `lab_day10.ipynb`.
> Bảng dưới sinh bằng `python -m lab.report_table` từ `report/evidence/*.json`.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | – | – | 0.111 | 0.084–0.143 | 0.86 | 0.162 | 0.029 | 0.243 |
| E1 | LoRA LLM | 33.030.144 | 6.55 | 49.3 | 0.252 | 0.213–0.291 | 1.29 | 0.205 | 0.167 | 0.518 |
| E2 | + projector | 142.135.296 | 8.25 | 49.9 | 0.334 | 0.294–0.375 | 1.11 | 0.159 | 0.281 | 0.599 |
| E3 | + LoRA ViT | 148.426.752 | 10.36 | 55.0 | 0.452 | 0.411–0.495 | 0.90 | 0.300 | 0.377 | 0.709 |
| E4 | E3 với `coord_scale=pixel` (E3 = `norm1000`) | 148.426.752 | 10.34 | 54.8 | 0.394 | 0.352–0.435 | 1.05 | 0.239 | 0.311 | 0.699 |

Ảnh minh hoạ: `report/figures/gt_samples.jpg` (nhãn), `report/figures/<best>_grid.jpg` và `<best>_{dung,sai}_{1,2,3}.jpg` (xanh lá = box đúng, màu = box dự đoán), `report/figures/loss.png`.

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?

Đếm trên cây model thật (`Qwen3VLForConditionalGeneration` dựng từ config trên thiết bị meta, test `lab_tests/test_vlm_cpu.py`):

| Khối | Tên module | Tham số |
|---|---|---|
| ViT (24 block + patch embed + pos embed) | `model.visual.blocks.*`, … | ≈ 306,2M |
| Projector | `model.visual.merger` (27,27M) + `model.visual.deepstack_merger_list` (3 merger, 81,83M; lấy đặc trưng ở lớp 5, 11, 17 của ViT) | ≈ 109,1M |
| LLM (36 lớp + embedding) | `model.language_model.*` | ≈ 4.022,5M (embedding 389M dùng chung với `lm_head`) |
| Tổng (không đếm trùng lm_head) | | ≈ 4.437,8M |

- **E1**: chỉ thêm LoRA r=16 vào 252 module của LLM (`q,k,v,o,gate,up,down` × 36 lớp) → 33.030.144 tham số huấn luyện; ViT và projector đóng băng.
- **E2**: E1 + mở đầy đủ projector → 142.135.296 (hơn E1 đúng 109.105.152 = số tham số projector).
- **E3**: E2 + LoRA trên 96 module ViT (`attn.qkv, attn.proj, mlp.linear_fc1, mlp.linear_fc2` × 24 block) → 348 module LoRA, 148.426.752 tham số.
- Trên Colab, `meta.trainable_params` của tôi: E1 = 33.030.144, E2 = 142.135.296, E3 = 148.426.752 (khớp số đếm ở trên).

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
`parse_valid_rate` của E0 = 0.958 (298/311 ảnh parse được, 13 ảnh hỏng). Đếm lại trên `report/preds/e0.jsonl` (311 ảnh, không có ảnh nào thực sự rỗng):

| Lỗi | Số lượng | Ghi chú |
|---|---|---|
| Output không parse được (bị cắt giữa chừng) | 13 ảnh | model lặp liên tục box `crack` nhỏ xếp nối nhau cho đến hết token, JSON không đóng |
| Box `crack` trải gần hết chiều ngang (rộng ≥ 800/1000) | 62 box | crack bị vẽ thành một dải phủ cả mặt đường, IoU thấp với box thật mảnh; kéo theo 182 FP của lớp crack |
| Trả `[]` dù ảnh luôn có ≥ 1 box | 31 ảnh | bỏ sót hoàn toàn, góp vào 533 FN |
| Box trùng y hệt trong cùng ảnh | 9 box | |

Hai lỗi phổ biến nhất: (1) lặp box crack đến mức output bị cắt, (2) box crack quá to. Ví dụ output thô (`e0_sai_2`, ảnh có 1 crack + 4 pothole, model trả 3 box crack giống hệt nhau phủ cả khung, TP = 0, FP = 3, FN = 5):

```
[{"label": "crack", "bbox_2d": [0, 375, 998, 750]}, {"label": "crack", "bbox_2d": [0, 375, 998, 750]}, {"label": "crack", "bbox_2d": [0, 375, 998, 750]}]
```

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Qwen3-VL được tiền huấn luyện để grounding trên thang 0–1000 tương đối với kích thước ảnh, nên toạ độ không phụ thuộc độ phân giải (ảnh còn bị resize về 640×352 trước khi vào ViT, nên "pixel" của model khác pixel gốc). Một box dự đoán đúng trên thang 0–1000 mà bị chấm như pixel sẽ bị quy đổi lần nữa: trục x chia cho 640 rồi nhân 1000 (×1,56), trục y ×2,78 (có thể vượt 1000), nên box bị kéo dãn và lệch về phía phải/dưới. Ngược lại, nhãn pixel chấm như 0–1000 thì box co về góc trên-trái. Với box nhỏ (diện tích trung vị ≈ 0,7% khung hình), IoU gần như luôn < 0,5 → TP ≈ 0.

Kiểm chứng: (1) test `test_wrong_scale_breaks_f1`: nhãn pixel chấm như 0–1000 cho F1 = 0, còn đáp án huấn luyện chấm đúng `--coord` cho F1 = 1 với cả hai thang (`test_ground_truth_answer_scores_perfect`); (2) chấm lại dự đoán E1 bằng `--coord pixel` (`report/extra/e1_scored_as_pixel.json`): F1 từ 0.252 xuống 0.000 (TP = 0, FP = 543, FN = 583), đúng như dự đoán: dù model đã học đúng thang 0–1000, cách chấm sai thang làm mọi box trượt ngưỡng IoU.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Một ví dụ huấn luyện của tôi dài 425 token: 220 token ảnh (`<|image_pad|>`, lưới 22×40 patch 16 px, gộp 2×2), 146 token prompt + template, và chỉ ≈ 59 token đáp án JSON + `<|im_end|>\n` (ảnh có 2 box). Tôi gán `labels = -100` cho toàn bộ tiền tố prompt (đã kiểm tra prompt là tiền tố chính xác của chuỗi đầy đủ), nên loss chỉ tính trên đáp án.

Nếu tính cả prompt: (1) ≈ 86% số token có loss sẽ là prompt cố định, gradient bị chi phối bởi việc học thuộc câu lệnh vốn giống hệt nhau ở mọi mẫu, nên tín hiệu học toạ độ bị pha loãng khoảng 7 lần; (2) token ảnh là placeholder, embedding thật được thay bằng đặc trưng ViT, nên "dự đoán" `<|image_pad|>` kế tiếp là bài toán vô nghĩa (luôn đúng, loss ≈ 0, hoặc làm lệch đầu ra); (3) model có thể học sinh lại prompt thay vì trả lời. Lúc suy luận ta không bao giờ yêu cầu model sinh prompt, nên đó là phép tính bỏ phí. Kiểm tra: giải mã các token có `labels != -100` cho ra đúng chuỗi JSON đáp án + `<|im_end|>\n` (test `test_image_tokens_and_label_mask`).

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

Lượng tử NF4 thay trọng số `Linear` bằng `Linear4bit`: trọng số lưu dạng số nguyên 4-bit đóng gói (uint8) + hằng số scale theo khối, không phải tensor số thực liên tục. Gradient không cập nhật được trực tiếp lên giá trị 4-bit (không có dtype khả vi, và mỗi bước lại phải lượng tử lại, làm mất cập nhật nhỏ), nên trên lớp 4-bit chỉ có thể gắn LoRA, không thể huấn luyện đầy đủ. Xử lý: không lượng tử ViT (gồm `visual.merger` và `visual.deepstack_merger_list`) để giữ ở fp16. Bẫy tôi gặp thật: lần chạy đầu dùng `llm_int8_skip_modules=["visual", "lm_head"]` như pseudo code, ô E0 in ra ViT 209,8M, merger 13,64M, deepstack 40,94M, đúng một nửa số thật (415,3M / 27,27M / 81,83M), vì `Linear4bit` đóng gói 2 giá trị 4-bit vào một byte. Nguyên nhân: transformers 5.19 khớp tên bằng `re.match` (từ đầu tên), nên `"visual"` không khớp `model.visual.merger.linear_fc1` và toàn bộ ViT đã bị lượng tử. Tôi sửa thành `["model.visual", "lm_head"]`, thêm `check_vit_not_quantized()` để dừng ngay nếu còn `Linear4bit` trong ViT, và test `test_skip_modules_keep_whole_vit_unquantized`. Sau khi sửa, ô E0 in ra ViT = 415.347.712 tham số (merger 27.271.680, deepstack 81.833.472), dtype float16, còn `q_proj` của LLM là `Linear4bit`. Kết quả E0 lần đầu được bỏ, E0–E4 chạy lại từ đầu.

Projector phải ở float32 vì (1) cập nhật Adam với lr 2e-4 thường nhỏ hơn bước lượng tử của fp16 (≈ 1e-3 tương đối) nên cộng vào trọng số fp16 sẽ bị làm tròn mất; (2) `GradScaler` không cho unscale gradient fp16 (`Attempting to unscale FP16 gradients`). Vì vậy tôi giữ bản trọng số chính fp32 (`p.data = p.data.float()`), còn phép nhân vẫn chạy fp16 nhờ `autocast`. Lúc suy luận, trọng số projector đã lưu (`extra_trainable.pt`) được chép lại vào model fp16.

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?

Ảnh là frame cắt từ video dashcam: hai frame liền kề gần như trùng nhau (cùng ổ gà, cùng vết nứt, chỉ dịch vài pixel). Chia ngẫu nhiên theo ảnh thì gần như mọi ảnh test đều có "anh em sinh đôi" trong train, nên model chỉ cần nhớ cảnh là được điểm cao. F1 đo được khi đó là khả năng nhớ, không phải khả năng khái quát sang đoạn đường mới, nên sẽ cao hơn F1 thật khi triển khai. `prepare_data.py` gom theo phiên (ngày + giờ trong tên file) và gán cả phiên vào một split; ô dữ liệu trong notebook xác nhận không phiên nào xuất hiện ở hai split.

| Split | Ảnh | Phiên | pothole | crack | manhole |
|---|---|---|---|---|---|
| train | 1.422 | 13 | 1.074 | 1.750 | 691 |
| val | 276 | 2 | 66 | 470 | 103 |
| test | 311 | 3 | 121 | 299 | 163 |

Không ảnh nào rỗng (mọi ảnh có ít nhất 1 box). Test chỉ gồm 3 phiên quay, nên 311 ảnh test thực chất là 3 đoạn đường, ít độc lập hơn con số 311 gợi ý. Hệ quả: val chỉ có 66 pothole nên F1 pothole trên val dao động mạnh; tôi dùng 40 ảnh val để gỡ lỗi và chỉ chấm test một lần cho mỗi setup.

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
| | F1 (CI95) | VRAM đỉnh | Phút | Tham số huấn luyện |
|---|---|---|---|---|
| E1 | 0.252 (0.213–0.291) | 6.55 GB | 49.3 | 33,0M |
| E2 | 0.334 (0.294–0.375) | 8.25 GB | 49.9 | 142,1M |
| E3 | 0.452 (0.411–0.495) | 10.36 GB | 55.0 | 148,4M |

Mở projector (E1→E2) tăng F1 +0.082 (+33% tương đối), tốn thêm 1,70 GB VRAM và chỉ 0,6 phút. Thêm LoRA ViT (E2→E3) tăng tiếp +0.118 (+35%), tốn thêm 2,11 GB và 5,1 phút, chỉ thêm 6,3M tham số (4,4%) nhưng recall tăng 0.319 → 0.432 và FP/ảnh giảm 1.11 → 0.90. Cả hai bước đều đáng; E3 cho F1 gấp 1,8 lần E1 với VRAM 10,36 GB vẫn nằm trong T4 16 GB. Đáng nhất là E3, và bước ViT có lợi suất cao nhất: ảnh dashcam nhỏ, ViT gốc không tách tốt ổ gà/vết nứt nên cần điều chỉnh ở phía thị giác, không chỉ ở LLM. Chi phí thời gian gần như phẳng (49→55 phút) nên VRAM mới là giá phải trả.


### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
Độ rộng CI95 (hi − lo): E0 0.059, E1 0.078, E2 0.081, E3 0.084, E4-pixel 0.083, tức khoảng ±0.04 quanh F1. Với 311 ảnh nhưng chỉ 3 phiên quay, các ảnh trong cùng phiên tương quan, nên CI thật có thể rộng hơn con số bootstrap theo ảnh (nếu bootstrap theo ảnh).

Đứng vững: E0 ≪ E1 (0.143 < 0.213), E2 < E3 (0.375 < 0.411) và E1 < E3, các CI không chồng nhau. Mỏng manh: E1 vs E2, khe hở giữa hai CI chỉ 0.003 (0.291 vs 0.294), nên chỉ nên nói "E2 nhiều khả năng tốt hơn E1", không nên nói chắc chắn. Mức chênh 0.08 tương đương một nửa độ rộng CI, vì vậy những chênh lệch dưới ~0.05 giữa hai setup bất kỳ trên bộ test này không phân biệt được với nhiễu.


### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
F1 của E3 (setup tốt nhất): manhole 0.709, crack 0.377, pothole 0.300. Pothole kém nhất; nó cũng là lớp duy nhất giảm ở E2 (0.205 → 0.159) trước khi tăng ở E3, và thấp hơn crack dù có số box train nhiều hơn manhole (1.074 so với 691), nên không thể giải thích chỉ bằng số mẫu. Giả thuyết của tôi (chưa kiểm chứng bằng thống kê kích thước box theo lớp): ổ gà là vùng không có hình dạng cố định, biên mờ, dễ lẫn với vá đường/bóng/vết nứt dày, và box nhỏ nên lệch vài pixel đã làm IoU < 0,5; còn nắp cống có hình tròn/vuông đặc trưng nên model dễ định vị. Crack (0.377) thuộc loại vật thể mảnh, dài, box có độ phủ thấp nên IoU rất nhạy. Ảnh minh hoạ: `report/figures/e3_grid.jpg`, `e3_dung_1..3.jpg`, `e3_sai_1..3.jpg` (đối chiếu với mô tả lớp pothole/crack ở trên).


### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
Biến E4: thang toạ độ nhãn/chấm `coord_scale`, giữ nguyên mọi thứ khác của E3. `norm1000` (chính là E3): F1 0.452 (0.411–0.495). `pixel`: F1 0.394 (0.352–0.435), parse_valid 0.997 (có 1 lỗi cú pháp, còn norm1000 là 1.000), P 0.405 / R 0.383 so với 0.473 / 0.432.

Chênh lệch −0.058 theo hướng có lợi cho norm1000 như lý thuyết dự đoán (Câu 3). Tuy nhiên hai CI chồng nhau trên đoạn 0.411–0.435, nên chỉ với bộ test này tôi không khẳng định được chắc chắn. Mức tin cậy: trung bình, vì hướng chênh nhất quán ở cả ba lớp (pothole −0.061, crack −0.066, manhole −0.010; lớp manhole gần như bằng nhau) và ở cả P lẫn R, nhưng mới chạy một seed và một lần chia test. Muốn kết luận chắc cần bootstrap hiệu F1 ghép cặp theo ảnh hoặc chạy lại nhiều seed.


## Nhìn lại

Lỗi lượng tử hoá ViT: do tên module bị khớp sai nên toàn bộ ViT bị nén xuống 4-bit. Ban đầu mình không nhận ra điều này và phải chạy lại từ E0 đến E4.

E0 trả về box crack phủ kín cả khung ảnh. F1 chỉ cho thấy điểm số thấp, nhưng không phân biệt được đó là lỗi vị trí hay lỗi lặp box.

Test chỉ gồm 3 phiên quay, nên khoảng tin cậy có thể lạc quan hơn so với thực tế.
