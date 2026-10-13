Họ tên: Doan Phuong Linh
`dataset_version`: d10-21e4070301

> Mã: `lab/` (labels, vlm, reward, viz, report_table) · test của tôi: `lab_tests/` · notebook: `lab_day10.ipynb`.
> Bảng dưới sinh bằng `python -m lab.report_table` từ `report/evidence/*.json`.

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | <điền> | – | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> |
| E1 | LoRA LLM | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> |
| E2 | + projector | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> |
| E3 | + LoRA ViT | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> |
| E4 | <điền: biến = giá trị> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> | <điền> |

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
- Trên Colab, `meta.trainable_params` của tôi: E1 = <điền>, E2 = <điền>, E3 = <điền>.

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
<điền: parse_valid_rate của E0, bảng đếm lỗi từ ô "Câu 2" trong notebook, và dán 1 output thô>

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?

Qwen3-VL được tiền huấn luyện để grounding trên thang 0–1000 tương đối với kích thước ảnh, nên toạ độ không phụ thuộc độ phân giải (ảnh còn bị resize về 640×352 trước khi vào ViT, nên "pixel" của model khác pixel gốc). Một box dự đoán đúng trên thang 0–1000 mà bị chấm như pixel sẽ bị quy đổi lần nữa: trục x chia cho 640 rồi nhân 1000 (×1,56), trục y ×2,78 (có thể vượt 1000), nên box bị kéo dãn và lệch về phía phải/dưới. Ngược lại, nhãn pixel chấm như 0–1000 thì box co về góc trên-trái. Với box nhỏ (diện tích trung vị ≈ 0,7% khung hình), IoU gần như luôn < 0,5 → TP ≈ 0.

Kiểm chứng: (1) test `test_wrong_scale_breaks_f1`: nhãn pixel chấm như 0–1000 cho F1 = 0, còn đáp án huấn luyện chấm đúng `--coord` cho F1 = 1 với cả hai thang (`test_ground_truth_answer_scores_perfect`); (2) chấm lại dự đoán E1 bằng `--coord pixel` (`report/extra/e1_scored_as_pixel.json`): F1 từ <điền> xuống <điền>.

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?

Một ví dụ huấn luyện của tôi dài 425 token: 220 token ảnh (`<|image_pad|>`, lưới 22×40 patch 16 px, gộp 2×2), 146 token prompt + template, và chỉ ≈ 59 token đáp án JSON + `<|im_end|>\n` (ảnh có 2 box). Tôi gán `labels = -100` cho toàn bộ tiền tố prompt (đã kiểm tra prompt là tiền tố chính xác của chuỗi đầy đủ), nên loss chỉ tính trên đáp án.

Nếu tính cả prompt: (1) ≈ 86% số token có loss sẽ là prompt cố định, gradient bị chi phối bởi việc học thuộc câu lệnh vốn giống hệt nhau ở mọi mẫu, nên tín hiệu học toạ độ bị pha loãng khoảng 7 lần; (2) token ảnh là placeholder, embedding thật được thay bằng đặc trưng ViT, nên "dự đoán" `<|image_pad|>` kế tiếp là bài toán vô nghĩa (luôn đúng, loss ≈ 0, hoặc làm lệch đầu ra); (3) model có thể học sinh lại prompt thay vì trả lời. Lúc suy luận ta không bao giờ yêu cầu model sinh prompt, nên đó là phép tính bỏ phí. Kiểm tra: giải mã các token có `labels != -100` cho ra đúng chuỗi JSON đáp án + `<|im_end|>\n` (test `test_image_tokens_and_label_mask`).

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?

Lượng tử NF4 thay trọng số `Linear` bằng `Linear4bit`: trọng số lưu dạng số nguyên 4-bit đóng gói (uint8) + hằng số scale theo khối, không phải tensor số thực liên tục. Gradient không cập nhật được trực tiếp lên giá trị 4-bit (không có dtype khả vi, và mỗi bước lại phải lượng tử lại, làm mất cập nhật nhỏ), nên trên lớp 4-bit chỉ có thể gắn LoRA, không thể huấn luyện đầy đủ. Xử lý: `llm_int8_skip_modules=["visual", "lm_head"]` giữ toàn bộ ViT (gồm `visual.merger` và `visual.deepstack_merger_list`) ở fp16; ô E0 trong notebook in dtype ViT = <điền: float16> còn `q_proj` của LLM là `Linear4bit`.

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
<điền: so sánh F1 ± CI95, VRAM đỉnh, phút giữa E1/E2/E3; kết luận setup nào đáng nhất>

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
<điền: độ rộng CI95 (hi − lo) của từng setup; các CI có chồng nhau không; kết luận nào đứng vững (vd. E1 ≫ E0) và kết luận nào không (vd. E2 vs E3)>

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
<điền: F1 từng lớp của setup tốt nhất; giải thích; tham chiếu ảnh trong report/figures/>

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
<điền: biến E4, F1 ± CI95 của hai giá trị, kết luận có vượt CI95 không>

## Nhìn lại

<điền: điều bạn chưa hiểu, lỗi dữ liệu hoặc mô hình bạn gặp mà công cụ chấm chưa bắt được>
