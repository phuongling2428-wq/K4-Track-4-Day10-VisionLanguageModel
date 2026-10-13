"""Kiểm tra phần dữ liệu và setup LoRA trên CPU (không cần GPU / trọng số).

Các test dùng processor/config thật của Qwen3-VL-4B sẽ bị bỏ qua nếu không tải được.
"""
import re

import pytest
import torch

transformers = pytest.importorskip("transformers")

from lab.labels import PROMPT, objects_to_answer  # noqa: E402
from lab.vlm import LLM_TARGETS, MODEL_ID, VIT_TARGETS, collate, encode_example, target_pattern  # noqa: E402


@pytest.fixture(scope="module")
def processor():
    try:
        return transformers.AutoProcessor.from_pretrained(MODEL_ID)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"không tải được processor: {e}")


@pytest.fixture(scope="module")
def meta_model():
    accelerate = pytest.importorskip("accelerate")
    try:
        cfg = transformers.AutoConfig.from_pretrained(MODEL_ID)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"không tải được config: {e}")
    with accelerate.init_empty_weights():
        return transformers.Qwen3VLForConditionalGeneration._from_config(cfg)


def _example(processor, objects):
    from PIL import Image
    img = Image.new("RGB", (640, 360), (90, 90, 90))
    return encode_example(processor, img, PROMPT, objects_to_answer(objects))


def test_image_tokens_and_label_mask(processor):
    objs = [{"label": "pothole", "yolo": [0.5, 0.5, 0.2, 0.1]}]
    ex = _example(processor, objs)
    pad = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    assert int((ex["input_ids"] == pad).sum()) == 220
    assert ex["image_grid_thw"].tolist() == [[1, 22, 40]]
    learned = processor.tokenizer.decode(ex["input_ids"][ex["labels"] != -100])
    assert learned == objects_to_answer(objs) + "<|im_end|>\n"


def test_collate_pads_tokens_and_concats_pixels(processor):
    a = _example(processor, [])
    b = _example(processor, [{"label": "crack", "yolo": [0.2, 0.3, 0.1, 0.1]},
                             {"label": "manhole", "yolo": [0.6, 0.7, 0.1, 0.1]}])
    out = collate([a, b], processor.tokenizer.pad_token_id)
    L = max(a["input_ids"].shape[1], b["input_ids"].shape[1])
    for k in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
        assert out[k].shape == (2, L)
    short = 0 if a["input_ids"].shape[1] < L else 1
    assert (out["labels"][short, -1] == -100) and (out["attention_mask"][short, -1] == 0)
    assert out["pixel_values"].shape[0] == a["pixel_values"].shape[0] + b["pixel_values"].shape[0]
    assert out["image_grid_thw"].shape == (2, 3)


def test_lora_targets_match_expected_counts(meta_model):
    names = [n for n, _ in meta_model.named_modules()]
    assert sum(bool(re.fullmatch(LLM_TARGETS, n)) for n in names) == 252
    assert sum(bool(re.fullmatch(VIT_TARGETS, n)) for n in names) == 96
    assert sum(bool(re.fullmatch(target_pattern("llm+vit"), n)) for n in names) == 348
    # projector không bị regex ViT bắt nhầm
    assert not any(re.fullmatch(VIT_TARGETS, n) for n in names if "merger" in n)


def test_projector_size(meta_model):
    size = sum(p.numel() for n, p in meta_model.named_parameters()
               if "visual.merger" in n or "visual.deepstack_merger_list" in n)
    assert 105e6 < size < 112e6


def test_peft_trainable_counts(meta_model):
    """E1 vs E2 vs E3 trên model meta (cấu hình như apply_setup, không lượng tử)."""
    peft = pytest.importorskip("peft")
    import copy
    counts = {}
    for name, targets, proj in (("e1", "llm", False), ("e2", "llm", True), ("e3", "llm+vit", True)):
        m = copy.deepcopy(meta_model)
        for p in m.parameters():
            p.requires_grad_(False)
        m = peft.get_peft_model(m, peft.LoraConfig(r=16, lora_alpha=32, target_modules=target_pattern(targets)))
        if proj:
            for n, p in m.named_parameters():
                if "visual.merger" in n or "visual.deepstack_merger_list" in n:
                    p.requires_grad_(True)
        counts[name] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert 105e6 < counts["e2"] - counts["e1"] < 112e6
    assert counts["e3"] > counts["e2"]
    print(counts)
