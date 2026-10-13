"""Bước 3–7: nạp Qwen3-VL-4B 4-bit, dựng dữ liệu + mask loss, setup LoRA, huấn luyện, suy luận."""
import gc
import json
import math
import random
import re
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from lab.labels import objects_to_answer
from scripts.labcommon import read_jsonl

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
LLM_TARGETS = r".*language_model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"
VIT_TARGETS = r".*visual\.blocks\.\d+\.(attn\.(qkv|proj)|mlp\.linear_fc(1|2))"
PROJECTOR_KEYS = ("visual.merger", "visual.deepstack_merger_list")
EXTRA_FILE = "extra_trainable.pt"


SKIP_MODULES = ["model.visual", "lm_head"]


# ---------------------------------------------------------------- model
def check_vit_not_quantized(model):
    """Dừng ngay nếu có Linear 4-bit nào trong ViT/projector (dấu hiệu skip-modules không khớp)."""
    bad = [n for n, m in model.named_modules() if "visual" in n and "4bit" in type(m).__name__]
    if bad:
        raise RuntimeError(f"{len(bad)} module ViT bị lượng tử 4-bit (vd. {bad[0]}): kiểm tra SKIP_MODULES")


def load_model(model_id=MODEL_ID):
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

    quant = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        # ViT (gồm cả projector) và lm_head giữ fp16: merger lượng tử 4-bit thì không huấn luyện đầy đủ được.
        # transformers 5.x khớp tên bằng re.match (từ ĐẦU tên module), nên "visual" không khớp
        # "model.visual.merger..." và cả ViT bị lượng tử: phải ghi đầy đủ tiền tố.
        llm_int8_skip_modules=SKIP_MODULES)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_id, quantization_config=quant, dtype=torch.float16, device_map={"": 0})
    check_vit_not_quantized(model)
    processor = AutoProcessor.from_pretrained(model_id)
    processor.tokenizer.padding_side = "left"
    return model, processor


def cleanup():
    """Gọi sau `del model` để trả VRAM trước khi nạp model khác."""
    gc.collect()
    torch.cuda.empty_cache()


def load_image(data_dir, entry):
    return Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")


def user_message(image, prompt):
    return {"role": "user", "content": [{"type": "image", "image": image},
                                        {"type": "text", "text": prompt}]}


# ---------------------------------------------------------------- bước 4: dữ liệu
def encode_example(processor, image, prompt, answer):
    kw = dict(tokenize=True, return_dict=True, return_tensors="pt")
    user = user_message(image, prompt)
    assistant = {"role": "assistant", "content": [{"type": "text", "text": answer}]}
    prompt_enc = processor.apply_chat_template([user], add_generation_prompt=True, **kw)
    full_enc = processor.apply_chat_template([user, assistant], add_generation_prompt=False, **kw)
    n = prompt_enc["input_ids"].shape[1]
    if not torch.equal(full_enc["input_ids"][0, :n], prompt_enc["input_ids"][0]):
        raise ValueError("prompt không phải tiền tố của chuỗi đầy đủ: không mask loss được")
    labels = full_enc["input_ids"].clone()
    labels[:, :n] = -100  # chỉ học phần trả lời (JSON + <|im_end|>)
    out = dict(full_enc)
    out["labels"] = labels
    return out


class GroundingDataset(torch.utils.data.Dataset):
    def __init__(self, entries, data_dir, processor, prompt, coord="norm1000"):
        self.entries, self.data_dir = entries, data_dir
        self.processor, self.prompt, self.coord = processor, prompt, coord

    def __len__(self):
        return len(self.entries)

    def answer(self, e):
        return objects_to_answer(e["objects"], self.coord, e["width"], e["height"])

    def __getitem__(self, i):
        e = self.entries[i]
        return encode_example(self.processor, load_image(self.data_dir, e), self.prompt, self.answer(e))


PAD_VALUES = {"attention_mask": 0, "mm_token_type_ids": 0, "labels": -100}


def collate(batch, pad_id):
    """Pad các trường theo token (phải); nối pixel_values/image_grid_thw theo chiều 0, không pad."""
    length = max(b["input_ids"].shape[1] for b in batch)
    out = {}
    for key in ("input_ids", "attention_mask", "mm_token_type_ids", "labels"):
        if key not in batch[0]:
            continue
        val = PAD_VALUES.get(key, pad_id)
        out[key] = torch.cat([F.pad(b[key], (0, length - b[key].shape[1]), value=val) for b in batch])
    for key in ("pixel_values", "image_grid_thw"):
        out[key] = torch.cat([b[key] for b in batch])
    return out


def subsample(entries, fraction, seed=42):
    if fraction >= 1.0:
        return list(entries)
    k = max(1, round(len(entries) * fraction))
    return sorted(random.Random(seed).sample(list(entries), k), key=lambda e: e["image_id"])


# ---------------------------------------------------------------- bước 5: LoRA / đóng băng
def target_pattern(targets):
    if targets == "llm":
        return LLM_TARGETS
    if targets == "llm+vit":
        return rf"(?:{LLM_TARGETS})|(?:{VIT_TARGETS})"
    raise ValueError(targets)


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def apply_setup(model, targets="llm", train_projector=False, r=16, alpha=32, dropout=0.05,
                gradient_checkpointing=True):
    """E1: targets='llm'. E2: + train_projector. E3: targets='llm+vit' + train_projector."""
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        # use_reentrant=False + bitsandbytes 4-bit + autocast fp16 trên T4 báo CheckpointError
        # ("139 tensor lúc forward, 0 lúc recompute"); bản reentrant là cấu hình QLoRA kinh điển.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.config.use_cache = False
        # Không checkpoint ViT: E1/E2 ViT đóng băng nên không lưu activation; E3 tốn thêm ~1,4 GB.
        # (Checkpoint reentrant cần đầu vào requires_grad, mà pixel_values thì không.)
        for n, m in model.named_modules():
            if "visual" in n and hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False
    model.enable_input_require_grads()

    pattern = target_pattern(targets)
    matched = [n for n, _ in model.named_modules() if re.fullmatch(pattern, n)]
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout,
                                             target_modules=pattern))
    projector = 0
    if train_projector:
        for n, p in model.named_parameters():
            if any(k in n for k in PROJECTOR_KEYS):
                p.data = p.data.float()  # fp32 master weights: GradScaler không unscale grad fp16
                p.requires_grad_(True)
                projector += p.numel()
    for p in model.parameters():  # adapter LoRA cũng phải fp32
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    trainable, total = count_params(model)
    info = {"matched_modules": len(matched), "trainable_params": trainable,
            "projector_params": projector, "total_params": total}
    print(f"LoRA trên {len(matched)} module · tham số huấn luyện {trainable:,} "
          f"(projector {projector:,}) / {total:,}")
    return model, info


# ---------------------------------------------------------------- bước 6: huấn luyện
def projector_state(model):
    return {n: p.detach().to("cpu", copy=True) for n, p in model.named_parameters()
            if p.requires_grad and any(k in n for k in PROJECTOR_KEYS)}


def save_checkpoint(model, out_dir):
    out_dir = Path(out_dir)
    model.save_pretrained(out_dir / "adapter")  # chỉ chứa LoRA
    extra = projector_state(model)
    if extra:  # projector huấn luyện đầy đủ KHÔNG nằm trong adapter
        torch.save(extra, out_dir / EXTRA_FILE)


def peak_vram_gb():
    return torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0


def train(model, dataset, collate_fn, out_dir, lr=2e-4, batch_size=2, grad_accum=4, max_steps=300,
          warmup_ratio=0.05, save_every=100, seed=42, log_every=10, num_workers=2):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(seed)
    torch.manual_seed(seed)
    cuda = torch.cuda.is_available()
    device = next(p.device for p in model.parameters()) if cuda else torch.device("cpu")
    amp = ("cuda", torch.float16, True) if cuda else ("cpu", torch.bfloat16, False)  # CPU: chạy thử fp32
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    warm = max(1, int(warmup_ratio * max_steps))

    def lr_lambda(s):
        if s < warm:
            return (s + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn,
        num_workers=num_workers, generator=torch.Generator().manual_seed(seed))

    model.train()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0, it, log = time.time(), iter(loader), []
    for step in range(1, max_steps + 1):
        step_loss = 0.0
        for _ in range(grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(amp[0], dtype=amp[1], enabled=amp[2]):
                loss = model(**batch).loss / grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"loss không hữu hạn ở bước {step}: {loss.item()} (tràn fp16?)")
            scaler.scale(loss).backward()
            step_loss += loss.item()
        scaler.unscale_(opt)
        if step == 1:  # mọi tham số huấn luyện phải nhận gradient, nếu không là setup sai
            no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
            if no_grad:
                raise RuntimeError(f"{len(no_grad)} tham số không có gradient, vd. {no_grad[:3]}")
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        sched.step()
        log.append({"step": step, "loss": step_loss, "grad_norm": gnorm, "lr": sched.get_last_lr()[0]})
        if step % log_every == 0 or step == 1:
            el = (time.time() - t0) / 60
            print(f"step {step:4d}/{max_steps} loss {step_loss:.4f} gnorm {gnorm:.2f} "
                  f"lr {log[-1]['lr']:.2e} · {el:.1f} phút · VRAM đỉnh "
                  f"{peak_vram_gb():.2f} GB", flush=True)
        if step % save_every == 0 or step == max_steps:
            save_checkpoint(model, out_dir)
    run = {
        "trainable_params": sum(p.numel() for p in params),
        "peak_vram_gb": round(peak_vram_gb(), 2),
        "train_minutes": round((time.time() - t0) / 60, 1),
        "max_steps": max_steps, "batch_size": batch_size, "grad_accum": grad_accum, "lr": lr,
        "n_train_images": len(dataset), "final_loss": log[-1]["loss"],
    }
    (out_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    (out_dir / "train_log.json").write_text(json.dumps(log), encoding="utf-8")
    return run


# ---------------------------------------------------------------- bước 7: suy luận
def load_trained(run_dir, model_id=MODEL_ID, is_trainable=False, base=None):
    """Model gốc 4-bit → adapter LoRA → nạp lại projector đã huấn luyện (nếu có).

    base=(model, processor) để dùng model gốc đã nạp sẵn (chạy thử trên CPU).
    """
    from peft import PeftModel

    run_dir = Path(run_dir)
    model, processor = base if base is not None else load_model(model_id)
    model = PeftModel.from_pretrained(model, run_dir / "adapter", is_trainable=is_trainable)
    extra_path = run_dir / EXTRA_FILE
    if extra_path.exists():
        extra = torch.load(extra_path, map_location="cpu")
        params = dict(model.named_parameters())
        missing = [n for n in extra if n not in params]
        if missing:
            raise KeyError(f"không khớp tên tham số projector: {missing[:3]}")
        with torch.no_grad():
            for n, t in extra.items():
                params[n].copy_(t.to(params[n].dtype))
        print(f"đã nạp {len(extra)} tensor projector từ {extra_path.name}")
    model.eval()
    return model, processor


@torch.no_grad()
def predict(model, processor, entries, data_dir, prompt, out_path, batch_size=4,
            max_new_tokens=256, resume=True):
    """Ghi {"image_id", "raw"} từng dòng; resume=True bỏ qua ảnh đã có (khi Colab bị ngắt)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["image_id"] for r in read_jsonl(out_path)} if resume and out_path.exists() else set()
    todo = [e for e in entries if e["image_id"] not in done]
    model.eval()
    t0 = time.time()
    with open(out_path, "a" if done else "w", encoding="utf-8") as f:
        for i in range(0, len(todo), batch_size):
            batch = todo[i:i + batch_size]
            msgs = [[user_message(load_image(data_dir, e), prompt)] for e in batch]
            inputs = processor.apply_chat_template(
                msgs, tokenize=True, add_generation_prompt=True, return_dict=True,
                return_tensors="pt", processor_kwargs={"padding": True}).to(model.device)
            out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                                 use_cache=True)
            raws = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                                          skip_special_tokens=True)
            for e, raw in zip(batch, raws):
                f.write(json.dumps({"image_id": e["image_id"], "raw": raw}, ensure_ascii=False) + "\n")
            f.flush()
            n = i + len(batch)
            if n % (batch_size * 10) == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} ảnh · {(time.time() - t0) / 60:.1f} phút", flush=True)
    return out_path
