"""Thống kê split, vẽ nhãn và ảnh minh hoạ dự đoán đúng/sai (box dự đoán + box đúng)."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw  # noqa: E402

from lab.labels import box_to_pixel, objects_to_boxes  # noqa: E402
from scripts.labcommon import CLASSES, read_jsonl  # noqa: E402
from scripts.score import _gt_boxes, _match, _to_norm1000, parse_output  # noqa: E402

GT_COLOR = (0, 220, 0)
PRED_COLOR = {"pothole": (255, 60, 60), "crack": (255, 170, 0), "manhole": (60, 140, 255)}


def split_stats(splits_dir="data/splits"):
    """In số ảnh/box từng split và kiểm tra không phiên nào nằm ở hai split."""
    sessions, rows = {}, []
    for s in ("train", "val", "test"):
        entries = read_jsonl(Path(splits_dir) / f"{s}.jsonl")
        boxes = Counter(o["label"] for e in entries for o in e["objects"])
        empty = sum(1 for e in entries if not e["objects"])
        sessions[s] = {e["session"] for e in entries}
        rows.append((s, len(entries), len(sessions[s]), empty, *(boxes[c] for c in CLASSES)))
    print(f"{'split':6s} {'ảnh':>5s} {'phiên':>5s} {'rỗng':>5s} " + " ".join(f"{c:>8s}" for c in CLASSES))
    for r in rows:
        print(f"{r[0]:6s} {r[1]:5d} {r[2]:5d} {r[3]:5d} " + " ".join(f"{v:8d}" for v in r[4:]))
    leak = (sessions["train"] & sessions["val"]) | (sessions["train"] & sessions["test"]) \
        | (sessions["val"] & sessions["test"])
    print("phiên nằm ở >1 split:", sorted(leak) or "không có")
    return rows, leak


def _draw_box(d, box, color, text=None, width=2):
    d.rectangle(box, outline=color, width=width)
    if text:
        tx, ty = box[0] + 2, max(0, box[1] - 11)
        d.rectangle((tx - 1, ty, tx + 6 * len(text) + 2, ty + 10), fill=color)
        d.text((tx, ty - 1), text, fill=(0, 0, 0))


def draw_gt(entry, data_dir="data"):
    """Vẽ nhãn gốc: đi qua JSON 0–1000 rồi quy ngược về pixel (kiểm tra bước 2)."""
    im = Image.open(Path(data_dir) / entry["image_path"]).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = entry["width"], entry["height"]
    for o in entry["objects"]:  # box YOLO gốc (mảnh, trắng)
        xc, yc, bw, bh = o["yolo"]
        d.rectangle(((xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h),
                    outline=(255, 255, 255), width=1)
    for b in objects_to_boxes(entry["objects"]):  # box sau khi đi qua thang 0–1000
        _draw_box(d, box_to_pixel(b["bbox_2d"], "norm1000", w, h), PRED_COLOR[b["label"]], b["label"])
    return im


def grid(images, cols=3, captions=None):
    w, h = images[0].size
    cap_h = 16 if captions else 0
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * (h + cap_h)), (255, 255, 255))
    d = ImageDraw.Draw(out)
    for i, im in enumerate(images):
        x, y = (i % cols) * w, (i // cols) * (h + cap_h)
        out.paste(im, (x, y + cap_h))
        if captions:
            d.text((x + 4, y + 2), captions[i], fill=(0, 0, 0))
    return out


def per_image(entries, preds, coord="norm1000"):
    """TP/FP/FN từng ảnh theo đúng luật của score.py."""
    out = []
    for e in entries:
        raw = preds.get(e["image_id"])
        parsed = parse_output(raw) if raw is not None else None
        pred = _to_norm1000(parsed, coord, e["width"], e["height"]) if parsed else []
        per, _ = _match(_gt_boxes(e), pred)
        tp, fp, fn = (sum(v[k] for v in per.values()) for k in range(3))
        out.append({"entry": e, "raw": raw, "pred": pred, "valid": parsed is not None,
                    "tp": tp, "fp": fp, "fn": fn})
    return out


def draw_prediction(item, data_dir="data"):
    e = item["entry"]
    im = Image.open(Path(data_dir) / e["image_path"]).convert("RGB")
    d = ImageDraw.Draw(im)
    w, h = e["width"], e["height"]
    for g in _gt_boxes(e):
        _draw_box(d, box_to_pixel(g["bbox"], "norm1000", w, h), GT_COLOR, "GT " + g["label"], width=3)
    for p in item["pred"]:
        color = PRED_COLOR.get(p["label"], (255, 0, 255))
        _draw_box(d, box_to_pixel(p["bbox"], "norm1000", w, h), color, p["label"])
    return im


def pick_examples(items, n_good=3, n_bad=3):
    """Đúng: ảnh có box, không FP/FN, nhiều TP nhất. Sai: nhiều FP+FN nhất (gồm JSON hỏng)."""
    good = sorted((i for i in items if i["tp"] and not i["fp"] and not i["fn"]),
                  key=lambda i: (-i["tp"], i["entry"]["image_id"]))[:n_good]
    bad = sorted((i for i in items if i["fp"] + i["fn"]),
                 key=lambda i: (-(i["fp"] + i["fn"]), i["entry"]["image_id"]))[:n_bad]
    return good, bad


def caption(item):
    tag = "" if item["valid"] else " JSON hỏng"
    return f"{item['entry']['image_id'][:28]} TP={item['tp']} FP={item['fp']} FN={item['fn']}{tag}"


def make_figures(pred_path, out_dir, name, splits_dir="data/splits", data_dir="data", split="test",
                 coord="norm1000", n_good=3, n_bad=3):
    entries = read_jsonl(Path(splits_dir) / f"{split}.jsonl")
    preds = {r["image_id"]: r["raw"] for r in read_jsonl(pred_path)}
    items = per_image(entries, preds, coord)
    good, bad = pick_examples(items, n_good, n_bad)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for kind, group in (("dung", good), ("sai", bad)):
        for k, it in enumerate(group, 1):
            path = out_dir / f"{name}_{kind}_{k}.jpg"
            draw_prediction(it, data_dir).save(path, quality=90)
            records.append({"file": path.name, "kind": kind, "image_id": it["entry"]["image_id"],
                            "tp": it["tp"], "fp": it["fp"], "fn": it["fn"], "valid": it["valid"],
                            "gt": [o["label"] for o in it["entry"]["objects"]],
                            "pred": [p["label"] for p in it["pred"]], "raw": (it["raw"] or "")[:300]})
    sel = good + bad
    grid([draw_prediction(i, data_dir) for i in sel], cols=3,
         captions=[caption(i) for i in sel]).save(out_dir / f"{name}_grid.jpg", quality=88)
    (out_dir / f"{name}_examples.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    return records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="report/figures")
    ap.add_argument("--split", default="test")
    ap.add_argument("--coord", default="norm1000", choices=["norm1000", "pixel"])
    a = ap.parse_args(argv)
    for r in make_figures(a.pred, a.out, a.name, split=a.split, coord=a.coord):
        print(r["file"], r["kind"], f"TP={r['tp']} FP={r['fp']} FN={r['fn']}", r["gt"], "→", r["pred"])


if __name__ == "__main__":
    main()
