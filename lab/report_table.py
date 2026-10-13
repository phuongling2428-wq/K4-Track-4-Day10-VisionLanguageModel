"""In bảng tổng hợp Markdown từ report/evidence/*.json để dán vào REPORT.md."""
import argparse
import json
from pathlib import Path

SETUP = {"e0": "zero-shot", "e1": "LoRA LLM", "e2": "+ projector", "e3": "+ LoRA ViT"}


def _fmt_int(v):
    return f"{int(v):,}".replace(",", ".") if isinstance(v, (int, float)) else ""


def row(key, d):
    m, pc, meta = d["micro"], d["per_class"], d.get("meta", {})
    lo, hi = d["f1_ci95"]
    setup = SETUP.get(key, key[3:] if key.startswith("e4_") else key)
    return (f"| {d['name']} | {setup} | {_fmt_int(meta.get('trainable_params', 0))} "
            f"| {meta.get('peak_vram_gb', '')} | {meta.get('train_minutes', '')} "
            f"| {m['f1']:.3f} | {lo:.3f}–{hi:.3f} | {d['fp_per_image']:.2f} "
            f"| {pc['pothole']['f1']:.3f} | {pc['crack']['f1']:.3f} | {pc['manhole']['f1']:.3f} |")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--evidence", default="report/evidence")
    a = ap.parse_args(argv)
    files = sorted(Path(a.evidence).glob("e*.json"))
    print("| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh "
          "| F1 pothole | F1 crack | F1 manhole |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for p in files:
        print(row(p.stem, json.loads(p.read_text(encoding="utf-8"))))
    print()
    print("| Mã | parse_valid | P | R | mean IoU | TP | FP | FN |")
    print("|---|---|---|---|---|---|---|---|")
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        m = d["micro"]
        print(f"| {d['name']} | {d['parse_valid_rate']:.3f} | {m['precision']:.3f} | {m['recall']:.3f} "
              f"| {d['mean_iou']:.3f} | {m['tp']} | {m['fp']} | {m['fn']} |")


if __name__ == "__main__":
    main()
