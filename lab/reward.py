"""Bước 9 (E5): reward IoU kiểm chứng được cho GRPO. Python thuần, test được không cần GPU."""
import json

from scripts.score import IOU_THR, iou_xyxy, parse_output


def gt_boxes_norm1000(objects):
    """Danh sách (label, bbox thang 0–1000) từ objects YOLO của một entry."""
    out = []
    for o in objects:
        xc, yc, w, h = o["yolo"]
        out.append((o["label"], ((xc - w / 2) * 1000, (yc - h / 2) * 1000,
                                 (xc + w / 2) * 1000, (yc + h / 2) * 1000)))
    return out


def iou_reward(completion, gt):
    """0 nếu JSON hỏng; ngược lại 0.2 + 0.8 * tổng IoU cặp khớp / max(n_gt, n_pred, 1).

    gt: list (label, (x1, y1, x2, y2)) thang 0–1000. Box bịa bị phạt qua mẫu số.
    """
    pred = parse_output(completion)
    if pred is None:
        return 0.0
    if not gt and not pred:
        return 1.0
    pairs = sorted(((iou_xyxy(g[1], p["bbox"]), i, j)
                    for i, g in enumerate(gt) for j, p in enumerate(pred)
                    if g[0] == p["label"]), reverse=True)
    used_g, used_p, total = set(), set(), 0.0
    for v, i, j in pairs:
        if v < IOU_THR:
            break
        if i in used_g or j in used_p:
            continue
        used_g.add(i)
        used_p.add(j)
        total += v
    return 0.2 + 0.8 * total / max(len(gt), len(pred), 1)


def _text(completion):
    # GRPOTrainer trả completion dạng hội thoại [{"role": "assistant", "content": ...}]
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def grpo_reward(completions, gt, **kwargs):
    """Chữ ký reward function của trl.GRPOTrainer; `gt` là cột JSON của dataset."""
    return [iou_reward(_text(c), [(lab, tuple(bb)) for lab, bb in json.loads(g)])
            for c, g in zip(completions, gt)]
