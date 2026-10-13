"""Bước 2: nhãn YOLO → JSON grounding (thang 0–1000 hoặc pixel) và prompt."""
import json

from scripts.labcommon import CLASSES

COORDS = ("norm1000", "pixel")


def yolo_to_box(label, yolo, coord="norm1000", width=640, height=360):
    """YOLO (xc, yc, w, h chuẩn hoá [0,1]) → {"label", "bbox_2d": [x1, y1, x2, y2]} số nguyên."""
    if coord not in COORDS:
        raise ValueError(f"coord phải là một trong {COORDS}")
    xc, yc, w, h = yolo
    sx, sy = (1000, 1000) if coord == "norm1000" else (width, height)
    box = [(xc - w / 2) * sx, (yc - h / 2) * sy, (xc + w / 2) * sx, (yc + h / 2) * sy]
    lim = [sx, sy, sx, sy]
    box = [min(max(int(round(v)), 0), m) for v, m in zip(box, lim)]
    return {"label": label, "bbox_2d": box}


def objects_to_boxes(objects, coord="norm1000", width=640, height=360):
    boxes = [yolo_to_box(o["label"], o["yolo"], coord, width, height) for o in objects]
    # trên→dưới, trái→phải: thứ tự ổn định cho model học
    boxes.sort(key=lambda b: (b["bbox_2d"][1], b["bbox_2d"][0], b["label"]))
    return boxes


def objects_to_answer(objects, coord="norm1000", width=640, height=360):
    """Chuỗi JSON đáp án (gọn, không khoảng trắng); "[]" nếu ảnh không có hư hại."""
    return json.dumps(objects_to_boxes(objects, coord, width, height), separators=(",", ":"))


def box_to_pixel(bbox, coord="norm1000", width=640, height=360):
    """Quy box (thang coord) về pixel để vẽ."""
    if coord == "pixel":
        return tuple(float(v) for v in bbox)
    x1, y1, x2, y2 = bbox
    return (x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height)


def build_prompt(coord="norm1000", width=640, height=360):
    if coord == "norm1000":
        scale = ("Coordinates are integers on a 0-1000 scale relative to the image size "
                 "(0 = left/top edge, 1000 = right/bottom edge), not pixels.")
    elif coord == "pixel":
        scale = (f"Coordinates are integer pixel positions in the {width}x{height} image "
                 f"(x from 0 to {width}, y from 0 to {height}).")
    else:
        raise ValueError(f"coord phải là một trong {COORDS}")
    return (
        "Detect every road damage in this dashcam image. "
        f"Allowed labels: {', '.join(CLASSES)}. "
        "Answer with ONLY a JSON list and no other text, in the form "
        '[{"label": "pothole", "bbox_2d": [x1, y1, x2, y2]}], '
        "where (x1, y1) is the top-left and (x2, y2) the bottom-right corner. "
        f"{scale} If there is no damage, answer []."
    )


PROMPT = build_prompt("norm1000")
