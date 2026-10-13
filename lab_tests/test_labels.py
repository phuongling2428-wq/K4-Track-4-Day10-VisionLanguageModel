import json

import pytest

from lab.labels import box_to_pixel, build_prompt, objects_to_answer, yolo_to_box
from scripts.score import _gt_boxes, evaluate, parse_output


def test_yolo_to_box_norm1000():
    assert yolo_to_box("pothole", [0.5, 0.5, 0.2, 0.1]) == {"label": "pothole", "bbox_2d": [400, 450, 600, 550]}


def test_yolo_to_box_clamps_to_scale():
    assert yolo_to_box("crack", [0.02, 0.98, 0.1, 0.1])["bbox_2d"] == [0, 930, 70, 1000]


def test_yolo_to_box_pixel():
    assert yolo_to_box("manhole", [0.5, 0.5, 0.25, 0.5], "pixel", 640, 360)["bbox_2d"] == [240, 90, 400, 270]


def test_answer_sorted_compact_and_empty():
    objs = [{"label": "crack", "yolo": [0.8, 0.8, 0.1, 0.1]},
            {"label": "pothole", "yolo": [0.2, 0.2, 0.1, 0.1]}]
    ans = objects_to_answer(objs)
    assert " " not in ans
    assert [b["label"] for b in json.loads(ans)] == ["pothole", "crack"]
    assert objects_to_answer([]) == "[]"


def test_roundtrip_back_to_pixels():
    yolo = [0.3125, 0.4, 0.1, 0.05]
    b = yolo_to_box("pothole", yolo)["bbox_2d"]
    x1, y1, x2, y2 = box_to_pixel(b, "norm1000", 640, 360)
    ref = ((yolo[0] - yolo[2] / 2) * 640, (yolo[1] - yolo[3] / 2) * 360,
           (yolo[0] + yolo[2] / 2) * 640, (yolo[1] + yolo[3] / 2) * 360)
    for got, exp in zip((x1, y1, x2, y2), ref):
        assert abs(got - exp) <= 0.64  # sai số làm tròn ≤ nửa đơn vị của thang 0–1000


@pytest.mark.parametrize("coord", ["norm1000", "pixel"])
def test_ground_truth_answer_scores_perfect(coord):
    """Đáp án huấn luyện chấm bằng score.py (đúng --coord) phải đạt F1 = 1."""
    entries = [{"image_id": f"i{k}", "width": 640, "height": 360,
                "objects": [{"label": "crack", "yolo": [0.1 + 0.2 * k, 0.5, 0.1, 0.08]},
                            {"label": "manhole", "yolo": [0.5, 0.8, 0.12, 0.1]}]} for k in range(4)]
    entries.append({"image_id": "empty", "width": 640, "height": 360, "objects": []})
    preds = {e["image_id"]: objects_to_answer(e["objects"], coord) for e in entries}
    res = evaluate(entries, preds, coord=coord, n_boot=20)
    assert res["parse_valid_rate"] == 1.0
    assert res["micro"]["f1"] == 1.0


def test_wrong_scale_breaks_f1():
    """Nhãn pixel nhưng chấm như 0–1000: box co về góc trên-trái, F1 ≈ 0."""
    e = {"image_id": "a", "width": 640, "height": 360,
         "objects": [{"label": "pothole", "yolo": [0.6, 0.6, 0.2, 0.2]}]}
    res = evaluate([e], {"a": objects_to_answer(e["objects"], "pixel")}, coord="norm1000", n_boot=20)
    assert res["micro"]["f1"] == 0.0
    assert parse_output(objects_to_answer(e["objects"]))[0]["bbox"] == pytest.approx(_gt_boxes(e)[0]["bbox"])


def test_prompt_mentions_scale_and_classes():
    p = build_prompt("norm1000")
    assert "0-1000" in p and all(c in p for c in ("pothole", "crack", "manhole"))
    assert "640x360" in build_prompt("pixel")
