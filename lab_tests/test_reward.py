import json

import pytest

from lab.reward import grpo_reward, gt_boxes_norm1000, iou_reward

GT = [("pothole", (100, 100, 300, 300)), ("crack", (500, 500, 700, 600))]
PERFECT = json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 300, 300]},
                      {"label": "crack", "bbox_2d": [500, 500, 700, 600]}])


def test_perfect():
    assert iou_reward(PERFECT, GT) == pytest.approx(1.0)


def test_broken_json():
    assert iou_reward('[{"label": "pothole", "bbox_2d": [1, 2', GT) == 0.0
    assert iou_reward("I see a pothole.", GT) == 0.0


def test_fenced_json_still_valid():
    assert iou_reward("```json\n" + PERFECT + "\n```", GT) == pytest.approx(1.0)


def test_wrong_class_gets_only_format_reward():
    wrong = json.dumps([{"label": "manhole", "bbox_2d": [100, 100, 300, 300]},
                        {"label": "manhole", "bbox_2d": [500, 500, 700, 600]}])
    assert iou_reward(wrong, GT) == pytest.approx(0.2)


def test_extra_box_is_penalised():
    extra = json.loads(PERFECT) + [{"label": "crack", "bbox_2d": [0, 0, 50, 50]}]
    assert iou_reward(json.dumps(extra), GT) == pytest.approx(0.2 + 0.8 * 2 / 3)


def test_low_iou_not_matched():
    shifted = json.dumps([{"label": "pothole", "bbox_2d": [250, 250, 450, 450]}])
    assert iou_reward(shifted, GT[:1]) == pytest.approx(0.2)


def test_partial_iou_counts():
    # IoU = 200*150 / (200*200) = 0.75
    r = iou_reward(json.dumps([{"label": "pothole", "bbox_2d": [100, 100, 300, 250]}]), GT[:1])
    assert r == pytest.approx(0.2 + 0.8 * 0.75)


def test_empty_cases():
    assert iou_reward("[]", []) == 1.0
    assert iou_reward("[]", GT) == pytest.approx(0.2)  # trả rỗng không được thưởng IoU
    assert iou_reward(PERFECT, []) == pytest.approx(0.2)  # bịa box trên ảnh rỗng


def test_one_giant_box_hack_is_not_rewarded():
    giant = json.dumps([{"label": "pothole", "bbox_2d": [0, 0, 1000, 1000]}])
    assert iou_reward(giant, GT) == pytest.approx(0.2)


def test_one_pred_cannot_match_two_gt():
    gt = [("crack", (100, 100, 200, 200)), ("crack", (100, 100, 200, 200))]
    one = json.dumps([{"label": "crack", "bbox_2d": [100, 100, 200, 200]}])
    assert iou_reward(one, gt) == pytest.approx(0.2 + 0.8 * 1 / 2)


def test_grpo_signature_with_conversational_completions():
    gt = json.dumps([[lab, list(bb)] for lab, bb in GT])
    comps = [[{"role": "assistant", "content": PERFECT}], [{"role": "assistant", "content": "oops"}]]
    assert grpo_reward(comps, [gt, gt], trainer_state=None) == [pytest.approx(1.0), 0.0]


def test_gt_from_yolo():
    (lab, bb), = gt_boxes_norm1000([{"label": "crack", "yolo": [0.5, 0.5, 0.2, 0.1]}])
    assert lab == "crack" and bb == pytest.approx((400, 450, 600, 550))
