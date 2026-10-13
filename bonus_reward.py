"""Pure Python IoU reward for the optional Day 10 GRPO experiment."""
from __future__ import annotations

import json
from scripts.score import iou_xyxy, parse_output


def reward_iou(completion: str, ground_truth: list[dict]) -> float:
    """Score a generated JSON list against normalized 0–1000 GT boxes.

    Invalid JSON earns 0. Correct empty/empty earns 1. Other valid outputs earn
    0.2 plus matched IoU scaled by the larger of GT and prediction counts.
    """
    prediction = parse_output(completion)
    if prediction is None:
        return 0.0
    if not prediction and not ground_truth:
        return 1.0
    pairs = sorted(
        ((iou_xyxy(tuple(gt["bbox"]), pred["bbox"]), gi, pi)
         for gi, gt in enumerate(ground_truth)
         for pi, pred in enumerate(prediction)
         if gt["label"] == pred["label"]),
        reverse=True,
    )
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    total_iou = 0.0
    for overlap, gi, pi in pairs:
        if overlap < 0.5:
            break
        if gi in used_gt or pi in used_pred:
            continue
        used_gt.add(gi)
        used_pred.add(pi)
        total_iou += overlap
    return 0.2 + 0.8 * total_iou / max(len(ground_truth), len(prediction), 1)


def trl_reward(completions, ground_truth, **kwargs):
    """TRL reward callback; ``ground_truth`` comes from the dataset column."""
    if len(completions) != len(ground_truth):
        raise ValueError("completion and ground_truth batch sizes differ")
    result = []
    for completion, truth in zip(completions, ground_truth):
        if isinstance(completion, list):
            completion = "".join(str(part.get("content", "")) for part in completion)
        if isinstance(truth, str):
            truth = json.loads(truth)
        result.append(reward_iou(completion, truth))
    return result
