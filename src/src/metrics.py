from __future__ import annotations

import re
import string
from collections.abc import Sequence

import torch
from overrides import overrides
from torchmetrics import Metric


def _remove_articles(text: str) -> str:
    return re.sub(r"\b(?:a|an|the)\b", "", text)


def _remove_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_answer(answer: str) -> str:
    return _remove_articles(_remove_punctuation(answer.lower()))


class Perplexity(Metric):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, answer_probs: torch.Tensor, mask: torch.Tensor | None = None) -> None:
        if mask is not None:
            answer_probs = answer_probs.clone()
            answer_probs[~mask] = float("NaN")

        # It doesn't matter the log and exp base as long as they are the same because they cancel out.
        self.total += (-answer_probs.log().nanmean(dim=-1)).exp().sum()
        self.count += len(answer_probs)

    @overrides
    def compute(self) -> torch.Tensor:
        return self.total / self.count


def get_best_evidence_spans(start_scores: torch.Tensor, end_scores: torch.Tensor, mask: torch.Tensor,
                            top_k: int = 1, are_probs: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    start_scores[~mask] = end_scores[~mask] = -float("inf")

    if are_probs:
        candidates = start_scores.unsqueeze(dim=2) @ end_scores.unsqueeze(dim=1)
    else:
        candidates = start_scores.unsqueeze(dim=2) + end_scores.unsqueeze(dim=1)

    lower_triangle = torch.tril(torch.ones(*candidates.shape[1:], device=mask.device, dtype=bool), diagonal=-1)  # noqa
    candidates[..., lower_triangle] = -float("inf")

    span_count = candidates.shape[1] * candidates.shape[2]

    scores_flat = candidates.view(-1, span_count)
    idx_sort = scores_flat.topk(k=min(top_k, span_count), dim=1)[1]

    start = idx_sort.div(candidates.shape[2], rounding_mode="trunc")
    end = idx_sort % candidates.shape[2]

    return start, end

def f1(p: float, r: float) -> float:
    if p == 0 or r == 0:
        return 0
    return 2 * p * r / (p + r)

def iou(span_1: tuple[float, float], span_2: tuple[float, float]) -> float:
    num = len(
        set(range(round(span_1[0]), round(span_1[1])))
        & set(range(round(span_2[0]), round(span_2[1])))
    )
    denominator = len(
        set(range(round(span_1[0]), round(span_1[1])))
        | set(range(round(span_2[0]), round(span_2[1])))
    )
    return 0 if denominator == 0 else num / denominator


class IouF1(Metric):
    def __init__(self, threshold: float = 0.5, pred_threshold: float = 1.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.threshold = threshold
        self.pred_threshold = pred_threshold

        self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, preds: Sequence[Sequence[tuple[float, float]]],
               target: Sequence[Sequence[tuple[float, float]]]) -> None:
        for preds_instance, target_instance in zip(preds, target):
            if len(preds_instance) == 0 and len(target_instance) == 0:
                micro_f1 = 1
            else:
                iou_s = {i: max((iou(p, g) for g in target_instance), default=0)
                         for i, p in enumerate(preds_instance)}

                for i in range(len(preds_instance)):  # Delete the repeated predictions.
                    p_i = preds_instance[i]
                    for j in range(i + 1, len(preds_instance)):
                        p_j = preds_instance[j]
                        if iou(p_i, p_j) > self.pred_threshold:
                            del iou_s[j if iou_s[i] >= iou_s[j] else i]

                threshold_tps = sum(iou_value >= self.threshold for iou_value in iou_s.values())
                micro_p = threshold_tps / len(preds_instance) if len(preds_instance) > 0 else 0
                micro_r = threshold_tps / len(target_instance) if len(target_instance) > 0 else 0
                micro_f1 = f1(micro_p, micro_r)

            self.total += micro_f1

        self.count += len(preds)

    @overrides
    def compute(self) -> float:
        return self.total / self.count
