from __future__ import annotations

import itertools
import re
import string
from collections import defaultdict
from collections.abc import Iterable, Sequence

import torch
from overrides import overrides
from torchmetrics import Metric

Span = tuple[float, float]


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


def compute_f1(p: float, r: float) -> float:
    if p == 0 or r == 0:
        return 0
    return 2 * p * r / (p + r)


def compute_iou(span1: Span, span2: Span) -> float:
    start1, end1 = span1
    start2, end2 = span2

    start_intersection = max(start1, start2)
    end_intersection = min(end1, end2)

    if (intersection_size := end_intersection - start_intersection) <= 0:
        return 0.0

    span1_size = end1 - start1
    span2_size = end2 - start2

    assert span1_size >= 0
    assert span2_size >= 0

    return intersection_size / (span1_size + span2_size - intersection_size)


def compute_iou_multiple(span: Span, spans: Iterable[Span]) -> Iterable[float]:
    for s in spans:
        yield compute_iou(span, s)


class IouF1(Metric):  # noqa
    def __init__(self, iou_thresholds: Sequence[float] = tuple(torch.linspace(0.5, 0.95, 10).tolist()),
                 repeated_pred_threshold: float = 1.0) -> None:
        super().__init__()
        # For some reason, if this is a tensor it's not moved to the correct device automatically.
        # Anyway, we don't need it to be a tensor as we basically treat it as a list.
        self.iou_thresholds = iou_thresholds.tolist() if isinstance(iou_thresholds, torch.Tensor) else iou_thresholds
        self.repeated_pred_threshold = repeated_pred_threshold

        self.add_state("total", default=torch.zeros(len(self.iou_thresholds)), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def _compute_f1_instance(self, preds_instance: Sequence[Span], target_instance: Sequence[Span]) -> torch.Tensor:
        # FIXME: shouldn't allow the same target interval be assigned to different predictions.
        #   we should "rule them out" once we match them, or something like that.

        iou_scores = {i: max(compute_iou_multiple(p, target_instance), default=0) for i, p in enumerate(preds_instance)}

        for i, p_i in enumerate(preds_instance):
            for j in range(i + 1, len(preds_instance)):
                p_j = preds_instance[j]
                if compute_iou(p_i, p_j) > self.repeated_pred_threshold:
                    del iou_scores[j if iou_scores[i] >= iou_scores[j] else i]  # Delete the repeated predictions.

        iou_score_tensor = torch.tensor(list(iou_scores.values()), device=self.device)

        f1_scores = []

        for iou_threshold in self.iou_thresholds:
            threshold_tps = (iou_score_tensor >= iou_threshold).sum()
            # We compute the min because the predictions may be assigned to multiple targets and pass the maximum then,
            # or also because if the threshold is 0, we may also pass the maximum allowed.
            micro_p = min(threshold_tps / len(preds_instance), 1) if len(preds_instance) > 0 else 0
            micro_r = min(threshold_tps / len(target_instance), 1) if len(target_instance) > 0 else 0
            f1_scores.append(compute_f1(micro_p, micro_r))

        return torch.tensor(f1_scores, device=self.device)

    @overrides(check_signature=False)
    def update(self, preds: Sequence[Sequence[Span]], target: Sequence[Sequence[Span]],
               alternative_targets: Sequence[Sequence[Sequence[Span]]] | None = None) -> None:
        alternative_targets = alternative_targets or [()] * len(preds)

        self.total += sum(
            torch.stack([self._compute_f1_instance(preds_instance, target_instance)
                         for target_instance in itertools.chain([main_target_instance],
                                                                alternative_targets_instance)]).max(dim=0).values
            for preds_instance, main_target_instance, alternative_targets_instance in zip(preds, target,
                                                                                          alternative_targets)
        )

        self.count += len(preds)

    @overrides
    def compute(self) -> torch.Tensor:
        return self.total / self.count


def interpolated_precision_recall(
        precision: torch.Tensor,  # Shape: (T, S)
        recall: torch.Tensor,  # Shape: (T, S)
) -> torch.Tensor:  # Shape: (T,)
    T = precision.shape[0]

    zero = torch.zeros((T, 1), dtype=precision.dtype, device=precision.device)
    one = torch.ones((T, 1), dtype=precision.dtype, device=precision.device)

    m_precision = torch.hstack([zero, precision, zero]).flip(dims=(-1,)).cummax(dim=-1)[0].flip(dims=(-1,))

    m_recall = torch.hstack([zero, recall, one])

    result = torch.empty(T, device=recall.device)

    # Not sure how to vectorize this:
    for i, (m_precision_t, m_recall_t) in enumerate(zip(m_precision, m_recall)):
        n_idx = torch.where(m_recall_t[1:] != m_recall_t[:-1])[0]  # noqa
        result[i] = ((m_recall_t[n_idx + 1] - m_recall_t[n_idx]) * m_precision_t[n_idx + 1]).sum()

    return result


# Inspired by https://github.com/activitynet/ActivityNet/blob/82304fa/Evaluation/eval_detection.py#L182
# but supporting multiple sets of target spans.
class AveragePrecision(Metric):  # noqa
    def __init__(self, iou_thresholds: Sequence[float] = tuple(torch.linspace(0.5, 0.95, 10).tolist())) -> None:
        super().__init__()
        # For some reason, if this is a tensor it's not moved to the correct device automatically.
        # Anyway, we don't need it to be a tensor as we basically treat it as a list.
        self.iou_thresholds = iou_thresholds.tolist() if isinstance(iou_thresholds, torch.Tensor) else iou_thresholds

        self.add_state("tp", default=[], dist_reduce_fx="cat")
        self.add_state("fp", default=[], dist_reduce_fx="cat")
        self.add_state("target_span_count", default=torch.zeros(len(self.iou_thresholds)), dist_reduce_fx="sum")

    @overrides(check_signature=False)
    def update(self, preds: Sequence[Sequence[Span]], target: Sequence[Sequence[Span]],
               alternative_targets: Sequence[Sequence[Sequence[Span]]] | None = None) -> None:
        # The spans inside each prediction instance are assumed to be sorted by confidence score.

        alternative_targets = alternative_targets or [()] * len(preds)

        pred_span_count = sum(len(pred_instance) for pred_instance in preds)

        tp = torch.empty((len(self.iou_thresholds), pred_span_count), device=self.device)
        fp = torch.empty((len(self.iou_thresholds), pred_span_count), device=self.device)

        target_span_count = torch.zeros(len(self.iou_thresholds), device=self.device)
        pred_span_index = 0

        iou_threshold_range = torch.arange(len(self.iou_thresholds), device=self.device)

        for preds_instance, main_target_instance, alternative_targets_instance in zip(preds, target,
                                                                                      alternative_targets):
            all_target_instance = [main_target_instance] + list(alternative_targets_instance)

            tp_preds_instance = torch.zeros((len(self.iou_thresholds), len(all_target_instance), len(preds_instance)),
                                            device=self.device)
            fp_preds_instance = torch.zeros((len(self.iou_thresholds), len(all_target_instance), len(preds_instance)),
                                            device=self.device)

            for idx_target_instance, target_instance in enumerate(all_target_instance):
                assigned_target_instance_spans = defaultdict(set)
                for idx_preds_instance_span, preds_instance_span in enumerate(preds_instance):
                    iou_scores = torch.tensor(list(compute_iou_multiple(preds_instance_span, target_instance)),
                                              device=self.device)

                    for idx_iou_threshold, iou_threshold in enumerate(self.iou_thresholds):
                        # `tolist()` so we use integers instead of tensors for indexing.
                        for idx_target_instance_span in iou_scores.argsort(descending=True).tolist():
                            if iou_scores[idx_target_instance_span] < iou_threshold:
                                fp_preds_instance[idx_iou_threshold, idx_target_instance, idx_preds_instance_span] = 1
                                break

                            if idx_target_instance_span not in assigned_target_instance_spans[idx_iou_threshold]:
                                tp_preds_instance[idx_iou_threshold, idx_target_instance, idx_preds_instance_span] = 1
                                assigned_target_instance_spans[idx_iou_threshold].add(idx_target_instance_span)
                                break
                        else:
                            fp_preds_instance[idx_iou_threshold, idx_target_instance, idx_preds_instance_span] = 1

            # We want to choose the target that maximizes the precision, and we break ties based on the fewest spans
            # (for a higher recall).
            target_instance_lengths = torch.tensor([len(target_instance) for target_instance in all_target_instance],
                                                   device=self.device)
            tp_per_target_instance_and_iou_threshold = tp_preds_instance.sum(dim=-1)
            idxs_target_instance = [max(((idx_t, (t_tp, -t_l))
                                         for (idx_t, (t_tp, t_l)) in
                                         enumerate(zip(tp_per_target_instance_and_iou_threshold[idx_iou_threshold],
                                                       target_instance_lengths))),
                                        key=lambda x: x[1])[0] for idx_iou_threshold in range(len(self.iou_thresholds))]

            target_span_count += torch.tensor([len(all_target_instance[idx_t]) for idx_t in idxs_target_instance],
                                              device=self.device)

            tp[:, pred_span_index:pred_span_index + len(preds_instance)] = tp_preds_instance[
                iou_threshold_range, idxs_target_instance]
            fp[:, pred_span_index:pred_span_index + len(preds_instance)] = fp_preds_instance[
                iou_threshold_range, idxs_target_instance]

            pred_span_index += len(preds_instance)

        self.tp.append(tp)
        self.fp.append(fp)

        self.target_span_count += target_span_count

    @overrides
    def compute(self) -> torch.Tensor:  # Shape: (T,)
        tp_cum_sum = torch.cat(self.tp, dim=-1).cumsum(dim=-1)
        fp_cum_sum = torch.cat(self.fp, dim=-1).cumsum(dim=-1)

        precision = tp_cum_sum / (tp_cum_sum + fp_cum_sum)
        recall = tp_cum_sum / self.target_span_count.view(-1, 1)

        return interpolated_precision_recall(precision, recall)
