import warnings
from collections.abc import Iterable, Mapping, Sequence
from typing import Literal

from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge

warnings.filterwarnings("ignore")  # filter user warning for BLEU when overlap is 0


def evaluate_qa(model_name: str, preds: Sequence[str], dataset: Iterable[Mapping[str, str]]) -> None:
    sources = [instance["question"] for instance in dataset]
    labels = [[instance["answer"]] for instance in dataset]

    evaluation = QAEvaluation(sources, preds, labels)

    print(f"------------------{model_name} Baseline----------------------")
    print(f"Exact Match: {round(evaluation.exact_match() * 100, 2)}%")
    print(f"BLEU 1: {round(evaluation.bleu(1) * 100, 2)}%")
    print(f"BLEU 2: {round(evaluation.bleu(2) * 100, 2)}%")
    print(f"BLEU 3: {round(evaluation.bleu(3) * 100, 2)}%")
    print(f"ROUGE 1: {round(evaluation.rouge(1) * 100, 2)}%")
    print(f"ROUGE 1: {round(evaluation.rouge(2) * 100, 2)}%")
    print(f"ROUGE 3: {round(evaluation.rouge(3) * 100, 2)}%")


def evaluate_evidence(model_name: str, preds: Sequence[Sequence[tuple[float, float]]],
                      dataset: Iterable[Mapping[str, str]]) -> None:
    """ 
    QAEvaluation for evidence finding.

    predictions: list of lists of span list, that is: [[[start1, end1], [start2, end2]], ...]
        The innermost list gives the predicted start, end.
        The second innermost list gives all the span predictions for an instance.
        The outermost list is the predictions for all instances.
    """
    gt_spans = [instance["evidence"] for instance in dataset]
    evidence_evaluation = EvidenceEvaluation(preds, gt_spans)

    print(f"------------------{model_name} Baseline----------------------")
    print(f"IOU F1: {round(evidence_evaluation.iou_f1() * 100, 2)}%")


class QAEvaluation:
    def __init__(self, sources: Sequence[str], preds: Sequence[str], labels: Sequence[Sequence[str]]) -> None:
        assert isinstance(labels[0], Sequence)

        self.sources = sources
        self.preds = preds
        self.labels = labels

        assert len(self.sources) == len(self.preds) == len(self.labels)

    def exact_match(self) -> float:
        return sum(1 for pred, label in zip(self.preds, self.labels) if pred in label) / len(self.preds)

    def bleu(self, n: Literal[1, 2, 3, 4]) -> float:
        # individual BLEU n-gram score
        pred_tokens = [word_tokenize(pred) for pred in self.preds]
        label_tokens = [[word_tokenize(label) for label in l_labels] for l_labels in self.labels]

        assert 1 <= n <= 4
        weights = [0, 0, 0, 0]
        weights[n - 1] = 1

        return sum(sentence_bleu(label_tok, pred_tok, weights=tuple(weights))
                   for pred_tok, label_tok in zip(pred_tokens, label_tokens)) / len(self.preds)

    def rouge(self, n: Literal[1, 2, 3, 4, 5, "l"], t: Literal["n", "l", "w"] = "n",
              stats: Literal["p", "r", "f"] = "p") -> float:
        """ 
        stats: "p": precision; "r": recall; "f": f1
        t: Rouge type:
            ROUGE-N: Overlap of N-grams between the system and reference summaries.
            ROUGE-L: Longest Common Subsequence (LCS) based statistics. Longest common 
                        subsequence problem takes into account sentence level structure
                        similarity naturally and identifies longest co-occurring in 
                        sequence n-grams automatically.
            ROUGE-W: Weighted LCS-based statistics that favors consecutive LCSes.
        """
        assert n in {1, 2, 3, 4, 5, "l"}
        evaluator = Rouge(metrics=[f"rouge-{t}"], max_n=n)
        return sum(max(evaluator.get_scores(pred, label)[f"rouge-{n}"][stats] for label in labels)
                   for pred, labels in zip(self.preds, self.labels)) / len(self.preds)


class EvidenceEvaluation:
    def __init__(self, preds: Sequence[Sequence[tuple[float, float]]],
                 labels: Sequence[Sequence[tuple[float, float]]]) -> None:
        self.preds = preds
        self.labels = labels

    @staticmethod
    def _f1(_p: float, _r: float) -> float:
        if _p == 0 or _r == 0:
            return 0
        return 2 * _p * _r / (_p + _r)

    @staticmethod
    def _calculate_iou(span_1: tuple[float, float], span_2: tuple[float, float]) -> float:
        num = len(
            set(range(round(span_1[0]), round(span_1[1])))
            & set(range(round(span_2[0]), round(span_2[1])))
        )
        denominator = len(
            set(range(round(span_1[0]), round(span_1[1])))
            | set(range(round(span_2[0]), round(span_2[1])))
        )
        return 0 if denominator == 0 else num / denominator

    def iou_f1(self, threshold: float = 0.5, pred_threshold: float = 1.0) -> float:
        """
        IoU-F1
        
        Code originally implemented for paper Micromodels for Efficient, Explainable, and Reusable Systems:
            A Case Study on Mental Health by Lee et al.
        Reference: https://github.com/MichiganNLP/micromodels/blob/empathy/empathy/run_experiments.py#L292-L333
        
        threshold: take into account the prediction when its iou with gold span is larger than this threshold.
        pred_threshold: if two predictions' IoU smaller than it, will only consider the prediction span with the
        larger value. '1' means that all prediction spans will be considered.
        """
        assert len(self.preds) == len(self.labels)

        f1_sum = 0

        for pred_spans, gold_spans in zip(self.preds, self.labels):
            if len(pred_spans) == 0 and len(gold_spans) == 0:
                micro_f1 = 1
            else:
                iou_s = {i: max((self._calculate_iou(pred_span, gold_span) for gold_span in gold_spans), default=0)
                         for i, pred_span in enumerate(pred_spans)}

                for i in range(len(pred_spans)):  # Delete the repeated predictions.
                    ps_1 = pred_spans[i]
                    for j in range(i + 1, len(pred_spans)):
                        ps_2 = pred_spans[j]
                        if self._calculate_iou(ps_1, ps_2) > pred_threshold:
                            del iou_s[j if iou_s[i] >= iou_s[j] else i]

                threshold_tps = sum(x >= threshold for x in iou_s.values())
                micro_p = threshold_tps / len(pred_spans) if len(pred_spans) > 0 else 0
                micro_r = threshold_tps / len(gold_spans) if len(gold_spans) > 0 else 0
                micro_f1 = self._f1(micro_p, micro_r)

            f1_sum += micro_f1

        return f1_sum / len(self.preds)


###########################################################


def test() -> None:
    TEST_SOURCE = ["he began by starting"]
    TEST_PREDS = ["he began by starting"]
    TEST_LABELS = [["he began by asd", "he began asd ads"]]

    TEST_EVIDENCE_PREDS = [[(1.2, 3.1), (4.5, 6.7)]]
    TEST_EVIDENCE_LABELS = [[(1.2, 3.5), (2.3, 5.0)]]

    qa_evaluation = QAEvaluation(TEST_SOURCE, TEST_PREDS, TEST_LABELS)
    print(qa_evaluation.exact_match())
    print(qa_evaluation.bleu(2))
    print(qa_evaluation.rouge(2))

    evidence_evaluation = EvidenceEvaluation(TEST_EVIDENCE_PREDS, TEST_EVIDENCE_LABELS)
    print(evidence_evaluation.iou_f1())


if __name__ == "__main__":
    test()
