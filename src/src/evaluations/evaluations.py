import warnings
from collections import defaultdict
from src.dataloader import VQADataset

import numpy as np
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from tqdm import tqdm

warnings.filterwarnings("ignore")   # filter user warning for BLEU when overlap is 0


def evaluate(model_name, preds, test_data):
    """ 
    Evaluation for QA
    """
    sources = [itm['source'] for itm in test_data]
    labels = [[itm['target']] for itm in test_data]

    evl = Evaluation(sources, preds, labels)    
    
    print(f"------------------{model_name} Baseline----------------------")
    print(f"Exact Match: {round(evl.exact_match() * 100, 2)}%")
    print(f"BLEU 1: {round(evl.BLEU(1) * 100, 2)}%")
    print(f"BLEU 2: {round(evl.BLEU(2) * 100, 2)}%")
    print(f"BLEU 3: {round(evl.BLEU(3) * 100, 2)}%")
    print(f"ROUGE 1: {round(evl.ROUGE(1) * 100, 2)}%")
    print(f"ROUGE 1: {round(evl.ROUGE(2) * 100, 2)}%")
    print(f"ROUGE 3: {round(evl.ROUGE(3) * 100, 2)}%")


def evidence_evaluation(model_name: str, preds: list, test_data: VQADataset):
    """ 
    Evaluation for evidence finding
    preds: list of list of span list, that is: [[[start1, end1], [start2, end2]], ...]
        The most inner list gives the predicted start, end
        The second inner list gives all the span predictions for an instance
        The outmost list is the predictions for all instances
    """
    gt_spans = [itm["target_period"] for itm in test_data]
    evl = Evidence_Evaluation(preds, gt_spans)    
    
    print(f"------------------{model_name} Baseline----------------------")
    print(f"IOU F1: {round(evl.iou_f1() * 100, 2)}%")


class Evaluation():
    """ Evaluation class for QA """

    def __init__(self, sources: list, preds: list, labels: list):
        
        assert isinstance(labels[0], list)
        self.sources = sources
        self.preds = preds
        self.labels = labels
        assert len(self.sources) == len(self.preds) == len(self.labels)

    def exact_match(self):
        corrects = 0
        for pred, label in zip(self.preds, self.labels):
            if pred in label:
                corrects += 1
        return corrects / len(self.preds)
    
    def BLEU(self,N):
        # individual BLEU N-gram score
        self.pred_toks = [word_tokenize(pred) for pred in self.preds]
        self.label_toks = [[word_tokenize(label) for label in llabels] for llabels in self.labels]

        assert N >=1 and N <= 4
        weights = [0, 0, 0, 0]
        weights[N - 1] = 1

        tt_bleus = 0

        for pred_tok, label_tok in zip(self.pred_toks, self.label_toks):
            bleu = sentence_bleu(label_tok, pred_tok, weights = tuple(weights))
            tt_bleus += bleu
        return tt_bleus / len(self.preds)
        
    
    def ROUGE(self, N, t='n', stats='p'):
        """ 
        stats: 'p': precision; 'r': recall; 'f': f1
        t: Rouge type:
            ROUGE-N: Overlap of N-grams between the system and reference summaries.
            ROUGE-L: Longest Common Subsequence (LCS) based statistics. Longest common 
                        subsequence problem takes into account sentence level structure
                        similarity naturally and identifies longest co-occurring in 
                        sequence n-grams automatically.
            ROUGE-W: Weighted LCS-based statistics that favors consecutive LCSes .
        """
        assert N in [1, 2, 3, 4, 5, 'l']
        evaluator = Rouge(metrics=[f'rouge-{t}'], max_n=N)

        tt_rouge = 0
        for pred, labels in zip(self.preds, self.labels):
            rouge = []
            for label in labels:
                score = evaluator.get_scores(pred, label)
                rouge.append(score[f'rouge-{N}'][stats])
            tt_rouge += max(rouge)
        return tt_rouge / len(self.preds)


class Evidence_Evaluation():
    """ Evaluation for evidence finding """

    def __init__(self, preds: list, labels: list):
        self.preds = preds 
        self.labels = labels

    def _f1(self, _p, _r):
        if _p == 0 or _r == 0:
            return 0
        return 2 * _p * _r / (_p + _r)

    def _calculate_iou(self, span_1, span_2):
        num = len(
                set(range(round(span_1[0]), round(span_1[1])))
                & set(range(round(span_2[0]), round(span_2[1])))
        )
        denom = len(
            set(range(round(span_1[0]), round(span_1[1])))
            | set(range(round(span_2[0]), round(span_2[1])))
        )
        iou = 0 if denom == 0 else num / denom
        return iou

    def iou_f1(self, threshold=0.5, pred_threshold=1):
        """
        IOU-F1
        
        Code originally implemented for paper Micromodels for Efficient, Explainable, and Reusable Systems:
            A Case Study on Mental Health by Lee et al.
        Reference: https://github.com/MichiganNLP/micromodels/blob/empathy/empathy/run_experiments.py#L292-L333
        
        threshold: take into account the prediction when its iou with gold span is larger than
            this threshold.
        pred_threshold: if two predictions' iou smaller than it, will only consider the predic-
            tion span with the larger value. '1' means that all prediction spans will be consi-
            dered.
        """
        predictions = self.preds
        gold = self.labels
        assert len(predictions) == len(gold)
        all_f1_vals = []

        for idx, pred_spans in tqdm(enumerate(predictions)):
            gold_spans = gold[idx]

            ious = defaultdict(float)

            # repeated predictions
            overlapped = list()

            for i in range(len(pred_spans)):
                ps_1 = pred_spans[i]
                for j in range(i + 1, len(pred_spans)):
                    ps_2 = pred_spans[j]
                    iou = self._calculate_iou(ps_1, ps_2)
                    if iou > pred_threshold:
                        overlapped.append((i, j))


            for i, pred_span in enumerate(pred_spans):
                best_iou = 0.0
                for gold_span in gold_spans:

                    iou = self._calculate_iou(pred_span, gold_span)

                    if iou > best_iou:
                        best_iou = iou
                ious[i] = best_iou
            
            # delete overlapped predictions
            for (i, j) in overlapped:
                assert i in ious and j in ious
                if ious[i] >= ious[j]:
                    del ious[j]
                else:
                    del ious[i]

            threshold_tps = sum(int(x >= threshold) for x in ious.values())

            micro_r = threshold_tps / len(gold_spans) if len(gold_spans) > 0 else 0
            micro_p = threshold_tps / len(pred_spans) if len(pred_spans) > 0 else 0
            micro_f1 = self._f1(micro_r, micro_p)
            if len(pred_spans) == 0 and len(gold_spans) == 0:
                all_f1_vals.append(1)
            else:
                all_f1_vals.append(micro_f1)

        return np.mean(all_f1_vals)



###########################################################
###########################################################
# testing for the evaluation class


TEST_SOURCE = ["he began by starting"]
TEST_PREDS = ["he began by starting"]
TEST_LABELS = [["he began by asd", "he began asd ads"]]

TEST_EVIDENCE_PREDS = [[[1.2, 3.1], [4.5, 6.7]]]
TEST_EVIDENCE_LABELS = [[[1.2, 3.5], [2.3, 5.0]]]


if __name__ == "__main__":
    evl = Evaluation(TEST_SOURCE, TEST_PREDS, TEST_LABELS)
    print(evl.exact_match())
    print(evl.BLEU(2))
    print(evl.ROUGE(2))

    ev_evl = Evidence_Evaluation(TEST_EVIDENCE_PREDS, TEST_EVIDENCE_LABELS)
    print(ev_evl.iou_f1())
    








