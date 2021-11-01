from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from rouge import Rouge
import warnings

warnings.filterwarnings("ignore")   # filter user warning for BLEU when overlap is 0


def evaluate(model_name, preds, test_data):
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


class Evaluation():

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



TEST_SOURCE = ["he began by starting"]
TEST_PREDS = ["he began by starting"]
TEST_LABELS = [["he began by asd", "he began asd ads"]]


if __name__ == "__main__":
    evl = Evaluation(TEST_SOURCE, TEST_PREDS, TEST_LABELS)
    print(evl.exact_match())
    print(evl.BLEU(2))
    print(evl.ROUGE(2))
    








