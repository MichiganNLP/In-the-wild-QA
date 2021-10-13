from nltk.translate.bleu_score import sentence_bleu
from nltk import word_tokenize
from rouge import Rouge
import warnings

warnings.filterwarnings("ignore")   # filter user warning for BLEU when overlap is 0


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
        
    
    def ROUGE(self, N, stats='p'):
        """ 
        stats: 'p': precision; 'r': recall; 'f': f1
        """
        assert N in [1, 2, 3, 4, 5, 'l']
        evaluator = Rouge(metrics=[f'rouge-{N}'])

        tt_rouge = 0
        for pred, labels in zip(self.preds, self.labels):
            rouge = []
            for label in labels:
                score = evaluator.get_scores(pred, label)
                rouge.append(score[0][f'rouge-{N}'][stats])
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
    








