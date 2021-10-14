
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import ConcatDataset
import numpy as np

from closest_rtr.rtr_dataloader import RTRDataset
from evaluations.evaluations import Evaluation
from tqdm import tqdm


def closest_rtr(args):

    embedding_model = SentenceTransformer(args.embedding_model)

    train_data = RTRDataset(args.train_data, embedding_model=embedding_model)
    dev_data = RTRDataset(args.dev_data, embedding_model=embedding_model)

    # NOTE: here we only use train data as the corpus
    # as dev data is the same as test data in our testing
    # train_dev_data = ConcatDataset([train_data, dev_data])

    test_data = RTRDataset(args.test_data, embedding_model=embedding_model)

    preds = []
    for test_d in tqdm(iter(test_data)):
        question_emb = test_d['source_embeddings']
        similarity_scores = [util.pytorch_cos_sim(question_emb, itm["source_embeddings"]) for itm in train_data]
        max_idx = np.argmax(similarity_scores)

        pred = train_data[max_idx]['target']
        preds.append(pred)
    
    
    sources = [itm['source'] for itm in test_data]
    labels = [[itm['target']] for itm in test_data]
    evl = Evaluation(sources, preds, labels)    
    
    print("------------------Closest Retrieval Text Baseline----------------------")
    print(f"Exact Match: {round(evl.exact_match() * 100, 2)}%")
    print(f"BLEU 1: {round(evl.BLEU(1) * 100, 2)}%")
    print(f"BLEU 2: {round(evl.BLEU(2) * 100, 2)}%")
    print(f"BLEU 3: {round(evl.BLEU(3) * 100, 2)}%")
    print(f"ROUGE 1: {round(evl.ROUGE(1) * 100, 2)}%")
    print(f"ROUGE 2: {round(evl.ROUGE(2) * 100, 2)}%")
    print(f"ROUGE 3: {round(evl.ROUGE(3) * 100, 2)}%")
