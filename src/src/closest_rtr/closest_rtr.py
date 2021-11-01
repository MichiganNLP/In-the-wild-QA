
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import ConcatDataset
import numpy as np

from src.closest_rtr.rtr_dataloader import RTRDataset
from src.evaluations.evaluations import evaluate
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
    
    evaluate("Closest Retrieval Text", preds, test_data)
    