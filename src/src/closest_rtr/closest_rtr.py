import argparse
import os

import numpy as np
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
from transformers import AutoTokenizer

from src.closest_rtr.rtr_dataloader import RTRDataset
from src.evaluations.evaluations import evaluate_qa


def evaluate_closest_rtr(args: argparse.Namespace) -> None:
    embedding_model = SentenceTransformer(args.embedding_model)

    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    train_data_path = RTRDataset(args.train_data_path, tokenizer=tokenizer, embedding_model=embedding_model)

    dataset = RTRDataset(args.test_data_path, tokenizer=tokenizer, embedding_model=embedding_model)

    preds = []
    for instance in tqdm(dataset):
        question_emb = instance["source_embeddings"]
        similarity_scores = [util.pytorch_cos_sim(question_emb, instance["source_embeddings"])
                             for instance in train_data_path]
        max_idx = np.argmax([s.cpu() for s in similarity_scores]).item()

        pred = train_data_path[max_idx]["target"]
        preds.append(pred)

    evaluate_qa("Closest Retrieval Text", preds, dataset)
