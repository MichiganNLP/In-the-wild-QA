import argparse
import multiprocessing

import sentence_transformers.util
import torch

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = multiprocessing.cpu_count() // max(torch.cuda.device_count(), 1)


def evaluate_closest_rtr(args: argparse.Namespace) -> None:
    data_module = VideoQAWithEvidenceDataModule(args)
    train_data_loader = data_module.train_dataloader()
    test_data_loader = data_module.test_dataloader()

    embedding_model = sentence_transformers.SentenceTransformer(args.embedding_model)
    embedding_model.eval()

    with torch.inference_mode():
        train_embeddings = embedding_model.encode([question_instance for batch in train_data_loader
                                                   for question_instance in batch["question"]],
                                                  convert_to_tensor=True)
        test_embeddings = embedding_model.encode([question_instance for batch in test_data_loader
                                                  for question_instance in batch["question"]],
                                                 convert_to_tensor=True)

        similarity_scores = sentence_transformers.util.pytorch_cos_sim(test_embeddings, train_embeddings)

        train_answers = [answer_instance for batch in train_data_loader for answer_instance in batch["answer"]]

        most_similar_ids = similarity_scores.argmax(dim=-1)
        predictions = [train_answers[most_similar_id.item()] for most_similar_id in most_similar_ids]

        evaluate_qa("Closest Retrieval Text", predictions, test_data_loader.dataset)
