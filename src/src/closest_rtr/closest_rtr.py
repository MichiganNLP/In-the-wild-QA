import argparse

import sentence_transformers
import torch

from src.evaluations.evaluations import evaluate_qa
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_closest_rtr(args: argparse.Namespace) -> None:
    data_module = VideoQAWithEvidenceDataModule(args)
    train_data_loader = data_module.train_dataloader()
    test_data_loader = data_module.test_dataloader()

    embedding_model = sentence_transformers.SentenceTransformer(args.embedding_model)
    embedding_model.eval()
    embedding_model.to(DEVICE)

    model_kwargs = {"convert_to_tensor": True, "normalize_embeddings": True, "device": DEVICE}

    with torch.inference_mode():
        train_embeddings = torch.cat([embedding_model.encode(batch["question"], **model_kwargs)
                                      for batch in train_data_loader])
        test_embeddings = torch.cat([embedding_model.encode(batch["question"], **model_kwargs)
                                     for batch in test_data_loader])

        similarity_scores = test_embeddings @ train_embeddings.T

        train_answers = [answer_instance for batch in train_data_loader for answer_instance in batch["answer"]]

        most_similar_ids = similarity_scores.argmax(dim=-1)
        preds = [train_answers[most_similar_id.item()] for most_similar_id in most_similar_ids]

        evaluate_qa("Closest Retrieval Text", preds, test_data_loader.dataset)
