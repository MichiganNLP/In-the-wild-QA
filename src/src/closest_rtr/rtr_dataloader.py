from collections.abc import Mapping
from typing import Any

from overrides import overrides
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataset


class RTRDataset(VideoQAWithEvidenceDataset):
    def __init__(self, data_dir: str, tokenizer: PreTrainedTokenizerBase, embedding_model: SentenceTransformer) -> None:
        super().__init__(data_dir, tokenizer)
        self.embedding_model = embedding_model

        self.inputs = [d["question"] for d in self.instances]
        self.targets = [d["answer"] for d in self.instances]

        self.input_embeddings = [self.embedding_model.encode(input_, convert_to_tensor=True)
                                 for input_ in self.inputs]

    @overrides
    def __getitem__(self, i: int) -> Mapping[str, Any]:
        return {"source": self.inputs[i], "target": self.targets[i], "source_embeddings": self.input_embeddings[i]}
