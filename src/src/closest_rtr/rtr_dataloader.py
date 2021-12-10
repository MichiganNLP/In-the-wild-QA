import json
from typing import Any, Mapping

from src.dataloader import VQADataset


class RTRDataset(VQADataset):
    def __init__(self, data_dir: str, embedding_model) -> None:
        super().__init__(data_dir)
        self.embedding_model = embedding_model

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        return {"source": self.inputs[i], "target": self.targets[i], "source_embeddings": self.input_embeddings[i]}

    def _build(self) -> None:
        with open(self.data_dir) as f:
            data = json.load(f)

        # corpus of the answers
        self.inputs.extend(question["question"] for d in data for question in d["questions"])
        self.targets.extend(question["answers"] for d in data for question in d["questions"])

        self.input_embeddings = [self.embedding_model.encode(itm, convert_to_tensor=True) for itm in self.inputs]
