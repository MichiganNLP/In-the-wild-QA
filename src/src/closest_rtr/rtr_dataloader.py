import json
from typing import Any, Mapping

from src.dataloader import VQADataset


class RTRDataset(VQADataset):
    def __init__(self, data_dir: str, embedding_model) -> None:
        self.embedding_model = embedding_model
        super().__init__(data_dir)

    def __getitem__(self, i: int) -> Mapping[str, Any]:
        return {"source": self.inputs[i], "target": self.targets[i], "source_embeddings": self.input_embeddings[i]}

    def _build(self) -> None:
        with open(self.data_dir) as f:
            data = json.load(f)

        # corpus of the answers
        self.inputs.extend(d["question"] for d in data)
        self.targets.extend(d["answer"] for d in data)

        self.input_embeddings = [self.embedding_model.encode(itm, convert_to_tensor=True) for itm in self.inputs]
