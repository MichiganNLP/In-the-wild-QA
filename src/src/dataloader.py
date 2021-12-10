import json
from typing import Any, Mapping

from torch.utils.data import Dataset


class VQADataset(Dataset):
    def __init__(self, data_dir: str) -> None:
        self.data_dir = data_dir

        self.inputs = []
        self.targets = []
        self.target_periods = []
        self.durations = []

        self._build()

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        return {
            "source": self.inputs[index],
            "target": self.targets[index],
            "target_period": self.target_periods[index],
            "duration": self.durations[index],
        }

    def _build(self) -> None:
        with open(self.data_dir) as f:
            data = json.load(f)

        # corpus of the answers
        self.inputs.extend(d["question"] for d in data)
        self.targets.extend(d["answers"] for d in data)
        self.target_periods.extend([[float(v[0]), float(v[1])] for span in d["evidences"] for v in span.values()]
                                   for d in data)
        self.durations.extend(float(d["duration"]) for d in data)

        assert len(self.inputs) == len(self.targets) == len(self.target_periods) == len(self.durations)
