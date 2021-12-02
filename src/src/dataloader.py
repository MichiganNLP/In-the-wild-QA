import re
import os
import glob
import json
from torch.utils.data import Dataset



class VQADataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        
        self.inputs = []
        self.targets = []
        self.target_periods = []
        self.durations = []

        self._build()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return {
            "source": self.inputs[index], 
            "target": self.targets[index],
            "target_period": self.target_periods[index],
            "duration": self.durations[index]
        }
    
    def _build(self):
        with open(self.data_dir, 'r') as f:
            data = json.load(f)
        
        # corpus of the answers
        self.inputs.extend([d["question"] for d in data])
        self.targets.extend([d["correct_answer"] for d in data])
        self.target_periods.extend([[[float(v[0]), float(v[1])] for span in d["evidences"] for v in span.values()] for d in data])
        self.durations.extend([float(d["duration"]) for d in data])
        assert len(self.inputs) == len(self.targets) == len(self.target_periods) == len(self.durations)
