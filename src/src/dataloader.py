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

        self._build()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return {"source": self.inputs[index], "target": self.targets[index]}
    
    def _build(self):
        with open(self.data_dir, 'r') as f:
            data = json.load(f)
        
        # corpus of the answers
        self.inputs.extend([question["question"] for d in data for question in d["questions"]])
        self.targets.extend([question["answers"] for d in data for question in d["questions"]])