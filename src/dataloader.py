import re
import os
import glob
import json
from torch.utils.data import Dataset


def get_dataset(tokenizer, type_path, args):
    return DBDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)


class VQADataset(Dataset):
    def __init__(self, data_dir, is_test=False, embedding_model=None, tokenizer=None, max_len=512):
        self.data_dir = data_dir
        self.is_test = is_test
        
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []
        self.embedding_model = embedding_model

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