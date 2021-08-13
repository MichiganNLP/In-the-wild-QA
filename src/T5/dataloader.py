import re
import os
import glob
import json
from torch.utils.data import Dataset, DataLoader


def get_dataset(tokenizer, type_path, args):
    return DBDataset(tokenizer=tokenizer, data_dir=args.data_dir, type_path=type_path, max_len=args.max_seq_length)


class DBDataset(Dataset):
    def __init__(self, tokenizer, data_dir, type_path, max_len=512, test=False):
        self.test=test
        self.train_data_dir = os.path.join(data_dir, f'{type_path}.json')
        
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.inputs = []
        self.targets = []

        self._build()
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()    # might need to squeeze

        if not self.test:
            target_ids = self.targets[index]["input_ids"].squeeze()
            target_mask = self.targets[index]["attention_mask"].squeeze()    # might need to squeeze

            return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        else:
            return {"source_ids": source_ids, "source_mask": src_mask}
    
    def _build(self):
        self._buil_examples_from_files(self.train_data_dir)
    
    def _buil_examples_from_files(self, data):
            with open(self.train_data_dir, 'r') as f:
                    data = json.load(f)
            
            for itm in data:
                prefix = itm["question"]
                prefix += " </s>"

                if not self.test:
                    # tokenize inputs
                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                            [prefix], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                    )
                    for l in itm["labels"]:
                        sql = l["query"]
                        sql += " </s>"

                        # tokenize targets
                        tokenized_targets = self.tokenizer.batch_encode_plus(
                                [sql], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                        )

                        # for training, each input and target will be added
                        self.inputs.append(tokenized_inputs)
                        self.targets.append(tokenized_targets)
                else:
                    # for testing time, inputs should not be repeatly added 
                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                            [prefix], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                        )
                    self.inputs.append(tokenized_inputs)
