import json

from dataloader import VQADataset

def get_dataset(tokenizer, data_dir, args):
    return T5Dataset(tokenizer=tokenizer, data_dir=data_dir, max_len=args.max_seq_length)


class T5Dataset(VQADataset):

    def __init__(self, data_dir, tokenizer, is_test=False, is_zero_shot=False, max_len=512):
        self.tokenizer = tokenizer
        self.is_test = is_test
        self.is_zero_shot = is_zero_shot
        self.max_len = max_len
        super(T5Dataset, self).__init__(data_dir)
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()    # might need to squeeze

        if not self.is_test:
            target_ids = self.targets[index]["input_ids"].squeeze()
            target_mask = self.targets[index]["attention_mask"].squeeze()    # might need to squeeze

            return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}
        else:
            return {"source_ids": source_ids, "source_mask": src_mask}
    
    def _build(self):
        self._buil_examples_from_files()
    
    def _buil_examples_from_files(self):
        with open(self.data_dir, 'r') as f:
            data = json.load(f)
        
        for d in data:
            for itm in d["questions"]:
                question = itm["question"]

                if self.is_zero_shot:
                    question += "[question] <extra_id_0>"
                else:
                    question += " </s>"
                    answer = itm["answers"]
                    answer += " </s>"

                if not self.is_test:
                    # tokenize inputs
                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                            [question], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                    )

                    # tokenize targets
                    tokenized_targets = self.tokenizer.batch_encode_plus(
                            [answer], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                    )

                    # for training, each input and target will be added
                    self.inputs.append(tokenized_inputs)
                    self.targets.append(tokenized_targets)

                else:
                    # for testing time, inputs should not be repeatly added 
                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                            [question], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                        )
                    self.inputs.append(tokenized_inputs)
