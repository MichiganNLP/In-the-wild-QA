import json
from dataloader import VQADataset


class RTRDataset(VQADataset):

    def __init__(self, data_dir, embedding_model):
        self.embedding_model = embedding_model
        super(RTRDataset, self).__init__(data_dir)

    def __getitem__(self, index):
        return {"source": self.inputs[index], "target": self.targets[index], "source_embeddings": self.input_embeddings[index]}
    
    def _build(self):
        with open(self.data_dir, 'r') as f:
            data = json.load(f)
        
        # corpus of the answers
        self.inputs.extend([question["question"] for d in data for question in d["questions"]])
        self.targets.extend([question["answers"] for d in data for question in d["questions"]])

        self.input_embeddings = [self.embedding_model.encode(itm, convert_to_tensor=True) for itm in self.inputs]
