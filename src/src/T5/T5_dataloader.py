import json
import torch

from src.dataloader import VQADataset
from src.utils.utils import read_hdf5
from torch.nn.utils.rnn import pad_sequence


def get_dataset(tokenizer, data_dir, args):
    if args.model_type == "T5_text_and_visual":
        return T5Dataset(data_dir=data_dir, tokenizer=tokenizer, include_visual=True, max_len=args.max_seq_length, \
            max_vid_len=args.max_vid_length, path_to_visual_file=args.path_to_visual_file, visual_size=args.visual_size)
    return T5Dataset(data_dir=data_dir, tokenizer=tokenizer, max_len=args.max_seq_length)


def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = int(lengths.max())
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


class T5Dataset(VQADataset):

    def __init__(self, data_dir, tokenizer, is_test=False, is_zero_shot=False, include_visual=False, \
        max_len=512, max_vid_len=None, path_to_visual_file=None, visual_size=None):

        self.tokenizer = tokenizer
        self.is_test = is_test
        self.is_zero_shot = is_zero_shot
        self.max_len = max_len
        self.include_visual = include_visual
        self.max_vid_len = max_vid_len
        self.visual_features = read_hdf5(path_to_visual_file)

        self._build_visual_features()

        self.visual_size = visual_size
        assert all([x.shape[-1] == self.visual_size for _, x in self.visual_features.items()])
        self.input_visuals = []
        super(T5Dataset, self).__init__(data_dir)

    def _build_visual_features(self):
        x = pad_sequence([torch.tensor(x[:self.max_vid_len, :]) for x in self.visual_features.values()], batch_first=True)
        lengths = torch.Tensor([min(itm.shape[0], self.max_vid_len) for itm in self.visual_features.values()])
        attention_masks = get_mask_from_sequence_lengths(lengths)
        assert attention_masks.shape[0] == x.shape[0]
        self.visual_attention_masks = {}
        for i, k in zip(range(x.shape[0]), self.visual_features):
            self.visual_features[k] = x[i]
            self.visual_attention_masks[k] = attention_masks[i]
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()    # might need to squeeze

        visual_ids = torch.Tensor([])
        visual_mask = torch.Tensor([])
        target_ids = torch.Tensor([])
        target_mask = torch.Tensor([])

        if self.include_visual:
            visual_ids = self.input_visuals[index]["input_ids"].squeeze()
            visual_mask = self.input_visuals[index]["attention_mask"].squeeze()

        if not self.is_test:
            target_ids = self.targets[index]["input_ids"].squeeze()
            target_mask = self.targets[index]["attention_mask"].squeeze()    # might need to squeeze

        return {"source_ids": source_ids, 
                "source_mask": src_mask, 
                "visual_ids": visual_ids,
                "visual_mask": visual_mask,
                "target_ids": target_ids, 
                "target_mask": target_mask}

    def _build(self):
        self._buil_examples_from_files()
    
    def _buil_examples_from_files(self):
        with open(self.data_dir, 'r') as f:
            data = json.load(f)
        
        for d in data:
            start_time = d["start_time"]
            end_time = d["end_time"]
            video_id = d["video_id"]

            for itm in d["questions"]:
                question = itm["question"]

                if self.is_zero_shot:
                    question += " <extra_id_0>"
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

                    if self.include_visual:
                        self.input_visuals.append({
                            "input_ids": self.visual_features[video_id],
                            "attention_mask": self.visual_attention_masks[video_id]})

                    self.targets.append(tokenized_targets)

                else:
                    # for testing time, inputs should not be repeatly added 
                    tokenized_inputs = self.tokenizer.batch_encode_plus(
                            [question], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                        )
                    self.inputs.append(tokenized_inputs)
                    if self.include_visual:
                        self.input_visuals.append({
                            "input_ids": self.visual_features[video_id],
                            "attention_mask": self.visual_attention_masks[video_id]})
