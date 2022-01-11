import json

import numpy as np
import torch
from src.dataloader import VQADataset
from src.utils.utils import isfloat, read_hdf5
from torch.nn.utils.rnn import pad_sequence


def get_dataset(tokenizer, data_dir, args):
    if args.model_type == "T5_text_and_visual":
        return T5Dataset(data_dir=data_dir, tokenizer=tokenizer, include_visual=True, max_len=args.max_seq_length, \
            max_vid_len=args.max_vid_length, path_to_visual_file=args.path_to_visual_file, visual_size=args.visual_size, \
            sample_rate=args.sample_rate)
    elif args.model_type == "T5_evidence":
        return T5Dataset(data_dir=data_dir, tokenizer=tokenizer, include_visual=True, max_len=args.max_seq_length, \
            max_vid_len=args.max_vid_length, path_to_visual_file=args.path_to_visual_file, visual_size=args.visual_size, \
            sample_rate=args.sample_rate, is_evidence=True)
    return T5Dataset(data_dir=data_dir, tokenizer=tokenizer, max_len=args.max_seq_length)


def get_mask_from_sequence_lengths(lengths: torch.Tensor) -> torch.Tensor:
    max_length = int(lengths.max())
    return torch.arange(max_length).expand(len(lengths), max_length) < lengths.unsqueeze(1)  # noqa


class T5Dataset(VQADataset):

    def __init__(self, data_dir, tokenizer, is_test=False, is_zero_shot=False, is_evidence=False, \
        include_visual=False, max_len=512, max_vid_len=None, path_to_visual_file=None, \
        visual_size=None, sample_rate=None):

        self.tokenizer = tokenizer
        self.is_test = is_test
        self.is_zero_shot = is_zero_shot
        self.is_evidence = is_evidence

        self.max_len = max_len
        self.include_visual = include_visual
        self.max_vid_len = max_vid_len
        self.visual_features = read_hdf5(path_to_visual_file)

        self.sample_rate = sample_rate
        self.visual_size = visual_size

        self._build_visual_features()

        assert all([x.shape[-1] == self.visual_size for _, x in self.visual_features.items()])
        self.input_visuals = []
        self.evidences = []

        super(T5Dataset, self).__init__(data_dir)

    def _build_visual_features(self):
        
        sample_rate = self.sample_rate

        vfs = self.visual_features.values()

        if sample_rate:
            # Sample video features
            vfs = []
            for x in self.visual_features.values():
                repeat_times = (sample_rate - x.shape[0] % sample_rate) % sample_rate
                supplement_itms = np.repeat(x[-1, :].reshape((1, -1)), repeat_times, axis=0)
                full_x = np.concatenate((x, supplement_itms), axis=0)
                downsampled_x = np.mean(full_x.reshape(-1, sample_rate, self.visual_size), axis=1)
                vfs.append(downsampled_x)
        
        # Trim video features
        processed_features = pad_sequence([torch.tensor(x[:self.max_vid_len, :]) for x in vfs], batch_first=True)
        lengths = torch.Tensor([min(itm.shape[0], self.max_vid_len) for itm in vfs])

        attention_masks = get_mask_from_sequence_lengths(lengths)
        assert attention_masks.shape[0] == processed_features.shape[0]
        self.visual_attention_masks = {}
        for i, k in zip(range(processed_features.shape[0]), self.visual_features):
            self.visual_features[k] = processed_features[i]
            self.visual_attention_masks[k] = attention_masks[i]
    
    def __getitem__(self, index):
        source_ids = self.inputs[index]["input_ids"].squeeze()
        src_mask = self.inputs[index]["attention_mask"].squeeze()    # might need to squeeze

        visual_ids = torch.Tensor([])
        visual_mask = torch.Tensor([])
        target_ids = torch.Tensor([])
        target_mask = torch.Tensor([])
        evidences = []

        if self.include_visual:
            visual_ids = self.input_visuals[index]["input_ids"].squeeze()
            visual_mask = self.input_visuals[index]["attention_mask"].squeeze()

        if not self.is_test and not self.is_evidence:
            target_ids = self.targets[index]["input_ids"].squeeze()
            target_mask = self.targets[index]["attention_mask"].squeeze()    # might need to squeeze
        
        if not self.is_test and self.is_evidence:
            evidences = self.evidences[index]

        return {"source_ids": source_ids, 
                "source_mask": src_mask, 
                "visual_ids": visual_ids,
                "visual_mask": visual_mask,
                "target_ids": target_ids, 
                "target_mask": target_mask,
                "evidence": evidences}

    def _build(self):
        self._buil_examples_from_files()
    
    def _buil_examples_from_files(self):
        with open(self.data_dir, 'r') as f:
            data = json.load(f)
        
        for d in data:
            question = d["question"]
            
            # NOTE: for debugging
            # video_id = d["video_id"]

            if self.is_zero_shot:
                question += " <extra_id_0>"
            elif not self.is_evidence:
                question += " </s>"
                answer = d["answers"]
                answer += " </s>"

            if not self.is_test:
                # tokenize inputs
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                        [question], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                )

                # start_with_zero = False

                if self.is_evidence:
                    # only return the start and end position as the int
                    evidences = d["evidences"]
                    this_example_evidences = []
                    
                    #NOTE: debugging purpose, only use the single evidence
                    for evidence in [evidences[0]]:
                        for _, [start_time, end_time] in evidence.items():
                            assert isfloat(start_time) and isfloat(end_time)
                            # if round(float(start_time)) == 0:
                            #     start_with_zero = True
                            this_example_evidences.append([round(float(start_time)), round(float(end_time))])
                    # if start_with_zero:
                    #     continue
                    self.evidences.append(this_example_evidences)

                else:
                    # not finding evidence, thus tokenize targets
                    tokenized_targets = self.tokenizer.batch_encode_plus(
                            [answer], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                    )
                    self.targets.append(tokenized_targets)

                # for training, each input and target will be added
                self.inputs.append(tokenized_inputs)
            
                # NOTE: for debugging
                # if self.include_visual:
                #     self.input_visuals.append({
                #         "input_ids": self.visual_features[video_id],
                #         "attention_mask": self.visual_attention_masks[video_id]})
                context = d["context"]
                tokenized_context = self.tokenizer.batch_encode_plus(
                        [context], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                )
                self.input_visuals.append(tokenized_context)

            else:
                # ###############################
                # # NOTE: just for debugging
                # start_with_zero = False

                # if self.is_evidence:
                #     # only return the start and end position as the int
                #     evidences = d["evidences"]
                #     this_example_evidences = []
                    
                #     #NOTE: debugging purpose, only use the single evidence
                #     for evidence in [evidences[0]]:
                #         for _, [start_time, end_time] in evidence.items():
                #             assert isfloat(start_time) and isfloat(end_time)
                #             if round(float(start_time)) == 0:
                #                 start_with_zero = True
                #             this_example_evidences.append([round(float(start_time)), round(float(end_time))])
                #     if start_with_zero:
                #         continue
                # #####################################
                # for testing time, inputs should not be repeatly added 
                tokenized_inputs = self.tokenizer.batch_encode_plus(
                        [question], max_length=self.max_len, pad_to_max_length=True, return_tensors="pt"
                    )
                self.inputs.append(tokenized_inputs)
                if self.include_visual:
                    self.input_visuals.append({
                        "input_ids": self.visual_features[video_id],
                        "attention_mask": self.visual_attention_masks[video_id]})
