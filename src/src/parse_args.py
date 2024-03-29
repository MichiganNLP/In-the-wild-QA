import os
from dataclasses import dataclass, field
from typing import Iterable, Optional, Union

import torch
from transformers.hf_argparser import DataClassType

# Can't use `from __future__ import annotations` here. See https://github.com/huggingface/transformers/pull/15795
# From the next version of transformers (after v4.17.0) it should be possible.


MODEL_CHOICES = [
    "random",
    "most_common_ans",
    "closest_rtr",
    "t5_train",
    "t5_zero_shot",
    "t5_text_and_visual",
    "t5_evidence",
    "t5_evidence_io",
    "t5_multi_task",
    "clip_decoder",
]

EMBEDDING_CHOICES = [
    "stsb-roberta-base",
    "stsb-bert-large",
    "stsb-distilbert-base",
    "stsb-roberta-large",
]


@dataclass
class TrainAndTestArguments:
    train_data_path: str = "example_data/wildQA-data/dev.json"
    dev_data_path: str = "example_data/wildQA-data/test.json"
    num_workers: int = len(os.sched_getaffinity(0)) // max(torch.cuda.device_count(), 1)
    output_ckpt_dir: Optional[str] = None
    model_name_or_path: str = "t5-base"
    max_seq_length: int = field(
        default=512,
        metadata={"help": "maximum of the text sequence. Truncate if exceeded."}
    )
    beam_size: Optional[int] = None
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    train_batch_size: Optional[int] = 32
    eval_batch_size: Optional[int] = 32
    num_train_epochs: int = 100
    gradient_accumulation_steps: Optional[int] = None
    n_gpu: int = 1
    early_stop_callback: bool = field(
        default=False,
        metadata={"help": "whether we allow early stop in training."}
    )
    opt_level: Optional[int] = field(
        default=None,
        metadata={"help": "optimization level. you can find out more on optimisation levels here"
                          " https://nvidia.github.io/apex/amp.html#opt-levels-and-properties"}
    )
    fp_16: bool = field(
        default=False,
        metadata={"help": "if you want to enable 16-bit training then install apex and set this to true."}
    )
    max_grad_norm: float = 1.0
    seed: int = 42
    profiler: Optional[str] = None
    use_tpu: bool = False
    wandb_project: str = "In-the-wild-QA"
    wandb_name: str = field(
        default=None,
        metadata={"help": "name of this run."}
    )
    wandb_entity: str = field(
        default="in-the-wild-vqa-um",
        metadata={"help": "your account to for wandb."}
    )
    wandb_offline: bool = field(
        default=False,
        metadata={"help": "if set true, we will not have wandb record online"}
    )


@dataclass
class ClosestRetrievalArguments:
    embedding_model: str = field(
        default="stsb-roberta-base",
        metadata={"help": "model types for calculating embedding, more models available at "
                          "https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/"
                          "edit#gid=0"}
    )

    def __post_init__(self):
        if self.embedding_model not in EMBEDDING_CHOICES:
            raise ValueError(f"Please select from {', '.join(EMBEDDING_CHOICES)}")


@dataclass
class _VisualBaseArguments:
    visual_avg_pool_size: Optional[int] = field(
        default=None,
        metadata={"help": "sampling rate for visual features. Only Used for visual features."}
    )
    visual_features_path: str = "https://www.dropbox.com/s/vt2kjdqr7mnxg2q/WildQA_I3D_avg_pool.hdf5?dl=1"
    visual_size: int = field(
        default=1024,
        metadata={"help": "visual embedding dimension."}
    )
    max_vid_length: int = field(
        default=2048,
        metadata={"help": "maximum length of the visual input. Truncate the exceeded part."}
    )


@dataclass
class T5TextVisualTrainArguments(_VisualBaseArguments):
    pass


#########################################################################################
#########################################################################################
# Argument classes for Video Evidence Finding part.


@dataclass
class RandomArguments:
    span_prediction_count: int = field(
        default=5,
        metadata={"help": "number of predicted evidence"}
    )


@dataclass
class T5EvidenceFindingTrainArguments(T5TextVisualTrainArguments):
    pass


@dataclass
class T5EvidenceIOTrainArguments(T5TextVisualTrainArguments):
    pass


#########################################################################################
#########################################################################################
# Argument classes for Multi-Tasking part.


@dataclass
class T5MultiTaskTrainArguments(T5TextVisualTrainArguments):
    vqa_weight: float = field(
        default=1.0,
        metadata={"help": "weight for VQA part"}
    )
    evidence_weight: float = field(
        default=1.0,
        metadata={"help": "weight for evidence finding part"}
    )


#########################################################################################
#########################################################################################
# Argument classes for the CLIP models


@dataclass
class CLIPDecoderBasics:
    pretrained_clip_ckpt_path: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "ckpt name to the pre-trained CLIP model"}
    )
    frames_path: str = field(
        default="/scratch/mihalcea_root/mihalcea0/shared_data/video_features/frames",
        metadata={"help": "path to the directory that contains all the extracted frames"}
    )


@dataclass
class CLIPDecoderTrainArguments(CLIPDecoderBasics, T5TextVisualTrainArguments):
    pass


def model_type_to_dataclass_types(model_type: str) -> Union[DataClassType, Iterable[DataClassType]]:
    return {
        "random": [TrainAndTestArguments, RandomArguments],
        "most_common_ans": TrainAndTestArguments,
        "closest_rtr": [TrainAndTestArguments, ClosestRetrievalArguments],
        "t5_train": TrainAndTestArguments,
        "t5_zero_shot": TrainAndTestArguments,
        "t5_text_and_visual": [TrainAndTestArguments, T5TextVisualTrainArguments],
        "t5_evidence": [TrainAndTestArguments, T5EvidenceFindingTrainArguments],
        "t5_evidence_io": [TrainAndTestArguments, T5EvidenceIOTrainArguments],
        "t5_multi_task": [TrainAndTestArguments, T5MultiTaskTrainArguments],
        "clip_decoder": [TrainAndTestArguments, CLIPDecoderTrainArguments],
    }[model_type]
