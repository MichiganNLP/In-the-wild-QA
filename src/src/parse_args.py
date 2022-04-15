import multiprocessing
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional, Union

import torch
from transformers.hf_argparser import DataClassType


# Can't use `from __future__ import annotations` here. See https://github.com/huggingface/transformers/pull/15795
# From the next version of transformers (after v4.17.0) it should be possible.


MODEL_CHOICES = [
    "random",
    "most_common_ans",
    "closest_rtr",
    "T5_train",
    "T5_zero_shot",
    "T5_text_and_visual",
    "T5_evidence",
    "T5_evidence_IO",
    "T5_multi_task",
    "violet_decoder",
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
    train_data_path: str = "example_data/wildQA-data/train.json"
    dev_data_path: str = "example_data/wildQA-data/dev.json"
    test_data_path: str = "example_data/wildQA-data/test.json"
    num_workers: int = multiprocessing.cpu_count() // max(torch.cuda.device_count(), 1)
    output_ckpt_dir: Optional[str] = None
    model_name_or_path: str = "t5-base"
    max_seq_length: int = field(
        default=512,
        metadata={"help": "maximum of the text sequence. Truncate if exceeded."}
    )
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    adam_epsilon: float = 1e-8
    warmup_steps: int = 0
    train_batch_size: Optional[int] = 8
    eval_batch_size: Optional[int] = 8
    num_train_epochs: int = 100
    gradient_accumulation_steps: int = 16
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
    profiler: Optional[Literal["simple", "advanced", "pytorch"]] = None
    use_tpu: bool = False
    test_after_train: bool = False
    wandb_project: str = "In-the-wild-QA"
    wandb_name: str = field(
        default=None,
        metadata={"help": "name of this run."}
    )
    wandb_entity: str = field(
        default=None,
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
        metadata={"help": "model types for calculating embedding, more models available at\
                        https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0"}
    )

    def __post_init__(self):
        if self.embedding_model not in EMBEDDING_CHOICES:
            raise ValueError(f"Please select from {', '.join(EMBEDDING_CHOICES)}")


@dataclass
class T5ZeroShotArguments:
    max_seq_length: int = field(
        default=512,
        metadata={"help": "maximum length of the text length. Truncate the exceeded part."}
    )
    batch_size: int = 32
    pred_out_dir: Optional[str] = None
    pred_num: int = field(
        default=None,
        metadata={"help": "number of predictions made."}
    )
    beam_size: Optional[int] = None
    model_name_or_path: str = "t5-base"


@dataclass
class _VisualBaseArguments:
    visual_avg_pool_size: int = field(metadata={"help": "sampling rate for visual features."})
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
class T5TextVisualTrainArguments(TrainAndTestArguments, _VisualBaseArguments):
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
# Argument classes for Pre-trained model, VIOLET encoder + T5 decoder and CLIP encoder + T5 decoder


@dataclass
class VIOLETDecoderTrainArguments(T5TextVisualTrainArguments):
    pretrained_violet_ckpt_path: str = field(
        default="ckpts/pre-trained_violet/ckpt_violet_pretrain.pt",
        metadata={"help": "path to the pre-trained VIOLET model checkpoint"}
    )
    path_to_frames: str = field(
        default="video_features/frames",
        metadata={"help": "path to the directory that contains all the extracted frames"}
    )
    size_img: int = field(
        default=224,
        metadata={"help": "image size to convert. We use the default 224 consistent with the original Violet paper."}
    )


@dataclass
class CLIPDecoderBasics:
    pretrained_clip_ckpt_path: str = field(
        default="openai/clip-vit-base-patch32",
        metadata={"help": "ckpt name to the pre-trained CLIP model"}
    )
    path_to_frames: str = field(
        default="video_features/frames",
        metadata={"help": "path to the directory that contains all the extracted frames"}
    )

@dataclass
class CLIPDecoderTrainArguments(CLIPDecoderBasics, T5TextVisualTrainArguments):
    pass


def model_type_to_dataclass_types(model_type: str) -> Union[DataClassType, Iterable[DataClassType]]:
    return {
        "random": [TrainAndTestArguments, RandomArguments],
        "most_common_ans": TrainAndTestArguments,
        "closest_rtr":  [TrainAndTestArguments, ClosestRetrievalArguments],
        "T5_train": TrainAndTestArguments,
        "T5_zero_shot": [TrainAndTestArguments, T5ZeroShotArguments],
        "T5_text_and_visual": [TrainAndTestArguments, T5TextVisualTrainArguments],
        "T5_evidence": [TrainAndTestArguments, T5EvidenceFindingTrainArguments],
        "T5_evidence_IO": [TrainAndTestArguments, T5EvidenceIOTrainArguments],
        "T5_multi_task": [TrainAndTestArguments, T5MultiTaskTrainArguments],
        "violet_decoder": [TrainAndTestArguments, VIOLETDecoderTrainArguments],
        "clip_decoder": [TrainAndTestArguments, CLIPDecoderTrainArguments],
    }[model_type]
