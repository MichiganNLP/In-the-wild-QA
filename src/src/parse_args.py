from __future__ import annotations

import multiprocessing
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

import torch

# Can't use `from __future__ import annotations` here. See https://github.com/huggingface/transformers/pull/15795
# From the next version of transformers (after v4.17.0) it should be possible.
from transformers.hf_argparser import DataClassType


@dataclass
class DataPathArguments:
    train_data_path: str = "example_data/wildQA-data/train.json"
    dev_data_path: str = "example_data/wildQA-data/dev.json"
    test_data_path: str = "example_data/wildQA-data/test.json"
    num_workers: int = multiprocessing.cpu_count() // max(torch.cuda.device_count(), 1)


MODEL_CHOICES = [
    "random_text",
    "most_common_ans",
    "closest_rtr",
    "T5_train",
    "T5_zero_shot",
    "T5_text_and_visual",
    "random_evidence",
    "T5_evidence",
    "T5_evidence_IO",
    "T5_multi_task",
    "violet_decoder",
    "clip_decoder",
]

#########################################################################################
#########################################################################################
# Argument classes for Video QA part.

EMBEDDING_CHOICES = [
    "stsb-roberta-base",
    "stsb-bert-large",
    "stsb-distilbert-base",
    "stsb-roberta-large",
]


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
class T5TrainArguments:
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


@dataclass
class _T5EvalBaseArguments:
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


@dataclass
class T5EvalArguments(_T5EvalBaseArguments):
    ckpt_path: Optional[str] = None


@dataclass
class T5ZeroShotArguments(_T5EvalBaseArguments):
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
class T5TextVisualTrainArguments(T5TrainArguments, _VisualBaseArguments):
    pass


@dataclass
class T5TextVisualEvalArguments(T5EvalArguments, _VisualBaseArguments):
    pass


#########################################################################################
#########################################################################################
# Argument classes for Video Evidence Finding part.


@dataclass
class RandomEvidenceArguments:
    pred_num: int = field(
        default=5,
        metadata={"help": "number of predicted evidence"}
    )


@dataclass
class T5EvidenceFindingTrainArguments(T5TextVisualTrainArguments):
    pass


@dataclass
class T5EvidenceFindingEvalArguments(T5TextVisualEvalArguments):
    pass


@dataclass
class T5EvidenceIOTrainArguments(T5TextVisualTrainArguments):
    pass


@dataclass
class T5EvidenceIOEvalArguments(T5TextVisualEvalArguments):
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


@dataclass
class T5MultiTaskEvalArguments(T5TextVisualEvalArguments):
    pass


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
class VIOLETDecoderEvalArguments(T5TextVisualEvalArguments):
    pass


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


@dataclass
class CLIPDecoderEvalArguments(CLIPDecoderBasics, T5TextVisualEvalArguments):
    pass


#########################################################################################
#########################################################################################
# Argument classes for others.


@dataclass
class WandbArguments:
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


def model_type_to_dataclass_types(model_type: str) -> DataClassType | Iterable[DataClassType]:
    return {
        "random_text": DataPathArguments,
        "most_common_ans": DataPathArguments,
        "closest_rtr":  [DataPathArguments, ClosestRetrievalArguments],
        "random_evidence": [DataPathArguments, RandomEvidenceArguments],
        "T5_train": [DataPathArguments, T5TrainArguments, WandbArguments],
        "T5_zero_shot": [DataPathArguments, T5ZeroShotArguments],
        "T5_text_and_visual": [DataPathArguments, T5TextVisualTrainArguments, WandbArguments],
        "T5_evidence": [DataPathArguments, T5EvidenceFindingTrainArguments, WandbArguments],
        "T5_evidence_IO": [DataPathArguments, T5EvidenceIOTrainArguments, WandbArguments],
        "T5_multi_task": [DataPathArguments, T5MultiTaskTrainArguments, WandbArguments],
        "violet_decoder": [DataPathArguments, VIOLETDecoderTrainArguments, WandbArguments],
        "clip_decoder": [DataPathArguments, CLIPDecoderTrainArguments, WandbArguments],
    }[model_type]
