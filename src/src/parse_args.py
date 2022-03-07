import multiprocessing
from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class DataPathArguments:
    train_data: str = field(
        default="example_data/wildQA-data/train.json",
        metadata={"help": "train data path"}
    )
    dev_data: str = field(
        default="example_data/wildQA-data/dev.json",
        metadata={"help": "dev data path"}
    )
    test_data: str = field(
        default="example_data/wildQA-data/test.json",
        metadata={"help": "test data path"}
    )
    num_workers: int = multiprocessing.cpu_count() // max(torch.cuda.device_count(), 1)


MODEL_CHOICES = [
    "random_text",
    "most_common_ans",
    "closest_rtr",
    "T5_train",
    "T5_eval",
    "T5_zero_shot",
    "T5_text_and_visual",
    "T5_text_visual_eval",
    "random_evidence",
    "T5_evidence",
    "T5_evidence_eval",
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
    output_ckpt_dir: str = field(
        default=None,
        metadata={"help": "path to the output checkpoints"}
    )
    model_name_or_path: str = field(
        default="t5-base",
        metadata={"help": "types of T5"}
    )
    max_seq_length: int = field(
        default=512,
        metadata={"help": "maximum of the text sequence. Truncate if exceeded part."}
    )
    learning_rate: float = field(
        default=3e-4,
        metadata={"help": "learning rate for the training."}
    )
    weight_decay: float = field(
        default=0.0,
        metadata={"help": "weight decay for the training phase."}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "adam epsilon for the training phase."}
    )
    warmup_steps: int = field(
        default=0,
        metadata={"help": "steps for warming up in the training."}
    )
    train_batch_size: int = field(
        default=8,
        metadata={"help": "batch size for training."}
    )
    eval_batch_size: int = field(
        default=8,
        metadata={"help": "batch size for evaluation."}
    )
    num_train_epochs: int = field(
        default=100,
        metadata={"help": "maximum number of epochs for training."}
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "number of steps taken before gradient updates."}
    )
    n_gpu: int = field(
        default=1,
        metadata={"help": "number of gpus to use."}
    )
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
    max_grad_norm: float = field(
        default=1.0,
        metadata={"help": "if you enable 16-bit training then set this to a sensible value, 0.5 is a good default."}
    )
    seed: int = field(
        default=42,
        metadata={"help": "random seed for T5 training phase."}
    )
    use_tpu: bool = field(
        default=False,
        metadata={"help": "whether to use TPU. We do not support TPU."}
    )


@dataclass
class _T5EvalBaseArguments:
    max_seq_length: int = field(
        default=512,
        metadata={"help": "maximum length of the text length. Truncate the exceeded part."}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "batch size for testing."}
    )
    pred_out_dir: str = field(
        default=None,
        metadata={"help": "prediction output directory."}
    )
    pred_num: int = field(
        default=None,
        metadata={"help": "number of predictions made."}
    )
    beam_size: int = field(
        default=None,
        metadata={"help": "beam size for search."}
    )


@dataclass
class T5EvalArguments(_T5EvalBaseArguments):
    ckpt_path: str = field(
        default=None,
        metadata={"help": "path to checkpoint to load."}
    )


@dataclass
class T5ZeroShotArguments(_T5EvalBaseArguments):
    model_name_or_path: str = field(
        default="t5-base",
        metadata={"help": "T5 model type for zero shot."}
    )


@dataclass
class _VisualBaseArguments:
    sample_rate: int = field(
        metadata={"help": "sampling rate for visual features."}
    )
    path_to_visual_file: str = field(
        default="https://www.dropbox.com/s/vt2kjdqr7mnxg2q/WildQA_I3D_avg_pool.hdf5?dl=1",
        metadata={"help": "path to visual input files"}
    )
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
# Argument classes for others.


@dataclass
class WandbArguments:
    wandb_project: str = field(
        default="In-the-wild-QA",
        metadata={"help": "wandb project name."}
    )
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
