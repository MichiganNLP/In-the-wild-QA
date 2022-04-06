import argparse
import json
import os
from collections.abc import Mapping
from typing import Iterator

import torch
import torch.cuda
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.metrics import get_best_evidence_spans
from src.transformer_models.model import AnswerWithEvidenceModule, TYPE_BATCH
from src.video_qa_with_evidence_dataset import VideoQAWithEvidenceDataModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_best_evidence_span(outs: tuple[torch.Tensor, torch.Tensor],
                           batch: TYPE_BATCH) -> Iterator[Mapping[str, torch.Tensor]]:
    start, end = get_best_evidence_spans(start_scores=outs[0], end_scores=outs[1], mask=batch["visual_mask"])
    for start_instance, end_instance, start_scores_instance, end_scores_instance in zip(start, end, *outs):
        yield {"start": start_instance[0], "end": end_instance[0],
               "score": start_scores_instance[start_instance[0]] + end_scores_instance[end_instance[0]]}


def transformer_eval(args: argparse.Namespace) -> None:
    model_class = AnswerWithEvidenceModule

    if args.model_type == "T5_zero_shot":
        os.environ["TOKENIZERS_PARALLELISM"] = "0"
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = model_class(tokenizer=tokenizer, **args.__dict__)
    else:
        model = model_class.load_from_checkpoint(args.ckpt_path)
        tokenizer = model.tokenizer

    data_module = VideoQAWithEvidenceDataModule(args, tokenizer=tokenizer)
    data_loader = data_module.test_dataloader()

    model.eval()
    model.to(DEVICE)

    outputs = []
    seq2seq_outputs = []
    evidence_outputs = []

    with torch.inference_mode():
        for batch in tqdm(data_loader):
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            if args.model_type == "T5_text_visual_eval":
                outs = model.model.generate(input_ids=batch["question_ids"], attention_mask=batch["question_mask"],
                                            visual=batch["visual"], visual_attention_mask=batch["visual_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)
            elif args.model_type in {"T5_evidence_eval", "T5_evidence_IO_eval"}:
                outs = model.model.predict(masked_caption_ids=batch["question_ids"],
                                           attention_mask=batch["question_mask"], visual=batch["visual"],
                                           visual_attention_mask=batch["visual_mask"])
            elif args.model_type == "T5_multi_task_eval":
                evidence_outs = model.model.predict(masked_caption_ids=batch["question_ids"],
                                                    attention_mask=batch["question_mask"], visual=batch["visual"],
                                                    visual_attention_mask=batch["visual_mask"])
                seq2seq_outs = model.model.generate(input_ids=batch["question_ids"],
                                                    attention_mask=batch["question_mask"], visual=batch["visual"],
                                                    visual_attention_mask=batch["visual_mask"],
                                                    max_length=args.max_seq_length, num_beams=args.beam_size,
                                                    num_return_sequences=args.pred_num)
            elif args.model_type == "clip_decoder_eval":
                outs = model.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                            visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)
            else:
                outs = model.model.generate(input_ids=batch["question_ids"], attention_mask=batch["question_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)

            if args.model_type == "T5_evidence_eval":
                outputs.extend(get_best_evidence_span(outs, batch))
            elif args.model_type == "T5_evidence_IO_eval":
                for start_end_scores in outs.values():
                    # TODO: duplicate / None if just 1 prediction but pred_num is larger than 1
                    outputs.extend({"score": score, "start": start, "end": end}
                                   for start, end, score in start_end_scores[:args.pred_num])
            elif args.model_type == "T5_multi_task_eval":
                seq2seq_outputs.extend(model.tokenizer.decode(ids) for ids in seq2seq_outs)
                evidence_outputs.extend(get_best_evidence_span(evidence_outs, batch))
            elif args.model_type == "clip_decoder_eval":
                outputs.extend(model.tokenizer["decoder"].decode(ids) for ids in outs)
            else:  # QA part answer generation
                outputs.extend(model.tokenizer.decode(ids) for ids in outs)

    os.makedirs(args.pred_out_dir, exist_ok=True)

    if args.model_type == "T5_multi_task_eval":
        with open(f"{args.pred_out_dir}/preds-{args.model_type}-vqa-{args.pred_num}.txt", "w") as file:
            file.write("\n".join(seq2seq_outputs))
        with open(f"{args.pred_out_dir}/preds-{args.model_type}-evidence-{args.pred_num}.txt", "w") as file:
            file.write("\n".join(json.dumps(e) for e in evidence_outputs))
    else:  # single task evaluation for each sub-task
        with open(f"{args.pred_out_dir}/preds-{args.model_type}-{args.pred_num}.txt", "w") as file:
            file.write("\n".join((json.dumps(e) for e in outputs) if args.model_type.startswith("T5_evidence")
                                 else outputs))
