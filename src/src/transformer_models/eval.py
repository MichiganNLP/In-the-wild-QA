import argparse
import json
import os

import torch
import torch.cuda
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from src.transformer_models.model import FineTuner
from src.transformer_models.video_qa_with_evidence_dataset import VideoQAWithEvidenceForT5DataModule


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _process_evidence_out(outs, batch, outputs):
    batch_size = outs[0].shape[0]
    predicted_span = {"score": -float("inf"), "start": -1, "end": -1}
    vid_len = torch.count_nonzero(batch["visual_mask"], dim=1)

    for b in range(batch_size):
        for i in range(vid_len[b].item()):
            start_score = outs[0][b, i]
            start = i
            for j in range(i + 1, vid_len[b].item()):
                end_score = outs[1][b, j]
                end = j
                if start_score + end_score > predicted_span["score"]:
                    # if start_score > predicted_span["score"]:
                    predicted_span["score"] = (start_score + end_score).item()
                    # predicted_span["score"] = start_score.item()
                    predicted_span["start"] = start
                    predicted_span["end"] = end
    outputs.append(predicted_span)
    return outputs


def transformer_eval(args: argparse.Namespace) -> None:
    os.environ["TOKENIZERS_PARALLELISM"] = "0"
    model_class = FineTuner

    if args.model_type == "T5_zero_shot":
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model = model_class(tokenizer=tokenizer, **args.__dict__)
    else:
        model = model_class.load_from_checkpoint(args.ckpt_path)
        tokenizer = model.tokenizer

    data_loader = VideoQAWithEvidenceForT5DataModule(args, tokenizer=tokenizer).test_dataloader()

    model.model.eval()
    model.model.to(DEVICE)

    outputs = []
    seq2seq_outputs, evidence_outputs = [], []

    with torch.inference_mode():
        for batch in tqdm(data_loader):
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            if args.model_type == "T5_text_visual_eval":
                outs = model.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                            visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)
            elif args.model_type in {"T5_evidence_eval", "T5_evidence_IO_eval"}:
                outs = model.model.predict(masked_caption_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                           visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"])
            elif args.model_type == "T5_multi_task_eval":
                evidence_outs = model.model.predict(masked_caption_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                           visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"])
                seq2seq_outs = model.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                            visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)
            elif args.model_type == "clip_decoder_eval":
                outs = model.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                    visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                                    max_length=args.max_seq_length, num_beams=args.beam_size,
                                    num_return_sequences=args.pred_num)
            else:
                outs = model.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                            visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)

            if args.model_type == "T5_evidence_eval":
                outputs = _process_evidence_out(outs, batch, outputs)

            elif args.model_type == "T5_evidence_IO_eval":
                for b, start_end_scores in outs.items():
                    # TODO: duplicate / None if just 1 prediction but pred_num is larger than 1
                    predicted_start_ends = start_end_scores[:args.pred_num]
                    outputs.extend([{
                        "score": score,
                        "start": start,
                        "end": end
                    } for [start, end, score] in predicted_start_ends])
            elif args.model_type == "T5_multi_task_eval":

                seq2seq_outputs.extend(model.tokenizer.decode(ids) for ids in seq2seq_outs)
                evidence_outputs = _process_evidence_out(evidence_outs, batch, evidence_outputs)
            
            elif args.model_type == "clip_decoder_eval":
                outputs.extend(model.tokenizer["decoder_tokenizer"].decode(ids) for ids in outs)

            else:  # QA part answer generation
                outputs.extend(model.tokenizer.decode(ids) for ids in outs)

    os.makedirs(args.pred_out_dir, exist_ok=True)

    if not args.model_type == "T5_multi_task_eval":
        # single task evaluation for each sub-task
        with open(f"{args.pred_out_dir}/preds-{args.model_type}-{args.pred_num}.txt", "w") as file:
            file.write("\n".join((json.dumps(e) for e in outputs) if args.model_type in {"T5_evidence_eval", "T5_evidence_IO_eval"} else outputs))
    else:
        with open(f"{args.pred_out_dir}/preds-{args.model_type}-vqa-{args.pred_num}.txt", "w") as file:
            file.write("\n".join(seq2seq_outputs))
        with open(f"{args.pred_out_dir}/preds-{args.model_type}-evidence-{args.pred_num}.txt", "w") as file:
            file.write("\n".join(json.dumps(e) for e in evidence_outputs))
