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

    with torch.inference_mode():
        for batch in tqdm(data_loader):
            batch = {k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in batch.items()}

            if args.model_type == "T5_text_visual_eval":
                outs = model.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                            visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)
            elif args.model_type == "T5_evidence_eval":
                outs = model.model.predict(masked_caption_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                           visual=batch["visual_ids"], visual_attention_mask=batch["visual_mask"])
            else:
                outs = model.model.generate(input_ids=batch["source_ids"], attention_mask=batch["source_mask"],
                                            max_length=args.max_seq_length, num_beams=args.beam_size,
                                            num_return_sequences=args.pred_num)

            if args.model_type == "T5_evidence_eval":
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
            else:  # QA part answer generation
                outputs.extend(model.tokenizer.decode(ids) for ids in outs)

    os.makedirs(args.pred_out_dir, exist_ok=True)

    with open(f"{args.pred_out_dir}/preds-{args.model_type}-{args.pred_num}.txt", "w") as file:
        file.write("\n".join((json.dumps(e) for e in outputs) if args.model_type == "T5_evidence_eval" else outputs))
