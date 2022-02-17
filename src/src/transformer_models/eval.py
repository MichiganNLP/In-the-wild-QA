import argparse
import json
import os

import torch
import torch.cuda
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from src.transformer_models.t5_dataloader import T5Dataset
from src.transformer_models.model import FineTuner, my_collate


def transformer_eval(args: argparse.Namespace) -> None:
    # load model from the ckpt path
    if args.model_type == "T5_zero_shot":
        model = FineTuner(args)
        dataset = T5Dataset(tokenizer=model.tokenizer, data_dir=args.test_data, is_test=True, is_zero_shot=True)

    else:
        model = FineTuner.load_from_checkpoint(args.ckpt_path)

        if args.model_type == "T5_text_visual_eval":
            dataset = T5Dataset(data_dir=args.test_data, is_test=True, tokenizer=model.tokenizer, include_visual=True,
                                max_len=args.max_seq_length, max_vid_len=args.max_vid_length,
                                path_to_visual_file=args.path_to_visual_file, visual_size=args.visual_size,
                                sample_rate=args.sample_rate)
        elif args.model_type == "T5_evidence_eval":
            dataset = T5Dataset(data_dir=args.test_data, is_test=True, tokenizer=model.tokenizer, include_visual=True,
                                max_len=args.max_seq_length, max_vid_len=args.max_vid_length,
                                path_to_visual_file=args.path_to_visual_file, visual_size=args.visual_size,
                                sample_rate=args.sample_rate, is_evidence=True)
        else:
            dataset = T5Dataset(tokenizer=model.tokenizer, data_dir=args.test_data, is_test=True)

    model.model.eval()
    # NOTE: assume we have GPU resources in testing
    model.model.cuda()

    if args.model_type == "T5_evidence_eval":
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=my_collate)
    else:
        loader = DataLoader(dataset, batch_size=args.batch_size)

    outputs = []

    for batch in tqdm(loader):
        if args.model_type == "T5_text_visual_eval":
            outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                        attention_mask=batch['source_mask'].cuda(),
                                        visual=batch["visual_ids"].cuda(),
                                        visual_attention_mask=batch['visual_mask'].cuda(),
                                        max_length=args.max_seq_length,
                                        num_beams=args.beam_size,
                                        num_return_sequences=args.pred_num)
        elif args.model_type == "T5_evidence_eval":
            outs = model.model.predict(masked_caption_ids=batch['source_ids'].cuda(),
                                       attention_mask=batch['source_mask'].cuda(),
                                       visual=batch["visual_ids"].cuda(),
                                       visual_attention_mask=batch['visual_mask'].cuda())
        else:
            outs = model.model.generate(input_ids=batch['source_ids'].cuda(),
                                        attention_mask=batch['source_mask'].cuda(),
                                        max_length=args.max_seq_length,
                                        num_beams=args.beam_size,
                                        num_return_sequences=args.pred_num)

        if args.model_type == "T5_evidence_eval":
            batch_size, N, _ = outs[0].shape
            assert N == 1
            predicted_span = {
                "score": -float('inf'),
                "start": -1,
                "end": -1
            }
            vid_len = torch.count_nonzero(batch["visual_mask"], dim=1)

            for b in range(batch_size):
                for i in range(vid_len[b].item()):
                    start_score = outs[0][b, 0, i]
                    start = i
                    for j in range(i + 1, vid_len[b].item()):
                        end_score = outs[1][b, 0, j]
                        end = j
                        if start_score + end_score > predicted_span["score"]:
                            # if start_score > predicted_span["score"]:
                            predicted_span["score"] = (start_score + end_score).item()
                            # predicted_span["score"] = start_score.item()
                            predicted_span["start"] = start
                            predicted_span["end"] = end
            outputs.append(predicted_span)
        else:
            # QA part answer generation
            dec = [model.tokenizer.decode(ids) for ids in outs]
            outputs.extend(dec)

    # NOTE: 2 places: 
    # OOV + , places
    # def fix_oov(pred: str) -> str:
    #     fixed = pred.replace(" ⁇ ", "<")
    #     fixed = fixed.replace("<unk>", "<")
    #     return fixed

    if not os.path.exists(args.pred_out_dir):
        os.makedirs(args.pred_out_dir)

    if args.model_type == "T5_evidence_eval":
        with open(f'{args.pred_out_dir}/preds-{args.model_type}-{args.pred_num}.txt', 'w') as f:
            f.write('\n'.join(json.dumps(e) for e in outputs))
    else:
        with open(f'{args.pred_out_dir}/preds-{args.model_type}-{args.pred_num}.txt', 'w') as f:
            f.write('\n'.join(outputs))
