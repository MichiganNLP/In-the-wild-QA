import os
import textwrap
import argparse
import torch
import torch.cuda
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from src.T5.T5_dataloader import T5Dataset
from src.T5.model import T5FineTuner




def T5_eval(args):
    # load model from the ckpt path
    if args.model_type != "T5_zero_shot":
        model = T5FineTuner.load_from_checkpoint(args.ckpt_path)

        if args.model_type == "T5_text_visual_eval":
            dataset = T5Dataset(data_dir=args.test_data, is_test=True, tokenizer=model.tokenizer, include_visual=True, max_len=args.max_seq_length, \
                max_vid_len=args.max_vid_length, path_to_visual_file=args.path_to_visual_file, visual_size=args.visual_size)
        else:
            dataset = T5Dataset(tokenizer=model.tokenizer, data_dir=args.test_data, is_test=True)
    else:
        model = T5FineTuner(args)
        dataset = T5Dataset(tokenizer=model.tokenizer, data_dir=args.test_data, is_test=True, is_zero_shot=True)

    model.model.eval()
    # NOTE: assume we have GPU resources in testing
    model.model.cuda()


    loader = DataLoader(dataset, batch_size=args.batch_size)

    outputs = []
    inputs = []


    for batch in tqdm(loader):

        if args.model_type == "T5_text_visual_eval":
            outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                                    attention_mask=batch['source_mask'].cuda(), 
                                    visual=batch["visual_ids"].cuda(),
                                    visual_attention_mask=batch['visual_mask'].cuda(),
                                    max_length=args.max_seq_length,
                                    num_beams=args.beam_size,
                                    num_return_sequences=args.pred_num)
        else:
            outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                                    attention_mask=batch['source_mask'].cuda(), 
                                    max_length=args.max_seq_length,
                                    num_beams=args.beam_size,
                                    num_return_sequences=args.pred_num)

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

    with open(f'{args.pred_out_dir}/preds-{args.model_type}-{args.pred_num}.txt', 'w') as f:
        f.write('\n'.join(outputs))
