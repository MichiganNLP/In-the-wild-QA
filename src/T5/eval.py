import os
import textwrap
import argparse
import torch
import torch.cuda
from tqdm.auto import tqdm
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader

from T5.T5_dataloader import T5Dataset
from T5.model import T5FineTuner




def T5_eval(args):
    # load model from the ckpt path
    model = T5FineTuner.load_from_checkpoint(args.ckpt_path)
    model.model.eval()
    # NOTE: assume we have GPU resources in testing
    model.model.cuda()

    dataset = T5Dataset(tokenizer=model.tokenizer, data_dir=args.test_data, is_test=True)
    loader = DataLoader(dataset, batch_size=args.batch_size)

    outputs = []
    inputs = []

    for batch in tqdm(loader):
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

    with open(f'{args.pred_out_dir}/preds-{args.pred_num}.txt', 'w') as f:
        f.write('\n'.join(outputs))
