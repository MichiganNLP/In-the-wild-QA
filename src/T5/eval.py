import os
import textwrap
import argparse
import torch
import torch.cuda
from tqdm.auto import tqdm
from sklearn import metrics

from dataloader import DBDataset
from model import T5FineTuner
from torch.utils.data import Dataset, DataLoader


args_dict = dict(
    data_dir="dataset/academic",
    ckpt_path='ckpt/epoch=46-val_sent_acc=0.00.ckpt',
    batch_size=32,
    max_seq_length=512,
    pred_out_dir="preds/academic",
    pred_num=5,
)


args = argparse.Namespace(**args_dict)

# load model from the ckpt path
model = T5FineTuner.load_from_checkpoint(args.ckpt_path)
model.model.eval()
# NOTE: assume we have GPU resources in testing
model.model.cuda()

dataset = DBDataset(model.tokenizer, args.data_dir, 'test', test=True)
loader = DataLoader(dataset, batch_size=args.batch_size)

outputs = []
inputs = []

for batch in tqdm(loader):
    outs = model.model.generate(input_ids=batch['source_ids'].cuda(), 
                            attention_mask=batch['source_mask'].cuda(), 
                            max_length=args.max_seq_length,
                            num_beams=5,
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
