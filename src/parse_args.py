import argparse

def parse_args():
    parser = argparse.ArgumentParser(prog='model_type')
    subparsers = parser.add_subparsers(help='sub-command help', dest='model_type')

    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--train_data', help='train data path', required=True)
    parent_parser.add_argument('--dev_data', help='dev data path', required=True)
    parent_parser.add_argument('--test_data', help='test data path', required=True)
    parent_parser.add_argument('--random_state', type=int, default=42, help='random state number')

    # random text baseline model arguments
    parser_rdn_text = subparsers.add_parser('random_text', help='random text baseline', parents=[parent_parser])

    # random baseline model arguments
    parser_rtr = subparsers.add_parser('closest_rtr', help='closest retrieve text baseline', parents=[parent_parser])
    parser_rtr.add_argument('--embedding_model', default="stsb-roberta-base", choices=['stsb-roberta-base', 
                        'stsb-bert-large', 'stsb-distilbert-base', 'stsb-roberta-large'],
                        help='model types for calculating embedding, more models available at\
                        https://docs.google.com/spreadsheets/d/14QplCdTCDwEmTqrn1LH4yrbKvdogK4oQvYO1K1aPR5M/edit#gid=0')
    

    # T5 model arguments
    parser_T5 = subparsers.add_parser('T5_train', help='T5 text baseline', parents=[parent_parser])
    parser_T5.add_argument("--output_ckpt_dir", required=True)
    parser_T5.add_argument("--model_name_or_path", default='t5-base')
    parser_T5.add_argument("--tokenizer_name_or_path", default='t5-base')
    parser_T5.add_argument("--max_seq_length", default=512)
    parser_T5.add_argument("--learning_rate", default=3e-4)
    parser_T5.add_argument("--weight_decay", default=0.0)
    parser_T5.add_argument("--adam_epsilon", default=1e-8)
    parser_T5.add_argument("--warmup_steps", default=0)
    parser_T5.add_argument("--train_batch_size", default=8, type=int)
    parser_T5.add_argument("--eval_batch_size", default=8, type=int)
    parser_T5.add_argument("--num_train_epochs", default=100, type=int)
    parser_T5.add_argument("--gradient_accumulation_steps", default=16, type=int)
    parser_T5.add_argument("--n_gpu", default=1)
    parser_T5.add_argument("--early_stop_callback", default=False)

    # if you want to enable 16-bit training then install apex and set this to true
    parser_T5.add_argument("--fp_16", default=False)

     # you can find out more on optimisation levels here https://nvidia.github.io/apex/amp.html#opt-levels-and-properties
    parser_T5.add_argument("--opt_level", default='01')

    # if you enable 16-bit training then set this to a sensible value, 0.5 is a good default
    parser_T5.add_argument("--max_grad_norm", default=1.0)
    parser_T5.add_argument("--seed", default=42)
    parser_T5.add_argument("--use_tpu", default=False)

    parser_T5.add_argument("--wandb_project", default='In-the-wild-VQA')
    parser_T5.add_argument("--wandb_name", required=True)

    # T5 model evaluate arguments
    parser_T5_eval = subparsers.add_parser('T5_eval', help='T5 text baseline', parents=[parent_parser])
    parser_T5_eval.add_argument("--ckpt_path")
    parser_T5_eval.add_argument("--max_seq_length", default=512, type=int)
    parser_T5_eval.add_argument("--batch_size", default=32, type=int)
    parser_T5_eval.add_argument("--pred_out_dir", help="prediction output directory")
    parser_T5_eval.add_argument("--pred_num", type=int, help="number of predictions made")
    parser_T5_eval.add_argument("--beam_size", type=int, help="beam size for search")

    args = parser.parse_args()
    return args