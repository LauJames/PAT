import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import json
import torch
import logging
import time
from transformers import BertTokenizerFast, BertConfig
from bert_ranker.models import pairwise_bert
from bert_ranker.dataloader.dataset import MSMARCO_PR_Pair_Dataset
from bert_ranker.network_trainer import Trainer

import argparse
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser('Pytorch')
    # Input and output configs
    parser.add_argument("--output_dir", default=curdir + '/results', type=str,
                        help="the folder to output predictions")
    parser.add_argument("--mode", default='train', type=str,
                        help="train/eval_full_dev1000/eval_subsmall_dev/train_pseudo/dl2019")

    # Training procedure
    parser.add_argument("--from_triples", default=True, type=bool,
                        help="Using the official triples train set or not")
    parser.add_argument("--seed", default=42, type=str,
                        help="random seed")
    parser.add_argument("--num_epochs", default=2, type=int,
                        help="Number of epochs for training.")
    parser.add_argument("--num_training_instances", default=-1, type=int,
                        help="Number of training instances for training (if num_training_instances != -1 then num_epochs is ignored).")
    parser.add_argument("--validate_every_epochs", default=1, type=int,
                        help="Run validation every <validate_every_epochs> epochs.")
    parser.add_argument("--validate_every_steps", default=750, type=int,
                        help="Run validation every <validate_every_steps> steps.")
    parser.add_argument("--num_validation_batches", default=-1, type=int,
                        help="Run validation for a sample of <num_validation_batches>. To run on all instances use -1.")
    parser.add_argument("--train_batch_size", default=256, type=int,
                        help="Training batch size.")
    parser.add_argument("--val_batch_size", default=2048, type=int,
                        help="Validation and test batch size.")
    parser.add_argument("--sample_data", default=2560000, type=int,
                        help="Amount of data to sample for training and eval. If no sampling required use -1.")
    parser.add_argument("--use_dev_triple", default=False, type=bool,
                        help="whether use dev triples to select the best model for pseudo label and extract.")
    parser.add_argument("--pseudo_final", default=False, type=bool,
                        help="whether use the best saved model to pseudo datasets")

    # Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--max_seq_len", default=256, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--lr", default=7e-6, type=float, required=False,
                        help="Learning rate.")
    parser.add_argument("--max_grad_norm", default=0.1, type=float, required=False,
                        help="Max gradient normalization.")
    parser.add_argument("--accumulation_steps", default=1, type=float, required=False,
                        help="gradient accumulation.")
    parser.add_argument("--warmup_portion", default=0.1, type=float, required=False,
                        help="warmup portion.")
    parser.add_argument("--loss_function", default="label-smoothing-cross-entropy", type=str, required=False,
                        help="Loss function (default is 'cross-entropy').")
    parser.add_argument("--smoothing", default=0.1, type=float, required=False,
                        help="Smoothing hyperparameter used only if loss_function is label-smoothing-cross-entropy.")

    args = parser.parse_args()
    args.model_name = 'pairwise-BERT-ranker'

    logger = logging.getLogger("Pytorch")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    # log directory
    log_dir = curdir + '/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split()[:3])
    if args.from_triples:
        log_path = log_dir + '/pairwise.mspr_triples.' + now_time + '.' + args.mode + '.log'
        args.run_id = args.transformer_model + '.pairwise.triples.' + now_time
        tensorboard_dir = curdir + '/tensorboard_dir/pairwise_triples_' + args.transformer_model + '/'
    else:
        log_path = log_dir + '/pairwise.mspr.' + now_time + '.' + args.mode + '.log'
        args.run_id = args.transformer_model + '.pairwise.' + now_time
        tensorboard_dir = curdir + '/tensorboard_dir/pairwise_' + args.transformer_model + '/'

    if os.path.exists(log_path):
        os.remove(log_path)

    if args.mode == 'train':
        writer_train = SummaryWriter(tensorboard_dir + 'train')
    elif args.mode == 'train_pseudo':
        writer_train = SummaryWriter(tensorboard_dir + 'train_pseudo')
    elif args.mode == 'train_pseudo_same':
        writer_train = SummaryWriter(tensorboard_dir + 'train_pseudo_same')
    elif args.mode == 'train_nq':
        writer_train = SummaryWriter(tensorboard_dir + 'train_nq')
    else:
        writer_train = None

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Runing with configurations: {}'.format(json.dumps(args.__dict__, indent=4)))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)

    # Instantiate transformer model to be used
    model = pairwise_bert.BertForPairwiseLearning.from_pretrained(args.transformer_model,
                                                                  loss_function=args.loss_function,
                                                                  smoothing=args.smoothing)

    data_obj = MSMARCO_PR_Pair_Dataset(tokenizer=tokenizer)

    if args.mode in ['train_pseudo', 'eval_pseudo_full_dev1000', 'eval_pseudo_subsmall']:
        model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.pseudo.' + args.transformer_model + '.pth'
    elif args.mode in ['train_pseudo_same', 'eval_pseudo_same_full_dev1000', 'dl2019_same_pseudo']:
        model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.pseudo.same.' + args.transformer_model + '.pth'
    elif args.mode in ['train_nq']:
        model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.nq.' + args.transformer_model + '.pth'
    else:
        model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'

    trainer = Trainer(
        model=model,
        data_class=data_obj,
        tokenizer=tokenizer,
        model_path=model_path,
        validation_metric=['ndcg_cut_10', 'map', 'recip_rank'],
        monitor_metric='ndcg_cut_10',
        args=args,
        writer_train=writer_train,
        run_id=args.run_id,
        logger=logger)

    if args.mode in ['train', 'train_nq', 'train_pseudo', 'train_pseudo_same']:
        trainer.train_ranker(mode=args.mode)
        writer_train.close()
    else:
        raise ValueError("Error mode !!!")
    print("Done!")


if __name__ == "__main__":
    main()