import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import json
import torch
import logging
import time
from transformers import BertTokenizerFast
from bert_ranker.models import pairwise_bert
from bert_ranker.dataloader.dataset_imitate import Imitation_Dataset
from bert_ranker.network_trainer import Trainer

import argparse
from torch.utils.tensorboard import SummaryWriter


def main():
    parser = argparse.ArgumentParser('Pytorch')
    # Input and output configs

    parser.add_argument("--output_dir", default=curdir + '/results', type=str,
                        help="the folder to output predictions")
    parser.add_argument("--save_model", default=curdir + '/saved_models', type=bool, required=False,
                        help="Save trained model at the end of training.")
    parser.add_argument("--mode", default='train', type=str,
                        help="train/eval_full_dev1000/eval_subsmall_dev")

    # Training procedure
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
    parser.add_argument("--train_batch_size", default=128, type=int,
                        help="Training batch size.")
    parser.add_argument("--val_batch_size", default=1024, type=int,
                        help="Validation and test batch size.")
    parser.add_argument("--sample_data", default=-1, type=int,
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
    parser.add_argument("--max_grad_norm", default=1, type=float, required=False,
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

    # sample_config_tail = 'top_10_last_35'
    # sample_config_tail = 'top_15_last_19'
    # sample_config_tail = 'top_16_last_17'
    sample_config_tail = 'top_20_last_10'
    # sample_config_tail = 'top_25_last_4'
    # sample_config_tail = 'top_15_last_59'
    # sample_config_tail = 'top_16_last_55'
    # sample_config_tail = 'top_20_last_40'
    # sample_config_tail = 'top_25_last_28'

    # log directory
    log_dir = curdir + '/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split()[:3])

    log_path = log_dir + '/imitation.pairwise.mspr.' + sample_config_tail + '.' + now_time + '.' + args.mode + '.log'
    args.run_id = 'imitation.' + args.transformer_model + '.pairwise.' + now_time
    tensorboard_dir = curdir + '/tensorboard_dir/imitation_pairwise_triples_' + args.transformer_model + '/'

    if os.path.exists(log_path):
        os.remove(log_path)

    if args.mode == 'train':
        writer_train = SummaryWriter(tensorboard_dir + 'train')
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

    data_obj = Imitation_Dataset(tokenizer=tokenizer, sample_config_tail=sample_config_tail)

    # load fine-tuned pairwise ranker
    start_model_path = curdir + '/saved_models/' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'

    # model_path = curdir + '/saved_models/Imitation.bert_large.straight.' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'
    # model_path = curdir + '/saved_models/Imitation.bert_large.further.' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'
    # model_path = curdir + '/saved_models/Imitation.MiniLM.straight.' + model.__class__.__name__ + '.' + sample_config_tail + '.' + args.transformer_model + '.pth'
    model_path = curdir + '/saved_models/Imitation.MiniLM.further.' + model.__class__.__name__ + '.' + sample_config_tail + '.' + args.transformer_model + '.pth'

    trainer = Trainer(
        model=model,
        data_class=data_obj,
        tokenizer=tokenizer,
        model_path=model_path,
        start_model_path=start_model_path,
        validation_metric=['ndcg_cut_10', 'map', 'recip_rank'],
        monitor_metric='ndcg_cut_10',
        args=args,
        writer_train=writer_train,
        run_id=args.run_id,
        logger=logger)

    if args.mode in ['train']:
        trainer.train_ranker(mode=args.mode)
        writer_train.close()
    elif args.mode in ['eval_subsmall_dev', 'eval_full_dev1000']:
        with torch.no_grad():
            model = trainer.load_model()
            print("load {} ".format(model_path))
            trainer.dev_pairwise(mode=args.mode, model=model)
    else:
        raise ValueError("Error mode !!!")
    print("Done!")


if __name__ == "__main__":
    main()