"""
Single Query Attack method target imitation ranker on MSMARCO Passage Ranking
Using the generate triggers to test its transferability
"""
import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import json
import torch
import logging
import time

import numpy as np
import argparse
import bisect
from tqdm import tqdm
from torch import cuda


from transformers import BertTokenizer, BertTokenizerFast, BertForNextSentencePrediction
from bert_ranker.models import pairwise_bert
from bert_ranker.models.bert_lm import BertForLM
from apex import amp
from attack_methods import gen_adversarial_trigger_pair_passage
from data_utils import prepare_data_and_scores

device = 'cuda:0' if cuda.is_available() else 'cpu'


def main():
    parser = argparse.ArgumentParser('IR_Attack')

    parser.add_argument('--verbose', default=True, type=bool,
                        help='Print every iteration')
    parser.add_argument('--mode', default='train', type=str,
                        help='train/transfer')

    # target known model config
    parser.add_argument("--experiment_name", default='pairwise.v1', type=str)
    parser.add_argument("--data_name", default="dl", type=str)
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--max_seq_len", default=256, type=int, required=False,
                        help="Maximum sequence length for the inputs.")
    parser.add_argument("--tri_len", default=8, type=int, help="Maximun trigger length for generation.")
    parser.add_argument("--topk", default=50, type=int, help="Top k sampling for beam search")
    parser.add_argument('--max_iter', type=int, default=20, help='maximum iteraiton')
    parser.add_argument("--beta", default=0.6, type=float, help="Coefficient for language model loss.")
    parser.add_argument("--gamma", default=100., type=float, help="Coefficient for NSP model loss.")
    parser.add_argument('--stemp', type=float, default=1.0, help='temperature of softmax')
    parser.add_argument('--repetition_penalty', type=float, default=1.0, help='penalty of repetition')
    parser.add_argument('--lr', type=float, default=0.0075, help='optimization step size')
    parser.add_argument("--num_beams", default=5, type=int, help="Number of beams")
    parser.add_argument("--num_filters", default=10, type=int, help="Number of num_filters words to be filtered")
    parser.add_argument('--perturb_iter', type=int, default=30, help='PPLM iteration')
    parser.add_argument('--patience_limit', type=int, default=2, help="Patience for early stopping.")
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument("--seed", default=666, type=str, help="random seed")
    parser.add_argument('--regularize', default=False, type=bool, help='Use regularize to decrease perplexity')
    parser.add_argument("--fp16", default=True, type=bool, help="Whether to use apex to accelerate.")

    args = parser.parse_args()

    logger = logging.getLogger("Pytorch")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    log_dir = curdir + '/logs'
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    now_time = '_'.join(time.asctime(time.localtime(time.time())).split()[:3])
    log_path = log_dir + '/attack.{}.mspr.spec.lm_{}.nsp_{}.{}.log'.format(args.experiment_name, args.beta, args.gamma,
                                                                           now_time)

    if os.path.exists(log_path):
        os.remove(log_path)

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.info('Runing with configurations: {}'.format(json.dumps(args.__dict__, indent=4)))

    tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
    if args.mode == 'train':
        train_trigger(args, logger, tokenizer)
    else:
        raise ValueError('Not implemented error!')


def train_trigger(args, logger, tokenizer):
    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(
        experiment_name=args.experiment_name,
        data_name=args.data_name,
        top_k=10,
        least_num=5)

    model = pairwise_bert.BertForPairwiseLearning.from_pretrained(args.transformer_model)
    model.to(device)
    if args.experiment_name == 'pairwise.v2':
        model_path = prodir + '/bert_ranker/saved_models/Imitation.MiniLM.L12.v2.' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'
    elif args.experiment_name == 'pairwise.v1':
        model_path = prodir + '/bert_ranker/saved_models/Imitation.bert_large.further_train.' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'
    elif args.experiment_name == 'pairwise.wo.imitation':
        model_path = prodir + '/bert_ranker/saved_models/' + model.__class__.__name__ + '.' + args.transformer_model + '.pth'
    logger.info("Load model from: {}".format(model_path))
    print("Load model from: {}".format(model_path))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    lm_model = BertForLM.from_pretrained('bert-base-uncased')
    lm_model.to(device)
    lm_model.eval()
    for param in lm_model.parameters():
        param.requires_grad = False

    nsp_model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')
    nsp_model.to(device)
    nsp_model.eval()
    for param in nsp_model.parameters():
        param.requires_grad = False

    model, lm_model, nsp_model = amp.initialize([model, lm_model, nsp_model])

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_10_cnt = 0
    cnt = 0
    boost_rank_list = []
    q_candi_trigger_dict = dict()

    used_qids = list(queries.keys())
    # random.shuffle(used_qids)

    for qid in tqdm(used_qids, desc="Processing"):
        torch.manual_seed(args.seed + cnt)
        torch.cuda.manual_seed_all(args.seed + cnt)
        cnt += 1

        tmp_trigger_dict = {}
        query = queries[qid]
        best = best_query_sent[qid]
        best_score = best[0]
        best_sent = ' '.join(best[1:5])

        old_scores = query_scores[qid][::-1]

        for did in target_q_passage[qid]:
            raw_passage = passages_dict[did]
            trigger, new_score, trigger_cands = gen_adversarial_trigger_pair_passage(
                query=query,
                best_sent=best_sent,
                raw_passage=raw_passage,
                model=model,
                tokenizer=tokenizer,
                device=device,
                logger=logger,
                args=args,
                lm_model=lm_model,
                nsp_model=nsp_model)

            msg = f'Query={query}\n' \
                  f'Best true sentences={best_sent}\n' \
                  f'Best similarity score={best_score}\n' \
                  f'Trigger={trigger}\n' \
                  f'Similarity core={new_score}\n'

            tmp_trigger_dict[did] = trigger

            print(msg)
            logger.info(msg)
            if args.verbose:
                logger.info('---Rank shifts for less relevant documents---')

                old_rank, old_score = target_q_passage[qid][did]
                new_rank = len(old_scores) - bisect.bisect_left(old_scores, new_score)

                total_docs_cnt += 1
                boost_rank_list.append(old_rank - new_rank)
                if old_rank > new_rank:
                    success_cnt += 1
                    if new_rank <= 500:
                        less_500_cnt += 1
                        if new_rank <= 100:
                            less_100_cnt += 1
                            if new_rank <= 50:
                                less_50_cnt += 1
                                if new_rank <= 10:
                                    less_10_cnt += 1

                print(f'Query id={qid}, Doc id={did}, '
                      f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
                logger.info(f'Query id={qid}, Doc id={did}, '
                            f'old score={old_score:.2f}, new score={new_score:.2f}, old rank={old_rank}, new rank={new_rank}')
            print('\n\n')
            logger.info('\n\n')

        q_candi_trigger_dict[qid] = tmp_trigger_dict

    boost_success_rate = success_cnt / (total_docs_cnt + 0.0) * 100
    less_500_rate = less_500_cnt / (total_docs_cnt + 0.0) * 100
    less_100_rate = less_100_cnt / (total_docs_cnt + 0.0) * 100
    less_50_rate = less_50_cnt / (total_docs_cnt + 0.0) * 100
    less_10_rate = less_10_cnt / (total_docs_cnt + 0.0) * 100
    avg_boost_rank = np.average(boost_rank_list)
    res_str = 'Boost Success Rate: {}\n' \
              'Average Boost Rank: {}\n' \
              'less than 500 Rate: {}\n' \
              'less than 100 Rate: {}\n' \
              'less than 50 Rate: {}\n' \
              'less than 10 Rate: {}\n'.format(boost_success_rate, avg_boost_rank, less_500_rate, less_100_rate,
                                               less_50_rate, less_10_rate)
    print(res_str)
    logger.info(res_str)

    with open(curdir + '/saved_results/cands_triggers_{}_{}_{}.json'.format(args.experiment_name, args.beta, args.gamma), 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(q_candi_trigger_dict, ensure_ascii=False))
        print('Trigger saved!')

    print('Test all triggers on imitation model...')

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_10_cnt = 0
    for qid in tqdm(used_qids, desc="Processing"):
        query = queries[qid]
        best = best_query_sent[qid]
        best_sent = best[1]
        old_scores = query_scores[qid][::-1]

        query_ids = tokenizer.encode(query, add_special_tokens=True)
        query_ids = torch.tensor(query_ids, device=device).unsqueeze(0)

        best_sent_ids = tokenizer.encode(best_sent, add_special_tokens=True)
        best_sent_ids = torch.tensor(best_sent_ids[1:], device=device).unsqueeze(0)

        placeholder_input_ids_neg = torch.cat([query_ids, best_sent_ids], dim=-1)
        placeholder_type_ids_neg = torch.cat([torch.zeros_like(query_ids), torch.ones_like(best_sent_ids)], dim=-1)
        placeholder_attention_mask_neg = torch.ones_like(placeholder_input_ids_neg)

        triggers = q_candi_trigger_dict[qid]
        for did in target_q_passage[qid]:
            tmp_best_new_score = -1e9
            old_rank, raw_score = target_q_passage[qid][did]
            for t_did, _ in triggers.items():

                triggered_passage = triggers[t_did] + ' ' + passages_dict[did]
                triggered_passage_ids = tokenizer.encode(triggered_passage, add_special_tokens=True)
                triggered_passage_ids = torch.tensor(triggered_passage_ids[1:], device=device).unsqueeze(0)

                input_ids_pos = torch.cat([query_ids, triggered_passage_ids], dim=-1)
                token_type_ids_pos = torch.cat([torch.zeros_like(query_ids), torch.ones_like(triggered_passage_ids)],
                                               dim=-1)
                attention_mask_pos = torch.ones_like(input_ids_pos)

                outputs = model(input_ids_pos=input_ids_pos,
                                attention_mask_pos=attention_mask_pos,
                                token_type_ids_pos=token_type_ids_pos,
                                input_ids_neg=placeholder_input_ids_neg,
                                attention_mask_neg=placeholder_attention_mask_neg,
                                token_type_ids_neg=placeholder_type_ids_neg)[0]
                new_score = outputs[0, 1].item()

                if tmp_best_new_score < new_score:
                    tmp_best_new_score = new_score
                print("New high score: {:.4f}".format(new_score))
                print("Trigger: {}".format(triggers[t_did]))

            new_rank = len(old_scores) - bisect.bisect_left(old_scores, tmp_best_new_score)

            total_docs_cnt += 1
            boost_rank_list.append(old_rank - new_rank)
            if old_rank > new_rank:
                success_cnt += 1
                if new_rank <= 500:
                    less_500_cnt += 1
                    if new_rank <= 100:
                        less_100_cnt += 1
                        if new_rank <= 50:
                            less_50_cnt += 1
                            if new_rank <= 10:
                                less_10_cnt += 1

            print(f'Query id={qid}, Doc id={did}, '
                  f'old score={raw_score:.4f}, new score={tmp_best_new_score:.4f}, old rank={old_rank}, new rank={new_rank}')

        print('\n\n')

    boost_success_rate = success_cnt / (total_docs_cnt + 0.0) * 100
    less_500_rate = less_500_cnt / (total_docs_cnt + 0.0) * 100
    less_100_rate = less_100_cnt / (total_docs_cnt + 0.0) * 100
    less_50_rate = less_50_cnt / (total_docs_cnt + 0.0) * 100
    less_10_rate = less_10_cnt / (total_docs_cnt + 0.0) * 100
    avg_boost_rank = np.average(boost_rank_list)
    res_str = 'Boost Success Rate: {}\n' \
              'Average Boost Rank: {}\n' \
              'less than 500 Rate: {}\n' \
              'less than 100 Rate: {}\n' \
              'less than 50 Rate: {}\n' \
              'less than 10 Rate: {}\n'.format(boost_success_rate, avg_boost_rank, less_500_rate, less_100_rate,
                                               less_50_rate, less_10_rate)
    print(res_str)


if __name__ == "__main__":
    main()
