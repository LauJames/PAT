"""
Using the generate triggers to test its transferability based on pairwise ranker
"""
import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import json
import torch
import numpy as np
import bisect
from tqdm import tqdm
from torch import cuda


from transformers import BertTokenizerFast, BertConfig, BertModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from bert_ranker.models import pairwise_bert
from apex import amp
from data_utils import prepare_data_and_scores
from bert_ranker.models.bert_cat import BERT_Cat

device = 'cuda:0' if cuda.is_available() else 'cpu'


def main():
    lm_param = 0.6
    nsp_param = 5.0
    target_imitation = 'pairwise.v2'
    # target_imitation = 'pairwise.wo.imitation'
    # experiment_name = 'ms-marco-MiniLM-L-12-v2'
    transformer_model_name = "bert-large-uncased"
    # transformer_model_name = "ms-marco-MiniLM-L-12-v2"
    # transformer_model_name = "pairwise.v2"
    if transformer_model_name == 'bert-large-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(transformer_model_name)
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-large-msmarco")
    elif transformer_model_name == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(transformer_model_name)
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
    elif transformer_model_name == 'pt-tinybert-msmarco':
        tokenizer = AutoTokenizer.from_pretrained("nboost/pt-tinybert-msmarco")
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-tinybert-msmarco")
    elif transformer_model_name == 'distilbert-cat-margin_mse-T2-msmarco':
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased")
        model = BERT_Cat.from_pretrained("sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco")
    elif transformer_model_name == 'ms-marco-MiniLM-L-12-v2':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    elif transformer_model_name == 'pairwise.v2':
        model = pairwise_bert.BertForPairwiseLearning.from_pretrained('bert-base-uncased')
        model_path = prodir + '/bert_ranker/saved_models/Imitation.MiniLM.L12.v2.BertForPairwiseLearning.bert-base-uncased.pth'
        model.load_state_dict(torch.load(model_path))
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    elif transformer_model_name == 'pairwise.v1':
        model = pairwise_bert.BertForPairwiseLearning.from_pretrained('bert-base-uncased')
        model_path = prodir + '/bert_ranker/saved_models/Imitation.bert_large.further_train.BertForPairwiseLearning.bert-base-uncased.pth'
        model.load_state_dict(torch.load(model_path))
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    model = amp.initialize(model)

    # load triggers
    trigger_path = curdir + '/saved_results/cands_triggers_{}_{}_{}.json'.format(target_imitation, lm_param, nsp_param)
    with open(trigger_path, 'r', encoding='utf-8') as fin:
        q_trigger_dict = json.loads(fin.readline())

    # load victim models results
    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(
        experiment_name=transformer_model_name,
        data_name='dl',
        mode='test',
        top_k=10,
        least_num=50)

    total_docs_cnt = 0
    success_cnt = 0
    less_500_cnt = 0
    less_100_cnt = 0
    less_50_cnt = 0
    less_10_cnt = 0
    boost_rank_list = []
    query_keys = list(queries.keys())

    with torch.no_grad():
        for qid in tqdm(query_keys):
            query = queries[qid]
            old_scores = query_scores[qid][::-1]

            triggers = q_trigger_dict[qid]

            for did in target_q_passage[qid]:
                tmp_best_new_score = -1e9
                old_rank, raw_score = target_q_passage[qid][did]
                for t_did, _ in triggers.items():

                    triggered_passage = triggers[t_did] + ' ' + passages_dict[did]

                    batch_encoding = tokenizer([[query, triggered_passage]], max_length=256, padding="max_length",
                                               truncation=True, return_tensors='pt')

                    if transformer_model_name == 'ms-marco-MiniLM-L-12-v2':
                        outputs = model(**(batch_encoding.to(device)))
                        new_score = outputs.logits.squeeze().item()
                    elif transformer_model_name == 'bert-large-uncased':
                        outputs = model(**(batch_encoding.to(device)))
                        new_score = outputs.logits[0, -1].item()
                    else:
                        pos_input_ids = batch_encoding['input_ids'].to(device)
                        pos_token_type_ids = batch_encoding['token_type_ids'].to(device)
                        pos_attention_mask = batch_encoding['attention_mask'].to(device)
                        neg_input_ids = batch_encoding['input_ids'].to(device)
                        neg_token_type_ids = batch_encoding['token_type_ids'].to(device)
                        neg_attention_mask = batch_encoding['attention_mask'].to(device)
                        outputs = model(input_ids_pos=pos_input_ids,
                                        attention_mask_pos=pos_attention_mask,
                                        token_type_ids_pos=pos_token_type_ids,
                                        input_ids_neg=neg_input_ids,
                                        attention_mask_neg=neg_attention_mask,
                                        token_type_ids_neg=neg_token_type_ids,
                                        )[0]
                        new_score = outputs[0, 1].item()

                    if tmp_best_new_score < new_score:
                        tmp_best_new_score = new_score
                        print("New high score: {:.4f}".format(new_score))
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
    print("{}\n to: \t {}".format(trigger_path, transformer_model_name))


if __name__ == "__main__":
    main()