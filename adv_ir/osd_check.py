import csv
from itertools import count
import plistlib
import sys
import os
from collections import defaultdict

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import json
import logging
import time

import numpy as np
import pandas as pd
from tqdm import tqdm
from torch import cuda
import torch.nn.functional as F
from copy import deepcopy
from sklearn.feature_extraction.text import TfidfVectorizer

from data_utils import prepare_data_and_scores


save_folder = curdir + '/osd'
# raw passage
used_passages_path = save_folder + '/passages.txt'
ranker_results_dir = prodir + '/bert_ranker/results/runs'
mspr_data_folder = prodir + '/data/msmarco_passage'

csv_folder = curdir + '/human_eval'
# passage concatentated with query/trigger
query_plus_path = csv_folder + '/passage_query.csv'
pat_path = csv_folder + '/passage_pat.csv'

def get_all_passages():
    run_file = ranker_results_dir + '/runs.ms-marco-MiniLM-L-12-v2.public.bert.msmarco.dl2019.csv'
    target_q_dict = defaultdict(list)
    # 首先加载所有q对应的passage,rank,score
    with open(run_file) as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split('\t')
            rank = int(rank)
            score = float(score)
            target_q_dict[qid].append((pid, rank, score))
    
    collection_path = mspr_data_folder + '/collection_queries/collection.tsv'
    collection_df = pd.read_csv(collection_path, sep='\t', names=['docid', 'document_string'])
    collection_df['docid'] = collection_df['docid'].astype(str)
    collection_str = collection_df.set_index('docid').to_dict()['document_string']

    used_qids = list(target_q_dict.keys())

    pid_list = []
    for qid in tqdm(used_qids):
        for pid_tup in target_q_dict[qid]:
            pid_list.append(pid_tup[0])
    
    uni_pid_list = list(set(pid_list))
    passage_list = []
    for pid in uni_pid_list:
        passage_list.append(collection_str[pid])
    
    with open(used_passages_path, 'w') as fout:
        for tmp in passage_list:
            fout.write(tmp + '\n')


def build_tfidf():
    passage_list = []
    with open(used_passages_path, 'r') as fin:
        for line in fin:
            passage_list.append(line.strip())
    
    vectorizer = TfidfVectorizer()
    # train
    tfidf_vector = vectorizer.fit(passage_list)
    return vectorizer


def get_max_tfidf_passage(passage, vectorizer):
    tfidf_list = vectorizer.transform([passage])[0]
    return np.max(tfidf_list)


def get_query_doc_tfidf_match_score(query, passage, vectorizer):
    vocab = vectorizer.vocabulary_
    query_vec = TfidfVectorizer()
    query_vec.fit([query])
    query_word_list = query_vec.get_feature_names()

    passage_vec = TfidfVectorizer()
    passage_vec.fit([passage])
    passage_word_list = passage_vec.get_feature_names()
    

    passage_fit = vectorizer.transform([passage]).toarray()[0]
    target_words = list(set(query_word_list) & set(passage_word_list))
    num = len(query_word_list)
    pq_score = 0
    for tmp_word in target_words:
        try:
            tmp_word_id = vocab[tmp_word]
            pq_score += passage_fit[tmp_word_id]
            # num += 1
        except:
            print("unknown")
    return pq_score, num

def get_osd_score(max_tfidf, match_score, num):
    spamicity_score = match_score / (num * max_tfidf)
    return spamicity_score


def load_query_triggered_passages(fpath):
    query_list = []
    passage_list = []
    is_first = True
    with open(fpath, 'r') as f:
        for line in f:
            if is_first:
                is_first = False
                continue
            qid, pid, query, passage = line.strip().split('\t')
            query_list.append(query)
            passage_list.append(passage)
    
    return query_list, passage_list


def get_spam_scores(target_path, vectorizer):
    q_list, p_list = load_query_triggered_passages(target_path)
    qplus_score_list = []
    for tmp_query, tmp_passage in zip(q_list, p_list):
        tmp_max_tfidf = get_max_tfidf_passage(tmp_query, vectorizer)
        tmp_match_score, tmp_num = get_query_doc_tfidf_match_score(tmp_query, tmp_passage, vectorizer)
        tmp_spam_score = get_osd_score(tmp_max_tfidf, tmp_match_score, tmp_num)
        qplus_score_list.append(tmp_spam_score)

    return qplus_score_list


def detection_rate(target_list, threshold=0.3, eps=1e-12):
    positive_cnt = len([i for i in target_list if i > threshold])
    return positive_cnt / (len(target_list) + eps)

if __name__ == "__main__":
    vectorizer = build_tfidf()

    qplus_score_list = get_spam_scores(query_plus_path,vectorizer)
    pat_score_list = get_spam_scores(pat_path,vectorizer)

    # print(qplus_score_list)
    # print(pat_score_list)

    threshold = 0.05

    print("Detection rate of Query+: {}".format(detection_rate(qplus_score_list, threshold=threshold)))
    print("Detection rate of PAT: {}".format(detection_rate(pat_score_list, threshold=threshold)))


