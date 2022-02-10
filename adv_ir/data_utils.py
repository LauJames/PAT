import os
import sys
from collections import defaultdict

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import pandas as pd
import pickle as pkl

mspr_data_folder = prodir + '/data/msmarco_passage'
ranker_results_dir = prodir + '/bert_ranker/results/runs'
trec_dl_data_folder = prodir + '/data/trec_dl_2019'


def pick_target_query_doc_and_best_scores(target_name='pointwise',
                                          data_name='dl',
                                          top_k=10,
                                          least_num=10):
    if data_name == 'dl':
        if target_name == 'imitate.v2':
            run_file = ranker_results_dir + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM.csv'
        elif target_name == 'imitate.v1':
            run_file = ranker_results_dir + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_bert_large.csv'
        elif target_name == 'pairwise.pseudo':
            run_file = ranker_results_dir + '/runs/runs.bert-base-uncased.pairwise.triples.pseudo.csv'
        elif target_name == 'pairwise.wo.imitation':
            run_file = ranker_results_dir + '/runs.bert-base-uncased.pairwise.triples.dl2019.csv'
        elif target_name == 'pointwise':
            run_file = ranker_results_dir + '/runs.bert-base-uncased.pointwise.triples.2M.dl2019.csv'
        elif target_name == 'mini':
            run_file = ranker_results_dir + '/runs.ms-marco-MiniLM-L-12-v2.public.bert.msmarco.dl2019.csv'
        elif target_name == 'large':
            run_file = ranker_results_dir + '/runs.bert-large-uncased.public.bert.msmarco.dl2019.csv'
        else:
            raise ValueError("Experiment name Error!")
    else:
        raise ValueError("Data name does not exist!")

    target_q_dict = defaultdict(list)
    query_scores = defaultdict(list)
    target_q_pid = defaultdict(dict)
    all_qid_pid_dict = defaultdict(list)
    with open(run_file) as f:
        for line in f:
            qid, _, pid, rank, score, _ = line.strip().split('\t')
            rank = int(rank)
            score = float(score)
            target_q_dict[qid].append((pid, rank, score))

    # get target query and tail passages with its rank
    for qid in target_q_dict.keys():
        least_query_scores = target_q_dict[qid][-least_num:]
        for pid, rank, score in least_query_scores:
            target_q_pid[qid][pid] = (rank, score)
        for pid, _, t_socre in target_q_dict[qid]:
            query_scores[qid].append(t_socre)
            all_qid_pid_dict[qid].append(pid)

    best_query_score = {}
    # get best scores
    all_qid_list = list(target_q_dict.keys())
    for qid in all_qid_list:
        if len(target_q_dict[qid]) < 100:
            target_q_pid.pop(qid)
            query_scores.pop(qid)
            all_qid_pid_dict.pop(qid)
            continue
        top_query_scores = target_q_dict[qid][:top_k]
        scores = [t[2] for t in top_query_scores]
        pids = [t[0] for t in top_query_scores]
        best_query_score[qid] = (scores, pids)
    return target_q_pid, query_scores, best_query_score, all_qid_pid_dict


def prepare_data_and_scores(target_name='mini',
                            data_name='dl',
                            mode='',
                            top_k=10,
                            least_num=10):
    if data_name == 'dl':
        collection_path = mspr_data_folder + '/collection.tsv'
        queries_path = trec_dl_data_folder + '/msmarco-test2019-queries.tsv'
        if mode == 'test' and 'pseudo' not in target_name:
            preprocessed_pkl_path = curdir + '/tmp_data/{}.{}.test.pkl'.format(data_name, target_name)
        else:
            preprocessed_pkl_path = curdir + '/tmp_data/{}.{}.pkl'.format(data_name, target_name)
    else:
        raise ValueError("Error data name!")

    if not os.path.exists(preprocessed_pkl_path):
        target_q_pid, query_scores, best_query_score, all_qid_pid_dict = pick_target_query_doc_and_best_scores(
            target_name, data_name, top_k, least_num)
        print("{} does not exist, creating it...".format(preprocessed_pkl_path))
        # load doc_id=pid to string
        collection_df = pd.read_csv(collection_path, sep='\t', names=['docid', 'document_string'])
        collection_df['docid'] = collection_df['docid'].astype(str)
        collection_str = collection_df.set_index('docid').to_dict()['document_string']

        # load query
        query_df = pd.read_csv(queries_path, names=['qid', 'query_string'], sep='\t')
        query_df['qid'] = query_df['qid'].astype(str)
        queries_str = query_df.set_index('qid').to_dict()['query_string']

        passages_dict = {}
        best_query_sent = defaultdict(list)
        queries = {}
        for qid in target_q_pid.keys():
            queries[qid] = queries_str[qid]

            if qid not in best_query_sent:
                best_query_sent[qid].append(max(best_query_score[qid][0]))

            for pid in all_qid_pid_dict[qid]:
                passages_dict[pid] = collection_str[pid]
                if pid in best_query_score[qid][1]:
                    best_query_sent[qid].append(collection_str[pid])

        with open(preprocessed_pkl_path, 'wb') as f:
            pkl.dump(target_q_pid, f)
            pkl.dump(query_scores, f)
            pkl.dump(best_query_sent, f)
            pkl.dump(queries, f)
            pkl.dump(passages_dict, f)
    else:
        print("Load data from {}".format(preprocessed_pkl_path))
        with open(preprocessed_pkl_path, 'rb') as f:
            target_q_pid = pkl.load(f)
            query_scores = pkl.load(f)
            best_query_sent = pkl.load(f)
            queries = pkl.load(f)
            passages_dict = pkl.load(f)

    return target_q_pid, query_scores, best_query_sent, queries, passages_dict


def get_query_passage_by_qid_rank(qid, rank):
    target_q_passage, query_scores, best_query_sent, queries, passages_dict = prepare_data_and_scores(
        target_name='pairwise.v2',
        data_name='dl',
        top_k=10,
        least_num=10)
    query = queries[qid]
    for did in target_q_passage[qid]:
        old_rank, _ = target_q_passage[qid][did]
        if old_rank == rank:
            passage = passages_dict[did]
            print("Query:\t{}".format(query))
            print("Passage:\t{}".format(passage))
            print("Rank:\t{}".format(rank))


if __name__ == "__main__":
    prepare_data_and_scores(target_name='pairwise.v2', data_name='dl', top_k=10, least_num=10)










