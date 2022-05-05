import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))

import pandas as pd
import collections
import random

ms_data_folder = prodir + '/data/msmarco_passage'
sampled_triples_path = prodir + '/data/msmarco_passage/triples_from_runs'
runs_data_folder = prodir + '/bert_ranker/results/runs'

runs_bert_large = runs_data_folder + '/runs.bert-large-uncased.public.bert.msmarco.eval_full_dev1000.csv'
runs_MiniLM_L_12 = runs_data_folder + '/runs.ms-marco-MiniLM-L-12-v2.public.bert.msmarco.eval_full_dev1000.csv'

random_seed = 666


def sample_from_dev_runs(run_path, save_pre_fix, top_n=25, last_sample=4):
    relevant_pairs_dict = collections.defaultdict(list)
    with open(run_path, 'r') as f:
        for line in f:
            qid, _, did, _, _, _ = line.strip().split('\t')
            relevant_pairs_dict[qid].append(did)

    sampled_triples_ids = []
    for seed, (qid, did_list) in enumerate(relevant_pairs_dict.items()):
        # for top n
        tmp_top_n = top_n if len(did_list) > top_n else len(did_list)
        top_n_dids = did_list[:tmp_top_n]
        last_dids = did_list[tmp_top_n:]
        for i in range(tmp_top_n):
            random.seed(random_seed + seed + i)
            pos_did = top_n_dids[i]
            for j in range(i + 1, tmp_top_n):
                neg_did = top_n_dids[j]
                sampled_triples_ids.append((qid, pos_did, neg_did))
            # for top n corresponding random negative sampling from another
            if len(last_dids) < last_sample:
                selected_last_neg = last_dids
            else:
                selected_last_neg = random.sample(last_dids, last_sample)
            for tmp_did in selected_last_neg:
                sampled_triples_ids.append((qid, pos_did, tmp_did))
            # print(len(sampled_triples_ids))

    # save qid pos_did neg_did list
    print("Sampled {} triples from: {}".format(len(sampled_triples_ids), run_path))

    # load doc_id to string
    collection_df = pd.read_csv("{}/collection.tsv".format(ms_data_folder), sep='\t',
                                names=['docid', 'document_string'])
    collection_df['docid'] = collection_df['docid'].astype(str)
    collection_str = collection_df.set_index('docid').to_dict()['document_string']

    # load query
    query_df = pd.read_csv("{}/queries.dev.tsv".format(ms_data_folder), names=['qid', 'query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(str)
    queries_str = query_df.set_index('qid').to_dict()['query_string']

    sampled_triples_text_list = []
    for (qid, pos_did, neg_did) in sampled_triples_ids:
        sampled_triples_text_list.append((queries_str[qid], collection_str[pos_did], collection_str[neg_did]))

    final_text_triples_df = pd.DataFrame(sampled_triples_text_list)
    save_text_path = save_pre_fix + '_text.top_{}_last_{}.csv'.format(top_n, last_sample)
    final_text_triples_df.to_csv(save_text_path, sep='\t', index=False, header=False)
    print("Saved sampled triples text into : {}".format(save_text_path))


if __name__ == "__main__":
    # save_triples_prefix = sampled_triples_path + '/minilm_l12_sampled_triples'
    save_triples_prefix = sampled_triples_path + '/bert_large_sampled_triples'
    sample_from_dev_runs(runs_bert_large, save_triples_prefix)



