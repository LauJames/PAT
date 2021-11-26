import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))

from tqdm import tqdm

import pandas as pd
import collections
import pickle as pkl

data_folder = prodir + '/data/trec_dl_2019'
qrel_path = data_folder + '/2019qrels-pass.txt'
transformed_qrels_path = data_folder + "/trec_dl_2019_qrels.tsv"
test_query_passage_path = data_folder + '/msmarco-passagetest2019-top1000.tsv'
queries_path = data_folder + '/msmarco-test2019-queries.tsv'


def build_test_trec_data_from_qrel_and_queries():
    build_set_path = data_folder + "/trec_dl2019_passage_test1000_full.tsv"
    relevant_pairs = set()
    relevant_dict = collections.defaultdict(dict)
    qid_list = []
    with open(qrel_path, 'r') as f:
        for line in f:
            qid, _, did, rating = line.strip().split(' ')
            qid_list.append(qid)
            if int(rating) > 0:
                relevant_pairs.add('\t'.join([qid, did]))
                relevant_dict[qid][did] = int(rating)

    qid_list = list(set(qid_list))
    samples_with_label_list = []
    pos_cnt = 0
    with open(test_query_passage_path, 'r') as f:
        for line in f:
            query_id, doc_id, query, doc = line.strip().split('\t')
            if query_id in qid_list:
                label = 0
                if '\t'.join([query_id, doc_id]) in relevant_pairs:
                    label = relevant_dict[query_id][doc_id]
                    # label = 1
                    pos_cnt += 1
                samples_with_label_list.append((query_id, doc_id, query, doc, label))
            else:
                continue
    print("Positive labels count: {}".format(pos_cnt))
    final_df = pd.DataFrame(samples_with_label_list, columns=["qid", "did", "query", "passage", "label"])
    final_df.to_csv(build_set_path, sep='\t', index=False, header=True)
    print("Done!")


def gen_input_from_csv_to_pkl():
    csv_path = data_folder + "/trec_dl2019_passage_test1000_full.tsv"
    pkl_path = data_folder + "/trec_dl2019_passage_test1000_full.pkl"
    sample_df = pd.read_csv(csv_path, sep="\t", names=["qid", "did", "query", "passage", "label"], header=0)
    instances = []
    labels = []
    qids = []
    pids = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        instances.append((row["query"], row["passage"]))
        labels.append(row["label"])
        qids.append(row["qid"])
        pids.append(row["did"])

    with open(pkl_path, 'wb') as f:
        pkl.dump(instances, f)
        pkl.dump(labels, f)
        pkl.dump(qids, f)
        pkl.dump(pids, f)
    print("Total of {} instances were cached.".format(len(labels)))


def transform_qrels():
    qrel_list = []
    with open(qrel_path, 'r') as f:
        for line in f:
            qid, _, did, rating = line.strip().split()
            if int(rating) > 1:
                qrel_list.append((qid, 'Q0', did, 1))
            else:
                qrel_list.append((qid, 'Q0', did, 0))
    final_df = pd.DataFrame(qrel_list)
    final_df.to_csv(transformed_qrels_path, sep='\t', index=False, header=False)


if __name__ == "__main__":
    transform_qrels()

    # build_test_trec_data_from_qrel_and_queries()
    # gen_input_from_csv_to_pkl()
