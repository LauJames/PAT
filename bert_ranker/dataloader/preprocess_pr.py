import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))

from tqdm import tqdm

import csv
import gzip
import codecs
import pandas as pd
import os
import tarfile
import shutil
import zipfile
import collections
import pickle as pkl

data_folder = prodir + '/data/msmarco_passage'


def transform_trec2020pr_to_dfs(path):
    """
    Transforms TREC 2020 Passage Ranking files (https://microsoft.github.io/TREC-2020-Deep-Learning/)
    to train, valid and test dfs containing only positive query-passage combinations.

    Args:
        path: str with the path for the TREC folder containing:
            - collection.tar.gz (uncompressed: collection.tsv)
            - queries.tar.gz (uncompressed: queries.train.tsv, queries.dev.tsv)
            - qrels.dev.tsv
            - qrels.train.tsv

    Returns: (train, valid, test) pandas DataFrames
    """
    query_df_train = pd.read_csv("{}/queries.train.tsv".format(path), names=['qid', 'query_string'], sep='\t')
    query_df_train['qid'] = query_df_train['qid'].astype(int)
    queries_str_train = query_df_train.set_index('qid').to_dict()['query_string']

    query_df_dev = pd.read_csv("{}/queries.dev.tsv".format(path), names=['qid', 'query_string'], sep='\t')
    query_df_dev['qid'] = query_df_dev['qid'].astype(int)
    queries_str_dev = query_df_dev.set_index('qid').to_dict()['query_string']

    collection_str = pd.read_csv("{}/collection.tsv".format(path), sep='\t', names=['docid', 'document_string']) \
        .set_index('docid').to_dict()['document_string']

    qrels_train = pd.read_csv("{}/qrels.train.tsv".format(path), sep="\t", names=["topicid", "_", "docid", "rel"])
    qrels_dev = pd.read_csv("{}/qrels.dev.tsv".format(path), sep="\t", names=["topicid", "_", "docid", "rel"])

    train = []
    for idx, row in tqdm(qrels_train.sort_values("topicid").iterrows()):
        train.append([queries_str_train[row["topicid"]], collection_str[row["docid"]]])
    train_df = pd.DataFrame(train, columns=["query", "passage"])

    dev = []
    for idx, row in tqdm(qrels_dev.sort_values("topicid").iterrows()):
        dev.append([queries_str_dev[row["topicid"]], collection_str[row["docid"]]])
    all_dev_df = pd.DataFrame(dev, columns=["query", "passage"])

    dev_df, test_df = all_dev_df[0:all_dev_df.shape[0] // 2], all_dev_df[all_dev_df.shape[0] // 2:]

    return train_df, dev_df, test_df


def trec_dl_processor(data_folder):
    """
    Extracts the files downloaded and process them into a DF with ["query", "passage"]
    """
    collection_tar = tarfile.open(data_folder + "/collection.tar.gz")
    collection_tar.extractall(data_folder)
    collection_tar.close()
    queries_tar = tarfile.open(data_folder + "/queries.tar.gz")
    queries_tar.extractall(data_folder)
    queries_tar.close()

    train, valid, test = transform_trec2020pr_to_dfs(data_folder)
    train.to_csv(data_folder + "/train.tsv", sep="\t", index=False)
    valid.to_csv(data_folder + "/valid.tsv", sep="\t", index=False)
    test.to_csv(data_folder + "/test.tsv", sep="\t", index=False)


# build dev set, especially for the sub small set
def build_data_from_qrel_and_queries(qrels_path, queries_path, set_name):
    # using the official dev1000 as the negtive sample reference
    dev_top1000_path = data_folder + "/top1000.dev"
    build_set_path = data_folder + "/sampled_set/{}.tsv".format(set_name)

    relevant_pairs = set()
    with open(qrels_path, 'r') as f:
        for line in f:
            qid, _, did, _ = line.strip().split('\t')
            relevant_pairs.add('\t'.join([qid, did]))

    dev_top1000_df = pd.read_csv(dev_top1000_path, sep='\t', names=['qid', 'pid', 'query', 'passage'])

    query_df = pd.read_csv(queries_path, names=['qid', 'query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(int)

    sample_df = pd.DataFrame()
    for idx, row in query_df.sort_values('qid').iterrows():
        tmp_df = dev_top1000_df.loc[dev_top1000_df['qid'] == row['qid']]
        sample_df = pd.concat([sample_df, tmp_df], axis=0, ignore_index=True)
    sample_csv_path = data_folder + "/top1000.{}.dev.tsv".format(set_name)
    sample_df.to_csv(sample_csv_path, sep='\t', index=False, header=False)

    queries_docs = collections.defaultdict(list)
    query_ids = {}
    with open(sample_csv_path, 'r') as f:
        for i, line in enumerate(f):
            query_id, doc_id, query, doc = line.strip().split('\t')
            label = 0
            if '\t'.join([query_id, doc_id]) in relevant_pairs:
                label = 1
            queries_docs[query].append((doc_id, doc, label))
            query_ids[query] = query_id

    # Add fake paragraphs to the queries that have less than num_eval_docs.
    queries = list(queries_docs.keys())  # Need to copy keys before iterating.
    total_pad_num = 0
    for query in queries:
        docs = queries_docs[query]
        pad_num = max(0, 1000 - len(docs))
        if pad_num != 0:
            print(pad_num)
        total_pad_num += pad_num
        docs += pad_num * [('00000000', 'FAKE DOCUMENT', 0)]
        queries_docs[query] = docs
    print("Add {} fake documents".format(total_pad_num))
    assert len(
        set(len(docs) == 1000 for docs in queries_docs.values())) == 1, (
        'Not all queries have {} docs'.format(1000))

    # save
    sample_with_label_list = []
    for query, doc_ids_docs in queries_docs.items():
        query_id = query_ids[query]
        doc_ids, docs, labels = zip(*doc_ids_docs)
        for doc_id, doc, label in zip(doc_ids, docs, labels):
            sample_with_label_list.append((query_id, doc_id, query, doc, label))
    final_df = pd.DataFrame(sample_with_label_list, columns=["qid", "did", "query", "passage", "label"])
    final_df.to_csv(build_set_path, sep='\t', index=False, header=True)
    print("Done!")


def build_dev_data_from_qrel_run_queries(qrels_path, run_path, queries_path, set_name):
    build_set_path = data_folder + "/sampled_set/{}.tsv".format(set_name)
    qid_list = []
    relevant_pairs = set()
    with open(qrels_path, 'r') as f:
        for line in f:
            qid, _, did, _ = line.strip().split('\t')
            qid_list.append(qid)
            relevant_pairs.add('\t'.join([qid, did]))

    qid_list = list(set(qid_list))
    runs = {}
    with open(run_path, 'r') as f:
        for line in f:
            qid, doc_id, rank = line.strip().split()
            if qid not in runs:
                runs[qid] = []
            runs[qid].append(doc_id)

    # load doc_id to string
    collection_df = pd.read_csv("{}/collection.tsv".format(data_folder), sep='\t', names=['docid', 'document_string'])
    collection_df['docid'] = collection_df['docid'].astype(str)
    collection_str = collection_df.set_index('docid').to_dict()['document_string']

    # load query
    query_df = pd.read_csv(queries_path, names=['qid', 'query_string'], sep='\t')
    query_df['qid'] = query_df['qid'].astype(str)
    queries_str = query_df.set_index('qid').to_dict()['query_string']

    sample_with_label_list = []
    no_pos_cnt = 0
    for query_id, docs in tqdm(runs.items()):
        pos_cnt = 0
        if query_id in qid_list:
            if len(docs) != 1000:
                print("Do not meet 1000 documents!!!")

            for doc_id in docs:
                label = 0
                if '\t'.join([query_id, doc_id]) in relevant_pairs:
                    label = 1
                    pos_cnt += 1
                sample_with_label_list.append((query_id, doc_id, queries_str[query_id], collection_str[doc_id], label))
        if pos_cnt == 0:
            no_pos_cnt += 1
    print("There are {} queries having no postive sample...".format(no_pos_cnt))
    final_df = pd.DataFrame(sample_with_label_list, columns=["qid", "did", "query", "passage", "label"])
    final_df.to_csv(build_set_path, sep='\t', index=False, header=True)
    print("{} saved done!".format(len(final_df)))


def build_dev_from_top1000_and_rel(qrels_path, set_name):
    dev_top1000_path = data_folder + "/top1000.dev"
    build_set_path = data_folder + "/sampled_set/{}.tsv".format(set_name)

    relevant_pairs = set()
    qid_list = []
    with open(qrels_path, 'r') as f:
        for line in f:
            qid, _, did, _ = line.strip().split('\t')
            qid_list.append(qid)
            relevant_pairs.add('\t'.join([qid, did]))

    qid_list = list(set(qid_list))
    sample_with_label_list = []
    pos_cnt = 0
    with open(dev_top1000_path, 'r') as f:
        for i, line in enumerate(f):
            query_id, doc_id, query, doc = line.strip().split('\t')
            if query_id in qid_list:
                label = 0
                if '\t'.join([query_id, doc_id]) in relevant_pairs:
                    label = 1
                    pos_cnt += 1
                sample_with_label_list.append((query_id, doc_id, query, doc, label))
            else:
                continue
    print("Positive labels count: {}".format(pos_cnt))
    final_df = pd.DataFrame(sample_with_label_list, columns=["qid", "did", "query", "passage", "label"])
    final_df.to_csv(build_set_path, sep='\t', index=False, header=True)
    print("Done!")


def gen_dev_input_from_csv_to_pkl(set_name):
    csv_path = data_folder + "/sampled_set/{}.tsv".format(set_name)
    sample_df = pd.read_csv(csv_path, sep="\t", names=["qid", "did", "query", "passage", "label"], header=0)
    instances = []
    labels = []
    qids = []
    pids = []
    for _, row in tqdm(sample_df.iterrows(), total=len(sample_df)):
        # instances.append((row["query"], row["passage"], row["passage"]))
        instances.append((row["query"], row["passage"]))
        labels.append(row["label"])
        qids.append(row["qid"])
        pids.append(row["did"])

    with open(data_folder + "/sampled_set/{}.dev.pkl".format(set_name), 'wb') as f:
        pkl.dump(instances, f)
        pkl.dump(labels, f)
        pkl.dump(qids, f)
        pkl.dump(pids, f)
    print("Total of {} instances were cached.".format(len(labels)))


def gen_pairwise_for_eval_triples(sample_num=512000):
    train_triples_csv_path = data_folder + "/triples.train.small.tsv"
    pkl_path = data_folder + "/sampled_set/triples.dev.pairwise.{}.pkl".format(sample_num)
    train_df = pd.read_csv(train_triples_csv_path, sep='\t', skiprows=20480000, names=["query", "pos", "neg"],
                           nrows=sample_num)
    dev_instances = []
    dev_labels = []
    for _, rows in tqdm(train_df.iterrows(), total=len(train_df)):
        # query, reL_doc, ns_doc
        dev_instances.append((rows["query"], rows["pos"], rows["neg"]))
        dev_labels.append(1)
        # add reverse
        dev_instances.append((rows["query"], rows["neg"], rows["pos"],))
        dev_labels.append(0)

    with open(pkl_path, 'wb') as f:
        pkl.dump(dev_instances, f)
        pkl.dump(dev_labels, f)
    print("Total of {} instances are cached into {}".format(len(dev_labels), pkl_path))


# build dev set from top1000.dev.tsv to get (query, passage, label)
def gen_dev1000_data(data_path, num_eval_docs=1000, add_fake=True):
    relevant_pairs = set()
    qrels_file = data_folder + "/qrels.dev.tsv"
    save_file = data_folder + "/sampled_set/dev_1000_triple_without_fake.tsv"
    ids_file = data_folder + '/sampled_set/query_passage_ids_dev.txt'
    if add_fake:
        save_file = data_folder + "/sampled_set/dev_1000_triple.tsv"
    with open(qrels_file, 'r') as f:
        for line in f:
            qid, _, did, _ = line.strip().split('\t')
            relevant_pairs.add('\t'.join([qid, did]))

    queries_docs = collections.defaultdict(list)
    query_ids = {}
    with open(data_path + '/top1000.dev', 'r') as f:
        for i, line in enumerate(f):
            query_id, doc_id, query, doc = line.strip().split('\t')
            label = 0
            if '\t'.join([query_id, doc_id]) in relevant_pairs:
                label = 1
            queries_docs[query].append((doc_id, doc, label))
            query_ids[query] = query_id

    # Add fake paragraphs to the queries that have less than num_eval_docs.
    queries = list(queries_docs.keys())  # Need to copy keys before iterating.
    for query in queries:
        docs = queries_docs[query]
        if add_fake:
            docs += max(
                0, num_eval_docs - len(docs)) * [('00000000', 'FAKE DOCUMENT', 0)]
        queries_docs[query] = docs

    assert len(
        set(len(docs) == num_eval_docs for docs in queries_docs.values())) == 1, (
        'Not all queries have {} docs'.format(num_eval_docs))

    # save
    dev_list = []
    ids_f = open(ids_file, 'w')
    for query, doc_ids_docs in queries_docs.items():
        query_id = query_ids[query]
        doc_ids, docs, labels = zip(*doc_ids_docs)
        for doc_id, doc, label in zip(doc_ids, docs, labels):
            dev_list.append((query, doc, label))
            ids_f.write('\t'.join([query_id, doc_id]) + '\n')
    dev_df = pd.DataFrame(dev_list, columns=["query", "passage", "label"])
    dev_df.to_csv(save_file, sep='\t', index=False, header=True)
    ids_f.close()


if __name__ == "__main__":
    # build dev set, used for evaluating final model
    qrels_path = data_folder + '/qrels.dev.tsv'
    queries_path = data_folder + '/collection_queries/queries.dev.small.tsv'

    set_name = 'run_bm25'
    runs_path = data_folder + '/runs_from_public/run.bm25.dev.small.tsv'
    build_dev_data_from_qrel_run_queries(qrels_path, runs_path, queries_path, set_name)
    gen_dev_input_from_csv_to_pkl(set_name)

    # build sub small dev set, used for accerlating evaluation during training process
    set_name = 'run_sub_small'
    runs_path = data_folder + '/runs_from_public/run.dev.sub_small.tsv'
    build_dev_data_from_qrel_run_queries(qrels_path, runs_path, queries_path, set_name)
    gen_dev_input_from_csv_to_pkl(set_name)
