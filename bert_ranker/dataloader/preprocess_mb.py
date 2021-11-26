import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))

from tqdm import tqdm
import pandas as pd
import pickle as pkl

data_folder = prodir + '/data/trec_mb_2014'
query_toks_path = data_folder + '/a.toks'
content_toks_path = data_folder + '/b.toks'
rel_judgement_path = data_folder + '/sim.txt'
rel_id_path = data_folder + '/id.txt'
url_path = data_folder + '/url.txt'


def get_list_from_txt(fpath):
    rlist = []
    with open(fpath, 'r', encoding='UTF-8') as f:
        for line in f:
            tmp_list = line.strip().replace('\n', '')
            rlist.append(tmp_list)
    return rlist


def build_mb_to_csv(build_csv_path):
    queries = get_list_from_txt(query_toks_path)
    docs = get_list_from_txt(content_toks_path)
    labels = get_list_from_txt(rel_judgement_path)
    runs_list = get_list_from_txt(rel_id_path)
    url_list = get_list_from_txt(url_path)

    qid_list, did_list = [], []
    for tmp_str in runs_list:
        qid, _, did, _, _, _ = tmp_str.split()
        qid_list.append(qid)
        did_list.append(did)

    sample_list = []
    for qid, did, query, passage, label in zip(qid_list, did_list, queries, docs, labels):
        sample_list.append((qid, did, query, passage, label))

    final_df = pd.DataFrame(sample_list, columns=["qid", "did", "query", "passage", "label"])
    final_df.to_csv(build_csv_path, sep='\t', index=False, header=True)
    print("Done!")


def gen_input_from_csv_to_pkl(csv_path):
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

    with open(data_folder + "/trec_mb_2014.pkl", 'wb') as f:
        pkl.dump(instances, f)
        pkl.dump(labels, f)
        pkl.dump(qids, f)
        pkl.dump(pids, f)
    print("Total of {} instances were cached.".format(len(labels)))


def build_mb_queries_doc_dict_to_csv():
    queries = get_list_from_txt(query_toks_path)
    docs = get_list_from_txt(content_toks_path)
    runs_list = get_list_from_txt(rel_id_path)

    qid_list, did_list = [], []
    for tmp_str in runs_list:
        qid, _, did, _, _, _ = tmp_str.split()
        qid_list.append(qid)
        did_list.append(did)

    qid_dict = {}
    did_dict = {}
    for qid, did, query, passage in zip(qid_list, did_list, queries, docs):
        qid_dict[qid] = query
        did_dict[did] = passage

    queries = []
    for k, v in qid_dict.items():
        queries.append((k, v))
    final_df = pd.DataFrame(queries)
    final_df.to_csv(data_folder + '/mb_2014_queries.tsv', sep='\t', index=False, header=False)

    passages = []
    for k, v in did_dict.items():
        passages.append((k, v))
    final_df = pd.DataFrame(passages)
    final_df.to_csv(data_folder + '/mb_2014_collections.tsv', sep='\t', index=False, header=False)

    print("Done!")


if __name__ == "__main__":
    csv_path = data_folder + '/trec_mb_2014_no_url.csv'
    build_mb_to_csv(csv_path)
    gen_input_from_csv_to_pkl(csv_path)

