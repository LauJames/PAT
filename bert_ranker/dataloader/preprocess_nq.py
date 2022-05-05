import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))

from tqdm import tqdm
import pandas as pd
import collections
import gzip
import json
import random

data_folder = prodir + '/data/nq'
triples_per_10_neg_csv_path = data_folder + '/train_triples_per_10_neg.csv'
BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])
random_seed = 42


def transform_nq_to_dfs():
    train_gz_path = data_folder + '/biencoder-nq-adv-hn-train.json.gz'
    with gzip.open(train_gz_path, 'r') as fin:
        data = fin.read()
        obj = json.loads(data.decode('utf-8'))
    return obj


def sample_triples(query, pos_list, neg_list, per_neg=10):
    if len(neg_list) == 0:
        return None
    else:
        triples_list = []
        for tmp_pos in pos_list:
            random.seed(random_seed)
            sampled_neg_list = random.choices(neg_list, k=per_neg)
            for tmp_neg in sampled_neg_list:
                triples_list.append((query, tmp_pos, tmp_neg))
        return triples_list


def get_triples(json_obj):
    triple_list = []
    for json_sample in tqdm(json_obj, desc='Processing:'):
        norm_query = json_sample["question"].replace("’", "'")

        positive_ctxs = json_sample["positive_ctxs"]
        ctxs = [ctx for ctx in positive_ctxs if "score" in ctx]
        if ctxs:
            positive_ctxs = ctxs

        negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
        hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []

        for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
            if "title" not in ctx:
                ctx["title"] = None

        def create_passage(ctx):
            return ctx["title"] + ctx["text"]

        positive_passages = [create_passage(ctx) for ctx in positive_ctxs]
        negative_passages = [create_passage(ctx) for ctx in negative_ctxs]
        hard_negative_passages = [create_passage(ctx) for ctx in hard_negative_ctxs]

        tmp_triples_normal = sample_triples(norm_query, positive_passages, negative_passages)
        if tmp_triples_normal is not None:
            triple_list.extend(tmp_triples_normal)
        tmp_triples_hard = sample_triples(norm_query, positive_passages, hard_negative_passages)
        if tmp_triples_hard is not None:
            triple_list.extend(tmp_triples_hard)

    print("Total triples: {}".format(len(triple_list)))

    final_df = pd.DataFrame(triple_list, columns=["query", "pos", "neg"])
    final_df.to_csv(triples_per_10_neg_csv_path, sep='\t', index=False, header=True)


if __name__ == '__main__':
    json_obj = transform_nq_to_dfs()
    get_triples(json_obj)

