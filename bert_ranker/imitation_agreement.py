import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import numpy as np
import collections
import math

data_folder = prodir + '/data/msmarco_passage'
runs_folder = prodir + '/bert_ranker/results/runs/'

runs_folder = curdir + '/results/runs'

runs_folder = curdir + '/results/runs'

bert_large_run = data_folder + '/runs_from_public/BERT_Large_dev_run.tsv'
minilm_l12_v2_dl_runs = runs_folder + '/runs.ms-marco-MiniLM-L-12-v2.public.bert.msmarco.dl2019.csv'

simulator_dl_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Tue_Nov_2.dl2019.csv'
simulator_imitation_ms_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Thu_Oct_28.eval_full_dev1000_imitation.csv'
simulator_imitation_dl_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Thu_Nov_4.dl2019_imitation.csv'
simulator_imitation_minilm_further_dl_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Thu_Nov_11.dl2019_imitation_miniLM_further_train.csv'
simulator_imitation_minilm_straight_dl_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Thu_Nov_11.dl2019_imitation_miniLM_straight.csv'
simulator_imitation_straight_bert_large_dl_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Tue_Nov_9.dl2019_imitation_straight_bert_large.csv'
simulator_imitation_bert_large_further_dl_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Wed_Nov_10.dl2019_imitation_bert_large_344_further_train.csv'
simulator_imitation_pseudo_same_dl_run = runs_folder + '/runs.bert-base-uncased.pairwise.triples.Wed_Nov_10.dl2019_same_pseudo.csv'

dl_imitate_further_MinilM_top25_last_4_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_miniLM_further.top_25_last_4.csv'
dl_imitate_straight_mini_top20_last10_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_straight.top_20_last_10.csv'
dl_imitate_straight_mini_top15_last19_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_straight.top_15_last_19.csv'
dl_imitate_further_mini_top15_last19_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_further.top_15_last_19.csv'
dl_imitate_straight_mini_top25_last28_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_straight.top_25_last_28.csv'
dl_imitate_straight_mini_top20_last40_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_straight.top_20_last_40.csv'
dl_imitate_further_mini_top25_last28_runs = runs_folder +'/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_further.top_25_last_28.csv'
dl_imitate_straight_mini_top15_last59_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_straight.top_15_last_59.csv'
dl_imitate_straight_mini_top20_last_40_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_straight.top_20_last_40.csv'
dl_imitate_straight_mini_top25_last_4_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_straight.top_25_last_4.csv'
dl_imitate_further_mini_top25_last_4_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_further.top_25_last_4.csv'
dl_imitate_further_mini_top20_last10_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_further.top_20_last_10.csv'
dl_imitate_further_mini_top20_last10_64_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_further.top_20_last_10_64.csv'
dl_imitate_further_mini_top20_last10_512_runs = runs_folder + '/runs.bert-base-uncased.pairwise.triples.dl2019_imitation_MiniLM_further.top_20_last_10_512.csv'


def rbo_score(l1, l2, p):
    if not l1 or not l2:
        return 0
    s1 = set()
    s2 = set()
    max_depth = len(l1)
    score = 0.0
    for d in range(max_depth):
        s1.add(l1[d])
        s2.add(l2[d])
        avg_overlap = len(s1 & s2) / (d + 1)
        score += math.pow(p, d) * avg_overlap
    return (1 - p) * score


def load_runs(runs_path):
    runs = collections.defaultdict(list)
    with open(runs_path, 'r') as f:
        for line in f:
            qid, _, did, _, _, _ = line.strip().split('\t')
            runs[qid].append(did)
    return runs


def top_n_overlap(runs_path_1, runs_path_2, topn=10):
    runs1 = load_runs(runs_path_1)
    runs2 = load_runs(runs_path_2)

    sim_ratio_list = []
    for qid, dids in runs1.items():
        target_dids = dids[:topn]
        another_dids = runs2[qid][:topn]
        tmp_sim_cnt = 0
        tmp_cnt = 0
        for did in target_dids:
            if did in another_dids:
                tmp_sim_cnt += 1
            tmp_cnt += 1
        sim_ratio_list.append(tmp_sim_cnt / (tmp_cnt + 0.0))
    print("Top@{} functional similarity: {}".format(topn, np.mean(sim_ratio_list)))


def avg_rbo(runs_path_1, runs_path_2, topn=10, p=0.9):
    runs1 = load_runs(runs_path_1)
    runs2 = load_runs(runs_path_2)

    rbo_list = []
    for qid, dids in runs1.items():
        target_dids = dids[:topn]
        another_dids = runs2[qid][:topn]

        tmp_rbo = rbo_score(target_dids, another_dids, p=p)
        rbo_list.append(tmp_rbo)

    print("Top@{} functional similarity: \t{}".format(topn, np.mean(rbo_list)))


if __name__ == "__main__":
    victim = minilm_l12_v2_dl_runs
    simulator = dl_imitate_further_mini_top20_last10_runs
    p = 0.7
    top_n_overlap(simulator, victim, topn=10)
    avg_rbo(victim, simulator, topn=1000, p=p)


