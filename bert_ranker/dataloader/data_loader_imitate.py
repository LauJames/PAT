import os
import sys

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))
sys.path.insert(0, prodir)

import pandas as pd
import pickle as pkl
from tqdm import tqdm
import random
import torch
from transformers import BertTokenizerFast


class Imitation_Dataset(object):
    def __init__(self, tokenizer, random_seed=666):
        self.tokenizer = tokenizer
        self.random_seed = random_seed
        self.ms_folder_name = prodir + '/data/msmarco_passage'
        self.triples_from_runs_folder = self.ms_folder_name + '/triples_from_runs'
        self.triples_from_runs_bert_large = self.triples_from_runs_folder + '/bert_large_sampled_triples_text.top_20_last_10.csv'
        self.triples_from_runs_distilbert_cat = self.triples_from_runs_folder + '/distilbert_cat_sampled_triples_text.top_20_last_10.csv'
        self.triples_from_runs_minilm_l12_v2 = self.triples_from_runs_folder + '/minilm_l12_sampled_triples_text.top_20_last_10.csv'

        self.dev_sub_small_path = self.ms_folder_name + "/sampled_set/run_sub_small.dev.pkl"
        self.dev_top1000_full_path = self.ms_folder_name + "/sampled_set/run_bm25.dev.pkl"

        self.test_trec_dl_2019_path = prodir + '/data/trec_dl_2019/trec_dl2019_passage_test1000_full.pkl'
        self.test_trec_mb_2014_path = prodir + '/data/trec_mb_2014/trec_mb_2014.pkl'

    def _load_from_triples_to_pairwise(self):
        pkl_path = self.ms_folder_name + '/triples_from_runs/bert_large_sampled_triples_text.top_20_last_10.pkl'
        # pkl_path = self.ms_folder_name + '/triples_from_runs/distilbert_cat_sampled_triples_text.top_20_last_10.pkl'
        # pkl_path = self.ms_folder_name + '/triples_from_runs/minilm_l12_sampled_triples_text.top_20_last_10.pkl'
        # pkl_path = self.ms_folder_name + '/triples_from_runs/bert_large_sampled_triples_text.top_20_last_10.pkl'
        print("Load data from {}".format(pkl_path))
        if not os.path.exists(pkl_path):
            print("{} not exists.\n Load pairwise sample from the {} ...".format(pkl_path,
                                                                                 self.triples_from_runs_bert_large))
            train_df = pd.read_csv(self.triples_from_runs_bert_large, sep='\t', names=["query", "pos", "neg"])
            train_instances = []
            train_labels = []
            for _, rows in tqdm(train_df.iterrows(), total=len(train_df)):
                # query, reL_doc, ns_doc
                train_instances.append((rows["query"], rows["pos"], rows["neg"]))
                train_labels.append(1)
                # add reverse
                train_instances.append((rows["query"], rows["neg"], rows["pos"],))
                train_labels.append(0)

            with open(pkl_path, 'wb') as f:
                pkl.dump(train_instances, f)
                pkl.dump(train_labels, f)
        else:
            with open(pkl_path, 'rb') as f:
                train_instances = pkl.load(f)
                train_labels = pkl.load(f)
        print("Total of {} instances are loaded.".format(len(train_labels)))
        return train_instances, train_labels

    def data_generator_pairwise_triple(self, mode='train', epoch_sample_num=None, random_seed=666, batch_size=32,
                                       max_seq_len=128):
        if mode == 'train':
            examples, labels = self._load_from_triples_to_pairwise()
        else:
            raise ValueError("Error mode: {}!".format(mode))

        total_cnt = len(labels)

        list_pack = list(zip(examples, labels))
        random.seed(random_seed)
        random.shuffle(list_pack)
        examples[:], labels[:] = zip(*list_pack)

        for i in tqdm(range(0, total_cnt, batch_size), desc='Processing:'):
            tmp_examples = examples[i: i + batch_size]
            tmp_labels = torch.tensor(labels[i: i + batch_size], dtype=torch.long)

            batch_encoding_pos = self.tokenizer([(e[0], e[1]) for e in tmp_examples],
                                                max_length=max_seq_len, padding="max_length", truncation=True,
                                                return_tensors='pt')
            batch_encoding_neg = self.tokenizer([(e[0], e[2]) for e in tmp_examples],
                                                max_length=max_seq_len, padding="max_length", truncation=True,
                                                return_tensors='pt')
            yield batch_encoding_pos, batch_encoding_neg, tmp_labels

    def data_generator_mono_dev(self, batch_size=32, max_seq_len=128, mode='dev'):

        if mode in ['dev', 'eval_subsmall_dev', 'test']:
            pkl_path = self.dev_sub_small_path
        elif mode in ['eval_full_dev1000']:
            pkl_path = self.dev_top1000_full_path
        elif 'dl2019' in mode:
            pkl_path = self.test_trec_dl_2019_path
        elif 'mb2014' in mode:
            pkl_path = self.test_trec_mb_2014_path
        else:
            raise ValueError("Error mode !!!")
        if os.path.exists(pkl_path):
            with open(pkl_path, 'rb') as f:
                print("Loading instances from {}".format(pkl_path))
                examples = pkl.load(f)
                labels = pkl.load(f)
                qids = pkl.load(f)
                pids = pkl.load(f)
        else:
            raise ValueError("{} not exists".format(pkl_path))

        for i in tqdm(range(0, len(labels), batch_size), desc='Processing:'):
            tmp_examples = examples[i: i + batch_size]
            tmp_qids = qids[i: i + batch_size]
            tmp_pids = pids[i: i + batch_size]
            tmp_labels = torch.tensor(labels[i: i + batch_size], dtype=torch.long)

            batch_encoding_pos = self.tokenizer([(e[0], e[1]) for e in tmp_examples],
                                                max_length=max_seq_len, padding="max_length", truncation=True,
                                                return_tensors='pt')
            yield batch_encoding_pos, tmp_labels, tmp_qids, tmp_pids


if __name__ == "__main__":
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    data_class = Imitation_Dataset(tokenizer=tokenizer)
    cnt = 0
    for ins_p, ins_n, labels in data_class.data_generator_pairwise_triple():
        cnt += 1
        if cnt > 2:
            break