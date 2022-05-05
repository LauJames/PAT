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


class MSMARCO_PR_Pair_Dataset(object):
    def __init__(self, tokenizer, random_seed=666):
        self.tokenizer = tokenizer
        self.random_seed = random_seed

        self.dataset_folder_name = prodir + '/data/msmarco_passage'

        self.train_sample_file_path = None
        self.train_triples_small_path = self.dataset_folder_name + "/triples.train.small.tsv"
        self.nq_train_triples_per_10_path = prodir + '/data/nq/train_triples_per_10_neg.csv'
        self.dev_sub_small_path = self.dataset_folder_name + "/sampled_set/run_sub_small.dev.pkl"
        self.dev_top1000_full_path = self.dataset_folder_name + "/sampled_set/run_bm25.dev.pkl"
        self.dev_top1000_small_path = self.dataset_folder_name + "/sampled_set/top1000.small.dev.pkl"
        self.dev_triples_path = self.dataset_folder_name + "/sampled_set/triples.dev.pairwise.25600.pkl"

        self.victim_pseudo_same_triples = self.dataset_folder_name + "/pseudo_set/pairwise.pseudo.same.256000.pkl"
        self.test_trec_dl_2019_path = prodir + '/data/trec_dl_2019/trec_dl2019_passage_test1000_full.pkl'
        self.test_trec_dl_2019_200q_path = prodir + '/data/trec_dl_2019/trec_dl2019_passage_200q_test1000_full.pkl'
        self.test_trec_mb_2014_path = prodir + '/data/trec_mb_2014/trec_mb_2014.pkl'

    def _load_from_triples_to_pointwise(self, sample_num):
        pkl_path = self.dataset_folder_name + "/sampled_set/pointwise.train.small.{}.pkl".format(sample_num)
        if not os.path.exists(pkl_path):
            print("{} not exists.\n Sampling ## point-wise ## sample from the {} ...".format(pkl_path,
                                                                                             self.train_triples_small_path))
            train_df = pd.read_csv(self.train_triples_small_path, sep='\t',
                                   nrows=sample_num if sample_num != -1 else None, names=["query", "pos", "neg"])
            train_instances = []
            train_labels = []
            for _, rows in tqdm(train_df.iterrows(), total=len(train_df)):
                train_instances.append((rows["query"], rows["pos"]))
                train_labels.append(1)
                train_instances.append((rows["query"], rows["neg"]))
                train_labels.append(0)
            with open(pkl_path, 'wb') as f:
                pkl.dump(train_instances, f)
                pkl.dump(train_labels, f)
        else:
            with open(pkl_path, 'rb') as f:
                train_instances = pkl.load(f)
                train_labels = pkl.load(f)
        print("Total of {} instances are loaded from: {}".format(len(train_labels), pkl_path))
        return train_instances, train_labels

    def _load_from_triples_to_pairwise(self, sample_num):
        pkl_path = self.dataset_folder_name + "/sampled_set/triples.train.small.{}.pkl".format(sample_num)
        print("Load data from {}".format(pkl_path))
        if not os.path.exists(pkl_path):
            print("{} not exists.\n Sampling ## pair-wise ## sample from the {} ...".format(pkl_path,
                                                                                            self.train_triples_small_path))
            train_df = pd.read_csv(self.train_triples_small_path, sep='\t',
                                   nrows=sample_num if sample_num != -1 else None, names=["query", "pos", "neg"])
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

    def _load_nq_from_triples_to_pairwise(self, sample_num):
        pkl_path = prodir + '/data/nq/sampled_set/nq.triples.train.{}.pkl'.format(sample_num)
        print("Load data from {}".format(pkl_path))
        if not os.path.exists(pkl_path):
            print("{} not exists.\n Sampling ## pairwise ## sample from the {} ...".format(pkl_path,
                                                                                           self.nq_train_triples_per_10_path))
            train_df = pd.read_csv(self.nq_train_triples_per_10_path, sep='\t',
                                   nrows=sample_num if sample_num != -1 else None, names=["query", "pos", "neg"])
            train_instances = []
            train_labels = []
            for _, rows in tqdm(train_df.iterrows(), total=len(train_df)):
                # query, reL_passage, ns_passage
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

        tmp = list(zip(train_instances, train_labels))
        random.seed(self.random_seed)
        random.shuffle(tmp)
        train_instances[:], train_labels[:] = zip(*tmp)

        return train_instances, train_labels

    def data_generator_pseudo_pairwise(self, mode='pseudo_pair', epoch_sample_num=None, batch_size=32, max_seq_len=128):
        if mode == 'pseudo_same':
            examples, labels = self._load_from_triples_to_pairwise(sample_num=epoch_sample_num)
        else:
            raise ValueError('Please check the mode you enter!!!')

        total_cnt = len(examples)

        for i in tqdm(range(0, total_cnt, batch_size), desc='Processing:'):
            tmp_examples = examples[i: i + batch_size]
            tmp_labels = torch.tensor(labels[i: i + batch_size], dtype=torch.long)

            batch_encoding_pos = self.tokenizer([(e[0], e[1]) for e in tmp_examples],
                                                max_length=max_seq_len, padding="max_length", truncation=True,
                                                return_tensors='pt')
            batch_encoding_neg = self.tokenizer([(e[0], e[2]) for e in tmp_examples],
                                                max_length=max_seq_len, padding="max_length", truncation=True,
                                                return_tensors='pt')
            yield batch_encoding_pos, batch_encoding_neg, tmp_labels, tmp_examples

    def data_generator_pointwise_triple(self, mode='train', epoch_sample_num=None, random_seed=666, batch_size=32,
                                        max_seq_len=128):
        if mode == 'train':
            examples, labels = self._load_from_triples_to_pointwise(sample_num=epoch_sample_num)
        elif mode == 'train_pseudo':
            pkl_path = self.dataset_folder_name + "/pseudo_set/pointwise.pseudo.small.{}.pkl".format(epoch_sample_num)
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    examples = pkl.load(f)
                    labels = pkl.load(f)
                print("Total of {} instances are loaded from: {}".format(len(labels), pkl_path))
            else:
                raise OSError("{} not exist!".format(pkl_path))
        else:
            raise ValueError("Error mode: {}!".format(mode))

        total_cnt = len(examples)

        list_pack = list(zip(examples, labels))
        random.seed(random_seed)
        random.shuffle(list_pack)
        examples[:], labels[:] = zip(*list_pack)

        for i in tqdm(range(0, total_cnt, batch_size), desc='Processing:'):
            tmp_examples = examples[i: i + batch_size]
            tmp_labels = torch.tensor(labels[i: i + batch_size], dtype=torch.long)

            batch_encoding = self.tokenizer([(e[0], e[1]) for e in tmp_examples],
                                            max_length=max_seq_len, padding="max_length", truncation=True,
                                            return_tensors='pt')

            yield batch_encoding, tmp_labels

    def data_generator_pairwise_triple(self, mode='train', epoch_sample_num=None, random_seed=666, batch_size=32,
                                       max_seq_len=128):
        if mode == 'train':
            examples, labels = self._load_from_triples_to_pairwise(sample_num=epoch_sample_num)
        elif mode == 'train_nq':
            examples, labels = self._load_nq_from_triples_to_pairwise(sample_num=epoch_sample_num)
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

        if mode in ['dev', 'eval_subsmall_dev', 'test', 'eval_pseudo_subsmall', 'eval_subsmall_imitation']:
            pkl_path = self.dev_sub_small_path
        elif mode in ['eval_full_dev1000', 'eval_full_dev1000_imitation',
                      'eval_pseudo_full_dev1000', 'eval_full_dev1000_same_pseudo']:
            pkl_path = self.dev_top1000_full_path
        elif mode in ['dl2019', 'dl2019_imitation', 'nq_dl2019']:
            pkl_path = self.test_trec_dl_2019_path
        elif mode == 'dl2019_200q':
            pkl_path = self.test_trec_dl_2019_200q_path
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

    def data_generator_pseudo_point(self, epoch_sample_num=None, batch_size=32, max_seq_len=128):
        examples, labels = self._load_from_triples_to_pointwise(sample_num=epoch_sample_num)
        total_cnt = len(examples)

        for i in tqdm(range(0, total_cnt, batch_size), desc='Processing:'):
            tmp_examples = examples[i: i + batch_size]
            tmp_labels = torch.tensor(labels[i: i + batch_size], dtype=torch.long)

            batch_encoding = self.tokenizer([(e[0], e[1]) for e in tmp_examples],
                                            max_length=max_seq_len, padding="max_length", truncation=True,
                                            return_tensors='pt')

            yield batch_encoding, tmp_labels, tmp_examples

    def data_generator_pairwise_dev_triple(self, batch_size=32, max_seq_len=128):
        if os.path.exists(self.dev_triples_path):
            with open(self.dev_triples_path, 'rb') as f:
                examples = pkl.load(f)
                labels = pkl.load(f)
            print("Total of {} instances are loaded from: {}".format(len(labels), self.dev_triples_path))

        total_cnt = len(examples)
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


if __name__ == "__main__":
    ns_sample = 'bm25'
    mode = 'train'
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    data_class = MSMARCO_PR_Pair_Dataset(tokenizer=tokenizer)




