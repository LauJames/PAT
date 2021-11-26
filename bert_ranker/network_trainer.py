import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import metrics
import bert_ranker_utils

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import torch.optim as optim
import pickle as pkl
from apex import amp
from bert_ranker.models import pairwise_bert
from sklearn.metrics import f1_score, confusion_matrix, classification_report


class Trainer(object):
    """
    Parent class for all ranking models.
    """

    def __init__(self, model, data_class, tokenizer, args, model_path, start_model_path=None, writer_train=None,
                 run_id='BERT', monitor_metric='ndcg_cut_10', validation_metric=None, logger=None):

        # ArgumentParser format
        self.random_seed = args.seed
        self.num_epochs = args.num_epochs
        self.num_training_instances = args.num_training_instances
        self.validate_every_epochs = args.validate_every_epochs
        self.validate_every_steps = args.validate_every_steps
        self.num_validation_batches = args.num_validation_batches
        self.batch_size_train = args.train_batch_size
        self.batch_size_eval = args.val_batch_size
        self.sample_num = args.sample_data
        self.use_dev_triple = args.use_dev_triple
        self.pseudo_final = args.pseudo_final

        self.max_seq_len = args.max_seq_len
        self.lr = args.lr
        self.max_grad_norm = args.max_grad_norm
        self.accumulation_steps = args.accumulation_steps
        self.warmup_portion = args.warmup_portion

        self.transformer_model = args.transformer_model
        self.loss_function = args.loss_function
        self.smoothing = args.smoothing

        self.data_class = data_class
        self.tokenizer = tokenizer
        self.monitor_metric = monitor_metric
        self.validation_metric = validation_metric
        self.model_name = args.model_name

        self.num_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Device {}".format(self.device))
        print("Num GPU {}".format(self.num_gpu))

        self.model = model.to(self.device)

        self.writer_train = writer_train
        self.logger = logger

        self.model_path = model_path
        self.output_dir = curdir + '/results'
        self.run_id = run_id

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level='O1')

        if self.num_gpu > 1:
            devices = [v for v in range(self.num_gpu)]
            self.model = nn.DataParallel(self.model, device_ids=devices)
        if start_model_path is not None:
            self.model = self.load_state(self.model, start_model_path, self.num_gpu)

    def load_state(self, model, model_path, gpu_num):
        if gpu_num > 1:
            model.module.load_state_dict(torch.load(model_path))
        else:
            model.load_state_dict(torch.load(model_path))
        print("{} loaded!".format(model_path))
        self.logger.info("{} loaded!".format(model_path))
        return model

    def load_model(self):
        if self.model_name == 'pairwise-BERT-ranker':
            model = pairwise_bert.BertForPairwiseLearning.from_pretrained(self.transformer_model,
                                                                          loss_function=self.loss_function,
                                                                          smoothing=self.smoothing)
        else:
            raise ValueError("{} model class is not exist!".format(self.model_name))

        model.to(self.device)
        model = amp.initialize(model, opt_level='O1')
        if self.num_gpu > 1:
            devices = [v for v in range(self.num_gpu)]
            model = nn.DataParallel(model, device_ids=devices)
            model.module.load_state_dict(torch.load(self.model_path))
        else:
            model.load_state_dict(torch.load(self.model_path))
        return model

    def train_ranker(self, mode='train'):
        max_val = 0
        save_best = True
        global_step = 0

        data_generator = self.data_class.data_generator_pairwise_triple

        for epoch in range(self.num_epochs):
            print("Train the models for epoch {} with batch size {}\n".format(epoch, self.batch_size_train))
            self.logger.info("Train the models for epoch {} with batch size {}".format(epoch, self.batch_size_train))
            self.model.train()

            epoch_instance = 0
            epoch_step = 0
            early_stop_cnt = 0

            for batch_encoding_pos, batch_encoding_neg, tmp_labels in data_generator(mode=mode,
                                                                                     epoch_sample_num=self.sample_num,
                                                                                     random_seed=self.random_seed + epoch,
                                                                                     batch_size=self.batch_size_train,
                                                                                     max_seq_len=self.max_seq_len):

                pos_input_ids = batch_encoding_pos['input_ids'].to(self.device)
                pos_token_type_ids = batch_encoding_pos['token_type_ids'].to(self.device)
                pos_attention_mask = batch_encoding_pos['attention_mask'].to(self.device)
                neg_input_ids = batch_encoding_neg['input_ids'].to(self.device)
                neg_token_type_ids = batch_encoding_neg['token_type_ids'].to(self.device)
                neg_attention_mask = batch_encoding_neg['attention_mask'].to(self.device)
                true_labels = tmp_labels.to(self.device)
                outputs = self.model(
                    input_ids_pos=pos_input_ids,
                    attention_mask_pos=pos_attention_mask,
                    token_type_ids_pos=pos_token_type_ids,
                    input_ids_neg=neg_input_ids,
                    attention_mask_neg=neg_attention_mask,
                    token_type_ids_neg=neg_token_type_ids,
                    labels=true_labels
                )
                loss = outputs[0]

                if self.num_gpu > 1:
                    loss = loss.mean()

                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()

                self.writer_train.add_scalar('loss', loss, global_step)

                nn.utils.clip_grad_norm_(self.model.parameters(),
                                         self.max_grad_norm)
                # nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.max_grad_norm)

                self.optimizer.step()
                self.optimizer.zero_grad()
                global_step += 1
                epoch_step += 1
                epoch_instance += pos_input_ids.shape[0]

                if self.num_training_instances != -1 and epoch_instance >= self.num_training_instances:
                    print("Reached num_training_instances of {} ({} batches). Early stopping.\n".format(
                        self.num_training_instances, epoch_step))
                    self.logger.info("Reached num_training_instances of {} ({} batches). Early stopping.".format(
                        self.num_training_instances, epoch_step))
                    break

                # logging for steps
                is_validation_step = (self.validate_every_steps > 0 and global_step % self.validate_every_steps == 0)
                if is_validation_step:
                    with torch.no_grad():
                        res_array = self.dev_pairwise(mode='dev', model=self.model)
                        if self.use_dev_triple and 'pseudo' not in mode:
                            mac_f1 = self.dev_triple_pairwise(mode='dev_triple', model=self.model)
                        else:
                            mac_f1 = 0

                        if res_array[self.monitor_metric] + mac_f1 > max_val and save_best:
                            max_val = res_array[self.monitor_metric] + mac_f1
                            if self.num_gpu > 1:
                                torch.save(self.model.module.state_dict(), self.model_path)
                            else:
                                torch.save(self.model.state_dict(), self.model_path)
                            self.logger.info("Saved !")
                            print("\nSaved !")
                            early_stop_cnt = 0
                        else:
                            early_stop_cnt += 1

                if early_stop_cnt > 3:
                    print("early stop this epoch")
                    self.logger.info("early stop this epoch")
                    break

            is_validation_epoch = (self.validate_every_epochs > 0 and (epoch % self.validate_every_epochs == 0))
            if is_validation_epoch:
                with torch.no_grad():
                    res_array = self.dev_pairwise(mode='dev', model=self.model)
                    if self.use_dev_triple and 'pseudo' not in mode:
                        mac_f1 = self.dev_triple_pairwise(mode='dev_triple', model=self.model)
                    else:
                        mac_f1 = 0

                    if res_array[self.monitor_metric] + mac_f1 > max_val and save_best:
                        max_val = res_array[self.monitor_metric] + mac_f1
                        if self.num_gpu > 1:
                            torch.save(self.model.module.state_dict(), self.model_path)
                        else:
                            torch.save(self.model.state_dict(), self.model_path)
                        self.logger.info("Saved !")
                        print("\nSaved !")

        with torch.no_grad():
            model = self.load_model()
            self.dev_pairwise(mode='test', model=model)
            if self.pseudo_final and 'pseudo' not in mode:
                self.pseudo_pairwise(mode='test', model=model)

    def dev_pairwise(self, mode='dev', model=None):
        model.eval()

        all_logits = []
        all_labels = []
        all_softmax_logits = []
        all_qids = []
        all_pids = []
        cnt = 0

        for batch_encoding, tmp_labels, tmp_qids, tmp_pids in self.data_class.data_generator_mono_dev(mode=mode,
                                                                                                      batch_size=self.batch_size_eval,
                                                                                                      max_seq_len=self.max_seq_len):
            cnt += 1

            pos_input_ids = batch_encoding['input_ids'].to(self.device)
            pos_token_type_ids = batch_encoding['token_type_ids'].to(self.device)
            pos_attention_mask = batch_encoding['attention_mask'].to(self.device)
            neg_input_ids = batch_encoding['input_ids'].to(self.device)
            neg_token_type_ids = batch_encoding['token_type_ids'].to(self.device)
            neg_attention_mask = batch_encoding['attention_mask'].to(self.device)
            true_labels = tmp_labels.to(self.device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
            )
            logits = outputs[0]

            all_labels += true_labels.int().tolist()
            all_logits += logits[:, 1].tolist()
            all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()
            all_qids += tmp_qids
            all_pids += tmp_pids

            if self.num_validation_batches != -1 and cnt > self.num_validation_batches and mode == 'dev':
                break

        # accumulates per query
        all_labels, _ = bert_ranker_utils.accumulate_list_by_qid(all_labels, all_qids)
        all_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_logits, all_qids)
        all_softmax_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_softmax_logits, all_qids)
        all_pids, all_qids = bert_ranker_utils.accumulate_list_by_qid(all_pids, all_qids)

        res = metrics.evaluate_and_aggregate(all_logits, all_labels, ['ndcg_cut_10', 'map', 'recip_rank', 'MRR@10'])
        for metric, v in res.items():
            print("\n{} {} : {:3f}".format(mode, metric, v))
            self.logger.info("{} {} : {:3f}".format(mode, metric, v))

        if mode not in ['dev', 'test']:
            # Saving predictions and labels to a file
            # For MSMARCO
            top_k = 100
            run_list = []
            for probs, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(probs).argsort()[::-1]
                top_qids = np.array(qids)[sorted_idx[:top_k]]
                top_pids = np.array(pids)[sorted_idx[:top_k]]
                for rank, (t_qid, t_pid) in enumerate(zip(top_qids, top_pids)):
                    run_list.append((t_qid, t_pid, rank + 1))
            run_df = pd.DataFrame(run_list, columns=["qid", "pid", "rank"])
            run_df.to_csv(self.output_dir + "/run." + self.run_id + '.' + mode + ".csv", sep='\t', index=False,
                          header=False)

            # For TREC eval
            runs_list = []
            for scores, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(scores).argsort()[::-1]
                sorted_scores = np.array(scores)[sorted_idx]
                sorted_qids = np.array(qids)[sorted_idx]
                sorted_pids = np.array(pids)[sorted_idx]
                for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                    runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Pair'))
            runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
            runs_df.to_csv(self.output_dir + '/runs/runs.' + self.run_id + '.' + mode + '.csv', sep='\t', index=False,
                           header=False)
        return res

    def dev_triple_pairwise(self, mode='dev_triple', model=None):
        model.eval()
        all_pre_labels = []
        all_labels = []

        for batch_encoding_pos, batch_encoding_neg, tmp_labels in self.data_class.data_generator_pairwise_dev_triple(
                mode=mode,
                batch_size=self.batch_size_eval,
                max_seq_len=self.max_seq_len):
            pos_input_ids = batch_encoding_pos['input_ids'].to(self.device)
            pos_token_type_ids = batch_encoding_pos['token_type_ids'].to(self.device)
            pos_attention_mask = batch_encoding_pos['attention_mask'].to(self.device)
            neg_input_ids = batch_encoding_neg['input_ids'].to(self.device)
            neg_token_type_ids = batch_encoding_neg['token_type_ids'].to(self.device)
            neg_attention_mask = batch_encoding_neg['attention_mask'].to(self.device)
            true_labels = tmp_labels.to(self.device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
                labels=true_labels
            )
            logits_dif = outputs[-1]
            pre_labels = torch.argmax(logits_dif, dim=-1).tolist()
            all_pre_labels += pre_labels
            all_labels += tmp_labels.tolist()

        macro_f1 = f1_score(all_labels, all_pre_labels, average='macro')
        print(classification_report(all_labels, all_pre_labels))
        self.logger.info(classification_report(all_labels, all_pre_labels))
        self.logger.info("Pairwise triples {}: {}".format(mode, macro_f1))
        return macro_f1

    def pseudo_pairwise(self, model=None, mode='pseudo'):
        model.eval()

        all_pseudo_labels = []
        all_examples = []
        tru_laebls = []
        cnt = 0

        for batch_encoding_pos, batch_encoding_neg, tmp_labels, tmp_examples in self.data_class.data_generator_pseudo_pairwise(
                mode=mode,
                epoch_sample_num=self.sample_num,
                batch_size=self.batch_size_eval,
                max_seq_len=self.max_seq_len):
            pos_input_ids = batch_encoding_pos['input_ids'].to(self.device)
            pos_token_type_ids = batch_encoding_pos['token_type_ids'].to(self.device)
            pos_attention_mask = batch_encoding_pos['attention_mask'].to(self.device)
            neg_input_ids = batch_encoding_neg['input_ids'].to(self.device)
            neg_token_type_ids = batch_encoding_neg['token_type_ids'].to(self.device)
            neg_attention_mask = batch_encoding_neg['attention_mask'].to(self.device)
            true_labels = tmp_labels.to(self.device)
            outputs = model(
                input_ids_pos=pos_input_ids,
                attention_mask_pos=pos_attention_mask,
                token_type_ids_pos=pos_token_type_ids,
                input_ids_neg=neg_input_ids,
                attention_mask_neg=neg_attention_mask,
                token_type_ids_neg=neg_token_type_ids,
                labels=true_labels
            )
            logits_dif = outputs[-1]
            pre_labels = torch.argmax(logits_dif, dim=-1).tolist()
            all_pseudo_labels += pre_labels
            all_examples += tmp_examples
            tru_laebls += tmp_labels.tolist()

            cnt += 1
            if cnt % 1000 == 0:
                print(confusion_matrix(tru_laebls, all_pseudo_labels))
                print(classification_report(tru_laebls, all_pseudo_labels))
        if mode == 'pseudo_same':
            pkl_path = self.data_class.victim_pseudo_same_triples
        else:
            pkl_path = self.data_class.victim_pseudo_1x_another_triples
        with open(pkl_path, 'wb') as f:
            pkl.dump(all_examples, f)
            pkl.dump(all_pseudo_labels, f)
        print("Total of {} instances are pseudo labeled.".format(len(all_pseudo_labels)))


