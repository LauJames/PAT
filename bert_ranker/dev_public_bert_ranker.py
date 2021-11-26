import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)


import time
import pandas as pd
import numpy as np
from transformers import BertTokenizerFast, BertConfig, BertModel, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoModel
from bert_ranker.dataloader.dataset import MSMARCO_PR_Pair_Dataset
import bert_ranker_utils
import metrics
import torch
import torch.nn as nn
import argparse
from apex import amp
from models.bert_cat import BERT_Cat


def main():
    parser = argparse.ArgumentParser('Pytorch')
    # Input and output configs
    parser.add_argument("--output_dir", default=curdir + '/results', type=str,
                        help="the folder to output predictions")
    parser.add_argument("--mode", default='mb2014', type=str,
                        help="eval_full_dev1000/eval_pseudo_full_dev1000/dl2019/eval_subsmall_dev")

    # Training procedure
    parser.add_argument("--seed", default=42, type=str,
                        help="random seed")

    parser.add_argument("--val_batch_size", default=1024, type=int,
                        help="Validation and test batch size.")

    # Model hyperparameters
    parser.add_argument("--transformer_model", default="bert-base-uncased", type=str, required=False,
                        help="Bert model to use (default = bert-base-cased).")
    parser.add_argument("--max_seq_len", default=256, type=int, required=False,
                        help="Maximum sequence length for the inputs.")

    args = parser.parse_args()
    args.model_name = 'pub-ranker'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    now_time = '_'.join(time.asctime(time.localtime(time.time())).split()[:3])
    args.run_id = args.transformer_model + '.public.bert.msmarco.' + now_time
    output_dir = curdir + '/results'

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.transformer_model == 'bert-large-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-large-msmarco")
    elif args.transformer_model == 'bert-base-uncased':
        tokenizer = BertTokenizerFast.from_pretrained(args.transformer_model)
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-bert-base-uncased-msmarco")
    elif args.transformer_model == 'pt-tinybert-msmarco':
        tokenizer = AutoTokenizer.from_pretrained("nboost/pt-tinybert-msmarco")
        model = AutoModelForSequenceClassification.from_pretrained("nboost/pt-tinybert-msmarco")
    elif args.transformer_model == 'distilbert-cat-margin_mse-T2-msmarco':
        tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased")  # honestly not sure if that is the best way to go, but it works :)
        model = BERT_Cat.from_pretrained("sebastian-hofstaetter/distilbert-cat-margin_mse-T2-msmarco")
    elif args.transformer_model == 'condenser':
        tokenizer = AutoTokenizer.from_pretrained("Luyu/condenser")
        model = AutoModel.from_pretrained('Luyu/condenser')
    elif args.transformer_model == 'ms-marco-MiniLM-L-12-v2':
        tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
        model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")

    data_obj = MSMARCO_PR_Pair_Dataset(tokenizer=tokenizer)

    model.to(device)
    model = amp.initialize(model, opt_level='O1')
    num_gpu = torch.cuda.device_count()
    if num_gpu > 1:
        devices = [v for v in range(num_gpu)]
        model = nn.DataParallel(model, device_ids=devices)

    with torch.no_grad():
        model.eval()

        all_logits = []
        all_flat_labels = []
        all_softmax_logits = []
        all_qids = []
        all_pids = []
        cnt = 0

        for batch_encoding, tmp_labels, tmp_qids, tmp_pids in data_obj.data_generator_mono_dev(mode=args.mode,
                                                                                               batch_size=args.val_batch_size,
                                                                                               max_seq_len=args.max_seq_len):
            cnt += 1
            if args.transformer_model == 'distilbert-cat-margin_mse-T2-msmarco':
                input_ids = batch_encoding['input_ids'].to(device)
                attention_mask = batch_encoding['attention_mask'].to(device)
                scores = model(input_ids=input_ids, attention_mask=attention_mask).squeeze()

                all_logits += scores.tolist()
                all_softmax_logits += scores.tolist()
            elif args.transformer_model == 'ms-marco-MiniLM-L-12-v2':
                outputs = model(**batch_encoding)

                scores = outputs.logits.squeeze()
                all_logits += scores.tolist()
                all_softmax_logits += scores.tolist()
            elif args.transformer_model == 'condenser':
                outputs = model(**batch_encoding)
                scores = outputs.logits[:, 1]
                all_logits += scores.tolist()
                all_softmax_logits += scores.tolist()
            else:
                input_ids = batch_encoding['input_ids'].to(device)
                token_type_ids = batch_encoding['token_type_ids'].to(device)
                attention_mask = batch_encoding['attention_mask'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids,
                )
                logits = outputs[0]

                all_logits += logits[:, 1].tolist()
                all_softmax_logits += torch.softmax(logits, dim=1)[:, 1].tolist()

            # for ms-marco-MiniLM-L-12-v2
            all_flat_labels += tmp_labels
            all_qids += tmp_qids
            all_pids += tmp_pids

        # accumulates per query
        all_labels, _ = bert_ranker_utils.accumulate_list_by_qid(all_flat_labels, all_qids)
        all_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_logits, all_qids)
        all_softmax_logits, _ = bert_ranker_utils.accumulate_list_by_qid(all_softmax_logits, all_qids)
        all_pids, all_qids = bert_ranker_utils.accumulate_list_by_qid(all_pids, all_qids)

        res = metrics.evaluate_and_aggregate(all_logits, all_labels, ['ndcg_cut_10', 'map', 'recip_rank', 'MRR@10'])
        for metric, v in res.items():
            print("\n{} {} : {:3f}".format(args.mode, metric, v))

        validation_metric = ['MAP', 'RPrec', 'MRR', 'MRR@10', 'NDCG', 'NDCG@10']
        all_metrics = np.zeros(len(validation_metric))
        query_cnt = 0
        for labels, logits, probs in zip(all_labels, all_logits, all_softmax_logits):
            #
            gt = set(list(np.where(np.array(labels) > 0)[0]))
            pred_docs = np.array(probs).argsort()[::-1]

            all_metrics += metrics.metrics(gt, pred_docs, validation_metric)
            query_cnt += 1
        all_metrics /= query_cnt
        print("\n" + "\t".join(validation_metric))
        print("\t".join(["{:4f}".format(x) for x in all_metrics]))

        if args.mode not in ['dev', 'test']:
            # Saving predictions and labels to a file
            # For MSMARCO
            top_k = 100
            run_list = []
            for probs, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(probs).argsort()[::-1]
                # top_probs = np.array(probs)[sorted_idx[:top_k]]
                top_qids = np.array(qids)[sorted_idx[:top_k]]
                top_pids = np.array(pids)[sorted_idx[:top_k]]
                for rank, (t_qid, t_pid) in enumerate(zip(top_qids, top_pids)):
                    run_list.append((t_qid, t_pid, rank + 1))
            run_df = pd.DataFrame(run_list, columns=["qid", "pid", "rank"])
            run_df.to_csv(output_dir + "/run." + args.run_id + '.' + args.mode + ".csv", sep='\t', index=False,
                          header=False)

            # For TREC eval
            runs_list = []
            for scores, qids, pids in zip(all_logits, all_qids, all_pids):
                sorted_idx = np.array(scores).argsort()[::-1]
                sorted_scores = np.array(scores)[sorted_idx]
                sorted_qids = np.array(qids)[sorted_idx]
                sorted_pids = np.array(pids)[sorted_idx]
                for rank, (t_qid, t_pid, t_score) in enumerate(zip(sorted_qids, sorted_pids, sorted_scores)):
                    runs_list.append((t_qid, 'Q0', t_pid, rank + 1, t_score, 'BERT-Point'))
            runs_df = pd.DataFrame(runs_list, columns=["qid", "Q0", "pid", "rank", "score", "runid"])
            runs_df.to_csv(output_dir + '/runs/runs.' + args.run_id + '.' + args.mode + '.csv', sep='\t', index=False,
                           header=False)


if __name__ == "__main__":
    main()
