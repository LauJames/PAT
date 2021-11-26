import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(os.path.dirname(curdir))

import pytrec_eval

METRICS = {'map',
           'recip_rank',
           'ndcg_cut'}

RECALL_AT_W_CAND = {
    'R_10@1',
    'R_10@2',
    'R_10@5',
    'R_2@1',
    'R_1@1000'
}

MRR_AT_K = {
    'MRR@10',
    # 'MRR@1000'
}


def recall_at_with_k_candidates(preds, labels, k, at):
    """
    Calculates recall with k candidates. labels list must be sorted by relevance.

    Args:
        preds: float list containing the predictions.
        labels: float list containing the relevance labels.
        k: number of candidates to consider.
        at: threshold to cut the list.

    Returns: float containing Recall_k@at
    """
    num_rel = len([l for l in labels if l >= 1])
    # 'removing' candidates (relevant has to be in first positions in labels)
    preds = preds[:k]
    labels = labels[:k]

    sorted_labels = [x for _, x in sorted(zip(preds, labels), reverse=True)]
    hits = len([l for l in sorted_labels[:at] if l >= 1])
    return hits/num_rel


def evaluate_models(results):
    """
    Calculate METRICS for each model in the results dict

    Args:
        results: dict containing one key for each model and inside them pred and label keys.
        For example:
             results = {
              'model_1': {
                 'preds': [[1,2],[1,2]],
                 'labels': [[1,2],[1,2]]
               }
            }.
    Returns: dict with the METRIC results per model and query.
    """

    for model in results.keys():
        preds = results[model]['preds']
        labels = results[model]['labels']
        run = {}
        qrel = {}
        for i, p in enumerate(preds):
            run['q{}'.format(i+1)] = {}
            qrel['q{}'.format(i+1)] = {}
            for j, _ in enumerate(range(len(p))):
                run['q{}'.format(i+1)]['d{}'.format(j+1)] = float(preds[i][j])
                qrel['q{}'.format(i + 1)]['d{}'.format(j + 1)
                                          ] = int(labels[i][j])
        evaluator = pytrec_eval.RelevanceEvaluator(qrel, METRICS)
        results[model]['eval'] = evaluator.evaluate(run)

        # for MRR at k
        for mrr_metric in MRR_AT_K:
            at_k = int(mrr_metric.split("@")[-1])
            for i, p in enumerate(preds):
                run['q{}'.format(i+1)] = {}
                qrel['q{}'.format(i+1)] = {}
                if len(preds[i]) < at_k:
                    topn = len(preds[i])
                else:
                    topn = at_k
                for j, _ in enumerate(range(topn)):
                    run['q{}'.format(i+1)]['d{}'.format(j+1)
                                           ] = float(preds[i][j])
                    qrel['q{}'.format(i + 1)]['d{}'.format(j + 1)
                                              ] = int(labels[i][j])
            evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'recip_rank'})
            tmp_mrr_res = evaluator.evaluate(run)

            # merge mrr into results dict
            for query in qrel.keys():
                results[model]['eval'][query][mrr_metric] = tmp_mrr_res[query]['recip_rank']

        # for query in qrel.keys():
        #     # for Recall@N
        #     preds = []
        #     labels = []
        #     for doc in run[query].keys():
        #         preds.append(run[query][doc])
        #         labels.append(qrel[query][doc])

        #     for recall_metric in RECALL_AT_W_CAND:
        #         cand = int(recall_metric.split("@")[0].split("R_")[1])
        #         at = int(recall_metric.split("@")[-1])
        #         results[model]['eval'][query][recall_metric] = recall_at_with_k_candidates(preds, labels, cand, at)
    return results


def evaluate(preds, labels):
    qrels = {}
    qrels['model'] = {}
    qrels['model']['preds'] = preds
    qrels['model']['labels'] = labels

    results = evaluate_models(qrels)
    return results


def evaluate_and_aggregate(preds, labels, metrics):
    """
    Calculate evaluation metrics for a pair of preds and labels.

    Aggregates the results only for the evaluation metrics in metrics arg.

    Args:
        preds: list of lists of floats with predictions for each query.
        labels: list of lists with of floats with relevance labels for each query.
        metrics: list of str with the metrics names to aggregate.

    Returns: dict with the METRIC results per model and query.
    """
    results = evaluate(preds, labels)

    agg_results = {}
    for metric in metrics:
        res = 0
        per_q_values = []
        for q in results['model']['eval'].keys():
            per_q_values.append(results['model']['eval'][q][metric])
            res += results['model']['eval'][q][metric]
        res /= len(results['model']['eval'].keys())
        agg_results[metric] = res

    return agg_results


import numpy as np


def average_precision(gt, pred):
    """
    Computes the average precision.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    gt: set
         A set of ground-truth elements (order doesn't matter)
    pred: list
          A list of predicted elements (order does matter)
    Returns
    -------
    score: double
        The average precision over the input lists
    """

    if not gt:
        return 0.0

    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(pred):
        if p in gt and p not in pred[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)

    return score / max(1.0, len(gt))


def NDCG(gt, pred, use_graded_scores=False):
    score = 0.0
    for rank, item in enumerate(pred):
        if item in gt:
            if use_graded_scores:
                grade = 1.0 / (gt.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)

    norm = 0.0
    for rank in range(len(gt)):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)


def NDCG_at_k(gt, pred, use_graded_scores=False, k=10):
    score = 0.0
    for rank, item in enumerate(pred[:k]):
        if item in gt:
            if use_graded_scores:
                grade = 1.0 / (gt.index(item) + 1)
            else:
                grade = 1.0
            score += grade / np.log2(rank + 2)

    norm = 0.0
    norm_len = min(len(gt), k)
    for rank in range(norm_len):
        if use_graded_scores:
            grade = 1.0 / (rank + 1)
        else:
            grade = 1.0
        norm += grade / np.log2(rank + 2)
    return score / max(0.3, norm)


def metrics(gt, pred, metrics_map):
    '''
    Returns a numpy array containing metrics specified by metrics_map.
    gt: ground-truth items
    pred: predicted items
    '''
    out = np.zeros((len(metrics_map),), np.float32)

    if ('MAP' in metrics_map):
        avg_precision = average_precision(gt=gt, pred=pred)
        out[metrics_map.index('MAP')] = avg_precision

    if ('RPrec' in metrics_map):
        intersec = len(gt & set(pred[:len(gt)]))
        out[metrics_map.index('RPrec')] = intersec / max(1., float(len(gt)))

    if 'MRR' in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred):
            if item in gt:
                score = 1.0 / (rank + 1.0)
            break
        out[metrics_map.index('MRR')] = score

    if 'MRR@10' in metrics_map:
        score = 0.0
        for rank, item in enumerate(pred[:10]):
            if item in gt:
                score = 1.0 / (rank + 1.0)
            break
        out[metrics_map.index('MRR@10')] = score

    if ('NDCG' in metrics_map):
        out[metrics_map.index('NDCG')] = NDCG(gt, pred)

    if ('NDCG@10' in metrics_map):
        out[metrics_map.index('NDCG@10')] = NDCG_at_k(gt, pred, k=10)

    return out
