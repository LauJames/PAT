# PAT
Core implementation of Paper: "Imitation Adversarial Attacks for Black-box Neural Ranking Models."

# Requirements
- Python 3.6 or higher
- Pytorch==1.10.0
- transformers==4.6.1
- sentence-transformers==2.1.0
- apex==0.1
- tqdm
- nltk
- pytrec_eval
- trectools

# Environment
- Tesla V100 32GB GPU x 8
- CUDA 11.2
- Memory 256GB

# Datasets
- [MSMARCO Passage Ranking](https://microsoft.github.io/msmarco/)
- [TREC Deep Learning Track 2019](https://microsoft.github.io/msmarco/TREC-Deep-Learning-2019)
- [TREC Microblog Track](https://github.com/jinfengr/neural-tweet-search)

## Model Imitation
- Data Processing
  Build Dev data into pickle file to speedup the evaluation.
  1. MSMARCO Passage Ranking
  `python ./bert_ranker/dataloader/preprocess_pr.py`
  2. TREC DL2019
  `python ./bert_ranker/dataloader/preprocess_trec_dl.py`
  3. TREC MB2014
  `python ./bert_ranker/dataloader/preprocess_mb.py`

- Train Pairwise-BERT Ranker from scratch
  `python ./bert_ranker/run_pairwise_ranker.py`

- Get runs file (TREC Format) from the publicly available ranking model.
  `python ./bert_ranker/dev_public_bert_ranker.py`

- Sample training data from runs file of public model to train imitation model.
 `python ./bert_ranker/dataloader/sample_runs.py`

- Train imitation model using sampled data.
 `python ./bert_ranker/run_imitation_pairwise.py`

- Evaluate the similarity between imitation model and victim model using runs file.
 `python imitation_agreement.py

- Evaluate ranking performance using runs file
  Note that the evaluation metrics during training and development are not consistent with the official evaluation method.
  We get the standard ranking performance by official trec tools, which are implemented in trec_eval_tools.py

## Text Ranking Attack Via PAT

- The data preprocessing is implemented in ./adv_ir/data_utils.py
  We need extract the query, query id, scores (imitation model), and target candidate passages from runs file.

- The Pairwise Anchor-based Trigger generation is implemented in ./adv_ir/attack_methods.py
  function name: gen_adversarial_trigger_pair_passage()

- For generating adversarial triggers for ranking attack.
  `python pat_attack.py`
> Note that we adopted the fine-tuned BERT LM from [Song et al.(2020)](https://github.com/csong27/collision-bert/blob/43eda087bf6d632bdb150d98e934206327f8d082/scripts/ft_bert_lm.py)

- Test the transferability of triggers
  `python transfer_attack.py`
