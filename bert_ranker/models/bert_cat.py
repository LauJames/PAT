import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

from transformers import AutoTokenizer, AutoModel, PreTrainedModel, PretrainedConfig
import torch


class BERT_Cat_Config(PretrainedConfig):
    model_type = "BERT_Cat"
    bert_model: str
    trainable: bool = True


class BERT_Cat(PreTrainedModel):
    """
    The vanilla/mono BERT concatenated (we lovingly refer to as BERT_Cat) architecture
    -> requires input concatenation before model, so that batched input is possible
    """
    config_class = BERT_Cat_Config
    base_model_prefix = "bert_model"

    def __init__(self,
                 cfg) -> None:
        super().__init__(cfg)

        self.bert_model = AutoModel.from_pretrained(cfg.bert_model)

        for p in self.bert_model.parameters():
            p.requires_grad = cfg.trainable

        self._classification_layer = torch.nn.Linear(self.bert_model.config.hidden_size, 1)

    def forward(self,
                input_ids=None,
                attention_mask=None):
        # vecs = self.bert_model(**query_n_doc_sequence)[0][:,0,:] # assuming a distilbert model here
        vecs = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[0][:, 0, :]
        score = self._classification_layer(vecs)
        return score