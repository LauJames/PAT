import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))

import torch
from transformers import BertModel, BertForSequenceClassification,  BertForNextSentencePrediction, BertForMaskedLM
from transformers.models.bert.modeling_bert import BertSelfAttention, BertAttention, BertLayer, BertEncoder


class BertSelfAttentionPast(BertSelfAttention):
    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False,
            output_attentions=False):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        if layer_past is not None:
            if cache_query:
                past_q = layer_past[2]
                query_layer = torch.cat((past_q, query_layer), dim=-2)

            past_k, past_v = layer_past[0], layer_past[1]
            key_layer = torch.cat((past_k, key_layer), dim=-2)
            value_layer = torch.cat((past_v, value_layer), dim=-2)

        if cache_query:
            present = torch.stack([key_layer, value_layer, query_layer])
        else:
            present = torch.stack([key_layer, value_layer])

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if layer_past is None and attention_mask is not None:
            attention_scores += attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs, present) if output_attentions else (context_layer, present)
        return outputs


class BertAttentionPast(BertAttention):
    def __init__(self, config):
        super().__init__(config)
        self.self = BertSelfAttentionPast(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False):
        self_outputs = self.self(
            hidden_states, attention_mask, head_mask, encoder_hidden_states,
            encoder_attention_mask, layer_past, cache_query
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


class BertLayerPast(BertLayer):
    def __init__(self, config):
        super().__init__(config)
        self.attention = BertAttentionPast(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            layer_past=None,
            cache_query=False):
        self_attention_outputs = self.attention(hidden_states, attention_mask,
                                                head_mask, layer_past=layer_past,
                                                cache_query=cache_query)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output, attention_mask, head_mask,
                encoder_hidden_states, encoder_attention_mask
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


class BertEncoderPast(BertEncoder):
    def __init__(self, config):
        super().__init__(config)
        self.output_past = getattr(config, 'output_past', True)
        self.layer = nn.ModuleList(
            [BertLayerPast(config) for _ in range(config.num_hidden_layers)])
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            head_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            past=None,
            cache_query=False):
        if past is None:
            past = [None] * len(self.layer)

        all_hidden_states = ()
        all_attentions = ()
        presents = ()

        for i, (layer_module, layer_past) in enumerate(zip(self.layer, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states, attention_mask, head_mask[i], encoder_hidden_states,
                encoder_attention_mask, layer_past, cache_query
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            present = layer_outputs[-1]
            if self.output_past:
                presents = presents + (present,)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs


class MaskedMultiHeadSelfAttention(MultiHeadSelfAttention):
    def forward(self, query, key, value, layer_past=None,
                mask=None, head_mask=None, output_attentions=False):
        bs, q_length, dim = query.size()
        dim_per_head = self.dim // self.n_heads

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.n_heads * dim_per_head)

        q = shape(self.q_lin(query))  # (bs, n_heads, q_length, dim_per_head)
        k = shape(self.k_lin(key))  # (bs, n_heads, k_length, dim_per_head)
        v = shape(self.v_lin(value))  # (bs, n_heads, k_length, dim_per_head)
        if layer_past is not None:
            past_k, past_v = layer_past[0], layer_past[1]
            k = torch.cat((past_k, k), dim=-2)
            v = torch.cat((past_v, v), dim=-2)

        present = torch.stack([k, v])

        q = q / math.sqrt(dim_per_head)  # (bs, n_heads, q_length, dim_per_head)
        # (bs, n_heads, q_length, k_length)
        scores = torch.matmul(q, k.transpose(2, 3))
        if layer_past is None and mask is not None:
            scores += mask

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        weights = self.dropout(weights)  # (bs, n_heads, q_length, k_length)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = self.out_lin(context)  # (bs, q_length, dim)

        if output_attentions:
            return context, present, weights
        else:
            return context, present


class MaskedTransformerBlock(TransformerBlock):
    def __init__(self, config):
        super().__init__(config)
        self.attention = MaskedMultiHeadSelfAttention(config)
        self.output_attentions = config.output_attentions

    def forward(self, x, layer_past=None, attn_mask=None, head_mask=None):
        sa_output = self.attention(query=x, key=x, value=x, layer_past=layer_past,
                                   mask=attn_mask, head_mask=head_mask)
        if self.output_attentions:
            # (bs, seq_length, dim), (bs, n_heads, seq_length, seq_length)
            sa_output, sa_present, sa_weights = sa_output
        else:
            assert type(sa_output) == tuple
            sa_output, sa_present = sa_output
        sa_output = self.sa_layer_norm(sa_output + x)  # (bs, seq_length, dim)

        # Feed Forward Network
        ffn_output = self.ffn(sa_output)  # (bs, seq_length, dim)
        ffn_output = self.output_layer_norm(
            ffn_output + sa_output)  # (bs, seq_length, dim)

        output = (ffn_output, sa_present)
        if self.output_attentions:
            output = (sa_weights,) + output
        return output


class MaskedTransformer(Transformer):
    def __init__(self, config):
        super().__init__(config)
        self.output_past = getattr(config, 'output_past', True)
        layer = MaskedTransformerBlock(config)
        self.layer = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(config.n_layers)])

    def forward(self, x, past=None, attn_mask=None, head_mask=None):
        if past is None:
            past = [None] * len(self.layer)

        all_hidden_states = ()
        all_attentions = ()
        presents = ()

        hidden_state = x
        for i, (layer_module, layer_past) in enumerate(zip(self.layer, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_state,)

            layer_outputs = layer_module(x=hidden_state, layer_past=layer_past,
                                         attn_mask=attn_mask, head_mask=head_mask[i])
            hidden_state = layer_outputs[-2]
            present = layer_outputs[-1]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                assert len(layer_outputs) == 3
                attentions = layer_outputs[0]
                all_attentions = all_attentions + (attentions,)
            else:
                assert len(layer_outputs) == 2

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_state,)

        outputs = (hidden_state,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs



class BertForLM(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertAutoRegressiveModel(config)
        self.start_idx = 1
        self.init_weights()

    def prepare_inputs_for_generation(self, input_ids, past):
        # only last token for inputs_ids if past is defined in kwargs
        if past:
            input_ids = input_ids[:, -1].unsqueeze(-1)
        return {"input_ids": input_ids, "past": past}

    def enforce_repetition_penalty_(self, lprobs, batch_size, num_beams, prev_output_tokens, repetition_penalty):
        """repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858). """
        for i in range(batch_size * num_beams):
            for previous_token in set(prev_output_tokens[i].tolist()):
                # if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
                if lprobs[i, previous_token] < 0:
                    lprobs[i, previous_token] *= repetition_penalty
                else:
                    lprobs[i, previous_token] /= repetition_penalty

    def _reorder_cache(self, past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            reordered_past += (tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),)
        return reordered_past


    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            masked_lm_labels=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            labels=None,
            one_hot_labels=None,
            past=None
    ):
        label_start_idx = 1
        if inputs_embeds is not None:
            start_embeds = self.get_input_embeddings().weight[self.start_idx]
            inputs_embeds = torch.cat([start_embeds.view(1, 1, -1), inputs_embeds], 1)
            label_start_idx = 0

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past=past
        )

        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        # Add hidden states and attention if they are here
        outputs = (prediction_scores,) + outputs[2:]

        # we are doing next-token prediction;
        # shift prediction scores and input ids by one
        if one_hot_labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = one_hot_labels[:, label_start_idx:, :].contiguous()
            nll = -torch.log_softmax(prediction_scores, -1)
            ltr_lm_loss = torch.sum(nll * lm_labels, -1).mean()
            outputs = (ltr_lm_loss,) + outputs
        elif labels is not None:
            prediction_scores = prediction_scores[:, :-1, :].contiguous()
            lm_labels = labels[:, label_start_idx:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            ltr_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (ltr_lm_loss,) + outputs
        return outputs