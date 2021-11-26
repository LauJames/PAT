import sys
import os

curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(curdir))
prodir = os.path.dirname(curdir)

import torch
from pattern.text.en import singularize, pluralize
from attack_utils import create_constraints, get_inputs_filter_ids, get_sub_masks, STOPWORDS

BOS_TOKEN = '[unused0]'


def find_filters(query, model, tokenizer, device, k=500):
    words = [w for w in tokenizer.vocab if w.isalpha() and w not in STOPWORDS]
    inputs = tokenizer.batch_encode_plus([[query, w] for w in words], max_length=32, padding="max_length")
    all_input_ids = torch.tensor(inputs['input_ids'], device=device)
    all_token_type_ids = torch.tensor(inputs['token_type_ids'], device=device)
    all_attention_masks = torch.tensor(inputs['attention_mask'], device=device)
    n = len(words)
    batch_size = 1024
    n_batches = n // batch_size + 1
    all_scores = []
    for i in range(n_batches):
        input_ids = all_input_ids[i * batch_size: (i + 1) * batch_size]
        token_type_ids = all_token_type_ids[i * batch_size: (i + 1) * batch_size]
        attention_masks = all_attention_masks[i * batch_size: (i + 1) * batch_size]
        outputs = model(input_ids_pos=input_ids,
                        attention_mask_pos=attention_masks,
                        token_type_ids_pos=token_type_ids,
                        input_ids_neg=input_ids,
                        attention_mask_neg=attention_masks,
                        token_type_ids_neg=token_type_ids)
        scores = outputs[0][:, 1]
        all_scores.append(scores)

    all_scores = torch.cat(all_scores)
    _, top_indices = torch.topk(all_scores, k)
    filters = set([words[i.item()] for i in top_indices])
    return [w for w in filters if w.isalpha()]


def add_single_plural(text, tokenizer, contain_sub=False):
    tokens = tokenizer.tokenize(text)
    contains = []
    if contain_sub:
        for word in tokenizer.vocab:
            if word.isalpha() and len(word) > 2:
                for t in tokens:
                    if len(t) > 2 and word != t and (word.startswith(t) or t.startswith(word)):
                        contains.append(word)

    for t in tokens[:]:
        if not t.isalpha():
            continue
        sig_t = singularize(t)
        plu_t = pluralize(t)
        if sig_t != t and sig_t in tokenizer.vocab:
            tokens.append(sig_t)
        if plu_t != t and plu_t in tokenizer.vocab:
            tokens.append(plu_t)

    return [w for w in tokens + contains if w not in STOPWORDS]


def gen_adversarial_trigger_pair_passage(query, best_sent, raw_passage, model, tokenizer, device, args, logger,
                                         nsp_model=None, lm_model=None):
    word_embedding = model.get_input_embeddings().weight.detach()

    if lm_model is not None:
        lm_word_embedding = lm_model.get_input_embeddings().weight.detach()

    vocab_size = word_embedding.size(0)
    input_mask = torch.zeros(vocab_size, device=device)

    query_ids = tokenizer.tokenize(query, add_special_tokens=False)
    removed_query_tokens = []
    for word in query_ids:
        if word.isalpha() and len(word) > 2:
            if not word.startswith('##'):
                removed_query_tokens.append(word)
    remove_q_ids = tokenizer.convert_tokens_to_ids(removed_query_tokens)
    input_mask[remove_q_ids] = 1e-9
    if args.verbose:
        logger.info(','.join(removed_query_tokens))
        print(','.join(removed_query_tokens) + '\n')

    sub_mask = torch.zeros(vocab_size, device=device)

    # [q_len + 2]
    query_ids = tokenizer.encode(query, add_special_tokens=True)
    query_ids = torch.tensor(query_ids, device=device).unsqueeze(0)

    seq_len = args.tri_len
    # [50, q_len + 2]
    batch_query_ids = torch.cat([query_ids] * args.topk, 0)
    # [50, vocab_size]
    stopwords_mask = create_constraints(seq_len, tokenizer, device)

    # raw passage transform to ids
    passage_ids = tokenizer.encode(raw_passage, add_special_tokens=True)
    # remove [CLS]
    passage_ids = torch.tensor(passage_ids[1:], device=device).unsqueeze(0)

    # best_sent transform to ids
    best_sent_ids = tokenizer.encode(best_sent, add_special_tokens=True)
    # remove [CLS]
    best_sent_ids = torch.tensor(best_sent_ids[1:], device=device).unsqueeze(0)

    def relaxed_to_word_embs(x):
        # convert relaxed inputs to word embedding by softmax attention
        masked_x = x + input_mask + sub_mask
        if args.regularize:
            masked_x += stopwords_mask
        p = torch.softmax(masked_x / args.stemp, -1)
        # [vocab_size] * [vocab_size, 1024]
        x = torch.mm(p, word_embedding)
        # add embeddings for period and SEP
        x = torch.cat([x, word_embedding[tokenizer.sep_token_id].unsqueeze(0)])
        return p, x.unsqueeze(0)

    def get_lm_loss(p):
        x = torch.mm(p.detach(), lm_word_embedding).unsqueeze(0)
        return lm_model(inputs_embeds=x, one_hot_labels=p.unsqueeze(0))[0]

    def ids_to_emb(input_ids):
        input_ids_one_hot = torch.nn.functional.one_hot(input_ids, vocab_size).float()
        input_emb = torch.einsum('blv,vh->blh', input_ids_one_hot, word_embedding)
        return input_emb

    # some constants
    # [50]
    sep_tensor = torch.tensor([tokenizer.sep_token_id] * args.topk, device=device)
    # [50, 1, 1024]
    batch_sep_embeds = word_embedding[sep_tensor].unsqueeze(1)
    nsp_labels = torch.zeros((1,), dtype=torch.long, device=device)
    labels = torch.ones((1,), dtype=torch.long, device=device)
    repetition_penalty = args.repetition_penalty

    best_trigger = None
    best_score = -1e9
    prev_score = -1e9
    trigger_cands = []
    patience = 0

    var_size = (seq_len, vocab_size)
    z_i = torch.normal(mean=0., std=1.0, size=var_size, requires_grad=True, device=device)
    for it in range(args.max_iter):
        optimizer = torch.optim.Adam([z_i], lr=args.lr)
        for j in range(args.perturb_iter):
            optimizer.zero_grad()
            # relaxation
            # [16, 30522], [1, 17, 1024] - add [SEP]
            p_inputs, z_trigger_embeds = relaxed_to_word_embs(z_i)
            # forward to BERT with relaxed inputs
            # transform query into embedd vector [1, seq_len + 2] --> [1, seq_len + 2, vocab_size]
            query_emb = ids_to_emb(query_ids)
            best_sent_emb = ids_to_emb(best_sent_ids)
            passage_emb = ids_to_emb(passage_ids)

            # [CLS + query + SEP, trigger, SEP] > [CLS + query + SEP, best, SEP]
            concat_inputs_emb_pos = torch.cat([query_emb, z_trigger_embeds], dim=1)
            concat_inputs_emb_neg = torch.cat([query_emb, best_sent_emb], dim=1)

            outputs = model(
                inputs_embeds_pos=concat_inputs_emb_pos,
                inputs_embeds_neg=concat_inputs_emb_neg[:, :256, :],
                labels=labels
            )
            loss, cls_logits = outputs[0], outputs[1]

            if args.beta > 0.:
                lm_loss = get_lm_loss(p_inputs)
                loss += args.beta * lm_loss

            if args.gamma > 0.:
                concat_nsp_embs = torch.cat([
                    word_embedding[tokenizer.cls_token_id].unsqueeze(0).unsqueeze(0),
                    z_trigger_embeds,
                    passage_emb], dim=1)[:, :256, :]
                nsp_loss = nsp_model.forward(inputs_embeds=concat_nsp_embs, labels=nsp_labels, return_dict=True)['loss']
                loss += args.gamma * nsp_loss

            loss.backward()
            optimizer.step()
            if args.verbose and (j + 1) % 10 == 0:
                logger.info('It{}-{}, loss={}'.format(it, j + 1, loss.item()))
                print('It{}-{}, loss={}'.format(it, j + 1, loss.item()))

        # detach to free GPU memory
        z_i = z_i.detach()
        # [seq_len, 50]
        _, topk_tokens = torch.topk(z_i, args.topk)
        # [50, seq_len, vocab_size]
        probs_i = torch.softmax(z_i / args.stemp, -1).unsqueeze(0).expand(args.topk, seq_len, vocab_size)

        # [num_beam=10, 16]
        output_so_far = None
        # beam search left to right, get candidate trigger
        for t in range(seq_len):
            tmp_topk_tokens = topk_tokens[t]
            # [50, vocab_size]
            tmp_topk_onehot = torch.nn.functional.one_hot(tmp_topk_tokens, vocab_size).float()
            next_clf_scores = []
            for j in range(args.num_beams):
                next_beam_scores = torch.zeros(tokenizer.vocab_size, device=device) - 1e9
                if output_so_far is None:
                    context = probs_i.clone()
                else:
                    output_len = output_so_far.shape[1]
                    beam_topk_output = output_so_far[j].unsqueeze(0).expand(args.topk, output_len)
                    beam_topk_output = torch.nn.functional.one_hot(beam_topk_output, vocab_size)
                    context = torch.cat([beam_topk_output.float(), probs_i[:, output_len:].clone()], 1)
                context[:, t] = tmp_topk_onehot
                search_context_embeds = torch.einsum('blv,vh->blh', context, word_embedding)

                # [50, seq_len, vocab_size]
                batch_query_emb = ids_to_emb(batch_query_ids)

                context_embeds = torch.cat([search_context_embeds, batch_sep_embeds], 1)
                # concat inputs and context embeddings
                concat_inputs_emb_pos = torch.cat([batch_query_emb, context_embeds], dim=1)
                clf_logits = model(
                    inputs_embeds_pos=concat_inputs_emb_pos,
                    inputs_embeds_neg=concat_inputs_emb_pos
                )[0]
                clf_scores = clf_logits[:, 1].detach().float()
                next_beam_scores.scatter_(0, tmp_topk_tokens, clf_scores)
                next_clf_scores.append(next_beam_scores.unsqueeze(0))

            next_clf_scores = torch.cat(next_clf_scores, 0)
            next_scores = next_clf_scores + input_mask + sub_mask

            if args.regularize:
                next_scores += stopwords_mask[t]
            if output_so_far is None:
                next_scores[1:] = -1e9
            if output_so_far is not None and repetition_penalty > 1.0:
                lm_model.enforce_repetition_penalty_(next_scores, 1, args.num_beams, output_so_far, repetition_penalty)

            # re-organize to group the beam together
            # [batch_size, num_beams * vocab_size]
            next_scores = next_scores.view(1, args.num_beams * vocab_size)
            next_scores, next_tokens = torch.topk(next_scores, args.num_beams, dim=1, largest=True, sorted=True)
            # next batch beam content
            next_sent_beam = []
            for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[0], next_scores[0])):
                # get beam and token IDs
                beam_id = torch.div(beam_token_id, vocab_size, rounding_mode='trunc')
                token_id = beam_token_id % vocab_size
                next_sent_beam.append((beam_token_score, token_id, beam_id))

            next_batch_beam = next_sent_beam
            # sanity check / prepare next batch
            assert len(next_batch_beam) == args.num_beams
            beam_tokens = torch.tensor([x[1] for x in next_batch_beam], device=device)
            beam_idx = torch.tensor([x[2] for x in next_batch_beam], device=device)

            # re-order batch
            if output_so_far is None:
                output_so_far = beam_tokens.unsqueeze(1)
            else:
                output_so_far = output_so_far[beam_idx, :]
                output_so_far = torch.cat([output_so_far, beam_tokens.unsqueeze(1)], dim=-1)
            # end of beam search
        # [num_beams, tri_len + 1]
        pad_output_so_far = torch.cat([output_so_far, sep_tensor[:args.num_beams].unsqueeze(1)], dim=1)

        concat_input_ids_pos = torch.cat([batch_query_ids[:args.num_beams], pad_output_so_far], 1)
        token_type_ids_pos = torch.cat([torch.zeros_like(batch_query_ids[:args.num_beams]),
                                        torch.ones_like(pad_output_so_far)], dim=1)
        attention_mask_pos = torch.ones_like(concat_input_ids_pos)

        final_clf_logits = model(
            input_ids_pos=concat_input_ids_pos,
            attention_mask_pos=attention_mask_pos,
            token_type_ids_pos=token_type_ids_pos,
            input_ids_neg=concat_input_ids_pos,
            attention_mask_neg=attention_mask_pos,
            token_type_ids_neg=token_type_ids_pos
        )[0]
        actual_clf_scores = final_clf_logits[:, 1]
        sorter = torch.argsort(actual_clf_scores, -1, descending=True)
        # validation the output
        valid_idx = sorter[0]

        curr_best = output_so_far[valid_idx]
        next_z_i = torch.nn.functional.one_hot(curr_best, vocab_size).float()
        eps = args.eps
        next_z_i = (next_z_i * (1 - eps)) + (1 - next_z_i) * eps / (vocab_size - 1)
        z_i = torch.nn.Parameter(torch.log(next_z_i), True)

        curr_score = actual_clf_scores[valid_idx].item()

        if curr_score > best_score:
            patience = 0
            best_score = curr_score
            best_trigger = tokenizer.decode(curr_best.cpu().tolist())
            print(curr_score)

        if curr_score <= prev_score:
            patience += 1
        if patience > args.patience_limit:
            break

        prev_score = curr_score

    return best_trigger, best_score, trigger_cands
