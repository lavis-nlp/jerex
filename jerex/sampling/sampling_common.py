import random

import torch

from jerex import util


def create_positive_mentions(doc, context_size, include_orig_spans=False):
    """ Creates positive samples of entity mentions according to ground truth annotations """

    pos_mention_spans, pos_mention_masks, pos_mention_sizes, pos_mention_orig_spans = [], [], [], []
    for e in doc.entities:
        for m in e.entity_mentions:
            pos_mention_spans.append(m.span)
            pos_mention_masks.append(create_span_mask(*m.span, context_size))
            pos_mention_sizes.append(len(m.tokens))
            pos_mention_orig_spans.append(m.orig_span)

    return (pos_mention_spans, pos_mention_masks, pos_mention_sizes) + ((
        pos_mention_orig_spans,) if include_orig_spans else ())


def create_negative_mentions(doc, pos_mention_spans, neg_mention_count,
                             max_span_size, context_size, overlap_ratio=0.5):
    """ Creates negative samples of entity mentions, i.e. spans that do not match a ground truth mention """
    neg_dist_mention_spans, neg_dist_mention_sizes = [], []
    neg_overlap_mention_spans, neg_overlap_mention_sizes = [], []

    for sentence in doc.sentences:
        sentence_token_count = len(sentence.tokens)

        for size in range(1, max_span_size + 1):
            for i in range(0, (sentence_token_count - size) + 1):
                span = sentence.tokens[i:i + size].span

                if span not in pos_mention_spans:
                    ov = False

                    # check if span is inside a ground truth span
                    for s1, s2 in pos_mention_spans:
                        if span[0] >= s1 and span[1] <= s2:
                            ov = True
                            break

                    if ov:
                        neg_overlap_mention_spans.append(span)
                        neg_overlap_mention_sizes.append(size)
                    else:
                        neg_dist_mention_spans.append(span)
                        neg_dist_mention_sizes.append(size)

    # count of (inside) overlapping negative mentions and distinct negative mentions
    overlap_neg_count = min(len(neg_overlap_mention_spans), int(neg_mention_count * overlap_ratio))
    dist_neg_count = neg_mention_count - overlap_neg_count

    # sample negative entity mentions
    neg_overlap_mention_samples = random.sample(list(zip(neg_overlap_mention_spans, neg_overlap_mention_sizes)),
                                                overlap_neg_count)
    neg_overlap_mention_spans, neg_overlap_mention_sizes = zip(
        *neg_overlap_mention_samples) if neg_overlap_mention_samples else ([], [])
    neg_overlap_mention_masks = [create_span_mask(*span, context_size) for span in neg_overlap_mention_spans]

    neg_dist_mention_samples = random.sample(list(zip(neg_dist_mention_spans, neg_dist_mention_sizes)),
                                             min(len(neg_dist_mention_spans), dist_neg_count))

    neg_dist_mention_spans, neg_dist_mention_sizes = zip(*neg_dist_mention_samples) if neg_dist_mention_samples else (
        [], [])
    neg_dist_mention_masks = [create_span_mask(*span, context_size) for span in neg_dist_mention_spans]

    neg_mention_spans = list(neg_overlap_mention_spans) + list(neg_dist_mention_spans)
    neg_mention_sizes = list(neg_overlap_mention_sizes) + list(neg_dist_mention_sizes)
    neg_mention_masks = list(neg_overlap_mention_masks) + list(neg_dist_mention_masks)

    return neg_mention_spans, neg_mention_sizes, neg_mention_masks


def create_mention_candidates(doc, max_span_size, context_size):
    """ Creates candidates of entity mentions, i.e. all token spans of the document up to a specific length """
    mention_spans = []
    mention_orig_spans = []
    mention_masks = []
    mention_sizes = []
    mention_sent_indices = []

    for sentence in doc.sentences:
        sentence_token_count = len(sentence.tokens)

        for size in range(1, max_span_size + 1):
            for i in range(0, (sentence_token_count - size) + 1):
                span_tokens = sentence.tokens[i:i + size]
                span = span_tokens.span
                mention_spans.append(span)
                mention_orig_spans.append(span_tokens.orig_span)
                mention_masks.append(create_span_mask(*span, context_size))
                mention_sizes.append(size)
                mention_sent_indices.append(sentence.index)

    return mention_masks, mention_sizes, mention_spans, mention_orig_spans, mention_sent_indices


def create_pos_coref_pairs(doc, pos_mention_spans):
    """ Create positive coreference samples, i.e. pairs of mentions that are coreferent according to ground truth"""
    pos_coref_mention_pairs, pos_coref_mention_spans, pos_coref_eds = [], [], []

    for e in doc.entities:
        for m1 in e.entity_mentions:
            for m2 in e.entity_mentions:
                if m1 != m2:
                    s1, s2 = m1.span, m2.span
                    m1_phrase = m1.phrase.strip()
                    m2_phrase = m2.phrase.strip()

                    pos_coref_mention_pairs.append((pos_mention_spans.index(s1), pos_mention_spans.index(s2)))
                    pos_coref_mention_spans.append((s1, s2))
                    pos_coref_eds.append(util.get_edit_distance(m1_phrase, m2_phrase))

    return pos_coref_mention_pairs, pos_coref_mention_spans, pos_coref_eds


def create_neg_coref_pairs(doc, pos_mention_spans, neg_rel_count):
    """ Create negative coreference samples, i.e. pairs of mentions that are not
    coreferent according to ground truth """
    neg_coref_spans = []
    neg_eds = []

    # add mentions of document
    all_mentions = []
    for e in doc.entities:
        for m in e.entity_mentions:
            all_mentions.append(m)

    # filter uncoreferent pairs of all document mentions
    for m1 in all_mentions:
        for m2 in all_mentions:
            if m1.entity != m2.entity:
                m1_phrase = m1.phrase.strip()
                m2_phrase = m2.phrase.strip()

                s1, s2 = m1.span, m2.span
                neg_coref_spans.append((s1, s2))
                neg_eds.append(util.get_edit_distance(m1_phrase, m2_phrase))

    neg_samples = list(zip(neg_coref_spans, neg_eds))
    neg_samples = random.sample(neg_samples, min(len(neg_samples), neg_rel_count))
    neg_coref_spans, neg_eds = zip(*neg_samples) if neg_samples else ([], [])
    neg_coref_spans = list(neg_coref_spans)
    neg_eds = list(neg_eds)

    neg_coref_mention_pairs = [(pos_mention_spans.index(s1), pos_mention_spans.index(s2)) for s1, s2 in neg_coref_spans]

    return neg_coref_mention_pairs, neg_coref_spans, neg_eds


def create_coref_candidates(doc, pos_mention_spans):
    """ Create coreference candidates, i.e. all pairs of mentions of the document """
    coref_spans = []
    eds = []

    all_mentions = []

    for e in doc.entities:
        for m in e.entity_mentions:
            all_mentions.append(m)

    for m1 in all_mentions:
        for m2 in all_mentions:
            if m1 != m2:
                m1_phrase = m1.phrase.strip()
                m2_phrase = m2.phrase.strip()

                s1, s2 = m1.span, m2.span
                coref_spans.append((s1, s2))
                # add edit distance between the two mentions
                eds.append(util.get_edit_distance(m1_phrase, m2_phrase))

    coref_mention_pairs = [(pos_mention_spans.index(s1), pos_mention_spans.index(s2)) for s1, s2 in coref_spans]
    return coref_mention_pairs, coref_spans, eds


def create_entities(doc, pos_mention_spans):
    """ Create samples of ground truth entity clusters with type """
    entities = []
    entity_types = []

    for e in doc.entities:
        entities.append([pos_mention_spans.index(m.span) for m in e.entity_mentions])
        entity_types.append(e.entity_type.index)

    return entities, entity_types


def create_entity_pairs(doc):
    """ Create pairs of entity indices """
    entity_pairs = []

    for i1, _ in enumerate(doc.entities):
        for i2, _ in enumerate(doc.entities):
            if i1 != i2:
                pair = (i1, i2)
                entity_pairs.append(pair)
    return entity_pairs


def create_pos_relations(doc, rel_type_count):
    """ Creates positive relation samples, i.e. actual relations according to ground truth """
    pos_rel_entity_pairs, pos_rel_types = [], []

    # mapping from entity pair to relations
    rels_between_entities = dict()

    for rel in doc.relations:
        head, tail = rel.head_entity, rel.tail_entity
        head_idx, tail_idx = doc.entities.index(head), doc.entities.index(tail)
        pair = (head_idx, tail_idx)

        if pair not in rels_between_entities:
            rels_between_entities[pair] = []

        rels_between_entities[pair].append(rel)

    for pair, rels in rels_between_entities.items():
        rel_types = [r.relation_type.index for r in rels]
        one_hot = [(1 if t in rel_types else 0) for t in range(0, rel_type_count)]

        pos_rel_entity_pairs.append(pair)
        pos_rel_types.append(one_hot)

    return pos_rel_entity_pairs, pos_rel_types, rels_between_entities


def create_neg_relations(entities, rels_between_entities, rel_type_count, neg_rel_count):
    """ Creates negative relation samples, i.e. pairs of entities that are unrelated according to ground truth """

    neg_unrelated = []

    # search unrelated entity pairs
    for i1, _ in enumerate(entities):
        for i2, _ in enumerate(entities):
            if i1 != i2:
                pair = (i1, i2)
                if pair not in rels_between_entities:
                    neg_unrelated.append(pair)

    # sample negative relations
    neg_unrelated = random.sample(neg_unrelated, min(len(neg_unrelated), neg_rel_count))
    neg_rel_entity_pairs, neg_rel_types = [], []

    for pair in neg_unrelated:
        one_hot = [0] * rel_type_count
        neg_rel_entity_pairs.append(pair)
        neg_rel_types.append(one_hot)

    return neg_rel_entity_pairs, neg_rel_types


def create_rel_mention_pairs(doc, rel_entity_pairs, pos_mention_spans, context_size, offset_mp=0, offset_ep=0):
    """ Create pairs of all mentions of two entity clusters. Used in multi-instance relation classifier """
    rel_entity_pair_mp = []
    rel_mention_pair_ep = []
    rel_mention_pairs = []
    rel_ctx_masks = []
    rel_token_distances = []
    rel_sentence_distances = []

    for pair_idx, pair in enumerate(rel_entity_pairs):
        head = doc.entities[pair[0]]
        tail = doc.entities[pair[1]]

        entity_pair_mp = []

        for m1 in head.entity_mentions:
            for m2 in tail.entity_mentions:
                entity_pair_mp.append(len(rel_mention_pairs) + offset_mp)
                rel_mention_pair_ep.append(pair_idx + offset_ep)

                idx1 = pos_mention_spans.index(m1.span)
                idx2 = pos_mention_spans.index(m2.span)

                mention_pair = (idx1, idx2)
                rel_mention_pairs.append(mention_pair)

                # context between mentions
                rel_ctx_masks.append(create_rel_mask(m1.span, m2.span, context_size))

                # token and sentence distance of mentions
                token_dist = get_mention_token_dist(m1, m2)
                sentence_dist = abs(m1.sentence.index - m2.sentence.index)
                rel_token_distances.append(token_dist)
                rel_sentence_distances.append(sentence_dist)

        rel_entity_pair_mp.append(entity_pair_mp)

    return (rel_entity_pair_mp, rel_mention_pair_ep, rel_mention_pairs, rel_ctx_masks,
            rel_token_distances, rel_sentence_distances)


def get_mention_token_dist(m1, m2):
    """ Returns distance in tokens between two mentions """
    succ = m1.tokens[0].doc_index < m2.tokens[0].doc_index
    first = m1 if succ else m2
    second = m2 if succ else m1

    return max(0, second.tokens[0].doc_index - first.tokens[-1].doc_index)


def get_mention_token_dist_tensors(m1, m2):
    """ Returns distance in tokens between two mentions """
    succ = m1[0] < m2[0]
    first = m1 if succ else m2
    second = m2 if succ else m1

    d = second[0] - (first[1] - 1)
    if d < 0:
        return torch.tensor(0, dtype=torch.long, device=m1.device)
    return d


def create_span_mask(start, end, context_size):
    """ creates tensor for masking a token span  """
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    """ creates tensor for masking the span between two spans (used to mask the span between two entity mentions) """
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_span_mask(start, end, context_size)
    return mask


def create_context_tensors(encodings):
    """ Creates tensor of sub-word encodings and context masks """
    context_size = len(encodings)
    # token indices
    encodings = torch.tensor(encodings, dtype=torch.long)

    # masking of tokens
    context_masks = torch.ones(context_size, dtype=torch.bool)

    return encodings, context_masks


def create_mention_tensors(ctx_size, pos_mention_spans, pos_mention_masks, pos_mention_sizes,
                           neg_mention_spans=None, neg_mention_masks=None, neg_mention_sizes=None):
    """ Combines positive and negative mention samples into tensors """
    neg_mention_spans = neg_mention_spans if neg_mention_spans else []
    neg_mention_masks = neg_mention_masks if neg_mention_masks else []
    neg_mention_sizes = neg_mention_sizes if neg_mention_sizes else []

    mention_spans = pos_mention_spans + neg_mention_spans
    mention_masks = pos_mention_masks + neg_mention_masks
    mention_sizes = pos_mention_sizes + neg_mention_sizes
    mention_types = [1] * len(pos_mention_spans) + [0] * len(neg_mention_spans)

    if mention_masks:
        mention_types = torch.tensor(mention_types, dtype=torch.long)
        mention_masks = torch.stack(mention_masks)
        mention_sizes = torch.tensor(mention_sizes, dtype=torch.long)
        mention_spans = torch.tensor(mention_spans, dtype=torch.long)
        mention_sample_masks = torch.ones([mention_masks.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg entities)
        mention_types = torch.zeros([1], dtype=torch.long)
        mention_masks = torch.ones([1, ctx_size], dtype=torch.bool)
        mention_sizes = torch.zeros([1], dtype=torch.long)
        mention_spans = torch.zeros([1, 2], dtype=torch.long)
        mention_sample_masks = torch.zeros([1], dtype=torch.bool)

    return mention_types, mention_masks, mention_sizes, mention_spans, mention_sample_masks


def create_mention_candidate_tensors(ctx_size, mention_masks, mention_sizes, mention_spans,
                                     mention_orig_spans, mention_sent_indices):
    """ Creates necessary tensors for entity mention candidates """
    if mention_masks:
        mention_spans = torch.tensor(mention_spans, dtype=torch.long)
        mention_orig_spans = torch.tensor(mention_orig_spans, dtype=torch.long)
        mention_masks = torch.stack(mention_masks)
        mention_sizes = torch.tensor(mention_sizes, dtype=torch.long)
        mention_sent_indices = torch.tensor(mention_sent_indices, dtype=torch.long)
        mention_sample_masks = torch.tensor([1] * mention_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        mention_spans = torch.zeros([1, 2], dtype=torch.long)
        mention_orig_spans = torch.zeros([1, 2], dtype=torch.long)
        mention_masks = torch.ones([1, ctx_size], dtype=torch.bool)
        mention_sizes = torch.zeros([1], dtype=torch.long)
        mention_sent_indices = torch.zeros([1], dtype=torch.long)
        mention_sample_masks = torch.zeros([1], dtype=torch.bool)

    return mention_masks, mention_sizes, mention_spans, mention_orig_spans, mention_sent_indices, mention_sample_masks


def create_coref_tensors(pos_coref_mention_pairs, pos_eds,
                         neg_coref_mention_pairs=None, neg_eds=None):
    """ Combines positive and negative corefeference samples into tensors """
    neg_coref_mention_pairs = neg_coref_mention_pairs if neg_coref_mention_pairs else []
    neg_eds = neg_eds if neg_eds else []

    coref_mention_pairs = pos_coref_mention_pairs + neg_coref_mention_pairs
    coref_ed = pos_eds + neg_eds
    coref_types = [1] * len(pos_coref_mention_pairs) + [0] * len(neg_coref_mention_pairs)

    if coref_mention_pairs:
        coref_mention_pairs = torch.tensor(coref_mention_pairs, dtype=torch.long)
        coref_types = torch.tensor(coref_types, dtype=torch.long)
        coref_ed = torch.tensor(coref_ed, dtype=torch.long)
        coref_sample_masks = torch.ones([coref_mention_pairs.shape[0]], dtype=torch.bool)
    else:
        coref_mention_pairs = torch.zeros([1, 2], dtype=torch.long)
        coref_types = torch.zeros([1], dtype=torch.long)
        coref_ed = torch.zeros([1], dtype=torch.long)
        coref_sample_masks = torch.zeros([1], dtype=torch.long)

    return coref_mention_pairs, coref_types, coref_ed, coref_sample_masks


def create_entity_tensors(entities, entity_types):
    """ Creates necessary tensors for multi-instance entity typing """
    if entities:
        entity_masks = util.padded_stack([torch.ones(len(e), dtype=torch.bool) for e in entities])
        entities = util.padded_stack([torch.tensor(e, dtype=torch.long) for e in entities])
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_sample_masks = torch.ones([entities.shape[0]], dtype=torch.bool)
    else:
        entity_masks = torch.ones([1, 1], dtype=torch.bool)
        entities = torch.zeros([1, 1], dtype=torch.long)
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return entities, entity_masks, entity_types, entity_sample_masks


def create_entity_pair_tensors(rel_entity_pairs):
    """ Creates tensors of entity (cluster) pairs """
    if rel_entity_pairs:
        rel_entity_pairs = torch.tensor(rel_entity_pairs, dtype=torch.long)
        rel_sample_masks = torch.ones([rel_entity_pairs.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pairs)
        rel_entity_pairs = torch.zeros([1, 2], dtype=torch.long)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    return rel_entity_pairs, rel_sample_masks


def create_rel_global_tensors(pos_rel_entity_pairs, pos_rel_types,
                              neg_rel_entity_pairs, neg_rel_types):
    """ Combine positive and negative samples for the global relation classifier """
    rel_entity_pairs = pos_rel_entity_pairs + neg_rel_entity_pairs
    rel_types = pos_rel_types + neg_rel_types

    if rel_entity_pairs:
        rel_entity_pairs = torch.tensor(rel_entity_pairs, dtype=torch.long)
        rel_types = torch.tensor(rel_types, dtype=torch.long)
        rel_sample_masks = torch.ones([rel_entity_pairs.shape[0]], dtype=torch.bool)
    else:
        # corner case handling (no pos/neg relations)
        rel_entity_pairs = torch.zeros([1, 2], dtype=torch.long)
        rel_types = torch.zeros([1], dtype=torch.long)
        rel_sample_masks = torch.zeros([1], dtype=torch.bool)

    return rel_entity_pairs, rel_types, rel_sample_masks


def create_rel_mi_tensors(context_size, pos_rel_entity_pair_mp, pos_rel_mention_pair_ep, pos_rel_mention_pairs,
                          pos_rel_ctx_masks, pos_rel_token_distances,
                          pos_rel_sentence_distances, neg_rel_entity_pair_mp=None, neg_rel_mention_pair_ep=None,
                          neg_rel_mention_pairs=None, neg_rel_ctx_masks=None,
                          neg_rel_token_distances=None, neg_rel_sentence_distances=None):
    """ Combine positive and negative samples of mention pairs for multi-instance relation classifier """
    neg_rel_entity_pair_mp = neg_rel_entity_pair_mp if neg_rel_entity_pair_mp else []
    neg_rel_mention_pair_ep = neg_rel_mention_pair_ep if neg_rel_mention_pair_ep else []
    neg_rel_mention_pairs = neg_rel_mention_pairs if neg_rel_mention_pairs else []
    neg_rel_ctx_masks = neg_rel_ctx_masks if neg_rel_ctx_masks else []
    neg_rel_token_distances = neg_rel_token_distances if neg_rel_token_distances else []
    neg_rel_sentence_distances = neg_rel_sentence_distances if neg_rel_sentence_distances else []

    rel_entity_pair_mp = pos_rel_entity_pair_mp + neg_rel_entity_pair_mp
    rel_mention_pair_ep = pos_rel_mention_pair_ep + neg_rel_mention_pair_ep
    rel_mention_pairs = pos_rel_mention_pairs + neg_rel_mention_pairs
    rel_ctx_masks = pos_rel_ctx_masks + neg_rel_ctx_masks
    rel_token_distances = pos_rel_token_distances + neg_rel_token_distances
    rel_sentence_distances = pos_rel_sentence_distances + neg_rel_sentence_distances

    if rel_entity_pair_mp:
        rel_pair_masks = util.padded_stack([torch.ones(len(e), dtype=torch.bool) for e in rel_entity_pair_mp])
        rel_entity_pair_mp = util.padded_stack([torch.tensor(e, dtype=torch.long) for e in rel_entity_pair_mp])
        rel_mention_pair_ep = torch.tensor(rel_mention_pair_ep, dtype=torch.long)
        rel_mention_pairs = torch.tensor(rel_mention_pairs, dtype=torch.long)
        rel_ctx_masks = torch.stack(rel_ctx_masks)
        rel_token_distances = torch.tensor(rel_token_distances, dtype=torch.long)
        rel_sentence_distances = torch.tensor(rel_sentence_distances, dtype=torch.long)
    else:
        # corner case handling (no pos/neg relations)
        rel_pair_masks = torch.zeros([1, 1], dtype=torch.bool)
        rel_entity_pair_mp = torch.zeros([1, 1], dtype=torch.long)
        rel_mention_pair_ep = torch.zeros([1], dtype=torch.long)
        rel_mention_pairs = torch.zeros([1, 2], dtype=torch.long)
        rel_ctx_masks = torch.ones([1, context_size], dtype=torch.bool)
        rel_token_distances = torch.zeros([1], dtype=torch.long)
        rel_sentence_distances = torch.zeros([1], dtype=torch.long)

    return (rel_entity_pair_mp, rel_mention_pair_ep, rel_mention_pairs, rel_ctx_masks,
            rel_pair_masks, rel_token_distances, rel_sentence_distances)


def collate_fn_padding(batch):
    """ Pads a batch with zero values """
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch
