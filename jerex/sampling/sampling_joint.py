from jerex.entities import Document
from jerex.sampling.sampling_common import create_mention_candidates, create_context_tensors, create_mention_tensors, \
    create_coref_tensors, create_entity_tensors, create_rel_global_tensors, create_rel_mi_tensors, \
    create_pos_relations, create_neg_relations, create_rel_mention_pairs, \
    create_pos_coref_pairs, create_neg_coref_pairs, create_entities, create_positive_mentions, \
    create_negative_mentions, create_mention_candidate_tensors


def create_joint_train_sample(doc: Document, neg_mention_count: int, neg_rel_count: int, neg_coref_count: int,
                              max_span_size: int, neg_mention_overlap_ratio: float, rel_type_count: int,
                              size_embeddings_count: int):
    encodings = doc.encodings  # document sub-word encoding
    context_size = len(encodings)

    # positive entity mentions
    pos_mention_spans, pos_mention_masks, pos_mention_sizes = create_positive_mentions(doc, context_size)

    # negative entity mentions
    neg_mention_spans, neg_mention_sizes, neg_mention_masks = create_negative_mentions(doc, pos_mention_spans,
                                                                                       neg_mention_count,
                                                                                       max_span_size,
                                                                                       context_size,
                                                                                       overlap_ratio=neg_mention_overlap_ratio)

    # entities
    entities, entity_types = create_entities(doc, pos_mention_spans)

    # positive coreference pairs
    pos_coref_mention_pairs, pos_coref_spans, pos_eds = create_pos_coref_pairs(doc, pos_mention_spans)

    # negative coreference pairs
    neg_coref_mention_pairs, neg_coref_spans, neg_eds = create_neg_coref_pairs(doc, pos_mention_spans,
                                                                               neg_coref_count)

    # positive relations
    pos_rel_entity_pairs, pos_rel_types, rels_between_entities = create_pos_relations(doc, rel_type_count)

    (pos_rel_entity_pair_mp, pos_rel_mention_pair_ep, pos_rel_mention_pairs, pos_rel_ctx_masks,
     pos_rel_token_distances, pos_rel_sentence_distances) = create_rel_mention_pairs(doc, pos_rel_entity_pairs,
                                                                                     pos_mention_spans, context_size)

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities
    neg_rel_entity_pairs, neg_rel_types = create_neg_relations(entities, rels_between_entities,
                                                               rel_type_count, neg_rel_count)

    (neg_rel_entity_pair_mp, neg_rel_mention_pair_ep, neg_rel_mention_pairs, neg_rel_ctx_masks,
     neg_rel_token_distances, neg_rel_sentence_distances) = create_rel_mention_pairs(doc, neg_rel_entity_pairs,
                                                                                     pos_mention_spans, context_size,
                                                                                     offset_mp=len(
                                                                                         pos_rel_mention_pairs),
                                                                                     offset_ep=len(
                                                                                         pos_rel_entity_pairs))

    encodings, context_masks = create_context_tensors(encodings)

    mention_types, mention_masks, mention_sizes, mention_spans, mention_sample_masks = create_mention_tensors(
        context_size,
        pos_mention_spans,
        pos_mention_masks,
        pos_mention_sizes,
        neg_mention_spans,
        neg_mention_masks,
        neg_mention_sizes)

    (coref_mention_pairs, coref_types, coref_eds, coref_sample_masks) = create_coref_tensors(
        pos_coref_mention_pairs, pos_eds, neg_coref_mention_pairs, neg_eds)

    entities, entity_masks, entity_types, entity_sample_masks = create_entity_tensors(entities, entity_types)

    rel_entity_pairs, rel_types, rel_sample_masks = create_rel_global_tensors(pos_rel_entity_pairs, pos_rel_types,
                                                                              neg_rel_entity_pairs, neg_rel_types)

    (rel_entity_pair_mp, rel_mention_pair_ep, rel_mention_pairs, rel_ctx_masks, rel_pair_masks,
     rel_token_distances,
     rel_sentence_distances) = create_rel_mi_tensors(
        context_size,
        pos_rel_entity_pair_mp, pos_rel_mention_pair_ep,
        pos_rel_mention_pairs,
        pos_rel_ctx_masks,
        pos_rel_token_distances,
        pos_rel_sentence_distances,
        neg_rel_entity_pair_mp, neg_rel_mention_pair_ep,
        neg_rel_mention_pairs,
        neg_rel_ctx_masks,
        neg_rel_token_distances,
        neg_rel_sentence_distances)

    assert len(mention_masks) == len(mention_sizes) == len(mention_sample_masks) == len(mention_types)
    assert len(coref_mention_pairs) == len(coref_sample_masks) == len(coref_types) == len(coref_eds)
    assert len(entities) == len(entity_types)
    assert len(rel_entity_pairs) == len(rel_types)
    assert mention_sizes.max().item() < size_embeddings_count, f"You should increase the `size_embeddings_count` config.model section atleast to {mention_sizes.max()}"

    return dict(encodings=encodings, context_masks=context_masks, mention_masks=mention_masks,
                mention_sizes=mention_sizes, mention_types=mention_types, mention_sample_masks=mention_sample_masks,
                entities=entities, entity_masks=entity_masks, entity_types=entity_types,
                entity_sample_masks=entity_sample_masks,
                coref_mention_pairs=coref_mention_pairs, coref_types=coref_types,
                coref_eds=coref_eds, coref_sample_masks=coref_sample_masks,
                rel_entity_pairs=rel_entity_pairs, rel_types=rel_types, rel_types_evidence=rel_types,
                rel_sample_masks=rel_sample_masks,
                rel_entity_pair_mp=rel_entity_pair_mp, rel_mention_pair_ep=rel_mention_pair_ep,
                rel_mention_pairs=rel_mention_pairs, rel_ctx_masks=rel_ctx_masks, rel_pair_masks=rel_pair_masks,
                rel_token_distances=rel_token_distances, rel_sentence_distances=rel_sentence_distances)


def create_joint_inference_sample(doc, max_span_size: int):
    encodings = doc.encodings
    context_size = len(encodings)

    # create mention candidates
    (mention_masks, mention_sizes, mention_spans,
     mention_orig_spans, mention_sent_indices) = create_mention_candidates(doc, max_span_size, context_size)

    (mention_masks, mention_sizes, mention_spans,
     mention_orig_spans, mention_sent_indices, mention_sample_masks) = create_mention_candidate_tensors(context_size,
                                                                                                        mention_masks,
                                                                                                        mention_sizes,
                                                                                                        mention_spans,
                                                                                                        mention_orig_spans,
                                                                                                        mention_sent_indices)

    encodings, context_masks = create_context_tensors(encodings)

    return dict(encodings=encodings, context_masks=context_masks, mention_masks=mention_masks,
                mention_sizes=mention_sizes, mention_spans=mention_spans, mention_sample_masks=mention_sample_masks,
                mention_orig_spans=mention_orig_spans, mention_sent_indices=mention_sent_indices)
