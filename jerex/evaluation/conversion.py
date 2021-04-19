import torch


def convert_gt_cluster(entity, include_entity_type=False):
    t = set([m.orig_span for m in entity.entity_mentions])

    if include_entity_type:
        t = (t, entity.entity_type)

    return t


def convert_gt_relation(relation, include_entity_type=False):
    head = convert_gt_cluster(relation.head_entity, include_entity_type)
    tail = convert_gt_cluster(relation.tail_entity, include_entity_type)

    return head, tail, relation.relation_type


def convert_pred_mentions(mention_clf: torch.tensor, mention_orig_spans: torch.tensor):
    mention_nonzero = mention_clf.nonzero().view(-1)
    converted_mentions = [tuple(span) for span in mention_orig_spans[mention_nonzero].tolist()]
    assert len(converted_mentions) == len(set(converted_mentions))

    return converted_mentions


def convert_pred_clusters(mention_orig_spans, clusters, clusters_sample_masks, entity_types, entity_clf=None):
    return_entities = True
    if entity_clf is None:
        entity_clf = torch.ones((clusters.shape[0], 1)).to(mention_orig_spans.device)
        return_entities = False

    pred_clusters = []
    for cs, ms, e in zip(clusters.tolist(), clusters_sample_masks.tolist(), entity_clf):
        cluster = []

        for c, m in zip(cs, ms):
            if m:
                cluster.append(c)

        if cluster:
            entity_type_idx = e.argmax().item()
            entity_type = entity_types[entity_type_idx]
            pred_clusters.append((cluster, entity_type))

    converted_clusters = []
    converted_entities = []

    for c, entity_type in pred_clusters:
        pred_cluster = set()
        for m_idx in c:
            mention_span = tuple(mention_orig_spans[m_idx].tolist())
            pred_cluster.add(mention_span)

        converted_clusters.append(pred_cluster)
        converted_entities.append((pred_cluster, entity_type))

    if return_entities:
        return converted_clusters, converted_entities
    return converted_clusters


def convert_pred_relations(rel_clf: torch.tensor, rel_entity_pairs: torch.tensor,
                           converted_entities: list, relation_types):
    rel_entity_pairs, rel_types, rel_scores = convert_pred_relations_raw(rel_clf, rel_entity_pairs)

    converted_relations = []
    converted_relations_et = []

    for rel, rel_type, score in zip(rel_entity_pairs, rel_types, rel_scores):
        e1, e1_type = converted_entities[rel[0]]
        e2, e2_type = converted_entities[rel[1]]

        rel_type = relation_types[rel_type]
        converted_relations.append((e1, e2, rel_type))
        converted_relations_et.append(((e1, e1_type), (e2, e2_type), rel_type))

    return converted_relations, converted_relations_et


def convert_pred_relations_raw(rel_clf, entity_pairs):
    rel_class_count = rel_clf.shape[-1]

    # get predicted relation labels and corresponding entity pairs
    rel_clf = rel_clf.view(-1)
    rel_nonzero = rel_clf.nonzero().view(-1)
    rel_scores = rel_clf[rel_nonzero]

    rel_types = rel_nonzero % rel_class_count
    rel_indices = rel_nonzero // rel_class_count

    rels = entity_pairs[rel_indices]
    rels = rels.tolist()
    rel_types = rel_types.tolist()
    rel_scores = rel_scores.tolist()

    return rels, rel_types, rel_scores

