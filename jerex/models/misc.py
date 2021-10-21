import torch
from sklearn.cluster import AgglomerativeClustering

from jerex import util
from jerex.sampling import sampling_common


def create_coref_mention_pairs(valid_mentions, mention_spans, encodings, tokenizer):
    """ Creates pairs of of mentions associated edit distances and sample masks for coreference resolution """
    batch_size = valid_mentions.shape[0]

    batch_coref_pairs = []
    batch_coref_eds = []
    batch_coref_sample_masks = []

    for i in range(batch_size):
        pairs = []
        eds = []
        pair_sample_masks = []

        # get spans classified as mentions
        non_zero_indices = valid_mentions[i].nonzero().view(-1)
        non_zero_spans = mention_spans[i][non_zero_indices].tolist()
        non_zero_indices = non_zero_indices.tolist()

        sample_encodings = encodings[i]

        # create relations and masks
        for i1, s1 in zip(non_zero_indices, non_zero_spans):
            for i2, s2 in zip(non_zero_indices, non_zero_spans):
                if i1 != i2:
                    s1_phrase = tokenizer.decode(sample_encodings[s1[0]:s1[1]].tolist()).strip()
                    s2_phrase = tokenizer.decode(sample_encodings[s2[0]:s2[1]].tolist()).strip()
                    ed = util.get_edit_distance(s1_phrase, s2_phrase)

                    pairs.append((i1, i2))
                    eds.append(ed)
                    pair_sample_masks.append(1)

        if pairs:
            # case: more than two spans classified as entity mentions
            batch_coref_pairs.append(torch.tensor(pairs, dtype=torch.long))
            batch_coref_eds.append(torch.tensor(eds, dtype=torch.long))
            batch_coref_sample_masks.append(torch.tensor(pair_sample_masks, dtype=torch.bool))
        else:
            # case: no more than two spans classified as entity mentions
            batch_coref_pairs.append(torch.zeros([1, 2], dtype=torch.long))
            batch_coref_eds.append(torch.zeros([1], dtype=torch.long))
            batch_coref_sample_masks.append(torch.zeros([1], dtype=torch.bool))

    # stack
    batch_coref_pairs = util.padded_stack(batch_coref_pairs).to(valid_mentions.device)
    batch_coref_eds = util.padded_stack(batch_coref_eds).to(valid_mentions.device)
    batch_coref_sample_masks = util.padded_stack(batch_coref_sample_masks).to(valid_mentions.device)

    return batch_coref_pairs, batch_coref_eds, batch_coref_sample_masks


def create_rel_global_entity_pairs(batch_entity_reprs: torch.tensor, entity_sample_masks: torch.tensor):
    """ Creates pairs of entity clusters and associated pair sample masks for relation classification """
    batch_size = batch_entity_reprs.shape[0]

    batch_rel_entity_pairs = []
    batch_rel_sample_masks = []

    for i in range(batch_size):
        pairs = []
        pair_sample_masks = []

        sample_masks = entity_sample_masks[i]
        entity_reprs = batch_entity_reprs[i]

        for i1 in range(entity_reprs.shape[0]):
            for i2 in range(entity_reprs.shape[0]):
                if i1 != i2 and sample_masks[i1] and sample_masks[i2]:
                    pairs.append((i1, i2))
                    pair_sample_masks.append(1)

        if pairs:
            batch_rel_entity_pairs.append(torch.tensor(pairs, dtype=torch.long))
            batch_rel_sample_masks.append(torch.tensor(pair_sample_masks, dtype=torch.bool))
        else:
            batch_rel_entity_pairs.append(torch.zeros([1, 2], dtype=torch.long))
            batch_rel_sample_masks.append(torch.zeros([1], dtype=torch.bool))

    # stack
    batch_rel_entity_pairs = util.padded_stack(batch_rel_entity_pairs).to(batch_entity_reprs.device)
    batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks).to(batch_entity_reprs.device)

    return batch_rel_entity_pairs, batch_rel_sample_masks


def create_clusters(coref_clf: torch.tensor, mention_pairs: torch.tensor, pair_sample_mask: torch.tensor,
                    valid_mentions: torch.tensor, threshold: float):
    """ Creates entity clusters by complete linkage based on mention pair
    similarities from coreference resolution step """
    batch_size = coref_clf.shape[0]

    coref_clf = torch.sigmoid(coref_clf) * pair_sample_mask.float()

    batch_clusters = []
    batch_clusters_sample_masks = []

    for i in range(batch_size):
        doc_valid_mentions = valid_mentions[i].nonzero().view(-1).tolist()
        clusters = None

        if len(doc_valid_mentions) == 1:
            clusters = [[0]]
        elif doc_valid_mentions:
            # we only cluster valid mentions (according to mention localization step)
            # these must later (after clustering) be mapped back to indices in full mention candidate tensor
            mapping, mapping_rev = dict(), dict()
            for mention_idx in doc_valid_mentions:
                m_idx = len(mapping)
                mapping[mention_idx] = len(mapping)
                mapping_rev[m_idx] = mention_idx

            # create similarity matrix, initial high similarity of mentions to itself
            similarities = torch.zeros([len(mapping), len(mapping)])
            similarities = similarities.fill_diagonal_(1)

            # fill similarity matrix
            doc_pair_sample_mask = pair_sample_mask[i].bool()
            doc_coref_clf = coref_clf[i][doc_pair_sample_mask]
            doc_pairs = mention_pairs[i][doc_pair_sample_mask]

            for p, v in zip(doc_pairs, doc_coref_clf):
                similarities[mapping[p[0].item()], mapping[p[1].item()]] = v

            # similarities to distances
            distances = 1 - similarities

            # apply complete linkage clustering
            agg_clustering = AgglomerativeClustering(n_clusters=None, affinity='precomputed',
                                                     linkage='complete', distance_threshold=1 - threshold)
            assignments = agg_clustering.fit_predict(distances)

            # convert clusters to list
            mx = max(assignments)
            clusters = [[] for _ in range(mx + 1)]
            for mention_idx, cluster_index in enumerate(assignments):
                clusters[cluster_index].append(mapping_rev[mention_idx])  # map back

        # -> tensors
        if clusters:
            batch_clusters.append(util.padded_stack([torch.tensor(list(c), dtype=torch.long) for c in clusters]))
            sample_mask = util.padded_stack([torch.ones([len(c)], dtype=torch.bool) for c in clusters])
            batch_clusters_sample_masks.append(sample_mask)
        else:
            batch_clusters.append(torch.zeros([1, 1], dtype=torch.long))
            batch_clusters_sample_masks.append(torch.zeros([1, 1], dtype=torch.bool))

    # stack
    batch_clusters = util.padded_stack(batch_clusters).to(coref_clf.device)
    batch_clusters_sample_masks = util.padded_stack(batch_clusters_sample_masks).to(coref_clf.device)

    return batch_clusters, batch_clusters_sample_masks


def create_local_entity_pairs(batch_clusters, batch_cluster_sample_masks,
                              batch_mention_spans, batch_mention_sent_indices,
                              batch_mention_orig_spans, context_size):
    """ Creates pairs of of mentions associated edit distances and sample masks for coreference resolution """
    batch_size = batch_clusters.shape[0]

    rel_entity_pair_mp = []
    rel_mention_pair_ep = []
    rel_entity_pairs = []
    rel_mention_pairs = []
    rel_ctx_masks = []
    rel_token_distances = []
    rel_sentence_distances = []
    rel_mention_pair_masks = []

    for i in range(batch_size):
        clusters = batch_clusters[i]
        cluster_sample_masks = batch_cluster_sample_masks[i]
        cluster_sample_masks2 = cluster_sample_masks.any(-1)

        mention_spans = batch_mention_spans[i]
        mention_token_spans = batch_mention_orig_spans[i]
        mention_sent_indices = batch_mention_sent_indices[i]

        doc_rel_entity_pair_mp = []
        doc_rel_mention_pair_ep = []
        doc_rel_entity_pairs = []
        doc_rel_mention_pairs = []
        doc_rel_ctx_masks = []
        doc_rel_token_distances = []
        doc_rel_sentence_distances = []
        doc_rel_mention_pair_masks = []

        pair_idx = 0
        # for all pairs of entity clusters...
        for i1, c1 in enumerate(clusters):
            for i2, c2 in enumerate(clusters):
                if i1 != i2 and cluster_sample_masks2[i1] and cluster_sample_masks2[i2]:
                    entity_pair_mp = []

                    # ...pair all mentions of the two clusters
                    for j1, m1 in enumerate(c1):
                        for j2, m2 in enumerate(c2):
                            if cluster_sample_masks[i1, j1] and cluster_sample_masks[i2, j2]:
                                # ...and create necessary data for multi-instance relation classification
                                entity_pair_mp.append(len(doc_rel_mention_pairs))
                                doc_rel_mention_pair_ep.append(pair_idx)

                                # indices of mentions of pair
                                pair = (m1, m2)
                                doc_rel_mention_pairs.append(pair)

                                s1, s2 = mention_spans[m1], mention_spans[m2]
                                t1, t2 = mention_token_spans[m1], mention_token_spans[m2]
                                sent1, sent2 = mention_sent_indices[m1], mention_sent_indices[m2]

                                # mask to get context between mention pair
                                ctx_mask = sampling_common.create_rel_mask(s1, s2, context_size)
                                doc_rel_ctx_masks.append(ctx_mask)

                                # distance in sentences and tokens between mention pair
                                token_dist = sampling_common.get_mention_token_dist_tensors(t1, t2)
                                sentence_dist = abs(sent1 - sent2)
                                doc_rel_token_distances.append(token_dist)
                                doc_rel_sentence_distances.append(sentence_dist)

                    doc_rel_entity_pairs.append((i1, i2))
                    doc_rel_entity_pair_mp.append(entity_pair_mp)
                    doc_rel_mention_pair_masks.append(torch.ones(len(entity_pair_mp), dtype=torch.bool))

                    pair_idx += 1

        if doc_rel_mention_pairs:
            rel_entity_pairs.append(torch.tensor(doc_rel_entity_pairs, dtype=torch.long))
            rel_mention_pair_masks.append(util.padded_stack(doc_rel_mention_pair_masks))
            rel_entity_pair_mp.append(util.padded_stack([torch.tensor(e, dtype=torch.long)
                                                         for e in doc_rel_entity_pair_mp]))
            rel_mention_pair_ep.append(torch.tensor(doc_rel_mention_pair_ep, dtype=torch.long))
            rel_mention_pairs.append(torch.tensor(doc_rel_mention_pairs))
            rel_ctx_masks.append(torch.stack(doc_rel_ctx_masks))
            rel_token_distances.append(torch.stack(doc_rel_token_distances))
            rel_sentence_distances.append(torch.stack(doc_rel_sentence_distances))
        else:
            rel_entity_pairs.append(torch.zeros([1, 2], dtype=torch.long))
            rel_mention_pair_masks.append(torch.zeros([1, 1], dtype=torch.bool))
            rel_entity_pair_mp.append(torch.zeros([1, 1], dtype=torch.long))
            rel_mention_pair_ep.append(torch.zeros([1], dtype=torch.long))
            rel_mention_pairs.append(torch.zeros([1, 2], dtype=torch.long))
            rel_ctx_masks.append(torch.zeros([1, context_size], dtype=torch.bool))
            rel_token_distances.append(torch.zeros([1], dtype=torch.long))
            rel_sentence_distances.append(torch.zeros([1], dtype=torch.long))

    # stack
    rel_entity_pairs = util.padded_stack(rel_entity_pairs).to(batch_clusters.device)
    rel_mention_pair_masks = util.padded_stack(rel_mention_pair_masks).to(batch_clusters.device)
    rel_entity_pair_mp = util.padded_stack(rel_entity_pair_mp).to(batch_clusters.device)
    rel_mention_pair_ep = util.padded_stack(rel_mention_pair_ep).to(batch_clusters.device)
    rel_mention_pairs = util.padded_stack(rel_mention_pairs).to(batch_clusters.device)
    rel_ctx_masks = util.padded_stack(rel_ctx_masks).to(batch_clusters.device)
    rel_token_distances = util.padded_stack(rel_token_distances).to(batch_clusters.device)
    rel_sentence_distances = util.padded_stack(rel_sentence_distances).to(batch_clusters.device)

    return (rel_entity_pair_mp, rel_mention_pair_ep, rel_entity_pairs, rel_mention_pairs,
            rel_ctx_masks, rel_token_distances, rel_sentence_distances, rel_mention_pair_masks)
