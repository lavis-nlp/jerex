import torch
from torch import nn as nn

from jerex import util


class RelationClassificationMultiInstance(nn.Module):
    def __init__(self, hidden_size, entity_types, relation_types, meta_embedding_size,
                 token_dist_embeddings_count, sentence_dist_embeddings_count, prop_drop):
        super().__init__()

        self.pair_linear = nn.Linear(hidden_size * 5 + 2 * meta_embedding_size, hidden_size)
        self.rel_linear = nn.Linear(hidden_size + 2 * meta_embedding_size, hidden_size)
        self.rel_classifier = nn.Linear(hidden_size, relation_types)

        self.token_distance_embeddings = nn.Embedding(token_dist_embeddings_count, meta_embedding_size)
        self.sentence_distance_embeddings = nn.Embedding(sentence_dist_embeddings_count, meta_embedding_size)
        self.entity_type_embeddings = nn.Embedding(entity_types, meta_embedding_size)

        self.dropout = nn.Dropout(prop_drop)
        self._relation_types = relation_types

    def forward(self, entity_pair_reprs, h, mention_reprs, rel_entity_pair_mp, rel_mention_pair_ep,
                rel_mention_pairs, rel_ctx_masks, rel_pair_masks, rel_token_distances, rel_sentence_distances,
                rel_entity_types, max_pairs=None):
        batch_size = entity_pair_reprs.shape[0]

        # relations
        # obtain relation logits
        # chunk processing to reduce memory usage
        max_pairs = max_pairs if max_pairs is not None else rel_mention_pairs.shape[1]
        rel_mention_pair_reprs = torch.zeros([batch_size, rel_mention_pairs.shape[1], 768]).to(self._device)
        h_large = h.unsqueeze(1)
        h_large = h_large.repeat(1, max_pairs, 1, 1)
        for i in range(0, rel_mention_pairs.shape[1], max_pairs):
            # classify relation candidates
            chunk_rel_mention_pair_ep = rel_mention_pair_ep[:, i:i + max_pairs]
            chunk_rel_mention_pairs = rel_mention_pairs[:, i:i + max_pairs]
            chunk_rel_ctx_masks = rel_ctx_masks[:, i:i + max_pairs]
            chunk_rel_token_distances = rel_token_distances[:, i:i + max_pairs]
            chunk_rel_sentence_distances = rel_sentence_distances[:, i:i + max_pairs]

            chunk_h = h_large[:, :chunk_rel_ctx_masks.shape[1], :, :]
            chunk_rel_logits = self._create_mention_pair_representations(
                entity_pair_reprs, chunk_rel_mention_pair_ep, chunk_rel_mention_pairs,
                chunk_rel_ctx_masks,
                chunk_rel_token_distances,
                chunk_rel_sentence_distances, mention_reprs, chunk_h)

            rel_mention_pair_reprs[:, i:i + max_pairs, :] = chunk_rel_logits

        # classify relation candidates, get logits for each relation type per entity pair
        rel_clf = self._classify_relations(rel_mention_pair_reprs, rel_entity_pair_mp,
                                           rel_pair_masks, rel_entity_types)

        return rel_clf

    def _create_mention_pair_representations(self, entity_pair_reprs, chunk_rel_mention_pair_ep,
                                             rel_mention_pairs, rel_ctx_masks,
                                             rel_token_distances, rel_sentence_distances,
                                             mention_reprs, h):
        rel_token_distances = self.token_distance_embeddings(rel_token_distances)
        rel_sentence_distances = self.sentence_distance_embeddings(rel_sentence_distances)

        rel_mention_pair_reprs = util.batch_index(mention_reprs, rel_mention_pairs)

        s = rel_mention_pair_reprs.shape
        rel_mention_pair_reprs = rel_mention_pair_reprs.view(s[0], s[1], -1)

        # ctx max pooling
        m = ((rel_ctx_masks == 0).float() * (-1e30)).unsqueeze(-1)
        rel_ctx = m + h
        # max pooling
        rel_ctx, rel_ctx_indices = rel_ctx.max(dim=2)

        # set the context vector of neighboring or adjacent spans to zero
        rel_ctx[rel_ctx_masks.bool().any(-1) == 0] = 0

        entity_pair_reprs = util.batch_index(entity_pair_reprs, chunk_rel_mention_pair_ep)

        local_repr = torch.cat([rel_ctx, rel_mention_pair_reprs, entity_pair_reprs,
                                rel_token_distances, rel_sentence_distances], dim=2)

        local_repr = self.dropout(self.pair_linear(local_repr))

        return local_repr

    def _classify_relations(self, rel_mention_pair_reprs, rel_entity_pair_mp, rel_pair_masks, rel_entity_types):
        local_repr = util.batch_index(rel_mention_pair_reprs, rel_entity_pair_mp)

        local_repr += (rel_pair_masks.unsqueeze(-1) == 0).float() * (-1e30)
        local_repr = local_repr.max(dim=2)[0]

        rel_entity_types = self.entity_type_embeddings(rel_entity_types)
        rel_entity_types = rel_entity_types.view(rel_entity_types.shape[0], rel_entity_types.shape[1], -1)

        rel_repr = torch.cat([local_repr, rel_entity_types], dim=2)
        rel_repr = self.dropout(torch.relu(self.rel_linear(rel_repr)))

        # classify relation candidates
        rel_logits = self.rel_classifier(rel_repr)
        rel_logits = rel_logits.squeeze(dim=-1)

        return rel_logits

    @property
    def _device(self):
        return self.rel_classifier.weight.device
