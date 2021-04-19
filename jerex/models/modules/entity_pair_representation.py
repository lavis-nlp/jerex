import torch
from torch import nn as nn
from jerex import util


class EntityPairRepresentation(nn.Module):
    def __init__(self, hidden_size, entity_types, meta_embedding_size, prop_drop):
        super().__init__()

        self.entity_pair_linear = nn.Linear(hidden_size * 2 + meta_embedding_size * 2, hidden_size)
        self.entity_embeddings = nn.Embedding(entity_types, meta_embedding_size)

        self.dropout = nn.Dropout(prop_drop)

    def forward(self, entity_reprs, rel_entity_types, pairs):
        rel_entity_types = self.entity_embeddings(rel_entity_types)

        batch_size = pairs.shape[0]

        entity_pairs = util.batch_index(entity_reprs, pairs)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        rel_entity_types = rel_entity_types.view(rel_entity_types.shape[0], rel_entity_types.shape[1], -1)
        entity_pair_repr = self.entity_pair_linear(torch.cat([entity_pairs, rel_entity_types], dim=2))
        entity_pair_repr = self.dropout(torch.relu(entity_pair_repr))

        return entity_pair_repr


class EntityPairRepresentationCat(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, entity_reprs, pairs):
        batch_size = pairs.shape[0]

        entity_pairs = util.batch_index(entity_reprs, pairs)
        entity_pair_repr = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        return entity_pair_repr
