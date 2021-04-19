from torch import nn as nn

from jerex import util


class EntityRepresentation(nn.Module):
    def __init__(self, prop_drop):
        super().__init__()

        self.dropout = nn.Dropout(prop_drop)

    def forward(self, mention_reprs, entities, entity_masks):
        mention_clusters = util.batch_index(mention_reprs, entities)
        entity_masks = entity_masks.unsqueeze(-1)

        # max pool entity clusters
        m = (entity_masks == 0).float() * (-1e30)
        mention_spans_pool = mention_clusters + m
        entity_reprs = mention_spans_pool.max(dim=2)[0]
        entity_reprs = self.dropout(entity_reprs)

        return entity_reprs