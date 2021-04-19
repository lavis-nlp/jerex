from torch import nn as nn


class RelationClassificationGlobal(nn.Module):
    def __init__(self, hidden_size, relation_types):
        super().__init__()

        self.rel_classifier = nn.Linear(hidden_size, relation_types)

    def forward(self, entity_pair_reprs):
        rel_clf = self.rel_classifier(entity_pair_reprs)

        return rel_clf
