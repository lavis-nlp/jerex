import torch
from torch import nn as nn


class EntityClassification(nn.Module):
    def __init__(self, hidden_size, entity_types, prop_drop):
        super().__init__()

        self.linear = nn.Linear(hidden_size, hidden_size)
        self.entity_classifier = nn.Linear(hidden_size, entity_types)
        self.dropout = nn.Dropout(prop_drop)

    def forward(self, entity_reprs):
        entity_reprs = self.dropout(torch.relu(self.linear(entity_reprs)))
        entity_clf = self.entity_classifier(entity_reprs)

        return entity_clf
