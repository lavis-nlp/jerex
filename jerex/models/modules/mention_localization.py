import torch
from torch import nn as nn


class MentionLocalization(nn.Module):
    def __init__(self, hidden_size, meta_embedding_size, size_embeddings_count, prop_drop):
        super().__init__()

        self.linear = nn.Linear(hidden_size + meta_embedding_size, hidden_size)
        self.mention_classifier = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(prop_drop)
        self.size_embeddings = nn.Embedding(size_embeddings_count, meta_embedding_size)

    def forward(self, mention_reprs, mention_sizes):
        size_embeddings = self.size_embeddings(mention_sizes)

        # classify entity mentions
        mention_reprs = torch.cat([mention_reprs, size_embeddings], dim=2)
        mention_reprs = self.dropout(torch.relu(self.linear(mention_reprs)))
        mention_clf = self.mention_classifier(mention_reprs)
        mention_clf = mention_clf.squeeze(dim=-1)
        return mention_clf
