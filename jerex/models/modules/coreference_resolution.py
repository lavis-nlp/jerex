import torch
from torch import nn as nn
from jerex import util


class CoreferenceResolution(nn.Module):
    def __init__(self, hidden_size, meta_embedding_size, ed_embeddings_count, prop_drop):
        super().__init__()

        self.coref_linear = nn.Linear(hidden_size * 2 + meta_embedding_size, hidden_size)
        self.coref_classifier = nn.Linear(hidden_size, 1)

        self.coref_ed_embeddings = nn.Embedding(ed_embeddings_count, meta_embedding_size)

        self.dropout = nn.Dropout(prop_drop)

    def forward(self, mention_reprs, coref_mention_pairs, coref_eds, max_pairs=None):
        batch_size = coref_mention_pairs.shape[0]

        # classify corefs
        coref_clf = torch.zeros([batch_size, coref_mention_pairs.shape[1]]).to(self._device)

        # coref
        # obtain coref logits
        # chunk processing to reduce memory usage
        max_pairs = max_pairs if max_pairs is not None else coref_mention_pairs.shape[1]
        coref_eds = self.coref_ed_embeddings(coref_eds)
        for i in range(0, coref_mention_pairs.shape[1], max_pairs):
            chunk_corefs = coref_mention_pairs[:, i:i + max_pairs]
            chunk_coref_eds = coref_eds[:, i:i + max_pairs]
            chunk_coref_clf = self._classify_corefs(mention_reprs, chunk_corefs, chunk_coref_eds)
            coref_clf[:, i:i + max_pairs] = chunk_coref_clf

        return coref_clf

    def _classify_corefs(self, mention_reprs, coref_mention_pairs, coref_eds):
        batch_size = coref_mention_pairs.shape[0]

        # get pairs of entity mention representations
        mention_pairs1 = util.batch_index(mention_reprs, coref_mention_pairs)
        mention_pairs = mention_pairs1.view(batch_size, mention_pairs1.shape[1], -1)

        coref_repr = torch.cat([mention_pairs, coref_eds], dim=2)
        coref_repr = torch.relu(self.coref_linear(coref_repr))
        coref_repr = self.dropout(coref_repr)

        # classify coref candidates
        chunk_coref_logits = self.coref_classifier(coref_repr)
        chunk_coref_logits = chunk_coref_logits.squeeze(dim=-1)
        return chunk_coref_logits

    @property
    def _device(self):
        return self.coref_classifier.weight.device