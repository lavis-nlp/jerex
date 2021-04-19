import torch
from torch import nn as nn


class MentionRepresentation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, h, mention_masks, max_spans=None):
        mention_reprs = torch.zeros([mention_masks.shape[0], mention_masks.shape[1],
                                     h.shape[-1]]).to(h.device)

        max_spans = max_spans if max_spans is not None else mention_masks.shape[1]
        h_large = h.unsqueeze(1).repeat(1, max_spans, 1, 1)
        for i in range(0, mention_masks.shape[1], max_spans):
            chunk_mention_masks = mention_masks[:, i:i + max_spans]
            chunk_mention_reprs = self._forward(chunk_mention_masks, h_large)
            mention_reprs[:, i:i + max_spans, :] = chunk_mention_reprs

        return mention_reprs

    def _forward(self, mention_masks, h):
        h = h[:, :mention_masks.shape[1], :, :]

        # max pool entity candidate spans
        m = (mention_masks.unsqueeze(-1) == 0).float() * (-1e30)
        mention_reprs = m + h
        mention_reprs = mention_reprs.max(dim=2)[0]

        return mention_reprs
