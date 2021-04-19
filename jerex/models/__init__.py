import torch
from torch import nn
from transformers import BertPreTrainedModel, BertConfig, BertTokenizer

from jerex.models.classification_models import RelClassificationMultiInstanceModel, \
    EntityClassificationModel, CoreferenceResolutionModel, MentionLocalizationModel, RelClassificationGlobal
from jerex.models.joint_models import JointGlobalModel, JointMultiInstanceModel

_MODELS = {
    # joint models
    'joint_multi_instance': JointMultiInstanceModel,
    'joint_global': JointGlobalModel,

    # sub-task specific models
    'mention_localization': MentionLocalizationModel,
    'coreference_resolution': CoreferenceResolutionModel,
    'entity_classification': EntityClassificationModel,
    'relation_classification_multi_instance': RelClassificationMultiInstanceModel,
    'relation_classification_global': RelClassificationGlobal,
}


def get_model(name):
    return _MODELS[name]


def create_model(model_class: BertPreTrainedModel, encoder_config: BertConfig, tokenizer: BertTokenizer,
                 encoder_path=None, entity_types: dict = None, relation_types: dict = None,
                 prop_drop: float = 0.1, meta_embedding_size: int = 25,
                 size_embeddings_count: int = 10, ed_embeddings_count: int = 300,
                 token_dist_embeddings_count: int = 700, sentence_dist_embeddings_count: int = 50,
                 mention_threshold: float = 0.5, coref_threshold: float = 0.5, rel_threshold: float = 0.5,
                 position_embeddings_count: int = 700, cache_path=None):
    params = dict(config=encoder_config,
                  # JEREX model parameters
                  cls_token=tokenizer.convert_tokens_to_ids('[CLS]'),
                  entity_types=len(entity_types),
                  relation_types=len(relation_types),
                  prop_drop=prop_drop,
                  meta_embedding_size=meta_embedding_size,
                  size_embeddings_count=size_embeddings_count,
                  ed_embeddings_count=ed_embeddings_count,
                  token_dist_embeddings_count=token_dist_embeddings_count,
                  sentence_dist_embeddings_count=sentence_dist_embeddings_count,
                  mention_threshold=mention_threshold,
                  coref_threshold=coref_threshold,
                  rel_threshold=rel_threshold,
                  tokenizer=tokenizer,
                  cache_dir=cache_path,
                  )

    if encoder_path is not None:
        model = model_class.from_pretrained(encoder_path, **params)
    else:
        model = model_class(**params)

    # conditionally increase position embedding count
    if encoder_config.max_position_embeddings < position_embeddings_count:
        old = model.bert.embeddings.position_embeddings

        new = nn.Embedding(position_embeddings_count, encoder_config.hidden_size)
        new.weight.data[:encoder_config.max_position_embeddings, :] = old.weight.data
        model.bert.embeddings.position_embeddings = new
        model.bert.embeddings.register_buffer("position_ids",
                                              torch.arange(position_embeddings_count).expand((1, -1)))

        encoder_config.max_position_embeddings = position_embeddings_count

    return model
