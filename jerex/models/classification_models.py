import torch
from transformers import BertConfig
from transformers import BertModel
from transformers import BertPreTrainedModel

from jerex import util
from jerex.evaluation.classification_evaluator import MentionClassificationEvaluator, RelClassificationEvaluator, \
    EntityClassificationEvaluator, CorefClassificationEvaluator
from jerex.loss import RelationClassificationLoss, MentionLocalizationLoss, EntityClassificationLoss, \
    CoreferenceResolutionLoss
from jerex.task_types import TaskType
from jerex.models import misc
from jerex.models.modules.coreference_resolution import CoreferenceResolution
from jerex.models.modules.entity_classification import EntityClassification
from jerex.models.modules.entity_pair_representation import EntityPairRepresentationCat, EntityPairRepresentation
from jerex.models.modules.entity_representation import EntityRepresentation
from jerex.models.modules.mention_localization import MentionLocalization
from jerex.models.modules.mention_representation import MentionRepresentation
from jerex.models.modules.relation_classification_global import RelationClassificationGlobal
from jerex.models.modules.relation_classification_multi_instance import RelationClassificationMultiInstance


class MentionLocalizationModel(BertPreTrainedModel):
    """ Mention localization model """

    TASK_TYPE = TaskType.MENTION_LOCALIZATION
    LOSS = MentionLocalizationLoss
    EVALUATOR = MentionClassificationEvaluator
    MONITOR_METRIC = ('mention', 'f1_micro')

    def __init__(self, config: BertConfig,
                 meta_embedding_size: int, size_embeddings_count: int, prop_drop: float,
                 mention_threshold: float, *args, **kwargs):
        super(MentionLocalizationModel, self).__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.mention_representation = MentionRepresentation()
        self.mention_localization = MentionLocalization(config.hidden_size, meta_embedding_size,
                                                        size_embeddings_count, prop_drop)

        self._mention_threshold = mention_threshold

        # weight initialization
        self.init_weights()

    def forward(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                mention_sizes: torch.tensor, mention_sample_masks: torch.tensor,
                max_spans=None, inference=False, **kwargs):
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()

        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)
        mention_clf = self.mention_localization(mention_reprs, mention_sizes)

        if inference:
            mention_clf = torch.sigmoid(mention_clf)
            mention_clf[mention_clf < self._mention_threshold] = 0
            mention_clf *= mention_sample_masks

        return dict(mention_clf=mention_clf)


class CoreferenceResolutionModel(BertPreTrainedModel):
    """ Coreference resolution model """

    TASK_TYPE = TaskType.COREFERENCE_RESOLUTION
    LOSS = CoreferenceResolutionLoss
    EVALUATOR = CorefClassificationEvaluator
    MONITOR_METRIC = ('coref', 'f1_micro')

    def __init__(self, config: BertConfig,
                 meta_embedding_size: int, ed_embeddings_count: int, prop_drop: float, coref_threshold: float, *args, **kwargs):
        super(CoreferenceResolutionModel, self).__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.mention_representation = MentionRepresentation()
        self.coreference_resolution = CoreferenceResolution(config.hidden_size, meta_embedding_size,
                                                            ed_embeddings_count, prop_drop)

        self._coref_threshold = coref_threshold

        # weight initialization
        self.init_weights()

    def forward(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                mention_sample_masks: torch.tensor, coref_mention_pairs: torch.tensor,
                coref_eds: torch.tensor, coref_sample_masks: torch.tensor,
                max_spans=None, max_coref_pairs=None, valid_mentions=None, inference=False, **kwargs):
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()
        coref_sample_masks = coref_sample_masks.float()

        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)

        coref_clf = self.coreference_resolution(mention_reprs, coref_mention_pairs, coref_eds, max_pairs=max_coref_pairs)

        if inference:
            if valid_mentions is None:
                valid_mentions = torch.ones(mention_sample_masks.shape, dtype=torch.float).to(context_masks.device)
                valid_mentions *= mention_sample_masks

            clusters, clusters_sample_masks = misc.create_clusters(coref_clf, coref_mention_pairs, coref_sample_masks,
                                                                   valid_mentions, self._coref_threshold)

            coref_clf = torch.sigmoid(coref_clf)
            coref_clf[coref_clf < self._coref_threshold] = 0
            coref_clf *= coref_sample_masks

            return dict(coref_clf=coref_clf, clusters=clusters, clusters_sample_masks=clusters_sample_masks)

        return dict(coref_clf=coref_clf)


class EntityClassificationModel(BertPreTrainedModel):
    """ DocRED Classification model """

    TASK_TYPE = TaskType.ENTITY_CLASSIFICATION
    LOSS = EntityClassificationLoss
    EVALUATOR = EntityClassificationEvaluator
    MONITOR_METRIC = ('entity', 'f1_micro')

    def __init__(self, config: BertConfig, prop_drop: float, entity_types: int, *args, **kwargs):
        super(EntityClassificationModel, self).__init__(config)

        # Transformer model
        self.bert = BertModel(config, add_pooling_layer=False)

        self.mention_representation = MentionRepresentation()

        self.entity_representation = EntityRepresentation(prop_drop)
        self.entity_classification = EntityClassification(config.hidden_size, entity_types, prop_drop)

        # weight initialization
        self.init_weights()

    def forward(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                mention_sizes: torch.tensor, entity_sample_masks: torch.tensor,
                entities: torch.tensor, entity_masks: torch.tensor, max_spans=None, inference=False, **kwargs):
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()

        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)

        entity_reprs = self.entity_representation(mention_reprs, entities, entity_masks)
        entity_clf = self.entity_classification(entity_reprs)

        if inference:
            entity_clf = torch.softmax(entity_clf, dim=-1)
            entity_clf *= entity_sample_masks.float().unsqueeze(-1)

        return dict(entity_clf=entity_clf)


class RelClassificationMultiInstanceModel(BertPreTrainedModel):
    """ Relation classification model using a multi-instance approach """

    TASK_TYPE = TaskType.RELATION_CLASSIFICATION
    LOSS = RelationClassificationLoss
    EVALUATOR = RelClassificationEvaluator
    MONITOR_METRIC = ('rel', 'f1_micro')

    def __init__(self, config: BertConfig, relation_types: int, entity_types: int, meta_embedding_size: int,
                 token_dist_embeddings_count: int,
                 sentence_dist_embeddings_count: int,
                 prop_drop: float, rel_threshold: float, *args, **kwargs):
        super(RelClassificationMultiInstanceModel, self).__init__(config)

        # Transformer model
        self.bert = BertModel(config, add_pooling_layer=False)

        self.mention_representation = MentionRepresentation()
        self.entity_representation = EntityRepresentation(prop_drop)
        self.entity_pair_representation = EntityPairRepresentationCat(config.hidden_size)
        self.relation_classification = RelationClassificationMultiInstance(config.hidden_size, entity_types,
                                                                           relation_types, meta_embedding_size,
                                                                           token_dist_embeddings_count,
                                                                           sentence_dist_embeddings_count,
                                                                           prop_drop)

        self._rel_threshold = rel_threshold
        self._relation_types = relation_types

        # weight initialization
        self.init_weights()

    def forward(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                entities: torch.tensor, entity_masks: torch.tensor,
                rel_entity_pairs: torch.tensor, rel_sample_masks: torch.tensor,
                rel_entity_pair_mp: torch.tensor, rel_mention_pair_ep: torch.tensor,
                rel_mention_pairs: torch.tensor, rel_ctx_masks: torch.tensor, rel_pair_masks: torch.tensor,
                rel_token_distances: torch.tensor, rel_sentence_distances: torch.tensor, entity_types: torch.tensor,
                max_spans: bool = None, max_rel_pairs: bool = None, inference: bool = False, *args, **kwargs):
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()
        entity_masks = entity_masks.float()

        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)
        entity_reprs = self.entity_representation(mention_reprs, entities, entity_masks)
        entity_pair_reprs = self.entity_pair_representation(entity_reprs, rel_entity_pairs)

        rel_entity_types = util.batch_index(entity_types, rel_entity_pairs)
        rel_clf = self.relation_classification(entity_pair_reprs, h, mention_reprs,
                                               rel_entity_pair_mp, rel_mention_pair_ep,
                                               rel_mention_pairs, rel_ctx_masks, rel_pair_masks,
                                               rel_token_distances, rel_sentence_distances, rel_entity_types,
                                               max_pairs=max_rel_pairs)

        if inference:
            rel_clf = torch.sigmoid(rel_clf)
            rel_clf[rel_clf < self._rel_threshold] = 0
            rel_clf *= rel_sample_masks.unsqueeze(-1)

        return dict(rel_clf=rel_clf)


class RelClassificationGlobal(BertPreTrainedModel):
    """ Relation classification model using global entity representations """

    TASK_TYPE = TaskType.RELATION_CLASSIFICATION
    LOSS = RelationClassificationLoss
    EVALUATOR = RelClassificationEvaluator
    MONITOR_METRIC = ('rel', 'f1_micro')

    def __init__(self, config: BertConfig, relation_types: int, entity_types: int, meta_embedding_size: int,
                 prop_drop: float, rel_threshold, *args, **kwargs):
        super(RelClassificationGlobal, self).__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.mention_representation = MentionRepresentation()
        self.entity_representation = EntityRepresentation(prop_drop)
        self.entity_pair_representation = EntityPairRepresentation(config.hidden_size, entity_types,
                                                                   meta_embedding_size, prop_drop)
        self.relation_classification = RelationClassificationGlobal(config.hidden_size, relation_types)

        self._rel_threshold = rel_threshold
        self._relation_types = relation_types

        # weight initialization
        self.init_weights()

    def forward(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                entities: torch.tensor, entity_masks: torch.tensor,
                entity_types: torch.tensor, rel_entity_pairs: torch.tensor, rel_sample_masks: torch.tensor,
                max_spans=None, inference=False, **kwargs):
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()
        entity_masks = entity_masks.float()

        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)
        entity_reprs = self.entity_representation(mention_reprs, entities, entity_masks)

        rel_entity_types = util.batch_index(entity_types, rel_entity_pairs)
        entity_pair_reprs = self.entity_pair_representation(entity_reprs, rel_entity_types, rel_entity_pairs)
        rel_clf = self.relation_classification(entity_pair_reprs)

        if inference:
            rel_clf = torch.sigmoid(rel_clf)
            rel_clf[rel_clf < self._rel_threshold] = 0
            rel_clf *= rel_sample_masks.unsqueeze(-1)

        return dict(rel_clf=rel_clf)
