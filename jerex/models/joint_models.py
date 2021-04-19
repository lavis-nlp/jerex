from abc import abstractmethod

import torch
from transformers import BertConfig, BertTokenizer
from transformers import BertModel
from transformers import BertPreTrainedModel

from jerex import util
from jerex.evaluation.joint_evaluator import JointEvaluator
from jerex.loss import JointLoss
from jerex.task_types import TaskType
from jerex.models import misc
from jerex.models.modules.coreference_resolution import CoreferenceResolution
from jerex.models.modules.entity_classification import EntityClassification
from jerex.models.modules.entity_pair_representation import EntityPairRepresentation, EntityPairRepresentationCat
from jerex.models.modules.entity_representation import EntityRepresentation
from jerex.models.modules.mention_localization import MentionLocalization
from jerex.models.modules.mention_representation import MentionRepresentation
from jerex.models.modules.relation_classification_global import RelationClassificationGlobal
from jerex.models.modules.relation_classification_multi_instance import RelationClassificationMultiInstance


class JointBaseModel(BertPreTrainedModel):
    """ Base model of joint multi-instance and joint global models """

    def __init__(self, config: BertConfig, relation_types: int, entity_types: int,
                 meta_embedding_size: int, size_embeddings_count: int, ed_embeddings_count: int, prop_drop: float,
                 mention_threshold, coref_threshold, rel_threshold, tokenizer, *args, **kwargs):
        super(JointBaseModel, self).__init__(config)

        # Transformer model
        self.bert = BertModel(config, add_pooling_layer=False)

        self.mention_representation = MentionRepresentation()
        self.mention_localization = MentionLocalization(config.hidden_size, meta_embedding_size,
                                                        size_embeddings_count, prop_drop)
        self.coreference_resolution = CoreferenceResolution(config.hidden_size, meta_embedding_size,
                                                            ed_embeddings_count, prop_drop)
        self.entity_representation = EntityRepresentation(prop_drop)
        self.entity_classification = EntityClassification(config.hidden_size, entity_types, prop_drop)

        self._mention_threshold = mention_threshold
        self._coref_threshold = coref_threshold
        self._rel_threshold = rel_threshold

        self._relation_types = relation_types
        self._tokenizer = tokenizer

    def _forward_train_common(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                              mention_sizes: torch.tensor, entities: torch.tensor, entity_masks: torch.tensor,
                              coref_mention_pairs: torch.tensor, coref_eds, max_spans=None, max_coref_pairs=None,
                              **kwargs):
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()
        entity_masks = entity_masks.float()

        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']
        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)
        entity_reprs = self.entity_representation(mention_reprs, entities, entity_masks)

        mention_clf = self.mention_localization(mention_reprs, mention_sizes)
        entity_clf = self.entity_classification(entity_reprs)
        coref_clf = self.coreference_resolution(mention_reprs, coref_mention_pairs, coref_eds,
                                                max_pairs=max_coref_pairs)

        return h, mention_reprs, entity_reprs, mention_clf, entity_clf, coref_clf

    def _forward_inference_common(self, encodings: torch.tensor, context_masks: torch.tensor,
                                  mention_masks: torch.tensor,
                                  mention_sizes: torch.tensor, mention_spans: torch.tensor,
                                  mention_sample_masks: torch.tensor, max_spans=None, max_coref_pairs=None):
        # get contextualized token embeddings from last transformer layer
        context_masks = context_masks.float()
        mention_masks = mention_masks.float()
        mention_sample_masks = mention_sample_masks.float()

        # embed documents
        h = self.bert(input_ids=encodings, attention_mask=context_masks)['last_hidden_state']

        # get mention representations
        mention_reprs = self.mention_representation(h, mention_masks, max_spans=max_spans)

        # classify mentions
        mention_clf = self.mention_localization(mention_reprs, mention_sizes)
        valid_mentions = ((torch.sigmoid(mention_clf) >= self._mention_threshold).float() *
                          mention_sample_masks)

        # create mention pairs
        coref_mention_pairs, coref_mention_eds, coref_sample_masks = misc.create_coref_mention_pairs(
            valid_mentions, mention_spans, encodings, self._tokenizer)
        coref_sample_masks = coref_sample_masks.float()

        # classify coreferences
        coref_clf = self.coreference_resolution(mention_reprs, coref_mention_pairs, coref_mention_eds,
                                                max_pairs=max_coref_pairs)

        # create clusters
        clusters, clusters_sample_masks = misc.create_clusters(coref_clf, coref_mention_pairs,
                                                               coref_sample_masks,
                                                               valid_mentions, self._coref_threshold)
        entity_sample_masks = clusters_sample_masks.any(-1).float()

        # create entity representations
        entity_reprs = self.entity_representation(mention_reprs, clusters, clusters_sample_masks.float())

        # classify entities
        entity_clf = self.entity_classification(entity_reprs)

        return (h, mention_reprs, entity_reprs, clusters, entity_sample_masks, coref_sample_masks,
                clusters_sample_masks, mention_clf, entity_clf, coref_clf)

    def _apply_thresholds(self, mention_clf, coref_clf, entity_clf, rel_clf,
                          mention_sample_masks, coref_sample_masks, entity_sample_masks,
                          rel_sample_masks):
        mention_clf = torch.sigmoid(mention_clf)
        mention_clf[mention_clf < self._mention_threshold] = 0
        mention_clf *= mention_sample_masks.float()

        coref_clf = torch.sigmoid(coref_clf)
        coref_clf[coref_clf < self._coref_threshold] = 0
        coref_clf *= coref_sample_masks.float()

        entity_clf = torch.softmax(entity_clf, dim=-1)
        entity_clf *= entity_sample_masks.float().unsqueeze(-1)

        rel_clf = torch.sigmoid(rel_clf)
        rel_clf[rel_clf < self._rel_threshold] = 0
        rel_clf *= rel_sample_masks.float().unsqueeze(-1)

        return mention_clf, coref_clf, entity_clf, rel_clf

    @abstractmethod
    def _forward_train(self, *args, **kwargs):
        pass

    @abstractmethod
    def _forward_inference(self, *args, **kwargs):
        pass

    def forward(self, *args, inference=False, **kwargs):
        if not inference:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_inference(*args, **kwargs)


class JointMultiInstanceModel(JointBaseModel):
    """ Span-based model to jointly extract entity mentions, coreferences and relations using
    a multi-instance approach for entity and relation classification """

    TASK_TYPE = TaskType.JOINT
    LOSS = JointLoss
    MONITOR_METRIC = ('rel_nec', 'f1_micro')
    EVALUATOR = JointEvaluator

    def __init__(self, config: BertConfig, relation_types: int, entity_types: int,
                 meta_embedding_size: int, size_embeddings_count: int, ed_embeddings_count: int,
                 token_dist_embeddings_count: int, sentence_dist_embeddings_count: int, prop_drop: float,
                 mention_threshold: float, coref_threshold: float, rel_threshold: float, tokenizer: BertTokenizer,
                 *args, **kwargs):
        super(JointMultiInstanceModel, self).__init__(config, relation_types, entity_types,
                                                      meta_embedding_size, size_embeddings_count, ed_embeddings_count,
                                                      prop_drop,
                                                      mention_threshold, coref_threshold, rel_threshold, tokenizer)

        self.entity_classification = EntityClassification(config.hidden_size, entity_types, prop_drop)
        self.entity_pair_representation = EntityPairRepresentationCat(config.hidden_size)
        self.relation_classification = RelationClassificationMultiInstance(config.hidden_size, entity_types,
                                                                           relation_types, meta_embedding_size,
                                                                           token_dist_embeddings_count,
                                                                           sentence_dist_embeddings_count,
                                                                           prop_drop)

        # weight initialization
        self.init_weights()

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                       mention_sizes: torch.tensor, entities: torch.tensor, entity_masks: torch.tensor,
                       coref_mention_pairs: torch.tensor, rel_entity_pairs: torch.tensor,
                       rel_mention_pairs: torch.tensor, rel_ctx_masks: torch.tensor,
                       rel_entity_pair_mp: torch.tensor, rel_mention_pair_ep: torch.tensor,
                       rel_pair_masks: torch.tensor, rel_token_distances: torch.tensor,
                       rel_sentence_distances: torch.tensor, entity_types: torch.tensor, coref_eds: torch.tensor,
                       max_spans: bool = None, max_coref_pairs: bool = None, max_rel_pairs: bool = None, *args,
                       **kwargs):
        res = self._forward_train_common(encodings, context_masks, mention_masks, mention_sizes, entities, entity_masks,
                                         coref_mention_pairs, coref_eds, max_coref_pairs=max_coref_pairs,
                                         max_spans=max_spans)
        h, mention_reprs, entity_reprs, mention_clf, entity_clf, coref_clf = res

        entity_pair_reprs = self.entity_pair_representation(entity_reprs, rel_entity_pairs)

        rel_entity_types = util.batch_index(entity_types, rel_entity_pairs)
        rel_clf = self.relation_classification(entity_pair_reprs, h, mention_reprs,
                                               rel_entity_pair_mp, rel_mention_pair_ep,
                                               rel_mention_pairs, rel_ctx_masks, rel_pair_masks,
                                               rel_token_distances, rel_sentence_distances,
                                               rel_entity_types, max_pairs=max_rel_pairs)

        return dict(mention_clf=mention_clf, entity_clf=entity_clf, coref_clf=coref_clf, rel_clf=rel_clf)

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                           mention_sizes: torch.tensor, mention_spans: torch.tensor,
                           mention_sample_masks: torch.tensor, mention_sent_indices: torch.tensor,
                           mention_orig_spans: torch.tensor, max_spans: bool = None,
                           max_coref_pairs: bool = None, max_rel_pairs: bool = None, *args, **kwargs):
        res = self._forward_inference_common(encodings, context_masks,
                                             mention_masks, mention_sizes, mention_spans,
                                             mention_sample_masks, max_spans=max_spans, max_coref_pairs=max_coref_pairs)
        (h, mention_reprs, entity_reprs, clusters, entity_sample_masks, mention_pair_sample_masks,
         clusters_sample_masks, mention_clf, entity_clf, coref_clf) = res

        # create entity pairs
        (rel_entity_pair_mp, rel_mention_pair_ep, rel_entity_pairs,
         rel_mention_pairs, rel_ctx_masks, rel_token_distances,
         rel_sentence_distances, rel_mention_pair_masks) = misc.create_local_entity_pairs(clusters,
                                                                                  clusters_sample_masks,
                                                                                  mention_spans, mention_sent_indices,
                                                                                  mention_orig_spans,
                                                                                  context_masks.shape[-1])
        rel_sample_masks = rel_mention_pair_masks.any(dim=-1)

        # create entity pair representations
        entity_pair_reprs = self.entity_pair_representation(entity_reprs, rel_entity_pairs)

        # classify relations
        entity_types = entity_clf.argmax(dim=-1)
        rel_entity_types = util.batch_index(entity_types, rel_entity_pairs)
        rel_clf = self.relation_classification(entity_pair_reprs, h, mention_reprs,
                                               rel_entity_pair_mp, rel_mention_pair_ep,
                                               rel_mention_pairs, rel_ctx_masks, rel_mention_pair_masks,
                                               rel_token_distances, rel_sentence_distances, rel_entity_types,
                                               max_pairs=max_rel_pairs)

        # thresholding and masking
        mention_clf, coref_clf, entity_clf, rel_clf = self._apply_thresholds(mention_clf, coref_clf, entity_clf,
                                                                             rel_clf,
                                                                             mention_sample_masks,
                                                                             mention_pair_sample_masks,
                                                                             entity_sample_masks,
                                                                             rel_sample_masks)

        return dict(mention_clf=mention_clf, coref_clf=coref_clf, entity_clf=entity_clf, rel_clf=rel_clf,
                    clusters=clusters, clusters_sample_masks=clusters_sample_masks, rel_entity_pairs=rel_entity_pairs)


class JointGlobalModel(JointBaseModel):
    """ Span-based model to jointly extract entity mentions, coreferences and relations using
    global entity representations for relation classification """

    TASK_TYPE = TaskType.JOINT
    LOSS = JointLoss
    MONITOR_METRIC = ('rel_nec', 'f1_micro')
    EVALUATOR = JointEvaluator

    def __init__(self, config: BertConfig, relation_types: int, entity_types: int,
                 meta_embedding_size: int, size_embeddings_count: int,
                 ed_embeddings_count: int, prop_drop: float,
                 mention_threshold, coref_threshold, rel_threshold, tokenizer, *args, **kwargs):
        super(JointGlobalModel, self).__init__(config, relation_types, entity_types,
                                               meta_embedding_size, size_embeddings_count, ed_embeddings_count,
                                               prop_drop, mention_threshold, coref_threshold,
                                               rel_threshold, tokenizer)

        self.entity_pair_representation = EntityPairRepresentation(config.hidden_size, entity_types,
                                                                   meta_embedding_size, prop_drop)
        self.relation_classification = RelationClassificationGlobal(config.hidden_size, relation_types)

        # weight initialization
        self.init_weights()

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                       mention_sizes: torch.tensor, entities: torch.tensor, entity_masks: torch.tensor,
                       entity_types: torch.tensor, coref_mention_pairs: torch.tensor, rel_entity_pairs: torch.tensor,
                       coref_eds, max_spans=None, max_coref_pairs=None, *args, **kwargs):
        res = self._forward_train_common(encodings, context_masks, mention_masks, mention_sizes, entities, entity_masks,
                                         coref_mention_pairs, coref_eds, max_spans=max_spans,
                                         max_coref_pairs=max_coref_pairs)
        h, mention_reprs, entity_reprs, mention_clf, entity_clf, coref_clf = res

        rel_entity_types = util.batch_index(entity_types, rel_entity_pairs)
        entity_pair_reprs = self.entity_pair_representation(entity_reprs, rel_entity_types, rel_entity_pairs)
        rel_clf = self.relation_classification(entity_pair_reprs)

        return dict(mention_clf=mention_clf, entity_clf=entity_clf, coref_clf=coref_clf, rel_clf=rel_clf)

    def _forward_inference(self, encodings: torch.tensor, context_masks: torch.tensor, mention_masks: torch.tensor,
                           mention_sizes: torch.tensor, mention_spans: torch.tensor,
                           mention_sample_masks: torch.tensor, max_spans=None, max_coref_pairs=None, *args, **kwargs):
        res = self._forward_inference_common(encodings, context_masks,
                                             mention_masks, mention_sizes, mention_spans,
                                             mention_sample_masks, max_coref_pairs=max_coref_pairs, max_spans=max_spans)
        (h, mention_reprs, entity_reprs, clusters, entity_sample_masks, mention_pair_sample_masks,
         clusters_sample_masks, mention_clf, entity_clf, coref_clf) = res

        # create entity pairs
        rel_entity_pairs, rel_sample_masks = misc.create_rel_global_entity_pairs(entity_reprs, entity_sample_masks)
        rel_sample_masks = rel_sample_masks.float()

        # create entity pair representations
        entity_types = entity_clf.argmax(dim=-1)
        rel_entity_types = util.batch_index(entity_types, rel_entity_pairs)
        entity_pair_reprs = self.entity_pair_representation(entity_reprs, rel_entity_types, rel_entity_pairs)

        # classify relations
        rel_clf = self.relation_classification(entity_pair_reprs)

        # thresholding and masking
        mention_clf, coref_clf, entity_clf, rel_clf = self._apply_thresholds(mention_clf, coref_clf, entity_clf,
                                                                             rel_clf,
                                                                             mention_sample_masks,
                                                                             mention_pair_sample_masks,
                                                                             entity_sample_masks,
                                                                             rel_sample_masks)

        return dict(mention_clf=mention_clf, coref_clf=coref_clf, entity_clf=entity_clf, rel_clf=rel_clf,
                    clusters=clusters, clusters_sample_masks=clusters_sample_masks, rel_entity_pairs=rel_entity_pairs)
