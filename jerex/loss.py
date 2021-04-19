from abc import ABC

import torch


class Loss(ABC):
    def compute(self, *args, **kwargs):
        pass


class JointLoss(Loss):
    def __init__(self, task_weights=None):
        self._mention_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._entity_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self._coref_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
        self._task_weights = task_weights if task_weights else [1, 1, 1, 1]

    def compute(self, mention_clf, entity_clf, coref_clf, rel_clf, mention_types,
                entity_types, coref_types, rel_types, mention_sample_masks, entity_sample_masks,
                coref_sample_masks, rel_sample_masks, **kwargs):
        loss_dict = dict()

        losses = []

        # mention loss
        mention_clf = mention_clf.view(-1)
        mention_types = mention_types.view(-1).float()
        mention_sample_masks = mention_sample_masks.view(-1).float()

        mention_loss = self._mention_criterion(mention_clf, mention_types)
        mention_loss = (mention_loss * mention_sample_masks).sum() / mention_sample_masks.sum()

        losses.append(mention_loss)
        loss_dict['mention_loss'] = mention_loss

        # coref loss
        coref_sample_masks = coref_sample_masks.view(-1).float()
        coref_count = coref_sample_masks.sum()

        if coref_count.item() != 0:
            coref_clf = coref_clf.view(-1)
            coref_types = coref_types.view(-1).float()

            sample_coref_loss = self._coref_criterion(coref_clf, coref_types)
            coref_loss = (sample_coref_loss * coref_sample_masks).sum() / coref_count

            losses.append(coref_loss)
            loss_dict['coref_loss'] = coref_loss
        else:
            losses.append(0)

        # entity loss
        entity_clf = entity_clf.view(-1, entity_clf.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        sample_entity_loss = self._entity_criterion(entity_clf, entity_types)
        entity_loss = (sample_entity_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        losses.append(entity_loss)
        loss_dict['entity_loss'] = entity_loss

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        if rel_count.item() != 0:
            rel_clf = rel_clf.view(-1, rel_clf.shape[-1])
            rel_types = rel_types.view(-1, rel_types.shape[-1])

            rel_loss = self._rel_criterion(rel_clf, rel_types.float())
            rel_loss = rel_loss.sum(-1)
            rel_loss = (rel_loss * rel_sample_masks).sum() / rel_count

            losses.append(rel_loss)
            loss_dict['rel_loss'] = rel_loss
        else:
            losses.append(0)

        train_loss = sum([task_loss * weight for task_loss, weight in zip(losses, self._task_weights)])
        loss_dict['loss'] = train_loss

        return loss_dict


class MentionLocalizationLoss(Loss):
    def __init__(self, *args, **kwargs):
        self._criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def compute(self, mention_clf, mention_types, mention_sample_masks, **kwargs):
        loss_dict = dict()

        mention_clf = mention_clf.view(-1)
        mention_types = mention_types.view(-1).float()
        mention_sample_masks = mention_sample_masks.view(-1).float()

        train_loss = self._criterion(mention_clf, mention_types)
        train_loss = (train_loss * mention_sample_masks).sum() / mention_sample_masks.sum()

        loss_dict['loss'] = train_loss

        return loss_dict


class CoreferenceResolutionLoss(Loss):
    def __init__(self, *args, **kwargs):
        self._criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def compute(self, coref_clf, coref_types, coref_sample_masks, **kwargs):
        loss_dict = dict()

        coref_sample_masks = coref_sample_masks.view(-1).float()
        coref_count = coref_sample_masks.sum()

        coref_clf = coref_clf.view(-1)
        coref_types = coref_types.view(-1).float()

        train_loss = self._criterion(coref_clf, coref_types)
        train_loss = (train_loss * coref_sample_masks).sum() / coref_count

        loss_dict['loss'] = train_loss

        return loss_dict


class EntityClassificationLoss(Loss):
    def __init__(self, *args, **kwargs):
        self._criterion = torch.nn.CrossEntropyLoss(reduction='none')

    def compute(self, entity_clf, entity_types, entity_sample_masks, **kwargs):
        loss_dict = dict()

        entity_clf = entity_clf.view(-1, entity_clf.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        train_loss = self._criterion(entity_clf, entity_types)
        train_loss = (train_loss * entity_sample_masks).sum() / entity_sample_masks.sum()

        loss_dict['loss'] = train_loss

        return loss_dict


class RelationClassificationLoss(Loss):
    def __init__(self, *args, **kwargs):
        self._rel_criterion = torch.nn.BCEWithLogitsLoss(reduction='none')

    def compute(self, rel_clf, rel_types, rel_sample_masks, **kwargs):
        loss_dict = dict()

        # relation loss
        rel_sample_masks = rel_sample_masks.view(-1).float()
        rel_count = rel_sample_masks.sum()

        rel_clf = rel_clf.view(-1, rel_clf.shape[-1])
        rel_types = rel_types.view(-1, rel_types.shape[-1])

        train_loss = self._rel_criterion(rel_clf, rel_types.float())
        train_loss = train_loss.sum(-1)
        train_loss = (train_loss * rel_sample_masks).sum() / rel_count

        loss_dict['loss'] = train_loss

        return loss_dict
