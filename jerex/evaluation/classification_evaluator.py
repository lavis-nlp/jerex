import json
import os
from typing import List

import jerex.evaluation.scoring
from jerex import util
from jerex.entities import Document
from jerex.evaluation import conversion, scoring
from jerex.evaluation.evaluator import Evaluator

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class MentionLocalizationEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_batch(self, mention_clf, batch: dict):
        batch_size = mention_clf.shape[0]
        return self._convert_documents(batch_size, mention_clf=mention_clf,
                                       mention_orig_spans=batch['mention_orig_spans'])

    def _convert_documents(self, batch_size, **kwargs):
        converted_batch = []

        for b in range(batch_size):
            converted_mentions = conversion.convert_pred_mentions(**{k: v[b] for k, v in kwargs.items()})
            converted_batch.append(converted_mentions)

        return converted_batch

    def convert_gt(self, docs: List[Document]):
        converted_docs = []
        for i, doc in enumerate(docs):
            doc_mentions = util.flatten([e.entity_mentions for e in doc.entities])
            gt_mentions = [mention.orig_span for mention in doc_mentions]

            converted_docs.append(gt_mentions)

        return converted_docs

    def compute_metrics(self, ground_truth, predictions):
        print("Evaluation")

        print("")
        print("--- Mentions ---")
        print("")
        mention_eval = jerex.evaluation.scoring.score(ground_truth, predictions, print_results=True)

        return dict(mention=mention_eval)

    def store_predictions(self, predictions, documents, path):
        converted_predictions = [dict(mentions=doc_mentions) for doc_mentions in predictions]
        with open(path, 'w') as predictions_file:
            json.dump(converted_predictions, predictions_file)

    def store_examples(self, ground_truth, predictions, documents, path):
        pass


class CoreferenceResolutionEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_batch(self, coref_clf, clusters, clusters_sample_masks, batch: dict):
        batch_size = coref_clf.shape[0]
        return self._convert_documents(batch_size, clusters=clusters,
                                       clusters_sample_masks=clusters_sample_masks,
                                       mention_orig_spans=batch['mention_orig_spans'])

    def _convert_documents(self, batch_size, **kwargs):
        converted_batch = []

        for b in range(batch_size):
            converted_clusters = self.__convert_document(**{k: v[b] for k, v in kwargs.items()})
            converted_batch.append(converted_clusters)

        return converted_batch

    def __convert_document(self, clusters, clusters_sample_masks, mention_orig_spans):
        converted_clusters = conversion.convert_pred_clusters(mention_orig_spans, clusters,
                                                              clusters_sample_masks, self._entity_types)
        return converted_clusters

    def convert_gt(self, docs: List[Document]):
        converted_docs = []
        for doc in docs:
            gt_clusters = [conversion.convert_gt_cluster(e) for e in doc.entities]
            converted_docs.append(gt_clusters)

        return converted_docs

    def compute_metrics(self, ground_truth, predictions):
        print("Evaluation")

        print("")
        print("--- Clusters (Coreference Resolution) ---")
        print("")
        coref_eval = scoring.score(ground_truth, predictions, print_results=True)

        return dict(coref=coref_eval)

    def store_predictions(self, predictions, documents, path):
        converted_predictions = [dict(clusters=[list(c) for c in doc_clusters]) for doc_clusters in predictions]
        with open(path, 'w') as predictions_file:
            json.dump(converted_predictions, predictions_file)

    def store_examples(self, ground_truth, predictions, documents, path):
        pass


class EntityClassificationEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_batch(self, entity_clf, batch: dict):
        batch_size = entity_clf.shape[0]
        return self._convert_documents(batch_size, entity_clf=entity_clf,
                                       entity_sample_masks=batch['entity_sample_masks'])

    def _convert_documents(self, batch_size, **kwargs):
        converted_batch = []
        for b in range(batch_size):
            converted_entities = self._convert_pred_entities(**{k: v[b] for k, v in kwargs.items()})
            converted_batch.append(converted_entities)
        return converted_batch

    def _convert_pred_entities(self, entity_clf, entity_sample_masks):
        entity_types = []

        for i, (e, m) in enumerate(zip(entity_clf, entity_sample_masks)):
            if m:
                entity_type_idx = e.argmax().item()
                entity_type = self._entity_types[entity_type_idx]
                entity_types.append((i, entity_type))

        return entity_types

    def convert_gt(self, docs: List[Document]):
        converted_docs = []

        for doc in docs:
            gt_entities = [(i, entity.entity_type) for i, entity in enumerate(doc.entities)]
            converted_docs.append(gt_entities)

        return converted_docs

    def compute_metrics(self, ground_truth, predictions):
        print("Evaluation")

        print("")
        print("--- Entities ---")
        print("")
        entity_eval = scoring.score(ground_truth, predictions, type_idx=1, print_results=True)

        return dict(entity=entity_eval)

    def store_predictions(self, predictions, documents, path):
        converted_predictions = [[dict(entity=e[0], type=e[1].identifier)
                                  for e in doc_entities] for doc_entities in predictions]
        with open(path, 'w') as predictions_file:
            json.dump(converted_predictions, predictions_file)

    def store_examples(self, ground_truth, predictions, documents, path):
        pass


class RelClassificationEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_batch(self, rel_clf, batch: dict):
        batch_size = rel_clf.shape[0]
        return self._convert_documents(batch_size, rel_clf=rel_clf, entity_pairs=batch['rel_entity_pairs'])

    def _convert_documents(self, batch_size, **kwargs):
        converted_batch = []

        for b in range(batch_size):
            converted_relations = self._convert_pred_relations(**{k: v[b] for k, v in kwargs.items()})
            converted_batch.append(converted_relations)
        return converted_batch

    def _convert_pred_relations(self, rel_clf, entity_pairs):
        rel_entity_pairs, rel_types, rel_scores = conversion.convert_pred_relations_raw(rel_clf, entity_pairs)

        converted_relations = []

        for rel, rel_type_idx, score in zip(rel_entity_pairs, rel_types, rel_scores):
            rel_type_idx = self._relation_types[rel_type_idx]
            converted_relations.append((rel[0], rel[1], rel_type_idx))

        assert len(converted_relations) == len(set(converted_relations))
        return converted_relations

    def convert_gt(self, docs: List[Document]):
        converted_docs = []
        for doc in docs:
            gt_relations = [self._convert_gt_relation(doc, rel) for rel in doc.relations]
            converted_docs.append(gt_relations)

        return converted_docs

    def _convert_gt_relation(self, doc, relation):
        head_idx = doc.entities.index(relation.head_entity)
        tail_idx = doc.entities.index(relation.tail_entity)

        return head_idx, tail_idx, relation.relation_type

    def compute_metrics(self, ground_truth, predictions):
        print("Evaluation")

        print("")
        print("--- Relations ---")
        print("")
        rel_eval = scoring.score(ground_truth, predictions, type_idx=2, print_results=True)

        return dict(rel=rel_eval)

    def store_predictions(self, predictions, documents, path):
        results = []

        for i, doc in enumerate(documents):
            pred_relations = predictions[i]

            for r in pred_relations:
                result = dict(title=doc.title, h_idx=r[0], t_idx=r[1], r=r[2].short_name, evidence=[])
                results.append(result)

        with open(path, 'w') as predictions_file:
            json.dump(results, predictions_file)

    def store_examples(self, ground_truth, predictions, documents, path):
        pass
