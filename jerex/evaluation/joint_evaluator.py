import json
import os

import jinja2
import torch
from typing import List, Tuple, Dict

import jerex.evaluation.scoring
from jerex import util
from jerex.entities import Document
from jerex.evaluation import conversion, scoring
from jerex.evaluation.evaluator import Evaluator

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class JointEvaluator(Evaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def convert_batch(self, mention_clf, coref_clf, entity_clf, rel_clf,
                      clusters, clusters_sample_masks, rel_entity_pairs, batch: dict):
        batch_size = mention_clf.shape[0]

        return self._convert_documents(batch_size, mention_clf=mention_clf,
                                       entity_clf=entity_clf, rel_clf=rel_clf,
                                       clusters=clusters, clusters_sample_masks=clusters_sample_masks,
                                       rel_entity_pairs=rel_entity_pairs, mention_orig_spans=batch['mention_orig_spans'])

    def _convert_documents(self, batch_size, **kwargs):
        converted_batch = []
        for b in range(batch_size):
            converted_doc = self.__convert_document(**{k: v[b] for k, v in kwargs.items()})
            converted_batch.append(converted_doc)

        return converted_batch

    def __convert_document(self, mention_clf: torch.tensor, clusters, clusters_sample_masks,
                           entity_clf, rel_clf, rel_entity_pairs, mention_orig_spans):
        converted_mentions = conversion.convert_pred_mentions(mention_clf, mention_orig_spans)
        converted_clusters, converted_entities = conversion.convert_pred_clusters(mention_orig_spans, clusters,
                                                                                  clusters_sample_masks,
                                                                                  self._entity_types, entity_clf)
        converted_relations, converted_relations_et = conversion.convert_pred_relations(rel_clf, rel_entity_pairs,
                                                                                        converted_entities,
                                                                                        self._relation_types)

        return converted_mentions, converted_clusters, converted_entities, converted_relations, converted_relations_et

    def convert_gt(self, docs: List[Document]):
        converted_docs = []
        for doc in docs:
            doc_mentions = util.flatten([e.entity_mentions for e in doc.entities])
            gt_mentions = [mention.orig_span for mention in doc_mentions]

            gt_clusters = [conversion.convert_gt_cluster(e) for e in doc.entities]
            gt_entities = [conversion.convert_gt_cluster(e, True) for e in doc.entities]

            gt_relations = [conversion.convert_gt_relation(rel) for rel in doc.relations]
            gt_relations_et = [conversion.convert_gt_relation(rel, True) for rel in doc.relations]

            converted_docs.append((gt_mentions, gt_clusters, gt_entities, gt_relations, gt_relations_et))

        return converted_docs

    def compute_metrics(self, ground_truth, predictions):
        print("Evaluation")

        gt_mentions, gt_clusters, gt_entities, gt_relations, gt_relations_et = zip(*ground_truth)
        pred_mentions, pred_clusters, pred_entities, pred_relations, pred_relations_et = zip(*predictions)

        print("")
        print("--- Entity Mentions ---")
        # print("A mention is considered correct if the mention's span is predicted correctly")
        print("")
        mention_eval = scoring.score(gt_mentions, pred_mentions, print_results=True)

        print("")
        print("--- Clusters (Coreference Resolution) ---")
        # print("A cluster is considered correct if all member mention spans are predicted correctly")
        print("")
        coref_eval = scoring.score(gt_clusters, pred_clusters, print_results=True)

        print("")
        print("--- Entities ---")
        # print("An entity is considered correct if all mention spans and "
        #       "the entity type are predicted correctly")
        print("")
        entity_eval = scoring.score(gt_entities, pred_entities, type_idx=1, print_results=True)

        print("")
        print("--- Relations ---")
        print("Without entity type")
        # print("A relation is considered correct if the relation type and the spans of the two "
        #       "related entities are predicted correctly (entity type is not considered)")
        print("")
        rel_eval = scoring.score(gt_relations, pred_relations, type_idx=2, print_results=True)

        print("")
        print("With entity type")
        # print("A relation is considered correct if the relation type and the two "
        #       "related entities are predicted correctly (in span and entity type)")
        print("")
        rel_nec_eval = scoring.score(gt_relations_et, pred_relations_et, type_idx=2,
                                     print_results=True)

        return dict(mention=mention_eval, coref=coref_eval,
                    entity=entity_eval, rel=rel_eval,
                    rel_nec=rel_nec_eval)

    def store_predictions(self, predictions, documents, path):
        converted_predictions = []

        for doc_predictions, doc in zip(predictions, documents):
            doc_converted_predictions = dict()
            mentions, clusters, entities, relations, relations_et = doc_predictions

            doc_converted_predictions['tokens'] = [t.phrase for t in doc.tokens]
            doc_converted_predictions['mentions'] = mentions
            doc_converted_predictions['clusters'] = [[mentions.index(s) for s in c] for c in clusters]
            doc_converted_predictions['entities'] = [dict(cluster=clusters.index(e[0]),
                                                          type=e[1].identifier) for e in entities]
            doc_converted_predictions['relations'] = [dict(head=entities.index(r[0]),
                                                           tail=entities.index(r[1]),
                                                           type=r[2].identifier) for r in relations_et]

            converted_predictions.append(doc_converted_predictions)

        with open(path, 'w') as predictions_file:
            json.dump(converted_predictions, predictions_file)

    def store_examples(self, ground_truth, predictions, documents, path):
        gt_mentions, gt_clusters, gt_entities, gt_relations, gt_relations_et = zip(*ground_truth)
        pred_mentions, pred_clusters, pred_entities, pred_relations, pred_relations_et = zip(*predictions)

        example_docs = []

        for i, doc in enumerate(documents):
            # entities
            example_doc = self._convert_example(doc, gt_mentions[i], pred_mentions[i],
                                                gt_clusters[i], pred_clusters[i],
                                                gt_entities[i], pred_entities[i],
                                                gt_relations_et[i], pred_relations_et[i])
            example_docs.append(example_doc)

        self._store_examples(example_docs, path, template='joint_examples.html')

    def _convert_example(self, doc: Document, gt_mentions: List[Tuple], pred_mentions: List[Tuple],
                         gt_clusters: List[Tuple], pred_clusters: List[Tuple],
                         gt_entities: List[Tuple], pred_entities: List[Tuple],
                         gt_relations: List[Tuple], pred_relations: List[Tuple]):
        tokens = [t.phrase for t in doc.tokens]
        mention_tmp_args = self._get_tp_fn_fp(gt_mentions, pred_mentions,
                                              tokens, self._mention_to_html)
        cluster_tmp_args = self._get_tp_fn_fp(gt_clusters, pred_clusters,
                                              tokens, self._cluster_to_html)

        entity_tmp_args = self._get_tp_fn_fp(gt_entities, pred_entities,
                                             tokens, self._entity_to_html, type_idx=1)
        relation_tmp_args = self._get_tp_fn_fp(gt_relations, pred_relations,
                                               tokens, self._rel_to_html, type_idx=2)

        text = " ".join(tokens)
        return dict(mentions=mention_tmp_args, clusters=cluster_tmp_args,
                    entities=entity_tmp_args, relations=relation_tmp_args, text=text)

    def _store_examples(self, example_docs: List[Dict], file_path: str, template: str):
        template_path = os.path.join(SCRIPT_PATH, 'templates', template)

        # read template
        with open(os.path.join(SCRIPT_PATH, template_path)) as f:
            template = jinja2.Template(f.read())

        # write to disc
        template.stream(docs=example_docs).dump(file_path)

    def _get_tp_fn_fp(self, gt, pred, tokens, to_html, type_idx=None):
        if gt or pred:
            scores = jerex.evaluation.scoring.score_single(gt, pred, type_idx=type_idx)
        else:
            scores = dict(zip(jerex.evaluation.scoring.METRIC_LABELS, [100.] * 6))

        union = []
        for s in gt:
            if s not in union:
                union.append(s)

        for s in pred:
            if s not in union:
                union.append(s)

        # true positives
        tp = []
        # false negatives
        fn = []
        # false positives
        fp = []

        for s in union:
            type_verbose = s[type_idx].verbose_name if type_idx is not None else None

            if s in gt:
                if s in pred:
                    tp.append(dict(text=to_html(s, tokens), type=type_verbose, c='tp'))
                else:
                    fn.append(dict(text=to_html(s, tokens), type=type_verbose, c='fn'))
            else:
                fp.append(dict(text=to_html(s, tokens), type=type_verbose, c='fp'))

        return dict(results=tp + fp + fn, counts=dict(tp=len(tp), fp=len(fp), fn=len(fn)), scores=scores)

    def _mention_to_html(self, mention: Tuple, tokens: List[str]):
        start, end = mention[:2]

        tag_start = ' <span class="mention">'

        ctx_before = " ".join(tokens[:start])
        m = " ".join(tokens[start:end])
        ctx_after = " ".join(tokens[end:])

        html = ctx_before + tag_start + m + '</span> ' + ctx_after
        return html

    def _cluster_to_html(self, cluster: Tuple, tokens: List[str]):
        cluster = list(cluster)
        cluster = sorted(cluster)

        tag_start = ' <span class="mention">'
        html = ""

        last_end = None
        for mention in cluster:
            start, end = mention
            ctx_before = " ".join(tokens[last_end:start])
            m = " ".join(tokens[start:end])
            html += ctx_before + tag_start + m + '</span> '
            last_end = end

        html += " ".join(tokens[cluster[-1][1]:])
        return html

    def _entity_to_html(self, entity: Tuple, tokens: List[str]):
        cluster, entity_type = entity
        cluster = list(cluster)
        cluster = sorted(cluster)

        tag_start = ' <span class="mention">'
        html = ""

        last_end = None
        for mention in cluster:
            start, end = mention
            ctx_before = " ".join(tokens[last_end:start])
            m = " ".join(tokens[start:end])
            html += ctx_before + tag_start + m + '</span> '
            last_end = end

        html += " ".join(tokens[cluster[-1][1]:])
        return html

    def _rel_to_html(self, relation: Tuple, tokens: List[str]):
        head, tail, rel_type = relation

        mentions = []
        head_cluster, head_entity_type = head
        tail_cluster, tail_entity_type = tail
        head_cluster, tail_cluster = list(head_cluster), list(tail_cluster)
        for h in head_cluster:
            mentions.append((h[0], h[1], 'h'))
        for t in tail_cluster:
            mentions.append((t[0], t[1], 't'))

        mentions = sorted(mentions)

        head_tag = ' <span class="head"><span class="type">%s</span>' % head_entity_type
        tail_tag = ' <span class="tail"><span class="type">%s</span>' % tail_entity_type
        html = ""

        last_end = None
        for mention in mentions:
            start, end, h_or_t = mention
            ctx_before = " ".join(tokens[last_end:start])
            m = " ".join(tokens[start:end])
            html += ctx_before + (head_tag if h_or_t == 'h' else tail_tag) + m + '</span> '
            last_end = end

        html += " ".join(tokens[mentions[-1][1]:])
        return html
