import json
from collections import OrderedDict

import torch
from typing import List

from torch.utils.data import Dataset as TorchDataset
from tqdm import tqdm

from jerex import util
from jerex.entities import Relation, Document, Entity, Token, Sentence, EntityMention
from jerex.task_types import TaskType
from jerex.sampling import sampling_joint, sampling_classify


class DocREDDataset(TorchDataset):
    """ """
    TRAIN_MODE = 'train'
    INFERENCE_MODE = 'inference'

    def __init__(self, dataset_path, entity_types, relation_types, tokenizer, neg_mention_count=200,
                 neg_rel_count=200, neg_coref_count=200, max_span_size=10, neg_mention_overlap_ratio=0.5):
        self._dataset_path = dataset_path
        self._entity_types = entity_types
        self._relation_types = relation_types
        self._neg_mention_count = neg_mention_count
        self._neg_rel_count = neg_rel_count
        self._neg_coref_count = neg_coref_count
        self._max_span_size = max_span_size
        self._neg_mention_overlap_ratio = neg_mention_overlap_ratio
        self._tokenizer = tokenizer

        self._mode = DocREDDataset.TRAIN_MODE
        self._task = None

        self._documents = OrderedDict()
        self._entity_mentions = OrderedDict()
        self._entities = OrderedDict()
        self._relations = OrderedDict()

        # current ids
        self._doc_id = 0
        self._sid = 0
        self._rid = 0
        self._eid = 0
        self._meid = 0
        self._tid = 0

        self._parse_dataset(dataset_path)

    def switch_mode(self, mode):
        self._mode = mode

    def switch_task(self, task):
        self._task = task

    def _parse_dataset(self, dataset_path):
        documents = json.load(open(dataset_path))
        for document in tqdm(documents, desc="Parse dataset '%s'" % self._dataset_path):
            self._parse_document(document)

    def _parse_document(self, doc):
        title = doc['title']
        jsents = doc['sents']
        jrelations = doc['labels'] if 'labels' in doc else []
        jentities = doc['vertexSet']

        # parse tokens
        sentences, doc_encoding = self._parse_sentences(jsents)

        # parse entity mentions
        entities = self._parse_entities(jentities, sentences)

        # parse relations
        relations = self._parse_relations(jrelations, entities, sentences)

        # create document
        doc_tokens = util.flatten([s.tokens for s in sentences])
        self._create_document(doc_tokens, sentences, entities, relations, doc_encoding, title)

    def _parse_sentences(self, jsentences):
        sentences = []

        # full document sub-word encoding
        doc_encoding = []

        # parse tokens
        tok_doc_idx = 0
        for sidx, jtokens in enumerate(jsentences):
            sentence_tokens = []

            for tok_sent_idx, token_phrase in enumerate(jtokens):
                token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
                if not token_encoding:
                    token_encoding = [self._tokenizer.convert_tokens_to_ids('[UNK]')]
                span_start, span_end = (len(doc_encoding), len(doc_encoding) + len(token_encoding))

                token = self._create_token(tok_doc_idx, tok_sent_idx, span_start, span_end, token_phrase)

                sentence_tokens.append(token)
                doc_encoding += token_encoding

                tok_doc_idx += 1

            sentence = self._create_sentence(sidx, sentence_tokens)
            sentences.append(sentence)

        return sentences, doc_encoding

    def _parse_entities(self, jentities, sentences) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            mention_params = []
            for jentityMention in jentity:
                entity_type = self._entity_types[jentityMention['type']]
                start, end = jentityMention['pos'][0], jentityMention['pos'][1]

                # create entity mention
                sentence = sentences[jentityMention['sent_id']]
                tokens = sentence.tokens[start:end]._tokens
                phrase = " ".join([t.phrase for t in tokens])

                mention_params.append((entity_type, tokens, phrase, sentence))

            entity_type = mention_params[0][0]
            entity_phrase = mention_params[0][2]
            entity = self._create_entity(entity_type, entity_phrase)

            for _, tokens, phrase, sentence in mention_params:
                entity_mention = self._create_entity_mention(entity, tokens, sentence, phrase)

                entity.add_entity_mention(entity_mention)
                sentence.add_entity_mention(entity_mention)

            entities.append(entity)

        return entities

    def _parse_relations(self, jrelations, entities, sentences) -> List[Relation]:
        relations = []

        for jrelation in jrelations:
            relation_type = self._relation_types[jrelation['r']]

            head_idx = jrelation['h']
            tail_idx = jrelation['t']

            evidence = jrelation['evidence']
            evidence_sentences = [sentences[ev] for ev in evidence]

            # create relation
            head_entity = entities[head_idx]
            tail_entity = entities[tail_idx]

            relation = self._create_relation(relation_type, head_entity, tail_entity, evidence_sentences)
            relations.append(relation)

        return relations

    def _create_token(self, doc_index, sent_index, span_start, span_end, phrase) -> Token:
        token = Token(self._tid, doc_index, sent_index, span_start, span_end, phrase)
        self._tid += 1
        return token

    def _create_sentence(self, index: int, tokens: List[Token]) -> Sentence:
        sentence = Sentence(self._sid, index, tokens)
        self._sid += 1
        return sentence

    def _create_document(self, tokens, sentences, entities, relations, doc_encoding, title) -> Document:
        document = Document(self._doc_id, tokens, sentences, entities, relations, doc_encoding, title)
        self._documents[self._doc_id] = document
        self._doc_id += 1

        return document

    def _create_entity(self, entity_type, phrase) -> Entity:
        entity = Entity(self._eid, entity_type, phrase)
        self._entities[self._eid] = entity
        self._eid += 1
        return entity

    def _create_entity_mention(self, entity, tokens, sentence, phrase) -> EntityMention:
        mention = EntityMention(self._meid, entity, tokens, sentence, phrase)
        self._entity_mentions[self._meid] = mention
        self._meid += 1
        return mention

    def _create_relation(self, relation_type, head_entity, tail_entity, evidence_sentences) -> Relation:
        relation = Relation(self._rid, relation_type, head_entity, tail_entity, evidence_sentences)
        self._relations[self._rid] = relation
        self._rid += 1
        return relation

    def __len__(self):
        return len(self._documents)

    def __getitem__(self, index: int):
        doc = self._documents[index]

        if self._mode == DocREDDataset.TRAIN_MODE:
            if self._task == TaskType.JOINT:
                return sampling_joint.create_joint_train_sample(doc, self._neg_mention_count, self._neg_rel_count,
                                                                self._neg_coref_count,
                                                                self._max_span_size, self._neg_mention_overlap_ratio,
                                                                len(self._relation_types))
            elif self._task == TaskType.MENTION_LOCALIZATION:
                return sampling_classify.create_mention_classify_train_sample(doc, self._neg_mention_count,
                                                                              self._max_span_size,
                                                                              self._neg_mention_overlap_ratio)
            elif self._task == TaskType.COREFERENCE_RESOLUTION:
                return sampling_classify.create_coref_classify_train_sample(doc, self._neg_coref_count)
            elif self._task == TaskType.ENTITY_CLASSIFICATION:
                return sampling_classify.create_entity_classify_sample(doc)
            elif self._task == TaskType.RELATION_CLASSIFICATION:
                return sampling_classify.create_rel_classify_train_sample(doc, self._neg_rel_count,
                                                                          len(self._relation_types))
            else:
                raise Exception('Invalid task')

        elif self._mode == DocREDDataset.INFERENCE_MODE:
            if self._task == TaskType.JOINT:
                samples = sampling_joint.create_joint_inference_sample(doc, self._max_span_size)
            elif self._task == TaskType.MENTION_LOCALIZATION:
                samples = sampling_classify.create_mention_classify_inference_sample(doc, self._max_span_size)
            elif self._task == TaskType.COREFERENCE_RESOLUTION:
                samples = sampling_classify.create_coref_classify_inference_sample(doc)
            elif self._task == TaskType.ENTITY_CLASSIFICATION:
                samples = sampling_classify.create_entity_classify_sample(doc)
            elif self._task == TaskType.RELATION_CLASSIFICATION:
                samples = sampling_classify.create_rel_classify_inference_sample(doc)
            else:
                raise Exception('Invalid task')

            samples['doc_ids'] = torch.tensor(doc.doc_id, dtype=torch.long)
            return samples
        else:
            raise Exception('Invalid mode')

    @property
    def documents(self):
        return list(self._documents.values())

    @property
    def entity_mentions(self):
        return list(self._entity_mentions.values())

    @property
    def entities(self):
        return list(self._entities.values())

    @property
    def relations(self):
        return list(self._relations.values())

    @property
    def document_count(self):
        return len(self._documents)

    @property
    def entity_count(self):
        return len(self._entity_mentions)

    @property
    def relation_count(self):
        return len(self._relations)
