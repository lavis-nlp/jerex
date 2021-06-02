from typing import List


class RelationType:
    def __init__(self, identifier, index, short_name, verbose_name, symmetric=False):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name
        self._symmetric = symmetric

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    @property
    def symmetric(self):
        return self._symmetric

    def __str__(self):
        return self._verbose_name

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, RelationType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class EntityType:
    def __init__(self, identifier, index, short_name, verbose_name):
        self._identifier = identifier
        self._index = index
        self._short_name = short_name
        self._verbose_name = verbose_name

    @property
    def identifier(self):
        return self._identifier

    @property
    def index(self):
        return self._index

    @property
    def short_name(self):
        return self._short_name

    @property
    def verbose_name(self):
        return self._verbose_name

    def __str__(self):
        return self._verbose_name

    def __repr__(self):
        return self.__str__()

    def __int__(self):
        return self._index

    def __eq__(self, other):
        if isinstance(other, EntityType):
            return self._identifier == other._identifier
        return False

    def __hash__(self):
        return hash(self._identifier)


class Token:
    def __init__(self, tid: int, doc_index: int, sent_index: int, span_start: int, span_end: int, phrase: str):
        self._tid = tid  # ID within the corresponding dataset
        self._doc_index = doc_index  # original token index in document
        self._sent_index = sent_index  # original token index in sentence

        self._span_start = span_start  # start of token span in document (inclusive)
        self._span_end = span_end  # end of token span in document (exclusive)
        self._phrase = phrase

    @property
    def doc_index(self):
        return self._doc_index

    @property
    def sent_index(self):
        return self._sent_index

    @property
    def span_start(self):
        return self._span_start

    @property
    def span_end(self):
        return self._span_end

    @property
    def span(self):
        return self._span_start, self._span_end

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Token):
            return self._tid == other._tid
        return False

    def __hash__(self):
        return hash(self._tid)

    def __str__(self):
        return self._phrase

    def __repr__(self):
        return self._phrase


class TokenSpan:
    def __init__(self, tokens):
        self._tokens = tokens

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def orig_span_start(self):
        return self._tokens[0].doc_index

    @property
    def orig_span_end(self):
        return self._tokens[-1].doc_index + 1

    @property
    def orig_span(self):
        return self.orig_span_start, self.orig_span_end

    def __getitem__(self, s):
        if isinstance(s, slice):
            return TokenSpan(self._tokens[s.start:s.stop:s.step])
        else:
            return self._tokens[s]

    def __iter__(self):
        return iter(self._tokens)

    def __len__(self):
        return len(self._tokens)

    def __str__(self):
        return ' '.join([str(t) for t in self._tokens])

    def __repr__(self):
        return self.__str__()


class Entity:
    def __init__(self, eid: int, entity_type: EntityType, phrase: str):
        self._eid = eid
        self._entity_type = entity_type
        self._phrase = phrase

        self._entity_mentions = []

    def add_entity_mention(self, mention):
        self._entity_mentions.append(mention)

    @property
    def entity_type(self):
        return self._entity_type

    @property
    def entity_mentions(self):
        return self._entity_mentions

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, Entity):
            return self._eid == other._eid
        return False

    def __hash__(self):
        return hash(self._eid)

    def __str__(self):
        return self._phrase


class EntityMention:
    def __init__(self, emid: int, entity: Entity, tokens: List[Token], sentence: 'Sentence', phrase: str):
        self._emid = emid  # ID within the corresponding dataset

        self._entity = entity

        self._tokens = tokens
        self._sentence = sentence
        self._phrase = phrase

    @property
    def entity(self):
        return self._entity

    @property
    def entity_type(self):
        return self._entity.entity_type

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def orig_span_start(self):
        return self._tokens[0].doc_index

    @property
    def orig_span_end(self):
        return self._tokens[-1].doc_index + 1

    @property
    def orig_span(self):
        return self.orig_span_start, self.orig_span_end

    @property
    def sentence(self):
        return self._sentence

    @property
    def phrase(self):
        return self._phrase

    def __eq__(self, other):
        if isinstance(other, EntityMention):
            return self._emid == other._emid
        return False

    def __hash__(self):
        return hash(self._emid)

    def __str__(self):
        return self._phrase


class Sentence:
    def __init__(self, sent_id: int, index: int, tokens: List[Token]):
        self._sent_id = sent_id  # ID within the corresponding dataset
        self._index = index
        self._tokens = tokens
        self._entity_mentions = []

    def add_entity_mention(self, entity_mention):
        self._entity_mentions.append(entity_mention)

    @property
    def sent_id(self):
        return self._sent_id

    @property
    def index(self):
        return self._index

    @property
    def span_start(self):
        return self._tokens[0].span_start

    @property
    def span_end(self):
        return self._tokens[-1].span_end

    @property
    def span(self):
        return self.span_start, self.span_end

    @property
    def entity_mentions(self):
        return self._entity_mentions

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    def __str__(self):
        return ' '.join([str(t) for t in self.tokens])

    def __repr__(self):
        return self.__str__()

    def __eq__(self, other):
        if isinstance(other, Sentence):
            return self._sent_id == other._sent_id
        return False

    def __hash__(self):
        return hash(self._sent_id)


class Relation:
    def __init__(self, rid: int, relation_type: RelationType, head_entity: Entity, tail_entity: Entity,
                 evidence_sentences: List[Sentence]):
        self._rid = rid  # ID within the corresponding dataset
        self._relation_type = relation_type

        self._head_entity = head_entity
        self._tail_entity = tail_entity

        self._evidence_sentences = evidence_sentences

    @property
    def relation_type(self):
        return self._relation_type

    @property
    def head_entity(self):
        return self._head_entity

    @property
    def tail_entity(self):
        return self._tail_entity

    @property
    def evidence_sentences(self):
        return self._evidence_sentences

    def __eq__(self, other):
        if isinstance(other, Relation):
            return self._rid == other._rid
        return False

    def __hash__(self):
        return hash(self._rid)


class Document:
    def __init__(self, doc_id: int, tokens: List[Token], sentences: List[Sentence],
                 entities: List[Entity], relations: List[Relation], encoding: List[int], title: str):
        self._doc_id = doc_id  # ID within the corresponding dataset

        self._sentences = sentences
        self._tokens = tokens
        self._entities = entities
        self._relations = relations

        # sub-word document encoding
        self._encoding = encoding

        self._title = title

    @property
    def doc_id(self):
        return self._doc_id

    @property
    def sentences(self):
        return self._sentences

    @property
    def entities(self):
        return self._entities

    @property
    def relations(self):
        return self._relations

    @property
    def tokens(self):
        return TokenSpan(self._tokens)

    @property
    def encodings(self):
        return self._encoding

    @encodings.setter
    def encodings(self, value):
        self._encoding = value

    @property
    def title(self):
        return self._title

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return ' '.join([str(s) for s in self.sentences])

    def __eq__(self, other):
        if isinstance(other, Document):
            return self._doc_id == other._doc_id
        return False

    def __hash__(self):
        return hash(self._doc_id)
