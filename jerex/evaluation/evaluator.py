import os

from abc import ABC, abstractmethod
from transformers import BertTokenizer
from typing import List

from jerex.entities import Document

SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__))


class Evaluator(ABC):
    def __init__(self, entity_types: dict, relation_types: dict, tokenizer: BertTokenizer):
        self._entity_types = {v.index: v for v in entity_types.values()}
        self._relation_types = {v.index: v for v in relation_types.values()}
        self._tokenizer = tokenizer

    @abstractmethod
    def convert_batch(self, *args, **kwargs):
        pass

    @abstractmethod
    def convert_gt(self, docs: List[Document]):
        pass

    @abstractmethod
    def compute_metrics(self, ground_truth, predictions):
        pass

    @abstractmethod
    def store_examples(self, ground_truth, predictions, documents, path):
        pass

    @abstractmethod
    def store_predictions(self, predictions, documents, path):
        pass
