import json
from collections import OrderedDict

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from jerex.datasets import DocREDDataset
from jerex.entities import EntityType, RelationType
from jerex.sampling.sampling_common import collate_fn_padding


class DocREDDataModule(pl.LightningDataModule):
    """ Reads entity/relation type specification and manages datasets for training/validation/testing"""
    def __init__(self, tokenizer: BertTokenizer, task_type: str, types_path: str = None,
                 train_path: str = None, valid_path: str = None, test_path: str = None,
                 entity_types: dict = None, relation_types: dict = None,
                 train_batch_size: int = 1, valid_batch_size: int = 1, test_batch_size: int = 1,
                 sampling_processes: int = 4, neg_mention_count: int = 50,
                 neg_relation_count: int = 50, neg_coref_count: int = 50,
                 max_span_size: int = 10, neg_mention_overlap_ratio: float = 0.5,
                 final_valid_evaluate: bool = False,
                 size_embeddings_count: int = 30):
        super().__init__()

        if types_path is not None:
            # load types
            types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + relation types

            self._entity_types = OrderedDict()
            self._relation_types = OrderedDict()

            # entities
            for i, (key, v) in enumerate(types['entities'].items()):
                entity_type = EntityType(key, i, v['short'], v['verbose'])
                self._entity_types[key] = entity_type

            # relations
            for i, (key, v) in enumerate(types['relations'].items()):
                relation_type = RelationType(key, i, v['short'], v['verbose'], v['symmetric'])
                self._relation_types[key] = relation_type

        elif entity_types is not None and relation_types is not None:
            self._entity_types = entity_types
            self._relation_types = relation_types
        else:
            raise Exception('You must either specify types_path or entity_types+relation_types')

        self._tokenizer = tokenizer
        self._task_type = task_type
        self._train_batch_size = train_batch_size
        self._valid_batch_size = valid_batch_size
        self._test_batch_size = test_batch_size
        self._sampling_processes = sampling_processes
        self._neg_mention_count = neg_mention_count
        self._neg_relation_count = neg_relation_count
        self._neg_coref_count = neg_coref_count
        self._max_span_size = max_span_size
        self._neg_mention_overlap_ratio = neg_mention_overlap_ratio
        self._size_embeddings_count = size_embeddings_count

        self._train_path = train_path
        self._valid_path = valid_path
        self._test_path = test_path

        self._train_dataset = None
        self._valid_dataset = None
        self._test_dataset = None

        self._final_valid_evaluate = final_valid_evaluate

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            if self._train_path is not None:
                self._train_dataset = DocREDDataset(dataset_path=self._train_path,
                                                    entity_types=self._entity_types,
                                                    relation_types=self._relation_types,
                                                    neg_mention_count=self._neg_mention_count,
                                                    neg_coref_count=self._neg_coref_count,
                                                    neg_rel_count=self._neg_relation_count,
                                                    max_span_size=self._max_span_size,
                                                    neg_mention_overlap_ratio=self._neg_mention_overlap_ratio,
                                                    tokenizer=self._tokenizer,
                                                    size_embeddings_count=self._size_embeddings_count)

                self._train_dataset.switch_task(self._task_type)
                self._train_dataset.switch_mode(DocREDDataset.TRAIN_MODE)

            if self._valid_path is not None:
                self._valid_dataset = DocREDDataset(dataset_path=self._valid_path,
                                                    entity_types=self._entity_types,
                                                    relation_types=self._relation_types,
                                                    max_span_size=self._max_span_size,
                                                    tokenizer=self._tokenizer,
                                                    size_embeddings_count=self._size_embeddings_count)

                self._valid_dataset.switch_task(self._task_type)
                self._valid_dataset.switch_mode(DocREDDataset.INFERENCE_MODE)

        if (stage == 'test' or stage is None) and (self._test_path is not None or
                                                   (self._final_valid_evaluate is True and
                                                    self._valid_path is not None)):
            if self._test_path is not None:
                self._test_dataset = DocREDDataset(dataset_path=self._test_path,
                                                   entity_types=self._entity_types,
                                                   relation_types=self._relation_types,
                                                   max_span_size=self._max_span_size,
                                                   tokenizer=self._tokenizer,
                                                   size_embeddings_count=self._size_embeddings_count)
            else:
                self._test_dataset = self._valid_dataset

            self._test_dataset.switch_task(self._task_type)
            self._test_dataset.switch_mode(DocREDDataset.INFERENCE_MODE)

    def train_dataloader(self):
        return DataLoader(self._train_dataset, batch_size=self._train_batch_size, shuffle=False, drop_last=True,
                          num_workers=self._sampling_processes,
                          collate_fn=collate_fn_padding)

    def val_dataloader(self):
        return DataLoader(self._valid_dataset, batch_size=self._valid_batch_size, shuffle=False, drop_last=False,
                          num_workers=self._sampling_processes,
                          collate_fn=collate_fn_padding)

    def test_dataloader(self):
        return DataLoader(self._test_dataset, batch_size=self._test_batch_size, shuffle=False, drop_last=False,
                          num_workers=self._sampling_processes,
                          collate_fn=collate_fn_padding)

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def valid_dataset(self):
        return self._valid_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def relation_types(self):
        return self._relation_types

    @property
    def relation_type_count(self):
        return len(self._relation_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)
