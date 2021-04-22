import os
import pickle
from multiprocessing import Lock

import pytorch_lightning as pl
import torch
import transformers
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AdamW, BertConfig, BertTokenizer

from configs import TrainConfig, TestConfig
from jerex import models, util
from jerex.data_module import DocREDDataModule

_predictions_write_lock = Lock()


class JEREXModel(pl.LightningModule):
    """ Implements the training, validation and testing routines of JEREX. """
    def __init__(self, model_type: str, tokenizer_path: str, encoder_path: str = None,
                 encoder_config_path: str = None, cache_path: str = None, lowercase: bool = False,
                 entity_types: dict = None, relation_types: dict = None,
                 prop_drop: float = 0.1,
                 meta_embedding_size: int = 25,
                 size_embeddings_count: int = 10,
                 ed_embeddings_count: int = 300,
                 token_dist_embeddings_count: int = 700,
                 sentence_dist_embeddings_count: int = 50,
                 mention_threshold: float = 0.5, coref_threshold: float = 0.5, rel_threshold: float = 0.5,
                 mention_weight: float = 1, entity_weight: float = 1, coref_weight: float = 1,
                 relation_weight: float = 1,
                 lr: float = 5e-5, lr_warmup: float = 0.1, weight_decay: float = 0.01,
                 position_embeddings_count: int = 700,
                 max_spans_train: int = None, max_spans_inference: int = None,
                 max_coref_pairs_train: int = None, max_coref_pairs_inference: int = None,
                 max_rel_pairs_train: int = None, max_rel_pairs_inference: int = None,
                 examples_filename: str = 'examples_test.html',
                 store_examples=True, store_predictions=True, predictions_filename='predictions.json',
                 tmp_predictions_filename='.predictions_tmp.json', **kwargs):
        super().__init__()

        self.save_hyperparameters()

        model_class = models.get_model(model_type)

        self._tokenizer = BertTokenizer.from_pretrained(tokenizer_path,
                                                        do_lower_case=lowercase,
                                                        cache_dir=cache_path)

        encoder_config = BertConfig.from_pretrained(encoder_config_path or encoder_path, cache_dir=cache_path)

        self.model = models.create_model(model_class, encoder_config=encoder_config, tokenizer=self._tokenizer,
                                         encoder_path=encoder_path, entity_types=entity_types,
                                         relation_types=relation_types,
                                         prop_drop=prop_drop, meta_embedding_size=meta_embedding_size,
                                         size_embeddings_count=size_embeddings_count,
                                         ed_embeddings_count=ed_embeddings_count,
                                         token_dist_embeddings_count=token_dist_embeddings_count,
                                         sentence_dist_embeddings_count=sentence_dist_embeddings_count,
                                         mention_threshold=mention_threshold, coref_threshold=coref_threshold,
                                         rel_threshold=rel_threshold,
                                         position_embeddings_count=position_embeddings_count,
                                         cache_path=cache_path)

        self._evaluator = model_class.EVALUATOR(entity_types, relation_types, self._tokenizer)

        task_weights = [mention_weight, coref_weight, entity_weight, relation_weight]  # loss weights of sub-components
        self._compute_loss = self.model.LOSS(task_weights=task_weights)

        self._lr = lr
        self._lr_warmup = lr_warmup
        self._weight_decay = weight_decay
        self._max_spans_train = max_spans_train
        self._max_spans_inference = max_spans_inference
        self._max_coref_pairs_train = max_coref_pairs_train
        self._max_coref_pairs_inference = max_coref_pairs_inference
        self._max_rel_pairs_train = max_rel_pairs_train
        self._max_rel_pairs_inference = max_rel_pairs_inference
        self._store_examples = store_examples
        self._store_predicitons = store_predictions

        # evaluation
        self._eval_valid_gt = None  # validation datasets converted for evaluation
        self._eval_test_gt = None  # test datasets converted for evaluation
        self._examples_filename = examples_filename
        self._predictions_filename = predictions_filename
        self._tmp_predictions_filename = tmp_predictions_filename

    def setup(self, stage):
        """ Setup is run once before training/testing starts """
        # depending on stage (training=fit or testing), convert ground truth for later evaluation
        if stage == 'fit':
            self._eval_valid_gt = self._evaluator.convert_gt(self.trainer.datamodule.valid_dataset.documents)
        elif stage == 'test':
            self._eval_test_gt = self._evaluator.convert_gt(self.trainer.datamodule.test_dataset.documents)

    def forward(self, inference=False, **batch):
        max_spans = self._max_spans_train if not inference else self._max_spans_inference
        max_coref_pairs = self._max_coref_pairs_train if not inference else self._max_coref_pairs_inference
        max_rel_pairs = self._max_rel_pairs_train if not inference else self._max_rel_pairs_inference

        outputs = self.model(**batch, max_spans=max_spans, max_coref_pairs=max_coref_pairs,
                             max_rel_pairs=max_rel_pairs, inference=inference)

        return outputs

    def training_step(self, batch, batch_idx):
        """ Implements a training step, i.e. calling of forward pass and loss computation """
        # this method is called by PL for every training step
        # the returned loss is optimized
        outputs = self(**batch)
        losses = self._compute_loss.compute(**outputs, **batch)
        loss = losses['loss']

        for tag, value in losses.items():
            self.log('train_%s' % tag, value.item())
        return loss

    def validation_step(self, batch, batch_idx):
        """ Implements a validation step, i.e. evaluation of validation dataset against ground truth """
        # this method is called by PL for every validation step
        # validation is run after every epoch (default)
        return self._inference(batch, batch_idx)

    def validation_epoch_end(self, outputs):
        """ Loads current epoch's validation set predictions from disk and computes validation metrics """
        # this method is called by PL after all validation steps have finished
        if self._do_eval():
            predictions = self._load_predictions()
            metrics = self._evaluator.compute_metrics(self._eval_valid_gt[:len(predictions)], predictions)

            # this metric is used to store the best model over epochs and later use it for testing
            score = metrics[self.model.MONITOR_METRIC[0]][self.model.MONITOR_METRIC[1]]
            self.log('valid_f1', score, sync_dist=self.trainer.use_ddp, sync_dist_op='max')

            self._delete_predictions()
        else:
            self.log('valid_f1', 0, sync_dist=self.trainer.use_ddp, sync_dist_op='max')

        self._barrier()

    def test_step(self, batch, batch_idx):
        """ Implements a test step, i.e. evaluation of test dataset against ground truth """
        return self._inference(batch, batch_idx)

    def test_epoch_end(self, outputs):
        """ Loads current epoch's test set predictions from disk and computes test metrics """
        if self._do_eval():
            predictions = self._load_predictions()

            # compute evaluation metrics
            metrics = self._evaluator.compute_metrics(self._eval_test_gt, predictions)

            # log metrics
            for task, metrics in metrics.items():
                for metric_name, metric_value in metrics.items():
                    self.log(f'{task}_{metric_name}', metric_value)

            if self._store_examples:
                docs = self.trainer.datamodule.test_dataset.documents
                self._evaluator.store_examples(self._eval_test_gt, predictions, docs, self._examples_filename)

            if self._store_predicitons:
                docs = self.trainer.datamodule.test_dataset.documents
                self._evaluator.store_predictions(predictions, docs, self._predictions_filename)

            self._delete_predictions()

        self._barrier()

    def _inference(self, batch, batch_index):
        """ Converts prediction results of an epoch and stores the predictions on disk for later evaluation"""
        output = self(**batch, inference=True)

        # evaluate batch
        predictions = self._evaluator.convert_batch(**output, batch=batch)

        # save predictions to disk
        with _predictions_write_lock:
            for doc_id, doc_predictions in zip(batch['doc_ids'], predictions):
                res = dict(doc_id=doc_id.item(), predictions=doc_predictions)
                with open(self._tmp_predictions_filename, 'ab+') as fp:
                    pickle.dump(res, fp)

    def configure_optimizers(self):
        """ Created and configures optimizer and learning rate schedule """
        # this method is called once by PL before training starts
        optimizer_params = self._get_optimizer_params()
        optimizer = AdamW(optimizer_params, lr=self._lr, weight_decay=self._weight_decay)

        dataloader = self.train_dataloader()

        train_doc_count = len(dataloader)
        updates_epoch = train_doc_count // dataloader.batch_size
        updates_total = updates_epoch * self.trainer.max_epochs

        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=int(self._lr_warmup * updates_total),
                                                                 num_training_steps=updates_total)
        return [optimizer], [{'scheduler': scheduler, 'name': 'learning_rate', 'interval': 'step', 'frequency': 1}]

    def _get_optimizer_params(self):
        """ Get parameters to optimize """
        param_optimizer = list(self.model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_params = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': self._weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

        return optimizer_params

    def _do_eval(self):
        """ Waits for all processes validation/testing end of epoch and
        decides evaluation process (with global rank 0)  """
        eval_proc = True
        self._barrier()

        if self.global_rank != 0:
            eval_proc = False

        return eval_proc

    def _barrier(self):
        """ When using ddp as accelerator, lets processes wait till all processes passed barrier """
        if self.trainer.use_ddp:
            torch.distributed.barrier(torch.distributed.group.WORLD)

    def _load_predictions(self):
        """ Load current epoch predictions from disk for evaluation """
        predictions = []
        with open(self._tmp_predictions_filename, 'rb') as fr:
            try:
                while True:
                    predictions.append(pickle.load(fr))
            except EOFError:
                pass

        predictions = sorted(predictions, key=lambda p: p['doc_id'])
        predictions = [p['predictions'] for p in predictions]
        return predictions

    def _delete_predictions(self):
        os.remove(self._tmp_predictions_filename)


def train(cfg: TrainConfig):
    """ Loads datasets, builds model and creates trainer for JEREX training"""
    if cfg.misc.seed is not None:
        pl.seed_everything(cfg.misc.seed)

    model_class = models.get_model(cfg.model.model_type)

    tokenizer = BertTokenizer.from_pretrained(cfg.model.tokenizer_path, do_lower_case=cfg.sampling.lowercase,
                                              cache_dir=cfg.misc.cache_path)

    # read datasets
    data_module = DocREDDataModule(types_path=cfg.datasets.types_path, tokenizer=tokenizer,
                                   task_type=model_class.TASK_TYPE,
                                   train_path=cfg.datasets.train_path,
                                   valid_path=cfg.datasets.valid_path,
                                   test_path=cfg.datasets.test_path,
                                   train_batch_size=cfg.training.batch_size,
                                   valid_batch_size=cfg.inference.valid_batch_size,
                                   test_batch_size=cfg.inference.test_batch_size,
                                   neg_mention_count=cfg.sampling.neg_mention_count,
                                   neg_relation_count=cfg.sampling.neg_relation_count,
                                   neg_coref_count=cfg.sampling.neg_coref_count,
                                   max_span_size=cfg.sampling.max_span_size,
                                   neg_mention_overlap_ratio=cfg.sampling.neg_mention_overlap_ratio)

    data_module.setup('fit')

    model = JEREXModel(model_type=cfg.model.model_type, encoder_path=cfg.model.encoder_path,
                       tokenizer_path=cfg.model.tokenizer_path,
                       cache_path=cfg.misc.cache_path,
                       lowercase=cfg.sampling.lowercase,
                       entity_types=data_module.entity_types,
                       relation_types=data_module.relation_types,
                       prop_drop=cfg.model.prop_drop,
                       meta_embedding_size=cfg.model.meta_embedding_size,
                       size_embeddings_count=cfg.model.size_embeddings_count,
                       ed_embeddings_count=cfg.model.ed_embeddings_count,
                       token_dist_embeddings_count=cfg.model.token_dist_embeddings_count,
                       sentence_dist_embeddings_count=cfg.model.sentence_dist_embeddings_count,
                       position_embeddings_count= cfg.model.position_embeddings_count,
                       mention_threshold=cfg.model.mention_threshold, coref_threshold=cfg.model.coref_threshold,
                       rel_threshold=cfg.model.rel_threshold,
                       mention_weight=cfg.loss.mention_weight, entity_weight=cfg.loss.entity_weight,
                       coref_weight=cfg.loss.coref_weight,
                       relation_weight=cfg.loss.relation_weight,
                       lr=cfg.training.lr, lr_warmup=cfg.training.lr_warmup, weight_decay=cfg.training.weight_decay,
                       max_spans_train=cfg.training.max_spans,
                       max_spans_inference=cfg.inference.max_spans,
                       max_coref_pairs_train=cfg.training.max_coref_pairs,
                       max_coref_pairs_inference=cfg.inference.max_coref_pairs,
                       max_rel_pairs_train=cfg.training.max_rel_pairs,
                       max_rel_pairs_inference=cfg.inference.max_rel_pairs,
                       max_span_size=cfg.sampling.max_span_size)

    checkpoint_callback = ModelCheckpoint(dirpath='checkpoint', mode='max', monitor='valid_f1')

    tb_logger = pl.loggers.TensorBoardLogger('.', 'tb')
    csv_logger = pl.loggers.CSVLogger('.', 'csv')

    trainer = pl.Trainer(callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
                         min_epochs=cfg.training.min_epochs, max_epochs=cfg.training.max_epochs,
                         logger=[tb_logger, csv_logger],
                         profiler=cfg.misc.profiler, gradient_clip_val=cfg.training.max_grad_norm,
                         gpus=cfg.distribution.gpus if cfg.distribution.gpus else None,
                         accelerator=cfg.distribution.accelerator, precision=cfg.misc.precision,
                         flush_logs_every_n_steps=cfg.misc.flush_logs_every_n_steps,
                         log_every_n_steps=cfg.misc.log_every_n_steps,
                         deterministic=cfg.misc.deterministic,
                         accumulate_grad_batches=cfg.training.accumulate_grad_batches,
                         prepare_data_per_node=cfg.distribution.prepare_data_per_node,
                         num_sanity_val_steps=0)

    trainer.fit(model, datamodule=data_module)

    if cfg.datasets.test_path is not None:
        # test
        data_module.setup('test')
        trainer.test(model, datamodule=data_module)


def test(cfg: TestConfig):
    """ Loads test dataset and model and creates trainer for JEREX testing """
    overrides = util.get_overrides_dict(mention_threshold=cfg.model.mention_threshold,
                                        coref_threshold=cfg.model.coref_threshold,
                                        rel_threshold=cfg.model.rel_threshold,
                                        cache_path=cfg.misc.cache_path)
    model = JEREXModel.load_from_checkpoint(cfg.model.model_path,
                                            tokenizer_path=cfg.model.tokenizer_path,
                                            encoder_config_path=cfg.model.encoder_config_path,
                                            max_spans_inference=cfg.inference.max_spans,
                                            max_coref_pairs_inference=cfg.inference.max_coref_pairs,
                                            max_rel_pairs_inference=cfg.inference.max_rel_pairs,
                                            encoder_path=None, **overrides)

    tokenizer = BertTokenizer.from_pretrained(model.hparams.tokenizer_path,
                                              do_lower_case=model.hparams.lowercase,
                                              cache_dir=model.hparams.cache_path)

    # read datasets
    model_class = models.get_model(model.hparams.model_type)
    data_module = DocREDDataModule(entity_types=model.hparams.entity_types,
                                   relation_types=model.hparams.relation_types,
                                   tokenizer=tokenizer, task_type=model_class.TASK_TYPE,
                                   test_path=cfg.dataset.test_path,
                                   test_batch_size=cfg.inference.test_batch_size,
                                   max_span_size=model.hparams.max_span_size)

    tb_logger = pl.loggers.TensorBoardLogger('.', 'tb')
    csv_logger = pl.loggers.CSVLogger('.', 'cv')

    trainer = pl.Trainer(logger=[tb_logger, csv_logger],
                         profiler="simple", gpus=cfg.distribution.gpus if cfg.distribution.gpus else None,
                         accelerator=cfg.distribution.accelerator, precision=cfg.misc.precision,
                         flush_logs_every_n_steps=cfg.misc.flush_logs_every_n_steps,
                         log_every_n_steps=cfg.misc.log_every_n_steps,
                         prepare_data_per_node=cfg.distribution.prepare_data_per_node)

    # test
    data_module.setup('test')
    trainer.test(model, datamodule=data_module)



