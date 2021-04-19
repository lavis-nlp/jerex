# JEREX: "Joint Entity-Level Relation Extractor"
PyTorch code for JEREX: "Joint Entity-Level Relation Extractor". For a description of the model and experiments, see our paper "An End-to-end Model for Entity-level Relation Extraction using Multi-instance Learning": https://arxiv.org/abs/2102.05980 (accepted at EACL 2021).


## Setup
### Requirements
- Required
  - Python 3.7+
  - PyTorch (tested with version 1.8.1)
  - PyTorch Lightning (tested with version 1.2.7)
  - transformers (+sentencepiece, e.g. with 'pip install transformers[sentencepiece]', tested with version 4.5.1)
  - hydra-core (tested with version 1.0.6)
  - scikit-learn (tested with version 0.21.3)
  - tqdm (tested with version 4.43.0)
  - numpy (tested with version 1.18.1)
  - jinja2 (tested with version 2.11.3)


### Fetch data
Fetch end-to-end (joint) DocRED [1] dataset split. For the original DocRED split, see https://github.com/thunlp/DocRED
```
bash ./scripts/fetch_dataset.sh
```

Fetch model checkpoints (joint multi-instance model (end-to-end split) and relation classification multi-instance model (original split)):
```
bash ./scripts/fetch_models.sh
```

## Examples
(1) Train JEREX using the end-to-end split
```
python ./jerex_train.py
```

(2) Evaluate JEREX on the end-to-end split (you need to fetch the model first):
```
python ./jerex_test.py
```

## Hyperparameters
- The hyperparameters used in our paper are set as default. You can adjust hyperparameters and other configuration settings in the 'train.yaml' and 'test.yaml' under ./configs
- A brief explanation of available configuration settings can be found in './configs.py'
- Besides the main JEREX model ('joint_multi_instance') and the 'global' baseline ('joint_global') you can also train each sub-component ('mention_localization', 'coreference_resolution', 'entity_classification',
    'relation_classification_multi_instance', 'relation_classification_global') individually. Just set 'model.model_type' accordingly (e.g. 'model.model_type: joint_global')

## Training/Inference speed and memory consumption
Performing a search over token spans (and pairs of spans) in the input document (as in JEREX) can be quite (CPU/GPU) memory demanding. If you run into memory issues (i.e. crashing of training/inference), these settings may help:
- Changing precision from fp32 to fp16 ('misc.precision: 16') will lower memory consumption.
- 'training.max_spans'/'training.max_coref_pairs'/'training.max_rel_pairs' (or 'inference.max_spans'/'inference.max_coref_pairs'/'inference.max_rel_pairs'): 
These settings restrict the number of spans/mention pairs for coreference resolution/mention pairs for MI relation classification that are processed simultaneously. 
Setting these to a lower number reduces training/inference speed, but lowers memory consumption. 
- The default setting of maximum span size is quite large. 
If the entity mentions in your dataset are usually shorter than 10 tokens, you can restrict the span search to less tokens (by setting 'sampling.max_span_size')

## References
```
[1] Yuan Yao, Deming Ye, Peng Li, Xu Han, Yankai Lin,Zhenghao Liu, Zhiyuan Liu, Lixin Huang, Jie Zhou,and Maosong Sun. 2019.  DocRED: A Large-ScaleDocument-Level  Relation  Extraction  Dataset.InProceedings of the 57th Annual Meeting of the As-sociation for Computational Linguistics, pages 764–777, Florence, Italy. ACL.
```
