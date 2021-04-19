import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import TrainConfig
from jerex import model, util

cs = ConfigStore.instance()
cs.store(name="train", node=TrainConfig)


@hydra.main(config_name='train', config_path='configs')
def train(cfg: TrainConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.datasets, 'train_path', 'valid_path', 'test_path', 'types_path')
    util.config_to_abs_paths(cfg.model, 'tokenizer_path', 'encoder_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')

    model.train(cfg)




if __name__ == '__main__':
    train()
