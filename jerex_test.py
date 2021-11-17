import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from configs import TestConfig
from jerex import model, util

cs = ConfigStore.instance()
cs.store(name="test", node=TestConfig)


@hydra.main(config_name='test', config_path='configs/docred_joint')
def test(cfg: TestConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    util.config_to_abs_paths(cfg.dataset, 'test_path')
    util.config_to_abs_paths(cfg.model, 'model_path', 'tokenizer_path', 'encoder_config_path')
    util.config_to_abs_paths(cfg.misc, 'cache_path')

    model.test(cfg)


if __name__ == '__main__':
    test()
