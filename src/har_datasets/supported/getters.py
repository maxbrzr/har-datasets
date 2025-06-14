from omegaconf import OmegaConf
from hydra import initialize, compose

from har_datasets.dataset.har_dataset import HARDataset
from har_datasets.config.config import HAR_DATASETS_DICT, DatasetId, HARConfig


def get_har_cfg(
    dataset_id: DatasetId, config_dir: str = "../../../config"
) -> HARConfig:
    # load dataset-specific config from dir
    with initialize(version_base=None, config_path=config_dir):
        cfg = compose(config_name="cfg", overrides=[f"dataset={dataset_id.value}"])
        cfg = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        cfg = HARConfig(**cfg)  # type: ignore

    assert isinstance(cfg, HARConfig)

    return cfg


def get_har_dataset(cfg) -> HARDataset:
    return HARDataset(cfg=cfg, parse=HAR_DATASETS_DICT[cfg.dataset.info.id])
