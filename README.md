# HAR Datasets

This library provides support for popular HAR (human activity recognition) datasets including

- metadata descriptions in [DCAT-AP](https://www.dcat-ap.de/) and [Croissant](https://github.com/mlcommons/croissant)
- downloading from original source with caching
- parsing into a centralized format with caching
- preparation via config with caching
- pytorch integration.

HAR datasets not included can be used by writing a custom config and parser, which can then be integrated easily.

# How to Install

```
pip install "git+https://github.com/maxbrzr/har-datasets.git"
```

This installs the library into the active environment.

# How To Use

```python
from har_datasets.dataset.har_dataset import HARDataset
from har_datasets.supported.getter import DatasetId, get_har_dataset_cfg_and_parser

cfg, parse = get_har_dataset_cfg_and_parser(dataset_id=DatasetId.UCI_HAR)
dataset = HARDataset(cfg=cfg, parse=parse)

train_loader, val_loader, test_loader = dataset.get_dataloaders()
```

For unsupported har datasets, a custom parse function and config can be implented and used instead.

# Supported HAR Datasets

- [x] [UCI-HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [x] [WISDM-12](https://www.cis.fordham.edu/wisdm/dataset.php)
- [] [RealWorld](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)
- [] [MotionSense](https://github.com/mmalekzadeh/motion-sense)
- [] [OPPORTUNITY](https://archive.ics.uci.edu/dataset/226/opportunity+activity+recognition)
- [] [PAMAP2](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)
- [] [Daphnet](https://archive.ics.uci.edu/dataset/245/daphnet+freezing+of+gait)
- [] [KU-HAR](https://data.mendeley.com/datasets/45f952y38r/5)
- [] [ExtraSensory](http://extrasensory.ucsd.edu/)
- [] [MHEALTH](https://archive.ics.uci.edu/dataset/319/mhealth+dataset)
- [] [UTD-MHAD](https://personal.utdallas.edu/~kehtar/UTD-MHAD.html)
- [] [USC-SIPI (USC-HAD)](https://sipi.usc.edu/had/)
- [] [HuGaDB](https://github.com/romanchereshnev/HuGaDB)
- [] [iSPL IMU-Stretch](https://github.com/thunguyenth/HAR_IMU_Stretch)
- [] [w-HAR](https://github.com/thunguyenth/HAR_IMU_Stretch)
- [] [SWELL](https://www.kaggle.com/datasets/qiriro/swell-heart-rate-variability-hrv)
- [] [WISDM-19](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)

# Common Format

Since all HAR datasets do not share a common format, specific parsers are provided for each datasets to bring it into a common format which simplifies preparation. As common format a single csv is specified, which contains sensor channels as different columns. Additionally it contains the columns

| Column         | Type  |
|----------------|-------|
| timestamp      | datetime in ns |
| subject_id     | int   |
| activity_id    | int   |
| activity_name  | str   |
| session_id     | int   |

All parsers must ensure to output this format. 

# Preparation Config

For benchmarking, all HAR datasets should be prepared in the same manner, e.g. sampling frequency. For this purpose, a centralized config is needed. Additionally dataset-specific configs are required, e.g. to select only a subset of sensor channels. For this purpose, a hierarchical config system using [hydra](https://hydra.cc/docs/intro/) and specified with yaml is utilized. The config is then validated with [pydantic](https://docs.pydantic.dev/latest/) against a specified [schema](./src/har_datasets/config/config.py) for debugging purposes.

# Preparation Features

- [x] Resampling
- [x] Normalization
- [x] Differentiation
- [x] Windowing
- [x] Spectrogram Generation
- [x] Class Weights Computation
- [x] Subject Cross Validation

# PyTorch Support

We provide [torch](https://pytorch.org/) datasets for easy integration into existing code bases.

# TODOS

- paper specifc time windows
- alle channel same
- configurable preprocessing piepline
- spectrogramm allows multiple transforms
- class resampling
- optional in mem or not, window chaching


