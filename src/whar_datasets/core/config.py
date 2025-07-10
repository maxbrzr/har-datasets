from enum import Enum
from typing import Callable, List, Tuple, TypeAlias, Dict
from pydantic import BaseModel, field_serializer
import pandas as pd

Parse: TypeAlias = Callable[
    [str, str], Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]
]

NON_CHANNEL_COLS: List[str] = [
    "subject_id",
    "activity_id",
    "session_id",
    "activity_name",
    "timestamp",
]


class NormType(str, Enum):
    STD_GLOBALLY = "std_globally"
    MIN_MAX_GLOBALLY = "min_max_globally"
    ROBUST_SCALE_GLOBALLY = "robust_scale_globally"
    STD_PER_SAMPLE = "std_per_sample"
    MIN_MAX_PER_SAMPLE = "min_max_per_sample"
    ROBUST_SCALE_PER_SAMPLE = "robust_scale_per_sample"


class GivenSplit(BaseModel):
    train_subj_ids: List[int]
    test_subj_ids: List[int]


class SubjCrossValSplit(BaseModel):
    subj_id_groups: List[List[int]]


class Split(BaseModel):
    given_split: GivenSplit | None
    subj_cross_val_split: SubjCrossValSplit | None
    val_percentage: float = 0.1  # in [0, 1]


class Training(BaseModel):
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 100
    seed: int = 0
    in_memory: bool = False
    split: Split
    normalization: NormType | None = NormType.STD_GLOBALLY


class Selections(BaseModel):
    activity_names: List[str]
    sensor_channels: List[str]


class SlidingWindow(BaseModel):
    window_time: float  # in seconds
    overlap: float  # in [0, 1]


class Caching(BaseModel):
    cache_parsed: bool = True
    cache_windows: bool = True
    cache_spectrograms: bool = True


class Preprocessing(BaseModel):
    selections: Selections
    sliding_window: SlidingWindow
    in_parallel: bool = False


class Parsing(BaseModel):
    parse: Parse
    activity_id_col: str = "activity_id"

    @field_serializer("parse")
    def serialize_func(self, func, _info):
        # Serialize the function by its name
        return func.__name__


class Info(BaseModel):
    id: str
    download_url: str
    sampling_freq: int
    num_of_subjects: int
    num_of_activities: int
    num_of_channels: int


class Dataset(BaseModel):
    info: Info
    parsing: Parsing
    preprocessing: Preprocessing
    training: Training
    caching: Caching = Caching()


class Spectrogram(BaseModel):
    window_size: int | None = 32
    overlap: int | None = None
    mode: str = "magnitude"


class Common(BaseModel):
    datasets_dir: str
    resampling_freq: int | None = None
    use_derivative: bool = False
    use_spectrogram: bool = False
    spectrogram: Spectrogram = Spectrogram()


class WHARConfig(BaseModel):
    common: Common  # common config applyed to all datasets
    dataset: Dataset  # dataset specific config
