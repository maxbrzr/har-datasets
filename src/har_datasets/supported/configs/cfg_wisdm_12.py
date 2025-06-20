from har_datasets.config.config import (
    Common,
    Dataset,
    GivenSplit,
    HARConfig,
    Info,
    Selections,
    SlidingWindow,
    Spectrogram,
    Split,
    SplitType,
    SubjCrossValSplit,
    Training,
)

cfg_wisdm_12 = HARConfig(
    common=Common(
        datasets_dir="./datasets",
        resampling_freq=None,
        normalization=None,
        include_derivative=False,
        sliding_window=SlidingWindow(window_time=2.56, overlap=0),
        spectrogram=Spectrogram(window_size=20, overlap=None, mode="magnitude"),
    ),
    dataset=Dataset(
        info=Info(
            dataset_id="wisdm_12",
            dataset_url="https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz",
            sampling_freq=20,
        ),
        selections=Selections(
            activity_names=[
                "Walking",
                "Jogging",
                "Upstairs",
                "Downstairs",
                "Sitting",
                "Standing",
            ],
            channels=[
                "accel_x",
                "accel_y",
                "accel_z",
            ],
        ),
        split=Split(
            split_type=SplitType.GIVEN,
            given_split=GivenSplit(
                train_subj_ids=list(range(1, 26)),
                test_subj_ids=list(range(26, 31)),
                val_subj_ids=list(range(31, 37)),
            ),
            subj_cross_val_split=SubjCrossValSplit(
                subj_id_groups=[
                    [1, 2, 3, 4, 5, 6],
                    [7, 8, 9, 10, 11, 12],
                    [13, 14, 15, 16, 17, 18],
                    [19, 20, 21, 22, 23, 24],
                    [25, 26, 27, 28, 29, 30],
                    [31, 32, 33, 34, 35, 36],
                ],
            ),
        ),
        training=Training(
            batch_size=32,
            shuffle=True,
            learning_rate=0.0001,
            num_epochs=100,
        ),
    ),
)
