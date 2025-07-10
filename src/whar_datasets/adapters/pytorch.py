import random
from typing import Tuple
import numpy as np
from torch import Tensor
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from tqdm import tqdm

from whar_datasets.core.processing import process
from whar_datasets.core.sampling import get_label, get_window
from whar_datasets.core.splitting import get_split
from whar_datasets.core.normalizing import get_norm_params, normalize_window
from whar_datasets.core.utils.loading import load_session_metadata, load_windowing
from whar_datasets.core.weighting import compute_class_weights
from whar_datasets.core.config import WHARConfig


class PytorchAdapter(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(self, cfg: WHARConfig, override_cache: bool = False):
        super().__init__()

        self.cfg = cfg

        self.cache_dir, self.windows_dir = process(cfg, override_cache)
        self.session_metadata = load_session_metadata(self.cache_dir)
        self.window_metadata, self.windows = load_windowing(
            self.cache_dir, self.windows_dir, self.cfg
        )

        print(f"subject_ids: {np.sort(self.session_metadata['subject_id'].unique())}")
        print(f"activity_ids: {np.sort(self.session_metadata['activity_id'].unique())}")

        self.seed = cfg.dataset.training.seed

        self.X: Tensor | None = None
        self.y: Tensor | None = None

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def get_dataloaders(
        self,
        train_batch_size: int,
        train_shuffle: bool = True,
        subj_cross_val_group_index: int | None = None,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # get split indices from config
        self.train_indices, self.val_indices, self.test_indices = get_split(
            self.cfg,
            self.session_metadata,
            self.window_metadata,
            subj_cross_val_group_index,
        )

        # get normalization parameters from train indices
        self.norm_params = get_norm_params(
            self.cfg,
            self.train_indices,
            self.windows_dir,
            self.window_metadata,
            self.windows,
        )

        # prepare windows
        self._prepare_windows()

        # specify split subsets
        train_set = Subset(self, self.train_indices)
        test_set = Subset(self, self.test_indices)
        val_set = Subset(self, self.val_indices)

        print(f"train: {len(train_set)} | val: {len(val_set)} | test: {len(test_set)}")

        # create dataloaders from split
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=train_batch_size,
            shuffle=train_shuffle,
            generator=self.generator,
        )

        val_loader = DataLoader(
            dataset=val_set,
            batch_size=len(val_set),
            shuffle=False,
            generator=self.generator,
        )

        test_loader = DataLoader(
            dataset=test_set,
            batch_size=1,
            shuffle=False,
            generator=self.generator,
        )

        return train_loader, val_loader, test_loader

    def get_class_weights(self, dataloader: DataLoader) -> dict:
        indices = dataloader.dataset.indices  # type: ignore
        assert indices is not None

        return compute_class_weights(
            self.session_metadata, self.window_metadata.iloc[indices]
        )

    def _prepare_windows(self):
        x_list = []
        y_list = []

        loop = tqdm(range(len(self.window_metadata)))
        loop.set_description("Preparing windows")

        for i in loop:
            label = get_label(i, self.window_metadata, self.session_metadata)
            window = get_window(
                i, self.cfg, self.windows_dir, self.window_metadata, self.windows
            )
            window = normalize_window(self.cfg, self.norm_params, window)

            y_list.append(label)
            x_list.append(torch.tensor(window.values, dtype=torch.float32))

        self.y = torch.tensor(y_list, dtype=torch.long)
        self.X = torch.stack(x_list)

        # free memory
        del self.windows
        self.windows = None

    def __len__(self) -> int:
        return len(self.window_metadata)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if self.X is None or self.y is None:
            raise RuntimeError(
                "Dataloaders must be initialized before accessing items."
            )

        return self.y[index], self.X[index]
