import math
import logging
from omegaconf import DictConfig, listconfig

import torch
import numpy
import random
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from dataloaders.dataset import WaveformDataset
from dataloaders.data_interface import WaveformInterface


log = logging.getLogger(__name__)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        cfg: DictConfig,
    ):
        super().__init__()

        # Get number of gpus
        if type(cfg.devices) == listconfig.ListConfig:
            num_gpus = len(cfg.devices)
        else:
            num_gpus = cfg.devices

        # Get batch sizes
        self.batch_size = math.floor(cfg.train.batch_size / num_gpus)
        log.info(f"Using batch size of {self.batch_size}")

        # Dataloader specific parameters
        self.cfg = cfg
        self.ignore_label = cfg.data.ignore_label

        # Number of workers
        self.num_workers = cfg.train.train_workers

        # Load data from interface
        data_interface = WaveformInterface(cfg)
        data_interface.load()
        self.train_data = data_interface.train_data
        self.val_data = data_interface.val_data
        self.test_data = data_interface.test_data

        # Grab label to index map
        self.label_to_index_map = data_interface.label_to_index_map

        training_samples = len(self.train_data["waveforms"])
        validation_samples = len(self.val_data["waveforms"])
        testing_samples = len(self.test_data["waveforms"])
        log.info(f"Training samples: {training_samples}")
        log.info(f"Validation samples: {validation_samples}")
        log.info(f"Testing samples: {testing_samples}")

        self.dataset_type = WaveformDataset

    def train_dataloader(self):
        dataset = self.dataset_type(self.train_data, self.cfg, is_test=False)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=True,
            sampler=None,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def val_dataloader(self):
        dataset = self.dataset_type(self.val_data, self.cfg, is_test=True)
        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=True,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )

    def test_dataloader(self):
        dataset = self.dataset_type(self.test_data, self.cfg, is_test=True)
        return DataLoader(
            dataset,
            batch_size=1,
            collate_fn=dataset.collate,
            num_workers=self.num_workers,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            worker_init_fn=seed_worker,
            generator=g,
        )
