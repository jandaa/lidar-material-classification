import logging
import random
from pathlib import Path

import hydra
from omegaconf import DictConfig

import torch
import pytorch_lightning as pl
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import label_binarize
from torchmetrics.functional.classification.precision_recall import precision_recall

from dataloaders.data_interface import WaveformInterface
from dataloaders.datamodule import DataModule
from model.trainer import TemporalConvolutionModel
from utils import eval_semantic


log = logging.getLogger("main")


class Trainer:
    """High level training class."""

    def __init__(self, cfg: DictConfig):

        # Save configuration
        self.cfg = cfg

        # Have to run this for some reason
        torch.cuda.is_available()

        # Set absolute checkpoint paths
        self.checkpoint_path = None
        if cfg.checkpoint:
            self.checkpoint_path = str(Path.cwd() / "checkpoints" / cfg.checkpoint)
            log.info(f"Resuming checkpoint: {Path(self.checkpoint_path).name}")

        # Load data interface
        log.info("Loading data interface")
        self.data_interface = WaveformInterface(self.cfg)

        # Create model
        log.info("Creating Model")
        self.model = TemporalConvolutionModel(cfg, self.data_interface)

        # Init variables
        self.trainer = self.get_trainer()
        self.data_loader = None

    def run_tasks(self):
        """Run all the tasks specified in configuration."""
        for task in self.cfg.tasks:
            if hasattr(self, task):
                log.info(f"Performing task: {task}")
                getattr(self, task)()
            else:
                raise NotImplementedError(f"Task {task} does not exist")

    def get_trainer(self):
        """Build a trainer for regular training"""

        log.info("Building Trainer")
        tb_logger = pl.loggers.TensorBoardLogger("lightning_logs/train")
        return pl.Trainer(
            logger=tb_logger,
            accelerator=self.cfg.accelerator,
            strategy=self.cfg.strategy,
            devices=self.cfg.devices,
            max_epochs=self.cfg.max_epochs,
            resume_from_checkpoint=self.checkpoint_path,
            check_val_every_n_epoch=int(self.cfg.check_val_every_n_epoch),
            limit_train_batches=self.cfg.limit_train_batches,
            limit_val_batches=self.cfg.limit_val_batches,
            limit_test_batches=self.cfg.limit_test_batches,
            accumulate_grad_batches=self.cfg.train.accumulate_grad_batches,
            deterministic=True,
            precision=self.cfg.precision,
            max_time=self.cfg.max_time,
            val_check_interval=self.cfg.val_check_interval,
        )

    def preprocess(self):
        """Preprocess waveforms."""
        self.data_interface.preprocess()

    def train(self):
        """Train on superivised data."""

        self.data_loader = DataModule(self.cfg)

        if self.checkpoint_path:
            checkpoint = torch.load(self.checkpoint_path)
            self.model.load_state_dict(checkpoint["state_dict"])

        log.info("starting training")
        self.trainer.fit(
            self.model,
            self.data_loader.train_dataloader(),
            self.data_loader.val_dataloader(),
        )
        log.info("Finished training")

    def train_random_forest(self):
        "Train a random forest classifier as a baseline."

        data_interface = WaveformInterface(self.cfg)
        data_interface.load()

        train = data_interface.train_data
        val = data_interface.val_data

        training_features = train["waveforms"].detach().cpu().numpy()
        validation_features = val["waveforms"].detach().cpu().numpy()

        training_labels = label_binarize(
            train["labels"].detach().cpu().numpy(),
            classes=list(range(len(self.cfg.data.materials))),
        )

        # Fit model
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=50,
            random_state=self.cfg.train.random_seed,
            n_jobs=self.cfg.train.train_workers,
        )
        model.fit(training_features, training_labels)

        # Validate
        y_score = model.predict(validation_features)

        precision, recall = precision_recall(
            torch.from_numpy(y_score).float(), val["labels"].int()
        )
        log.info(f"Precision: {precision}, Recall: {recall}")

        semantic_matches = {
            "pred": np.argmax(y_score, axis=1),
            "gt": val["labels"].int().squeeze().detach().cpu().numpy(),
        }

        mean_iou = eval_semantic.evaluate(
            semantic_matches,
            data_interface.index_to_label_map,
            -1,
            verbose=True,
        )
        log.info(f"mIOU: {mean_iou}")


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Set random seeds for reproductability
    seed = 42
    torch.manual_seed(seed)
    random.seed(0)
    np.random.seed(0)
    pl.seed_everything(seed, workers=True)

    # Create trainer and go through all desired tasks
    trainer = Trainer(cfg)
    trainer.run_tasks()


if __name__ == "__main__":
    main()
