import logging
from pathlib import Path
from omegaconf import DictConfig
import functools

import pytorch_lightning as pl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim.lr_scheduler import LambdaLR

# from torchmetrics.classification.precision_recall import Precision, Recall
from torchmetrics.functional.classification.precision_recall import precision_recall
from torchmetrics.functional.classification.precision_recall_curve import (
    precision_recall_curve,
)

from dataloaders.data_interface import WaveformInterface
from utils import eval_semantic

from model.model import TCN

log = logging.getLogger(__name__)


class LambdaStepLR(LambdaLR):
    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(
            optimizer, lr_lambda, last_step, verbose=False
        )

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(
            optimizer, lambda s: (1 - s / (max_iter + 1)) ** power, last_step
        )


def configure_optimizers(parameters, optimizer_cfg, scheduler_cfg):
    if optimizer_cfg.type == "Adam":
        optimizer = torch.optim.Adam(
            parameters,
            lr=optimizer_cfg.lr,
        )
    elif optimizer_cfg.type == "SGD":
        optimizer = torch.optim.SGD(
            parameters,
            lr=optimizer_cfg.lr,
            momentum=optimizer_cfg.momentum,
            dampening=optimizer_cfg.dampening,
            weight_decay=optimizer_cfg.weight_decay,
        )
    else:
        # TODO: Put error logging at high level try catch block
        log.error(f"Invalid optimizer type: {optimizer_cfg.type}")
        raise ValueError(f"Invalid optimizer type: {optimizer_cfg.type}")

    # Get scheduler if any is specified
    if not scheduler_cfg.type:
        log.info("No learning rate schedular specified")
        return optimizer
    elif scheduler_cfg.type == "ExpLR":
        scheduler = ExponentialLR(optimizer, scheduler_cfg.exp_gamma)
    elif scheduler_cfg.type == "PolyLR":
        scheduler = PolyLR(
            optimizer, max_iter=scheduler_cfg.max_iter, power=scheduler_cfg.poly_power
        )
    else:
        log.error(f"Invalid scheduler type: {scheduler_cfg.type}")
        raise ValueError(f"Invalid scheduler type: {scheduler_cfg.type}")

    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": scheduler_cfg.interval,
            "frequency": scheduler_cfg.frequency,
        },
    }


class TemporalConvolutionModel(pl.LightningModule):
    def __init__(
        self,
        cfg: DictConfig,
        data_interface: WaveformInterface,
    ):
        super().__init__()

        # Learning configuration
        self.cfg = cfg
        self.optimizer_cfg = cfg.optimizer
        self.scheduler_cfg = cfg.scheduler

        self.index_to_label_map = data_interface.index_to_label_map

        # self.semantic_criterion = nn.CrossEntropyLoss(
        #     ignore_index=cfg.data.ignore_label
        # )
        self.semantic_criterion = nn.NLLLoss()
        self.ignore_label_id = cfg.data.ignore_label

        self.model = TCN(cfg)

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        return configure_optimizers(parameters, self.optimizer_cfg, self.scheduler_cfg)

    def loss_fn(self, semantic_labels, semantic_scores):
        """Just return the semantic loss"""
        return self.semantic_criterion(semantic_scores, semantic_labels.long())

    def training_step(self, batch: dict, batch_idx: int):

        output = self.model(batch["waveforms"])
        loss = self.loss_fn(batch["labels"], output)

        # Log losses
        log = functools.partial(self.log, on_step=True, on_epoch=True)
        log("train_loss", loss)

        return loss

    def validation_step(self, batch: dict, batch_idx: int):
        output = self.model(batch["waveforms"])
        loss = self.loss_fn(batch["labels"], output)
        self.log("val_loss", loss, sync_dist=True)

        return self.get_matches_val(batch, output)

    def test_step(self, batch: dict, batch_idx: int):
        output = self.model(batch["waveforms"])

        return self.get_matches_test(batch, output)

    def validation_epoch_end(self, semantic_matches):
        semantic_matches = {
            "pred": np.concatenate([match["pred"] for match in semantic_matches], 0),
            "gt": np.concatenate([match["gt"] for match in semantic_matches], 0),
            "torch_output": torch.cat(
                [match["torch_output"] for match in semantic_matches], 0
            ),
            "torch_gt": torch.cat([match["torch_gt"] for match in semantic_matches], 0),
        }

        mean_iou = eval_semantic.evaluate(
            semantic_matches,
            self.index_to_label_map,
            self.ignore_label_id,
            verbose=True,
        )
        self.log("val_semantic_mIOU", mean_iou, sync_dist=True)

        # Compute precision recall
        output = semantic_matches["torch_output"]
        labels = semantic_matches["torch_gt"].int()
        precision, recall = precision_recall(output, labels)
        log.info(f"precision: {precision}, recall: {recall}")
        self.log("val_precision", precision, sync_dist=True)
        self.log("val_recall", recall, sync_dist=True)

    def get_matches_val(self, batch, output):
        """Get all gt and predictions for validation"""
        semantic_pred = output.detach().cpu().numpy().argmax(1)
        semantic_gt = batch["labels"].detach().cpu().numpy().astype(np.int)
        semantic_matches = {
            "gt": semantic_gt,
            "pred": semantic_pred,
            "torch_gt": batch["labels"],
            "torch_output": output,
        }

        return semantic_matches

    def get_matches_test(self, batch, output):
        """Generate test-time prediction to gt matches"""
        semantic_pred = output.detach().cpu().numpy().max(1)[1]
        semantic_gt = (
            batch["labels"].reshape((-1,)).detach().cpu().numpy().astype(np.int)
        )
        semantic_matches = {"gt": semantic_gt, "pred": semantic_pred}

        return semantic_matches

    def test_epoch_end(self, outputs) -> None:

        # Semantic eval
        semantic_matches = {}
        for output in outputs:
            scene_name = output["test_scene_name"]
            semantic_matches[scene_name] = {}
            semantic_matches[scene_name]["gt"] = output["semantic"]["gt"]
            semantic_matches[scene_name]["pred"] = output["semantic"]["pred"]

        eval_semantic.evaluate(
            semantic_matches,
            self.index_to_label_map,
            self.ignore_label_id,
        )
