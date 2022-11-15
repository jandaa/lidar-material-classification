from pathlib import Path
import numpy as np
import torch

# Leddartech DAS
from pioneer.das.api.platform import Platform

from utils.utils import load_waveforms


class WaveformInterface:
    def __init__(self, cfg):

        # Setup
        self.cfg = cfg
        self.dataset_path = Path(self.cfg.dataset_path)
        self.raw_data_path = self.dataset_path / cfg.data.raw_path
        self.preprocessed_path = self.dataset_path / cfg.data.preprocessed_path
        self.preprocessed_path.mkdir(parents=True, exist_ok=True)
        self.materials = cfg.data.materials

        self.label_to_index_map = {
            label: index for index, label in enumerate(self.materials)
        }
        self.index_to_label_map = {
            index: label for label, index in self.label_to_index_map.items()
        }

        # Split data
        self.train_data = None
        self.val_data = None
        self.test_data = None

    def load(self) -> None:
        """Load waveform data for all the materials requrested."""

        # Load raw waveform data
        waveform_map = load_waveforms(self.preprocessed_path, self.materials)

        # Get max number of features
        max_datapoints_per_class = min(
            waveform["high"].shape[0] for _, waveform in waveform_map.items()
        )
        max_datapoints_per_class = min(10000, max_datapoints_per_class)

        # Get features and labels
        waveforms = []
        labels = []
        for label, material in enumerate(self.materials):
            low = waveform_map[material]["low"]

            # Even out labels
            index = np.random.choice(
                low.shape[0], max_datapoints_per_class, replace=False
            )
            low = low[index, :]

            waveforms.append(low)
            labels.append(torch.ones(low.shape[0], dtype=np.int).reshape(-1, 1) * label)

        waveforms = torch.concat(waveforms, 0).float()
        labels = torch.concat(labels, 0).float()

        # Compute mean and variance
        # mean = np.mean(waveforms.detach().numpy())
        # std = np.std(waveforms.detach().numpy())

        # Split data
        indices = np.arange(waveforms.shape[0])
        np.random.shuffle(indices)
        split = int(self.cfg.train.train_split * indices.size)
        training_split = indices[:split]
        validation_split = indices[split:]

        self.train_data = {
            "waveforms": waveforms[training_split],
            "labels": labels[training_split],
        }
        self.val_data = {
            "waveforms": waveforms[validation_split],
            "labels": labels[validation_split],
        }
        self.test_data = self.val_data

    def preprocess(self) -> None:
        """Preprocess raw data by extracting waveforms from the das API."""

        # Background unknown waveforms
        unknown_highs = []
        unknown_lows = []

        for material in self.materials:

            # Unkown class is present in all materials
            if material == "unknown":
                continue

            pf = Platform(str(self.raw_data_path / material))

            # Bench Positions
            bench = pf.sensors["bench_bfc"]
            positions = bench["positions"]

            # Full waveforms
            pixell = pf.sensors["pixell_bfc"]
            echos = pixell["ech"]
            waveform = pixell["ftrr"]

            highs = []
            lows = []

            for frame_ind in range(len(positions)):
                angle = positions[frame_ind].raw.reshape(-1)[0][2]

                # Select angle range
                if self.cfg.preprocess.max_angle_in_deg != -1:
                    if abs(angle) > self.cfg.preprocess.max_angle_in_deg:
                        continue

                # Get valid indices
                indices = echos[frame_ind].indices

                # Get indicies which belong to boardz
                xyz = echos[frame_ind].get_point_cloud()
                board_inds = np.logical_and(xyz[:, 0] < 1.1, abs(xyz[:, 1]) < 0.75)

                # Grab waveforms
                high = waveform[frame_ind].raw["high"]["data"][indices][board_inds]
                low = waveform[frame_ind].raw["low"]["data"][indices][board_inds]

                highs.append(high)
                lows.append(low)

                # select all the unknowns
                high = waveform[frame_ind].raw["high"]["data"][indices][~board_inds]
                low = waveform[frame_ind].raw["low"]["data"][indices][~board_inds]

                unknown_highs.append(high)
                unknown_lows.append(low)

                # classes = np.ones(indices.size, dtype=np.int32)
                # classes[board_inds] = 0
                # colours = class_colours[classes]
                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector(xyz)
                # pcd.colors = o3d.utility.Vector3dVector(colours)
                # o3d.visualization.draw_geometries([pcd])

            torch.save(
                (np.vstack(highs), np.vstack(lows)),
                self.preprocessed_path / (material + ".pth"),
            )

        # save background pixels
        torch.save(
            (np.vstack(unknown_highs), np.vstack(unknown_lows)),
            self.preprocessed_path / "unknown.pth",
        )
