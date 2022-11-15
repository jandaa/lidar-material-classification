import logging
import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

import matplotlib
from joblib import load

matplotlib.use("TKAgg")
from matplotlib import pyplot as plt
from pathlib import Path
import numpy as np

import open3d as o3d
from utils.utils import get_random_colour
from pioneer.das.api.platform import Platform

from PIL import Image


def visualize_class_colours(class_colours):
    iter = len(class_colours)
    width_px = 1000
    new = Image.new(mode="RGB", size=(width_px, 120))

    for i in range(iter):

        newt = Image.new(
            mode="RGB",
            size=(width_px // iter, 100),
            color=tuple((class_colours[i]).astype(np.int)),
        )
        new.paste(newt, (i * width_px // iter, 10))

    new.show()


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Setup
    root_folder = Path(get_original_cwd())
    data_folder = root_folder / cfg.preprocess.data_path
    output_folder = root_folder / cfg.preprocess.point_cloud_path
    output_folder.mkdir(parents=True, exist_ok=True)
    materials = cfg.train.materials

    # Get colours for each class
    # class_colours = np.array(
    #     [
    #         [86, 209, 41],
    #         [247, 247, 17],
    #         [17, 44, 247],
    #         [247, 17, 216],
    #         [0, 0, 0],
    #     ]
    # )
    class_colours = np.array(
        [
            [0, 0, 0],
            [17, 17, 216],
            [216, 107, 4],
            [200, 200, 200],
            [247, 247, 5],
            [255, 0, 0],
        ]
    )
    # visualize_class_colours(class_colours)
    class_colours = class_colours / 255.0

    # Load model
    if cfg.train.save_model_filename:
        model = load(f"{cfg.train.save_model_filename}.joblib")
    else:
        logging.error("Missing model")

    for material in ["yellow"]:

        pf = Platform(str(data_folder / material))

        # Bench Positions
        bench = pf.sensors["bench_bfc"]
        positions = bench["positions"]

        # Full waveforms
        pixell = pf.sensors["pixell_bfc"]
        echos = pixell["ech"]
        waveform = pixell["ftrr"]

        for frame_ind in range(len(positions)):
            angle = positions[frame_ind].raw.reshape(-1)[0][2]

            # Select angle range
            if cfg.preprocess.max_angle_in_deg != -1:
                if abs(angle) > cfg.preprocess.max_angle_in_deg:
                    continue

            xyz = echos[frame_ind].get_point_cloud()
            indices = echos[frame_ind].indices

            distances = echos[frame_ind].distances
            # board_inds = np.logical_and(xyz[:, 0] < 1.1, abs(xyz[:, 1]) < 0.75)
            # predictions = np.ones(indices.size, dtype=np.int32)
            # predictions[board_inds] = 0

            # Grab waveforms
            high = waveform[frame_ind].raw["high"]["data"][indices]
            low = waveform[frame_ind].raw["low"]["data"][indices]

            # Make feature vectors
            high = high[
                :,
                list(
                    range(
                        cfg.train.features.high.min,
                        cfg.train.features.high.max,
                        cfg.train.features.high.step,
                    )
                ),
            ]
            low = low[
                :,
                list(
                    range(
                        cfg.train.features.low.min,
                        cfg.train.features.low.max,
                        cfg.train.features.low.step,
                    )
                ),
            ]
            feature = np.hstack([high, low])

            # Run model
            predictions = model.predict(feature)
            # prob_predictions = model.predict_proba(feature)

            # unknown_indices = predictions.sum(axis=1) == 0
            predictions = predictions.argmax(axis=1)
            # predictions[unknown_indices] = 4

            # Colourize point cloud based on predictions
            colours = class_colours[predictions]

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colours)
            o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    main()
