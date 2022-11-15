from pathlib import Path
import numpy as np
import torch


def load_waveforms(preprocessed_path: Path, materials: list):
    waveforms = {}
    for material in materials:
        high, low = torch.load(preprocessed_path / (material + ".pth"))
        waveforms[material] = {
            "high": torch.from_numpy(high),
            "low": torch.from_numpy(low),
        }
    return waveforms


def get_random_colour():
    return np.random.choice(range(256), size=3).astype(np.float) / 256.0
