import torch
from torch.utils.data.dataset import Dataset

from dataloaders import transforms


class WaveformDataset(Dataset):
    def __init__(self, data, cfg, is_test=False):
        super(WaveformDataset, self).__init__()

        # Save config
        self.cfg = cfg
        self.num_workers = cfg.train.train_workers
        self.ignore_label = cfg.data.ignore_label
        self.is_test = is_test

        # Save data
        self.data = data

        # Transformations
        self.augmentations = transforms.Compose(
            [
                # transforms.Normalize(mean=469.47983, std=6333.1953),
                # transforms.Scale(scale=1 / 32752.0)
            ]
        )

    def __len__(self):
        return len(self.data["waveforms"])

    def __getitem__(self, index):
        augmented_waveform = self.data["waveforms"][index]
        # augmented_waveform = self.augmentations(augmented_waveform)
        return {
            "waveform": augmented_waveform,
            "label": self.data["labels"][index],
        }

    def collate(self, batch):
        # waveforms = torch.concat(
        #     [datapoint["waveform"].unsqueeze(1).unsqueeze(0) for datapoint in batch],
        #     0,
        # )
        waveforms = torch.concat(
            [datapoint["waveform"].unsqueeze(0).unsqueeze(1) for datapoint in batch],
            0,
        )
        labels = torch.concat([datapoint["label"] for datapoint in batch], 0)
        return {"waveforms": waveforms, "labels": labels}
