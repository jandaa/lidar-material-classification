from torch import nn
from model.tcn import TemporalConvNet
import torch.nn.functional as F


class TCN(nn.Module):
    def __init__(self, cfg):
        super(TCN, self).__init__()

        self.tcn = TemporalConvNet(
            cfg.model.input_size,
            cfg.model.channels,
            kernel_size=cfg.model.kernel_size,
            dropout=cfg.model.dropout,
        )
        self.linear = nn.Linear(cfg.model.channels[-1], len(cfg.data.materials))
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        output = self.tcn(x)
        output = self.linear(output[:, :, -1])
        # output = self.softmax(output)
        return F.log_softmax(output, dim=1)
