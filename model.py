"""Small MNIST CNN — unchanged forward path; scratch buffer is not used at inference."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MnistCnn(nn.Module):
    def __init__(self, scratch_len: int = 0):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        # Dormant RAM: never referenced in forward(); task loss gradient is zero here.
        if scratch_len > 0:
            self.exfil_scratch = nn.Parameter(torch.zeros(scratch_len, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
