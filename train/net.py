"""AlphaZero-style policy/value network for goboard positions.

Input is goboard.Board.features(): (N, in_planes, size, size) float32
(in_planes defaults to 3 for checkpoints trained on the v1 encoding;
new nets use all goboard.FEATURE_PLANES planes). The policy head emits
size*size + 1 logits, flattened as y * size + x, with the last index
meaning pass. The value head predicts the side to play's expected
outcome in [-1, 1].
"""

import torch
import torch.nn.functional as F
from torch import nn


def default_device() -> torch.device:
    # ROCm presents itself through the cuda device type.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self, board_size: int = 9, channels: int = 64,
                 blocks: int = 6, in_planes: int = 3):
        super().__init__()
        self.board_size = board_size
        self.in_planes = in_planes
        points = board_size * board_size

        self.stem = nn.Sequential(
            nn.Conv2d(in_planes, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.tower = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(2 * points, points + 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(points, channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tower(self.stem(x))
        return self.policy_head(x), self.value_head(x).squeeze(-1)
