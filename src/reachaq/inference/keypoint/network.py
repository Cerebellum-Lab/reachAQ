"""The keypoint network: a small CNN that predicts one heatmap per body part.

One animal per camera view and a fixed field of view make the problem
single-instance, so there is no detector, box or non-maximum suppression. Each
body part gets a heatmap at a quarter of the input resolution and two offset
maps that refine the peak below one heatmap cell:

    input  (N, 1, S, S)      grayscale, 0..1
    output (N, 3K, S/4, S/4)  K heatmap logits, then K x-offsets, K y-offsets

A part's position is the centre of its peak cell plus that cell's offset, in
cells, times the stride. Its confidence is the sigmoid of the peak logit, so an
absent part - one trained against an empty heatmap - scores near zero.

The cameras are monochrome, so the network takes one channel. predict() is
handed three-channel frames; they are averaged, which is exact for frames that
carry the same value in every channel.

Everything here is plain torch so that live inference and training share one
definition and one CUDA runtime.
"""

import typing

import torch
from torch import nn

STRIDE = 4


def _conv(cin: int, cout: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, stride, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


class _Residual(nn.Module):
    """Two 3x3 convolutions around an identity shortcut."""

    def __init__(self, channels: int):
        super().__init__()
        self.first = _conv(channels, channels)
        self.second = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.second(self.first(x)))


def _stage(cin: int, cout: int, blocks: int) -> nn.Sequential:
    """Halve the resolution, then refine at the new width."""
    return nn.Sequential(_conv(cin, cout, 2),
                         *[_Residual(cout) for _ in range(blocks)])


class KeypointNet(nn.Module):
    """
    Encoder at strides 4-32, a top-down merge back to stride 4, one head.

    The deep stages are there for context rather than resolution: telling the
    right paw from the left, or the tongue from the lower lip, depends on
    where the part sits relative to the rest of the animal, which a stride-4
    feature alone cannot see. The top-down path carries that context back to
    the resolution the heatmaps are read at.
    """

    def __init__(self, keypoints: int, widths: typing.Sequence[int] = (32, 64, 128, 192),
                 blocks: typing.Sequence[int] = (1, 2, 2, 2), neck: int = 64):
        super().__init__()
        self.keypoints = int(keypoints)
        self.widths = tuple(int(w) for w in widths)
        self.blocks = tuple(int(b) for b in blocks)
        self.neck_width = int(neck)
        w4, w8, w16, w32 = self.widths
        b4, b8, b16, b32 = self.blocks

        self.stem = nn.Sequential(_conv(1, w4 // 2, 2), _conv(w4 // 2, w4, 2),
                                  *[_Residual(w4) for _ in range(b4)])
        self.s8 = _stage(w4, w8, b8)
        self.s16 = _stage(w8, w16, b16)
        self.s32 = _stage(w16, w32, b32)

        self.lateral = nn.ModuleList(
            nn.Conv2d(w, neck, 1) for w in (w4, w8, w16, w32))
        self.smooth = nn.ModuleList(_conv(neck, neck) for _ in range(3))
        self.head = nn.Sequential(_conv(neck, neck),
                                  nn.Conv2d(neck, 3 * self.keypoints, 1))
        # Start every heatmap near "absent". Most cells are background, and a
        # zero bias would begin training with a loss dominated by them.
        nn.init.constant_(self.head[-1].bias[:self.keypoints], -4.0)
        nn.init.zeros_(self.head[-1].bias[self.keypoints:])

    def config(self) -> dict:
        """What it takes to rebuild this network around a state_dict."""
        return {"keypoints": self.keypoints, "widths": list(self.widths),
                "blocks": list(self.blocks), "neck": self.neck_width}

    def forward(self, x):
        f4 = self.stem(x)
        f8 = self.s8(f4)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        top = self.lateral[3](f32)
        for level, feature in ((2, f16), (1, f8), (0, f4)):
            top = nn.functional.interpolate(top, size=feature.shape[-2:], mode="nearest")
            top = self.smooth[level](top + self.lateral[level](feature))
        return self.head(top)


def decode(output: torch.Tensor) -> torch.Tensor:
    """
    (N, 3K, H, W) network output -> (N, K, 3) x, y, confidence.

    x and y are in the pixels of the network input, with pixel centres at
    integers, the convention DeepLabCut labels are written in.
    """
    n, channels, height, width = output.shape
    k = channels // 3
    heat = output[:, :k].reshape(n, k, height * width)
    peak, index = heat.max(dim=2)
    offsets = output[:, k:].reshape(n, 2, k, height * width)
    gather = index.view(n, 1, k, 1).expand(n, 2, k, 1)
    dx, dy = offsets.gather(3, gather).squeeze(3).unbind(1)
    col = (index % width).to(output.dtype)
    row = torch.div(index, width, rounding_mode="floor").to(output.dtype)
    x = (col + 0.5 + dx) * STRIDE - 0.5
    y = (row + 0.5 + dy) * STRIDE - 0.5
    return torch.stack((x, y, torch.sigmoid(peak)), dim=2)
