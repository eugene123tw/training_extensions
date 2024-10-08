# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Positional encoding module."""

from __future__ import annotations

import math
from typing import Any, ClassVar

import torch
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter

from otx.algo.modules.norm import FrozenBatchNorm2d
from otx.algo.object_detection_3d.utils.utils import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float | None = None,
    ):
        """Initialize the PositionEmbeddingSine module.

        Args:
            num_pos_feats (int): Number of positional features.
            temperature (int): Temperature scaling factor.
            normalize (bool): Flag indicating whether to normalize the position embeddings.
            scale (Optional[float]): Scaling factor for the position embeddings. If None, default value is used.
        """
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            msg = "normalize should be True if scale is passed"
            raise ValueError(msg)
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward function for PositionEmbeddingSine module."""
        x = tensor_list.tensors
        mask = tensor_list.mask
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned."""

    def __init__(self, num_pos_feats: int = 256):
        """Positional embedding."""
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward pass of the PositionEmbeddingLearned module.

        Args:
            tensor_list (NestedTensor): Input tensor.

        Returns:
            torch.Tensor: Position embeddings.
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device) / w * 49
        j = torch.arange(h, device=x.device) / h * 49
        x_emb = self.get_embed(i, self.col_embed)
        y_emb = self.get_embed(j, self.row_embed)
        return (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )

    def get_embed(self, coord: torch.Tensor, embed: nn.Embedding) -> torch.Tensor:
        """Get the embedding for the given coordinates.

        Args:
            coord (torch.Tensor): The coordinates.
            embed (nn.Embedding): The embedding layer.

        Returns:
            torch.Tensor: The embedding for the coordinates.
        """
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long()
        ceil_coord = (floor_coord + 1).clamp(max=49)
        return embed(floor_coord) * (1 - delta) + embed(ceil_coord) * delta