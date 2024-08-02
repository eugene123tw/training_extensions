# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""RT-Detr loss, modified from https://github.com/lyuwenyu/RT-DETR."""

from __future__ import annotations

from typing import Callable

import torch
from torch import nn
from torchvision.ops import box_convert

from otx.algo.common.losses import GIoULoss, L1Loss
from otx.algo.common.utils.bbox_overlaps import bbox_overlaps
from otx.algo.detection.utils.matchers import HungarianMatcher
from otx.algo.detection.utils.matchers.hungarian_matcher import HungarianMatcher
from otx.core.utils.mask_util import polygon_to_bitmap


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


class NestedTensor:
    def __init__(self, tensors, mask: torch.Tensor):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)


def nested_tensor_from_tensor_list(tensor_list: list[torch.Tensor]):
    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        batch_size, num_channels, height, width = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((batch_size, height, width), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("Only 3-dimensional tensors are supported")
    return NestedTensor(tensor, mask)


def dice_loss(inputs, targets, num_boxes):
    """Compute the DICE loss, similar to generalized IOU for masks

    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs (0 for the negative class and 1 for the positive
                 class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_boxes


def sigmoid_focal_loss(
    inputs,
    targets,
    num_boxes,
    alpha: float = 0.25,
    gamma: float = 2,
):
    """Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.

    Args:
        inputs (`torch.FloatTensor` of arbitrary shape):
            The predictions for each example.
        targets (`torch.FloatTensor` with the same shape as `inputs`)
            A tensor storing the binary classification label for each element in the `inputs` (0 for the negative class
            and 1 for the positive class).
        alpha (`float`, *optional*, defaults to `0.25`):
            Optional weighting factor in the range (0,1) to balance positive vs. negative examples.
        gamma (`int`, *optional*, defaults to `2`):
            Exponent of the modulating factor (1 - p_t) to balance easy vs hard examples.

    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = nn.functional.binary_cross_entropy_with_logits(
        inputs,
        targets,
        reduction="none",
    )
    # add modulating factor
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes


class DetrCriterion(nn.Module):
    """This class computes the loss for DETR.

    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)

    Args:
        weight_dict (dict[str, int | float]): A dictionary containing the weights for different loss components.
        alpha (float, optional): The alpha parameter for the loss calculation. Defaults to 0.2.
        gamma (float, optional): The gamma parameter for the loss calculation. Defaults to 2.0.
        num_classes (int, optional): The number of classes. Defaults to 80.
    """

    def __init__(
        self,
        weight_dict: dict[str, int | float],
        alpha: float = 0.2,
        gamma: float = 2.0,
        num_classes: int = 80,
    ) -> None:
        """Create the criterion."""
        super().__init__()
        self.num_classes = num_classes
        self.matcher = HungarianMatcher(weight_dict={"cost_class": 2, "cost_bbox": 5, "cost_giou": 2})
        loss_bbox_weight = weight_dict["loss_bbox"] if "loss_bbox" in weight_dict else 1.0
        loss_giou_weight = weight_dict["loss_giou"] if "loss_giou" in weight_dict else 1.0
        self.loss_vfl_weight = weight_dict["loss_vfl"] if "loss_vfl" in weight_dict else 1.0
        self.alpha = alpha
        self.gamma = gamma
        self.lossl1 = L1Loss(loss_weight=loss_bbox_weight)
        self.giou = GIoULoss(loss_weight=loss_giou_weight)

    def loss_labels_vfl(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the vfl loss.

        Args:
            outputs (dict[str, torch.Tensor]): Model outputs.
            targets (List[Dict[str, torch.Tensor]]): List of target dictionaries.
            indices (List[Tuple[int, int]]): List of tuples of indices.
            num_boxes (int): Number of predicted boxes.
        """
        idx = self._get_src_permutation_idx(indices)

        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)
        ious = bbox_overlaps(
            box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
        )
        ious = torch.diag(ious).detach()

        src_logits = outputs["pred_logits"]
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target = nn.functional.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = ious.to(target_score_o.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        pred_score = nn.functional.sigmoid(src_logits).detach()
        weight = self.alpha * pred_score.pow(self.gamma) * (1 - target) + target_score

        loss = nn.functional.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction="none")
        loss = loss.mean(1).sum() * src_logits.shape[1] / num_boxes
        return {"loss_vfl": loss * self.loss_vfl_weight}

    def loss_boxes(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
        indices: list[tuple[int, int]],
        num_boxes: int,
    ) -> dict[str, torch.Tensor]:
        """Compute the losses re)L1 regression loss and the GIoU loss.

        Targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.

        Args:
            outputs (dict[str, torch.Tensor]): The outputs of the model.
            targets (list[dict[str, torch.Tensor]]): The targets.
            indices (list[tuple[int, int]]): The indices of the matched boxes.
            num_boxes (int): The number of boxes.

        Returns:
            dict[str, torch.Tensor]: The losses.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs["pred_boxes"][idx]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        loss_bbox = self.lossl1(src_boxes, target_boxes, avg_factor=num_boxes)
        loss_giou = self.giou(
            box_convert(src_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            avg_factor=num_boxes,
        )
        losses["loss_giou"] = loss_giou
        losses["loss_bbox"] = loss_bbox

        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        img_size = targets[0]["img_size"]
        # masks = [t["masks"] for t in targets]
        batch_polygons = [t["polygons"] for t in targets]

        masks = [
            torch.tensor(polygon_to_bitmap(polygons, *img_size), dtype=src_masks.dtype, device=src_masks.device)
            for polygons in batch_polygons
        ]

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks[tgt_idx]

        # stride: 32

        # upsample predictions to the target size
        src_masks = nn.functional.interpolate(
            src_masks[:, None],
            # size=target_masks.shape[-2:],
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        src_masks = src_masks[:, 0].flatten(1)

        # TODO: figure this out
        target_masks = target_masks[:, 2::4, 2::4]
        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(
        self,
        indices: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    @property
    def _available_losses(self) -> tuple[Callable]:
        return (self.loss_boxes, self.loss_labels_vfl)  # type: ignore[return-value]

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
            "bce": self.loss_labels_bce,
            "focal": self.loss_labels_focal,
            "vfl": self.loss_labels_vfl,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """This performs the loss computation.

        Args:
             outputs (dict[str, torch.Tensor]): dict of tensors, see the output
                specification of the model for the format
             targets (list[dict[str, torch.Tensor]]): list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        if "pred_boxes" not in outputs or "pred_logits" not in outputs:
            msg = "The model should return the predicted boxes and logits"
            raise ValueError(msg)

        outputs_without_aux = {k: v for k, v in outputs.items() if "aux" not in k}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(num_boxes)
            world_size = torch.distributed.get_world_size()
        num_boxes = torch.clamp(num_boxes / world_size, min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self._available_losses:
            l_dict = loss(outputs, targets, indices, num_boxes)
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self._available_losses:
                    if loss == "masks":
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # Logging is enabled only for the last layer
                        kwargs = {"log": False}

                    l_dict = loss(aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_aux_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # In case of cdn auxiliary losses. For rtdetr
        if "dn_aux_outputs" in outputs:
            if "dn_meta" not in outputs:
                msg = "dn_meta is not in outputs"
                raise ValueError(msg)
            indices = self.get_cdn_matched_indices(outputs["dn_meta"], targets)
            num_boxes = num_boxes * outputs["dn_meta"]["dn_num_group"]

            for i, aux_outputs in enumerate(outputs["dn_aux_outputs"]):
                # indices = self.matcher(aux_outputs, targets)
                for loss in self._available_losses:
                    kwargs = {}
                    l_dict = loss(aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_dn_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses

    @staticmethod
    def get_cdn_matched_indices(
        dn_meta: dict[str, list[torch.Tensor]],
        targets: list[dict[str, torch.Tensor]],
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """get_cdn_matched_indices.

        Args:
            dn_meta (dict[str, list[torch.Tensor]]): meta data for cdn
            targets (list[dict[str, torch.Tensor]]): targets
        """
        dn_positive_idx, dn_num_group = dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
        num_gts = [len(t["labels"]) for t in targets]
        device = targets[0]["labels"].device
        dn_match_indices = []
        for i, num_gt in enumerate(num_gts):
            if num_gt > 0:
                gt_idx = torch.arange(num_gt, dtype=torch.int64, device=device)
                gt_idx = gt_idx.tile(dn_num_group)
                if len(dn_positive_idx[i]) != len(gt_idx):
                    msg = f"len(dn_positive_idx[i]) != len(gt_idx), {len(dn_positive_idx[i])} != {len(gt_idx)}"
                    raise ValueError(msg)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append(
                    (
                        torch.zeros(0, dtype=torch.int64, device=device),
                        torch.zeros(0, dtype=torch.int64, device=device),
                    ),
                )

        return dn_match_indices
