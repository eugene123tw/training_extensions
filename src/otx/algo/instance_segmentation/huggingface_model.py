# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Huggingface universal segmentation implementations."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from transformers import AutoModelForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import (
    dice_loss,
    sample_point,
    sigmoid_cross_entropy_loss,
)

from otx.core.data.entity.base import OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.metrics.mean_ap import MaskRLEMeanAPFMeasureCallable
from otx.core.model.base import DefaultOptimizerCallable, DefaultSchedulerCallable
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel
from otx.core.schedulers import LRSchedulerListCallable
from otx.core.types.label import LabelInfoTypes
from otx.core.utils.mask_util import mask2bbox, polygon_to_bitmap

if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
    from transformers.utils.generic import ModelOutput

    from otx.core.metrics import MetricCallable


def select_masks(tgt_idx, mask_labels):
    gt_masks = []
    batch_size = torch.max(tgt_idx[0]) + 1
    for b in range(batch_size):
        gt_masks.append(mask_labels[b][tgt_idx[1][tgt_idx[0] == b]])
    return torch.cat(gt_masks, dim=0)


def loss_masks(
    self,
    masks_queries_logits: torch.Tensor,
    mask_labels: list[torch.Tensor],
    indices: tuple[np.array],
    num_masks: int,
) -> dict[str, torch.Tensor]:
    src_idx = self._get_predictions_permutation_indices(indices)
    tgt_idx = self._get_targets_permutation_indices(indices)
    # shape (batch_size * num_queries, height, width)
    pred_masks = masks_queries_logits[src_idx]
    # shape (batch_size, num_queries, height, width)

    # NOTE: instead of padding the masks, we select the masks without creating unnecessary masks
    target_masks = select_masks(tgt_idx, mask_labels)

    # No need to upsample predictions as we are using normalized coordinates
    pred_masks = pred_masks[:, None]
    target_masks = target_masks[:, None]

    # Sample point coordinates
    with torch.no_grad():
        point_coordinates = self.sample_points_using_uncertainty(
            pred_masks,
            lambda logits: self.calculate_uncertainty(logits),
            self.num_points,
            self.oversample_ratio,
            self.importance_sample_ratio,
        )

        point_labels = sample_point(target_masks.float(), point_coordinates, align_corners=False).squeeze(1)

    point_logits = sample_point(pred_masks, point_coordinates, align_corners=False).squeeze(1)

    losses = {
        "loss_mask": sigmoid_cross_entropy_loss(point_logits, point_labels, num_masks),
        "loss_dice": dice_loss(point_logits, point_labels, num_masks),
    }

    del pred_masks
    del target_masks
    return losses


class HuggingFaceModelForInstanceSeg(ExplainableOTXInstanceSegModel):
    def __init__(
        self,
        model_name_or_path: str,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
    ) -> None:
        self.model_name = model_name_or_path
        # NOTE: huggingface model loaded in _build_model phase
        self.load_from = None

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )

    def _build_model(self, num_classes: int) -> nn.Module:
        model = AutoModelForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
        model.criterion.loss_masks = MethodType(loss_masks, model.criterion)
        return model

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        if self.training:
            gt_masks = []
            for img_info, polygons, masks in zip(entity.imgs_info, entity.polygons, entity.masks):
                if len(masks) == 0:
                    masks = polygon_to_bitmap(polygons, *img_info.img_shape)
                gt_masks.append(tv_tensors.Mask(masks, device=img_info.device, dtype=torch.bool))

        return {
            "pixel_values": entity.images,
            "class_labels": entity.labels if self.training else None,
            "mask_labels": gt_masks if self.training else None,
        }

    def _customize_outputs(
        self,
        outputs: ModelOutput,
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return outputs.loss

        target_sizes = [(max(m.shape), max(m.shape)) for m in inputs.masks]
        masks, bboxes, labels, scores = self.post_process_instance_segmentation(
            outputs,
            inputs.imgs_info,
            target_sizes=target_sizes,
        )

        if self.explain_mode:
            msg = "Explain mode is not supported yet."
            raise NotImplementedError(msg)

        return InstanceSegBatchPredEntity(
            batch_size=inputs.batch_size,
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            masks=masks,
            polygons=[],
            labels=labels,
        )

    def post_process_instance_segmentation(
        self,
        outputs,
        imgs_info,
        target_sizes: list[tuple[int, int]] | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        # [batch_size, num_queries, num_classes+1]
        class_queries_logits = outputs.class_queries_logits
        # [batch_size, num_queries, height, width]
        masks_queries_logits = outputs.masks_queries_logits

        # Scale back to preprocessed image size - (384, 384) for all models
        masks_queries_logits = torch.nn.functional.interpolate(
            masks_queries_logits,
            size=(384, 384),
            mode="bilinear",
            align_corners=False,
        )

        device = masks_queries_logits.device
        num_classes = class_queries_logits.shape[-1] - 1
        num_queries = class_queries_logits.shape[-2]

        batch_scores: list[Tensor] = []
        batch_bboxes: list[tv_tensors.BoundingBoxes] = []
        batch_labels: list[torch.LongTensor] = []
        batch_masks: list[tv_tensors.Mask] = []

        for mask_pred, mask_cls, img_info, target_size in zip(
            masks_queries_logits,
            class_queries_logits,
            imgs_info,
            target_sizes,
        ):
            ori_h, ori_w = img_info.ori_shape
            scores = torch.nn.functional.softmax(mask_cls, dim=-1)[:, :-1]
            labels = torch.arange(num_classes, device=device).unsqueeze(0).repeat(num_queries, 1).flatten(0, 1)

            scores_per_image, topk_indices = scores.flatten(0, 1).topk(num_queries, sorted=False)
            labels_per_image = labels[topk_indices]

            topk_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
            mask_pred = mask_pred[topk_indices]
            pred_masks = (mask_pred > 0).float()

            # Calculate average mask prob
            mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (
                pred_masks.flatten(1).sum(1) + 1e-6
            )
            pred_scores = scores_per_image * mask_scores_per_image
            pred_classes = labels_per_image
            pred_masks = torch.nn.functional.interpolate(
                pred_masks.unsqueeze(0),
                size=target_size,
                mode="nearest",
            )[0][:, :ori_h, :ori_w]

            pred_boxes = mask2bbox(pred_masks)

            keep = (pred_masks.sum((1, 2)) > 10) & (pred_scores > 0.05)
            batch_masks.append(pred_masks[keep])
            batch_bboxes.append(pred_boxes[keep])
            batch_labels.append(pred_classes[keep])
            batch_scores.append(pred_scores[keep])

        return batch_masks, batch_bboxes, batch_labels, batch_scores
