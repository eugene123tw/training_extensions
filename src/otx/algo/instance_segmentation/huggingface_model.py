# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Huggingface universal segmentation implementations."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn
from torchvision import tv_tensors
from transformers import AutoImageProcessor
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerConfig,
    Mask2FormerForUniversalSegmentation,
    Mask2FormerLoss,
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


class OTXMask2FormerLoss(Mask2FormerLoss):
    def select_masks(self, tgt_idx, mask_labels):
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
        target_masks = self.select_masks(tgt_idx, mask_labels)

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


class OTXMask2FormerForUniversalSegmentation(Mask2FormerForUniversalSegmentation):
    def __init__(self, config: Mask2FormerConfig):
        super().__init__(config)
        self.criterion = OTXMask2FormerLoss(config=config, weight_dict=self.weight_dict)


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
        self.load_from = None

        super().__init__(
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.image_processor = AutoImageProcessor.from_pretrained(self.model_name)

    def _build_model(self, num_classes: int) -> nn.Module:
        # TODO: change this to universal segmentation model
        return OTXMask2FormerForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        gt_masks = []
        for img_info, polygons in zip(entity.imgs_info, entity.polygons):
            img_h, img_w = img_info.img_shape
            masks = polygon_to_bitmap(polygons, img_h, img_w)
            gt_masks.append(tv_tensors.Mask(masks, device=img_info.device, dtype=torch.bool))

        return {
            "pixel_values": entity.images,
            "class_labels": entity.labels,
            "mask_labels": gt_masks,
        }

    def _customize_outputs(
        self,
        outputs: ModelOutput,
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            return outputs.loss

        target_sizes = [(max(m.shape), max(m.shape)) for m in inputs.masks]
        results = self.image_processor.post_process_instance_segmentation(
            outputs,
            threshold=0.0,
            target_sizes=target_sizes,
            return_binary_maps=True,
        )

        scores: list[Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        for img_info, pred in zip(inputs.imgs_info, results):
            ori_h, ori_w = img_info.ori_shape
            scores.append(
                torch.tensor(
                    [r["score"] for r in pred["segments_info"]],
                    device=img_info.device,
                ),
            )
            labels.append(
                torch.tensor(
                    [r["label_id"] for r in pred["segments_info"]],
                    device=img_info.device,
                ),
            )
            bit_masks = tv_tensors.Mask(
                pred["segmentation"],
                dtype=torch.bool,
            )[:, :ori_h, :ori_w]
            masks.append(bit_masks)
            bboxes.append(
                tv_tensors.BoundingBoxes(
                    mask2bbox(bit_masks),
                    format="XYXY",
                    canvas_size=img_info.ori_shape,
                ),
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
