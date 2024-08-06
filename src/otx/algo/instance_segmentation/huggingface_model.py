# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""Huggingface universal segmentation implementations."""

from __future__ import annotations

from types import MethodType
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torch.cuda.amp import autocast
from torchvision import tv_tensors
from transformers import AutoModelForUniversalSegmentation, AutoProcessor
from transformers.models.mask2former.modeling_mask2former import (
    dice_loss,
    sample_point,
    sigmoid_cross_entropy_loss,
)
from transformers.models.oneformer.modeling_oneformer import (
    OneFormerHungarianMatcher,
    pair_wise_dice_loss,
    pair_wise_sigmoid_cross_entropy_loss,
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
        self.model_name_or_path = model_name_or_path
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
            pretrained_model_name_or_path=self.model_name_or_path,
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


class OTXOneFormerHungarianMatcher(OneFormerHungarianMatcher):
    @torch.no_grad()
    def forward(self, masks_queries_logits, class_queries_logits, mask_labels, class_labels) -> List[Tuple[Tensor]]:
        indices: list[tuple[np.array]] = []

        num_queries = class_queries_logits.shape[1]

        preds_masks = masks_queries_logits
        preds_probs = class_queries_logits
        # iterate through batch size
        for pred_probs, pred_mask, target_mask, labels in zip(preds_probs, preds_masks, mask_labels, class_labels):
            pred_probs = pred_probs.softmax(-1)
            # Compute the classification cost. Contrary to the loss, we don't use the NLL,
            # but approximate it in 1 - proba[target class].
            # The 1 is a constant that doesn't change the matching, it can be ommitted.
            cost_class = -pred_probs[:, labels]

            pred_mask = pred_mask[:, None]
            target_mask = target_mask[:, None].to(pred_mask.device)

            # all masks share the same set of points for efficient matching!
            point_coords = torch.rand(1, self.num_points, 2, device=pred_mask.device)

            # get ground truth labels
            target_mask = sample_point(
                target_mask.float(),
                point_coords.repeat(target_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            pred_mask = sample_point(
                pred_mask,
                point_coords.repeat(pred_mask.shape[0], 1, 1),
                align_corners=False,
            ).squeeze(1)

            with autocast(enabled=False):
                pred_mask = pred_mask.float()
                target_mask = target_mask.float()

                # compute the sigmoid ce loss
                cost_mask = pair_wise_sigmoid_cross_entropy_loss(pred_mask, target_mask)
                # Compute the dice loss
                cost_dice = pair_wise_dice_loss(pred_mask, target_mask)
                # final cost matrix
                cost_matrix = self.cost_mask * cost_mask + self.cost_class * cost_class + self.cost_dice * cost_dice
                cost_matrix = cost_matrix.reshape(num_queries, -1).cpu()
                # do the assigmented using the hungarian algorithm in scipy
                assigned_indices: tuple[np.array] = linear_sum_assignment(cost_matrix.cpu())
                indices.append(assigned_indices)

        # It could be stacked in one tensor
        matched_indices = [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices
        ]
        return matched_indices


class HuggingFaceOneFormerInstanceSeg(HuggingFaceModelForInstanceSeg):
    def __init__(
        self,
        model_name_or_path: str,
        label_info: LabelInfoTypes,
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MaskRLEMeanAPFMeasureCallable,
        torch_compile: bool = False,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            label_info=label_info,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
        )
        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            # NOTE: set is_training = True in order to randomly initialize a text encoder
            is_training=True,
        )
        self.task_token = self._generate_task_tokens()

    def _build_model(self, num_classes: int) -> nn.Module:
        model = AutoModelForUniversalSegmentation.from_pretrained(
            pretrained_model_name_or_path=self.model_name_or_path,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
            # NOTE: set is_training = True in order to randomly initialize a text encoder
            is_training=True,
        )
        model.matcher = OTXOneFormerHungarianMatcher(
            num_points=model.matcher.num_points,
            cost_class=model.matcher.cost_class,
            cost_mask=model.matcher.cost_mask,
            cost_dice=model.matcher.cost_dice,
        )
        model.criterion.loss_masks = MethodType(loss_masks, model.criterion)
        model.criterion.matcher = model.matcher
        return model

    def _generate_task_tokens(self, task: str = "instance"):
        return self.processor._preprocess_text([task])

    def _generate_txt_tokens(self, labels) -> Tensor:
        num_texts = self.model.config.num_queries - self.model.config.text_encoder_n_ctx
        text_list = ["an instance photo"] * num_texts
        for i, label in enumerate(labels[:num_texts]):
            text_list[i] = f"a photo with a {self.label_info.label_names[label]}"
        return self.processor._preprocess_text(text_list)

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        device = entity.images.device

        task_inputs = self.task_token.repeat(entity.batch_size, 1).to(device)
        if self.training:
            gt_masks = []
            text_inputs = []

            for img_info, polygons, masks, labels in zip(
                entity.imgs_info,
                entity.polygons,
                entity.masks,
                entity.labels,
            ):
                if len(masks) == 0:
                    masks = polygon_to_bitmap(polygons, *img_info.img_shape)
                gt_masks.append(tv_tensors.Mask(masks, device=device, dtype=torch.bool))
                text_inputs.append(self._generate_txt_tokens(labels).to(device))
            text_inputs = torch.stack(text_inputs, dim=0)
        return {
            "pixel_values": entity.images,
            "class_labels": entity.labels if self.training else None,
            "mask_labels": gt_masks if self.training else None,
            "text_inputs": text_inputs if self.training else None,
            "task_inputs": task_inputs,
        }

    def forward(
        self,
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        if self.training != self.model.model.is_training:
            self.model.model.is_training = self.training
            for layer in self.model.model.pixel_level_module.decoder.encoder.layers:
                layer.is_training = self.training
        return super().forward(inputs)
