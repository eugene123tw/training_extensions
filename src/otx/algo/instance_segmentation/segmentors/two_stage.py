# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This class and its supporting functions are adapted from the mmdet.
# Please refer to https://github.com/open-mmlab/mmdetection/

"""TwoStageDetector."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from otx.algo.utils.mmengine_utils import InstanceData
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity


class TwoStageDetector(nn.Module):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        neck: nn.Module,
        rpn_head: nn.Module,
        roi_head: nn.Module,
        roi_criterion: nn.Module,
        rpn_criterion: nn.Module,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.roi_criterion = roi_criterion
        self.rpn_criterion = rpn_criterion

    def _load_from_state_dict(
        self,
        state_dict: dict,
        prefix: str,
        local_metadata: dict,
        strict: bool,
        missing_keys: list[str] | str,
        unexpected_keys: list[str] | str,
        error_msgs: list[str] | str,
    ) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage weights into two-stage model."""
        bbox_head_prefix = prefix + ".bbox_head" if prefix else "bbox_head"
        bbox_head_keys = [k for k in state_dict if k.startswith(bbox_head_prefix)]
        rpn_head_prefix = prefix + ".rpn_head" if prefix else "rpn_head"
        rpn_head_keys = [k for k in state_dict if k.startswith(rpn_head_prefix)]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + bbox_head_key[len(bbox_head_prefix) :]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN."""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    @property
    def with_neck(self) -> bool:
        """bool: whether the detector has a neck."""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_bbox(self) -> bool:
        """bool: whether the detector has a bbox head."""
        return (hasattr(self, "roi_head") and self.roi_head.with_bbox) or (
            hasattr(self, "bbox_head") and self.bbox_head is not None
        )

    def forward(
        self,
        entity: torch.Tensor,
        mode: str = "tensor",
    ) -> dict[str, torch.Tensor] | list[InstanceData] | tuple[torch.Tensor] | torch.Tensor:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`DetDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle either back propagation or
        parameter update, which are supposed to be done in :meth:`train_step`.

        Args:
            inputs (torch.Tensor): The input tensor with shape
                (N, C, ...) in general.
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
                Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`DetDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == "loss":
            return self.loss(entity)
        if mode == "predict":
            return self.predict(entity)
        msg = f"Invalid mode {mode}. Only supports loss and predict mode."
        raise RuntimeError(msg)

    def extract_feat(self, batch_inputs: Tensor) -> tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        x = self.backbone(batch_inputs)
        if self.with_neck:
            x = self.neck(x)
        return x

    def loss(self, batch_inputs: InstanceSegBatchDataEntity) -> dict:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.

        Returns:
            dict: A dictionary of loss components
        """
        x = self.extract_feat(batch_inputs.images)

        # Copy data entity and set gt_labels to 0 in RPN
        rpn_entity = InstanceSegBatchDataEntity(
            images=torch.empty(0),
            batch_size=batch_inputs.batch_size,
            imgs_info=batch_inputs.imgs_info,
            bboxes=batch_inputs.bboxes,
            masks=batch_inputs.masks,
            labels=[torch.zeros_like(labels) for labels in batch_inputs.labels],
            polygons=batch_inputs.polygons,
        )

        cls_reg_targets, bbox_preds, cls_scores, rpn_results_list = self.rpn_head.prepare_loss_inputs(
            x,
            rpn_entity,
        )

        losses = self.rpn_criterion(
            cls_reg_targets=cls_reg_targets,
            bbox_preds=bbox_preds,
            cls_scores=cls_scores,
        )

        (
            bbox_results,
            mask_results,
            cls_reg_targets_roi,
            mask_targets,
            pos_labels,
        ) = self.roi_head.prepare_loss_inputs(
            x,
            rpn_results_list,
            batch_inputs,
        )

        labels, label_weights, bbox_targets, bbox_weights, valid_label_mask = cls_reg_targets_roi

        roi_losses = self.roi_criterion(
            cls_score=bbox_results["cls_score"],
            bbox_pred=bbox_results["bbox_pred"],
            labels=labels,
            label_weights=label_weights,
            bbox_targets=bbox_targets,
            bbox_weights=bbox_weights,
            mask_preds=mask_results["mask_preds"],
            mask_targets=mask_targets,
            pos_labels=pos_labels,
            valid_label_mask=valid_label_mask,
        )

        losses.update(roi_losses)

        return losses

    def predict(
        self,
        entity: InstanceSegBatchDataEntity,
        rescale: bool = True,
    ) -> list[InstanceData]:
        """Predict results from a batch of inputs and data samples with post-processing."""
        if not self.with_bbox:
            msg = "Bbox head is not implemented."
            raise NotImplementedError(msg)
        x = self.extract_feat(entity.images)

        rpn_results_list = self.rpn_head.predict(x, entity, rescale=False)

        return self.roi_head.predict(x, rpn_results_list, entity, rescale=rescale)

    def export(
        self,
        batch_inputs: torch.Tensor,
        batch_img_metas: list[dict],
    ) -> tuple[torch.Tensor, ...]:
        """Export for two stage detectors."""
        x = self.extract_feat(batch_inputs)

        rpn_results_list = self.rpn_head.export(
            x,
            batch_img_metas,
            rescale=False,
        )

        return self.roi_head.export(
            x,
            rpn_results_list,
            batch_img_metas,
            rescale=False,
        )
