import copy
import math
import re
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision import tv_tensors

from otx.algo.common.utils.utils import filter_scores_and_topk
from otx.algo.detection.backbones import PResNet
from otx.algo.detection.losses import RTDetrCriterion
from otx.algo.detection.necks import HybridEncoder
from otx.algo.detection.rtdetr import RTDETR
from otx.algo.instance_segmentation.heads.rtdetr_insg_seg_decoder import RTDETRInstSegTransformer
from otx.core.data.entity.base import ImageInfo, OTXBatchLossEntity
from otx.core.data.entity.instance_segmentation import InstanceSegBatchDataEntity, InstanceSegBatchPredEntity
from otx.core.model.instance_segmentation import ExplainableOTXInstanceSegModel


class RTDETR_INST(RTDETR):
    def _forward_features(self, images: Tensor, targets: dict[str, Any] | None = None):
        feats = self.backbone(images)
        encoded_feats, last_proj = self.encoder(feats)
        return self.decoder(encoded_feats, feats, last_proj, targets)

    def forward(
        self,
        images: Tensor,
        imgs_info: list[ImageInfo],
        targets: list[dict] | None = None,
    ):
        if self.multi_scale and self.training:
            sz = int(np.random.choice(self.multi_scale))
            images = F.interpolate(images, size=[sz, sz])

        for image, target in zip(images, targets):
            target["img_size"] = image.shape[-2:]

        output = self._forward_features(images, targets)
        if self.training:
            return self.criterion(output, targets)
        return self.postprocess(output, imgs_info)

    def postprocess(self, outputs: dict, imgs_info: list[ImageInfo], deploy_mode=False):
        scores: list[Tensor] = []
        bboxes: list[tv_tensors.BoundingBoxes] = []
        labels: list[torch.LongTensor] = []
        masks: list[tv_tensors.Mask] = []

        h, w = imgs_info[0].img_shape
        ori_h, ori_w = imgs_info[0].ori_shape
        scale_factor = [1 / s for s in imgs_info[0].scale_factor]  # h, w

        for pred_scores, pred_boxes, pred_masks in zip(
            outputs["pred_logits"],
            outputs["pred_boxes"],
            outputs["pred_masks"],
        ):
            pred_scores = pred_scores.sigmoid()
            pred_scores, pred_labels, keep_idxs, _ = filter_scores_and_topk(pred_scores, 0.05, self.num_top_queries)
            pred_boxes = pred_boxes[keep_idxs]
            pred_masks = pred_masks.sigmoid()[keep_idxs]

            pred_boxes = torchvision.ops.box_convert(pred_boxes, in_fmt="cxcywh", out_fmt="xyxy")
            pred_boxes *= pred_boxes.new_tensor([w, h]).repeat((1, 2))
            pred_boxes *= pred_boxes.new_tensor(scale_factor[::-1]).repeat((1, 2))

            pred_boxes[:, 0::2].clamp_(min=0, max=ori_w - 1)
            pred_boxes[:, 1::2].clamp_(min=0, max=ori_h - 1)
            keep_idxs = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]) > 0

            pred_boxes = pred_boxes[keep_idxs > 0]
            pred_labels = pred_labels[keep_idxs > 0]
            pred_scores = pred_scores[keep_idxs > 0]
            pred_masks = pred_masks[keep_idxs > 0]

            if len(pred_boxes):
                pred_masks = torch.nn.functional.interpolate(
                    pred_masks.unsqueeze(0),
                    size=(math.ceil(h * scale_factor[0]), math.ceil(w * scale_factor[1])),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)[..., :ori_h, :ori_w]
                pred_masks = pred_masks > 0.5
                scores.append(pred_scores)
                bboxes.append(tv_tensors.BoundingBoxes(pred_boxes, format="xyxy", canvas_size=(ori_h, ori_w)))
                labels.append(pred_labels)
                masks.append(tv_tensors.Mask(pred_masks, dtype=torch.bool))

        return scores, bboxes, labels, masks


class OTX_RTDETR_INST(ExplainableOTXInstanceSegModel):
    image_size = (1, 3, 640, 640)
    mean = (0.0, 0.0, 0.0)
    std = (255.0, 255.0, 255.0)

    def _customize_inputs(self, entity: InstanceSegBatchDataEntity) -> dict[str, Any]:
        return {
            "images": entity.images,
            "imgs_info": entity.imgs_info,
            "targets": [
                {"boxes": bb, "labels": ll, "masks": masks, "polygons": polygons}
                for bb, ll, masks, polygons in zip(entity.bboxes, entity.labels, entity.masks, entity.polygons)
            ],
        }

    def _customize_outputs(
        self,
        outputs: tuple[Tensor, Tensor, Tensor, Tensor],
        inputs: InstanceSegBatchDataEntity,
    ) -> InstanceSegBatchPredEntity | OTXBatchLossEntity:
        if self.training:
            if not isinstance(outputs, dict):
                raise TypeError(outputs)

            losses = OTXBatchLossEntity()
            for k, v in outputs.items():
                if isinstance(v, list):
                    losses[k] = sum(v)
                elif isinstance(v, Tensor):
                    losses[k] = v
                else:
                    msg = "Loss output should be list or torch.tensor but got {type(v)}"
                    raise TypeError(msg)
            return losses

        saliency_map = []  # TODO add saliency map and XAI feature
        feature_vector = []
        scores, bboxes, labels, masks = outputs

        return InstanceSegBatchPredEntity(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
            masks=masks,
            polygons=[],
            saliency_map=saliency_map,
            feature_vector=feature_vector,
        )

    def get_num_anchors(self) -> list[int]:
        """Gets the anchor configuration from model."""
        # TODO update anchor configuration

        return [1] * 10

    def configure_optimizers(self):
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        param_groups = self.get_optim_params(self.model.optimizer_configuration, self.model)
        optimizer = torch.optim.AdamW(param_groups, lr=1e-4, weight_decay=1e-4)
        # optimizer = self.optimizer_callable(param_groups)
        schedulers = self.scheduler_callable(optimizer)

        def ensure_list(item: Any) -> list:  # noqa: ANN401
            return item if isinstance(item, list) else [item]

        lr_scheduler_configs = []
        for scheduler in ensure_list(schedulers):
            lr_scheduler_config = {"scheduler": scheduler}
            if hasattr(scheduler, "interval"):
                lr_scheduler_config["interval"] = scheduler.interval
            if hasattr(scheduler, "monitor"):
                lr_scheduler_config["monitor"] = scheduler.monitor
            lr_scheduler_configs.append(lr_scheduler_config)

        return [optimizer], lr_scheduler_configs

    @staticmethod
    def get_optim_params(cfg: list[dict[str, Any]] | None, model: nn.Module):
        """Perform no bias decay and learning rate correction for the modules.
        E.g.:
            ^(?=.*a)(?=.*b).*$         means including a and b
            ^((?!b.)*a((?!b).)*$       means including a but not b
            ^((?!b|c).)*a((?!b|c).)*$  means including a but not (b | c)
        """
        if cfg is None:
            return model.parameters()

        cfg = copy.deepcopy(cfg)

        param_groups = []
        visited = []
        for pg in cfg:
            pattern = pg["params"]
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and len(re.findall(pattern, k)) > 0}
            pg["params"] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        names = [k for k, v in model.named_parameters() if v.requires_grad]

        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters() if v.requires_grad and k in unseen}
            param_groups.append({"params": params.values()})
            visited.extend(list(params.keys()))

        return param_groups


class RTDetrInstResNet18(OTX_RTDETR_INST):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r18vd_5x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(
            depth=18,
            pretrained=True,
            freeze_at=-1,
            return_idx=[1, 2, 3],
            num_stages=4,
            freeze_norm=False,
        )
        encoder = HybridEncoder(
            in_channels=[128, 256, 512],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            expansion=0.5,
            dim_feedforward=1024,
            eval_spatial_size=self.image_size[2:],
        )
        decoder = RTDETRInstSegTransformer(
            num_classes=num_classes,
            num_decoder_layers=3,
            backbone_feat_channels=[128, 256, 512],
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            eval_spatial_size=self.image_size[2:],
        )
        criterion = RTDetrCriterion(
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 5, "loss_giou": 2, "loss_mask": 1, "loss_dice": 1},
            losses=["vfl", "boxes", "masks"],
            num_classes=num_classes,
            gamma=2.0,
            alpha=0.75,
        )
        optimizer_configuration = [
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": 0.00001},
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": 0.00001},
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]

        return RTDETR_INST(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            num_classes=num_classes,
            criterion=criterion,
            optimizer_configuration=optimizer_configuration,
            multi_scale=[],
        )


class RTDetrInstResNet50(OTX_RTDETR_INST):
    load_from = (
        "https://github.com/lyuwenyu/storage/releases/download/v0.1/rtdetr_r50vd_2x_coco_objects365_from_paddle.pth"
    )

    def _build_model(self, num_classes: int) -> nn.Module:
        backbone = PResNet(depth=50, return_idx=[1, 2, 3], num_stages=4, freeze_norm=True, pretrained=True, freeze_at=0)
        encoder = HybridEncoder(
            in_channels=[512, 1024, 2048],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            expansion=1.0,
            dim_feedforward=1024,
            eval_spatial_size=self.image_size[2:],
        )
        decoder = RTDETRInstSegTransformer(
            num_classes=num_classes,
            backbone_feat_channels=[512, 1024, 2048],
            feat_channels=[256, 256, 256],
            feat_strides=[8, 16, 32],
            hidden_dim=256,
            num_levels=3,
            num_queries=300,
            eval_spatial_size=self.image_size[2:],
            num_decoder_layers=6,
            num_denoising=100,
            eval_idx=-1,
        )

        criterion = RTDetrCriterion(
            weight_dict={"loss_vfl": 1.0, "loss_bbox": 5, "loss_giou": 2, "loss_mask": 1, "loss_dice": 1},
            losses=["vfl", "boxes", "masks"],
            num_classes=num_classes,
            gamma=2.0,
            alpha=0.75,
        )

        optimizer_configuration = [
            {"params": "backbone", "lr": 0.00001},
            {"params": "^(?=.*decoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
            {"params": "^(?=.*encoder(?=.*bias|.*norm.*weight)).*$", "weight_decay": 0.0},
        ]

        return RTDETR_INST(
            backbone=backbone,
            encoder=encoder,
            decoder=decoder,
            criterion=criterion,
            num_classes=num_classes,
            optimizer_configuration=optimizer_configuration,
        )
