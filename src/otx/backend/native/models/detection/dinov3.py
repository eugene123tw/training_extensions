# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""DINOv3 model implementations."""

from __future__ import annotations

import copy
import re
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import torch
from torch import Tensor, nn
from torchvision.ops import box_convert
from torchvision.tv_tensors import BoundingBoxFormat


from otx.algo.detection.dinov3.eval.detection.models.backbone import build_dinov3_detection_backbone
from otx.algo.detection.dinov3.eval.detection.models.detr import PlainDETR
from otx.algo.detection.dinov3.eval.detection.models.position_encoding import PositionEncoding
from otx.algo.detection.dinov3.eval.detection.models.transformer import build_dinov3_transformer
from otx.algo.detection.dinov3.hub.backbones import dinov3_vit7b16
from otx.backend.native.exporter.base import OTXModelExporter
from otx.backend.native.exporter.native import OTXNativeModelExporter
from otx.backend.native.models.base import DataInputParams, DefaultOptimizerCallable, DefaultSchedulerCallable

from otx.backend.native.models.detection.base import OTXDetectionModel


from otx.backend.native.models.utils.utils import load_checkpoint
from otx.config.data import TileConfig
from otx.data.entity.base import OTXBatchLossEntity
from otx.data.entity.torch import OTXDataBatch, OTXPredBatch
from otx.metrics.fmeasure import MeanAveragePrecisionFMeasureCallable


from otx.backend.native.models.detection.rtdetr import RTDETR


if TYPE_CHECKING:
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable

    from otx.backend.native.schedulers import LRSchedulerListCallable
    from otx.metrics import MetricCallable
    from otx.types.label import LabelInfoTypes


class DINOv3DETR(RTDETR):

    pretrained_weights: ClassVar[dict[str, str]] = {
        "dinov3_7b": "/home/yuchunli/git/dinov3/weights/dinov3_vit7b16_combined.pth",
    }
    input_size_multiplier = 32

    def __init__(
        self,
        label_info: LabelInfoTypes,
        data_input_params: DataInputParams,
        model_name: Literal["dinov3_7b"] = "dinov3_7b",
        optimizer: OptimizerCallable = DefaultOptimizerCallable,
        scheduler: LRSchedulerCallable | LRSchedulerListCallable = DefaultSchedulerCallable,
        metric: MetricCallable = MeanAveragePrecisionFMeasureCallable,
        multi_scale: bool = False,
        torch_compile: bool = False,
        tile_config: TileConfig = TileConfig(enable_tiler=False),
    ) -> None:
        self.multi_scale = multi_scale
        super().__init__(
            label_info=label_info,
            data_input_params=data_input_params,
            model_name=model_name,
            optimizer=optimizer,
            scheduler=scheduler,
            metric=metric,
            torch_compile=torch_compile,
            tile_config=tile_config,
        )

    def _create_model(self, num_classes: int | None = None) -> PlainDETR:
        num_classes = num_classes if num_classes is not None else self.num_classes

        backbone = dinov3_vit7b16(pretrained=False)

        backbone = build_dinov3_detection_backbone(
            backbone,
            position_embedding=PositionEncoding.SINE,
            hidden_dim=768,
            num_feature_levels=1,
            n_windows_sqrt=3,
        )

        transformer = build_dinov3_transformer(
            d_model=768,
            nhead=8,
            num_feature_levels=1,
            two_stage=True,
            two_stage_num_proposals=600, # 300 one2one + 300 one2many
            mixed_selection=True,
            norm_type="pre_norm",
            decoder_type="global_rpe_decomp",
            proposal_feature_levels=4,
            proposal_in_stride=16,
            proposal_tgt_strides=[8, 16, 32, 64],
            proposal_min_size=50,
            add_transformer_encoder=True,
            num_encoder_layers=6,
        )

        optimizer_configuration = [
            # no weight decay for norm layers in backbone
            {"params": "^(?=.*backbone)(?=.*norm).*$", "weight_decay": 0.0, "lr": 0.00001},
            # lr for the backbone, but not norm layers is 0.00001
            {"params": "^(?=.*backbone)(?!.*norm).*$", "lr": 0.00001},
            # no weight decay for norm layers and biases in encoder and decoder layers
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bias)).*$", "weight_decay": 0.0},
        ]


        model = PlainDETR(
            backbone=backbone,
            transformer=transformer,
            num_classes=num_classes,
            num_feature_levels=1,
            aux_loss=True,
            with_box_refine=True,
            two_stage=True,
            num_queries_one2one=300, 
            num_queries_one2many=300,
            mixed_selection=True,
            optimizer_configuration=optimizer_configuration,
        )
        
        model.init_weights()
        load_checkpoint(model, self.pretrained_weights[self.model_name], map_location="cpu")

        # freeze backbone
        for param in model.backbone.parameters():
            param.requires_grad = False

        # freeze transformer.encoder
        for param in model.transformer.encoder.parameters():
            param.requires_grad = False

        return model

    def _customize_inputs(
        self,
        entity: OTXDataBatch,
        pad_size_divisor: int = 32,
        pad_value: int = 0,
    ) -> dict[str, Any]:
        targets: list[dict[str, Any]] = []
        # prepare bboxes for the model
        if entity.bboxes is not None and entity.labels is not None:
            for bb, ll in zip(entity.bboxes, entity.labels):
                # convert to cxcywh if needed
                if len(scaled_bboxes := bb):
                    converted_bboxes = (
                        box_convert(bb, in_fmt="xyxy", out_fmt="cxcywh") if bb.format == BoundingBoxFormat.XYXY else bb
                    )
                    # normalize the bboxes
                    scaled_bboxes = converted_bboxes / torch.tensor(bb.canvas_size[::-1]).tile(2)[None].to(
                        converted_bboxes.device,
                    )
                targets.append({"boxes": scaled_bboxes, "labels": ll})

        if self.explain_mode:
            return {"entity": entity}

        return {
            "images": entity.images,
            "targets": targets,
        }

    def _customize_outputs(
        self,
        outputs: list[torch.Tensor] | dict,  # type: ignore[override]
        inputs: OTXDataBatch,
    ) -> OTXPredBatch | OTXBatchLossEntity:
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

        original_sizes = [img_info.ori_shape for img_info in inputs.imgs_info]  # type: ignore[union-attr]
        scores, bboxes, labels = self.model.postprocess(outputs, original_sizes)

        if self.explain_mode:
            if not isinstance(outputs, dict):
                msg = f"Model output should be a dict, but got {type(outputs)}."
                raise ValueError(msg)

            if "feature_vector" not in outputs:
                msg = "No feature vector in the model output."
                raise ValueError(msg)

            if "saliency_map" not in outputs:
                msg = "No saliency maps in the model output."
                raise ValueError(msg)

            return OTXPredBatch(
                batch_size=len(outputs),
                images=inputs.images,
                imgs_info=inputs.imgs_info,
                scores=scores,
                bboxes=bboxes,
                labels=labels,
                feature_vector=[feature_vector.unsqueeze(0) for feature_vector in outputs["feature_vector"]],
                saliency_map=[saliency_map.to(torch.float32) for saliency_map in outputs["saliency_map"]],
            )

        return OTXPredBatch(
            batch_size=len(outputs),
            images=inputs.images,
            imgs_info=inputs.imgs_info,
            scores=scores,
            bboxes=bboxes,
            labels=labels,
        )

    def configure_optimizers(self) -> tuple[list[torch.optim.Optimizer], list[dict[str, Any]]]:
        """Configure an optimizer and learning-rate schedulers.

        Configure an optimizer and learning-rate schedulers
        from the given optimizer and scheduler or scheduler list callable in the constructor.
        Generally, there is two lr schedulers. One is for a linear warmup scheduler and
        the other is the main scheduler working after the warmup period.

        Returns:
            Two list. The former is a list that contains an optimizer
            The latter is a list of lr scheduler configs which has a dictionary format.
        """
        param_groups = self._get_optim_params(self.model.optimizer_configuration, self.model)
        optimizer = self.optimizer_callable(param_groups)
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
    def _get_optim_params(cfg: list[dict[str, Any]] | None, model: nn.Module) -> list[dict[str, Any]]:
        """Perform no bias decay and learning rate correction for the modules.

        The configuration dict should consist of regular expression pattern for the model parameters with "params" key.
        Other optimizer parameters can be added as well.

        E.g.:
            cfg = [{"params": "^((?!b).)*$", "lr": 0.01, "weight_decay": 0.0}, ..]
            The above configuration is for the parameters that do not contain "b".

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
            if "params" not in pg:
                msg = f"The 'params' key should be included in the configuration, but got {pg.keys()}"
                raise ValueError(msg)
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

    @property
    def _exporter(self) -> OTXModelExporter:
        """Creates OTXModelExporter object that can export the model."""
        return OTXNativeModelExporter(
            task_level_export_parameters=self._export_parameters,
            data_input_params=self.data_input_params,
            resize_mode="standard",
            swap_rgb=False,
            via_onnx=False,
            onnx_export_configuration={
                "input_names": ["images"],
                "output_names": ["bboxes", "labels", "scores"],
                "dynamic_axes": {
                    "images": {0: "batch"},
                    "boxes": {0: "batch", 1: "num_dets"},
                    "labels": {0: "batch", 1: "num_dets"},
                    "scores": {0: "batch", 1: "num_dets"},
                },
                "autograd_inlining": False,
                "opset_version": 16,
            },
            output_names=["bboxes", "labels", "scores"],
        )

    @property
    def _optimization_config(self) -> dict[str, Any]:
        """PTQ config for RT-DETR."""
        return {"model_type": "transformer"}

    @staticmethod
    def _forward_explain_detection(
        self,  # noqa: ANN001
        entity: OTXDataBatch,
        mode: str = "tensor",  # noqa: ARG004
    ) -> dict[str, torch.Tensor]:
        """Forward function for explainable detection model."""
        backbone_feats = self.encoder(self.backbone(entity.images))
        predictions = self.decoder(backbone_feats, explain_mode=True)

        raw_logits = DETR.split_and_reshape_logits(
            backbone_feats,
            predictions["raw_logits"],
        )

        saliency_map = self.explain_fn(raw_logits)
        feature_vector = self.feature_vector_fn(backbone_feats)
        predictions.update(
            {
                "feature_vector": feature_vector,
                "saliency_map": saliency_map,
            },
        )

        return predictions
