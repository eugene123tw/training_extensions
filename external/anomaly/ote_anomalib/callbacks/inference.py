"""
Inference Callbacks for OTE inference
"""

# Copyright (C) 2021 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.

from typing import Any, List

import numpy as np
import pytorch_lightning as pl
from anomalib.core.model.anomaly_module import AnomalyModule
from anomalib.utils.post_process import anomaly_map_to_color_map
from ote_anomalib.data import LabelNames
from ote_anomalib.logging import get_logger
from ote_sdk.entities.datasets import DatasetEntity
from ote_sdk.entities.label import LabelEntity
from ote_sdk.entities.result_media import ResultMediaEntity
from ote_sdk.entities.scored_label import ScoredLabel
from pytorch_lightning.callbacks import Callback
from ote_anomalib.data.utils import annotations_from_mask
from ote_sdk.entities.model_template import TaskType

logger = get_logger(__name__)


def get_inference_callback(task_type: str, ote_dataset: DatasetEntity, labels: List[LabelEntity]):
    if task_type == "classification":
        return AnomalyClassificationInferenceCallback(ote_dataset, labels)
    elif task_type == "segmentation":
        return AnomalySegmentationInferenceCallback(ote_dataset, labels)
    else:
        raise ValueError(f"Task type not recognized: {task_type}")


class AnomalyClassificationInferenceCallback(Callback):
    """Callback that updates the OTE dataset during inference."""

    def __init__(self, ote_dataset: DatasetEntity, labels: List[LabelEntity], task_type: TaskType):
        self.ote_dataset = ote_dataset
        self.normal_label = [label for label in labels if label.name == LabelNames.normal][0]
        self.anomalous_label = [label for label in labels if label.name == LabelNames.anomalous][0]
        self.task_type = task_type

    def on_predict_epoch_end(self, _trainer: pl.Trainer, pl_module: AnomalyModule, outputs: List[Any]):
        """Call when the predict epoch ends."""
        outputs = outputs[0]
        pred_scores = np.hstack([output["pred_scores"].cpu() for output in outputs])
        pred_labels = np.hstack([output["pred_labels"].cpu() for output in outputs])
        anomaly_maps = np.vstack([output["anomaly_maps"].cpu() for output in outputs])
        pred_masks = np.vstack([output["pred_masks"].cpu() for output in outputs])

        # Loop over dataset again to assign predictions
        for dataset_item, pred_score, pred_label, anomaly_map, pred_mask in zip(
            self.ote_dataset, pred_scores, pred_labels, anomaly_maps, pred_masks
        ):
            label = self.anomalous_label if pred_label else self.normal_label
            if self.task_type == TaskType.ANOMALY_CLASSIFICATION:
                probability = (1 - pred_score) if pred_score < 0.5 else pred_score
                dataset_item.append_labels([ScoredLabel(label=label, probability=float(probability))])
            elif self.task_type == TaskType.ANOMALY_SEGMENTATION:
                mask = pred_mask.squeeze().astype(np.uint8)
                dataset_item.append_annotations(annotations_from_mask(mask, self.normal_label, self.anomalous_label))
            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

            heatmap = anomaly_map_to_color_map(anomaly_map.squeeze(), normalize=False)
            heatmap_media = ResultMediaEntity(
                name="Anomaly Map",
                type="anomaly_map",
                annotation_scene=dataset_item.annotation_scene,
                numpy=heatmap,
            )
            dataset_item.append_metadata_item(heatmap_media)
            logger.info(
                "\n\tMin: %.3f, Max: %.3f, Threshold: %.3f, Assigned Label '%s', %.3f",
                pl_module.min_max.min.item(),
                pl_module.min_max.max.item(),
                pl_module.image_threshold.value.item(),
                label.name,
                pred_score,
            )


class AnomalySegmentationInferenceCallback(Callback):

    def __init__(self, ote_dataset: DatasetEntity, labels: List[LabelEntity]):
        raise NotImplementedError
