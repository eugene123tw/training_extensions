"""This module implements the ResultMediaEntity."""
# Copyright (C) 2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from typing import Optional

import numpy as np

from otx.api.entities.annotation import Annotation, AnnotationSceneEntity
from otx.api.entities.label import LabelEntity
from otx.api.entities.metadata import IMetadata
from otx.api.entities.shapes.rectangle import Rectangle


# pylint: disable=too-many-instance-attributes; Requires refactor
class ResultMediaEntity(IMetadata):
    """Represents a media (e.g. an image which was generated by a task).

    For instance, a `ResultMediaEntity` could be an attention map generated by a classification task.

    The result media contains media data, which is associated with a
    `otx.api.entities.annotation.AnnotationSceneEntity` and related to an optional
    `otx.api.entities.label.LabelEntity`.

    Example:
        >>> from otx.api.entities.annotation import (
            Annotation,
            AnnotationSceneEntity,
            AnnotationSceneKind,
            )
        >>> from otx.api.entities.id import ID
        >>> from otx.api.entities.label import Domain, LabelEntity
        >>> from otx.api.entities.result_media import ResultMediaEntity
        >>> from otx.api.entities.scored_label import LabelSource, ScoredLabel
        >>> from otx.api.entities.shapes.rectangle import Rectangle

        >>> source = LabelSource(
                user_id="user_entity", model_id=ID("efficientnet"), model_storage_id=ID("efficientnet-storage")
                )
        >>> falcon_label = LabelEntity(name="Falcon", domain=Domain.DETECTION)
        >>> eagle_label = LabelEntity(name="Eagle", domain=Domain.DETECTION)
        >>> falcon_bb = Rectangle(x1=0.0, y1=0.0, x2=0.5, y2=0.5)
        >>> falcon_scored_label = ScoredLabel(label=falcon_label, probability=0.9, label_source=source)
        >>> eagle_bb = Rectangle(x1=0.2, y1=0.2, x2=0.8, y2=0.8)
        >>> eagle_scored_label = ScoredLabel(label=eagle_label, probability=0.6, label_source=source)
        >>> annotation_scene = AnnotationSceneEntity(
                 annotations=[
                     Annotation(shape=falcon_bb, labels=[falcon_scored_label]),
                     Annotation(shape=eagle_bb, labels=[eagle_scored_label]),
                 ], kind=AnnotationSceneKind.PREDICTION
             )
        >>> ResultMediaEntity(
                name="Model Predictions",
                type="Bounding Box Annotations",
                annotation_scene=annotation_scene,
                numpy=image_array
            )

    Args:
        name (str): Name.
        type (str): The type of data (e.g. Attention map). This type is descriptive.
        annotation_scene (AnnotationScene Entity): Associated annotation which was generated by the task
                                alongside this media.
        numpy (np.ndarray): The data as a numpy array.
        roi (Optional[Annotation]): The ROI covered by this media. If null, assume the entire image. Defaults to None.
        label (Optional[LabelEntity]): A label associated with this media. Defaults to None.
    """

    # pylint: disable=redefined-builtin, too-many-arguments;
    def __init__(
        self,
        name: str,
        type: str,
        annotation_scene: AnnotationSceneEntity,
        numpy: np.ndarray,
        roi: Optional[Annotation] = None,
        label: Optional[LabelEntity] = None,
    ):
        self.name = name
        self.type = type
        self.annotation_scene = annotation_scene
        self.roi = Annotation(Rectangle.generate_full_box(), labels=[]) if roi is None else roi
        self.label = label
        self._numpy = np.copy(numpy)

    def __repr__(self):
        """Returns a string with all the attributes of the ResultMediaEntity."""
        return (
            "ResultMediaEntity("
            f"name={self.name}, "
            f"type={self.type}, "
            f"annotation_scene={self.annotation_scene}, "
            f"roi={self.roi}, "
            f"label={self.label})"
        )

    @property
    def width(self) -> int:
        """Returns the width of the result media."""
        return self.numpy.shape[1]

    @property
    def height(self) -> int:
        """Returns the height of the result media."""
        return self.numpy.shape[0]

    @property
    def numpy(self) -> np.ndarray:
        """Returns the data."""
        return self._numpy

    @numpy.setter
    def numpy(self, value):
        self._numpy = value

    def __eq__(self, other):
        """Checks if the annotation_scene and roi matches with the other ResultMediaEntity."""
        if isinstance(other, ResultMediaEntity):
            return self.annotation_scene == other.annotation_scene and self.roi == other.roi
        return False