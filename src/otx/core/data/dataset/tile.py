# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
"""OTX tile dataset."""

from __future__ import annotations

import logging as log
import time
from functools import wraps
from itertools import product
from typing import TYPE_CHECKING, Callable

import numpy as np
import torch
from datumaro import Bbox, DatasetItem, Image, Polygon
from datumaro import Dataset as DmDataset
from datumaro.components.media import BboxIntCoords, Image, RoIImage
from datumaro.plugins.tiling import Tile
from datumaro.plugins.tiling.util import (
    clip_x1y1x2y2,
    cxcywh_to_x1y1x2y2,
    x1y1x2y2_to_cxcywh,
    x1y1x2y2_to_xywh,
    xywh_to_x1y1x2y2,
)
from torchvision import tv_tensors

from otx.core.data.entity.base import ImageInfo
from otx.core.data.entity.detection import DetDataEntity
from otx.core.data.entity.instance_segmentation import InstanceSegDataEntity
from otx.core.data.entity.tile import (
    TileBatchDetDataEntity,
    TileBatchInstSegDataEntity,
    TileDetDataEntity,
    TileInstSegDataEntity,
)
from otx.core.types.task import OTXTaskType
from otx.core.utils.mask_util import polygon_to_bitmap

from .base import OTXDataset

if TYPE_CHECKING:
    from datumaro.components.media import BboxIntCoords

    from otx.core.config.data import TileConfig
    from otx.core.data.dataset.detection import OTXDetectionDataset
    from otx.core.data.dataset.instance_segmentation import OTXInstanceSegDataset
    from otx.core.data.entity.base import OTXDataEntity

# ruff: noqa: SLF001
# NOTE: Disable private-member-access (SLF001).
# This is a workaround so we could apply the same transforms to tiles as the original dataset.


def timing(f):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time.perf_counter()
        result = f(*args, **kw)
        te = time.perf_counter()
        print("func:%r took: %2.4f sec" % (f.__name__, te - ts))
        return result

    return wrap


class OTXTileTransform(Tile):
    """OTX tile transform.

    Different from the original Datumaro Tile transform,
    OTXTileTransform takes tile_size and overlap as input instead of grid size

    Args:
        extractor (DmDataset): Dataset subset to extract tiles from.
        tile_size (tuple[int, int]): Tile size.
        overlap (tuple[float, float]): Overlap ratio.
            Overlap values are clipped between 0 and 0.9 to ensure the stride is not too small.
        threshold_drop_ann (float): Threshold to drop annotations.
        with_full_img (bool): Include full image in the tiles.
    """

    def __init__(
        self,
        extractor: DmDataset,
        tile_size: tuple[int, int],
        overlap: tuple[float, float],
        with_full_img: bool,
    ) -> None:
        # NOTE: clip overlap to [0, 0.9]
        overlap = max(0, min(overlap[0], 0.9)), max(0, min(overlap[1], 0.9))
        super().__init__(
            extractor,
            (0, 0),
            overlap=overlap,
            threshold_drop_ann=1.0,
        )
        self._tile_size = tile_size
        self.with_full_img = with_full_img

    def shift_polygon(self, polygon: np.ndarray, offset: tuple[int, int]) -> np.ndarray:
        polygon[0::2] += offset[0]
        polygon[1::2] += offset[1]
        return polygon

    def tile_boxes_overlap(self, tile_box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Compute overlapping ratio over boxes.

        Args:
            tile_box (np.ndarray): box in shape (1, 4).
            boxes (np.ndarray): boxes in shape (N, 4).

        Returns:
            np.ndarray: matched indices.
        """
        x1, y1, x2, y2 = tile_box
        return (boxes[:, 0] > x1) & (boxes[:, 1] > y1) & (boxes[:, 2] < x2) & (boxes[:, 3] < y2)

    def transform_item(self, item: DatasetItem):
        items = []
        rois = self._extract_rois(item.media)

        gt_bboxes = []
        gt_polygons = []
        gt_labels = []
        for ann in item.annotations:
            if isinstance(ann, Bbox):
                gt_bboxes.append(ann.points)
                gt_labels.append(ann.label)
            elif isinstance(ann, Polygon):
                gt_polygons.append(np.array(ann.points))
        gt_labels = np.array(gt_labels, dtype=np.int64)
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)

        for idx, roi in enumerate(rois):
            roi_box = np.array(xywh_to_x1y1x2y2(*roi), dtype=np.float32)
            keep = self.tile_boxes_overlap(roi_box, gt_bboxes)

            if any(keep):
                tiled_labels = gt_labels[keep]
                tiled_bboxes = gt_bboxes[keep]
                tiled_polygons = [gt_polygon for gt_polygon, keep in zip(gt_polygons, keep) if keep]

                tiled_bboxes[:, 0] -= roi_box[0]
                tiled_bboxes[:, 1] -= roi_box[1]
                tiled_bboxes[:, 2] -= roi_box[0]
                tiled_bboxes[:, 3] -= roi_box[1]
                tiled_polygons = [self.shift_polygon(polygon, (-roi_box[0], -roi_box[1])) for polygon in tiled_polygons]

                annos = []
                for tile_box, tile_label, tile_polygon in zip(tiled_bboxes, tiled_labels, tiled_polygons):
                    x1, y1, w, h = x1y1x2y2_to_xywh(*tile_box)
                    bbox = Bbox(x=x1, y=y1, w=w, h=h, label=tile_label)
                    polygon = Polygon(points=tile_polygon, label=tile_label)
                    annos.extend([bbox, polygon])

                items += [
                    self.wrap_item(
                        item,
                        id=item.id + f"_tile_{idx}",
                        media=RoIImage.from_image(item.media, roi),
                        attributes=self._get_tiled_attributes(item, idx, roi),
                        annotations=annos,
                    ),
                ]
        return items

    def _extract_rois(self, image: Image) -> list[BboxIntCoords]:
        """Extracts Tile ROIs from the given image.

        Args:
            image (Image): Full image.

        Returns:
            list[BboxIntCoords]: list of ROIs.
        """
        if image.size is None:
            msg = "Image size is None"
            raise ValueError(msg)

        img_h, img_w = image.size
        tile_h, tile_w = self._tile_size
        h_ovl, w_ovl = self._overlap

        rois: list[BboxIntCoords] = []
        cols = range(0, img_w, int(tile_w * (1 - w_ovl)))
        rows = range(0, img_h, int(tile_h * (1 - h_ovl)))

        if self.with_full_img:
            rois += [x1y1x2y2_to_xywh(0, 0, img_w, img_h)]
        for offset_x, offset_y in product(cols, rows):
            x2 = min(offset_x + tile_w, img_w)
            y2 = min(offset_y + tile_h, img_h)
            c_x, c_y, w, h = x1y1x2y2_to_cxcywh(offset_x, offset_y, x2, y2)
            x1, y1, x2, y2 = cxcywh_to_x1y1x2y2(c_x, c_y, w, h)
            x1, y1, x2, y2 = clip_x1y1x2y2(x1, y1, x2, y2, img_w, img_h)
            x1, y1, x2, y2 = (int(v) for v in [x1, y1, x2, y2])
            rois += [x1y1x2y2_to_xywh(x1, y1, x2, y2)]

        log.info(f"image: {img_h}x{img_w} ~ tile_size: {self._tile_size}")
        log.info(f"{len(rows)}x{len(cols)} tiles -> {len(rois)} tiles")
        return rois


class OTXTileDatasetFactory:
    """OTX tile dataset factory."""

    @classmethod
    def create(
        cls,
        task: OTXTaskType,
        dataset: OTXDataset,
        tile_config: TileConfig,
    ) -> OTXTileDataset:
        """Create a tile dataset based on the task type and subset type.

        NOte: All task utilize the same OTXTileTrainDataset for training.
              In testing, we use different tile dataset for different task
              type due to different annotation format and data entity.

        Args:
            task (OTXTaskType): OTX task type.
            dataset (OTXDataset): OTX dataset.
            tile_config (TilerConfig): Tile configuration.

        Returns:
            OTXTileDataset: Tile dataset.
        """
        if dataset.dm_subset[0].subset == "train":
            return OTXTileTrainDataset(dataset, tile_config)

        if task == OTXTaskType.DETECTION:
            return OTXTileDetTestDataset(dataset, tile_config)
        if task in [OTXTaskType.ROTATED_DETECTION, OTXTaskType.INSTANCE_SEGMENTATION]:
            return OTXTileInstSegTestDataset(dataset, tile_config)
        msg = f"Unsupported task type: {task} for tiling"
        raise NotImplementedError(msg)


class OTXTileDataset(OTXDataset):
    """OTX tile dataset base class.

    Args:
        dataset (OTXDataset): OTX dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDataset, tile_config: TileConfig) -> None:
        super().__init__(
            dataset.dm_subset,
            dataset.transforms,
            dataset.mem_cache_handler,
            dataset.mem_cache_img_max_size,
            dataset.max_refetch,
        )
        self.tile_config = tile_config
        self._dataset = dataset

    def __len__(self) -> int:
        return len(self._dataset)

    @property
    def collate_fn(self) -> Callable:
        """Collate function from the original dataset."""
        return self._dataset.collate_fn

    def _get_item_impl(self, index: int) -> OTXDataEntity | None:
        """Get item implementation from the original dataset."""
        return self._dataset._get_item_impl(index)

    def _convert_entity(self, image: np.ndarray, dataset_item: DatasetItem) -> OTXDataEntity:
        """Convert a tile dataset item to OTXDataEntity."""
        msg = "Method _convert_entity is not implemented."
        raise NotImplementedError(msg)

    def get_tiles(self, image: np.ndarray, item: DatasetItem) -> tuple[list[OTXDataEntity], list[dict]]:
        """Retrieves tiles from the given image and dataset item.

        Args:
            image (np.ndarray): The input image.
            item (DatasetItem): The dataset item.

        Returns:
            A tuple containing two lists:
            - tile_entities (list[OTXDataEntity]): List of tile entities.
            - tile_attrs (list[dict]): List of tile attributes.
        """
        tile_ds = DmDataset.from_iterable([item])
        tile_ds = tile_ds.transform(
            OTXTileTransform,
            tile_size=self.tile_config.tile_size,
            overlap=(self.tile_config.overlap, self.tile_config.overlap),
            with_full_img=self.tile_config.with_full_img,
        )

        if item.subset == "val":
            # NOTE: filter validation tiles with annotations only to avoid evaluation on empty tiles.
            tile_ds = tile_ds.filter("/item/annotation", filter_annotations=True, remove_empty=True)

        tile_entities: list[OTXDataEntity] = []
        tile_attrs: list[dict] = []
        for tile in tile_ds:
            tile_entity = self._convert_entity(image, tile)
            # apply the same transforms as the original dataset
            transformed_tile = self._apply_transforms(tile_entity)
            if transformed_tile is None:
                msg = "Transformed tile is None"
                raise RuntimeError(msg)
            tile_entities.append(transformed_tile)
            tile_attrs.append(tile.attributes)
        return tile_entities, tile_attrs


class OTXTileTrainDataset(OTXTileDataset):
    """OTX tile train dataset.

    Args:
        dataset (OTXDataset): OTX dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDataset, tile_config: TileConfig) -> None:
        dm_dataset = dataset.dm_subset
        dm_dataset = dm_dataset.transform(
            OTXTileTransform,
            tile_size=tile_config.tile_size,
            overlap=(tile_config.overlap, tile_config.overlap),
            with_full_img=tile_config.with_full_img,
        )
        dataset.dm_subset = dm_dataset.filter("/item/annotation", filter_annotations=True, remove_empty=True)
        super().__init__(dataset, tile_config)


class OTXTileDetTestDataset(OTXTileDataset):
    """OTX tile detection test dataset.

    OTXTileDetTestDataset wraps a list of tiles (DetDataEntity) into a single TileDetDataEntity for testing/predicting.

    Args:
        dataset (OTXDetDataset): OTX detection dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXDetectionDataset, tile_config: TileConfig) -> None:
        super().__init__(dataset, tile_config)

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile detection test dataset."""
        return TileBatchDetDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileDetDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileDetDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileDetDataEntity: tile detection data entity that wraps a list of detection data entities.

        Note:
            Ignoring [override] check is necessary here since OTXDataset._get_item_impl exclusively permits
            the return of OTXDataEntity. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileDetDataEntity.
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        bbox_anns = [ann for ann in item.annotations if isinstance(ann, Bbox)]

        bboxes = (
            np.stack([ann.points for ann in bbox_anns], axis=0).astype(np.float32)
            if len(bbox_anns) > 0
            else np.zeros((0, 4), dtype=np.float32)
        )
        labels = torch.as_tensor([ann.label for ann in bbox_anns])

        tile_entities, tile_attrs = self.get_tiles(img_data, item)

        return TileDetDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            tile_attr_list=tile_attrs,
            ori_img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            ori_bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            ori_labels=labels,
        )

    def _convert_entity(self, image: np.ndarray, dataset_item: DatasetItem) -> DetDataEntity:
        """Convert a tile datumaro dataset item to DetDataEntity."""
        x1, y1, w, h = dataset_item.attributes["roi"]
        tile_img = image[y1 : y1 + h, x1 : x1 + w]
        tile_shape = tile_img.shape[:2]
        img_info = ImageInfo(
            img_idx=dataset_item.attributes["id"],
            img_shape=tile_shape,
            ori_shape=tile_shape,
        )
        return DetDataEntity(
            image=tile_img,
            img_info=img_info,
            # we don't need tile-level annotations
            bboxes=tv_tensors.BoundingBoxes(
                [],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=tile_shape,
            ),
            labels=torch.as_tensor([]),
        )


class OTXTileInstSegTestDataset(OTXTileDataset):
    """OTX tile inst-seg test dataset.

    OTXTileDetTestDataset wraps a list of tiles (InstanceSegDataEntity) into a single TileDetDataEntity
    for testing/predicting.

    Args:
        dataset (OTXInstanceSegDataset): OTX inst-seg dataset.
        tile_config (TilerConfig): Tile configuration.
    """

    def __init__(self, dataset: OTXInstanceSegDataset, tile_config: TileConfig) -> None:
        super().__init__(dataset, tile_config)

    @property
    def collate_fn(self) -> Callable:
        """Collate function for tile inst-seg test dataset."""
        return TileBatchInstSegDataEntity.collate_fn

    def _get_item_impl(self, index: int) -> TileInstSegDataEntity:  # type: ignore[override]
        """Get item implementation.

        Transform a single dataset item to multiple tiles using Datumaro tiling plugin, and
        wrap tiles into a single TileInstSegDataEntity.

        Args:
            index (int): Index of the dataset item.

        Returns:
            TileInstSegDataEntity: tile inst-seg data entity that wraps a list of inst-seg data entities.

        Note:
            Ignoring [override] check is necessary here since OTXDataset._get_item_impl exclusively permits
            the return of OTXDataEntity. Nevertheless, in instances involving tiling, it becomes
            imperative to encapsulate tiles within a unified entity, namely TileInstSegDataEntity.
        """
        item = self.dm_subset[index]
        img = item.media_as(Image)
        img_data, img_shape = self._get_img_data_and_shape(img)

        gt_bboxes, gt_labels, gt_masks, gt_polygons = [], [], [], []

        for annotation in item.annotations:
            if isinstance(annotation, Polygon):
                bbox = np.array(annotation.get_bbox(), dtype=np.float32)
                gt_bboxes.append(bbox)
                gt_labels.append(annotation.label)

                if self._dataset.include_polygons:
                    gt_polygons.append(annotation)
                else:
                    gt_masks.append(polygon_to_bitmap([annotation], *img_shape)[0])

        # convert xywh to xyxy format
        bboxes = np.array(gt_bboxes, dtype=np.float32)
        bboxes[:, 2:] += bboxes[:, :2]

        masks = np.stack(gt_masks, axis=0) if gt_masks else np.zeros((0, *img_shape), dtype=bool)
        labels = np.array(gt_labels, dtype=np.int64)

        tile_entities, tile_attrs = self.get_tiles(img_data, item)

        return TileInstSegDataEntity(
            num_tiles=len(tile_entities),
            entity_list=tile_entities,
            tile_attr_list=tile_attrs,
            ori_img_info=ImageInfo(
                img_idx=index,
                img_shape=img_shape,
                ori_shape=img_shape,
            ),
            ori_bboxes=tv_tensors.BoundingBoxes(
                bboxes,
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=img_shape,
            ),
            ori_labels=torch.as_tensor(labels),
            ori_masks=tv_tensors.Mask(masks, dtype=torch.uint8),
            ori_polygons=gt_polygons,
        )

    def _convert_entity(self, image: np.ndarray, dataset_item: DatasetItem) -> InstanceSegDataEntity:
        """Convert a tile dataset item to InstanceSegDataEntity."""
        x1, y1, w, h = dataset_item.attributes["roi"]
        tile_img = image[y1 : y1 + h, x1 : x1 + w]
        tile_shape = tile_img.shape[:2]
        img_info = ImageInfo(
            img_idx=dataset_item.attributes["id"],
            img_shape=tile_shape,
            ori_shape=tile_shape,
        )
        return InstanceSegDataEntity(
            image=tile_img,
            img_info=img_info,
            # we don't need tile-level annotations
            bboxes=tv_tensors.BoundingBoxes(
                [],
                format=tv_tensors.BoundingBoxFormat.XYXY,
                canvas_size=tile_shape,
            ),
            labels=torch.as_tensor([]),
            masks=tv_tensors.Mask(np.zeros((0, *tile_shape), dtype=bool)),
            polygons=[],
        )
