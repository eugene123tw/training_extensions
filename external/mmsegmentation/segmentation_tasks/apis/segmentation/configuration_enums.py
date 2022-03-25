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

from ote_sdk.configuration import ConfigurableEnum

class Models(ConfigurableEnum):
    """
    This Enum represents the types of models for inference
    """
    Segmentation = 'segmentation'
    BlurSegmentation = 'blur_segmentation'

class POTQuantizationPreset(ConfigurableEnum):
    """
    This Enum represents the quantization preset for post training optimization
    """

    PERFORMANCE = 'Performance'
    MIXED = 'Mixed'