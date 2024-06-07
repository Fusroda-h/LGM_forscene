from dataclasses import dataclass

from .view_sampler import ViewSamplerCfg
from typing import Optional


@dataclass
class DatasetCfgCommon:
    image_shape: list[int]
    background_color: list[float]
    cameras_are_circular: bool
    overfit_to_scene: Optional[str]
    view_sampler: ViewSamplerCfg
