# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# NOTE: This module was contributed by Markus Pielmeier on PR #63
from __future__ import annotations

import importlib
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from kiss_icp.config.config import (
    AdaptiveThresholdConfig,
    DataConfig,
    MappingConfig,
)
from pydantic_settings import BaseSettings


class KISSConfig(BaseSettings):
    out_dir: str = "results"
    data: DataConfig = DataConfig()
    mapping: MappingConfig = MappingConfig()
    adaptive_threshold: AdaptiveThresholdConfig = AdaptiveThresholdConfig()


def _yaml_source(config_file: Optional[Path]) -> Dict[str, Any]:
    data = None
    if config_file is not None:
        try:
            yaml = importlib.import_module("yaml")
        except ModuleNotFoundError:
            print("Custom configuration file specified but PyYAML is not installed on your system,"
                  ' run `pip install "kiss-icp[all]"`. You can also modify the config.py if your '
                  "system does not support PyYaml ")
            sys.exit(1)
        with open(config_file) as cfg_file:
            data = yaml.safe_load(cfg_file)
    return data or {}


def load_config(config_file: Optional[Path], deskew: Optional[bool],
                max_range: Optional[float]) -> KISSConfig:
    """Load configuration from an Optional yaml file. Additionally, deskew and max_range can be
    also specified from the CLI interface"""

    config = KISSConfig(**_yaml_source(config_file))

    # Override defaults from command line
    if deskew is not None:
        config.data.deskew = deskew
    if max_range is not None:
        config.data.max_range = max_range

    # Check if there is a possible mistake
    if config.data.max_range < config.data.min_range:
        print("[WARNING] max_range is smaller than min_range, settng min_range to 0.0")
        config.data.min_range = 0.0

    # Use specified voxel size or compute one using the max range
    if config.mapping.voxel_size is None:
        config.mapping.voxel_size = float(config.data.max_range / 100.0)

    return config


def write_config(config: KISSConfig, filename: str):
    with open(filename, "w") as outfile:
        try:
            yaml = importlib.import_module("yaml")
            yaml.dump(config.model_dump(), outfile, default_flow_style=False)
        except ModuleNotFoundError:
            outfile.write(str(config.model_dump()))
