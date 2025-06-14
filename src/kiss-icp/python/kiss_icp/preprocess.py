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
import numpy as np
from kiss_icp.config import KISSConfig
from kiss_icp.pybind import kiss_icp_pybind


def get_preprocessor(config: KISSConfig):
    return Preprocessor(config) if config.data.preprocess else Stubcessor()


class Stubcessor:

    def __call__(self, frame: np.ndarray):
        return frame


class Preprocessor(Stubcessor):

    def __init__(self, config: KISSConfig):
        self.config = config

    def __call__(self, frame: np.ndarray):
        if frame.shape[1] == 3:
            eigen_frame = kiss_icp_pybind._Vector3dVector(frame)
        elif frame.shape[1] == 4:
            eigen_frame = kiss_icp_pybind._Vector4dVector(frame)
        elif frame.shape[1] == kiss_icp_pybind._point_size():
            eigen_frame = kiss_icp_pybind._VectorNdVector(frame)
        elif frame.shape[1] > 4:
            eigen_frame = kiss_icp_pybind._VectorXdVector(frame)
        else:
            raise ValueError("Invalid shape")

        return np.asarray(
            kiss_icp_pybind._preprocess(
                eigen_frame,
                self.config.data.max_range,
                self.config.data.min_range,
            ))
