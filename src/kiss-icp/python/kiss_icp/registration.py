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
from kiss_icp.mapping import VoxelHashMap
from kiss_icp.pybind import kiss_icp_pybind


def register_frame(points: np.ndarray,
                   voxel_map: VoxelHashMap,
                   initial_guess: np.ndarray,
                   max_correspondance_distance: float,
                   kernel: float,
                   src_=None,
                   tgt_=None) -> np.ndarray:
    if points.shape[1] == 3:
        eigen_points = kiss_icp_pybind._Vector3dVector(points)
    elif points.shape[1] == kiss_icp_pybind._point_size():
        eigen_points = kiss_icp_pybind._VectorNdVector(points)
    elif points.shape[1] > 3:
        eigen_points = kiss_icp_pybind._VectorXdVector(points)
    else:
        raise ValueError("Invalid shape")

    initial_guess = initial_guess.astype(np.float64)

    if points.shape[1] == kiss_icp_pybind._point_size():
        do_return = True
        if src_ is None or tgt_ is None:
            do_return = False
            src_, tgt_ = np.array([[0, 0, 0]]), np.array([[0, 0, 0]])
        eigen_src_ = kiss_icp_pybind._Vector3dVector(src_)
        eigen_tgt_ = kiss_icp_pybind._Vector3dVector(tgt_)
        pose = kiss_icp_pybind._register_point_cloud(
            points=eigen_points,
            voxel_map=voxel_map._internal_map,
            initial_guess=initial_guess,
            max_correspondance_distance=max_correspondance_distance,
            kernel=kernel,
            src_=eigen_src_,
            tgt_=eigen_tgt_)
        if not do_return:
            return pose
        src_ = np.array(eigen_src_)
        tgt_ = np.array(eigen_tgt_)
        return pose, src_, tgt_

    return kiss_icp_pybind._register_point_cloud(
        points=eigen_points,
        voxel_map=voxel_map._internal_map,
        initial_guess=initial_guess,
        max_correspondance_distance=max_correspondance_distance,
        kernel=kernel,
    )
