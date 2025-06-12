from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from kiss_icp.pybind import kiss_icp_pybind
from numpy.typing import ArrayLike
from PIL import Image as PIL_Image


class KittiOdometry:

    def __init__(self, sequence: int, root_dir: Path, image_subsample: int = 1) -> None:
        self.correct_kitti_scan = lambda frame: np.asarray(
            kiss_icp_pybind._correct_kitti_scan(kiss_icp_pybind._Vector3dVector(frame)))
        self.same_cameras = [["camera"]]
        self.cameras = [x for sublist in self.same_cameras for x in sublist]

        self.root_dir = root_dir
        self.sequence = f"{sequence:02d}"
        self.image_subsample = image_subsample

        self.img_files = sorted(
            (self.root_dir / "sequences" / self.sequence / "image_2").glob("*.png"))
        self.pcl_files = sorted(
            (self.root_dir / "sequences" / self.sequence / "velodyne").glob("*.bin"))
        self.calib = self.read_calib()
        self.timestamps = self.read_times()
        self.gt_poses = self.read_poses()
        assert len(self.img_files) == len(
            self.pcl_files), "Number of images and point clouds does not match"
        assert len(self.img_files) == len(
            self.timestamps), "Number of images and timestamps does not match"
        assert len(self.img_files) == len(
            self.gt_poses), "Number of images and poses does not match"

    def __len__(self) -> int:
        return len(self.pcl_files)

    def read_image_files(self, frame_id: int) -> Dict[str, Path]:
        return {camera: self.img_files[camera][frame_id] for camera in self.cameras}

    def read_images(self, frame_id: int) -> Dict[str, ArrayLike]:
        image_path = self.img_files[frame_id]
        image = PIL_Image.open(image_path)
        if self.image_subsample > 1:
            image_res = (image.size[0] // self.image_subsample,
                         image.size[1] // self.image_subsample)
            image = image.resize(image_res, PIL_Image.BILINEAR)
        return {"camera": np.array(image)}

    def read_pcl(self, frame_id: int) -> ArrayLike:
        pcl_path = self.pcl_files[frame_id]
        pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(-1, 4)
        # Keep only xyz points
        pcl = pcl[:, :3].astype(np.float64)
        # Correct KITTI's HDL-64 calibration
        pcl = self.correct_kitti_scan(pcl)
        # Remove points behind the camera
        # pcl = pcl[pcl[:, 0] > 0]

        # Crop at 50m
        # depth = np.linalg.norm(pcl, axis=1)
        # pcl = pcl[depth < 50]

        return pcl

    def read_calib(self) -> Dict[str, ArrayLike]:
        calib_path = self.root_dir / "sequences" / self.sequence / "calib.txt"
        with open(calib_path, "r", encoding="UTF-8") as f:
            calib = f.readlines()
        P2 = np.array([float(x) for x in calib[2].strip("\n").split(" ")[1:]]).reshape(3, 4)
        # R0_rect = np.array([float(x) for x in calib[4].strip("\n").split(" ")[1:]]).reshape(3, 3)
        # # Add a 1 in bottom-right, reshape to 4 x 4
        # R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
        # R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
        Tr_velo_to_cam = np.array([float(x)
                                   for x in calib[4].strip("\n").split(" ")[1:]]).reshape(3, 4)
        Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
        return {"P2": P2, "Tr_velo_to_cam": Tr_velo_to_cam}

    def read_poses(self) -> List[ArrayLike]:
        assert hasattr(self, "calib"), "Calibration matrix not loaded"
        Tr = self.calib["Tr_velo_to_cam"]
        Tr_inv = np.linalg.inv(Tr)

        poses_path = self.root_dir / "poses" / f"{self.sequence}.txt"
        with open(poses_path, "r", encoding="UTF-8") as f:
            lines = f.readlines()
            poses = []
            for line in lines:
                values = np.array([float(x) for x in line.strip("\n").split(" ")])
                values = values.reshape(3, 4)
                pose = np.eye(4)
                pose[:3, :] = values

                # Convert to LiDAR frame
                pose = np.matmul(Tr_inv, np.matmul(pose, Tr))

                poses.append(pose)

        return poses

    def read_times(self) -> List[float]:
        times_path = self.root_dir / "sequences" / self.sequence / "times.txt"
        with open(times_path, "r", encoding="UTF-8") as f:
            lines = f.readlines()
        times = [float(line.strip("\n")) for line in lines]
        return times

    def project_pcl_to_image(self, pcl: ArrayLike, image: ArrayLike,
                             camera: str) -> Union[ArrayLike, ArrayLike, ArrayLike]:
        pcl_in_cam_raw = self.calib["P2"] @ self.calib[
            "Tr_velo_to_cam"] @ pcl  # 3x4 @ 4x4 @ 4xN = 3xN
        pcl_indices = np.where(pcl_in_cam_raw[2, :] > 0)[0]
        pcl_in_cam = pcl_in_cam_raw[:, pcl_indices]
        u, v = pcl_in_cam[:2, :] / pcl_in_cam[2, :] / self.image_subsample
        u_out = np.logical_or(u < 0, u > image.shape[1])
        v_out = np.logical_or(v < 0, v > image.shape[0])
        outlier = np.logical_or(u_out, v_out)
        pcl_indices = pcl_indices[~outlier]
        u, v = u[~outlier], v[~outlier]
        u, v = u.astype(int), v.astype(int)
        # pcl_in_cam = pcl_in_cam_raw[:, pcl_indices]

        return u, v, pcl_indices
