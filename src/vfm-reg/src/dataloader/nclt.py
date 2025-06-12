import csv
import re
import sys
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Union

import cv2
import numpy as np
import pandas as pd
import scipy
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
from robotcar_sdk.python.transform import build_se3_transform


class NCLT:

    def __init__(self,
                 sequence: int,
                 root_dir: Path,
                 image_subsample: int = 1,
                 high_level_api: bool = False) -> None:

        self.root_dir = root_dir
        self.sequence = sequence
        self.image_subsample = image_subsample

        # Cam0: sky facing / upwards
        # Cam1: side left
        # Cam2: rear left (in direction of travel)
        # Cam3: rear right
        # Cam4: side right
        # Cam5: forward
        self.same_cameras = [["Cam1"], ["Cam2"], ["Cam3"], ["Cam4"], ["Cam5"]]
        self.cameras = [x for sublist in self.same_cameras for x in sublist]

        self.calib = self.read_calib()
        self.undistortion_maps = self.read_undistortion_maps()
        self.undistortion_masks = self.read_undistortion_masks()
        self.camera_parameters = self.read_camera_parameters()

        self.timestamps_abs = self.read_times()
        self.timestamps = self.compute_relative_timestamps()  # Adapt API to KittiOdometry
        # self.gt_poses = self.read_poses()

        self.img_files = {
            camera: [
                self.root_dir / "images" / self.sequence / "lb3" / camera / f"{ts}.tiff"
                for ts in self.timestamps_abs["image"]
            ]
            for camera in self.cameras
        }
        self.pcl_files = [
            self.root_dir / "velodyne_data" / self.sequence / "velodyne_sync" / f"{ts}.bin"
            for ts in self.timestamps_abs["pcl"]
        ]

    def __len__(self) -> int:
        return len(self.pcl_files)

    def read_image_files(self, frame_id: int) -> Dict[str, Path]:
        return {camera: self.img_files[camera][frame_id] for camera in self.cameras}

    def read_images(self,
                    frame_id: int = -1,
                    crop: bool = True,
                    filenames: Optional[List[Path]] = None) -> Dict[str, ArrayLike]:
        assert frame_id != -1 or filenames is not None, "Either frame_id or filenames must be provided"
        if frame_id == -1:
            assert filenames is not None, "Cannot provide both frame_id and filenames"
        if filenames is not None:
            assert frame_id == -1, "Cannot provide both frame_id and filenames"

        if frame_id != -1:
            assert hasattr(
                self,
                "timestamps_abs") and "image" in self.timestamps_abs, "Image timestamps not loaded"
            filenames = [self.img_files[camera][frame_id] for camera in self.cameras]

        images = dict()
        for camera, image_path in zip(self.cameras, filenames):
            image = cv2.imread(str(image_path))
            # Undistort image
            mapu = self.undistortion_maps[camera]["mapu"]
            mapv = self.undistortion_maps[camera]["mapv"]
            mask = self.undistortion_maps[camera]["mask"]
            image = cv2.remap(image, mapu, mapv, cv2.INTER_LINEAR)
            image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_CUBIC)
            # Convert from BGR to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Crop black borders
            if crop:
                mask = self.undistortion_masks[camera]["mask"]
                coords = self.undistortion_masks[camera]["coords"]
                image = image[mask].reshape((coords[2], coords[3], 3))

            # Rotate by 90deg clockwise
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

            if self.image_subsample > 1:
                image_res = (image.shape[1] // self.image_subsample,
                             image.shape[0] // self.image_subsample)
                image = cv2.resize(image, image_res, interpolation=cv2.INTER_AREA)

            images[camera] = image

        return images

    def read_pcl(self, frame_id: int = -1, filename: Optional[Path] = None) -> ArrayLike:
        assert frame_id != -1 or filename is not None, "Either frame_id or filename must be provided"
        if frame_id == -1:
            assert filename is not None, "Cannot provide both frame_id and filename"
        if filename is not None:
            assert frame_id == -1, "Cannot provide both frame_id and filename"

        if filename is not None:
            pcl_path = filename
        else:
            pcl_path = self.pcl_files[frame_id]

        # From dataset SDK
        def convert(x_s, y_s, z_s):
            scaling = 0.005  # 5 mm
            offset = -100.0
            x = x_s * scaling + offset
            y = y_s * scaling + offset
            z = z_s * scaling + offset
            return x, y, z

        # From:
        # https://gitlab.kitware.com/keu-computervision/pylidar-slam/-/blob/master/slam/dataset/nclt_dataset.py
        binary = np.fromfile(pcl_path, dtype=np.int16)
        x = np.ascontiguousarray(binary[::4])
        y = np.ascontiguousarray(binary[1::4])
        z = np.ascontiguousarray(binary[2::4])
        x = x.astype(np.float32).reshape(-1, 1)
        y = y.astype(np.float32).reshape(-1, 1)
        z = z.astype(np.float32).reshape(-1, 1)
        x, y, z = convert(x, y, z)
        pcl = np.concatenate([x, y, z], axis=1)

        # Crop at 50m
        depth = np.linalg.norm(pcl, axis=1)
        pcl = pcl[depth < 50]

        return pcl

    def read_calib(self) -> Dict[str, ArrayLike]:
        calib = dict()

        # From dataset paper
        calib["lidar_in_ego"] = np.eye(4)
        calib["lidar_in_ego"][:3, :3] = R.from_euler("xyz", [0.807, 0.166, -90.703],
                                                     degrees=True).as_matrix()
        calib["lidar_in_ego"][:3, 3] = [0.002, -0.004, -0.957]
        calib["ego_in_lidar"] = np.linalg.inv(calib["lidar_in_ego"])

        return calib

    def read_undistortion_maps(self) -> Dict[str, Dict[str, ArrayLike]]:
        undistortion_maps = {}

        for camera in self.cameras:
            undistortion_map_file = self.root_dir / "cam_params" / f"U2D_{camera}_1616X1232.txt"

            # Adapted from dataset SDK
            with open(undistortion_map_file, 'r') as f:
                header = f.readline().rstrip()
                chunks = re.sub(r'[^0-9,]', '', header).split(',')
                mapu = np.zeros((int(chunks[1]), int(chunks[0])), dtype=np.float32)
                mapv = np.zeros((int(chunks[1]), int(chunks[0])), dtype=np.float32)
                for line in f.readlines():
                    chunks = line.rstrip().split(' ')
                    mapu[int(chunks[0]), int(chunks[1])] = float(chunks[3])
                    mapv[int(chunks[0]), int(chunks[1])] = float(chunks[2])
            # generate a mask
            mask = np.ones(mapu.shape, dtype=np.uint8)
            mask = cv2.remap(mask, mapu, mapv, cv2.INTER_LINEAR)
            kernel = np.ones((30, 30), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)

            undistortion_maps[camera] = {"mapu": mapu, "mapv": mapv, "mask": mask}

        return undistortion_maps

    def read_undistortion_masks(self) -> Dict[str, ArrayLike]:
        undistortion_masks = {camera: {"coords": [210, 450, 820, 700]} for camera in self.cameras}

        for camera in self.cameras:
            mask_coords = undistortion_masks[camera]["coords"]
            mask = np.zeros((1232, 1616), dtype=np.uint8)
            mask[mask_coords[0]:mask_coords[0] + mask_coords[2],
                 mask_coords[1]:mask_coords[1] + mask_coords[3]] = 1
            mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
            undistortion_masks[camera]["mask"] = mask.astype(bool)

        return undistortion_masks

    def read_camera_parameters(self) -> Dict[str, Dict[str, ArrayLike]]:
        camera_parameters = {}

        for camera in self.cameras:
            camera_id = camera[-1]
            K_file = self.root_dir / "cam_params" / f"K_cam{camera_id}.csv"
            x_lb3_file = self.root_dir / "cam_params" / f"x_lb3_c{camera_id}.csv"

            K = np.loadtxt(K_file, delimiter=",")
            x_lb3_ = np.loadtxt(x_lb3_file, delimiter=",")
            x_lb3 = np.eye(4)
            x_lb3[:3, 3] = x_lb3_[:3]
            x_lb3[:3, :3] = R.from_euler("xyz", x_lb3_[3:], degrees=True).as_matrix()

            camera_parameters[camera] = {"K": K, "x_lb3": x_lb3}

        return camera_parameters

    def read_poses(self,
                   absolute_poses: bool = False,
                   interpolate_pcl: bool = True) -> List[ArrayLike]:
        # Read poses at point cloud timestamps
        assert hasattr(
            self,
            "timestamps_abs") and "pcl" in self.timestamps_abs, "Point cloud timestamps not loaded"

        poses_file = self.root_dir / "ground_truth" / f"groundtruth_{self.sequence}.csv"

        with open(poses_file) as f:
            vo_reader = csv.reader(f)
            next(f)

            timestamps = []
            abs_poses = []

            upper_timestamp = max(self.timestamps_abs["pcl"])

            for row in tqdm(vo_reader, desc="Reading poses"):
                if np.any(np.isnan([float(v) for v in row[1:7]])):
                    continue

                timestamp = int(row[0])
                timestamps.append(timestamp)

                xyzrpy = [float(v) for v in row[1:7]]
                abs_pose = build_se3_transform(xyzrpy)
                abs_poses.append(abs_pose)

                if timestamp >= upper_timestamp:
                    break

        # Remove point clouds after last pose
        if max(timestamps) < max(self.timestamps_abs["pcl"]):
            keep_idx = np.array(self.timestamps_abs["pcl"]) <= max(timestamps)
            self.timestamps_abs["pcl"] = np.array(self.timestamps_abs["pcl"])[keep_idx].tolist()
            self.timestamps_abs["image"] = np.array(self.timestamps_abs["image"])[keep_idx].tolist()
        # Remove point clouds before first pose
        if min(timestamps) > min(self.timestamps_abs["pcl"]):
            keep_idx = np.array(self.timestamps_abs["pcl"]) >= min(timestamps)
            self.timestamps_abs["pcl"] = np.array(self.timestamps_abs["pcl"])[keep_idx].tolist()
            self.timestamps_abs["image"] = np.array(self.timestamps_abs["image"])[keep_idx].tolist()

        if interpolate_pcl:
            interp = scipy.interpolate.interp1d(timestamps, abs_poses, kind='nearest', axis=0)
            poses = interp(self.timestamps_abs["pcl"])
            # The interpolation accuracy with 'nearest' is about .002s with max .009s

            # interp = scipy.interpolate.interp1d(timestamps, timestamps, kind='nearest', axis=0)
            # interp_ts = interp(self.timestamps_abs["pcl"])
            # interp_ts = np.abs(np.asarray(interp_ts) - np.asarray(self.timestamps_abs["pcl"]))
            # print(np.max(interp_ts / 1e6), np.mean(interp_ts) /1e6)

            poses = np.asarray(poses)
        else:
            poses = np.asarray(abs_poses)

        # Start poses at origin
        if not absolute_poses:
            origin_pose = np.linalg.inv(poses[0])
            poses = [origin_pose @ pose for pose in poses]

        return poses

    def read_times(self) -> Union[List[int], List[int]]:

        img_files_dir = self.root_dir / "images" / self.sequence / "lb3" / "Cam1"
        img_files = list(img_files_dir.glob("*.tiff"))
        img_timestamps = sorted([int(f.stem) for f in img_files])

        pcl_files_dir = self.root_dir / "velodyne_data" / self.sequence / "velodyne_sync"
        pcl_files = list(pcl_files_dir.glob("*.bin"))
        pcl_timestamps = sorted([int(f.stem) for f in pcl_files])

        common_timestamps = sorted(list(set(img_timestamps) & set(pcl_timestamps)))
        # common_timestamps = pcl_timestamps

        return {"image": common_timestamps, "pcl": common_timestamps}

    def compute_relative_timestamps(self) -> List[float]:
        # Compute relative timestamps as microseconds (int)
        relative_pcl_timestamps = [
            ts - self.timestamps_abs["pcl"][0] for ts in self.timestamps_abs["pcl"]
        ]
        # Convert to seconds (float)
        relative_pcl_timestamps = [ts / 1e6 for ts in relative_pcl_timestamps]
        return relative_pcl_timestamps

    def project_pcl_to_image(self, pcl: ArrayLike, image: ArrayLike,
                             camera: str) -> Union[ArrayLike, ArrayLike, ArrayLike]:
        assert camera in self.cameras, f"Camera {camera} not available"

        K = self.camera_parameters[camera]["K"]
        x_lb3_c = self.camera_parameters[camera]["x_lb3"]  # Rotation matrix

        # From dataset SDK
        x_body_lb3 = np.eye(4)
        x_body_lb3[:3, 3] = [0.035, 0.002, -1.23]
        x_body_lb3[:3, :3] = R.from_euler("xyz", [-179.93, -0.23, 0.50], degrees=True).as_matrix()

        T_lb3_body = np.linalg.inv(x_body_lb3)
        T_c_lb3 = np.linalg.inv(x_lb3_c)
        T_c_body = T_c_lb3 @ T_lb3_body

        # Project point cloud to camera frame
        plc_ = T_c_body @ pcl

        # Project point cloud to image frame
        pcl_ = K @ plc_[:3, :]

        # Compute pixel coordinates
        x_im = pcl_[0] / pcl_[2] / self.image_subsample
        y_im = pcl_[1] / pcl_[2] / self.image_subsample
        z_im = pcl_[2]
        in_front = z_im > 0
        x_im = x_im[in_front]
        y_im = y_im[in_front]

        # Filter points outside image
        x_im = x_im.astype(int)
        y_im = y_im.astype(int)
        mask_coords = np.array(self.undistortion_masks[camera]["coords"]) // self.image_subsample
        x_out = np.logical_or(x_im < mask_coords[1], x_im >= mask_coords[1] + mask_coords[3])
        y_out = np.logical_or(y_im < mask_coords[0], y_im >= mask_coords[0] + mask_coords[2])
        in_image = np.logical_not(np.logical_or(x_out, y_out))
        x_im = x_im[in_image]
        y_im = y_im[in_image]
        x_im -= mask_coords[1]
        y_im -= mask_coords[0]

        # Filter points without RGB information
        in_rgb = []
        for i in range(len(x_im)):
            if np.any(image[y_im[i], x_im[i]] != 0):
                in_rgb.append(i)
        x_im = x_im[in_rgb]
        y_im = y_im[in_rgb]

        # Compute pcl coordinates
        pcl_indices = np.where(in_front)[0]
        pcl_indices = pcl_indices[in_image]
        pcl_indices = pcl_indices[in_rgb]

        return x_im, y_im, pcl_indices
