import sys
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from colour_demosaicing import demosaicing_CFA_Bayer_bilinear as demosaic
from numpy.typing import ArrayLike
from PIL import Image as PIL_Image
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))

from robotcar_sdk.python.camera_model import CameraModel
from robotcar_sdk.python.image import load_image
from robotcar_sdk.python.interpolate_poses import interpolate_ins_poses
from robotcar_sdk.python.transform import build_se3_transform

from vfm_reg.utils import *


class OxfordRobotcar:

    def __init__(self,
                 sequence: str,
                 root_dir: Path,
                 image_subsample: int = 1,
                 high_level_api: bool = False) -> None:

        self.root_dir = root_dir
        self.sequence = sequence
        self.image_subsample = image_subsample
        self.lidar_frequency = 10
        self.same_cameras = [["stereo/centre"], ["mono_left", "mono_right", "mono_rear"]]
        # self.same_cameras = [["stereo/centre"]]
        self.cameras = [x for sublist in self.same_cameras for x in sublist]

        models_dir = Path(__file__).parent / "robotcar_sdk" / "models"
        self.camera_model = {camera: CameraModel(models_dir, camera) for camera in self.cameras}
        self.calib = self.read_calib()

        if not high_level_api:
            self.timestamps_abs = self.read_times()
            # self.pcl_to_image_ts_transforms = self.compute_pcl_to_image_ts_transforms()
            self.gt_poses = self.read_poses()
            self.timestamps = self.compute_relative_timestamps()  # Adapt API to KittiOdometry

            self.img_files = {
                camera: [
                    self.root_dir / f"{self.sequence}-radar-oxford-10k" / camera / f"{ts}.png"
                    for ts in self.timestamps_abs["image"][camera]
                ]
                for camera in self.cameras
            }
            # Cache undistorted images for faster processing
            self.img_undistorted_files = {
                camera: [
                    self.root_dir / f"{self.sequence}-radar-oxford-10k" / f"{camera}_undistorted" /
                    f"{ts}.png" for ts in self.timestamps_abs["image"][camera]
                ]
                for camera in self.cameras
            }
            self.pcl_files = [
                self.root_dir / f"{self.sequence}-radar-oxford-10k" / "velodyne_left" / f"{ts}.bin"
                for ts in self.timestamps_abs["pcl"]
            ]
            for camera in self.cameras:
                assert len(self.img_files[camera]) == len(
                    self.pcl_files), f"Number of images {camera} and point clouds does not match"
            assert len(self.gt_poses) == len(
                self.pcl_files), "Number of poses and point clouds does not match"

        else:
            self.timestamps_abs = None
            self.gt_poses = None
            self.timestamps = None

    def __len__(self) -> int:
        return len(self.pcl_files)

    def read_image_files(self, frame_id: int) -> Dict[str, Path]:
        return {camera: self.img_files[camera][frame_id] for camera in self.cameras}

    def read_images(self,
                    frame_id: int = -1,
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
            use_cache = True
            filenames = [self.img_files[camera][frame_id] for camera in self.cameras]
        else:
            use_cache = False

        def read_image(camera, image_path, use_cache: bool = True):
            # Load undistorted image if available
            if use_cache and self.img_undistorted_files[camera][frame_id].exists():
                image = PIL_Image.open(self.img_undistorted_files[camera][frame_id])
            else:
                # image_path = self.img_files[camera][frame_id]
                image = PIL_Image.open(image_path)
                # Debayer image
                if camera == "stereo/centre":
                    image = demosaic(image, "GBRG")
                else:
                    image = demosaic(image, "RGGB")
                # Undistort image
                image = self.camera_model[camera].undistort(image).astype(np.uint8)
                image = PIL_Image.fromarray(image)

                # Remove hood from stereo/centre camera
                if camera == "stereo/centre":
                    image = image.crop((0, 0, image.size[0], image.size[1] - 150))
                # Remove area without LiDAR coverage
                else:
                    image = image.crop((0, 0, image.size[0], image.size[1] - 200))

                # Save for future loading
                if use_cache:
                    self.img_undistorted_files[camera][frame_id].parent.mkdir(parents=True,
                                                                              exist_ok=True)
                    image.save(self.img_undistorted_files[camera][frame_id])

            if self.image_subsample > 1:
                image_res = (image.size[0] // self.image_subsample,
                             image.size[1] // self.image_subsample)
                image = image.resize(image_res, PIL_Image.BILINEAR)
            return image

        images = dict()
        for camera, filename in zip(self.cameras, filenames):
            if not use_cache:
                image = read_image(camera, filename, False)

            else:
                try:
                    image = read_image(camera, filename)
                except:
                    self.img_undistorted_files[camera][frame_id].unlink()
                    try:
                        image = read_image(camera, filename)
                    except:
                        print(
                            f"Error loading image: {self.img_undistorted_files[camera][frame_id]}")
                        raise

            images[camera] = np.array(image)

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

        pcl = np.fromfile(pcl_path, dtype=np.float32).reshape(4, -1).T

        # Remove points on ego vehile
        depth = np.linalg.norm(pcl[:, :3], axis=1)
        pcl = pcl[depth > 2.5]

        # Crop at 50m
        depth = depth[depth > 2.5]
        pcl = pcl[depth < 50]

        # Keep only xyz points
        pcl = pcl[:, :3]
        return pcl

    def read_calib(self) -> Dict[str, ArrayLike]:
        calib = dict()

        extrinsics_dir = Path(__file__).parent / "robotcar_sdk" / "extrinsics"
        with open(extrinsics_dir / "velodyne_left.txt") as extrinsics_file:
            extrinsics = next(extrinsics_file)
            # LiDAR in ego frame (stereo/centre camera)
            calib["lidar_in_ego"] = np.asarray(
                build_se3_transform([float(x) for x in extrinsics.split(' ')]))

        for camera in self.cameras:
            if camera == "stereo/centre":
                camera_extrinsics_file = extrinsics_dir / "stereo.txt"
            else:
                camera_extrinsics_file = extrinsics_dir / f"{camera}.txt"
            with open(camera_extrinsics_file) as extrinsics_file:
                extrinsics = next(extrinsics_file)
                # Camera in ego frame (stereo/centre camera)
                calib[f"{camera}_in_ego"] = np.asarray(
                    build_se3_transform([float(x) for x in extrinsics.split(' ')]))

        with open(extrinsics_dir / "ins.txt") as extrinsics_file:
            extrinsics = next(extrinsics_file)
            # INS in ego frame (stereo/centre camera)
            calib["ins_in_ego"] = np.asarray(
                build_se3_transform([float(x) for x in extrinsics.split(' ')]))

        # LiDAR in INS
        calib["lidar_in_ins"] = np.linalg.solve(calib["ins_in_ego"], calib["lidar_in_ego"])
        # INS in LiDAR
        calib["ins_in_lidar"] = np.linalg.inv(calib["lidar_in_ins"])

        return calib

    def read_poses(self, absolute_poses: bool = False) -> List[ArrayLike]:
        # Read poses at point cloud timestamps
        assert hasattr(
            self,
            "timestamps_abs") and "pcl" in self.timestamps_abs, "Point cloud timestamps not loaded"

        ins_file = self.root_dir / f"{self.sequence}-radar-oxford-10k" / "gps" / "ins.csv"
        # Interpolate poses to match the timestamps of the point clouds
        poses = np.asarray(
            interpolate_ins_poses(ins_file, deepcopy(self.timestamps_abs["pcl"]),
                                  self.timestamps_abs["pcl"][0]))
        # Move poses from INS to the LiDAR frame
        poses = np.asarray([np.dot(pose, self.calib["ins_in_lidar"]) for pose in poses])

        # Remove timestamps and poses from NaN poses
        if np.isnan(poses).any():
            nan_idx = set([x[0] for x in np.argwhere(np.isnan(poses))])
            nan_idx = reversed(sorted(list(nan_idx)))
            for idx in nan_idx:
                poses = np.delete(poses, idx, axis=0)
                for k, v in self.timestamps_abs.items():
                    if k == 'image':
                        for k2, v2 in v.items():
                            del v2[idx]
                    else:
                        del v[idx]

        # Start poses at origin
        if not absolute_poses:
            origin_pose = np.linalg.inv(poses[0])
            poses = [origin_pose @ pose for pose in poses]

        return poses

    def read_times(self) -> Union[List[int], List[int]]:
        step = 20 // self.lidar_frequency

        pcl_timestamps_file = self.root_dir / f"{self.sequence}-radar-oxford-10k" / "velodyne_left.timestamps"
        pcl_timestamps = np.loadtxt(pcl_timestamps_file, delimiter=' ', usecols=[0],
                                    dtype=np.int64)[::step].tolist()

        def read_image_times(camera_type: str) -> List[int]:
            if camera_type == "stereo/centre":
                camera_type = "stereo"
            # Load synchronized image timestamps if available
            img_timestamps_file = self.root_dir / f"{self.sequence}-radar-oxford-10k" / f"{camera_type}_{step}.timestamps"
            if img_timestamps_file.exists():
                img_timestamps = np.loadtxt(img_timestamps_file,
                                            delimiter=' ',
                                            usecols=[0],
                                            dtype=np.int64).tolist()

            else:
                img_timestamps_file = img_timestamps_file.parent / f"{camera_type}.timestamps"
                img_timestamps = np.loadtxt(img_timestamps_file,
                                            delimiter=' ',
                                            usecols=[0],
                                            dtype=np.int64).tolist()

                # Find nearest image for each point cloud
                img_timestamps_ = []
                for pcl_ts in tqdm(pcl_timestamps,
                                   desc=f"Searching for nearest images | {camera_type}"):
                    img_ts = np.argmin([abs(pcl_ts - img_ts) for img_ts in img_timestamps])
                    img_timestamps_.append(img_timestamps[img_ts])
                img_timestamps = img_timestamps_

                # Save synchronized timestamps
                np.savetxt(img_timestamps_file.parent / f"{camera_type}_{step}.timestamps",
                           img_timestamps,
                           fmt='%i',
                           delimiter=' ')
            return img_timestamps

        img_timestamps = {camera: read_image_times(camera) for camera in self.cameras}

        return {"image": img_timestamps, "pcl": pcl_timestamps}

    def compute_relative_timestamps(self) -> List[float]:
        assert hasattr(self, "gt_poses"), "Ground truth poses not loaded"

        # Compute relative timestamps as microseconds (int)
        relative_pcl_timestamps = [
            ts - self.timestamps_abs["pcl"][0] for ts in self.timestamps_abs["pcl"]
        ]
        # Convert to seconds (float)
        relative_pcl_timestamps = [ts / 1e6 for ts in relative_pcl_timestamps]
        return relative_pcl_timestamps

    def compute_pcl_to_image_ts_transforms(self) -> List[ArrayLike]:
        # ToDo: Implement this to further improve the pcl to image projection

        # Compute transforms to move point clouds to image timestamps

        ins_file = self.root_dir / f"{self.sequence}-radar-oxford-10k" / "gps" / "ins.csv"

        # Poses are relative to this frame
        origin_timestamp = self.timestamps_abs["pcl"][0]

        pcl_to_image_ts_transforms = []
        for pcl_t, img_t in zip(self.timestamps_abs["pcl"], self.timestamps_abs["image"]):
            # Obtain INS pose at image and at point cloud timestamps
            poses = np.asarray(interpolate_ins_poses(ins_file, [img_t, pcl_t], origin_timestamp))
            # Relative pose to move point cloud to image timestamp
            relative_pose = np.linalg.inv(poses[1]) @ poses[0]
            print_msg(poses[0], '1')
            print_msg(poses[1], '2')
            print_msg(relative_pose, 'rel')
            pcl_to_image_ts_transforms.append(relative_pose)
        return pcl_to_image_ts_transforms

    def project_pcl_to_image(self, pcl: ArrayLike, image: ArrayLike,
                             camera: str) -> Union[ArrayLike, ArrayLike, ArrayLike]:
        assert camera in self.cameras, f"Camera {camera} not available"

        # Project point cloud to ego frame
        pcl_ = self.calib["lidar_in_ego"] @ pcl

        # Project point cloud to camera frame
        pcl_ = self.calib[f"{camera}_in_ego"] @ pcl_

        # Project point cloud to image frame
        pcl_ = np.linalg.solve(self.camera_model[camera].G_camera_image, pcl_)

        # Find points in front of image plane by checking z coordinate (depth)
        in_front = np.asarray([i for i in range(pcl_.shape[1]) if pcl_[2, i] >= 0])
        pcl_in_front = pcl_[:, in_front]

        # Compute pixel coordinates
        focal_length = self.camera_model[camera].focal_length
        principal_point = self.camera_model[camera].principal_point
        u = focal_length[0] * pcl_in_front[0, :] / pcl_in_front[2, :] + principal_point[0]
        v = focal_length[1] * pcl_in_front[1, :] / pcl_in_front[2, :] + principal_point[1]
        u /= self.image_subsample
        v /= self.image_subsample

        # Remove points outside image
        u_out = np.logical_or(u < 0, u > image.shape[1])
        v_out = np.logical_or(v < 0, v > image.shape[0])
        outlier = np.logical_or(u_out, v_out)

        pcl_indices = in_front[~outlier]
        u, v = u[~outlier], v[~outlier]
        u, v = u.astype(int), v.astype(int)
        return u, v, pcl_indices


if __name__ == "__main__":

    sequence = '2019-01-15-13-06-37'
    root_dir = Path('/data/robotcar')
    image_subsample = 1

    dataloader = OxfordRobotcar(sequence, root_dir, image_subsample)
