#!/usr/bin/env python

import argparse
import datetime
import json
import pickle
import random
import signal
import sys
from pathlib import Path
from time import time
from typing import Optional, Tuple

import hdbscan
import numpy as np
import open3d as o3d
import rospy
import scipy
import teaserpp_python as teaser
from kiss_icp.config import load_config
from kiss_icp.mapping import get_voxel_hash_map
from kiss_icp.registration import register_frame
from kiss_icp.voxelization import voxel_down_sample
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2 as PointCloudMsg
from sklearn.neighbors import KDTree
from tqdm import tqdm
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg

from vfm_reg.read_h5 import read_scenes
from vfm_reg.descriptors import *
from vfm_reg.image_features import ImageFeatureGenerator
from vfm_reg.utils import *
from pointdsc.PointDSC import PointDSC

np.set_printoptions(precision=2)

o3d.utility.random.seed(42)  # Effect unclear
random.seed(42)
np.random.seed(42)


class RegistrationNode:

    def __init__(self,
                 folder: Path,
                 interactive: bool,
                 cluster_removal_prob: int = 0,
                 scan_to_scan: Optional[str] = None) -> None:
        self.stop_publishing = False  # Track if the node wants to exit
        self.interactive = interactive
        self.cluster_removal_prob = cluster_removal_prob
        self.scan_to_scan = scan_to_scan
        assert self.scan_to_scan in [None, 'data', 'kitti']

        self.dino_generator = ImageFeatureGenerator("dinov2", use_featup=False)

        self.filenames = sorted(list(folder.glob(f"scene_*.h5")))
        # self.filenames = [self.filenames[i] for i in [1, 9, 10]] # NCLT figures
        # self.filenames = [self.filenames[i] for i in [2, 7, 9]] # RobotCar figures
        # self.filenames = [self.filenames[i] for i in [7]] # NCLT teaser figure
        # self.filenames = [self.filenames[i] for i in [4]] # NCLT robustness figure
        # self.filenames = [self.filenames[i] for i in [0]]  # NCLT overview figure
        self.scene_idx = 0
        self.scan_idx = 0
        self.map_descriptor_cache = {}

        self.rot_errors = {}
        self.trans_errors = {}
        self.registration_success = {}
        self.points_in_map = []

        pcl_output_topic = "/pcl"
        self.pcl_pub = [
            rospy.Publisher(f"{pcl_output_topic}/{i}", PointCloudMsg, queue_size=10)
            for i in range(6)
        ]
        correspondences_output_topic = "/correspondences"
        self.corresponce_pub = [
            rospy.Publisher(f"{correspondences_output_topic}/{i}", MarkerArrayMsg, queue_size=10)
            for i in range(2)
        ]

        self.config = load_config(None, deskew=False, max_range=None)
        print("Sigma:", self.config.adaptive_threshold.initial_threshold)
        print("Voxel size:", self.config.mapping.voxel_size)
        print("Max points per voxel:", self.config.mapping.max_points_per_voxel)
        print('=' * 80)

    def teaser_registration(self,
                            voxel_map,
                            raw_scan,
                            method: str,
                            run_icp: bool = False) -> ArrayLike:

        if method == 'fpfh':
            [src, tgt] = self.compute_correspondences(voxel_map[:, :3],
                                                      raw_scan[:, :3],
                                                      'fpfh',
                                                      mutual_filter=True)

        elif method == 'vfm':
            [src, tgt] = self.compute_vfm_correspondences(voxel_map, raw_scan)

        else:
            raise ValueError(f"Invalid method: {method}")
        print(f"[{method}] Correspondences: {src.shape[0]}")

        src, tgt = src.T, tgt.T

        solver_params = teaser.RobustRegistrationSolver.Params()
        solver_params.cbar2 = 1.0
        solver_params.noise_bound = 0.2  # should be similar to the voxel size
        solver_params.estimate_scaling = False
        solver_params.inlier_selection_mode = \
            teaser.RobustRegistrationSolver.INLIER_SELECTION_MODE.PMC_EXACT
        solver_params.rotation_tim_graph = \
            teaser.RobustRegistrationSolver.INLIER_GRAPH_FORMULATION.CHAIN
        solver_params.rotation_estimation_algorithm = \
            teaser.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
        solver_params.rotation_gnc_factor = 1.4
        solver_params.rotation_max_iterations = 10000
        solver_params.rotation_cost_threshold = 1e-16
        solver = teaser.RobustRegistrationSolver(solver_params)

        solver.solve(src, tgt)
        solution = solver.getSolution()
        teaser_pose = np.eye(4)
        teaser_pose[:3, :3] = solution.rotation
        teaser_pose[:3, 3] = solution.translation

        if run_icp:
            # Downsample the map
            voxel_hash_map = get_voxel_hash_map(self.config)
            voxel_hash_map.add_points(voxel_map[:, :3])

            # ICP-based refinement
            downsample_scan = voxel_down_sample(raw_scan[:, :3],
                                                self.config.mapping.voxel_size * 0.5)
            voxel_scan = voxel_down_sample(downsample_scan, self.config.mapping.voxel_size * 1.)

            # Orthogonalize the rotation matrix
            # https://math.stackexchange.com/questions/3292034/normalizing-a-quasi-rotation-matrix
            R = teaser_pose[:3, :3]
            while np.abs(1 - np.linalg.det(R)) > 1e-12:
                R = 3 / 2 * R - 1 / 2 * R @ R.T @ R
            teaser_pose[:3, :3] = R

            sigma = self.config.adaptive_threshold.initial_threshold
            pose = register_frame(points=voxel_scan,
                                  voxel_map=voxel_hash_map,
                                  initial_guess=teaser_pose,
                                  max_correspondance_distance=3 * sigma,
                                  kernel=sigma / 3)

            return teaser_pose, pose

        return teaser_pose, None

    def pointsdc_registration(self,
                              voxel_map,
                              raw_scan,
                              method: str,
                              n_points: int,
                              run_icp: bool = False) -> ArrayLike:

        # Code is adapted from PointDSC/demo_registration.py
        device = torch.device('cuda')

        # Downsample the map
        voxel_hash_map = get_voxel_hash_map(self.config)
        voxel_hash_map.add_points(voxel_map[:, :3])
        map_pcl = voxel_hash_map.point_cloud()
        map_pcl = voxel_map[:, :3]

        if method == 'fcgf':
            # downsample=0.3 is from the KITTI config
            fcgf_weight_path = Path(__file__).parent / 'fcgf' / '2019-07-31_19-37-00.pth'
            src_pts, src_features = extract_fcgf_features(raw_scan[:, :3], .3, fcgf_weight_path)
            tgt_pts, tgt_features = extract_fcgf_features(map_pcl[:, :3], .3, fcgf_weight_path)

        elif method == 'fpfh':
            # [src, tgt] = self.compute_correspondences(voxel_map[:, :3], raw_scan[:, :3], 'fpfh')
            src_pts, src_features = extract_fpfh_features(raw_scan[:, :3], .3, normalize=True)
            tgt_pts, tgt_features = extract_fpfh_features(map_pcl[:, :3], .3, normalize=True)

        else:
            raise ValueError(f"Invalid method: {method}")

        # Batched version of original code to fit the entire map into memory
        batch_size = 1000
        n_iter = src_features.shape[0] // batch_size + 1
        source_dis = []
        corr = []
        for i in range(n_iter):
            distance_i = np.sqrt(
                2 - 2 * (src_features[i * batch_size:(i + 1) * batch_size] @ tgt_features.T) + 1e-6)
            source_idx_i = np.argmin(distance_i, axis=1)
            source_dis_i = np.min(distance_i, axis=1)
            corr_i = np.concatenate([
                np.arange(i * batch_size, i * batch_size + source_idx_i.shape[0])[:, None],
                source_idx_i[:, None]
            ],
                                    axis=-1)
            source_dis.append(source_dis_i)
            corr.append(corr_i)
        source_dis = np.concatenate(source_dis)
        corr = np.concatenate(corr)

        # To reduce the memory usage, we only use n_points points (smallest distance)
        if corr.shape[0] > n_points:
            top_n_idx = np.argpartition(source_dis, n_points)[:n_points]
            corr = corr[top_n_idx]

        src_keypts = src_pts[corr[:, 0]]
        tgt_keypts = tgt_pts[corr[:, 1]]
        corr_pos = np.concatenate([src_keypts, tgt_keypts], axis=-1)
        corr_pos = corr_pos - corr_pos.mean(0)

        print(f"[{method}] Correspondences: {corr_pos.shape[0]}")

        # outlier rejection
        data = {
            'corr_pos': torch.from_numpy(corr_pos)[None].to(device).float(),
            'src_keypts': torch.from_numpy(src_keypts)[None].to(device).float(),
            'tgt_keypts': torch.from_numpy(tgt_keypts)[None].to(device).float(),
            'testing': True,
        }

        pointdsc_weight_path = Path(__file__).parent / 'pointdsc' / 'model_best.pkl'
        pointdsc_model = PointDSC(
            in_dim=6,
            num_layers=12,
            num_channels=128,
            num_iterations=10,
            ratio=0.1,
            sigma_d=1.2,
            k=40,
            nms_radius=0.6,
        ).to(device)
        pointdsc_model.load_state_dict(torch.load(pointdsc_weight_path, map_location=device),
                                       strict=False)
        pointdsc_model.eval()

        res = pointdsc_model(data)
        pointdsc_pose = res['final_trans'][0].detach().cpu().numpy().astype(np.float64)

        if run_icp:
            # ICP-based refinement
            downsample_scan = voxel_down_sample(raw_scan[:, :3],
                                                self.config.mapping.voxel_size * 0.5)
            voxel_scan = voxel_down_sample(downsample_scan, self.config.mapping.voxel_size * 1.)

            # Orthogonalize the rotation matrix
            # https://math.stackexchange.com/questions/3292034/normalizing-a-quasi-rotation-matrix
            R = pointdsc_pose[:3, :3]
            while np.abs(1 - np.linalg.det(R)) > 1e-12:
                R = 3 / 2 * R - 1 / 2 * R @ R.T @ R
            pointdsc_pose[:3, :3] = R

            sigma = self.config.adaptive_threshold.initial_threshold
            pose = register_frame(points=voxel_scan,
                                  voxel_map=voxel_hash_map,
                                  initial_guess=pointdsc_pose,
                                  max_correspondance_distance=3 * sigma,
                                  kernel=sigma / 3)

            return pointdsc_pose, pose

        return pointdsc_pose, None

    def ransac_registration(self,
                            voxel_map,
                            raw_scan,
                            method: str,
                            run_icp: bool = False) -> ArrayLike:
        if method == 'vfm':
            [src, tgt] = self.compute_vfm_correspondences(voxel_map, raw_scan)

        elif method in ['fpfh', 'dip', 'gedi', 'fcgf', 'gcl', 'spinnet']:
            [src, tgt] = self.compute_correspondences(voxel_map[:, :3], raw_scan[:, :3], method)

        else:
            raise ValueError(f"Invalid method: {method}")
        print(f"[{method}] Correspondences: {src.shape[0]}")

        # Find correspondence indices
        downsample_scan = voxel_down_sample(raw_scan[:, :3], self.config.mapping.voxel_size * 0.5)
        voxel_scan = voxel_down_sample(downsample_scan, self.config.mapping.voxel_size * 1.)
        voxel_hash_map = get_voxel_hash_map(self.config)
        voxel_hash_map.add_points(voxel_map[:, :3])
        voxel_map_3d = voxel_hash_map.point_cloud()
        print(f"[{method}] Map size: {voxel_map_3d.shape[0]}, Scan size: {voxel_scan.shape[0]}")
        src_tree = KDTree(voxel_scan, metric="euclidean")
        tgt_tree = KDTree(voxel_map_3d, metric="euclidean")
        src_dist, src_indices = src_tree.query(src, k=1, return_distance=True)
        tgt_dist, tgt_indices = tgt_tree.query(tgt, k=1, return_distance=True)

        # Filter out correspondences that cannot be found in the voxelized point clouds
        if src_dist.max() > .001 or tgt_dist.max() > .001:
            tgt_indices = tgt_indices[src_dist < .001]
            tgt_dist = tgt_dist[src_dist < .001]
            src_indices = src_indices[src_dist < .001]
            src_dist = src_dist[src_dist < .001]
            src_indices = src_indices[tgt_dist < .001]
            # src_dist = src_dist[tgt_dist < .001]
            tgt_indices = tgt_indices[tgt_dist < .001]
            # tgt_dist = tgt_dist[tgt_dist < .001]
        print(f"[{method}] Correspondences after filtering: {src_indices.shape[0]}")

        pcd_src = o3d.geometry.PointCloud()
        pcd_src.points = o3d.utility.Vector3dVector(voxel_scan)
        pcd_tgt = o3d.geometry.PointCloud()
        pcd_tgt.points = o3d.utility.Vector3dVector(voxel_map_3d)

        coors = o3d.utility.Vector2iVector(np.stack((src_indices, tgt_indices), axis=1))

        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcd_src,
            pcd_tgt,
            coors,
            10000,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(50000, 1),
        )
        ransac_pose = np.array(result.transformation)

        if run_icp:
            # Orthogonalize the rotation matrix
            # https://math.stackexchange.com/questions/3292034/normalizing-a-quasi-rotation-matrix
            R = ransac_pose[:3, :3]
            while np.abs(1 - np.linalg.det(R)) > 1e-12:
                R = 3 / 2 * R - 1 / 2 * R @ R.T @ R
            ransac_pose[:3, :3] = R

            # ICP-based refinement
            sigma = self.config.adaptive_threshold.initial_threshold
            pose = register_frame(points=voxel_scan,
                                  voxel_map=voxel_hash_map,
                                  initial_guess=ransac_pose,
                                  max_correspondance_distance=3 * sigma,
                                  kernel=sigma / 3)

            # Visualize the correspondences
            # pose[:3, 3] += np.array([100, -150, 0])
            # pose[:3, 3] += np.array([0, 0, -50])
            # pose[:3, 3] += np.array([0, 0, -1])
            # corr_src = voxel_scan[np.random.choice(np.squeeze(src_indices), 50, replace=False)]
            # corr_src = transform_pcl(corr_src, pose)
            # corr_tgt = voxel_map_3d[np.squeeze(tgt_indices)]
            # publish_correspondences((corr_src, corr_tgt), self.corresponce_pub[0], [1, 0, 0])

            return ransac_pose, pose

        return ransac_pose, None

    def icp_registration(self, voxel_map, raw_scan, initial_pose=None, dist=3) -> ArrayLike:
        # Voxelize: double-downsampling from KISS-ICP
        downsample_scan = voxel_down_sample(raw_scan, self.config.mapping.voxel_size * 0.5)
        voxel_scan = voxel_down_sample(downsample_scan, self.config.mapping.voxel_size * 1.)

        voxel_hash_map = get_voxel_hash_map(self.config)
        voxel_hash_map.add_points(voxel_map)

        map_pcl = voxel_hash_map.point_cloud()
        print(f"Map size: {map_pcl.shape[0]}, Scan size: {voxel_scan.shape[0]}")

        # Run ICP
        a = time()
        sigma = self.config.adaptive_threshold.initial_threshold
        if initial_pose is None:
            initial_pose = np.eye(4)  # The point cloud has already been transformed
        if raw_scan.shape[1] == 3:
            pose = register_frame(points=voxel_scan,
                                  voxel_map=voxel_hash_map,
                                  initial_guess=initial_pose,
                                  max_correspondance_distance=dist * sigma,
                                  kernel=sigma / dist)
        else:
            src_, tgt_ = np.array([[0, 0, 0]]), np.array([[0, 0, 0]])
            pose, src_, tgt_ = register_frame(points=voxel_scan,
                                              voxel_map=voxel_hash_map,
                                              initial_guess=initial_pose,
                                              max_correspondance_distance=dist * sigma,
                                              kernel=sigma / dist,
                                              src_=src_,
                                              tgt_=tgt_)
            print(f"ICP time: {(time() - a):.5f}")
            # print(f"Number of correspondences: {src_.shape}")
            publish_correspondences((src_, tgt_), self.corresponce_pub[0], [1, 0, 0])

        return pose

    def compute_vfm_correspondences(
        self, voxel_map, raw_scan, initial_pose=np.eye(4)) -> Tuple[np.ndarray, np.ndarray]:
        # Voxelize: double-downsampling from KISS-ICP
        downsample_scan = voxel_down_sample(raw_scan, self.config.mapping.voxel_size * 0.5)
        voxel_scan = voxel_down_sample(downsample_scan, self.config.mapping.voxel_size * 1.)

        voxel_hash_map = get_voxel_hash_map(self.config)
        voxel_hash_map.add_points(voxel_map)

        # Publish voxelized map and scan with initial guess
        map_pcl = voxel_hash_map.point_cloud()
        # publish_point_cloud(map_pcl, self.pcl_pub[0])
        pcl = transform_pcl(voxel_scan, initial_pose)
        # publish_point_cloud(pcl, self.pcl_pub[4])

        print(f"Map size: {map_pcl.shape[0]}, Scan size: {voxel_scan.shape[0]}")

        # Mimick the C++ code
        voxel_pcl = voxel_down_sample(pcl, 5.)
        # publish_point_cloud(voxel_pcl, self.pcl_pub[4])

        # Compute correspondences
        correspondences = voxel_hash_map.get_vfm_correspondences(voxel_pcl, .8)

        if correspondences[0].shape[0] < 75:
            print("[WARNING] Voxelized too sparse, retrying with a larger voxel size")
            voxel_pcl = voxel_down_sample(pcl, 1.0)
            correspondences = voxel_hash_map.get_vfm_correspondences(voxel_pcl, .8)

        return correspondences

    def compute_correspondences(self,
                                voxel_map,
                                raw_scan,
                                method: str,
                                mutual_filter: bool = False) -> Tuple[ArrayLike, ArrayLike]:

        if method in self.map_descriptor_cache:
            (down_map, feats_map) = self.map_descriptor_cache[method]
        else:
            (down_map, feats_map) = (None, None)

        if method == 'fpfh':
            down_scan, feats_scan = extract_fpfh_features(raw_scan, .1)
            if down_map is None:
                down_map, feats_map = extract_fpfh_features(voxel_map, .1)

        elif method == 'dip':
            dip_weight_path = Path(__file__).parent / 'dip' / 'final_chkpt.pth'
            down_scan, feats_scan = extract_dip_features(raw_scan, .1, dip_weight_path)
            if down_map is None:
                down_map, feats_map = extract_dip_features(voxel_map, .1, dip_weight_path)

        elif method == 'gedi':
            gedi_weight_path = Path(__file__).parent / 'gedi' / 'chkpt.tar'
            down_scan, feats_scan = extract_gedi_features(raw_scan, .1, np.inf, gedi_weight_path)
            if down_map is None:
                down_map, feats_map = extract_gedi_features(voxel_map, .1, np.inf, gedi_weight_path)

        elif method == 'gcl':
            gcl_weight_path = Path(__file__).parent / 'gcl' / 'kitti_chkpt.pth'
            down_scan, feats_scan = extract_gcl_features(raw_scan, .3, gcl_weight_path)
            if down_map is None:
                down_map, feats_map = extract_gcl_features(voxel_map, .3, gcl_weight_path)

        elif method == 'fcgf':
            fcgf_weight_path = Path(__file__).parent / 'fcgf' / '2019-07-31_19-37-00.pth'
            # Voxel size is from KITTI config file (for this checkpoint)
            down_scan, feats_scan = extract_fcgf_features(raw_scan, .3, fcgf_weight_path)
            if down_map is None:
                down_map, feats_map = extract_fcgf_features(voxel_map, .3, fcgf_weight_path)

        elif method == 'spinnet':
            spinnet_weigth_path = Path(__file__).parent / 'spinnet' / 'KITTI_best.pkl'
            n_points = 7500
            down_scan, feats_scan = extract_spinnet_features(raw_scan, n_points,
                                                             spinnet_weigth_path)
            if down_map is None:
                down_map, feats_map = extract_spinnet_features(voxel_map, n_points,
                                                               spinnet_weigth_path)

        else:
            raise ValueError(f"Invalid method: {method}")
        self.map_descriptor_cache[method] = (down_map, feats_map)
        torch.cuda.empty_cache()

        def find_correspondences(feats0, feats1, n_points=5000, mutual_filter=True):
            # This function is adapted from TEASER++
            # If mututal_filter is True, return all points that are mutual correspondences.

            def find_knn_cpu(feat0, feat1, return_distance=False):
                feat1tree = cKDTree(feat1)
                dists, nn_inds = feat1tree.query(feat0,
                                                 k=1,
                                                 workers=-1,
                                                 distance_upper_bound=np.inf)
                assert feat0.shape[0] == len(nn_inds)
                if return_distance:
                    return nn_inds, dists
                else:
                    return nn_inds

            # Find correspondences from 0 to 1
            nns01, dists = find_knn_cpu(feats0, feats1, return_distance=True)
            corres01_idx0 = np.arange(len(nns01))
            corres01_idx1 = nns01

            # Filter out points that are not matched
            # Unmatched points are assigned the last index + 1
            dists = dists[corres01_idx1 != len(feats1)]
            corres01_idx0 = corres01_idx0[corres01_idx1 != len(feats1)]
            corres01_idx1 = corres01_idx1[corres01_idx1 != len(feats1)]
            # From here on, not all points in 0 have a correspondence in 1

            if not mutual_filter:
                # Keep only top N correspondences (smallest distance)
                n = min(n_points, len(dists) - 1)
                top_n_idx = np.argpartition(dists, n)[:n]
                # dists = dists[top_n_idx] # Not used
                corres01_idx0 = corres01_idx0[top_n_idx]
                corres01_idx1 = corres01_idx1[top_n_idx]

                return corres01_idx0, corres01_idx1

            # Find correspondences from 1 to 0
            nns10 = find_knn_cpu(feats1, feats0, return_distance=False)
            # corres10_idx1 = np.arange(len(nns10))
            corres10_idx0 = nns10

            # Filter out points that are not matched
            # Unmatched points are assigned the last index + 1
            corres10_idx0 = corres10_idx0[corres10_idx0 != len(feats0)]
            # From here on, not all points in 1 have a correspondence in 0

            mutual_filter = (corres10_idx0[corres01_idx1] == corres01_idx0)
            corres_idx0 = corres01_idx0[mutual_filter]
            corres_idx1 = corres01_idx1[mutual_filter]

            # Filter out points that are not matched
            corres_idx0 = corres_idx0[corres_idx1 != len(feats1)]
            corres_idx1 = corres_idx1[corres_idx1 != len(feats1)]

            return corres_idx0, corres_idx1

        corrs_scan, corrs_map = find_correspondences(feats_scan,
                                                     feats_map,
                                                     n_points=5000,
                                                     mutual_filter=mutual_filter)
        source = down_scan[corrs_scan]
        target = down_map[corrs_map]
        return source, target

    def make_step(self) -> None:
        if self.scene_idx >= len(self.filenames):
            self.stop_publishing = True
            return

        print(f"Current scene ID: {self.scene_idx} | {self.filenames[self.scene_idx].name}")
        scene = read_scenes(self.filenames[self.scene_idx])

        # Accumulate the local map and voxelize
        scene_local_map = []
        voxel_size = .25
        pbar = tqdm(total=len(scene["map_poses"]), desc="Voxelizing map")
        for pose, pcl in zip(scene["map_poses"], scene["map_point_clouds"]):
            # Remove points w/o descriptors
            pcl = pcl[np.sum(pcl[:, 3:], axis=1) > 0]
            pcl = voxel_down_sample(pcl, voxel_size).astype(pcl.dtype)
            scene_local_map.append(transform_pcl(pcl, pose))
            pbar.update(1)
        scene_local_map = np.concatenate(scene_local_map, axis=0).astype(np.float32)
        pbar.close()
        if scene_local_map.shape[0] > 1000000:
            # Split up voxelization due to memory constraints
            mean_3d = np.mean(scene_local_map[:, :3], axis=0)
            map_a = voxel_down_sample(scene_local_map[scene_local_map[:, 0] > mean_3d[0]],
                                      voxel_size).astype(scene_local_map.dtype)
            map_b = voxel_down_sample(scene_local_map[scene_local_map[:, 0] <= mean_3d[0]],
                                      voxel_size).astype(scene_local_map.dtype)
            scene_local_map = np.concatenate([map_a, map_b], axis=0)
        else:
            scene_local_map = voxel_down_sample(scene_local_map,
                                                voxel_size).astype(scene_local_map.dtype)
        scene_local_map = scene_local_map[:, :3 + 384]
        print(f"Map voxelization ({voxel_size}m): {scene_local_map.shape[0]}")

        # Random alterations to the poses
        rng = np.random.RandomState(seed=42)
        rng_cluster_removal = np.random.RandomState(seed=42)

        # Iterate over the scans of other sequences
        for i, (pose,
                point_cloud) in enumerate(zip(scene["scene_poses"], scene["scene_point_clouds"])):
            results = {}
            print('=' * 80)
            self.scan_idx = i
            tmp = point_cloud.shape
            point_cloud = voxel_down_sample(point_cloud, .1).astype(point_cloud.dtype)
            print(f"Scan voxelization (0.1m): {tmp[0]} -> {point_cloud.shape[0]}")
            # =====================================================================
            # Improve GT values manually
            if 'robotcar' in self.filenames[self.scene_idx].as_posix():
                if self.scene_idx == 0:
                    pose[:3, 3] += np.array([15, 5, 0])
                elif self.scene_idx == 6:
                    pose[:3, 3] += np.array([3.5, 2, 0])
                elif self.scene_idx == 7:
                    pose[:3, 3] += np.array([10, 8, 0])
                elif self.scene_idx == 10:
                    pose[:3, 3] += np.array([5, 2, 0])
                elif self.scene_idx == 12:
                    pose[:3, 3] += np.array([3, 1, 0])
                elif self.scene_idx == 13:
                    pose[:3, 3] += np.array([4, 2, 0])
                elif self.scene_idx == 14:
                    pose[:3, 3] += np.array([-2, 2, 0])
                elif self.scene_idx == 15:
                    pose[:3, 3] += np.array([3, 2, 0])
                elif self.scene_idx == 17:
                    pose[:3, 3] += np.array([2, 2, 0])
                elif self.scene_idx == 18:
                    pose[:3, 3] += np.array([8, 2, 0])
                elif self.scene_idx == 21:
                    pose[:3, 3] += np.array([2, 1, 0])
                elif self.scene_idx == 23:
                    pose[:3, 3] += np.array([5, 2, 0])
                elif self.scene_idx == 24:
                    pose[:3, 3] += np.array([0, 2, 0])
            # =====================================================================

            # =====================================================================
            # EXPERIMENT: Load KITTI data instead
            if self.scan_to_scan == 'kitti':
                filenames = sorted(
                    Path('/data/kitti_odom/dataset/sequences/08/velodyne').glob("*.bin"))
                kitti_i = np.random.choice(len(filenames) - 10)
                scene_local_map = np.fromfile(filenames[kitti_i],
                                              dtype=np.float32).reshape(-1, 4)[:, :3]
                point_cloud = np.fromfile(filenames[kitti_i + 1],
                                          dtype=np.float32).reshape(-1, 4)[:, :3]
                scene_local_map = voxel_down_sample(scene_local_map,
                                                    .1).astype(scene_local_map.dtype)
                point_cloud = voxel_down_sample(point_cloud, .1).astype(point_cloud.dtype)
                print(f"Map {scene_local_map.shape[0]}, Scan {point_cloud.shape[0]}")
                pose = np.eye(4)
            # =====================================================================

            # Compute GT pose with KISS-ICP w/o noise
            print(f'{"--- Ground truth: ICP w/o noise ---":-^80}')
            local_map_ = scene_local_map[:, :3]
            gt_pose = self.icp_registration(local_map_, point_cloud[:, :3], pose)
            point_cloud_t = transform_pcl(point_cloud[:, :3], gt_pose)
            publish_point_cloud(point_cloud_t, self.pcl_pub[1])
            print(f'{"--- Ground truth: ICP w/o noise ---":-^80}')

            # =====================================================================
            # For the release files: add the GT poses to the files
            if False:
                release_file = self.filenames[
                    self.
                    scene_idx].parent / 'release' / f"{self.filenames[self.scene_idx].stem}.json"
                with open(release_file, 'r', encoding="utf-8") as f:
                    data = json.load(f)
                data["registration"][i]["pose"] = gt_pose.tolist()
                with open(release_file, 'w', encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=4)
                continue
            # =====================================================================

            # DEBUG: Crop map around the GT pose + random offset
            if False:
                center_position = gt_pose[:3, 3].copy()
                center_position[:2] = center_position[:2] + rng.uniform(-25, 25, 2)
                point_distances = np.array(scene_local_map[:, :2] - center_position[:2])
                point_distances = np.max(np.abs(point_distances), axis=1)
                local_map = scene_local_map[point_distances < 125]
            else:
                local_map = scene_local_map
            # =====================================================================
            # EXPERIMENT: dummy experiment showing that scan-to-scan is working
            if self.scan_to_scan == 'data':
                map_poses = np.array([pose[:3, 3] for pose in scene["map_poses"]])
                dist = np.linalg.norm(pose[:3, 3] - map_poses, axis=1)
                local_map = scene["map_point_clouds"][dist.argmin()]
                local_map = transform_pcl(local_map, scene["map_poses"][dist.argmin()])
                local_map = voxel_down_sample(local_map, .1).astype(local_map.dtype)
            # =====================================================================
            # EXPERIMENT: Filter the map based on color value
            if self.cluster_removal_prob > 0:
                remove_classes = [
                    # tree (broadleaf and coniferious)
                    np.array([[217, 60, 165], [118, 105, 57]], dtype=np.float32),
                ]
            else:
                remove_classes = []

            # Load cached filtered map
            if len(remove_classes) > 0:
                local_map_pca = self.dino_generator.run_pca(local_map[:, 3:], n_components=3)

                for remove_class in remove_classes:
                    # Indices are of those points that might be removed
                    del_idx = []
                    for color in remove_class:
                        distance = np.linalg.norm(local_map_pca - color, axis=1)
                        del_idx.append(np.flatnonzero(distance < 50))
                    del_idx = np.concatenate(del_idx)

                    # Remove isolated points from the segmentation
                    knn = FaissKNeighbors()
                    knn.fit(local_map[del_idx, :3], del_idx)
                    n_neighbors = knn.n_neighbors_in_radius(local_map[del_idx, :3], 10, .5)
                    del_idx = del_idx[n_neighbors >= 3]
                    keep_idx = np.arange(local_map.shape[0])
                    keep_idx = np.delete(keep_idx, del_idx)
                    # Extend the segmentation based on Euclidean distance
                    knn = FaissKNeighbors()
                    knn.fit(local_map[keep_idx, :3], keep_idx)
                    knn_idx = knn.query(local_map[del_idx, :3], 50, .5)
                    del_idx = np.concatenate([del_idx, knn_idx])
                    keep_idx = np.arange(local_map.shape[0])
                    keep_idx = np.delete(keep_idx, del_idx)

                    # =====================================================================
                    # 0) Visualize the original map
                    # local_map_rgb = np.c_[local_map[:, :3], local_map_pca]
                    # publish_point_cloud(local_map_rgb, self.pcl_pub[0])
                    # input("Press Enter to continue with next visualization...")
                    # =====================================================================
                    # 1) Colorize the tree-like points
                    # local_map_rgb = np.ones_like(local_map[:, :3])
                    # local_map_rgb[keep_idx] = .7 * np.ones((1, 3), dtype=np.float32)
                    # local_map_rgb[del_idx] = np.array([1, 0, 0], dtype=np.float32)
                    # local_map_rgb = np.c_[local_map[:, :3], local_map_rgb]
                    # publish_point_cloud(local_map_rgb, self.pcl_pub[0])
                    # input("Press Enter to continue with next visualization...")
                    # =====================================================================

                    a = time()
                    clusterer = hdbscan.HDBSCAN(min_cluster_size=100, min_samples=25)
                    cluster_labels = clusterer.fit_predict(local_map[del_idx, :3])
                    print(
                        f"Clustering time: {(time() - a):.5f} | Clusters: {cluster_labels.max() + 1} "
                        f"| Outliers: {np.sum(cluster_labels == -1)}")
                    del_idx = del_idx[cluster_labels != -1]
                    cluster_labels = cluster_labels[cluster_labels != -1]

                    # =====================================================================
                    # 2) Colorize the clusters
                    # cluster_colors = np.random.rand(cluster_labels.max() + 1, 3)
                    # local_map_rgb = np.ones_like(local_map[:, :3])
                    # local_map_rgb[keep_idx] = .7 * np.ones((1, 3), dtype=np.float32)
                    # local_map_rgb[del_idx] = cluster_colors[cluster_labels]
                    # local_map_rgb = np.c_[local_map[:, :3], local_map_rgb]
                    # publish_point_cloud(local_map_rgb, self.pcl_pub[0])
                    # input("Press Enter to continue with next visualization...")
                    # =====================================================================
                    # For EXPERIMENT: Add trees to map -> Save clusters in file
                    # for label in tqdm(range(cluster_labels.max()), desc="Saving clusters"):
                    #     cluster_pcl = local_map[del_idx][cluster_labels == label]
                    #     print(label, cluster_pcl.shape)
                    #     publish_point_cloud(cluster_pcl, self.pcl_pub[1])
                    #     user_input = input("Save (y), Skip (s), Exit (e)")
                    #     if user_input == 'e':
                    #         break
                    #     elif user_input == 's':
                    #         continue
                    #     cluster_pcl_file = Path(__file__).parent / 'cluster_pcl' / f"{label:03}.npy"
                    #     cluster_pcl_file.parent.mkdir(parents=True, exist_ok=True)
                    #     np.save(cluster_pcl_file, cluster_pcl)
                    # =====================================================================

                    # Remove entire clusters from the map
                    for label in range(cluster_labels.max() + 1):
                        # break # For visualization only
                        # If true, remove the cluster from the to-be-deleted set, i.e., keep in map.
                        remove_chance = self.cluster_removal_prob  # 0.5 # from [0, 1]
                        if rng_cluster_removal.standard_normal() > scipy.stats.norm.ppf(
                                remove_chance):
                            del_idx = del_idx[cluster_labels != label]
                            cluster_labels = cluster_labels[cluster_labels != label]
                    keep_idx = np.arange(local_map.shape[0])
                    keep_idx = np.delete(keep_idx, del_idx)

                    # =====================================================================
                    # 3) Visualize the to-be-deleted points
                    # local_map_rgb = np.ones_like(local_map[:, :3])
                    # local_map_rgb[keep_idx] = .7 * np.ones((1, 3), dtype=np.float32)
                    # local_map_rgb[del_idx] = np.array([1, 0, 0], dtype=np.float32)
                    # local_map_rgb = np.c_[local_map[:, :3], local_map_rgb]
                    # publish_point_cloud(local_map_rgb, self.pcl_pub[0])
                    # input("Press Enter to continue with next visualization...")
                    # =====================================================================

                    local_map = local_map[keep_idx]
                    local_map_pca = local_map_pca[keep_idx]
                    self.points_in_map.append(local_map.shape[0])

            # EXPERIMENT: Add trees from NCLT to Robotcar scene
            add_clusters = False
            if add_clusters:
                cluster_pcl_files = list((Path(__file__).parent / 'cluster_pcl').glob("*.npy"))
                sampled_files = rng_cluster_removal.choice(cluster_pcl_files,
                                                           int(self.cluster_removal_prob * 100),
                                                           replace=True).tolist()
                new_clusters = []
                for cluster_pcl_file in sampled_files:
                    cluster_pcl = np.load(cluster_pcl_file)
                    # Center the cluster
                    cluster_pcl[:, :3] -= np.mean(cluster_pcl[:, :3], axis=0)
                    # Position the cluster in the map
                    cluster_pose = gt_pose.copy()
                    idx = rng_cluster_removal.choice(local_map.shape[0], 1)
                    cluster_pose[:2, 3] = local_map[idx, :2]
                    # Align the cluster with the ground
                    dist = np.linalg.norm(local_map[:, :2] - cluster_pose[:2, 3].reshape(1, 2),
                                          axis=1)
                    max_max = np.max(local_map[dist < 2, 2])
                    cluster_pcl[:, :3] = transform_pcl(cluster_pcl[:, :3], cluster_pose)
                    cluster_pcl[:, 2] += (max_max - np.max(cluster_pcl[:, 2]))
                    new_clusters.append(cluster_pcl)
                cluster_pcl = np.concatenate(new_clusters, axis=0)
                local_map = np.concatenate([local_map, cluster_pcl], axis=0)
                cluster_pca = self.dino_generator.run_pca(cluster_pcl[:, 3:], n_components=3)
                local_map_pca = np.concatenate([local_map_pca, cluster_pca], axis=0)
                # =====================================================================
                # 4) Add the clusters back to the map
                # local_map_rgb = .7 * np.ones_like(local_map[:, :3], dtype=np.float32)
                # local_map_rgb[-cluster_pcl.shape[0]:] = np.array([1, 0, 0], dtype=np.float32)
                # local_map_rgb = np.c_[local_map[:, :3], local_map_rgb]
                # publish_point_cloud(local_map_rgb, self.pcl_pub[0])
                # input("Press Enter to continue with next visualization...")
                # =====================================================================

            if len(remove_classes) > 0:
                print(f"Semantic-filtered map: {local_map.shape[0]}")
                local_map_rgb = np.c_[local_map[:, :3], local_map_pca]
                publish_point_cloud(local_map_rgb, self.pcl_pub[0])
            else:
                self.publish_map(local_map, self.pcl_pub[0])
            # =====================================================================
            # local_map_pca = np.c_[local_map[:, :3],
            #                       self.dino_generator.run_pca(local_map[:, 3:], n_components=3)]
            # local_map_pca[:, 3:] = local_map_pca[:, 3:] / 255.
            # publish_point_cloud(local_map_pca, self.pcl_pub[0])
            # =====================================================================

            # =====================================================================
            # EXPERIMENT: dummy experiment showing that scan-to-scan is working
            if self.scan_to_scan is not None:
                # Simulate error of LiDAR odometry
                t_noise = np.r_[rng.normal(0, 10, 2), rng.normal(0, 1, 1)]
                r_noise = np.r_[rng.normal(0, 2, 2), rng.normal(0, 10, 1)]
                # =====================================================================
                initial_pose = gt_pose.copy()
                euler = R.from_matrix(initial_pose[:3, :3]).as_euler("xyz")
                initial_pose[:3, :3] = R.from_euler("xyz", euler + np.deg2rad(r_noise)).as_matrix()
                initial_pose[:3, 3] = initial_pose[:3, 3] + t_noise
            # =====================================================================
            else:
                initial_pose = np.eye(4)
            # =====================================================================

            # Apply the initial guess
            # IMPORTANT: All computed poses will be relative to this initial guess
            point_cloud = transform_pcl(point_cloud, initial_pose)
            publish_point_cloud(point_cloud[:, :3], self.pcl_pub[2])

            # =====================================================================
            # With RANSAC
            print(f'{"--- RANSAC ---":-^80}')
            descriptor_icp = [
                ('fpfh', True),
                ('dip', True),
                ('gedi', True),
                ('fcgf', True),
                ('gcl', True),
                ('spinnet', True),
                ('vfm', True),
            ]
            if i == 0 or self.scan_to_scan is not None or self.cluster_removal_prob > 0:
                self.map_descriptor_cache = {}
            for method, run_icp in descriptor_icp:
                if self.scan_to_scan == 'kitti' and method == 'vfm':
                    continue
                key = f"{method}_ransac"
                results[key], results[f"{key}_icp"] = self.ransac_registration(
                    local_map, point_cloud, method, run_icp)
                if (method, run_icp) != descriptor_icp[-1]:
                    print('-' * 40)
            # point_cloud_t = transform_pcl(point_cloud[:, :3], results['vfm_ransac_icp'])
            # point_cloud_t = np.c_[point_cloud_t[:, :3],
            #                       self.dino_generator.run_pca(point_cloud[:, 3:], n_components=3) / 255.]
            # point_cloud_t[:, 2] -= 30 # For figures in paper: results: 1; correspondences: 30
            # publish_point_cloud(point_cloud_t, self.pcl_pub[3])
            print(f'{"--- RANSAC ---":-^80}')
            # =====================================================================
            # With TEASER
            print(f'{"--- TEASER ---":-^80}')
            descriptor_icp = [
                ('fpfh', True),
                ('vfm', True),
            ]
            for method, run_icp in descriptor_icp:
                if self.scan_to_scan == 'kitti' and method == 'vfm':
                    continue
                key = f"{method}_teaser"
                results[key], results[f"{key}_icp"] = self.teaser_registration(
                    local_map, point_cloud, method, run_icp)
                if (method, run_icp) != descriptor_icp[-1]:
                    print('-' * 40)
            # point_cloud_t = transform_pcl(point_cloud[:, :3], results['fpfh_teaser'])
            # publish_point_cloud(point_cloud_t, self.pcl_pub[4])
            print(f'{"--- TEASER ---":-^80}')
            # =====================================================================
            # With PointDSC
            print(f'{"--- PointDSC ---":-^80}')
            descriptor_icp = [
                ('fpfh', True),
                ('fcgf', True),
            ]
            for method, run_icp in descriptor_icp:
                key = f"{method}_pointdsc"
                results[key], results[f"{key}_icp"] = self.pointsdc_registration(
                    local_map, point_cloud, method, 10000, run_icp)
                if (method, run_icp) != descriptor_icp[-1]:
                    print('-' * 40)
            # point_cloud_t = transform_pcl(point_cloud[:, :3], results['fcgf_pointdsc_icp'])
            # publish_point_cloud(point_cloud_t, self.pcl_pub[4])
            print(f'{"--- PointDSC ---":-^80}')
            # =====================================================================
            # Vanilla ICP
            print(f'{"--- Vanilla ICP ---":-^80}')
            results['icp'] = self.icp_registration(local_map[:, :3], point_cloud[:, :3], dist=7)
            # point_cloud_t = transform_pcl(point_cloud[:, :3], results['icp'])
            # publish_point_cloud(point_cloud_t, self.pcl_pub[3])
            print(f'{"--- Vanilla ICP ---":-^80}')
            # =====================================================================

            # Compute errors and print results
            print('=' * 80)
            print(
                f"Scene: {self.scene_idx+1}/{len(self.filenames)} ({self.filenames[self.scene_idx].name}) "
                f"| {i+1}/{len(scene['scene_poses'])}")
            print_msg(gt_pose, "GT pose")
            print_msg(initial_pose, "Initial guess")
            print('-' * 80)
            for k, v in results.items():
                if v is None:
                    continue
                v = v @ initial_pose
                rte, rre = self.compute_errors(gt_pose, v, k)
                if rte < .3 and rre < 15:
                    print_msg(v, k, color="green")
                else:
                    print_msg(v, k)
            print('-' * 80)
            print(f"Points in map: {local_map.shape}")
            print('=' * 80)

            if i < len(scene["scene_poses"]) - 1 and self.interactive:
                input("Press Enter to continue with next sequence...\n")

        self.scene_idx += 1

        # Print errors
        print('=' * 80)
        for method, rot_error in self.rot_errors.items():
            print(
                f"Rotation error ({method:<20}): {np.mean(rot_error):.3f} ± {np.std(rot_error):.3f}"
            )
        print('-' * 80)
        for method, trans_error in self.trans_errors.items():
            print(
                f"Translat error ({method:<20}): {np.mean(trans_error):.3f} ± {np.std(trans_error):.3f}"
            )
        print('-' * 80)
        thresholds = [
            (.3, 15),  # PointDSC
            (.6, 1.5),  # GCL
            (2, 5),  # D3Feat, SpinNet
        ]
        str = f"{'':<20}: "
        for threshold in thresholds:
            str += f"{threshold[0]:>3}, {threshold[1]:<3} | "
        print(str[:-2])
        for method in self.rot_errors.keys():
            str = f"{method:<20}: "
            for threshold in thresholds:
                str += f"{100 * self.compute_success_rate(method, *threshold):>8.2f} | "
            print(str[:-2])
        print('-' * 80)
        print(f"Points in map: {np.mean(self.points_in_map)}")
        print('=' * 80)

    def publish_map(self, map, publisher) -> None:
        voxel_hash_map = get_voxel_hash_map(self.config)
        voxel_hash_map.add_points(map)
        map_pcl = voxel_hash_map.point_cloud()
        publish_point_cloud(map_pcl, publisher)

    def compute_errors(self, pose: ArrayLike, gt_pose: ArrayLike,
                       method: str) -> Tuple[float, float]:
        # Translation: meter
        # Rotation: degree

        # Compute the rotation error as the geodesic distance (see TEASER)
        R = pose[:3, :3]
        R_gt = gt_pose[:3, :3]
        rot_error = abs(np.arccos(min(max(((R.T @ R_gt).trace() - 1) / 2, -1.0), 1.0)))
        rot_error = np.rad2deg(rot_error)

        # Compute the translation error as the 2-norm of the distance vector
        t = pose[:3, 3]
        t_gt = gt_pose[:3, 3]
        trans_error = np.linalg.norm(t - t_gt)

        if method not in self.rot_errors:
            self.rot_errors[method] = []
            self.trans_errors[method] = []
        self.rot_errors[method].append(rot_error)
        self.trans_errors[method].append(trans_error)

        return trans_error, rot_error

    def compute_success_rate(self, method: str, translation_threshold, rotation_threshold) -> float:
        successful_translation = np.array(self.trans_errors[method]) < translation_threshold
        successful_rotation = np.array(self.rot_errors[method]) < rotation_threshold
        success_rate = np.mean(successful_translation & successful_rotation)
        return success_rate


def signal_handler(sig, frame):
    print("Manual termination triggered")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=str)
    parser.add_argument("--interactive", action="store_true", help="Pauses after each scene.")
    args = parser.parse_args()

    datetime_now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # EXPERIMENT: Removing/adding trees from/to the map
    cluster_removal_probs = [
        ('000', 0),
        ('020', .2),
        ('040', .4),
        ('060', .6),
        ('080', .8),
        ('100', 1.0),
        ('010', .1),
        ('030', .3),
        ('050', .5),
        ('070', .7),
        ('090', .9),
    ]
    # Make this the default case for running the algorithm. I.e., no specific experiment
    if len(cluster_removal_probs) == 0:
        cluster_removal_probs = [(None, 0)]

    for cluster_removal_prob in cluster_removal_probs:
        scan_to_scan = None
        mode_str = f"_{cluster_removal_prob[0]}" if cluster_removal_prob[0] is not None else ""

        # EXPERIMENT:
        # 2x scan-to-scan registration (KITTI and NCLT/RobotCar),
        # 1x scan-to-map-registration (NCLT/RobotCar)
        # for scan_to_scan in ['kitti', 'data', None]:
        #     cluster_removal_prob = (None, 0)
        #     mode_str = f"_{scan_to_scan}" if scan_to_scan is not None else ""

        data_dir = Path(args.folder)
        node = RegistrationNode(data_dir, args.interactive, cluster_removal_prob[1], scan_to_scan)

        time_per_step = []
        rospy.init_node("registration_node")
        while not rospy.is_shutdown():
            start = time()
            node.make_step()
            time_per_step.append(time() - start)
            avg_time = np.mean(time_per_step)
            remaining_time = (len(node.filenames) - node.scene_idx) * avg_time
            remaining_hours = remaining_time // 3600
            remaining_minutes = (remaining_time % 3600) // 60
            print(f"\033[93m Average time per step: {avg_time:.3f} s | "
                  f"Remaining: {remaining_hours}h {remaining_minutes}min \033[0m")
            if node.stop_publishing:
                break
            if args.interactive:
                input("Press Enter to continue with next scene...\n")

        error_file = data_dir / f"errors_{datetime_now}" / f"mode{mode_str}.pkl"
        error_file.parent.mkdir(exist_ok=True, parents=True)
        with open(error_file, "wb") as f:
            pickle.dump(
                {
                    "rot": node.rot_errors,
                    "trans": node.trans_errors,
                    "points_in_map": node.points_in_map
                }, f)
