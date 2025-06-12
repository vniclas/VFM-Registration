import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d
from dataloader import NCLT, OxfordRobotcar
from kiss_icp.voxelization import voxel_down_sample
from tqdm import tqdm

from vfm_reg.utils import transform_pcl


def main(dataset_dir: Path, scene_file: Path):
    if 'nclt' in dataset_dir.name:
        Dataset = NCLT
        date_idx = 1
    elif 'robotcar' in dataset_dir.name:
        Dataset = OxfordRobotcar
        date_idx = 0
    else:
        raise ValueError("Unknown dataset")

    with open(scene_file, "r") as f:
        scene_data = json.load(f)

    # Create map
    map_date = scene_data["mapping"]["point_clouds"][0].split("/")[date_idx][:19]
    map_sequence = Dataset(map_date, dataset_dir)
    voxel_size = .25
    map_pcl = []
    pbar = tqdm(total=len(scene_data["mapping"]["point_clouds"]), desc="Creating map")
    for pcl_file, pcl_pose in zip(scene_data["mapping"]["point_clouds"],
                                  scene_data["mapping"]["poses"]):
        pcl_file = Path(dataset_dir) / pcl_file
        pcl = map_sequence.read_pcl(filename=pcl_file)
        pcl = voxel_down_sample(pcl, voxel_size).astype(pcl.dtype)
        map_pcl.append(transform_pcl(pcl, np.array(pcl_pose)))
        pbar.update(1)
    map_pcl = np.concatenate(map_pcl, axis=0).astype(np.float32)
    pbar.close()
    print("Voxelizing map...")
    if map_pcl.shape[0] > 1000000:
        # Split up voxelization due to memory constraints
        mean_3d = np.mean(map_pcl[:, :3], axis=0)
        map_a = voxel_down_sample(map_pcl[map_pcl[:, 0] > mean_3d[0]],
                                  voxel_size).astype(map_pcl.dtype)
        map_b = voxel_down_sample(map_pcl[map_pcl[:, 0] <= mean_3d[0]],
                                  voxel_size).astype(map_pcl.dtype)
        map_pcl = np.concatenate([map_a, map_b], axis=0)
    else:
        map_pcl = voxel_down_sample(map_pcl, voxel_size).astype(map_pcl.dtype)

    # Make Open3D point cloud
    map_pcl[:, 2] *= -1  # Invert z-axis
    map_pcd = o3d.geometry.PointCloud()
    map_pcd.points = o3d.utility.Vector3dVector(map_pcl)
    map_pcd.paint_uniform_color([0, 0, 1])

    # Loop over registration scans
    for i, registration in enumerate(scene_data["registration"]):
        print(f"Loading registration {i+1} of current scene...")

        registration_date = registration["point_cloud"].split("/")[date_idx][:19]
        registration_sequence = Dataset(registration_date, dataset_dir)
        pcl_file = Path(dataset_dir) / registration["point_cloud"]
        pcl = registration_sequence.read_pcl(filename=pcl_file)
        pcl_pose = registration["pose"]
        pcl = transform_pcl(pcl, np.array(pcl_pose))

        # Make Open3D point cloud
        pcl[:, 2] *= -1  # Invert z-axis
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcl)
        pcd.paint_uniform_color([1, 0, 0])

        # Visualize the scene
        if i == len(scene_data["registration"]) - 1:
            print("Close the window to continue with next scene...")
        else:
            print("Close the window to continue with next registration...")
        o3d.visualization.draw_geometries([map_pcd, pcd])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("scene_folder", type=str)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_folder)
    scene_dir = Path(args.scene_folder)
    if args.scene_folder[-5:] == ".json":
        scene_files = [scene_dir]
    else:
        scene_files = sorted(list(scene_dir.glob("*.json")))

    print(f"Found {len(scene_files)} scene file(s).")

    for scene_file in scene_files:
        main(dataset_dir, scene_file)
