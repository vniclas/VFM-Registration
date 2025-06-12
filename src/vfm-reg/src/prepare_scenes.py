import argparse
import json
from pathlib import Path
import h5py
from typing import List
import cv2

import numpy as np
from dataloader import NCLT, OxfordRobotcar
from kiss_icp.voxelization import voxel_down_sample
from tqdm import tqdm

from vfm_reg.image_features import ImageFeatureGenerator


def save_scene(filename: Path, sequences: List[str], map_poses, map_point_clouds, seq_poses,
               seq_point_clouds) -> None:
    # File structure:
    #   /map
    #     /sequence name
    #       /point_cloud
    #       /pose
    #   /scans
    #     /sequence name
    #       /point_cloud
    #       /pose

    filename.parent.mkdir(parents=True, exist_ok=True)
    file = h5py.File(filename, "w")

    map_group = file.create_group(f"map/{sequences[0]}")
    pose_group = map_group.create_group("pose")
    point_cloud_group = map_group.create_group("point_cloud")
    for j in range(len(map_poses)):
        pose_group.create_dataset(f"{j:03}", data=map_poses[j])
        point_cloud_group.create_dataset(f"{j:03}", data=map_point_clouds[j])

    scans_group = file.create_group(f"scans")
    for j in range(len(seq_poses)):
        # This sequence has no hits
        if seq_poses[j] is None:
            continue
        scan_group = scans_group.create_group(f"{sequences[j+1]}")
        scan_group.create_dataset("pose", data=seq_poses[j])
        scan_group.create_dataset("point_cloud", data=seq_point_clouds[j])

    file.close()


def create_descriptors(image_files, sequence, feature_generator, pcl):
    images = sequence.read_images(filenames=image_files)
    img_descriptors = {
        camera: feature_generator.get_image_features(image, upsample=True) \
            for camera, image in images.items()
    }

    # Set descriptors to zero if the image is empty
    for camera, features in img_descriptors.items():
        image = images[camera]
        mask = np.all(image == 0, axis=-1)
        features[mask] = np.zeros_like(features[mask])
        img_descriptors[camera] = features

    n_features = img_descriptors[list(img_descriptors.keys())[0]].shape[1]
    pcl_descriptors = np.zeros((pcl.shape[1], n_features), dtype=np.float32)
    pcl_indices, pcl_image_features = None, None

    # Project point cloud to images
    pcl = np.insert(pcl, 3, values=1, axis=1).T
    for camera, image in images.items():
        image_features = img_descriptors[camera]

        if isinstance(sequence, NCLT):
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        u, v, pcl_indices_ = sequence.project_pcl_to_image(pcl, image, camera)
        if len(pcl_indices_) == 0:
            continue

        if isinstance(sequence, NCLT):
            image_features = np.rot90(image_features, k=1, axes=(0, 1))  # clockwise

        if pcl_indices is None:
            pcl_indices = pcl_indices_
            pcl_image_features = np.stack([image_features[v_, u_, :] for u_, v_ in zip(u, v)],
                                          axis=0)
        else:
            pcl_indices = np.r_[pcl_indices, pcl_indices_]
            pcl_image_features = np.r_[
                pcl_image_features,
                np.stack([image_features[v_, u_, :] for u_, v_ in zip(u, v)], axis=0)]

        if isinstance(sequence, NCLT):
            image_features = np.rot90(image_features, k=3, axes=(0, 1))  # counter-clockwise

    # Check for double indices
    unique_indices, unique_indices_idx = np.unique(pcl_indices, return_index=True)
    if len(unique_indices) < pcl_indices.shape[0]:
        pcl_indices = pcl_indices[unique_indices_idx]
        # Take feature from any camera
        pcl_image_features = pcl_image_features[unique_indices_idx]
    pcl_descriptors = np.zeros((pcl.shape[1], pcl_image_features.shape[1]), dtype=np.float32)
    # If a point has no descriptor, leave it at zero
    pcl_descriptors[pcl_indices, :] = pcl_image_features.astype(np.float32)

    pcl = pcl[:3, :].T
    return pcl_descriptors


def main(dataset_dir: Path, scene_file: Path, output_dir: Path):
    if 'nclt' in dataset_dir.name:
        Dataset = NCLT
        date_idx = 1
    elif 'robotcar' in dataset_dir.name:
        Dataset = OxfordRobotcar
        date_idx = 0
    else:
        raise ValueError("Unknown dataset")

    image_subsample = 2
    feature_generator = ImageFeatureGenerator('dinov2', use_featup=False)

    with open(scene_file, "r") as f:
        scene_data = json.load(f)

    #  Restore sequence names
    sequences = []
    sequences.append(scene_data["mapping"]["point_clouds"][date_idx].split("/")[1])
    for seq in scene_data["registration"]:
        sequences.append(seq["point_cloud"].split("/")[date_idx])

    # Load map data
    map_sequence = Dataset(sequences[0], dataset_dir, high_level_api=True)
    map_point_clouds = []
    pbar = tqdm(total=len(scene_data["mapping"]["point_clouds"]), desc="Loading map")
    for i, pcl_file in enumerate(scene_data["mapping"]["point_clouds"]):
        pcl_file = dataset_dir / pcl_file
        pcl = map_sequence.read_pcl(filename=pcl_file)
        pcl = voxel_down_sample(pcl, 0.2).astype(pcl.dtype)

        image_files = [dataset_dir / file for file in scene_data["mapping"]["images"][i]]
        pcl_descriptors = create_descriptors(image_files, map_sequence, feature_generator, pcl)
        pcl = np.c_[pcl, pcl_descriptors]

        map_point_clouds.append(pcl)
        pbar.update(1)
    pbar.close()
    map_poses = [np.array(pose) for pose in scene_data["mapping"]["poses"]]

    # Load scan data
    seq_point_clouds = []
    seq_poses = []
    pbar = tqdm(total=len(scene_data["registration"]), desc="Loading scans")
    for i, registration in enumerate(scene_data["registration"]):
        registration_sequence = Dataset(sequences[i + 1], dataset_dir, high_level_api=True)
        pcl_file = Path(dataset_dir) / registration["point_cloud"]
        pcl = registration_sequence.read_pcl(filename=pcl_file)
        pcl = voxel_down_sample(pcl, 0.1).astype(pcl.dtype)

        image_files = [dataset_dir / file for file in registration["images"]]
        pcl_descriptors = create_descriptors(image_files, registration_sequence, feature_generator,
                                             pcl)
        pcl = np.c_[pcl, pcl_descriptors]

        seq_point_clouds.append(pcl)
        seq_poses.append(np.array(registration["pose"]))
        pbar.update(1)
    pbar.close()

    output_filename = output_dir / scene_file.name.replace(".json", ".h5")
    save_scene(output_filename, sequences, map_poses, map_point_clouds, seq_poses, seq_point_clouds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_folder", type=str)
    parser.add_argument("scene_folder", type=str)
    parser.add_argument("--output_folder", type=str, required=False, default=None)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_folder)
    scene_dir = Path(args.scene_folder)
    if args.output_folder is not None:
        output_dir = Path(args.output_folder)
    else:
        if args.scene_folder[-5:] == ".json":
            output_dir = scene_dir.parent / "processed_scenes"
        else:
            output_dir = scene_dir / "processed_scenes"

    if args.scene_folder[-5:] == ".json":
        scene_files = [scene_dir]
    else:
        scene_files = sorted(list(scene_dir.glob("*.json")))

    print(f"Found {len(scene_files)} scene file(s).")

    for scene_file in scene_files:
        main(dataset_dir, scene_file, output_dir)
