import h5py

from vfm_reg.utils import *

# File structure:
#   /map
#     /sequence name
#       /pose
#       /point_cloud
#       /clip
#   /scans
#     /sequence name
#       /pose
#       /point_cloud


def read_scenes(filename):
    file = h5py.File(filename, "r")

    map_poses, map_point_clouds, map_clip = [], [], []
    map_group = file["map"]
    for key in map_group.keys():
        # There should be only one key corresponding to the sequence name
        map_group = file["map"][key]
        for pose, point_cloud in zip(map_group["pose"].values(), map_group["point_cloud"].values()):
            map_poses.append(pose[()])
            map_point_clouds.append(point_cloud[()])
        if "clip" in map_group.keys():
            for clip in map_group["clip"].values():
                map_clip.append(clip[()])

    scene_poses, scene_point_clouds = [], []
    scans_group = file["scans"]
    for scan in scans_group:
        scan_group = scans_group[scan]
        scene_poses.append(scan_group["pose"][()])
        scene_point_clouds.append(scan_group["point_cloud"][()])

    file.close()

    scene = {
        "map_poses": map_poses,
        "map_point_clouds": map_point_clouds,
        "map_clip": map_clip,
        "scene_poses": scene_poses,
        "scene_point_clouds": scene_point_clouds,
    }

    return scene
