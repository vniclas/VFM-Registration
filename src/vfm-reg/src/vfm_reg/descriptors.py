from typing import Tuple

import MinkowskiEngine as ME
import numpy as np
import open3d as o3d
import pointnet2_ops.pointnet2_utils as pnt2
import torch
import torch.nn as nn
from dip.lrf import lrf
from dip.network import PointNetFeature
from fcgf.fcgf import ResUNetBN2C as FCGF
from gcl.model import load_model
from gedi.gedi import GeDi
from numpy.typing import ArrayLike
from spinnet.model import Descriptor_Net
from tqdm import trange


def extract_fpfh_features(pcl: ArrayLike,
                          voxel_size: float,
                          normalize: bool = False) -> Tuple[ArrayLike, ArrayLike]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])

    # Compute normals
    radius_normal = voxel_size * 2
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    # Voxelize
    pcd = pcd.voxel_down_sample(voxel_size)

    # Compute FPFH features
    radius_feature = voxel_size * 5
    features = o3d.pipelines.registration.compute_fpfh_feature(
        pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    features = np.array(features.data).T

    if normalize:
        features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

    return np.array(pcd.points), features


def extract_dip_features(pcl: ArrayLike, voxel_size: float, weight_path):
    dim = 32
    patch_size = 256
    batch_size = 500

    net = PointNetFeature(dim=dim, l2norm=True, tnet=True)
    net.load_state_dict(torch.load(weight_path))
    net.cuda()
    net.eval()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size)
    pcd_pts = np.asarray(pcd.points)

    frag_lrf = lrf(pcd=pcd,
                   pcd_tree=o3d.geometry.KDTreeFlann(pcd),
                   patch_size=patch_size,
                   lrf_kernel=.6 * np.sqrt(3),
                   viz=False)
    patches = np.empty((pcd_pts.shape[0], 3, patch_size))
    for i in trange(pcd_pts.shape[0], desc="[DIP] Computing LRF"):
        frag_lrf_pts, _, _ = frag_lrf.get(pcd_pts[i])
        patches[i] = frag_lrf_pts.T

    pcd_desc = np.empty((patches.shape[0], dim))
    for b in trange(int(np.ceil(patches.shape[0] / batch_size)),
                    desc="[DIP] Computing descriptors"):
        i_start = b * batch_size
        i_end = (b + 1) * batch_size
        if i_end > patches.shape[0]:
            i_end = patches.shape[0]

        pcd1_batch = torch.Tensor(patches[i_start:i_end]).cuda()
        with torch.no_grad():
            f, mx1, amx1 = net(pcd1_batch)
        pcd_desc[i_start:i_end] = f.cpu().detach().numpy()[:i_end - i_start]

    return pcd_pts, pcd_desc


def extract_fcgf_features(pcl: ArrayLike, voxel_size: float, weight_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fcgf_model = FCGF(1, 32, bn_momentum=0.05, conv1_kernel_size=7,
                      normalize_feature=True).to(device)
    checkpoint = torch.load(weight_path)
    fcgf_model.load_state_dict(checkpoint['state_dict'])
    fcgf_model.eval()

    # Adapted from PointDSC/misc/cal_fcgf.py =======
    feats = torch.ones((len(pcl), 1), dtype=torch.float32)

    # Voxelize xyz and feats
    coords = np.floor(pcl / voxel_size)
    coords, inds = ME.utils.sparse_quantize(coords, return_index=True)
    # Convert to batched coords compatible with ME
    coords = ME.utils.batched_coordinates([coords])
    return_coords = pcl[inds]

    feats = feats[inds]
    coords = coords.clone().detach().int()
    stensor = ME.SparseTensor(feats, coordinates=coords, device=device)

    fcgf_feats = fcgf_model(stensor).F
    # ==============================================

    pcl_voxelized = return_coords.astype(np.float32)
    fcgf_feats = fcgf_feats.detach().cpu().numpy()

    return pcl_voxelized, fcgf_feats


def extract_gedi_features(pcl: ArrayLike, voxel_size, n_points, weight_path):
    # Adapted from gedi/demo.py ====================

    config = {
        'dim': 32,  # descriptor output dimension
        'samples_per_batch': 250,  # batches to process the data on GPU
        'samples_per_patch_lrf': 4000,  # num. of point to process with LRF
        'samples_per_patch_out': 512,  # num. of points to sample for pointnet++
        'r_lrf': .5,  # LRF radius
        'fchkpt_gedi_net': weight_path  # path to checkpoint
    }
    gedi = GeDi(config)

    if pcl.shape[0] > n_points:
        indices = np.random.choice(pcl.shape[0], n_points, replace=False)
    else:
        indices = np.arange(pcl.shape[0])

    pts = torch.Tensor(pcl[indices, :3]).float()

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcl[:, :3])
    pcd = pcd.voxel_down_sample(voxel_size)
    _pcd = torch.tensor(np.asarray(pcd.points)).float()

    gedi_feats = gedi.compute(pts=pts, pcd=_pcd)

    del gedi
    return pcl[indices, :3], gedi_feats


def extract_gcl_features(pcl: ArrayLike, voxel_size, weight_path):
    # Adapted from GCL/scripts/test_kitti.py =========

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Model = load_model('ResUNetFatBN')
    gcl_model = Model(
        1,
        32,
        bn_momentum=0.05,
        conv1_kernel_size=5,
        normalize_feature=True,
    ).to(device)
    checkpoint = torch.load(weight_path)
    gcl_model.load_state_dict(checkpoint['state_dict'])
    gcl_model.eval()

    # Voxelize
    pcl_torch = torch.tensor(pcl)
    _, inds = ME.utils.sparse_quantize(pcl_torch / voxel_size, return_index=True)
    pcl_torch_th = pcl_torch[inds]
    coords = torch.floor(pcl_torch_th / voxel_size).int()
    feats = torch.ones((len(pcl_torch_th), 1)).float()
    coords_batch, feats_batch = ME.utils.sparse_collate([coords], [feats])

    sinput = ME.SparseTensor(feats_batch.to(device), coordinates=coords_batch.to(device))
    enc = gcl_model(sinput)
    F = enc.F.detach()

    return pcl_torch_th.numpy(), F.cpu().numpy()


def extract_spinnet_features(pcl: ArrayLike, n_points, weight_path):
    # Adapted from SpinNet/KITTI/Test/test_kitti.py ========

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vicinity = 2.0
    model = Descriptor_Net(vicinity, 9, 60, 30, 0.3, 30, 'KITTI')
    model = nn.DataParallel(model, device_ids=[0])
    model = model.to(device)
    model.load_state_dict(torch.load(weight_path))
    model.eval()

    if pcl.shape[0] > n_points:
        indices = np.random.choice(pcl.shape[0], n_points, replace=False)
        keypoints = pcl[indices]
    else:
        keypoints = pcl

    def select_patches(pts, refer_pts, vicinity, num_points_per_patch=1024):
        pts = torch.FloatTensor(pts).cuda().unsqueeze(0)
        refer_pts = torch.FloatTensor(refer_pts).cuda().unsqueeze(0)
        group_idx = pnt2.ball_query(vicinity, num_points_per_patch, pts, refer_pts)
        pts_trans = pts.transpose(1, 2).contiguous()
        new_points = pnt2.grouping_operation(pts_trans, group_idx)
        new_points = new_points.permute([0, 2, 3, 1])
        mask = group_idx[:, :, 0].unsqueeze(2).repeat(1, 1, num_points_per_patch)
        mask = (group_idx == mask).float()
        mask[:, :, 0] = 0
        mask[:, :, num_points_per_patch - 1] = 1
        mask = mask.unsqueeze(3).repeat([1, 1, 1, 3])
        new_pts = refer_pts.unsqueeze(2).repeat([1, 1, num_points_per_patch, 1])
        local_patches = new_points * (1 - mask).float() + new_pts * mask.float()
        local_patches = local_patches.squeeze(0)
        return local_patches

    num_points_per_patch = 2048
    local_patches = select_patches(pcl,
                                   keypoints,
                                   vicinity=vicinity,
                                   num_points_per_patch=num_points_per_patch)
    B = local_patches.shape[0]
    step_size = 100  # according to GPU memory
    desc_len = 32
    iter_num = int(np.ceil(B / step_size))
    desc_list = []
    for k in trange(iter_num, desc="[SpinNet] Computing descriptors"):
        if k == iter_num - 1:
            desc = model(local_patches[k * step_size:, :, :])
        else:
            desc = model(local_patches[k * step_size:(k + 1) * step_size, :, :])
        desc_list.append(desc.view(desc.shape[0], desc_len).detach().cpu().numpy())
        del desc
    desc = np.concatenate(desc_list, 0).reshape([B, desc_len])

    return keypoints, desc
