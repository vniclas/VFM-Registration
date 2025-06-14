// MIT License
//
// Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
// Stachniss.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
// NOTE: This implementation is heavily inspired in the original CT-ICP VoxelHashMap implementation,
// although it was heavily modifed and drastically simplified, but if you are using this module you
// should at least acknoowledge the work from CT-ICP by giving a star on GitHub
#pragma once

#include <tsl/robin_map.h>

#include <Eigen/Core>
#include <sophus/se3.hpp>
#include <vector>

#include "DescriptorSize.hpp"

namespace kiss_icp {
struct VoxelHashMap {
    using Vector3dVector = std::vector<Eigen::Vector3d>;
    using VectorNdVector = std::vector<VectorNd>;
    using VectorXdVector = std::vector<Eigen::VectorXd>;
    using Vector3dVectorTuple = std::tuple<Vector3dVector, Vector3dVector>;
    using VectorNdVectorTuple = std::tuple<VectorNdVector, Vector3dVector>;
    using VectorXdVectorTuple = std::tuple<VectorXdVector, VectorXdVector>;
    using Voxel = Eigen::Vector3i;

    struct VoxelBlock {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::Vector3d> points;
        int num_points_;
        inline void AddPoint(const Eigen::Vector3d &point) {
            if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
        }
    };
    struct VoxelBlockN{
        // buffer of points with a max limit of n_points
        std::vector<VectorNd> points;
        int num_points_;
        inline void AddPoint(const VectorNd &point) {
            if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
        }
    };
    struct VoxelBlockX {
        // buffer of points with a max limit of n_points
        std::vector<Eigen::VectorXd> points;
        int num_points_;
        inline void AddPoint(const Eigen::VectorXd &point) {
            if (points.size() < static_cast<size_t>(num_points_)) points.push_back(point);
        }
    };

    struct VoxelHash {
        size_t operator()(const Voxel &voxel) const {
            const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
            return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349669 ^ vec[2] * 83492791);
        }
    };

    explicit VoxelHashMap(double voxel_size, double max_distance, int max_points_per_voxel)
        : voxel_size_(voxel_size),
          max_distance_(max_distance),
          max_points_per_voxel_(max_points_per_voxel) {}

    Vector3dVectorTuple GetCorrespondences(const Vector3dVector &points,
                                           double max_correspondance_distance) const;
    VectorNdVectorTuple GetCorrespondences(const VectorNdVector &points,
                                           double max_correspondance_distance) const;
    VectorXdVectorTuple GetCorrespondences(const VectorXdVector &points,
                                           double max_correspondance_distance) const;

    Vector3dVectorTuple GetVFMCorrespondences(const VectorNdVector &points,
                                              double min_cosine_similarity) const;

    inline void Clear() { map_.clear(); map_n_.clear(); map_x_.clear(); }
    inline bool Empty() const { return map_.empty(); }
    inline bool EmptyN() const { return map_n_.empty(); }
    inline bool EmptyX() const { return map_x_.empty(); }
    void Update(const Vector3dVector &points, const Eigen::Vector3d &origin);
    void Update(const VectorNdVector &points, const Eigen::Vector3d &origin);
    void Update(const VectorXdVector &points, const Eigen::Vector3d &origin);

    void Update(const Vector3dVector &points, const Sophus::SE3d &pose);
    void Update(const VectorNdVector &points, const Sophus::SE3d &pose);
    void Update(const VectorXdVector &points, const Sophus::SE3d &pose);

    void AddPoints(const Vector3dVector &points);
    void AddPoints(const VectorNdVector &points);
    void AddPoints(const VectorXdVector &points);

    void RemovePointsFarFromLocation(const Eigen::Vector3d &origin);
    Vector3dVector Pointcloud() const;
    VectorNdVector PointcloudN() const;

    double voxel_size_;
    double max_distance_;
    int max_points_per_voxel_;
    tsl::robin_map<Voxel, VoxelBlock, VoxelHash> map_;
    tsl::robin_map<Voxel, VoxelBlockN, VoxelHash> map_n_;
    tsl::robin_map<Voxel, VoxelBlockX, VoxelHash> map_x_;
};
}  // namespace kiss_icp
