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
#include "VoxelHashMap.hpp"
#include "DescriptorSize.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>

#include <numeric>

#include <Eigen/Core>
#include <algorithm>
#include <limits>
#include <tuple>
#include <utility>
#include <vector>

#include <iostream>  // Remove this line
#include <chrono> // Remove this line

// This parameters are not intended to be changed, therefore we do not expose it
namespace {
struct ResultTuple {
    ResultTuple(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::Vector3d> source;
    std::vector<Eigen::Vector3d> target;
};

struct ResultTupleN {
    ResultTupleN(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<VectorNd> source;
    std::vector<Eigen::Vector3d> target;
};

struct ResultTupleX {
    ResultTupleX(std::size_t n) {
        source.reserve(n);
        target.reserve(n);
    }
    std::vector<Eigen::VectorXd> source;
    std::vector<Eigen::VectorXd> target;
};
}  // namespace

namespace kiss_icp {

VoxelHashMap::Vector3dVectorTuple VoxelHashMap::GetCorrespondences(
    const Vector3dVector &points, double max_correspondance_distance) const {
    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighbor = [&](const Eigen::Vector3d &point) {
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
        voxels.reserve(27);
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        using Vector3dVector = std::vector<Eigen::Vector3d>;
        Vector3dVector neighbors;
        neighbors.reserve(27 * max_points_per_voxel_);
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            // Vanilla KISS-iCP
            if (!map_.empty()) {
                auto search = map_.find(voxel);
                if (search != map_.end()) {
                    const auto &points = search->second.points;
                    if (!points.empty()) {
                        for (const auto &point : points) {
                            neighbors.emplace_back(point);
                        }
                    }
                }
            }
            // Modified version, with descriptors attached to points
            else if (!map_n_.empty()) {
                auto search = map_n_.find(voxel);
                if (search != map_n_.end()) {
                    const auto &points = search->second.points;
                    if (!points.empty()) {
                        for (const auto &point : points) {
                            neighbors.emplace_back(point.template head<3>());
                        }
                    }
                }
            }
        });

        Eigen::Vector3d closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        std::for_each(neighbors.cbegin(), neighbors.cend(), [&](const auto &neighbor) {
            double distance = (neighbor - point).squaredNorm();
            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            }
        });

        return closest_neighbor;
    };
    using points_iterator = std::vector<Eigen::Vector3d>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        // Identity
        ResultTuple(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, &GetClosestNeighbor](
            const tbb::blocked_range<points_iterator> &r, ResultTuple res) -> ResultTuple {
            auto &[src, tgt] = res;
            src.reserve(r.size());
            tgt.reserve(r.size());
            for (const auto &point : r) {
                Eigen::Vector3d closest_neighbors = GetClosestNeighbor(point);
                if ((closest_neighbors - point).norm() < max_correspondance_distance) {
                    src.emplace_back(point);
                    tgt.emplace_back(closest_neighbors);
                }
            }
            return res;
        },
        // 2nd lambda: Parallel reduction
        [](ResultTuple a, const ResultTuple &b) -> ResultTuple {
            auto &[src, tgt] = a;
            const auto &[srcp, tgtp] = b;
            src.insert(src.end(),  //
                       std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),  //
                       std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
}

constexpr double COSINE_THRESHOLD_ = 0.; // 0.90;

VoxelHashMap::VectorNdVectorTuple VoxelHashMap::GetCorrespondences(
    const VectorNdVector &points, double max_correspondance_distance) const {

    const int n_features = static_cast<int>(points.front().rows()) - 3;
    int call_i = 0;

    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighbor = [&](const VectorNd &point) {
        bool point_has_descriptors = point.template tail(n_features).sum() != 0;

        // Determines the voxel where the point is located -> int casting is used to floor the value
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
        voxels.reserve(27);
        // Iterate over the 3x3x3 neighborhood
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        // Determine all points of the same and neighboring voxels
        int n_suppressed_points = 0;
        Vector3dVector neighbors;
        neighbors.reserve(27 * max_points_per_voxel_);
        // Iterate over the neighboring voxels
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            auto search = map_n_.find(voxel);
            if (search != map_n_.end()) {
                // Access all points from this voxel
                const auto &points = search->second.points;  // from VoxelBlockN
                if (!points.empty()) {
                    for (const auto &voxel_point : points) {
                        // "Copy" all points from a voxel to the neighbors vector
                        neighbors.emplace_back(voxel_point.template head<3>());

                        // Copy points only if descriptors are sufficiently similar
                        // double cosine_similarity = 2.0;
                        // if (point_has_descriptors && voxel_point.template tail(n_features).sum() != 0) {
                        //     cosine_similarity = voxel_point.template tail(n_features).dot(point.template tail(n_features)) / (voxel_point.template tail(n_features).norm() * point.template tail(n_features).norm() + 1e-5);
                        //     cosine_similarity = .5 * (cosine_similarity + 1); // mapped from [-1, 1] to [0, 1]
                        // }
                        // // std::cout << cosine_similarity << std::endl;

                        // if (cosine_similarity > COSINE_THRESHOLD_) {
                        //     Eigen::Vector3d voxel_point_3d = voxel_point.template head<3>();
                        //     neighbors.emplace_back(voxel_point_3d);
                        // }
                        // else {
                        //     n_suppressed_points++;
                        // }
                    }
                }
            }
        });
        // std::cout << " < " << n_suppressed_points << " | " << neighbors.size() << " > " << std::endl;

        double min_cos_dist = std::numeric_limits<double>::max();
        double max_cos_dist = std::numeric_limits<double>::min();

        // Find the single (!) neighbor with the smallest distance
        // VectorNd closest_neighbor;
        Eigen::Vector3d closest_neighbor_3d;
        double closest_distance2 = std::numeric_limits<double>::max();
        double neighbor_cos_dist = -1.0;
        std::for_each(neighbors.cbegin(), neighbors.cend(), [&](const auto &neighbor) {

            double distance = (neighbor - point.template head<3>()).squaredNorm();
            // VFM-ICP
            // double cosine_distance = -1.0;
            // if (n_features > 0 && false) { // DISABLED
            //     // If either point or neighbor has no features, cosine_distance is 1.0
            //     if (point_has_descriptors && neighbor.template tail(n_features).sum() != 0) {
            //         double cosine_similarity = neighbor.template tail(n_features).dot(point.template tail(n_features)) / (neighbor.template tail(n_features).norm() * point.template tail(n_features).norm() + 1e-5);
            //         cosine_distance = .5 * (1 - cosine_similarity); // mapped from [0, 2] to [0, 1]

            //         // cosine_distance = std::clamp(cosine_distance, 0.01, 1.0); // avoid distance := 0.0

            //         if (cosine_distance < .1)
            //             distance *= .8;

            //         // if (cosine_distance < min_cos_dist) {
            //         //     min_cos_dist = cosine_distance;
            //         // }
            //         // if (cosine_distance > max_cos_dist) {
            //         //     max_cos_dist = cosine_distance;
            //         // }
            //     }
            //     // distance *= cosine_distance;

            // }

            if (distance < closest_distance2) {
                closest_neighbor_3d = neighbor;
                closest_distance2 = distance;
                // neighbor_cos_dist = cosine_distance;
            }
        });
        // std::cout << closest_distance2 << std::endl;
        // std::cout << "———————————————————————————————————";
        // if (call_i++ == 0)
            // std::cout << min_cos_dist << " | " << max_cos_dist << " | " << closest_distance2 << " | " << neighbor_cos_dist << " | " << std::distance(neighbors.cbegin(), neighbors.cend()) << std::endl;

        return closest_neighbor_3d;
    };
    using points_iterator = std::vector<VectorNd>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        // Identity
        ResultTupleN(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, &GetClosestNeighbor, n_features](
            const tbb::blocked_range<points_iterator> &r, ResultTupleN res) -> ResultTupleN {
            auto &[src, tgt] = res;
            src.reserve(r.size());
            tgt.reserve(r.size());
            for (const auto &point : r) {
                Eigen::Vector3d closest_neighbor = GetClosestNeighbor(point);
                if (closest_neighbor.size() == 0) {
                    continue;
                }
                // Keep point association (nearest neighbor) if Euclidean distance is below threshold
                double euclidean_distance = (closest_neighbor - point.head<3>()).norm();
                if (euclidean_distance < max_correspondance_distance) {
                    src.emplace_back(point);
                    tgt.emplace_back(closest_neighbor);
                }
            }
            return res;
        },
        // 2nd lambda: Parallel reduction
        [](ResultTupleN a, const ResultTupleN &b) -> ResultTupleN {
            auto &[src, tgt] = a;
            const auto &[srcp, tgtp] = b;
            src.insert(src.end(),  //
                       std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),  //
                       std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
}

VoxelHashMap::VectorXdVectorTuple VoxelHashMap::GetCorrespondences(
    const VectorXdVector &points, double max_correspondance_distance) const {

    const int n_features = static_cast<int>(points.front().rows()) - 3;
    // int call_i = 0;

    // Lambda Function to obtain the KNN of one point, maybe refactor
    auto GetClosestNeighbor = [&](const Eigen::VectorXd &point) {

        // Determines the voxel where the point is located -> int casting is used to floor the value
        auto kx = static_cast<int>(point[0] / voxel_size_);
        auto ky = static_cast<int>(point[1] / voxel_size_);
        auto kz = static_cast<int>(point[2] / voxel_size_);
        std::vector<Voxel> voxels;
        voxels.reserve(27);
        // Iterate over the 3x3x3 neighborhood
        for (int i = kx - 1; i < kx + 1 + 1; ++i) {
            for (int j = ky - 1; j < ky + 1 + 1; ++j) {
                for (int k = kz - 1; k < kz + 1 + 1; ++k) {
                    voxels.emplace_back(i, j, k);
                }
            }
        }

        // Determine all points of the same and neighboring voxels
        VectorXdVector neighbors;
        neighbors.reserve(27 * max_points_per_voxel_);
        std::for_each(voxels.cbegin(), voxels.cend(), [&](const auto &voxel) {
            auto search = map_x_.find(voxel);
            if (search != map_x_.end()) {
                const auto &points = search->second.points;  // from VoxelBlockX
                if (!points.empty()) {
                    for (const auto &point : points) {
                        // "Copy" all points from a voxel to the neighbors vector
                        neighbors.emplace_back(point);
                    }
                }
            }
        });

        double min_cos_dist = std::numeric_limits<double>::max();
        double max_cost_dist = std::numeric_limits<double>::min();

        // Find the single (!) neighbor with the smallest distance
        Eigen::VectorXd closest_neighbor;
        double closest_distance2 = std::numeric_limits<double>::max();
        bool point_has_descriptors = point.template tail(n_features).sum() != 0;
        std::for_each(neighbors.cbegin(), neighbors.cend(), [&](const auto &neighbor) {

            double distance = (neighbor.template head<3>() - point.template head<3>()).squaredNorm();
            // VFM-ICP
            if (n_features > 0) {
                double cosine_distance = 1.0;
                // If either point or neighbor has no features, cosine_distance is 1.0
                if (point_has_descriptors && neighbor.template tail(n_features).sum() != 0) {
                    double cosine_similarity = neighbor.template tail(n_features).dot(point.template tail(n_features)) / (neighbor.template tail(n_features).norm() * point.template tail(n_features).norm() + 1e-5);
                    // double cosine_distance = 1.0 - cosine_similarity;
                    cosine_distance = .5 * (1 - cosine_similarity); // mapped from [0, 2] to [0, 1]
                    // if (distance < max_correspondance_distance)
                    //     std::cout << distance << " | " << cosine_distance << std::endl;
                    // cosine_distance *= 10;
                    cosine_distance = std::clamp(cosine_distance, 0.01, 1.0); // avoid distance := 0.0
                }
                distance *= cosine_distance;

                    // if (cosine_distance < min_cos_dist) {
                    //     min_cos_dist = cosine_distance;
                    // }
                    // if (cosine_distance > max_cost_dist) {
                    //     max_cost_dist = cosine_distance;
                    // }
            }

            if (distance < closest_distance2) {
                closest_neighbor = neighbor;
                closest_distance2 = distance;
            }
        });
        // std::cout << "———————————————————————————————————";
        // if (call_i++ == 0)
        //     std::cout << min_cos_dist << " | " << max_cost_dist << " | " << closest_distance2 << " | " << std::distance(neighbors.cbegin(), neighbors.cend()) << std::endl;

        return closest_neighbor;
    };
    using points_iterator = std::vector<Eigen::VectorXd>::const_iterator;
    const auto [source, target] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<points_iterator>{points.cbegin(), points.cend()},
        // Identity
        ResultTupleX(points.size()),
        // 1st lambda: Parallel computation
        [max_correspondance_distance, &GetClosestNeighbor, n_features](
            const tbb::blocked_range<points_iterator> &r, ResultTupleX res) -> ResultTupleX {
            auto &[src, tgt] = res;
            src.reserve(r.size());
            tgt.reserve(r.size());
            // std::cout << r.size() << std::endl;
            for (const auto &point : r) {
                Eigen::VectorXd closest_neighbor = GetClosestNeighbor(point);
                if (closest_neighbor.size() == 0) {
                    continue;
                }
                // Keep point association (neares neighbor) if Euclidean distance is below threshold
                double euclidean_distance = (closest_neighbor.head<3>() - point.head<3>()).norm();
                // double cosine_similarity = 1.0;
                // if (n_features > 0) {
                //     cosine_similarity = closest_neighbor.tail(n_features).dot(point.tail(n_features)) / (closest_neighbor.tail(n_features).norm() * point.tail(n_features).norm() + 1e-5);
                // }
                // std::cout << euclidean_distance << " | " << cosine_similarity << std::endl;
                // VFM-ICP
                // if (euclidean_distance < max_correspondance_distance && cosine_similarity > 0.9) {
                if (euclidean_distance < max_correspondance_distance) {
                    src.emplace_back(point);
                    tgt.emplace_back(closest_neighbor);
                }
            }
            return res;
        },
        // 2nd lambda: Parallel reduction
        [](ResultTupleX a, const ResultTupleX &b) -> ResultTupleX {
            auto &[src, tgt] = a;
            const auto &[srcp, tgtp] = b;
            src.insert(src.end(),  //
                       std::make_move_iterator(srcp.begin()), std::make_move_iterator(srcp.end()));
            tgt.insert(tgt.end(),  //
                       std::make_move_iterator(tgtp.begin()), std::make_move_iterator(tgtp.end()));
            return a;
        });

    return std::make_tuple(source, target);
}

float l2_norm(std::vector<float> const& u) {
    float accum = 0.;
    for (int i = 0; i < static_cast<int>(u.size()); ++i) {
        accum += u[i] * u[i];
    }
    return static_cast<float>(sqrt(accum));
}

VoxelHashMap::Vector3dVectorTuple VoxelHashMap::GetVFMCorrespondences(
    const VectorNdVector &points, double min_cosine_similarity) const {

    int n_features = static_cast<int>(points.front().rows()) - 3;
    VectorNdVector map_points = PointcloudN();

    // Prepare data for faiss
    auto a = std::chrono::steady_clock::now();
    float* xb = new float[map_points.size() * n_features]; // database points
    float* xq = new float[points.size() * n_features]; // query points
    for (size_t i = 0; i < map_points.size(); i++) {
        Vectornd descriptors_d = map_points[i].template tail(n_features); // w/o 3D coordinates
        std::vector<float> descriptors(descriptors_d.data(), descriptors_d.data() + descriptors_d.size());
        faiss::fvec_renorm_L2(n_features, 1, descriptors.data());
        std::copy(descriptors.begin(), descriptors.end(), xb + i * n_features);
    }
    for (size_t i = 0; i < points.size(); i++) {
        Vectornd descriptors_d = points[i].template tail(n_features); // w/o 3D coordinates
        std::vector<float> descriptors(descriptors_d.data(), descriptors_d.data() + descriptors_d.size());
        faiss::fvec_renorm_L2(n_features, 1, descriptors.data());
        std::copy(descriptors.begin(), descriptors.end(), xq + i * n_features);
    }
    auto b = std::chrono::steady_clock::now();

    // Build faiss index
    faiss::IndexFlatIP faiss_index(n_features);
    faiss_index.add(map_points.size(), xb);
    auto c = std::chrono::steady_clock::now();

    // Search for the correspondences
    int k = 1; // only nearest neighbor
    faiss::idx_t* I = new faiss::idx_t[k * points.size()];
    float* D = new float[k * points.size()];
    bool* valid = new bool[k * points.size()];
    faiss_index.search(points.size(), xq, k, D, I);
    auto d = std::chrono::steady_clock::now();

    std::vector<double> euclidean_distances;
    std::vector<double> euclidean_distances_median;
    euclidean_distances.reserve(points.size());
    for (size_t i = 0; i < points.size(); i++) {
        for (int j = 0; j < k; j++) {
            if (D[i * k + j] < min_cosine_similarity) {
                valid[i * k + j] = false;
            }
            else {
               valid[i * k + j] = true;
                euclidean_distances.emplace_back((points[i].head<3>() - map_points[I[i * k + j]].head<3>()).norm());
                euclidean_distances_median.emplace_back((points[i].head<3>() - map_points[I[i * k + j]].head<3>()).norm());
            }
        }
        // if (D[i] < min_cosine_similarity) {
        //     valid[i] = false;
        // }
        // else {
        //     valid[i] = true;
        //     euclidean_distances.emplace_back((points[i].head<3>() - map_points[I[i]].head<3>()).norm());
        //     euclidean_distances_median.emplace_back((points[i].head<3>() - map_points[I[i]].head<3>()).norm());
        // }
    }

    // Filter out outliers based on euclidean distance
    // double mean = std::accumulate(euclidean_distances.begin(), euclidean_distances.end(), 0.0) / static_cast<double>(euclidean_distances.size());
    // double accum = 0.0;
    // std::for_each (euclidean_distances.begin(), euclidean_distances.end(), [&](const double d) {
    //     accum += (d - mean) * (d - mean);
    // });
    // double stddev = sqrt(accum / static_cast<double>(euclidean_distances.size()));
    // int n_outliers = 0;
    // auto it = euclidean_distances.begin();
    // for (size_t i = 0; i < points.size(); i++) {
    //     if (!valid[i]) {
    //         continue;
    //     }
    //     if (std::abs(*it - mean) > 1.0 * stddev) {
    //         valid[i] = false;
    //         n_outliers++;
    //     }
    //     it++;
    // }
    // std::cout << "Eucl. dist | Mean: " << mean << "\t Stddev: " << stddev << std::endl;
    // std::cout << "Max dist: " << *std::max_element(euclidean_distances.begin(), euclidean_distances.end()) << std::endl;

    // https://www.sciencedirect.com/science/article/pii/S0022103113000668

    // Compute median
    size_t n = euclidean_distances_median.size() / 2;
    // afterwards, nth element is the median
    std::nth_element(euclidean_distances_median.begin(), euclidean_distances_median.begin() + n, euclidean_distances_median.end());
    double median = euclidean_distances_median[n];
    // If the vector is even
    if (!(euclidean_distances_median.size() & 1)) {
        double max_value = *std::max_element(euclidean_distances_median.begin(), euclidean_distances_median.begin() + n);
        median = (median + max_value) / 2;
    }
    // std::cout << "Median: " << median << std::endl;
    // Compute median absolute deviation (MAD)
    std::vector<double> abs_diff(euclidean_distances.size());
    std::transform(euclidean_distances.begin(), euclidean_distances.end(), abs_diff.begin(), [&](const double d) {
        return std::abs(d - median);
    });
    n = abs_diff.size() / 2;
    std::nth_element(abs_diff.begin(), abs_diff.begin() + n, abs_diff.end());
    double mad = abs_diff[n];
    if (!(abs_diff.size() & 1)) {
        double max_value = *std::max_element(abs_diff.begin(), abs_diff.begin() + n);
        mad = (mad + max_value) / 2;
    }
    mad *= 1.4826;
    // std::cout << "MAD: " << mad << std::endl;
    int n_outliers = 0;
    auto it = euclidean_distances.begin();
    for (size_t i = 0; i < points.size(); i++) {
        for (int j = 0; j < k; j++) {
            if (!valid[i * k + j]) {
                continue;
            }
            if (std::abs(*it - median) > .75 * mad) {
                // valid[i * k + j] = false;
                // n_outliers++;
            }
            it++;
        }
    }

    // Create correspondences pair
    ResultTuple correspondences(points.size()); // ToDo: figure out initial size
    // it = euclidean_distances_filtered.begin();
    double mean_similarity = 0.0;
    for (size_t i = 0; i < points.size(); i++) {
        for (int j = 0; j < k; j++) {
            if (valid[i]) {
                correspondences.source.emplace_back(points[i].head<3>());
                correspondences.target.emplace_back(map_points[I[i * k + j]].head<3>());
                // std::cout << *it << "\t" << (points[i].head<3>() - map_points[I[i]].head<3>()).norm() << std::endl;
                // it++;
            }
            mean_similarity += D[i * k + j];
        }
    }
    // mean_similarity /= static_cast<double>(correspondences.source.size());
    mean_similarity /= static_cast<double>(points.size());
    // mean = std::accumulate(euclidean_distances_filtered.begin(), euclidean_distances_filtered.end(), 0.0) / static_cast<double>(euclidean_distances_filtered.size());
    // std::cout << "Filtered Eucl. dist | Mean: " << mean << std::endl;
    // std::cout << "Max dist: " << *std::max_element(euclidean_distances_filtered.begin(), euclidean_distances_filtered.end()) << std::endl;
    // std::cout << std::accumulate(valid, valid + points.size(), 0) << " valid" << std::endl;


    // std::cout << "Data preparation: " << std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count() << " ms\n"
    //           << "Index building:   " << std::chrono::duration_cast<std::chrono::milliseconds>(c - b).count() << " ms\n"
    //           << "Search:           " << std::chrono::duration_cast<std::chrono::milliseconds>(d - c).count() << " ms" << std::endl;

    std::cout << "Points: " << points.size()
              << " | Corrs.: " << correspondences.source.size()
              << " | Outliers: " << n_outliers
              << " | Mean sim.: " << mean_similarity << std::endl;

    // Clean up
    delete[] xb;
    delete[] xq;
    delete[] I;
    delete[] D;
    delete[] valid;

    return std::make_tuple(correspondences.source, correspondences.target);
}

std::vector<Eigen::Vector3d> VoxelHashMap::Pointcloud() const {
    std::vector<Eigen::Vector3d> points;

    if (map_.size() != 0) {
        points.reserve(max_points_per_voxel_ * map_.size());
        for (const auto &[voxel, voxel_block] : map_) {
            (void)voxel;
            for (const auto &point : voxel_block.points) {
                points.push_back(point);
            }
        }
    }
    else if (map_n_.size() != 0) {
        points.reserve(max_points_per_voxel_ * map_n_.size());
        for (const auto &[voxel, voxel_block] : map_n_) {
            (void)voxel;
            for (const auto &point : voxel_block.points) {
                points.push_back(point.template head<3>());
            }
        }
    }
    else if (map_x_.size() != 0) {
        points.reserve(max_points_per_voxel_ * map_x_.size());
        for (const auto &[voxel, voxel_block] : map_x_) {
            (void)voxel;
            for (const auto &point : voxel_block.points) {
                points.push_back(point.template head<3>());
            }
        }
    }

    return points;
}

std::vector<VectorNd> VoxelHashMap::PointcloudN() const {
    std::vector<VectorNd> points;

    if (map_n_.size() != 0) {
        points.reserve(max_points_per_voxel_ * map_n_.size());
        for (const auto &[voxel, voxel_block] : map_n_) {
            (void)voxel;
            for (const auto &point : voxel_block.points) {
                points.push_back(point);
            }
        }
    }

    return points;
}

void VoxelHashMap::Update(const Vector3dVector &points, const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const VectorNdVector &points, const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const VectorXdVector &points, const Eigen::Vector3d &origin) {
    AddPoints(points);
    RemovePointsFarFromLocation(origin);
}

void VoxelHashMap::Update(const Vector3dVector &points, const Sophus::SE3d &pose) {
    Vector3dVector points_transformed(points.size());
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) { return pose * point; });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::Update(const VectorNdVector &points, const Sophus::SE3d &pose) {
    VectorNdVector points_transformed(points.size());
    // Subtract x, y, z
    const long int n_features = points.front().rows() - 3;
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) {
                        VectorNd point_transformed;
                        // point_transformed.resize(3 + n_features);
                        point_transformed << pose * point.template head<3>(),
                                             point.template tail(n_features);
                        return point_transformed;
                    });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::Update(const VectorXdVector &points, const Sophus::SE3d &pose) {
    VectorXdVector points_transformed(points.size());
    // Subtract x, y, z
    const long int n_features = points.front().rows() - 3;
    std::transform(points.cbegin(), points.cend(), points_transformed.begin(),
                   [&](const auto &point) {
                        Eigen::VectorXd point_transformed;
                        point_transformed.resize(3 + n_features);
                        point_transformed << pose * point.template head<3>(),
                                             point.template tail(n_features);
                        return point_transformed;
                    });
    const Eigen::Vector3d &origin = pose.translation();
    Update(points_transformed, origin);
}

void VoxelHashMap::AddPoints(const Vector3dVector &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        auto voxel = Voxel((point / voxel_size_).template cast<int>());
        auto search = map_.find(voxel);
        if (search != map_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_.insert({voxel, VoxelBlock{{point}, max_points_per_voxel_}});
        }
    });
}

void VoxelHashMap::AddPoints(const VectorNdVector &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        auto voxel = Voxel((point.template head<3>() / voxel_size_).template cast<int>());
        auto search = map_n_.find(voxel);
        if (search != map_n_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_n_.insert({voxel, VoxelBlockN{{point}, max_points_per_voxel_}});
        }
    });
}

void VoxelHashMap::AddPoints(const VectorXdVector &points) {
    std::for_each(points.cbegin(), points.cend(), [&](const auto &point) {
        auto voxel = Voxel((point.template head<3>() / voxel_size_).template cast<int>());
        auto search = map_x_.find(voxel);
        if (search != map_x_.end()) {
            auto &voxel_block = search.value();
            voxel_block.AddPoint(point);
        } else {
            map_x_.insert({voxel, VoxelBlockX{{point}, max_points_per_voxel_}});
        }
    });
}

void VoxelHashMap::RemovePointsFarFromLocation(const Eigen::Vector3d &origin) {
    for (const auto &[voxel, voxel_block] : map_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt - origin).squaredNorm() > (max_distance2)) {
            map_.erase(voxel);
        }
    }

    for (const auto &[voxel, voxel_block] : map_n_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt.head<3>() - origin).squaredNorm() > (max_distance2)) {
            map_n_.erase(voxel);
        }
    }

    for (const auto &[voxel, voxel_block] : map_x_) {
        const auto &pt = voxel_block.points.front();
        const auto max_distance2 = max_distance_ * max_distance_;
        if ((pt.head<3>() - origin).squaredNorm() > (max_distance2)) {
            map_x_.erase(voxel);
        }
    }
}
}  // namespace kiss_icp
