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
#include "Registration.hpp"

#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>

#include <algorithm>
#include <chrono>  // Remove this line
#include <cmath>
#include <iostream>  // Remove this line
#include <sophus/se3.hpp>
#include <sophus/so3.hpp>
#include <tuple>

#include "DescriptorSize.hpp"
#include "Preprocessing.hpp"

namespace Eigen {
using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Matrix3_6d = Eigen::Matrix<double, 3, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
}  // namespace Eigen

namespace {

inline double square(double x) { return x * x; }

struct ResultTuple {
    ResultTuple() {
        JTJ.setZero();
        JTr.setZero();
    }

    ResultTuple operator+(const ResultTuple &other) {
        this->JTJ += other.JTJ;
        this->JTr += other.JTr;
        return *this;
    }

    Eigen::Matrix6d JTJ;
    Eigen::Vector6d JTr;
};

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::Vector3d> &points) {
    std::transform(points.cbegin(), points.cend(), points.begin(),
                   [&](const auto &point) { return T * point; });
}

void TransformPoints(const Sophus::SE3d &T, std::vector<VectorNd> &points) {
    // Subtract x, y, z
    const long int n_features = points.front().rows() - 3;
    std::transform(points.cbegin(), points.cend(), points.begin(), [&](const auto &point) {
        VectorNd point_transformed;
        point_transformed.resize(3 + n_features);
        point_transformed << T * point.template head<3>(), point.template tail(n_features);
        return point_transformed;
    });
}

void TransformPoints(const Sophus::SE3d &T, std::vector<Eigen::VectorXd> &points) {
    // Subtract x, y, z
    const long int n_features = points.front().rows() - 3;
    std::transform(points.cbegin(), points.cend(), points.begin(), [&](const auto &point) {
        Eigen::VectorXd point_transformed;
        point_transformed.resize(3 + n_features);
        point_transformed << T * point.template head<3>(), point.template tail(n_features);
        return point_transformed;
    });
}

constexpr int MAX_NUM_ITERATIONS_ = 1000;  // 500;
constexpr double ESTIMATION_THRESHOLD_ = 0.0001;
constexpr double EUCL_DIST_THRESHOLD_ = 0.01;

std::tuple<Eigen::Matrix6d, Eigen::Vector6d> BuildLinearSystem(
    // Input are the 3D correspondences
    const std::vector<Eigen::Vector3d> &source,
    const std::vector<Eigen::Vector3d> &target,
    double kernel) {
    auto compute_jacobian_and_residual = [&](auto i) {
        // ToDo: This is the residual computation that could be changed
        // This is only the Euclidean distance (KISS-ICP, SAGE-ICP)
        const Eigen::Vector3d residual = source[i] - target[i];
        Eigen::Matrix3_6d J_r;
        J_r.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
        J_r.block<3, 3>(0, 3) = -1.0 * Sophus::SO3d::hat(source[i]);
        return std::make_tuple(J_r, residual);
    };

    // Signature of tbb::parallel_reduce
    // 1: range
    // 2: identity: starting empty value
    // 3: kernel: parallel computation
    // 4: reduction function

    const auto &[JTJ, JTr] = tbb::parallel_reduce(
        // Range
        tbb::blocked_range<size_t>{0, source.size()},
        // Identity
        ResultTuple(),
        // 1st Lambda: Parallel computation
        [&](const tbb::blocked_range<size_t> &r, ResultTuple J) -> ResultTuple {
            auto Weight = [&](double residual2) {
                return square(kernel) / square(kernel + residual2);
            };
            auto &[JTJ_private, JTr_private] = J;
            for (auto i = r.begin(); i < r.end(); ++i) {
                const auto &[J_r, residual] = compute_jacobian_and_residual(i);
                const double w = Weight(residual.squaredNorm());
                JTJ_private.noalias() += J_r.transpose() * w * J_r;
                JTr_private.noalias() += J_r.transpose() * w * residual;
            }
            return J;
        },
        // 2nd Lambda: Parallel reduction of the private Jacobians
        [&](ResultTuple a, const ResultTuple &b) -> ResultTuple { return a + b; });

    return std::make_tuple(JTJ, JTr);
}
}  // namespace

namespace kiss_icp {

Sophus::SE3d RegisterFrame(const std::vector<Eigen::Vector3d> &frame,
                           const VoxelHashMap &voxel_map,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance,
                           double kernel) {
    if (voxel_map.Empty()) return initial_guess;

    // Equation (9)
    std::vector<Eigen::Vector3d> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    int j = 0;
    for (; j < MAX_NUM_ITERATIONS_; ++j) {
        // Equation (10)
        // auto begin = std::chrono::steady_clock::now();
        const auto &[src, tgt] = voxel_map.GetCorrespondences(source, max_correspondence_distance);
        if (src.size() == 0) {
            std::cout << "[3D] [" << j << "] No correspondences found" << std::endl;
            break;
        }
        // if (j == 0) {
        //     auto end = std::chrono::steady_clock::now();
        // std::cout << "[3D] [" << j << "] correspondences " << src.size() << std::endl;
        //     std::cout << "Elapsed: " <<
        //     static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds> (end -
        //     begin).count()) / 1000000.0 << std::endl;
        // }
        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(src, tgt, kernel);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (dx.norm() < ESTIMATION_THRESHOLD_) {
            // std::cout << "[3D] [" << j << "] correspondences " << src.size() << std::endl;
            break;
        }
        // if (j == MAX_NUM_ITERATIONS_ - 1) {
        //     std::cout << "[3D] [" << j << "] correspondences " << src.size() << std::endl;
        // }
    }
    // std::cout << "[3D] [" << j << "] finished" << std::endl;

    // Spit the final transformation
    return T_icp * initial_guess;
}

Sophus::SE3d RegisterFrame(const std::vector<VectorNd> &frame,
                           const VoxelHashMap &voxel_map,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance,
                           double kernel,
                           std::vector<Eigen::Vector3d> &src_,
                           std::vector<Eigen::Vector3d> &tgt_) {
    if (voxel_map.EmptyN()) return initial_guess;
    auto start = std::chrono::steady_clock::now();
    // Equation (9)
    std::vector<VectorNd> source = frame;
    TransformPoints(initial_guess, source);

    int j = 0;
    Sophus::SE3d T_icp = Sophus::SE3d();

    auto begin = std::chrono::steady_clock::now();
    // VFM-based correspondences
    double min_cosine_similarity = 0.8;
    // For the correspondences search, heavily downsample the source point cloud
    auto voxelized_source = VoxelDownsample(source, 5.0);
    if (voxelized_source.size() < 100) {
        std::cout << "[WARNING] Voxelized too sparse. Keep input." << std::endl;
        voxelized_source = source;
    }
    auto end = std::chrono::steady_clock::now();
    // std::cout << "Voxelization:     "
    //           << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << " ms"
    //           << std::endl;
    // std::cout << "Source " << source.size() << " | Voxelized " << voxelized_source.size() <<
    // std::endl; auto begin = std::chrono::steady_clock::now();
    auto [src_3d, tgt_3d] =
        voxel_map.GetVFMCorrespondences(voxelized_source, min_cosine_similarity);
    // auto end = std::chrono::steady_clock::now();
    // std::cout << "Elapsed: " <<
    // static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds> (end -
    // begin).count()) / 1000000.0 << std::endl;

    // Initialize euclidean distance
    double prev_euclidean_distance = 0.0;
    double max_dist = 0.0;
    for (size_t i = 0; i < src_3d.size(); ++i) {
        prev_euclidean_distance += (src_3d[i] - tgt_3d[i]).norm();
        max_dist = std::max(max_dist, (src_3d[i] - tgt_3d[i]).norm());
    }
    prev_euclidean_distance /= static_cast<double>(src_3d.size());
    // std::cout << "Init. mean " << prev_euclidean_distance << std::endl;
    // std::cout << "Init. max " << max_dist << std::endl;

    // From now on, we will only need the 3D points
    std::vector<Eigen::Vector3d> source_3d;
    for (const auto &point : source) {
        source_3d.emplace_back(point.head<3>());
    }

    // VFM ICP-loop
    for (; j < MAX_NUM_ITERATIONS_; j++) {
        // If no VFM correspondences, continue with vanilla ICP
        if (src_3d.size() == 0) {
            std::cout << "No correspondences found" << std::endl;
            break;
        }

        // Update transform
        const auto &[JTJ, JTr] = BuildLinearSystem(src_3d, tgt_3d, kernel);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source_3d);
        TransformPoints(estimation, src_3d);
        // Update iterations
        T_icp = estimation * T_icp;

        // Compute euclidean distance between corresponding points
        std::vector<double> euclidean_distances;
        std::vector<double> euclidean_distances_median;
        euclidean_distances.reserve(src_3d.size());
        for (size_t i = 0; i < src_3d.size(); ++i) {
            euclidean_distances.emplace_back((src_3d[i] - tgt_3d[i]).norm());
            euclidean_distances_median.emplace_back((src_3d[i] - tgt_3d[i]).norm());
        }
        double mean = std::accumulate(euclidean_distances.begin(), euclidean_distances.end(), 0.0) /
                      static_cast<double>(euclidean_distances.size());
        double accum = 0.0;
        std::for_each(euclidean_distances.begin(), euclidean_distances.end(),
                      [&](const double d) { accum += (d - mean) * (d - mean); });
        double stddev = sqrt(accum / static_cast<double>(euclidean_distances.size()));
        // std::cout << "[ND] [" << j << "] Eucl. dist | Mean: " << mean << "\t Stddev: " << stddev
        // << std::endl; std::cout << "[ND] [" << j << "] Delta Eucl. dist: " <<
        // std::abs(prev_euclidean_distance - mean) << std::endl; std::cout << "Max dist: " <<
        // *std::max_element(euclidean_distances.begin(), euclidean_distances.end()) << std::endl;

        // Compute median
        size_t n = euclidean_distances_median.size() / 2;
        // afterwards, nth element is the median
        std::nth_element(euclidean_distances_median.begin(), euclidean_distances_median.begin() + n,
                         euclidean_distances_median.end());
        double median = euclidean_distances_median[n];
        // If the vector is even
        if (!(euclidean_distances_median.size() & 1)) {
            double max_value = *std::max_element(euclidean_distances_median.begin(),
                                                 euclidean_distances_median.begin() + n);
            median = (median + max_value) / 2;
        }
        // std::cout << "Median: " << median << std::endl;
        // Compute median absolute deviation (MAD)
        std::vector<double> abs_diff(euclidean_distances.size());
        std::transform(euclidean_distances.begin(), euclidean_distances.end(), abs_diff.begin(),
                       [&](const double d) { return std::abs(d - median); });
        n = abs_diff.size() / 2;
        std::nth_element(abs_diff.begin(), abs_diff.begin() + n, abs_diff.end());
        double mad = abs_diff[n];
        if (!(abs_diff.size() & 1)) {
            double max_value = *std::max_element(abs_diff.begin(), abs_diff.begin() + n);
            mad = (mad + max_value) / 2;
        }
        mad *= 1.4826;
        // std::cout << "MAD: " << mad << std::endl;

        // Filter correspondences
        std::vector<Eigen::Vector3d> src_3d_, tgt_3d_;
        for (size_t i = 0; i < euclidean_distances.size(); ++i) {
            // if (std::abs(euclidean_distances[i] - mean) < 2.5 * stddev) {
            if (std::abs(euclidean_distances[i] - median) < 1.5 * mad) {
                src_3d_.emplace_back(src_3d[i]);
                tgt_3d_.emplace_back(tgt_3d[i]);
            }
        }
        if (src_3d_.size() != src_3d.size()) {
            // std::cout << "[ND] [" << j << "] " << src_3d.size() << " -> " << src_3d_.size() <<
            // std::endl;
        }
        src_3d = src_3d_;
        tgt_3d = tgt_3d_;

        if (std::abs(prev_euclidean_distance - mean) < EUCL_DIST_THRESHOLD_) {
            break;
        }
        prev_euclidean_distance = mean;
    }
    src_ = src_3d;
    tgt_ = tgt_3d;
    std::cout << "[ND] [" << j << "] finished VFM" << std::endl;

    auto end_2 = std::chrono::steady_clock::now();
    // std::cout << "Elapsed: " <<
    // static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds> (end_2 -
    // end).count()) / 1000000.0 << std::endl;

    // ICP-loop
    for (; j < MAX_NUM_ITERATIONS_; ++j) {
        // break;
        // Equation (10)
        const auto &[src_3d, tgt_3d] =
            voxel_map.GetCorrespondences(source_3d, max_correspondence_distance);
        if (src_3d.size() == 0) {
            std::cout << "No correspondences found" << std::endl;
        }
        // Equation (11)
        const auto &[JTJ, JTr] = BuildLinearSystem(src_3d, tgt_3d, kernel);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source_3d);
        // Update iterations
        T_icp = estimation * T_icp;

        // Transform points of return source correspondences
        TransformPoints(estimation, src_);

        // Termination criteria
        if (dx.norm() < ESTIMATION_THRESHOLD_) {
            // std::cout << "[ND] [" << j << "] correspondences " << src.size() << std::endl;
            break;
        }
    }
    std::cout << "[ND] [" << j << "] finished" << std::endl;

    auto end_3 = std::chrono::steady_clock::now();
    // std::cout << "Elapsed: " <<
    // static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds> (end_3 -
    // end_2).count()) / 1000000.0 << std::endl;

    // Spit the final transformation
    return T_icp * initial_guess;
}

Sophus::SE3d RegisterFrame(const std::vector<Eigen::VectorXd> &frame,
                           const VoxelHashMap &voxel_map,
                           const Sophus::SE3d &initial_guess,
                           double max_correspondence_distance,
                           double kernel) {
    if (voxel_map.EmptyX()) return initial_guess;

    // Equation (9)
    std::vector<Eigen::VectorXd> source = frame;
    TransformPoints(initial_guess, source);

    // ICP-loop
    Sophus::SE3d T_icp = Sophus::SE3d();
    for (int j = 0; j < MAX_NUM_ITERATIONS_; ++j) {
        // Equation (10)
        const auto &[src, tgt] = voxel_map.GetCorrespondences(source, max_correspondence_distance);
        if (j == 0) std::cout << "[XD] [" << j << "] correspondences " << src.size() << std::endl;
        // Equation (11)
        std::vector<Eigen::Vector3d> src_3d;
        for (const auto &point : src) {
            src_3d.emplace_back(point.head<3>());
        }
        std::vector<Eigen::Vector3d> tgt_3d;
        for (const auto &point : tgt) {
            tgt_3d.emplace_back(point.head<3>());
        }
        const auto &[JTJ, JTr] = BuildLinearSystem(src_3d, tgt_3d, kernel);
        const Eigen::Vector6d dx = JTJ.ldlt().solve(-JTr);
        const Sophus::SE3d estimation = Sophus::SE3d::exp(dx);
        // Equation (12)
        TransformPoints(estimation, source);
        // Update iterations
        T_icp = estimation * T_icp;
        // Termination criteria
        if (dx.norm() < ESTIMATION_THRESHOLD_) break;
    }
    // Spit the final transformation
    return T_icp * initial_guess;
}

}  // namespace kiss_icp
