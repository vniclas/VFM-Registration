#pragma once

#include <Eigen/Core>


// DINOv2
int constexpr DESCRIPTOR_SIZE = 384;

// MaskCLIP
// int constexpr DESCRIPTOR_SIZE = 512;

// 3D point + N descriptor size
using VectorNd = Eigen::Matrix<double, 3 + DESCRIPTOR_SIZE, 1>;
using Vectornd = Eigen::Matrix<double, DESCRIPTOR_SIZE, 1>;
