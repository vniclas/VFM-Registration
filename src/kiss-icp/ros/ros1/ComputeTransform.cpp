#include <ros/ros.h>
#include <iostream>
#include <sophus/se3.hpp>

// KISS-ICP-ROS
#include "kiss_icp_pkg/ComputeTransform.h"
#include "Utils.hpp"

// KISS-ICP
#include "kiss_icp/pipeline/KissICP.hpp"

namespace kiss_icp_ros {

using utils::EigenToPointCloud2;
using utils::GetTimestamps;
using utils::PointCloud2ToEigen;
using utils::PointCloud2ToEigenVFM;

bool computeTransform(kiss_icp_pkg::ComputeTransform::Request &req,
                      kiss_icp_pkg::ComputeTransform::Response &res) {
    // frame_a: source in local frame
    // frame_b: target in global frame
    // initial_guess: initial guess for pose of frame_a in global frame

    bool use_descriptors = false;

    std::cout << req.frame_a.header.seq << " " << req.frame_b.header.seq << std::endl;

    // Construct the main KISS-ICP odometry node
    kiss_icp::pipeline::KISSConfig config = kiss_icp::pipeline::KISSConfig();
    kiss_icp::pipeline::KissICP odometry = kiss_icp::pipeline::KissICP(config);

    const auto frame_a = PointCloud2ToEigenVFM(&req.frame_a, use_descriptors);
    const auto frame_b = PointCloud2ToEigenVFM(&req.frame_b, use_descriptors);

    Sophus::SE3d initial_guess = tf2::transformToSophus(req.initial_guess);

    const auto transform = odometry.ComputeTransform(frame_a, frame_b, initial_guess);

    res.transform.translation.x = transform.translation().x();
    res.transform.translation.y = transform.translation().y();
    res.transform.translation.z = transform.translation().z();

    res.transform.rotation.x = transform.unit_quaternion().x();
    res.transform.rotation.y = transform.unit_quaternion().y();
    res.transform.rotation.z = transform.unit_quaternion().z();
    res.transform.rotation.w = transform.unit_quaternion().w();

    return true;
}

}  // namespace kiss_icp_ros

int main(int argc, char **argv) {

    ros::init(argc, argv, "compute_transform_server");
    ros::NodeHandle nh;

    ros::ServiceServer service = nh.advertiseService("compute_transform", kiss_icp_ros::computeTransform);
    ROS_INFO("Ready to compute transform.");
    ros::spin();

    return 0;
}
