# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
cmake_minimum_required(VERSION 3.16...3.26)
project(kiss_icp_pkg VERSION 0.3.0 LANGUAGES CXX)

# add_compile_options(-std=c++11)
set(ignore ${CATKIN_INSTALL_INTO_PREFIX_ROOT})
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../cpp/kiss_icp/)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../cpp/kiss_icp ${CMAKE_CURRENT_BINARY_DIR}/kiss_icp)
else()
  # cmake_minimum_required(VERSION 3.18)
  # message(STATUS "Performing out-of-tree build, fetching KISS-ICP v${CMAKE_PROJECT_VERSION} Release from Github")
  # include(FetchContent)
  # FetchContent_Declare(
  #   ext_kiss_icp_core PREFIX kiss_icp_core
  #   URL https://github.com/PRBonn/kiss-icp/archive/refs/tags/v${CMAKE_PROJECT_VERSION}.tar.gz SOURCE_SUBDIR
  #       cpp/kiss_icp)
  # FetchContent_MakeAvailable(ext_kiss_icp_core)
endif()

if("$ENV{ROS_VERSION}" STREQUAL "1")
  message(STATUS "KISS-ICP ROS 1 wrapper will be compiled")
  find_package(
    catkin REQUIRED
    COMPONENTS geometry_msgs
               nav_msgs
               sensor_msgs
               geometry_msgs
               roscpp
               rosbag
               std_msgs
               tf2
               tf2_ros
               message_generation
               )
  add_service_files(FILES
    ComputeTransform.srv
  )
  generate_messages(
    DEPENDENCIES
    sensor_msgs
    geometry_msgs
  )

  catkin_package()

  # ROS 1 node
  add_executable(odometry_node ros1/OdometryServer.cpp)
  add_executable(compute_transform_server ros1/ComputeTransform.cpp)
  target_compile_features(odometry_node PUBLIC cxx_std_20)
  target_compile_features(compute_transform_server PUBLIC cxx_std_20)
  target_include_directories(odometry_node PUBLIC include ${catkin_INCLUDE_DIRS})
  target_include_directories(compute_transform_server PUBLIC include ${catkin_INCLUDE_DIRS})
  target_link_libraries(odometry_node kiss_icp::pipeline ${catkin_LIBRARIES})
  target_link_libraries(compute_transform_server kiss_icp::pipeline ${catkin_LIBRARIES})
  install(TARGETS odometry_node RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
  install(DIRECTORY launch DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})


elseif("$ENV{ROS_VERSION}" STREQUAL "2")
  message(STATUS "KISS-ICP ROS 2 wrapper will be compiled")

  # find_package(ament_cmake REQUIRED)
  # find_package(nav_msgs REQUIRED)
  # find_package(rcutils REQUIRED)
  # find_package(rclcpp REQUIRED)
  # find_package(rclcpp_components REQUIRED)
  # find_package(sensor_msgs REQUIRED)
  # find_package(tf2_ros REQUIRED)

  # # ROS 2 node
  # add_library(odometry_component SHARED ros2/OdometryServer.cpp)
  # target_compile_features(odometry_component PUBLIC cxx_std_20)
  # target_include_directories(odometry_component PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  # target_link_libraries(odometry_component kiss_icp::pipeline)
  # ament_target_dependencies(
  #   odometry_component
  #   rcutils
  #   rclcpp
  #   rclcpp_components
  #   nav_msgs
  #   sensor_msgs
  #   geometry_msgs
  #   tf2_ros)

  # rclcpp_components_register_node(odometry_component PLUGIN "kiss_icp_ros::OdometryServer" EXECUTABLE odometry_node)

  # install(TARGETS odometry_component LIBRARY DESTINATION lib RUNTIME DESTINATION lib/${PROJECT_NAME})
  # install(DIRECTORY launch rviz DESTINATION share/${PROJECT_NAME}/)

  # ament_package()

else()
  message(FATAL_ERROR "catkin or colcon not found KISS-ICP-ROS disabled")
endif()
