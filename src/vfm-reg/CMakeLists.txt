cmake_minimum_required(VERSION 2.8.3)
project(vfm_reg)

find_package(catkin REQUIRED COMPONENTS
    rospy
    sensor_msgs
)

catkin_package()

catkin_python_setup()

catkin_install_python(PROGRAMS
    src/registration_node.py
    DESTINATION
    ${CATKIN_PACKAGE_BIN_DESTINATION}
)
