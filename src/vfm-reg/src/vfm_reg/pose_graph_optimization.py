from typing import List, Union

import gtsam
from geometry_msgs.msg import Point as PointMsg
from numpy.typing import ArrayLike
from visualization_msgs.msg import Marker as MarkerMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg


class PoseGraphOptimizationGtsam():

    def __init__(self, min_factors: int = 1) -> None:
        self.min_factors = min_factors  # Minimum number of new factors to optimize

        self.new_factors = gtsam.NonlinearFactorGraph()
        self.initial_estimate = gtsam.Values()
        self.current_estimate = gtsam.Values()

        # Parameters from GLIM paper
        self.parameters = gtsam.ISAM2Params()
        self.parameters.setRelinearizeThreshold(0.1)
        self.parameters.relinearizeSkip = 1
        self.isam = gtsam.ISAM2(self.parameters)

        # Building the graph
        self.vertex_ids = []
        self.edges = []

        # Cache
        self.new_loops_count = 0

    @property
    def poses(self) -> List[ArrayLike]:
        poses = []
        for i in self.vertex_ids:
            try:
                poses.append(self.pose_at(i))
            except RuntimeError:
                continue
        return poses

    def __len__(self) -> int:
        return self.size()

    def size(self) -> int:
        return len(self.vertex_ids)

    def pose_at(self, vertex_id: int, return_gtsam: bool = False) -> Union[ArrayLike, gtsam.Pose3]:
        if vertex_id == -1:  # Last vertex
            return self.pose_at(self.vertex_ids[-1], return_gtsam)
        try:
            pose = self.current_estimate.atPose3(vertex_id)
            return pose if return_gtsam else pose.matrix()
        except RuntimeError:
            # print('WARNING: Falling back to initial estimate for vertex', vertex_id)
            pose = self.initial_estimate.atPose3(vertex_id)
            return pose if return_gtsam else pose.matrix()

    def optimize_if_needed(self) -> bool:
        # Optimize early if there is a new loop
        if self.new_loops_count == 1:
            # print('Loop optimization')
            self.optimize()
            return True

        # This is a regular batch optimization
        elif self.new_factors.size() >= self.min_factors:
            # print('Batch optimization')
            self.optimize()
            self.new_loops_count = 0
            return True
        return False

    def optimize(self):
        # Run the optimization
        self.isam.update(self.new_factors, self.initial_estimate)

        # Retrieve the optimized estimate
        self.current_estimate = self.isam.calculateEstimate()

        # Clear after optimization
        self.new_factors = gtsam.NonlinearFactorGraph()
        self.initial_estimate.clear()

    def add_prior_factor_pose(self, vertex_id: int, pose: ArrayLike):
        noise = gtsam.noiseModel.Isotropic.Precision(6, int(1e6))  # from GLIM paper
        self.new_factors.addPriorPose3(vertex_id, gtsam.Pose3(pose), noise)
        self.initial_estimate.insert(vertex_id, gtsam.Pose3(pose))
        # self.current_estimate.insert(vertex_id, gtsam.Pose3(pose))
        self.vertex_ids.append(vertex_id)

    def add_prior_factor_gtsam_pose(self, vertex_id: int, pose: gtsam.Pose3):
        noise = gtsam.noiseModel.Isotropic.Precision(6, int(1e6))  # from GLIM paper
        self.new_factors.addPriorPose3(vertex_id, pose, noise)
        self.initial_estimate.insert(vertex_id, pose)
        # self.current_estimate.insert(vertex_id, gtsam.Pose3(pose))
        self.vertex_ids.append(vertex_id)

    def add_odom_edge(self, vertex_id: int, measurement: gtsam.Pose3, sigma: float):
        noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)  # from GLIM paper
        self.new_factors.add(
            gtsam.BetweenFactorPose3(self.vertex_ids[-1], vertex_id, measurement, noise))
        # Estimate pose of the new vertex: previous_pose * measurement
        estimate = self.pose_at(-1, True).compose(measurement)
        self.initial_estimate.insert(vertex_id, estimate)
        self.vertex_ids.append(vertex_id)
        self.edges.append((self.vertex_ids[-2], vertex_id))

    def add_loop_edge(self, vertex_id1: int, vertex_id2: int, measurement: ArrayLike, sigma: float):
        noise = gtsam.noiseModel.Isotropic.Sigma(6, sigma)  # from GLIM paper
        self.new_factors.push_back(
            gtsam.BetweenFactorPose3(vertex_id1, vertex_id2, gtsam.Pose3(measurement), noise))
        self.edges.append((vertex_id1, vertex_id2))
        self.new_loops_count += 1

    def convert_to_rviz_message(self, frame_id, timestamp) -> MarkerArrayMsg:
        marker_array = MarkerArrayMsg()

        # Add vertex markers
        for i in self.vertex_ids:
            vertex_pose = self.pose_at(i)
            marker = MarkerMsg()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = 'vertices'
            marker.id = i
            marker.type = MarkerMsg.SPHERE
            marker.action = MarkerMsg.ADD
            marker.pose.position.x = vertex_pose[0, 3]
            marker.pose.position.y = vertex_pose[1, 3]
            marker.pose.position.z = vertex_pose[2, 3]
            marker.pose.orientation.w = 1
            marker.scale.x = 0.125
            marker.scale.y = 0.125
            marker.scale.z = 0.125
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker_array.markers.append(marker)

        # Add edge markers
        # ToDo: Sometimes this takes a long time (.1s) to run
        for i, edge in enumerate(self.edges):
            marker = MarkerMsg()
            marker.header.frame_id = frame_id
            marker.header.stamp = timestamp
            marker.ns = 'edges'
            marker.id = i
            marker.type = MarkerMsg.LINE_STRIP
            marker.action = MarkerMsg.ADD
            marker.pose.orientation.w = 1
            marker.scale.x = 0.025
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            for vertex_id in edge:
                vertex_pose = self.pose_at(vertex_id)
                point = PointMsg()
                point.x = vertex_pose[0, 3]
                point.y = vertex_pose[1, 3]
                point.z = vertex_pose[2, 3]
                marker.points.append(point)
            marker_array.markers.append(marker)

        return marker_array
