import math
from typing import Optional, Union

import faiss
import numpy as np
import rospy
import tf_conversions
from geometry_msgs.msg import Point as PointMsg
from geometry_msgs.msg import Pose as PoseMsg
from geometry_msgs.msg import Transform as TransformMsg
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation as R
from sensor_msgs.msg import PointCloud2 as PointCloudMsg
from sensor_msgs.msg import PointField
from visualization_msgs.msg import Marker as MarkerMsg
from visualization_msgs.msg import MarkerArray as MarkerArrayMsg


class FaissKNeighbors:

    def __init__(self):
        self.index = None
        self.y = None

    def fit(self, X, y):
        self.index = faiss.IndexFlatL2(X.shape[1])
        self.index.add(X.astype(np.float32))
        self.y = y

    def query(self, X, k, r):
        distances, indices = self.index.search(X.astype(np.float32), k)
        distances, indices = distances.flatten(), indices.flatten()
        indices = indices[distances > 0]
        distances = distances[distances > 0]
        indices = indices[distances < r]
        distances = distances[distances < r]
        return np.unique(self.y[indices])

    def n_neighbors_in_radius(self, X, k, r):
        distances, indices = self.index.search(X.astype(np.float32), k)
        indices[distances <= 0.0] = -1
        indices[distances > r] = -1
        n_neighbors = np.sum(indices != -1, axis=1)
        return n_neighbors


def transform_pcl(pcl: ArrayLike, transform: ArrayLike) -> ArrayLike:
    assert transform.shape == (4, 4), "Invalid shape"
    pcl_xyz = pcl[:, :3].T.copy()
    pcl_xyz = np.insert(pcl_xyz, 3, values=1, axis=0)
    pcl_xyz = transform @ pcl_xyz
    pcl_out = np.c_[pcl_xyz.T[:, :3], pcl[:, 3:]]
    assert pcl_out.shape == pcl.shape
    return pcl_out.astype(pcl.dtype)


def matrix_to_transform_msg(matrix: ArrayLike) -> TransformMsg:
    assert matrix.shape == (4, 4), "Invalid shape"
    msg = TransformMsg()
    msg.translation.x = matrix[0, 3]
    msg.translation.y = matrix[1, 3]
    msg.translation.z = matrix[2, 3]
    quaternion = tf_conversions.transformations.quaternion_from_matrix(matrix)
    quaternion /= np.linalg.norm(quaternion)
    msg.rotation.x = quaternion[0]
    msg.rotation.y = quaternion[1]
    msg.rotation.z = quaternion[2]
    msg.rotation.w = quaternion[3]
    return msg


def transform_msg_to_matrix(msg: TransformMsg) -> np.ndarray:
    translation = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    quaternion = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    matrix = tf_conversions.transformations.quaternion_matrix(quaternion)
    matrix[:3, 3] = translation
    return matrix


def matrix_to_pose_msg(matrix: ArrayLike) -> PoseMsg:
    assert matrix.shape == (4, 4), "Invalid shape"
    msg = PoseMsg()
    msg.position.x = matrix[0, 3]
    msg.position.y = matrix[1, 3]
    msg.position.z = matrix[2, 3]
    quaternion = tf_conversions.transformations.quaternion_from_matrix(matrix)
    quaternion /= np.linalg.norm(quaternion)
    msg.orientation.x = quaternion[0]
    msg.orientation.y = quaternion[1]
    msg.orientation.z = quaternion[2]
    msg.orientation.w = quaternion[3]
    return msg


def pose_msg_to_matrix(msg: PoseMsg) -> np.ndarray:
    translation = np.array([msg.position.x, msg.position.y, msg.position.z])
    quaternion = np.array(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
    matrix = tf_conversions.transformations.quaternion_matrix(quaternion)
    matrix[:3, 3] = translation
    return matrix


def print_msg(msg: Union[TransformMsg, PoseMsg, ArrayLike],
              string: str = "",
              color: Optional[str] = None) -> None:
    # return
    if isinstance(msg, TransformMsg):
        euler = tf_conversions.transformations.euler_from_quaternion(
            [msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
        translation = np.array([msg.translation.x, msg.translation.y, msg.translation.z])

    elif isinstance(msg, PoseMsg):
        euler = tf_conversions.transformations.euler_from_quaternion(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        translation = np.array([msg.position.x, msg.position.y, msg.position.z])

    elif isinstance(msg, np.ndarray):
        assert msg.shape == (4, 4), "Invalid shape"
        euler = R.from_matrix(msg[:3, :3]).as_euler("xyz")
        translation = msg[:3, 3]

    else:
        raise ValueError("Invalid message type")

    if string:
        string = f" | {string}"

    # Convert radians to degrees
    euler = np.array([math.degrees(euler[0]), math.degrees(euler[1]), math.degrees(euler[2])])
    # rospy.loginfo(
    #     f"{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f} | {euler[0]:.3f}, {euler[1]:.3f}, {euler[2]:.3f} | {string}"
    # )

    string = f"T = {translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f} ".ljust(30) + \
             f"| R = {euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}".ljust(30) + \
             f"{string}"

    if color is None:
        pass
    elif color == 'red':
        string = f"\033[91m{string}\033[0m"
    elif color == 'green':
        string = f"\033[92m{string}\033[0m"
    elif color == 'yellow':
        string = f"\033[93m{string}\033[0m"
    elif color == 'blue':
        string = f"\033[94m{string}\033[0m"
    else:
        raise ValueError("Invalid color")

    print(string)


def publish_point_cloud(point_cloud, publisher, frame_id="odom", stamp=None, mode="auto") -> None:
    if mode == "auto":
        if point_cloud.shape[1] == 3:
            mode = "xyz"
        elif point_cloud.shape[1] == 6:
            mode = "xyzrgb"
        else:
            # Fallback
            mode = "xyz"
    if mode not in ["xyz", "xyzrgb"]:
        raise ValueError(f"Invalid mode: {mode}")

    if stamp is None:
        stamp = rospy.Time.now()

    pcl_msg = PointCloudMsg()
    pcl_msg.header.frame_id = frame_id
    pcl_msg.header.stamp = stamp
    pcl_msg.height = 1
    pcl_msg.width = point_cloud.shape[0]
    pcl_msg.fields = [
        PointField("x", 0, PointField.FLOAT32, 1),
        PointField("y", 4, PointField.FLOAT32, 1),
        PointField("z", 8, PointField.FLOAT32, 1),
    ]
    if mode == "xyzrgb":
        pcl_msg.fields.extend([
            PointField("r", 12, PointField.FLOAT32, 1),
            PointField("g", 16, PointField.FLOAT32, 1),
            PointField("b", 20, PointField.FLOAT32, 1)
        ])
    pcl_msg.is_bigendian = False
    pcl_msg.point_step = 4 * len(pcl_msg.fields)
    pcl_msg.row_step = pcl_msg.point_step * pcl_msg.width
    pcl_msg.is_dense = True
    k = 3 if mode == "xyz" else 6
    pcl_msg.data = point_cloud[:, :k].astype(np.float32).tobytes()
    publisher.publish(pcl_msg)


def publish_correspondences(correspondences, publisher, color) -> None:
    marker_array = MarkerArrayMsg()
    timestamp = rospy.Time.now()

    # Add correspondence markers
    for (i, src), tgt in zip(enumerate(correspondences[0]), correspondences[1]):
        marker = MarkerMsg()
        marker.header.frame_id = "odom"
        marker.header.stamp = timestamp
        marker.ns = "correspondences"
        marker.id = i
        marker.type = MarkerMsg.LINE_STRIP
        marker.action = MarkerMsg.ADD
        marker.pose.orientation.w = 1
        marker.scale.x = 0.4
        marker.color.a = 1.0
        marker.color.r = color[0]
        marker.color.g = color[1]
        marker.color.b = color[2]
        point_src = PointMsg()
        point_src.x = src[0]
        point_src.y = src[1]
        point_src.z = src[2]
        point_tgt = PointMsg()
        point_tgt.x = tgt[0]
        point_tgt.y = tgt[1]
        point_tgt.z = tgt[2]
        marker.points.append(point_src)
        marker.points.append(point_tgt)
        marker_array.markers.append(marker)

    publisher.publish(marker_array)
