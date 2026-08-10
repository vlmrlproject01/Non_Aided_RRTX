import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, Twist
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Odometry, Path
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

from rrtx_planner.rrtx_algorithm_scan import (
    INFLATION_RADIUS,
    MAP_MAX_X,
    MAP_MAX_Y,
    MAP_MIN_X,
    MAP_MIN_Y,
    ROBOT_RADIUS,
    OccupancyGrid2D,
    RRTX,
    is_collision_free,
)


class RRTXPlanner(Node):
    """ROS 2 node that runs an RRTX planner and drives TurtleBot3."""

    # TurtleBot3 Burger limits
    MAX_LINEAR_VEL = 0.22
    MAX_ANGULAR_VEL = 2.84

    # Safer limits used by this controller
    CMD_LINEAR_LIMIT = 0.15
    CMD_ANGULAR_LIMIT = 0.50

    GOAL_TOLERANCE = 0.15
    WAYPOINT_TOLERANCE = 0.20
    MAX_SCAN_RANGE = 2.5
    PLANNING_PERIOD = 0.2
    RRTX_ITERATIONS = 100
    PATH_CHANGE_THRESHOLD = 0.25

    def __init__(self):
        super().__init__("rrtx_planner")
        self.get_logger().info("RRTX Planner Started!")

        self.declare_parameter("goal_x", 2.0)
        self.declare_parameter("goal_y", 2.0)

        self.goal = (
            float(self.get_parameter("goal_x").value),
            float(self.get_parameter("goal_y").value),
        )
        self.get_logger().info(f"Target goal: {self.goal}")

        # TF2 is used to account for the laser sensor's position relative
        # to the robot base.
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10,
        )
        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        # Publishers
        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.path_pub = self.create_publisher(Path, "/rrtx_path", 10)
        self.grid_pub = self.create_publisher(
            OccupancyGridMsg,
            "/rrtx_grid",
            10,
        )

        # Timer that repeatedly updates the plan and follows the latest path.
        self.plan_timer = self.create_timer(
            self.PLANNING_PERIOD,
            self.plan_callback,
        )

        self.current_position = None
        self.current_yaw = 0.0

        self.grid = OccupancyGrid2D(
            MAP_MIN_X,
            MAP_MAX_X,
            MAP_MIN_Y,
            MAP_MAX_Y,
        )
        self.grid.compute_inflated(INFLATION_RADIUS)

        self.planner = None
        self.planner_initialized = False
        self.latest_path = []
        self.current_waypoint_idx = 1

        self.last_log_time = self.get_clock().now()

    def odom_callback(self, msg):
        """Store the robot position and orientation from odometry."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.current_position = (x, y)

        q = msg.pose.pose.orientation
        self.current_yaw = self.euler_from_quaternion(
            q.x,
            q.y,
            q.z,
            q.w,
        )

        if not self.planner_initialized:
            self.planner = RRTX(
                start=self.current_position,
                goal=self.goal,
                grid=self.grid,
            )
            self.planner_initialized = True
            self.get_logger().info("RRTX Planner Initialized.")

        self.planner.update_start(self.current_position)

    def scan_callback(self, msg):
        """Convert laser hits into obstacle positions in the planner grid."""
        if not self.planner_initialized or self.current_position is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                "base_footprint",
                msg.header.frame_id,
                Time(),
            )
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y

            rotation = transform.transform.rotation
            sensor_yaw = self.euler_from_quaternion(
                rotation.x,
                rotation.y,
                rotation.z,
                rotation.w,
            )
        except TransformException as exc:
            self.throttled_log(
                f"Laser transform unavailable; using zero offset: {exc}",
                standard=False,
            )
            tx = 0.0
            ty = 0.0
            sensor_yaw = 0.0

        robot_x, robot_y = self.current_position
        angle = msg.angle_min

        for distance in msg.ranges:
            valid_distance = (
                msg.range_min <= distance <= self.MAX_SCAN_RANGE
                and np.isfinite(distance)
            )

            if not valid_distance:
                angle += msg.angle_increment
                continue

            # Laser frame coordinates
            laser_x = distance * math.cos(angle)
            laser_y = distance * math.sin(angle)

            # Convert from laser frame to robot-base frame
            base_x = tx + (
                laser_x * math.cos(sensor_yaw)
                - laser_y * math.sin(sensor_yaw)
            )
            base_y = ty + (
                laser_x * math.sin(sensor_yaw)
                + laser_y * math.cos(sensor_yaw)
            )

            # Convert from robot-base frame to odom frame
            map_x = robot_x + (
                base_x * math.cos(self.current_yaw)
                - base_y * math.sin(self.current_yaw)
            )
            map_y = robot_y + (
                base_x * math.sin(self.current_yaw)
                + base_y * math.cos(self.current_yaw)
            )

            self.grid.update_from_scan_hit(
                robot_x,
                robot_y,
                map_x,
                map_y,
            )

            angle += msg.angle_increment

        self.grid.compute_inflated(INFLATION_RADIUS)

        # Prevent the robot's own occupied cells from blocking its start.
        self.grid.clear_around(
            robot_x,
            robot_y,
            ROBOT_RADIUS,
        )

    def plan_callback(self):
        """Update obstacles, grow the planner tree and follow the latest path."""
        if not self.planner_initialized or self.current_position is None:
            return

        self.planner.sync_with_grid()
        self.planner.grow_tree(max_iter=self.RRTX_ITERATIONS)

        new_path = self.planner.extract_path()

        if new_path:
            if self.is_path_significantly_changed(
                new_path,
                self.latest_path,
            ):
                self.latest_path = new_path
                self.current_waypoint_idx = 1
                self.throttled_log(
                    "Global path changed significantly. "
                    "Waypoint tracking was reset.",
                    standard=False,
                )
            else:
                self.latest_path = new_path
        else:
            self.latest_path = []
            self.diagnose_no_path()

        self.publish_path(self.latest_path)
        self.publish_grid()
        self.follow_path()

    def diagnose_no_path(self):
        """Log a likely reason when the planner has no current path."""
        sx, sy = self.current_position
        gx, gy = self.goal

        def in_bounds(x, y):
            return (
                MAP_MIN_X <= x <= MAP_MAX_X
                and MAP_MIN_Y <= y <= MAP_MAX_Y
            )

        if not in_bounds(sx, sy):
            self.throttled_log(
                f"Start position {self.current_position} is outside the "
                f"planner grid bounds x[{MAP_MIN_X}, {MAP_MAX_X}] "
                f"y[{MAP_MIN_Y}, {MAP_MAX_Y}].",
                standard=False,
            )
        elif not in_bounds(gx, gy):
            self.throttled_log(
                f"Goal {self.goal} is outside the planner grid bounds "
                f"x[{MAP_MIN_X}, {MAP_MAX_X}] "
                f"y[{MAP_MIN_Y}, {MAP_MAX_Y}].",
                standard=False,
            )
        elif not is_collision_free(self.current_position, self.grid):
            self.throttled_log(
                "The robot's current position is marked as occupied "
                "in the planner grid.",
                standard=False,
            )
        elif not is_collision_free(self.goal, self.grid):
            self.throttled_log(
                "The goal position is currently occupied in the grid.",
                standard=False,
            )
        else:
            self.throttled_log(
                "No path found yet. The tree is still growing and searching.",
                standard=False,
            )

    def publish_path(self, path):
        """Publish the current path for visualization in RViz."""
        msg = Path()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()

        for x, y in path:
            pose = PoseStamped()
            pose.header = msg.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.path_pub.publish(msg)

    def publish_grid(self):
        """Publish the planner's inflated obstacle grid for RViz."""
        msg = OccupancyGridMsg()
        msg.header.frame_id = "odom"
        msg.header.stamp = self.get_clock().now().to_msg()

        msg.info.resolution = float(self.grid.resolution)
        msg.info.width = int(self.grid.width)
        msg.info.height = int(self.grid.height)
        msg.info.origin.position.x = float(self.grid.origin_x)
        msg.info.origin.position.y = float(self.grid.origin_y)
        msg.info.origin.orientation.w = 1.0

        # OccupancyGrid data is row-major:
        # 0 = free, 100 = occupied, -1 = unknown.
        data = np.where(self.grid.inflated, 100, 0).astype(np.int8)
        msg.data = data.flatten().tolist()

        self.grid_pub.publish(msg)

    def is_path_significantly_changed(self, path_a, path_b):
        """Return True when two paths differ enough to reset waypoint tracking."""
        if not path_a or not path_b:
            return True

        if len(path_a) != len(path_b):
            return True

        mid_idx = len(path_a) // 2
        distance = math.hypot(
            path_a[mid_idx][0] - path_b[mid_idx][0],
            path_a[mid_idx][1] - path_b[mid_idx][1],
        )
        return distance > self.PATH_CHANGE_THRESHOLD

    def follow_path(self):
        """Use a simple rotate-then-drive controller to follow waypoints."""
        if (
            not self.latest_path
            or len(self.latest_path) < 2
            or self.current_position is None
        ):
            self.stop_robot()
            return

        robot_x, robot_y = self.current_position

        goal_distance = math.hypot(
            self.goal[0] - robot_x,
            self.goal[1] - robot_y,
        )

        if goal_distance < self.GOAL_TOLERANCE:
            self.throttled_log(
                "Goal reached successfully!",
                standard=True,
            )
            self.stop_robot()
            return

        while self.current_waypoint_idx < len(self.latest_path):
            waypoint = self.latest_path[self.current_waypoint_idx]
            waypoint_distance = math.hypot(
                waypoint[0] - robot_x,
                waypoint[1] - robot_y,
            )

            if waypoint_distance < self.WAYPOINT_TOLERANCE:
                self.current_waypoint_idx += 1
            else:
                break

        if self.current_waypoint_idx >= len(self.latest_path):
            self.current_waypoint_idx = len(self.latest_path) - 1

        target_x, target_y = self.latest_path[self.current_waypoint_idx]
        target_distance = math.hypot(
            target_x - robot_x,
            target_y - robot_y,
        )

        desired_yaw = math.atan2(
            target_y - robot_y,
            target_x - robot_x,
        )
        yaw_error = math.atan2(
            math.sin(desired_yaw - self.current_yaw),
            math.cos(desired_yaw - self.current_yaw),
        )

        cmd = Twist()

        if abs(yaw_error) > 0.5:
            cmd.linear.x = 0.0
            cmd.angular.z = (
                self.CMD_ANGULAR_LIMIT
                if yaw_error > 0.0
                else -self.CMD_ANGULAR_LIMIT
            )
        else:
            linear_scale = 0.6 * target_distance
            cmd.linear.x = min(
                self.CMD_LINEAR_LIMIT,
                linear_scale,
            )
            cmd.angular.z = 1.2 * yaw_error

        cmd.linear.x = max(
            -self.MAX_LINEAR_VEL,
            min(self.MAX_LINEAR_VEL, cmd.linear.x),
        )
        cmd.angular.z = max(
            -self.MAX_ANGULAR_VEL,
            min(self.MAX_ANGULAR_VEL, cmd.angular.z),
        )

        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        """Publish a zero velocity command."""
        if not rclpy.ok():
            return

        try:
            self.cmd_pub.publish(Twist())
        except Exception as exc:
            self.get_logger().warning(
                f"Could not publish stop command: {exc}"
            )

    @staticmethod
    def euler_from_quaternion(x, y, z, w):
        """Return the yaw angle from a quaternion."""
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def throttled_log(self, text, standard=True):
        """Log a message no more than once every two seconds."""
        now = self.get_clock().now()

        if (now - self.last_log_time).nanoseconds > 2_000_000_000:
            if standard:
                self.get_logger().info(text)
            else:
                self.get_logger().warning(text)

            self.last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = RRTXPlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("RRTX Planner stopped by user.")
    finally:
        node.stop_robot()
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
