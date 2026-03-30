#!/usr/bin/env python3
"""
Task executor for face and ring detection + interaction.

Behaviour:
  1. Waits for Nav2 to become ready.
  2. Drives a configurable patrol route (waypoints covering all walls).
  3. Whenever a new face or ring is detected, cancels the current patrol goal
     and diverts to that detection.
       - Face  → navigate to APPROACH_DIST from it, call /greet_service ("Hello!")
       - Ring  → navigate to APPROACH_DIST from it, call /sayColor_service (color name)
  4. After handling a detection, resumes the patrol at the next waypoint.
  5. Any detections that arrive while already handling one are queued (FIFO).
  6. Stops when all patrol waypoints have been visited.

Parameters
----------
patrol_waypoints : list[float]
    Flat list of (x, y) pairs in map frame, e.g. [1.0, 0.5, 1.0, 2.0, ...].
    If not provided, a default rectangular perimeter is used — adjust this to
    match your actual map before running.
approach_dist : float  (default 0.8)
    Distance in metres at which the robot stops in front of a detection.
"""

import math
import threading

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

import tf2_ros
from tf2_ros import TransformException

from rins_robot.msg import FaceCoords, RingCoords
from rins_robot.srv import Speech


# ── helpers ───────────────────────────────────────────────────────────────────

def yaw_to_quat(yaw: float):
    """Return (x, y, z, w) quaternion for a pure-Z rotation."""
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def make_pose(frame: str, stamp, x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.header.stamp = stamp
    ps.pose.position.x = x
    ps.pose.position.y = y
    ps.pose.position.z = 0.0
    qx, qy, qz, qw = yaw_to_quat(yaw)
    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    ps.pose.orientation.w = qw
    return ps


# ── node ──────────────────────────────────────────────────────────────────────

class TaskExecutor(Node):
    """Patrol node that reacts to face/ring detections."""

    # State machine labels
    _S_IDLE      = 'IDLE'
    _S_PATROLLING = 'PATROLLING'
    _S_HANDLING  = 'HANDLING'
    _S_DONE      = 'DONE'

    def __init__(self):
        super().__init__('task_executor')

        # ── parameters ────────────────────────────────────────────────────────
        self.declare_parameter('approach_dist', 0.8)
        self.declare_parameter('patrol_waypoints', rclpy.Parameter.Type.DOUBLE_ARRAY)

        self._approach_dist: float = (
            self.get_parameter('approach_dist').get_parameter_value().double_value
        )

        # Default patrol: rectangular perimeter.
        # IMPORTANT: Replace / extend these to match your actual map walls.
        # Each pair (x, y) is a waypoint in the 'map' frame.
        _default_waypoints = [
            ( 1.5,  0.0),
            ( 1.5,  1.5),
            ( 0.0,  1.5),
            (-1.5,  1.5),
            (-1.5,  0.0),
            (-1.5, -1.5),
            ( 0.0, -1.5),
            ( 1.5, -1.5),
        ]
        try:
            flat = (
                self.get_parameter('patrol_waypoints')
                .get_parameter_value().double_array_value
            )
            if len(flat) >= 2:
                self._waypoints = [
                    (flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)
                ]
            else:
                self._waypoints = _default_waypoints
        except Exception:
            self._waypoints = _default_waypoints

        # ── callback group ─────────────────────────────────────────────────────
        self._cbg = ReentrantCallbackGroup()

        # ── Nav2 action client ─────────────────────────────────────────────────
        self._nav = ActionClient(
            self, NavigateToPose, 'navigate_to_pose',
            callback_group=self._cbg
        )

        # ── detection subscribers ──────────────────────────────────────────────
        self.create_subscription(
            FaceCoords, '/face_coords', self._on_faces, 10,
            callback_group=self._cbg
        )
        self.create_subscription(
            RingCoords, '/ring_coords', self._on_rings, 10,
            callback_group=self._cbg
        )

        # ── speech service clients ─────────────────────────────────────────────
        self._greet_cli = self.create_client(
            Speech, '/greet_service', callback_group=self._cbg
        )
        self._color_cli = self.create_client(
            Speech, '/sayColor_service', callback_group=self._cbg
        )

        # ── TF listener ───────────────────────────────────────────────────────
        self._tf_buffer   = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self)

        # ── internal state (protected by a lock) ──────────────────────────────
        self._lock              = threading.Lock()
        self._state             = self._S_IDLE
        self._waypoint_idx      = 0
        self._current_goal_hdl  = None   # accepted nav goal handle

        # Detections pending interaction: list of dicts
        #   {'type': 'face'|'ring', 'id': int, 'x': float, 'y': float, 'color': str}
        self._pending: list = []
        self._visited_faces: set = set()
        self._visited_rings: set = set()

        # ── start-up ──────────────────────────────────────────────────────────
        self.get_logger().info(
            f'Task executor ready. {len(self._waypoints)} patrol waypoints. '
            f'Approach dist={self._approach_dist} m. Waiting for Nav2…'
        )
        self._startup_timer = self.create_timer(
            1.0, self._startup_check, callback_group=self._cbg
        )

    # ── start-up ──────────────────────────────────────────────────────────────

    def _startup_check(self):
        if self._nav.wait_for_server(timeout_sec=0.1):
            self._startup_timer.cancel()
            self.get_logger().info('Nav2 ready — starting patrol.')
            with self._lock:
                self._state = self._S_PATROLLING
            self._advance()

    # ── detection callbacks ────────────────────────────────────────────────────

    def _on_faces(self, msg: FaceCoords):
        for pt, fid in zip(msg.points, msg.ids):
            with self._lock:
                if fid in self._visited_faces:
                    continue
                if any(p['type'] == 'face' and p['id'] == fid for p in self._pending):
                    continue
                self.get_logger().info(
                    f'New face  id={fid}  @ ({pt.x:.2f}, {pt.y:.2f})'
                )
                self._pending.append({
                    'type': 'face', 'id': fid,
                    'x': pt.x, 'y': pt.y, 'color': ''
                })
                self._maybe_interrupt()

    def _on_rings(self, msg: RingCoords):
        for pt, rid, color in zip(msg.points, msg.ids, msg.colors):
            with self._lock:
                if rid in self._visited_rings:
                    continue
                if any(p['type'] == 'ring' and p['id'] == rid for p in self._pending):
                    continue
                self.get_logger().info(
                    f'New ring  id={rid}  color={color}  @ ({pt.x:.2f}, {pt.y:.2f})'
                )
                self._pending.append({
                    'type': 'ring', 'id': rid,
                    'x': pt.x, 'y': pt.y, 'color': color
                })
                self._maybe_interrupt()

    def _maybe_interrupt(self):
        """Cancel the active patrol goal so we divert to the detection.

        Must be called with self._lock held.
        """
        if self._state == self._S_PATROLLING and self._current_goal_hdl is not None:
            self._current_goal_hdl.cancel_goal_async()

    # ── navigation helpers ─────────────────────────────────────────────────────

    def _robot_xy(self):
        """Return (x, y) of base_link in map frame, or (0, 0) on failure."""
        try:
            t = self._tf_buffer.lookup_transform(
                'map', 'base_link', rclpy.time.Time()
            )
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return 0.0, 0.0

    def _approach_pose(self, tx: float, ty: float) -> PoseStamped:
        """Pose APPROACH_DIST in front of (tx, ty), facing it."""
        rx, ry = self._robot_xy()
        dx, dy = tx - rx, ty - ry
        dist = math.hypot(dx, dy)
        if dist < 1e-3:
            dist = 1e-3
        # Back off from target toward robot
        ux, uy = -dx / dist, -dy / dist
        ax = tx + ux * self._approach_dist
        ay = ty + uy * self._approach_dist
        yaw = math.atan2(dy, dx)   # face the target
        return make_pose('map', self.get_clock().now().to_msg(), ax, ay, yaw)

    def _send_nav_goal(self, pose: PoseStamped, on_done):
        """Send a NavigateToPose goal; call on_done(success: bool) when finished."""
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose = pose

        send_future = self._nav.send_goal_async(goal_msg)

        def _goal_accepted(fut):
            handle = fut.result()
            if not handle.accepted:
                self.get_logger().warn('Nav goal rejected.')
                on_done(False)
                return
            with self._lock:
                self._current_goal_hdl = handle
            handle.get_result_async().add_done_callback(_got_result)

        def _got_result(fut):
            with self._lock:
                self._current_goal_hdl = None
            status = fut.result().status
            on_done(status == GoalStatus.STATUS_SUCCEEDED)

        send_future.add_done_callback(_goal_accepted)

    # ── main control flow ──────────────────────────────────────────────────────

    def _advance(self):
        """Decide what to do next: handle a pending detection or resume patrol."""
        with self._lock:
            state   = self._state
            pending = list(self._pending)

        if state == self._S_DONE:
            return

        if pending:
            self._handle_next()
        else:
            self._patrol_step()

    def _patrol_step(self):
        with self._lock:
            idx = self._waypoint_idx
            total = len(self._waypoints)

        if idx >= total:
            self.get_logger().info('Patrol complete. All waypoints visited.')
            with self._lock:
                self._state = self._S_DONE
            return

        wx, wy = self._waypoints[idx]
        with self._lock:
            self._waypoint_idx += 1
            self._state = self._S_PATROLLING

        self.get_logger().info(
            f'Patrol waypoint {idx + 1}/{total}  → ({wx:.2f}, {wy:.2f})'
        )
        pose = make_pose('map', self.get_clock().now().to_msg(), wx, wy)

        def _done(success):
            if not success:
                # Either cancelled (detection interrupt) or Nav2 failure — advance anyway
                pass
            self._advance()

        self._send_nav_goal(pose, _done)

    def _handle_next(self):
        with self._lock:
            if not self._pending:
                # Race: another thread already handled it
                self._state = self._S_PATROLLING
                self._advance()
                return
            det = self._pending.pop(0)
            self._state = self._S_HANDLING

        self.get_logger().info(
            f'Heading to {det["type"]} id={det["id"]}  '
            f'@ ({det["x"]:.2f}, {det["y"]:.2f})'
        )
        pose = self._approach_pose(det['x'], det['y'])

        def _arrived(success):
            if success:
                self._interact(det)
            else:
                self.get_logger().warn(
                    f'Could not reach {det["type"]} id={det["id"]} — skipping.'
                )

            with self._lock:
                if det['type'] == 'face':
                    self._visited_faces.add(det['id'])
                else:
                    self._visited_rings.add(det['id'])
                self._state = self._S_PATROLLING

            self._advance()

        self._send_nav_goal(pose, _arrived)

    # ── interactions ──────────────────────────────────────────────────────────

    def _interact(self, det: dict):
        if det['type'] == 'face':
            self._say(self._greet_cli, 'Hello!')
            self.get_logger().info(f'Greeted face id={det["id"]}.')
        else:
            color = det['color'] or 'unknown color'
            self._say(self._color_cli, color)
            self.get_logger().info(f'Announced ring id={det["id"]} color={color}.')

    def _say(self, client, text: str):
        if not client.service_is_ready():
            self.get_logger().warn(f'Speech service not ready, skipping: "{text}"')
            return
        req = Speech.Request()
        req.data = text
        # call_async — fire and forget (speech is non-blocking for the state machine)
        client.call_async(req)


# ── entry point ───────────────────────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = TaskExecutor()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
