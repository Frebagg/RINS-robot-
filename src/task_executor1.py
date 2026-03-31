#!/usr/bin/env python3
"""
Parameters
----------
n_waypoints    : int   (default 25)   number of random patrol waypoints
approach_dist  : float (default 0.8)  stop this far from a detected object (m)
min_wall_dist  : float (default 0.5)  minimum clearance from walls for sampled points (m)
patrol_waypoints : float[] (optional) explicit flat [x0,y0, x1,y1, ...] override
"""

import math
import random
import threading
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose, Spin
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import LaserScan
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy

import tf2_ros
from tf2_ros import TransformException

from rins_robot.msg import FaceCoords, RingCoords
from rins_robot.srv import Speech


# ── helpers ────────────────────────────────────────────────────────────────────

def yaw_to_quat(yaw: float):
    return 0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)


def make_pose(frame: str, stamp, x: float, y: float, yaw: float = 0.0) -> PoseStamped:
    ps = PoseStamped()
    ps.header.frame_id = frame
    ps.header.stamp = stamp
    ps.pose.position.x = x
    ps.pose.position.y = y
    qx, qy, qz, qw = yaw_to_quat(yaw)
    ps.pose.orientation.x = qx
    ps.pose.orientation.y = qy
    ps.pose.orientation.z = qz
    ps.pose.orientation.w = qw
    return ps


# ── waypoint node ──────────────────────────────────────────────────────────────

class WaypointNode:
    __slots__ = ('x', 'y', 'spin_after', 'dtype', 'det_id', 'color')

    def __init__(self, x, y, spin_after=True, dtype=None, det_id=None, color=''):
        self.x          = x
        self.y          = y
        self.spin_after = spin_after  
        self.dtype      = dtype      
        self.det_id     = det_id     
        self.color      = color       


# ── main node ──────────────────────────────────────────────────────────────────

class TaskExecutor(Node):

    _IDLE       = 'IDLE'
    _PATROLLING = 'PATROLLING'
    _HANDLING   = 'HANDLING'
    _SPINNING   = 'SPINNING'
    _DONE       = 'DONE'

    _DUP_DIST = 1.0

    def __init__(self):
        super().__init__('task_executor')

        # ── parameters ─────────────────────────────────────────────────────────
        self.declare_parameter('n_waypoints',        25)
        self.declare_parameter('approach_dist',      0.8)
        self.declare_parameter('min_wall_dist',      0.5)
        self.declare_parameter('patrol_waypoints',   [0.0])
        
        self.declare_parameter('area_spin_gain',     3.0) # Ce hocemo uporabljat razliko v povrsini kot pogoj za vrtenje

        self._n_wp          = self.get_parameter('n_waypoints').value
        self._approach      = self.get_parameter('approach_dist').value
        self._min_wall      = self.get_parameter('min_wall_dist').value
        self._area_spin_gain = self.get_parameter('area_spin_gain').value

        self._cbg = ReentrantCallbackGroup()

        # ── action clients ──────────────────────────────────────────────────────
        self._nav  = ActionClient(self, NavigateToPose, 'navigate_to_pose',
                                   callback_group=self._cbg)
        self._spin_client = ActionClient(self, Spin, 'spin',
                                          callback_group=self._cbg)

        self._greet_cli = self.create_client(Speech, '/greet_service',
                                              callback_group=self._cbg)
        self._color_cli = self.create_client(Speech, '/sayColor_service',
                                              callback_group=self._cbg)

        map_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(OccupancyGrid, '/map',
                                  self._on_map, map_qos,
                                  callback_group=self._cbg)
        self.create_subscription(FaceCoords, '/face_coords',
                                  self._on_faces, 10,
                                  callback_group=self._cbg)
        self.create_subscription(RingCoords, '/ring_coords',
                                  self._on_rings, 10,
                                  callback_group=self._cbg)
        self.create_subscription(LaserScan, '/scan_filtered',
                                  self._on_scan, 10,
                                  callback_group=self._cbg)

        self._tf_buf      = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buf, self)

        self._lock          = threading.Lock()
        self._state         = self._IDLE
        self._wq: deque[WaypointNode] = deque()
        self._current: Optional[WaypointNode] = None
        self._nav_handle    = None
        self._map_done      = False        

        self._visited_faces: set = set()
        self._visited_rings: set = set()

        # Stvari za avtomatsko vrtenje (ne uporabljamo rn)
        self._last_spin_area: float = 0.0  
        self._area_spin_pending: bool = False

        self.get_logger().info(
            f'TaskExecutor ready — waiting for /map  '
            f'(n_waypoints={self._n_wp}, approach_dist={self._approach} m)'
        )

    # ══════════════════════════════════════════════════════════════════════════
    # Map
    # ══════════════════════════════════════════════════════════════════════════

    def _on_map(self, msg: OccupancyGrid):
        with self._lock:
            if self._map_done:
                return
            self._map_done = True

        self.get_logger().info('Map received.')
        self._build_route(msg)

        self.get_logger().info('Waiting for Nav2…')
        self._nav.wait_for_server()
        self.get_logger().info('Nav2 ready — starting patrol.')
        self._advance()

    def _build_route(self, msg: OccupancyGrid):
        flat = (self.get_parameter('patrol_waypoints')
                    .get_parameter_value().double_array_value)
        if len(flat) >= 2:
            pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
            for x, y in pts:
                self._wq.append(WaypointNode(x, y, spin_after=True))
            self.get_logger().info(f'Using {len(pts)} waypoints.')
            return

        pts = self._sample_free(msg, self._n_wp)
        for x, y in pts:
            self._wq.append(WaypointNode(x, y, spin_after=False))
        self.get_logger().info(f'Sampled {len(pts)} random waypoints.')

    def _sample_free(self, msg: OccupancyGrid, n: int):
        res  = msg.info.resolution
        ox   = msg.info.origin.position.x
        oy   = msg.info.origin.position.y
        w    = msg.info.width
        h    = msg.info.height
        data = msg.data
        pad  = max(1, int(self._min_wall / res))

        cols_g = max(1, int(math.ceil(math.sqrt(n))))
        rows_g = max(1, int(math.ceil(n / cols_g)))
        cell_w = max(1, (w - 2 * pad) / cols_g)
        cell_h = max(1, (h - 2 * pad) / rows_g)

        buckets: dict = {}
        for r in range(pad, h - pad):
            for c in range(pad, w - pad):
                if data[r * w + c] != 0:
                    continue
                if not all(data[(r + dr) * w + (c + dc)] == 0
                           for dr in range(-pad, pad + 1)
                           for dc in range(-pad, pad + 1)):
                    continue
                gi = min(int((r - pad) / cell_h), rows_g - 1)
                gj = min(int((c - pad) / cell_w), cols_g - 1)
                buckets.setdefault((gi, gj), []).append(
                    (ox + (c + 0.5) * res, oy + (r + 0.5) * res)
                )

        if not buckets:
            self.get_logger().warn('No free cells found')
            return [(0.0, 0.0)]

        pts = [random.choice(v) for v in buckets.values()]
        random.shuffle(pts)
        return pts[:n]

    # ══════════════════════════════════════════════════════════════════════════
    # Scan area
    # ══════════════════════════════════════════════════════════════════════════

    # mogoce za poznejso uporabo zdaj ne rabimo
    def _on_scan(self, msg: LaserScan):
        area = self._scan_polygon_area(msg)
        with self._lock:
            if self._state not in (self._PATROLLING,):
                self._last_spin_area = area
                return
            gain = area - self._last_spin_area
            if gain > self._area_spin_gain and not self._area_spin_pending:
                self._area_spin_pending = True
                self.get_logger().info(
                    f'Room entry detected — visible area grew by {gain:.1f} m² '
                    f'({self._last_spin_area:.1f} → {area:.1f}). Queueing spin.'
                )
                spin_node = WaypointNode(0.0, 0.0, spin_after=False, dtype='__spin__')
                self._wq.appendleft(spin_node)
                if self._nav_handle is not None:
                    if self._current and self._current.dtype is None:
                        self._wq.insert(1, self._current)
                        self._current = None
                    self._nav_handle.cancel_goal_async()
                self._last_spin_area = area

    @staticmethod
    def _scan_polygon_area(msg: LaserScan) -> float:
        pts = []
        for i, r in enumerate(msg.ranges):
            if msg.range_min < r < msg.range_max:
                a = msg.angle_min + i * msg.angle_increment
                pts.append((r * math.cos(a), r * math.sin(a)))
        if len(pts) < 3:
            return 0.0
        area = 0.0
        n = len(pts)
        for i in range(n):
            j = (i + 1) % n
            area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
        return abs(area) * 0.5

    # ══════════════════════════════════════════════════════════════════════════
    # Detection callbacks
    # ══════════════════════════════════════════════════════════════════════════

    def _on_faces(self, msg: FaceCoords):
        for pt, fid in zip(msg.points, msg.ids):
            with self._lock:
                if fid in self._visited_faces:
                    continue
                if self._already_queued(pt.x, pt.y):
                    continue
                self.get_logger().info(
                    f'New face id={fid} @ ({pt.x:.2f}, {pt.y:.2f}) — inserting'
                )
                node = WaypointNode(pt.x, pt.y, spin_after=False,
                                    dtype='face', det_id=fid)
                self._insert_detection(node)

    def _on_rings(self, msg: RingCoords):
        for pt, rid, color in zip(msg.points, msg.ids, msg.colors):
            with self._lock:
                if rid in self._visited_rings:
                    continue
                if self._already_queued(pt.x, pt.y):
                    continue
                self.get_logger().info(
                    f'New ring id={rid} color={color} @ ({pt.x:.2f}, {pt.y:.2f}) — inserting'
                )
                node = WaypointNode(pt.x, pt.y, spin_after=False,
                                    dtype='ring', det_id=rid, color=color)
                self._insert_detection(node)

    def _already_queued(self, x, y):
        def close(n):
            return n.dtype is not None and math.hypot(n.x - x, n.y - y) < self._DUP_DIST
        if self._current and close(self._current):
            return True
        return any(close(n) for n in self._wq)

    def _insert_detection(self, node: WaypointNode):

        self._wq.appendleft(node)

        if self._state == self._PATROLLING and self._nav_handle is not None:
            if self._current is not None and self._current.dtype is None:
                self._wq.insert(1, self._current)
                self._current = None
            self._nav_handle.cancel_goal_async()

    # ══════════════════════════════════════════════════════════════════════════
    # Control flow
    # ══════════════════════════════════════════════════════════════════════════

    def _advance(self):
        with self._lock:
            if self._state == self._DONE:
                return
            if not self._wq:
                self._state = self._DONE
                self.get_logger().info('All waypoints visited — patrol complete!')
                return

            self._current = self._wq.popleft()
            n = self._current
            self._state = self._HANDLING if n.dtype else self._PATROLLING

        # Spin-in-place node: no navigation, just do the 360° immediately
        if n.dtype == '__spin__':
            self.get_logger().info('[SPINNING] Area-triggered 360° spin')
            self._do_spin()
            return

        self.get_logger().info(
            f'[{self._state}] → ({n.x:.2f}, {n.y:.2f})'
            + (f'  [{n.dtype} id={n.det_id}]' if n.dtype else '')
        )

        pose = (self._approach_pose(n.x, n.y) if n.dtype
                else make_pose('map', self.get_clock().now().to_msg(), n.x, n.y))

        self._send_nav(pose, self._on_nav_done)

    def _on_nav_done(self, status: int):
        with self._lock:
            n = self._current
            self._nav_handle = None

        if status == GoalStatus.STATUS_CANCELED:
            self._advance()
            return

        if status != GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().warn(f'Nav failed (status={status}) — skipping.')
            if n and n.dtype:
                with self._lock:
                    if n.dtype == 'face':
                        self._visited_faces.add(n.det_id)
                    else:
                        self._visited_rings.add(n.det_id)
            self._advance()
            return

        if n and n.dtype == 'face':
            self._say(self._greet_cli, 'Hello!')
            with self._lock:
                self._visited_faces.add(n.det_id)
                done = self._check_task_complete()
            if done:
                return
            self._advance()

        elif n and n.dtype == 'ring':
            self._say(self._color_cli, n.color or 'unknown')
            with self._lock:
                self._visited_rings.add(n.det_id)
                done = self._check_task_complete()
            if done:
                return
            self._advance()

        elif n and n.spin_after:
            self._do_spin()

        else:
            self._advance()

    def _check_task_complete(self) -> bool:
        """Return True (and stop) if all targets have been found. Lock held."""
        if len(self._visited_faces) >= 3 and len(self._visited_rings) >= 2:
            self._state = self._DONE
            self._wq.clear()
            self.get_logger().info(
                'Task complete! Found 3 faces and 2 rings — stopping patrol.'
            )
            return True
        self.get_logger().info(
            f'Progress: {len(self._visited_faces)}/3 faces, '
            f'{len(self._visited_rings)}/2 rings'
        )
        return False

    # ── 360 no scope ──────────────────────────────────────────────────────────────

    def _do_spin(self):
        with self._lock:
            self._state = self._SPINNING

        goal = Spin.Goal()
        goal.target_yaw = 2.0 * math.pi

        if not self._spin_client.wait_for_server(timeout_sec=3.0):
            self.get_logger().warn('Spin server not available — skipping spin.')
            self._advance()
            return

        fut = self._spin_client.send_goal_async(goal)
        fut.add_done_callback(self._on_spin_accepted)

    def _on_spin_accepted(self, fut):
        handle = fut.result()
        if not handle.accepted:
            self.get_logger().warn('Spin goal rejected — skipping.')
            self._advance()
            return
        def _spin_done(_):
            with self._lock:
                self._area_spin_pending = False
            self._advance()
        handle.get_result_async().add_done_callback(_spin_done)

    # ── navigation wrapper ─────────────────────────────────────────────────────

    def _send_nav(self, pose: PoseStamped, on_done):
        goal = NavigateToPose.Goal()
        goal.pose = pose
        fut = self._nav.send_goal_async(goal)

        def _accepted(f):
            handle = f.result()
            if not handle.accepted:
                self.get_logger().warn('Nav goal rejected.')
                on_done(GoalStatus.STATUS_ABORTED)
                return
            with self._lock:
                self._nav_handle = handle
            handle.get_result_async().add_done_callback(
                lambda rf: on_done(rf.result().status)
            )

        fut.add_done_callback(_accepted)

    # ── approach pose ──────────────────────────────────────────────────────────

    def _robot_xy(self):
        try:
            t = self._tf_buf.lookup_transform('map', 'base_link', rclpy.time.Time())
            return t.transform.translation.x, t.transform.translation.y
        except TransformException:
            return 0.0, 0.0

    def _approach_pose(self, tx: float, ty: float) -> PoseStamped:
        rx, ry = self._robot_xy()
        dx, dy = tx - rx, ty - ry
        dist = max(math.hypot(dx, dy), 1e-3)
        ux, uy = -dx / dist, -dy / dist
        ax = tx + ux * self._approach
        ay = ty + uy * self._approach
        yaw = math.atan2(dy, dx)
        return make_pose('map', self.get_clock().now().to_msg(), ax, ay, yaw)

    # ── speech ─────────────────────────────────────────────────────────────────

    def _say(self, client, text: str):
        if not client.service_is_ready():
            self.get_logger().warn(f'Speech service not ready — skipping: "{text}"')
            return
        req = Speech.Request()
        req.data = text
        client.call_async(req)  


# ── entry point ────────────────────────────────────────────────────────────────

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
