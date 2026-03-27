#!/usr/bin/env python3


import rclpy
import math
import os
import yaml
import numpy as np

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional

from geometry_msgs.msg import PoseStamped, Point, Quaternion

from robot_commander import RobotCommander, TaskResult
from rins_robot.msg import FaceCoords, RingCoords


# ---------------------------------------------------------------------------
# Waypoint shit
# ---------------------------------------------------------------------------

class WaypointKind(Enum):
    NORMAL = auto()  
    CORNER = auto()   
    FACE   = auto()  
    RING   = auto()  


@dataclass
class Waypoint:
    pose:       PoseStamped
    kind:       WaypointKind
    color:      Optional[str]        = None   
    confidence: float                = 0.0
    next:       Optional['Waypoint'] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# Map shit
# ---------------------------------------------------------------------------

# Nalozimo mapo iz direktorija
def load_pgm(yaml_path: str):

    with open(yaml_path, 'r') as f:
        info = yaml.safe_load(f)

    image_path = info['image']
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(yaml_path), image_path)

    from PIL import Image as PILImage
    img = np.array(PILImage.open(image_path).convert('L'))
    info['height'] = img.shape[0]
    info['width']  = img.shape[1]
    return img, info


def medial_axis_waypoints(pgm: np.ndarray,
                          resolution: float,
                          spacing: float = 0.8) -> list:
    """
    Skeletonise the free-space region and sample waypoints at *spacing* metres.

    Returns a list of (col, row) pixel coordinate tuples.
    """
    from skimage.morphology import skeletonize

    free = pgm > 205
    skel = skeletonize(free)

    rows, cols = np.where(skel)
    if len(rows) == 0:
        return []

    points_px = list(zip(cols.tolist(), rows.tolist()))   # (col, row)
    ordered   = _order_by_nn(points_px)

    step = max(1, int(round(spacing / resolution)))
    return ordered[::step]


def _order_by_nn(points: list) -> list:
    if not points:
        return []
    pts     = np.array(points, dtype=float)
    visited = np.zeros(len(pts), dtype=bool)
    order   = [0]
    visited[0] = True
    for _ in range(len(pts) - 1):
        last  = pts[order[-1]]
        dists = np.linalg.norm(pts - last, axis=1)
        dists[visited] = np.inf
        nearest = int(np.argmin(dists))
        order.append(nearest)
        visited[nearest] = True
    return [points[i] for i in order]

# Zgenerira zigzag pot
# To be determined ce bo final ce nebo predolgo trajalo
def generate_zigzag(waypoints: list,
                    pgm: np.ndarray,
                    amplitude: float = 0.25) -> list:

    if len(waypoints) < 3:
        return waypoints

    h, w   = pgm.shape[:2]
    result = []

    for i, (cx, cy) in enumerate(waypoints):
        if 0 < i < len(waypoints) - 1:
            px, py = waypoints[i - 1]
            nx, ny = waypoints[i + 1]
            tx, ty = nx - px, ny - py
            length = math.hypot(tx, ty)
            if length > 0:
                perp_x = -ty / length   
                perp_y =  tx / length
                sign   = 1 if i % 2 == 0 else -1
                cx2 = max(0.0, min(w - 1.0, cx + sign * perp_x * amplitude))
                cy2 = max(0.0, min(h - 1.0, cy + sign * perp_y * amplitude))
                result.append((cx2, cy2))
                continue
        result.append((float(cx), float(cy)))

    return result

# Ce je waypoint kje ob robu se moremo obrniti za 360 da preverimo vse
def find_corners(waypoints: list, angle_threshold: float = 30.0) -> set:

    corners    = set()
    thresh_rad = math.radians(angle_threshold)

    for i in range(1, len(waypoints) - 1):
        x0, y0 = waypoints[i - 1]
        x1, y1 = waypoints[i]
        x2, y2 = waypoints[i + 1]

        v1   = (x1 - x0, y1 - y0)
        v2   = (x2 - x1, y2 - y1)
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            continue

        dot   = v1[0] * v2[0] + v1[1] * v2[1]
        cos_a = max(-1.0, min(1.0, dot / (mag1 * mag2)))
        if math.acos(cos_a) > thresh_rad:
            corners.add(i)

    return corners


def _px_to_metric(col: float, row: float, info: dict):
    res    = info['resolution']
    ox, oy = info['origin'][0], info['origin'][1]
    height = info['height']

    x_m = ox + col * res
    y_m = oy + (height - 1 - row) * res
    return x_m, y_m


def _yaw_to_quat(yaw: float) -> Quaternion:
    return Quaternion(x=0.0, y=0.0,
                      z=math.sin(yaw * 0.5),
                      w=math.cos(yaw * 0.5))


def build_linked_list(waypoints: list,
                      corner_indices: set,
                      info: dict) -> Optional[Waypoint]:

    head = None
    tail = None

    for i, (col, row) in enumerate(waypoints):
        x_m, y_m = _px_to_metric(col, row, info)

        if i < len(waypoints) - 1:
            nc, nr     = waypoints[i + 1]
            nx_m, ny_m = _px_to_metric(nc, nr, info)
            yaw = math.atan2(ny_m - y_m, nx_m - x_m)
        else:
            yaw = 0.0

        pose                     = PoseStamped()
        pose.header.frame_id     = 'map'
        pose.pose.position.x     = x_m
        pose.pose.position.y     = y_m
        pose.pose.position.z     = 0.0
        pose.pose.orientation    = _yaw_to_quat(yaw)

        kind = WaypointKind.CORNER if i in corner_indices else WaypointKind.NORMAL
        node = Waypoint(pose=pose, kind=kind)

        if head is None:
            head = node
            tail = node
        else:
            tail.next = node
            tail = node         

    return head


# ---------------------------------------------------------------------------
# Navigator node
# ---------------------------------------------------------------------------

class RobotNavigator(RobotCommander):  


    MAP_YAML    = '/maps/task1.yaml'
    SPACING_M   = 0.8    
    AMPLITUDE_M = 0.25   
    CORNER_DEG  = 30.0   
    DEDUP_DIST  = 0.5   

    def __init__(self):
        super().__init__('robot_navigator')

        self.head: Optional[Waypoint] = None

        self.detected_head: Optional[Waypoint] = None
        self.detected_tail: Optional[Waypoint] = None

        self.face_sub = self.create_subscription(  
            FaceCoords,
            '/face_coords',
            self._face_callback,
            10)

        self.ring_sub = self.create_subscription(
            RingCoords,
            '/ring_coords',
            self._ring_callback,
            10)

        self.info('RobotNavigator initialised.')

    # ------------------------------------------------------------------
    # Detection callbacks
    # ------------------------------------------------------------------

    def _face_callback(self, msg: FaceCoords):
        for point in msg.points:     
            self._add_detection(point, WaypointKind.FACE)

    def _ring_callback(self, msg: RingCoords):
        for point, color in zip(msg.points, msg.colors):
            self._add_detection(point, WaypointKind.RING, color=color)

    def _add_detection(self, point: Point,
                       kind: WaypointKind,
                       color: Optional[str] = None):

        node = self.detected_head
        while node is not None:
            if node.kind == kind:
                dx = node.pose.pose.position.x - point.x
                dy = node.pose.pose.position.y - point.y
                if math.hypot(dx, dy) < self.DEDUP_DIST:
                    # Update position as a running average
                    node.pose.pose.position.x = (node.pose.pose.position.x + point.x) / 2.0
                    node.pose.pose.position.y = (node.pose.pose.position.y + point.y) / 2.0
                    return
            node = node.next

        pose                  = PoseStamped()
        pose.header.frame_id  = 'map'
        pose.pose.position.x  = point.x
        pose.pose.position.y  = point.y
        pose.pose.position.z  = point.z

        new_wp = Waypoint(pose=pose, kind=kind, color=color)

        if self.detected_head is None:
            self.detected_head = new_wp
            self.detected_tail = new_wp
        else:
            self.detected_tail.next = new_wp
            self.detected_tail = new_wp

        label = kind.name + (f' color={color}' if color else '')
        self.info(f'[Detection] New {label} at '
                  f'({point.x:.2f}, {point.y:.2f}, {point.z:.2f})')

    # ------------------------------------------------------------------
    # Entry point
    # ------------------------------------------------------------------

    def run(self):
        self.waitUntilNav2Active()

        pgm, info    = load_pgm(self.MAP_YAML)
        amplitude_px = self.AMPLITUDE_M / info['resolution']

        wps       = medial_axis_waypoints(pgm, info['resolution'], self.SPACING_M)
        wps       = generate_zigzag(wps, pgm, amplitude=amplitude_px)
        corners   = find_corners(wps, angle_threshold=self.CORNER_DEG)
        self.head = build_linked_list(wps, corners, info)

        self.execute_coverage()   
    
def main():
    rclpy.init()
    node = RobotNavigator()   # ← was "RoboNavigator()" (typo fixed)
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
