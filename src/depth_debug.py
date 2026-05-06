#!/usr/bin/env python3
"""
Ring detector for TurtleBot 4 with Orbbec Gemini 335L (RGB-D).

Pipeline
--------
1. Subscribe to synchronized RGB image + organized PointCloud2.
2. Per color (red, green, blue, black) build an HSV mask, find contours,
   fit ellipses, keep ring-shaped ones (hollow interior check).
3. NMS across all candidates.
4. For each surviving ellipse: walk the LOWER rim, look up XYZ from the
   organized cloud (gated by the color mask so the hanger arm and
   background don't pollute samples).
5. Fit a 3D circle to the rim points: SVD plane fit -> 2D circle in plane
   via Kasa's linear least squares. Reject if radius is implausible or
   residual is too large.
6. Build a Pose with the circle's center and the plane's normal as the
   approach direction (+X axis). Transform to map frame.
7. Per-color nearest-neighbor tracking; publish PoseArray + MarkerArray
   (sphere + arrow showing approach direction + text label).

Run
---
    ros2 run <your_pkg> ring_detector
or directly:
    python3 ring_detector.py --ros-args -r __ns:=/rins
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, Pose, PoseArray, PoseStamped, Quaternion
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
import sensor_msgs_py.point_cloud2 as pc2
from std_msgs.msg import ColorRGBA, Header
from tf2_ros import Buffer, TransformException, TransformListener
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped
from visualization_msgs.msg import Marker, MarkerArray

# ---------------------------------------------------------------------------
# Topic names for the Gemini 335L camera on the real TurtleBot4.
# Verify with `ros2 topic list` on the real robot if needed.
# ---------------------------------------------------------------------------
RGB_TOPIC = "/gemini/color/image_raw"
DEPTH_TOPIC = "/gemini/depth/image_raw"
PC_TOPIC = "/gemini/depth/points"
INFO_TOPIC = "/gemini/color/camera_info"
CAMERA_FRAME = "gemini_color_optical_frame"
TARGET_FRAME = "map"

# ---------------------------------------------------------------------------
# HSV thresholds. These are starting points -- tune on the real robot using
# the `debug/mask_<color>` topic published by this node.
# H in OpenCV is 0..179, S/V are 0..255.
# ---------------------------------------------------------------------------
COLOR_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    # Red wraps around the hue circle, so it needs two ranges.
    "red": [
        (np.array([0, 110, 70]),   np.array([10, 255, 255])),
        (np.array([170, 110, 70]), np.array([179, 255, 255])),
    ],
    "green": [
        (np.array([40, 80, 50]),   np.array([85, 255, 255])),
    ],
    "blue": [
        (np.array([95, 120, 50]),  np.array([130, 255, 255])),
    ],
    # Black is low V; we ignore hue. Tight S upper bound avoids picking up
    # dark-saturated colors like deep red.
    "black": [
        (np.array([0, 0, 0]),      np.array([179, 80, 60])),
    ],
}

# RGBA used for RViz markers, matched to the ring color.
MARKER_RGBA: dict[str, tuple[float, float, float, float]] = {
    "red":   (1.0, 0.1, 0.1, 0.9),
    "green": (0.1, 0.9, 0.1, 0.9),
    "blue":  (0.1, 0.3, 1.0, 0.9),
    "black": (0.05, 0.05, 0.05, 0.95),
}

# ---------------------------------------------------------------------------
# Detection / acceptance thresholds.
# ---------------------------------------------------------------------------
MIN_CONTOUR_AREA_PX = 250         # ignore tiny blobs
MAX_AXIS_RATIO = 4.0              # rings viewed from the side -- still ellipse-like
MIN_RING_AXIS_PX = 15             # minimum minor axis of the ellipse
RING_HOLLOWNESS_THRESH = 0.45     # interior fill ratio must be BELOW this
DEPTH_RIM_SAMPLES = 24            # how many points around the rim to sample
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 4.0
NMS_IOU_THRESH = 0.30             # suppress overlapping ellipses above this IoU
MIN_RIM_POINTS_3D = 6             # minimum valid 3D rim points to attempt fit
MAX_CIRCLE_FIT_RESIDUAL_M = 0.04  # mean point-to-circle distance, reject if larger
RING_RADIUS_MIN_M = 0.05          # plausible physical ring radius bounds
RING_RADIUS_MAX_M = 0.25

# Cluster newly-detected rings with previously-detected ones if they are
# within this distance (in map frame). Same color only.
ASSOCIATION_RADIUS_M = 0.35
# If a tracked ring hasn't been seen for this long, drop it.
TRACK_TIMEOUT_S = 60.0
# Minimum number of detections before we trust a ring enough to publish it.
MIN_DETECTIONS_FOR_PUBLISH = 3


@dataclass
class TrackedRing:
    color: str
    position: np.ndarray                # mean position in map frame, shape (3,)
    orientation: Quaternion = field(default_factory=lambda: Quaternion(x=0.0, y=0.0, z=0.0, w=1.0))
    detections: int = 1
    last_seen_stamp: float = 0.0
    samples: list[np.ndarray] = field(default_factory=list)

    def update(self, new_pos: np.ndarray, stamp: float,
               new_orientation: Optional[Quaternion] = None) -> None:
        # Running mean -- cheap, robust enough, and behaves well with N -> inf.
        self.samples.append(new_pos)
        if len(self.samples) > 50:
            self.samples.pop(0)
        self.position = np.mean(np.stack(self.samples, axis=0), axis=0)
        # For orientation we just take the latest -- proper averaging of
        # quaternions needs SLERP/Markley's method and isn't worth it here,
        # since the ring isn't moving and consecutive estimates should agree.
        if new_orientation is not None:
            self.orientation = new_orientation
        self.detections += 1
        self.last_seen_stamp = stamp


class RingDetector(Node):
    def __init__(self) -> None:
        super().__init__("ring_detector")

        # Sensor QoS: cameras typically publish BEST_EFFORT. Using a matching
        # profile prevents the subscriber from silently dropping every frame.
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Camera intrinsics -- filled in by the CameraInfo callback.
        self.fx: Optional[float] = None
        self.fy: Optional[float] = None
        self.cx: Optional[float] = None
        self.cy: Optional[float] = None

        # CameraInfo is no longer required for back-projection (the point
        # cloud has XYZ baked in), but we keep it because it's lightweight
        # and useful if you ever want to re-add a fallback path.
        self.create_subscription(
            CameraInfo, INFO_TOPIC, self._on_camera_info, sensor_qos
        )

        # Time-synchronized RGB + organized point cloud.
        # The cloud is "organized": its width/height match the camera image,
        # so cloud[v, u] gives the 3D point for pixel (u, v) -- this is what
        # makes the hybrid approach work cleanly.
        self.rgb_sub = Subscriber(self, Image, RGB_TOPIC, qos_profile=sensor_qos)
        self.pc_sub = Subscriber(self, PointCloud2, PC_TOPIC, qos_profile=sensor_qos)
        self.sync = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.pc_sub],
            queue_size=10,
            slop=0.15,  # cloud generation adds latency; allow a bit more slop
        )
        self.sync.registerCallback(self._on_frames)

        # Publishers.
        self.poses_pub = self.create_publisher(PoseArray, "rings/poses", 10)
        self.markers_pub = self.create_publisher(MarkerArray, "rings/markers", 10)
        self.debug_pub = self.create_publisher(Image, "rings/debug_image", 1)
        self.mask_pubs = {
            color: self.create_publisher(Image, f"rings/debug/mask_{color}", 1)
            for color in COLOR_RANGES
        }

        # tracks[color] -> list[TrackedRing]
        self.tracks: dict[str, list[TrackedRing]] = {c: [] for c in COLOR_RANGES}

        # Periodic publish of stable rings -- decoupled from frame rate so RViz
        # keeps showing markers even if no new frames arrive.
        self.create_timer(0.5, self._publish_state)

        self.get_logger().info(
            f"Ring detector up. RGB={RGB_TOPIC}  DEPTH={DEPTH_TOPIC}  "
            f"camera_frame={CAMERA_FRAME}  target_frame={TARGET_FRAME}"
        )

    # -----------------------------------------------------------------------
    # Subscribers
    # -----------------------------------------------------------------------
    def _on_camera_info(self, msg: CameraInfo) -> None:
        # K = [fx 0 cx; 0 fy cy; 0 0 1]
        if self.fx is None:
            self.get_logger().info("Got CameraInfo, locking intrinsics.")
        self.fx = float(msg.k[0])
        self.fy = float(msg.k[4])
        self.cx = float(msg.k[2])
        self.cy = float(msg.k[5])

    def _on_frames(self, rgb_msg: Image, pc_msg: PointCloud2) -> None:
        try:
            rgb = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"RGB conversion failed: {exc}")
            return

        # The cloud must be ORGANIZED (height > 1) for our pixel-based lookup
        # to work. If the driver publishes an unorganized cloud (height==1),
        # we'd need to project each point through K to find its pixel -- much
        # more expensive. The Gemini driver publishes organized clouds, but
        # warn loudly if that ever changes.
        if pc_msg.height <= 1:
            self.get_logger().warn(
                "PointCloud2 is unorganized (height=1). Pixel-indexed XYZ "
                "lookup won't work. Check the camera driver config.",
                throttle_duration_sec=5.0,
            )
            return

        # Sanity: cloud dimensions should match the image. If they're different
        # (e.g. depth at 640x480, color at 1280x720), we'd need to rescale the
        # ellipse coordinates. Easier to enforce a matched config.
        if (pc_msg.width, pc_msg.height) != (rgb.shape[1], rgb.shape[0]):
            self.get_logger().warn(
                f"Cloud size {pc_msg.width}x{pc_msg.height} != image size "
                f"{rgb.shape[1]}x{rgb.shape[0]}. Configure the driver to "
                "align them, or add rescaling here.",
                throttle_duration_sec=5.0,
            )
            return

        # Convert PointCloud2 to a (H, W, 3) numpy array of XYZ in camera frame.
        # NaN entries mark invalid pixels (no depth return). We keep them as
        # NaN so downstream code can filter them with np.isfinite.
        xyz = self._cloud_to_xyz_image(pc_msg)

        debug = rgb.copy()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
        h, w = rgb.shape[:2]

        # 1) Collect candidates across all colors. NMS later resolves overlaps.
        candidates: list[dict] = []
        for color, ranges in COLOR_RANGES.items():
            mask = self._build_mask(hsv, ranges)
            self._publish_mask(color, mask, rgb_msg.header)

            for ellipse, hollowness in self._find_ring_ellipses(mask):
                (_, _), (axa, axb), _ = ellipse
                area = math.pi * (axa / 2.0) * (axb / 2.0)
                score = hollowness * math.sqrt(max(area, 1.0))
                candidates.append({
                    "color": color,
                    "ellipse": ellipse,
                    "hollowness": hollowness,
                    "score": score,
                    "mask": mask,
                })

        # 2) NMS across colors.
        kept = self._nms_ellipses(candidates, (h, w), NMS_IOU_THRESH)

        # 3) For each survivor, fit a 3D circle to the rim points.
        for det in kept:
            fit = self._fit_ring_3d(det["ellipse"], xyz, det["mask"])
            if fit is None:
                continue
            center_cam, normal_cam, radius_m, residual_m = fit

            pose_cam = self._build_pose_from_center_normal(center_cam, normal_cam)
            pose_map = self._to_target_frame_pose(pose_cam, rgb_msg.header.stamp)
            if pose_map is None:
                continue

            pos_map = np.array([
                pose_map.position.x, pose_map.position.y, pose_map.position.z
            ], dtype=np.float64)
            self._associate_or_create(det["color"], pos_map, rgb_msg.header.stamp,
                                      orientation=pose_map.orientation)
            self._draw_detection(debug, det["ellipse"], det["color"],
                                 det["hollowness"], radius_m, residual_m)

        # Publish debug overlay.
        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = rgb_msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().warn(f"Debug publish failed: {exc}")

    # -----------------------------------------------------------------------
    # Color & shape detection
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_mask(hsv: np.ndarray, ranges: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask |= cv2.inRange(hsv, lo, hi)

        # Morphology: close small gaps in the rim, then open to kill speckle.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        return mask

    @staticmethod
    def _ellipse_mask(ellipse, shape: tuple[int, int]) -> np.ndarray:
        m = np.zeros(shape, dtype=np.uint8)
        cv2.ellipse(m, ellipse, 255, thickness=-1)
        return m

    @classmethod
    def _nms_ellipses(
        cls,
        candidates: list[dict],
        shape: tuple[int, int],
        iou_thresh: float,
    ) -> list[dict]:
        """
        Greedy NMS over ellipse candidates.

        IoU is computed exactly via filled-ellipse rasters -- ellipses don't
        have a closed-form IoU like axis-aligned boxes, and approximating with
        bounding boxes loses too much info for thin/rotated rings.
        Cost is fine: a handful of small uint8 ops per pair, and we only have
        a few candidates per frame in practice.
        """
        if not candidates:
            return []

        # Sort high-score first; greedy NMS keeps the best and suppresses overlaps.
        order = sorted(candidates, key=lambda d: d["score"], reverse=True)
        masks = [cls._ellipse_mask(d["ellipse"], shape) for d in order]
        areas = [int(np.count_nonzero(m)) for m in masks]

        kept: list[dict] = []
        suppressed = [False] * len(order)
        for i in range(len(order)):
            if suppressed[i]:
                continue
            kept.append(order[i])
            for j in range(i + 1, len(order)):
                if suppressed[j]:
                    continue
                inter = int(np.count_nonzero(cv2.bitwise_and(masks[i], masks[j])))
                union = areas[i] + areas[j] - inter
                if union <= 0:
                    continue
                if inter / union > iou_thresh:
                    suppressed[j] = True
        return kept

    def _find_ring_ellipses(self, mask: np.ndarray):
        """
        Yield (ellipse, hollowness) for contours that look like rings.

        The key trick: a ring's bounding ellipse should have a mostly EMPTY
        interior. We measure 'hollowness' as the fraction of the ellipse's
        interior that is masked. Filled disks fail this test.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        h, w = mask.shape

        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA_PX:
                continue
            if len(cnt) < 5:
                continue  # fitEllipse needs >=5 points

            try:
                ellipse = cv2.fitEllipse(cnt)
            except cv2.error:
                continue

            (_, _), (axis_a, axis_b), _ = ellipse
            minor = min(axis_a, axis_b)
            major = max(axis_a, axis_b)
            if minor < MIN_RING_AXIS_PX:
                continue
            if major / max(minor, 1e-3) > MAX_AXIS_RATIO:
                continue

            # Hollowness check: compare a filled-ellipse mask vs. the actual mask.
            filled = np.zeros_like(mask)
            cv2.ellipse(filled, ellipse, 255, thickness=-1)
            interior_area = float(np.count_nonzero(filled))
            if interior_area <= 0:
                continue
            overlap = float(np.count_nonzero(cv2.bitwise_and(filled, mask)))
            fill_ratio = overlap / interior_area
            if fill_ratio > RING_HOLLOWNESS_THRESH:
                # Looks like a filled disk, not a ring.
                continue

            # Hollowness here is "how empty the interior is" -- bigger == ringier.
            hollowness = 1.0 - fill_ratio
            yield ellipse, hollowness

    # -----------------------------------------------------------------------
    # 3D estimation (point-cloud based)
    # -----------------------------------------------------------------------
    @staticmethod
    def _cloud_to_xyz_image(pc_msg: PointCloud2) -> np.ndarray:
        """
        Convert a PointCloud2 to a (H, W, 3) float32 array of XYZ in camera frame.

        Why we don't just np.frombuffer the raw data: the cloud's point_step
        often includes padding (e.g. 32 bytes per point with x,y,z,rgb
        followed by reserved bytes). Going through sensor_msgs_py is robust
        to whatever layout the driver chose and handles big-endian, NaN
        sentinels, and missing-field cases consistently.
        """
        # read_points returns a structured array; we extract just XYZ.
        # skip_nans=False so the H*W shape is preserved -- we want NaN entries
        # to mark invalid pixels, not get silently dropped.
        pts = pc2.read_points(
            pc_msg, field_names=("x", "y", "z"), skip_nans=False, reshape_organized_cloud=True
        )
        # pts is a structured ndarray of shape (H, W); each entry has (x, y, z).
        # Stack into a regular (H, W, 3) float32 view.
        xyz = np.stack(
            [pts["x"], pts["y"], pts["z"]], axis=-1
        ).astype(np.float32, copy=False)
        return xyz

    def _fit_ring_3d(
        self,
        ellipse,
        xyz: np.ndarray,
        color_mask: np.ndarray,
    ) -> Optional[tuple[np.ndarray, np.ndarray, float, float]]:
        """
        Walk the ellipse rim, gather 3D rim points from the cloud (filtered
        by the color mask), and fit a 3D circle.

        Returns (center_cam, normal_cam, radius_m, residual_m), or None.
            center_cam:  (3,) ring center in camera frame
            normal_cam:  (3,) unit normal of the ring's plane
            radius_m:    fitted physical radius
            residual_m:  mean point-to-circle distance (lower = better fit)
        """
        (cx_px, cy_px), (axis_a, axis_b), angle_deg = ellipse
        h, w = xyz.shape[:2]
        a = axis_a / 2.0
        b = axis_b / 2.0
        ang = math.radians(angle_deg)
        cos_a, sin_a = math.cos(ang), math.sin(ang)

        rim_pts: list[np.ndarray] = []
        n_samples = DEPTH_RIM_SAMPLES * 2
        for i in range(n_samples):
            t = 2.0 * math.pi * i / n_samples
            x_local = a * math.cos(t)
            y_local = b * math.sin(t)
            u = int(round(cx_px + x_local * cos_a - y_local * sin_a))
            v = int(round(cy_px + x_local * sin_a + y_local * cos_a))
            if not (0 <= u < w and 0 <= v < h):
                continue
            # Lower-half preference -- the hanger arm contaminates the top.
            if v < cy_px + 0.15 * b:
                continue
            # Mask gate: only accept where the rim is actually colored.
            u0, u1 = max(0, u - 1), min(w, u + 2)
            v0, v1 = max(0, v - 1), min(h, v + 2)
            if not np.any(color_mask[v0:v1, u0:u1]):
                continue
            # Pull XYZ from the 3x3 patch and pick the point closest to the
            # camera that's in valid range. Closest, because if any pixel in
            # the patch saw the rim (foreground) and others saw what's behind
            # (background), the rim is the smaller Z value. This is the
            # equivalent of the masked-min trick on a depth image, but exact.
            patch = xyz[v0:v1, u0:u1].reshape(-1, 3)
            valid = np.isfinite(patch).all(axis=1)
            valid &= (patch[:, 2] > DEPTH_MIN_M) & (patch[:, 2] < DEPTH_MAX_M)
            if not np.any(valid):
                continue
            cand = patch[valid]
            # Closest by Z within the patch -- foreground wins ties.
            best_idx = int(np.argmin(cand[:, 2]))
            rim_pts.append(cand[best_idx])

        if len(rim_pts) < MIN_RIM_POINTS_3D:
            return None

        pts = np.stack(rim_pts, axis=0)  # (N, 3)

        # Fit a plane to the rim points (the ring lies in a plane).
        # Use SVD on the centered points -- the smallest singular vector is
        # the plane normal. Robust to outliers within reason; if you need more,
        # wrap this in RANSAC.
        centroid = pts.mean(axis=0)
        centered = pts - centroid
        # SVD: V[-1] is the direction of least variance == plane normal.
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        normal = vh[-1]
        normal = normal / (np.linalg.norm(normal) + 1e-9)

        # Project points onto the plane (2D coords in the plane) so we can
        # do a 2D circle fit -- much simpler than fitting in 3D directly.
        # Build an orthonormal basis (u_hat, v_hat) spanning the plane.
        # Pick u_hat as any vector orthogonal to the normal.
        if abs(normal[0]) < 0.9:
            u_hat = np.cross(normal, np.array([1.0, 0.0, 0.0]))
        else:
            u_hat = np.cross(normal, np.array([0.0, 1.0, 0.0]))
        u_hat = u_hat / (np.linalg.norm(u_hat) + 1e-9)
        v_hat = np.cross(normal, u_hat)

        pts_2d = np.stack(
            [centered @ u_hat, centered @ v_hat], axis=1
        )  # (N, 2)

        # 2D circle fit (Kasa method): solve the linear system
        #   2*xc*x + 2*yc*y + (R^2 - xc^2 - yc^2) = x^2 + y^2
        # Let c = R^2 - xc^2 - yc^2; the unknowns are (xc, yc, c).
        x = pts_2d[:, 0]
        y = pts_2d[:, 1]
        A = np.stack([2 * x, 2 * y, np.ones_like(x)], axis=1)  # (N, 3)
        rhs = x * x + y * y
        try:
            sol, *_ = np.linalg.lstsq(A, rhs, rcond=None)
        except np.linalg.LinAlgError:
            return None
        xc_2d, yc_2d, c = sol
        r2 = c + xc_2d * xc_2d + yc_2d * yc_2d
        if r2 <= 0:
            return None
        radius = float(math.sqrt(r2))

        # Reject implausible radii -- ring hoops are roughly known.
        if not (RING_RADIUS_MIN_M <= radius <= RING_RADIUS_MAX_M):
            return None

        # Residual: mean distance from each point to the fitted circle.
        # In-plane distance from each (x, y) to (xc, yc), minus the radius.
        dx = x - xc_2d
        dy = y - yc_2d
        radial = np.sqrt(dx * dx + dy * dy)
        residual = float(np.mean(np.abs(radial - radius)))
        if residual > MAX_CIRCLE_FIT_RESIDUAL_M:
            return None

        # Lift the 2D center back to 3D camera frame.
        center_cam = centroid + xc_2d * u_hat + yc_2d * v_hat

        return center_cam, normal, radius, residual

    @staticmethod
    def _build_pose_from_center_normal(
        center_cam: np.ndarray, normal_cam: np.ndarray
    ) -> PoseStamped:
        """
        Build a PoseStamped (in camera frame) where the orientation's +X axis
        points along the ring's normal.

        Convention choice: aligning +X with the normal means a robot looking at
        a "PoseStamped from this ring" knows which way is "out of the ring."
        That's the natural direction to approach from for grabbing or passing
        through it. If your nav stack expects a different convention (e.g. +Z
        is approach direction, common in manipulation), change the basis below.
        """
        # Pick a stable up vector that's not parallel to the normal so we can
        # build a full orthonormal frame. World up in camera frame would be
        # nicer but we don't have TF here -- just pick any non-parallel vec.
        up = np.array([0.0, -1.0, 0.0])  # camera "up" in optical frame
        if abs(np.dot(up, normal_cam)) > 0.95:
            up = np.array([1.0, 0.0, 0.0])

        x_axis = normal_cam / (np.linalg.norm(normal_cam) + 1e-9)
        z_axis = np.cross(x_axis, up)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
        y_axis = np.cross(z_axis, x_axis)

        # Rotation matrix [x y z] -> quaternion (Hamilton convention, w last
        # because that's what geometry_msgs uses).
        R = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are basis vecs
        q = RingDetector._mat_to_quat(R)

        ps = PoseStamped()
        ps.header.frame_id = CAMERA_FRAME
        ps.pose.position.x = float(center_cam[0])
        ps.pose.position.y = float(center_cam[1])
        ps.pose.position.z = float(center_cam[2])
        ps.pose.orientation.x = float(q[0])
        ps.pose.orientation.y = float(q[1])
        ps.pose.orientation.z = float(q[2])
        ps.pose.orientation.w = float(q[3])
        return ps

    @staticmethod
    def _mat_to_quat(R: np.ndarray) -> np.ndarray:
        """3x3 rotation matrix to quaternion (x, y, z, w). Shepperd's method."""
        m = R
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            w = 0.25 * s
            x = (m[2, 1] - m[1, 2]) / s
            y = (m[0, 2] - m[2, 0]) / s
            z = (m[1, 0] - m[0, 1]) / s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        return np.array([x, y, z, w], dtype=np.float64)

    def _to_target_frame_pose(self, pose_cam: PoseStamped, stamp) -> Optional[Pose]:
        """Transform a PoseStamped from camera frame into TARGET_FRAME."""
        try:
            tf = self.tf_buffer.lookup_transform(
                TARGET_FRAME,
                CAMERA_FRAME,
                stamp,
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f"TF {CAMERA_FRAME} -> {TARGET_FRAME} unavailable: {exc}",
                throttle_duration_sec=2.0,
            )
            return None

        # Transform position via PointStamped (we already have that machinery)
        # and orientation by composing quaternions.
        from tf2_geometry_msgs import do_transform_pose
        pose_cam.header.stamp = stamp
        try:
            out = do_transform_pose(pose_cam.pose, tf)
        except Exception as exc:
            self.get_logger().warn(f"do_transform_pose failed: {exc}")
            return None
        return out

    # -----------------------------------------------------------------------
    # Tracking
    # -----------------------------------------------------------------------
    def _associate_or_create(
        self,
        color: str,
        pos_map: np.ndarray,
        stamp,
        orientation: Optional[Quaternion] = None,
    ) -> None:
        stamp_s = stamp.sec + stamp.nanosec * 1e-9
        bucket = self.tracks[color]
        # Find nearest existing track of the same color.
        best: Optional[TrackedRing] = None
        best_d = float("inf")
        for t in bucket:
            d = float(np.linalg.norm(t.position - pos_map))
            if d < best_d:
                best_d = d
                best = t
        if best is not None and best_d < ASSOCIATION_RADIUS_M:
            best.update(pos_map, stamp_s, new_orientation=orientation)
        else:
            new_track = TrackedRing(
                color=color,
                position=pos_map.copy(),
                orientation=orientation if orientation is not None
                             else Quaternion(x=0.0, y=0.0, z=0.0, w=1.0),
                samples=[pos_map.copy()],
                last_seen_stamp=stamp_s,
            )
            bucket.append(new_track)

    def _gc_tracks(self, now_s: float) -> None:
        for color, bucket in self.tracks.items():
            self.tracks[color] = [
                t for t in bucket if (now_s - t.last_seen_stamp) < TRACK_TIMEOUT_S
            ]

    # -----------------------------------------------------------------------
    # Publishing
    # -----------------------------------------------------------------------
    def _publish_state(self) -> None:
        now = self.get_clock().now()
        now_s = now.nanoseconds * 1e-9
        self._gc_tracks(now_s)

        header = Header()
        header.stamp = now.to_msg()
        header.frame_id = TARGET_FRAME

        poses = PoseArray()
        poses.header = header
        markers = MarkerArray()

        # Always publish a DELETEALL first so removed tracks vanish in RViz.
        clear = Marker()
        clear.header = header
        clear.action = Marker.DELETEALL
        markers.markers.append(clear)

        marker_id = 0
        for color, bucket in self.tracks.items():
            for t in bucket:
                if t.detections < MIN_DETECTIONS_FOR_PUBLISH:
                    continue

                pose = Pose()
                pose.position = Point(x=float(t.position[0]),
                                      y=float(t.position[1]),
                                      z=float(t.position[2]))
                pose.orientation = t.orientation
                poses.poses.append(pose)

                r, g, b, a = MARKER_RGBA[color]

                sphere = Marker()
                sphere.header = header
                sphere.ns = f"ring_{color}"
                sphere.id = marker_id
                marker_id += 1
                sphere.type = Marker.SPHERE
                sphere.action = Marker.ADD
                sphere.pose = pose
                sphere.scale.x = sphere.scale.y = sphere.scale.z = 0.18
                sphere.color = ColorRGBA(r=r, g=g, b=b, a=a)
                sphere.lifetime = Duration(seconds=2).to_msg()
                markers.markers.append(sphere)

                # Arrow showing the ring's approach direction (its plane normal,
                # mapped to +X by _build_pose_from_center_normal). Useful for
                # checking that the orientation makes sense in RViz.
                arrow = Marker()
                arrow.header = header
                arrow.ns = f"ring_normal_{color}"
                arrow.id = marker_id
                marker_id += 1
                arrow.type = Marker.ARROW
                arrow.action = Marker.ADD
                arrow.pose = pose
                arrow.scale.x = 0.30  # length
                arrow.scale.y = 0.04
                arrow.scale.z = 0.04
                arrow.color = ColorRGBA(r=r, g=g, b=b, a=0.7)
                arrow.lifetime = Duration(seconds=2).to_msg()
                markers.markers.append(arrow)

                label = Marker()
                label.header = header
                label.ns = f"ring_label_{color}"
                label.id = marker_id
                marker_id += 1
                label.type = Marker.TEXT_VIEW_FACING
                label.action = Marker.ADD
                label.pose = Pose()
                label.pose.position = Point(x=float(t.position[0]),
                                            y=float(t.position[1]),
                                            z=float(t.position[2]) + 0.15)
                label.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
                label.scale.z = 0.12
                label.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
                label.text = f"{color} (n={t.detections})"
                label.lifetime = Duration(seconds=2).to_msg()
                markers.markers.append(label)

        self.poses_pub.publish(poses)
        self.markers_pub.publish(markers)

    def _publish_mask(self, color: str, mask: np.ndarray, header: Header) -> None:
        try:
            msg = self.bridge.cv2_to_imgmsg(mask, encoding="mono8")
            msg.header = header
            self.mask_pubs[color].publish(msg)
        except Exception as exc:
            self.get_logger().warn(f"Mask publish failed for {color}: {exc}")

    # -----------------------------------------------------------------------
    # Debug drawing
    # -----------------------------------------------------------------------
    @staticmethod
    def _draw_detection(img: np.ndarray, ellipse, color: str, hollowness: float,
                        radius_m: float = 0.0, residual_m: float = 0.0) -> None:
        bgr = {
            "red":   (0, 0, 255),
            "green": (0, 255, 0),
            "blue":  (255, 0, 0),
            "black": (60, 60, 60),
        }[color]
        cv2.ellipse(img, ellipse, bgr, 2)
        (cx_px, cy_px), _, _ = ellipse
        label = f"{color} h={hollowness:.2f} r={radius_m*100:.0f}cm res={residual_m*1000:.0f}mm"
        cv2.putText(
            img,
            label,
            (int(cx_px) - 80, int(cy_px) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            bgr,
            1,
            cv2.LINE_AA,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RingDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()