#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from rins_robot.msg import RingCoords
from geometry_msgs.msg import PointStamped, Point
from cv_bridge import CvBridge, CvBridgeError
from rclpy.qos import qos_profile_sensor_data

import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration


# ---------------------------------------------------------------------------
# Topic names — verify with `ros2 topic list` on the real robot and adjust.
# ---------------------------------------------------------------------------
RGB_TOPIC   = "/oakd/rgb/preview/image_raw"
DEPTH_TOPIC = "/oakd/stereo/image_raw"
PC_TOPIC    = "/oakd/stereo/points"

# TF frame published by the OAK-D on the real robot.
# Verify with `ros2 run tf2_tools view_frames`.
CAMERA_FRAME = "oakd_rgb_camera_optical_frame"


class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')

        # ------------------------------------------------------------------
        # Ellipse geometry parameters
        # ------------------------------------------------------------------
        self.min_contour_points = 20    # min contour points to fit ellipse
        self.max_axis_ratio     = 2.5   # max major/minor axis ratio
        self.min_axis_px        = 15    # min ellipse axis length [px]
        self.max_axis_px        = 250   # max ellipse axis length [px]
        self.inner_scale        = 0.45  # inner ellipse = outer * inner_scale

        # ------------------------------------------------------------------
        # Depth parameters — loosened vs sim for real OAK-D noise
        # ------------------------------------------------------------------
        self.min_valid_depth        = 0.3   # [m]
        self.max_valid_depth        = 5.0   # [m]
        self.binary_depth_min       = 0.3   # [m]
        self.binary_depth_max       = 5.0   # [m]
        self.depth_hole_thr         = 0.08  # center must be this much further than rim [m]
        self.min_rim_depth_points   = 5     # min valid depth samples on rim
                                            # (low so black rings still pass)

        # ------------------------------------------------------------------
        # Hole validation via RGB contrast
        # Used as fallback when depth is unavailable (black rings, bad IR).
        # Inner region must differ from outer ring region by this much in V.
        # ------------------------------------------------------------------
        self.min_hole_contrast_v = 20   # [0–255]

        # ------------------------------------------------------------------
        # Merge / clustering parameters
        # ------------------------------------------------------------------
        self.merge_distance_xy = 0.5    # [m]
        self.merge_distance_z  = 0.5    # [m]

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self.latest_depth = None        # float32 depth image [m]
        self.rings_2d     = []          # [(ellipse, color_name)] from current frame
        self.coords       = []          # [(id, Point, color)] unique rings
        self.next_ring_id = 1

        # ------------------------------------------------------------------
        # ROS infrastructure
        # ------------------------------------------------------------------
        self.bridge      = CvBridge()
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.image_sub = self.create_subscription(
            Image, RGB_TOPIC, self.image_callback, 1)
        self.depth_sub = self.create_subscription(
            Image, DEPTH_TOPIC, self.depth_callback, 1)
        self.pc_sub = self.create_subscription(
            PointCloud2, PC_TOPIC, self.pointcloud_callback, qos_profile_sensor_data)

        self.coord_publisher = self.create_publisher(RingCoords, "/ring_coords", 10)
        self.create_timer(1 / 5, self.publish_rings_callback)

        cv2.namedWindow("Binary Image",   cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window",   cv2.WINDOW_NORMAL)

        self.get_logger().info(
            f"RingDetector started.\n"
            f"  RGB:   {RGB_TOPIC}\n"
            f"  Depth: {DEPTH_TOPIC}\n"
            f"  PC:    {PC_TOPIC}\n"
            f"  Frame: {CAMERA_FRAME}"
        )

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        if self.latest_depth is None:
            cv2.imshow("Detected rings", cv_image)
            cv2.waitKey(1)
            return

        depth = self.latest_depth.copy()
        self.rings_2d.clear()

        # 1. Build binary image from depth
        thresh = self._depth_to_binary(depth)
        cv2.imshow("Binary Image", thresh)

        # 2. Find ellipse candidates from depth binary
        ellipse_candidates = self._find_ellipse_candidates(thresh)

        # 3. Validate each candidate as a ring
        vis = cv_image.copy()
        for ellipse in ellipse_candidates:
            cv2.ellipse(vis, ellipse, (255, 255, 0), 1)  # yellow = candidate

            is_ring = self._validate_ring(depth, cv_image, ellipse)
            if not is_ring:
                continue

            color_name = self.classify_ring_color(cv_image, ellipse)

            # Debug: log raw HSV stats to help retune thresholds on real robot
            self._log_hsv_debug(cv_image, ellipse, color_name)

            cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
            cv2.ellipse(vis, ellipse, (0, 255, 0), 2)
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(vis, color_name, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            self.rings_2d.append((ellipse, color_name))

        cv2.imshow("Detected rings", vis)
        cv2.waitKey(1)

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        depth_image = depth_image.astype(np.float32)
        depth_image[~np.isfinite(depth_image)] = 0.0
        self.latest_depth = depth_image

        cv2.imshow("Depth window", self._depth_to_gray(depth_image))
        cv2.waitKey(1)

    def pointcloud_callback(self, data):
        pts = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        pts = pts.reshape((data.height, data.width, 3))

        for ellipse, color in self.rings_2d:
            if color == "unknown":
                continue

            point_3d = self._get_ring_3d_point(pts, ellipse)
            if point_3d is None:
                continue

            point_map = self._camera_to_map(point_3d)
            if point_map is None:
                continue

            x, y, z = point_map.point.x, point_map.point.y, point_map.point.z
            if not np.isfinite([x, y, z]).all():
                continue

            self.get_logger().info(f"Ring detected: {color}  x={x:.2f} y={y:.2f} z={z:.2f}")

            merged = False
            for i, (ring_id, ring_pt, ring_color) in enumerate(self.coords):
                if ring_color != color:
                    continue
                if (abs(ring_pt.x - x) < self.merge_distance_xy and
                        abs(ring_pt.y - y) < self.merge_distance_xy and
                        abs(ring_pt.z - z) < self.merge_distance_z):
                    ring_pt.x = (ring_pt.x + x) / 2.0
                    ring_pt.y = (ring_pt.y + y) / 2.0
                    ring_pt.z = (ring_pt.z + z) / 2.0
                    self.coords[i] = (ring_id, ring_pt, ring_color)
                    self.get_logger().info(f"  → merged with ring #{ring_id}")
                    merged = True
                    break

            if not merged:
                p = Point()
                p.x, p.y, p.z = x, y, z
                self.coords.append((self.next_ring_id, p, color))
                self.get_logger().info(f"  → new ring #{self.next_ring_id}: {color}")
                self.next_ring_id += 1

        self.rings_2d.clear()

    def publish_rings_callback(self):
        msg = RingCoords()
        for ring_id, ring_pt, color in self.coords:
            if not np.isfinite([ring_pt.x, ring_pt.y, ring_pt.z]).all():
                continue
            msg.ids.append(ring_id)
            msg.points.append(ring_pt)
            msg.colors.append(color)
        self.coord_publisher.publish(msg)

    # ------------------------------------------------------------------
    # Detection pipeline
    # ------------------------------------------------------------------

    def _depth_to_binary(self, depth):
        """Build a binary image: white = depth in [binary_depth_min, binary_depth_max]."""
        valid = np.isfinite(depth) & (depth > self.min_valid_depth) & (depth < self.max_valid_depth)
        thresh = np.zeros(depth.shape, dtype=np.uint8)
        thresh[valid & (depth > self.binary_depth_min) & (depth < self.binary_depth_max)] = 255
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        return thresh

    def _find_ellipse_candidates(self, binary):
        """Find ellipses in the binary image that pass basic geometry checks."""
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        ellipses = []
        for cnt in contours:
            if cnt.shape[0] < self.min_contour_points:
                continue
            ellipse = cv2.fitEllipse(cnt)
            ax1, ax2 = ellipse[1]
            if ax1 <= 0 or ax2 <= 0:
                continue
            ratio = ax1 / ax2 if ax1 > ax2 else ax2 / ax1
            if ratio > self.max_axis_ratio:
                continue
            if ax1 < self.min_axis_px or ax2 < self.min_axis_px:
                continue
            if ax1 > self.max_axis_px or ax2 > self.max_axis_px:
                continue
            ellipses.append(ellipse)
        return ellipses

    def _validate_ring(self, depth, bgr_image, ellipse):
        """
        Two-path ring validation:

        Path A (normal): depth-based hole check — rim depth is closer than center.
                         Works for all colored rings with good IR return.

        Path B (black ring fallback): if rim has too few valid depth points,
                         fall back to RGB contrast check between inner and outer
                         ellipse regions. Black plastic absorbs IR so depth is
                         unreliable; the hole behind the ring will look different
                         in brightness from the ring itself.
        """
        rim_depths = self._get_ellipse_perimeter_depths(depth, ellipse)
        rim_depths = self._filter_valid_depths(rim_depths)

        if len(rim_depths) >= self.min_rim_depth_points:
            # Path A: depth-based
            rim_depth = float(np.median(rim_depths))
            cx = int(round(ellipse[0][0]))
            cy = int(round(ellipse[0][1]))

            if 0 <= cy < depth.shape[0] and 0 <= cx < depth.shape[1]:
                center_depth = float(depth[cy, cx])

                # Center returns no depth → hole in 3D space → ring
                if not np.isfinite(center_depth) or center_depth == 0 or center_depth >= self.max_valid_depth:
                    return True

                if center_depth <= self.min_valid_depth:
                    return False

                # Center must be further than rim by at least depth_hole_thr
                return center_depth > rim_depth + self.depth_hole_thr

            return False

        else:
            # Path B: RGB contrast fallback (mainly for black rings)
            return self._validate_ring_rgb_contrast(bgr_image, ellipse)

    def _validate_ring_rgb_contrast(self, bgr_image, ellipse):
        """
        Checks that the inner ellipse region (hole) looks different in brightness
        from the outer ring region. This catches black rings where depth is absent.
        """
        ring_mask, inner_mask, _ = self._make_ring_mask(bgr_image.shape, ellipse)

        if inner_mask.sum() == 0 or ring_mask.sum() == 0:
            return False

        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

        ring_pixels  = gray[ring_mask  > 0]
        inner_pixels = gray[inner_mask > 0]

        if len(ring_pixels) < 10 or len(inner_pixels) < 10:
            return False

        ring_v  = float(np.median(ring_pixels))
        inner_v = float(np.median(inner_pixels))

        contrast = abs(inner_v - ring_v)
        self.get_logger().debug(
            f"RGB contrast check: ring_v={ring_v:.1f} inner_v={inner_v:.1f} contrast={contrast:.1f}"
        )
        return contrast >= self.min_hole_contrast_v

    # ------------------------------------------------------------------
    # Color classification
    # ------------------------------------------------------------------

    def classify_ring_color(self, bgr_image, ellipse):
        """
        Classifies ring color from the ring-shaped pixel band (outer minus inner ellipse).

        HSV thresholds are tuned conservatively. If running on the real robot,
        check the debug logs for actual H/S/V medians and retune here.
        """
        ring_mask, _, _ = self._make_ring_mask(bgr_image.shape, ellipse)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        ring_pixels = hsv[ring_mask > 0]

        if len(ring_pixels) < 20:
            return "unknown"

        s = ring_pixels[:, 1]
        v = ring_pixels[:, 2]

        # Separate chromatic from achromatic pixels
        # Loosen saturation threshold vs sim (60 → 40) for real lighting
        colored = ring_pixels[s > 40]

        if len(colored) < 10:
            # Achromatic branch — classify by brightness
            median_v = float(np.median(v))
            if   median_v < 50:   return "black"
            elif median_v > 180:  return "white"
            else:                 return "gray"

        median_h = float(np.median(colored[:, 0]))
        median_s = float(np.median(colored[:, 1]))

        # Red wraps around 0/180 in OpenCV HSV
        if median_h < 10 or median_h >= 165:  return "red"
        elif median_h < 25:                    return "orange"
        elif median_h < 38:                    return "yellow"
        elif median_h < 85:                    return "green"
        elif median_h < 130:                   return "blue"
        elif median_h < 165:                   return "purple"

        return "unknown"

    def _log_hsv_debug(self, bgr_image, ellipse, classified_color):
        """Logs actual HSV median values. Use this to retune thresholds on real robot."""
        ring_mask, _, _ = self._make_ring_mask(bgr_image.shape, ellipse)
        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        pixels = hsv[ring_mask > 0]
        if len(pixels) == 0:
            return
        mh = float(np.median(pixels[:, 0]))
        ms = float(np.median(pixels[:, 1]))
        mv = float(np.median(pixels[:, 2]))
        self.get_logger().info(
            f"Ring HSV: H={mh:.1f} S={ms:.1f} V={mv:.1f} → classified as '{classified_color}'"
        )

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def _make_ring_mask(self, shape, ellipse):
        """Returns (ring_mask, inner_mask, inner_ellipse)."""
        inner_ellipse = (
            ellipse[0],
            (ellipse[1][0] * self.inner_scale, ellipse[1][1] * self.inner_scale),
            ellipse[2]
        )
        mask_outer = np.zeros(shape[:2], dtype=np.uint8)
        mask_inner = np.zeros(shape[:2], dtype=np.uint8)
        cv2.ellipse(mask_outer, ellipse,       255, thickness=-1)
        cv2.ellipse(mask_inner, inner_ellipse, 255, thickness=-1)
        ring_mask = cv2.subtract(mask_outer, mask_inner)
        return ring_mask, mask_inner, inner_ellipse

    def _get_ellipse_perimeter_depths(self, depth, ellipse, delta_deg=10):
        h, w = depth.shape[:2]
        center = (int(round(ellipse[0][0])), int(round(ellipse[0][1])))
        axes = (
            max(1, int(round(ellipse[1][0] / 2.0))),
            max(1, int(round(ellipse[1][1] / 2.0)))
        )
        angle = int(round(ellipse[2]))
        pts = cv2.ellipse2Poly(center, axes, angle, 0, 360, delta_deg)
        if pts is None or len(pts) == 0:
            return np.array([], dtype=np.float32)
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        return depth[ys, xs]

    def _filter_valid_depths(self, depths):
        depths = depths[np.isfinite(depths)]
        return depths[(depths > self.min_valid_depth) & (depths < self.max_valid_depth)]

    def _get_ring_3d_point(self, pointcloud, ellipse):
        h, w, _ = pointcloud.shape
        ring_mask, _, _ = self._make_ring_mask((h, w), ellipse)
        ys, xs = np.where(ring_mask > 0)
        if len(xs) == 0:
            return None
        pts = pointcloud[ys, xs, :]
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]
        if len(pts) == 0:
            return None
        median_pt = np.median(pts, axis=0)
        return median_pt if np.isfinite(median_pt).all() else None

    def _camera_to_map(self, point_3d):
        p = PointStamped()
        p.header.frame_id = CAMERA_FRAME
        p.header.stamp    = self.get_clock().now().to_msg()
        p.point.x = float(point_3d[0])
        p.point.y = float(point_3d[1])
        p.point.z = float(point_3d[2])
        try:
            transform = self.tf_buffer.lookup_transform(
                "map", p.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.5)   # longer timeout for real robot latency
            )
            return tfg.do_transform_point(p, transform)
        except Exception as e:
            self.get_logger().warn(f"TF failed: {e}")
            return None

    def _depth_to_gray(self, depth):
        d = depth.copy().astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[(d < self.min_valid_depth) | (d > self.max_valid_depth)] = 0.0
        out   = np.zeros(d.shape, dtype=np.uint8)
        valid = d[d > 0]
        if len(valid) == 0:
            return out
        mn, mx = np.min(valid), np.max(valid)
        if mx - mn < 1e-6:
            return out
        norm = (d - mn) / (mx - mn)
        norm[d == 0] = 1.0
        return (norm * 255).astype(np.uint8)


def main():
    rclpy.init(args=None)
    rclpy.spin(RingDetector())
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()