#!/usr/bin/env python3

"""
Ring detection node for TurtleBot4 with Gemini 355L camera (real world).

Overview
--------
- Subscribes to /gemini/color/image_raw  (RGB, bgr8)
- Subscribes to /gemini/depth/image_raw  (uint16, millimetres, aligned to colour)
- Subscribes to /gemini/depth/camera_info (pinhole intrinsics, received once)
- Uses ApproximateTimeSynchronizer → single callback per (RGB, depth) pair,
  eliminating all async race conditions from the original code.
- Detects rings via ellipse fitting on a depth-binary image.
- Rejects FAKE rings (printed images on boxes) by checking that the centre of
  a detected ellipse is SIGNIFICANTLY farther away than the ring perimeter –
  a real hole-in-the-middle ring has no surface behind it at close range,
  while a printed image has the box surface at the same depth.
- Back-projects pixel + depth → 3-D in camera frame via pinhole intrinsics,
  then transforms to map frame via TF2.
- Two-stage pending → confirmed filter prevents false positive rings from
  a single noisy frame from entering the final output.
- Merges duplicate detections of the same physical ring; resolves colour
  by majority vote over all confirmed hits.
- Publishes confirmed rings on /ring_coords.
- Displays annotated visualisation windows.
"""

import rclpy
import rclpy.time
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data

import message_filters

from sensor_msgs.msg import Image, CameraInfo
from rins_robot.msg import RingCoords
from geometry_msgs.msg import PointStamped, Point

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np

import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from tf2_ros import TransformException


# ═══════════════════════════════════════════════════════════════════════════
#  Tunable parameters  (one place to change everything)
# ═══════════════════════════════════════════════════════════════════════════

# ── Depth validity ─────────────────────────────────────────────────────────
DEPTH_MIN_M        = 0.30   # closer than this → sensor noise [m]
DEPTH_MAX_M        = 1.50   # rings are within ~1 m; discard everything beyond [m]

# ── Depth-binary range (objects to segment) ───────────────────────────────
BINARY_DEPTH_MIN_M = 0.25
BINARY_DEPTH_MAX_M = 1.50   # matches DEPTH_MAX_M

# ── Morphological kernel sizes ─────────────────────────────────────────────
MORPH_OPEN_K  = 1   # keep thin ring structures (1x1 open is effectively no-op)
MORPH_CLOSE_K = 3   # still close small contour gaps without over-smoothing

# ── Ellipse geometry filters ──────────────────────────────────────────────
MIN_CONTOUR_PTS  = 15    # minimum contour points to attempt ellipse fit
AXIS_RATIO_MAX   = 3.5   # max major/minor ratio (rejects very elongated ellipses)
AXIS_MIN_PX      = 8     # minor axis must be at least this many pixels
AXIS_MAX_PX      = 260   # major axis must be at most this many pixels (avoids huge blobs)
# Minimum circularity: 4π·area / perimeter².  Circle = 1.0, thin arc ≈ 0.
# Rings appear as roughly circular blobs; scene clutter (table edges, box
# corners) tends to be elongated/angular with much lower circularity.
MIN_CIRCULARITY  = 0.25

# ── Ring hole validation ───────────────────────────────────────────────────
# The inner ellipse used to sample the hole is this fraction of the outer one.
INNER_SCALE           = 0.45
# Minimum number of valid depth samples needed on the ring perimeter.
MIN_RING_DEPTH_PTS    = 8
# Patch half-size for centre-depth sampling [pixels]
CENTRE_PATCH_HALF     = 5
# Minimum number of valid depth pixels inside the centre patch.
MIN_CENTRE_PATCH_PTS  = 2
# Real ring: hole depth must exceed ring depth by at least this much [m].
# Raised from 0.05 → 0.10 to sit well above Gemini 355L depth noise (~3 cm).
# A printed image on a flat box surface has difference ≈ 0; real hole ≫ 0.10.
HOLE_DEPTH_MARGIN_M   = 0.10
# Fraction of hole-region pixels that must be invalid (zero / out-of-range)
# to accept the "strong hole" path.  Raised from 0.40 → 0.60 to reduce
# false positives from noisy scene regions that happen to have some bad pixels.
HOLE_INVALID_FRACTION = 0.60

# ── 3-D back-projection: depth sampling ───────────────────────────────────
# Patch around ellipse centre used to compute the median ring depth for 3-D.
BACKPROJ_PATCH_HALF  = 8   # 17×17 pixel patch

# ── Two-stage confirmation ─────────────────────────────────────────────────
PENDING_MINHITS       = 3    # hits in pending stage before promotion
PENDING_KEEPTIME_NS   = int(4e9)   # 4 s max age for a pending candidate (was 8 s)

# ── Spatial matching thresholds (map frame) [m] ───────────────────────────
PENDING_XY_THR  = 0.55
PENDING_Z_THR   = 0.55
MATCH_XY_THR    = 0.65
MATCH_Z_THR     = 0.65
MERGE_XY_THR    = 0.45
MERGE_Z_THR     = 0.45

# ── Publishing ─────────────────────────────────────────────────────────────
PUBLISH_MIN_HITS = 6    # confirmed ring must have this many hits to be published

# ── Time synchronisation ───────────────────────────────────────────────────
SYNC_QUEUE  = 10
SYNC_SLOP_S = 0.08

# ── Colour classification ─────────────────────────────────────────────────
# Minimum saturation to consider a pixel "coloured" (not grey/white/black)
COLOUR_SAT_THR = 45
# Minimum value (brightness) – discard very dark pixels which are unreliable
COLOUR_VAL_THR = 30

# ═══════════════════════════════════════════════════════════════════════════

class RingDetector(Node):

    def __init__(self):
        super().__init__('ring_detector')

        # ── Camera intrinsics (filled once from camera_info) ──────────────
        self.fx = self.fy = self.cx_cam = self.cy_cam = None
        self.camera_frame = None

        # ── ROS infrastructure ────────────────────────────────────────────
        self.bridge      = CvBridge()
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        sensor_qos = qos_profile_sensor_data

        self.info_sub = self.create_subscription(
            CameraInfo, '/gemini/depth/camera_info',
            self._camera_info_cb, sensor_qos)

        # Synchronised RGB + depth
        self._rgb_sub   = message_filters.Subscriber(
            self, Image, '/gemini/color/image_raw',  qos_profile=sensor_qos)
        self._depth_sub = message_filters.Subscriber(
            self, Image, '/gemini/depth/image_raw',  qos_profile=sensor_qos)
        self._sync = message_filters.ApproximateTimeSynchronizer(
            [self._rgb_sub, self._depth_sub],
            queue_size=SYNC_QUEUE, slop=SYNC_SLOP_S)
        self._sync.registerCallback(self._rgbd_cb)

        # ── Tracking state ────────────────────────────────────────────────
        # pending:   [(Point[map], hit_count, last_seen, color_votes:dict), ...]
        # confirmed: [(ring_id, Point[map], hit_count, last_seen, color_votes:dict), ...]
        self.pending   = []
        self.confirmed = []
        self.next_id   = 1

        # ── Rejection diagnostics (aggregated, logged once per second) ───
        self._rej_stats = {
            'frames': 0,
            'contours': 0,
            'candidates': 0,
            'accepted': 0,
            'rej_contour_pts': 0,
            'rej_circularity': 0,
            'rej_axis_invalid': 0,
            'rej_ratio': 0,
            'rej_size': 0,
            'rej_hole': 0,
            'rej_color': 0,
            'rej_depth': 0,
            'rej_tf': 0,
            'rej_nonfinite': 0,
        }
        self._last_rej_log_ns = self.get_clock().now().nanoseconds

        # ── Publisher ─────────────────────────────────────────────────────
        self.coord_pub = self.create_publisher(RingCoords, '/ring_coords', 10)
        self.create_timer(1.0 / 5.0, self._publish_cb)

        # ── Visualisation windows ─────────────────────────────────────────
        cv2.namedWindow('Ring detector – RGB',    cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ring detector – Binary', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Ring detector – Depth',  cv2.WINDOW_NORMAL)

        self.get_logger().info('RingDetector initialised – waiting for camera_info…')

    # ──────────────────────────────────────────────────────────────────────
    # Camera info
    # ──────────────────────────────────────────────────────────────────────

    def _camera_info_cb(self, msg: CameraInfo):
        if self.fx is not None:
            return
        k = msg.k
        self.fx, self.fy       = k[0], k[4]
        self.cx_cam, self.cy_cam = k[2], k[5]
        self.camera_frame        = msg.header.frame_id
        self.get_logger().info(
            f'Camera info: fx={self.fx:.1f} fy={self.fy:.1f} '
            f'cx={self.cx_cam:.1f} cy={self.cy_cam:.1f}  frame={self.camera_frame}')
        self.destroy_subscription(self.info_sub)

    # ──────────────────────────────────────────────────────────────────────
    # Main synchronised callback
    # ──────────────────────────────────────────────────────────────────────

    def _rgbd_cb(self, rgb_msg: Image, depth_msg: Image):
        if self.fx is None:
            self.get_logger().warn(
                'Waiting for camera_info…', throttle_duration_sec=5.0)
            return

        self._rej_stats['frames'] += 1

        # ── Decode ────────────────────────────────────────────────────────
        try:
            rgb   = self.bridge.imgmsg_to_cv2(rgb_msg,   'bgr8')
            depth_raw = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge: {e}')
            return

        # Gemini 355L depth is uint16 millimetres.
        # Median blur on the raw uint16 removes salt-and-pepper noise while
        # preserving edges better than a Gaussian would.
        depth_raw = cv2.medianBlur(depth_raw, 5)

        depth_m = depth_raw.astype(np.float32) / 1000.0
        depth_m[~np.isfinite(depth_m)] = 0.0

        # Hard clamp: anything beyond DEPTH_MAX_M is irrelevant (rings are
        # within ~1 m) and only adds noise to contour finding.
        depth_m[depth_m > DEPTH_MAX_M] = 0.0

        h_rgb, w_rgb = rgb.shape[:2]
        h_dep, w_dep = depth_m.shape[:2]

        # Rings are hung on a wall above the floor — blank out the lower half
        # of the depth image to eliminate floor/base clutter entirely.
        depth_m[h_dep // 2:, :] = 0.0

        # ── Build depth-binary segmentation mask ─────────────────────────
        valid = (depth_m > DEPTH_MIN_M) & (depth_m < DEPTH_MAX_M)
        binary = np.zeros(depth_m.shape, dtype=np.uint8)
        binary[valid & (depth_m > BINARY_DEPTH_MIN_M) & (depth_m < BINARY_DEPTH_MAX_M)] = 255

        ko = np.ones((MORPH_OPEN_K,  MORPH_OPEN_K),  dtype=np.uint8)
        kc = np.ones((MORPH_CLOSE_K, MORPH_CLOSE_K), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  ko)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kc)

        # ── Find contours and fit ellipses ────────────────────────────────
        # RETR_TREE: retrieve full hierarchy so we can check that a candidate
        # contour actually contains a child (the hole inside the ring).  A real
        # ring in a binary depth image has its outer edge as a parent contour
        # with at least one child contour representing the inner hole edge.
        # Flat printed rings and random blobs typically have NO child contours.
        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        candidates = []   # (ellipse_in_depth_coords, ellipse_in_rgb_coords)
        raw_ellipses_rgb = []

        scale_x = w_rgb / w_dep
        scale_y = h_rgb / h_dep

        for idx, cnt in enumerate(contours):
            self._rej_stats['contours'] += 1
            if cnt.shape[0] < MIN_CONTOUR_PTS:
                self._rej_stats['rej_contour_pts'] += 1
                continue

            # ── Circularity filter ────────────────────────────────────────
            area      = cv2.contourArea(cnt)
            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1e-3:
                self._rej_stats['rej_circularity'] += 1
                continue
            circularity = 4.0 * np.pi * area / (perimeter * perimeter)
            if circularity < MIN_CIRCULARITY:
                self._rej_stats['rej_circularity'] += 1
                continue

            ellipse_dep = cv2.fitEllipse(cnt)
            ax1, ax2 = ellipse_dep[1]
            if ax1 <= 0 or ax2 <= 0:
                self._rej_stats['rej_axis_invalid'] += 1
                continue
            major = max(ax1, ax2)
            minor = min(ax1, ax2)
            ratio = major / minor
            if ratio > AXIS_RATIO_MAX:
                self._rej_stats['rej_ratio'] += 1
                continue
            if minor < AXIS_MIN_PX or major > AXIS_MAX_PX:
                self._rej_stats['rej_size'] += 1
                continue

            # ── Topology check: real ring should have a child contour ─────
            # hierarchy shape: (1, N, 4)  → [next, prev, first_child, parent]
            has_child = (hierarchy[0][idx][2] != -1)
            if not has_child:
                # No inner hole contour found.  Still allow if depth hole check
                # passes strongly (the ring may be too small / close for the
                # inner edge to form a separate contour), but count the miss.
                self.get_logger().debug(
                    f'Contour {idx}: no child contour (topology miss); continuing to depth check')

            # ── Fake-ring rejection via depth hole check ──────────────────
            if not self._ellipse_has_real_hole(depth_m, ellipse_dep):
                self._rej_stats['rej_hole'] += 1
                continue

            # Scale ellipse centre & axes to RGB image coordinates
            cx_d, cy_d = ellipse_dep[0]
            ax1_d, ax2_d = ellipse_dep[1]
            ellipse_rgb = (
                (cx_d * scale_x, cy_d * scale_y),
                (ax1_d * scale_x, ax2_d * scale_y),
                ellipse_dep[2]
            )

            candidates.append((ellipse_dep, ellipse_rgb))
            self._rej_stats['candidates'] += 1
            raw_ellipses_rgb.append(ellipse_dep)   # for yellow candidate overlay

        # ── Visualisation ─────────────────────────────────────────────────
        vis   = rgb.copy()
        depth_vis = self._depth_to_gray(depth_m)

        now   = self.get_clock().now()
        stamp = rgb_msg.header.stamp

        for ellipse_dep, ellipse_rgb in candidates:
            # Colour classification on RGB image using scaled ellipse
            color_name = self._classify_color(rgb, ellipse_rgb)
            if color_name == 'unknown':
                self._rej_stats['rej_color'] += 1
                self.get_logger().debug('Colour classification uncertain; keeping candidate as unknown')

            # 3-D localisation via back-projection.
            # Use the RING PERIMETER depth (material surface), not the hole
            # centre depth (which is air/background and will give wrong 3-D).
            cx_d = int(round(ellipse_dep[0][0]))
            cy_d = int(round(ellipse_dep[0][1]))

            perim_depths = self._perimeter_depths(depth_m, ellipse_dep)
            valid_pd = perim_depths[
                np.isfinite(perim_depths) &
                (perim_depths > DEPTH_MIN_M) &
                (perim_depths < DEPTH_MAX_M)]

            if len(valid_pd) >= MIN_RING_DEPTH_PTS:
                depth_val = float(np.median(valid_pd))
            else:
                # Fallback: small patch around ellipse centre (may be the hole,
                # but at least gives a rough estimate so we don't throw away
                # valid rings that are partially occluded).
                depth_val = self._sample_depth_patch(depth_m, cx_d, cy_d)
                if depth_val is None:
                    self._rej_stats['rej_depth'] += 1
                    continue

            point_cam = self._backproject(cx_d, cy_d, depth_val)
            point_map = self._to_map(point_cam, self.camera_frame, stamp)
            if point_map is None:
                self._rej_stats['rej_tf'] += 1
                continue

            mx, my, mz = point_map.point.x, point_map.point.y, point_map.point.z
            if not np.isfinite([mx, my, mz]).all():
                self._rej_stats['rej_nonfinite'] += 1
                continue

            # Update tracking
            self._update_tracking(mx, my, mz, color_name, now)
            self._rej_stats['accepted'] += 1

            # Draw on visualisation
            try:
                cv2.ellipse(vis, ellipse_rgb, (0, 255, 0), 2)
            except Exception:
                pass
            cx_v = int(round(ellipse_rgb[0][0]))
            cy_v = int(round(ellipse_rgb[0][1]))
            cv2.circle(vis, (cx_v, cy_v), 4, (0, 0, 255), -1)
            cv2.putText(vis, f'{color_name} d={depth_val:.2f}m',
                        (cx_v + 8, cy_v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # ── Housekeeping ──────────────────────────────────────────────────
        self._remove_stale_pending(now)
        self._merge_confirmed()
        self._maybe_log_rejection_summary(now)

        # HUD
        n_pub = sum(1 for _, _, c, _, _ in self.confirmed if c >= PUBLISH_MIN_HITS)
        cv2.putText(vis,
                    f'Confirmed rings: {n_pub}  |  Pending: {len(self.pending)}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis,
                    f'Confirmed rings: {n_pub}  |  Pending: {len(self.pending)}',
                    (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('Ring detector – RGB',    vis)
        cv2.imshow('Ring detector – Binary', binary)
        cv2.imshow('Ring detector – Depth',  depth_vis)
        key = cv2.waitKey(1)
        if key == 27:
            rclpy.shutdown()

    # ──────────────────────────────────────────────────────────────────────
    # Fake-ring rejection  ← THE CRITICAL CHECK
    # ──────────────────────────────────────────────────────────────────────

    def _ellipse_has_real_hole(self, depth_m: np.ndarray, ellipse) -> bool:
        """
        Returns True only if the ellipse looks like a REAL ring with a physical
        hole, not a printed image on a box surface.

        Strategy
        --------
        1. Sample depth values on the perimeter of the outer ellipse → ring_depth.
        2. Sample a patch of depth values inside a scaled-down inner ellipse
           (the hole region) → hole_depths.
        3. Accept as real hole when:
           a) There are enough valid ring-perimeter samples.
           b) EITHER: hole has mostly invalid/infinite depth (robot sees air/far
              wall through the hole) → strong signal of a real hole.
              OR: median hole depth > ring depth + HOLE_DEPTH_MARGIN_M (hole
              region is farther away → real hole looking at the far background).
           c) Reject if hole depth ≈ ring depth (flat printed surface).
        """
        h, w = depth_m.shape[:2]

        # 1. Ring perimeter depth
        perim = self._perimeter_depths(depth_m, ellipse)
        valid_perim = perim[
            np.isfinite(perim) &
            (perim > DEPTH_MIN_M) &
            (perim < DEPTH_MAX_M)]
        if len(valid_perim) < MIN_RING_DEPTH_PTS:
            self.get_logger().debug(
                f'_ellipse_has_real_hole: insufficient perimeter depth samples ({len(valid_perim)})')
            return False
        ring_depth = float(np.median(valid_perim))

        # 2. Inner (hole) region sampling
        inner_ellipse = (
            ellipse[0],
            (ellipse[1][0] * INNER_SCALE, ellipse[1][1] * INNER_SCALE),
            ellipse[2]
        )
        # Draw the inner ellipse mask
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        try:
            cv2.ellipse(mask_inner, inner_ellipse, 255, thickness=-1)
        except Exception:
            self.get_logger().debug('_ellipse_has_real_hole: failed to draw inner ellipse mask')
            return False

        ys, xs = np.where(mask_inner > 0)
        if len(ys) == 0:
            self.get_logger().debug('_ellipse_has_real_hole: inner ellipse produced no pixels')
            return False

        hole_raw = depth_m[ys, xs]

        # Pixels with zero depth (sensor returns 0 for no-return = air/glass)
        # and finite-but-out-of-range are both "hole evidence"
        n_total   = len(hole_raw)
        n_invalid = int(np.sum(
            (hole_raw == 0) |
            ~np.isfinite(hole_raw) |
            (hole_raw < DEPTH_MIN_M)
        ))

        invalid_fraction = n_invalid / n_total if n_total > 0 else 1.0

        # Strong hole signal: most pixels in the hole region have no depth return
        # (looking through the air at a far wall, or outside the sensor range)
        if invalid_fraction >= HOLE_INVALID_FRACTION:
            return True

        # Moderate signal: hole depth is measurably farther than ring surface
        valid_hole = hole_raw[
            np.isfinite(hole_raw) &
            (hole_raw > DEPTH_MIN_M) &
            (hole_raw < DEPTH_MAX_M)]

        if len(valid_hole) < MIN_CENTRE_PATCH_PTS:
            # Not enough valid readings in hole → treat as invalid (can't tell)
            # but since invalid_fraction was < 0.55 there were readings → reject
            self.get_logger().debug(
                f'_ellipse_has_real_hole: insufficient hole depth samples ({len(valid_hole)}), invalid_fraction={invalid_fraction:.2f}')
            return False

        hole_depth = float(np.median(valid_hole))
        is_real = hole_depth > ring_depth + HOLE_DEPTH_MARGIN_M
        if not is_real:
            self.get_logger().debug(
                f'_ellipse_has_real_hole: rejected as fake-like (hole_depth={hole_depth:.2f}, ring_depth={ring_depth:.2f}, margin={HOLE_DEPTH_MARGIN_M:.2f})')
        return is_real

    # ──────────────────────────────────────────────────────────────────────
    # Ellipse helpers
    # ──────────────────────────────────────────────────────────────────────

    def _perimeter_depths(self, depth_m: np.ndarray, ellipse,
                          delta_deg: int = 8) -> np.ndarray:
        """Sample depth values around the outer ellipse perimeter."""
        h, w = depth_m.shape[:2]
        cx  = int(round(ellipse[0][0]))
        cy  = int(round(ellipse[0][1]))
        ax1 = max(1, int(round(ellipse[1][0] / 2.0)))
        ax2 = max(1, int(round(ellipse[1][1] / 2.0)))
        ang = int(round(ellipse[2]))

        pts = cv2.ellipse2Poly((cx, cy), (ax1, ax2), ang, 0, 360, delta_deg)
        if pts is None or len(pts) == 0:
            return np.array([], dtype=np.float32)
        xs = np.clip(pts[:, 0], 0, w - 1)
        ys = np.clip(pts[:, 1], 0, h - 1)
        return depth_m[ys, xs]

    def _sample_depth_patch(self, depth_m: np.ndarray,
                            cx: int, cy: int) -> float | None:
        """
        Return median valid depth from a small patch around (cx, cy).
        Used to get a robust depth estimate at the ring centre for 3-D
        back-projection (we use the ring's own depth, not the hole depth).
        Here we actually want the ring material depth, so we sample a ring-
        band patch rather than the dead centre – use the perimeter average
        instead (called as fallback in the main loop).
        For the primary estimate we sample a patch at the ellipse centre
        displaced slightly toward the ring material, but for simplicity we
        just take a small patch and pick valid, near values (ring surface).
        """
        h, w = depth_m.shape[:2]
        r = BACKPROJ_PATCH_HALF
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
        if y1 <= y0 or x1 <= x0:
            return None
        patch = depth_m[y0:y1, x0:x1]
        valid = patch[(patch > DEPTH_MIN_M) & (patch < DEPTH_MAX_M) &
                      np.isfinite(patch)]
        if len(valid) < 4:
            return None
        return float(np.median(valid))

    # ──────────────────────────────────────────────────────────────────────
    # Colour classification
    # ──────────────────────────────────────────────────────────────────────

    def _classify_color(self, bgr: np.ndarray, ellipse) -> str:
        """
        Classify the colour of a ring from the BGR image.

        Steps
        -----
        1. Build a ring-band mask (outer ellipse minus inner ellipse).
        2. Convert to HSV.
        3. Discard very dark pixels (shadows, depth holes casting shade).
        4. Separate chromatic from achromatic pixels.
        5. Use median hue of chromatic pixels, or brightness for achromatic.
        """
        h, w = bgr.shape[:2]

        # Build ring-band mask in RGB coordinates
        inner_ell = (
            ellipse[0],
            (ellipse[1][0] * INNER_SCALE, ellipse[1][1] * INNER_SCALE),
            ellipse[2]
        )
        mask_outer = np.zeros((h, w), dtype=np.uint8)
        mask_inner = np.zeros((h, w), dtype=np.uint8)
        try:
            cv2.ellipse(mask_outer, ellipse,    255, thickness=-1)
            cv2.ellipse(mask_inner, inner_ell,  255, thickness=-1)
        except Exception:
            return 'unknown'
        ring_mask = cv2.subtract(mask_outer, mask_inner)

        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        px  = hsv[ring_mask > 0]

        if len(px) < 20:
            return 'unknown'

        # Discard very dark pixels – unreliable hue
        px = px[px[:, 2] > COLOUR_VAL_THR]
        if len(px) < 10:
            return 'unknown'

        sat = px[:, 1]
        val = px[:, 2]
        hue = px[:, 0]

        chromatic = px[sat > COLOUR_SAT_THR]

        if len(chromatic) < 10:
            # Achromatic rings: only black is allowed.
            med_v = float(np.median(val))
            if med_v < 55:
                return 'black'
            return 'unknown'

        # Use median hue of chromatic pixels
        # OpenCV H range: 0–179
        # Red wraps around 0/179 – handle with circular median trick
        h_vals = chromatic[:, 0].astype(np.float32)

        # Shift reds above 170 to negative so the circular wrap is handled
        h_shifted = np.where(h_vals > 170, h_vals - 180.0, h_vals)
        med_h = float(np.median(h_shifted))
        if med_h < 0:
            med_h += 180.0

        if   med_h <  8  or med_h >= 172: return 'red'
        elif med_h < 22:                  return 'orange'
        elif med_h < 38:                  return 'yellow'
        elif med_h < 88:                  return 'green'
        elif med_h < 130:                 return 'blue'
        elif med_h < 172:                 return 'purple'

        return 'unknown'

    # ──────────────────────────────────────────────────────────────────────
    # 3-D geometry
    # ──────────────────────────────────────────────────────────────────────

    def _backproject(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """Pinhole back-projection: pixel (u,v) + depth → camera-frame XYZ."""
        X = (u - self.cx_cam) * depth_m / self.fx
        Y = (v - self.cy_cam) * depth_m / self.fy
        return np.array([X, Y, depth_m], dtype=float)

    def _to_map(self, point_cam: np.ndarray,
                frame_id: str, stamp) -> PointStamped | None:
        """Transform a camera-frame 3-D point to the map frame via TF2."""
        ps = PointStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp    = stamp
        ps.point.x = float(point_cam[0])
        ps.point.y = float(point_cam[1])
        ps.point.z = float(point_cam[2])

        timeout     = Duration(seconds=0.15)
        source_time = rclpy.time.Time.from_msg(stamp)

        try:
            tf = self.tf_buffer.lookup_transform(
                'map', frame_id, source_time, timeout)
            return tfg.do_transform_point(ps, tf)
        except TransformException as te:
            err = str(te).lower()
            if ('extrapolation' in err or 'future' in err or
                    'lookup would require' in err):
                try:
                    tf = self.tf_buffer.lookup_transform(
                        'map', frame_id, rclpy.time.Time(), timeout)
                    return tfg.do_transform_point(ps, tf)
                except TransformException as te2:
                    self.get_logger().debug(f'TF fallback failed: {te2}')
                    return None
            self.get_logger().debug(f'TF failed: {te}')
            return None

    # ──────────────────────────────────────────────────────────────────────
    # Tracking
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _xy_dist(ax, ay, bx, by) -> float:
        return float(np.hypot(ax - bx, ay - by))

    def _update_tracking(self, x, y, z, color, now):
        """Route a new map-frame detection through the pending/confirmed pipeline."""
        # 1. Try to associate with a confirmed ring
        if self._hit_confirmed(x, y, z, color, now):
            return
        # 2. Otherwise update / create a pending candidate
        self._hit_pending(x, y, z, color, now)

    def _hit_confirmed(self, x, y, z, color, now) -> bool:
        best_i, best_d = -1, float('inf')
        for i, (ring_id, pt, count, last_seen, votes) in enumerate(self.confirmed):
            d_xy = self._xy_dist(pt.x, pt.y, x, y)
            d_z  = abs(pt.z - z)
            if d_xy <= MATCH_XY_THR and d_z <= MATCH_Z_THR and d_xy < best_d:
                best_d = d_xy
                best_i = i
        if best_i < 0:
            return False

        ring_id, pt, count, _, votes = self.confirmed[best_i]
        # Weighted running mean
        w   = 1.0 / (count + 1)
        pt.x = pt.x * (1 - w) + x * w
        pt.y = pt.y * (1 - w) + y * w
        pt.z = pt.z * (1 - w) + z * w
        votes[color] = votes.get(color, 0) + 1
        self.confirmed[best_i] = (ring_id, pt, count + 1, now, votes)
        return True

    def _hit_pending(self, x, y, z, color, now):
        best_i, best_d = -1, float('inf')
        for i, (pt, count, last_seen, votes) in enumerate(self.pending):
            d_xy = self._xy_dist(pt.x, pt.y, x, y)
            d_z  = abs(pt.z - z)
            if d_xy <= PENDING_XY_THR and d_z <= PENDING_Z_THR and d_xy < best_d:
                best_d = d_xy
                best_i = i

        if best_i >= 0:
            pt, count, _, votes = self.pending[best_i]
            w   = 1.0 / (count + 1)
            pt.x = pt.x * (1 - w) + x * w
            pt.y = pt.y * (1 - w) + y * w
            pt.z = pt.z * (1 - w) + z * w
            votes[color] = votes.get(color, 0) + 1
            count += 1
            self.pending[best_i] = (pt, count, now, votes)

            if count >= PENDING_MINHITS:
                best_color = max(votes, key=votes.get)
                self.get_logger().info(
                    f'Ring confirmed! id={self.next_id} color={best_color} '
                    f'map=({pt.x:.2f},{pt.y:.2f},{pt.z:.2f})')
                self.confirmed.append((self.next_id, pt, count, now, votes))
                self.next_id += 1
                del self.pending[best_i]
            return

        # Brand-new candidate
        p = Point()
        p.x, p.y, p.z = float(x), float(y), float(z)
        self.pending.append((p, 1, now, {color: 1}))

    def _remove_stale_pending(self, now):
        self.pending = [
            (pt, c, t, v) for pt, c, t, v in self.pending
            if max(0, (now - t).nanoseconds) <= PENDING_KEEPTIME_NS
        ]

    def _merge_confirmed(self):
        """Merge confirmed rings that ended up too close to each other."""
        if len(self.confirmed) < 2:
            return
        merged = []
        used   = set()
        for i in range(len(self.confirmed)):
            if i in used:
                continue
            id_i, pt_i, c_i, t_i, v_i = self.confirmed[i]
            for j in range(i + 1, len(self.confirmed)):
                if j in used:
                    continue
                id_j, pt_j, c_j, t_j, v_j = self.confirmed[j]
                if (self._xy_dist(pt_i.x, pt_i.y, pt_j.x, pt_j.y) <= MERGE_XY_THR
                        and abs(pt_i.z - pt_j.z) <= MERGE_Z_THR):
                    total = c_i + c_j
                    pt_i.x = (pt_i.x * c_i + pt_j.x * c_j) / total
                    pt_i.y = (pt_i.y * c_i + pt_j.y * c_j) / total
                    pt_i.z = (pt_i.z * c_i + pt_j.z * c_j) / total
                    c_i = total
                    t_i = t_i if t_i > t_j else t_j
                    id_i = min(id_i, id_j)
                    for col, cnt in v_j.items():
                        v_i[col] = v_i.get(col, 0) + cnt
                    used.add(j)
                    self.get_logger().info(
                        f'Merged confirmed ring {id_j} → {id_i}')
            merged.append((id_i, pt_i, c_i, t_i, v_i))
        self.confirmed = merged

    def _maybe_log_rejection_summary(self, now):
        """Log aggregate rejection diagnostics once per second."""
        now_ns = now.nanoseconds
        if now_ns - self._last_rej_log_ns < int(1e9):
            return

        s = self._rej_stats
        self.get_logger().info(
            'Reject summary (1s): '
            f'frames={s["frames"]} contours={s["contours"]} candidates={s["candidates"]} accepted={s["accepted"]} '
            f'contour_pts={s["rej_contour_pts"]} circ={s["rej_circularity"]} axis={s["rej_axis_invalid"]} ratio={s["rej_ratio"]} '
            f'size={s["rej_size"]} hole={s["rej_hole"]} color={s["rej_color"]} depth={s["rej_depth"]} '
            f'tf={s["rej_tf"]} nonfinite={s["rej_nonfinite"]}'
        )

        for key in s:
            s[key] = 0
        self._last_rej_log_ns = now_ns

    # ──────────────────────────────────────────────────────────────────────
    # Publisher
    # ──────────────────────────────────────────────────────────────────────

    def _publish_cb(self):
        msg = RingCoords()
        for ring_id, pt, count, _, votes in self.confirmed:
            if count < PUBLISH_MIN_HITS:
                continue
            if not np.isfinite([pt.x, pt.y, pt.z]).all():
                continue
            best_color = max(votes, key=votes.get)
            msg.ids.append(ring_id)
            msg.points.append(pt)
            msg.colors.append(best_color)
        self.coord_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────
    # Visualisation helper
    # ──────────────────────────────────────────────────────────────────────

    def _depth_to_gray(self, depth_m: np.ndarray) -> np.ndarray:
        d = depth_m.copy()
        d[~np.isfinite(d)] = 0.0
        d[(d < DEPTH_MIN_M) | (d > DEPTH_MAX_M)] = 0.0
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


# ═══════════════════════════════════════════════════════════════════════════

def main():
    rclpy.init(args=None)
    node = RingDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()