#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2, math
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from rins_robot.msg import RingCoords
from geometry_msgs.msg import PointStamped, Vector3, Pose, Point
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, qos_profile_sensor_data

import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration

qos_profile = QoSProfile(
          durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
          reliability=QoSReliabilityPolicy.RELIABLE,
          history=QoSHistoryPolicy.KEEP_LAST,
          depth=1)

class RingDetector(Node):
    def __init__(self):
        super().__init__('transform_point')

        # Basic ROS stuff
        timer_frequency = 5
        timer_period = 1/timer_frequency

        # ellipse thresholds
        self.ecc_thr = 100
        self.ratio_thr = 1.5
        self.min_contour_points = 20

        # depth thresholds
        self.latest_depth = None
        self.depth_thr = 0.10
        self.min_valid_depth = 0.05
        self.max_valid_depth = 5.0

        # direct threshold on depth image (in meters)
        self.binary_depth_min = 0.5
        self.binary_depth_max = 3.5

        # ring validation
        self.inner_scale = 0.45
        self.min_ring_depth_points = 12
        self.min_center_depth_points = 8

        # merging detections into ring table
        self.merge_distance_xy = 0.5
        self.merge_distance_z = 0.5

        # tf + pointcloud
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            "/oakd/rgb/preview/depth/points",
            self.checkRing_callback,
            qos_profile_sensor_data
        )

        # ring table publisher
        self.coordPublisher = self.create_publisher(RingCoords, "/ring_coords", 10)
        self.publishTimer = self.create_timer(1/5, self.publishRings_callback)

        # variables for ring tracking
        self.rings_2d = []      # (cx, cy, color)
        self.coords = []        # (id, Point(), color)
        self.nextRingId = 1

        # bridge
        self.bridge = CvBridge()

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)

    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
            return

        if self.latest_depth is None:
            cv2.imshow("Detected rings", cv_image)
            cv2.waitKey(1)
            return

        depth = self.latest_depth.copy()

        if cv_image.shape[:2] != depth.shape[:2]:
            print("RGB and depth image shapes do not match")
            return

        # clear detections from previous frame
        self.rings_2d.clear()

        # Binarize the depth image directly using depth values in meters
        valid = np.isfinite(depth)
        valid = valid & (depth > self.min_valid_depth) & (depth < self.max_valid_depth)

        thresh = np.zeros(depth.shape, dtype=np.uint8)
        thresh[valid & (depth > self.binary_depth_min) & (depth < self.binary_depth_max)] = 255

        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Binary Image", thresh)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        contour_vis = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        cv2.drawContours(contour_vis, contours, -1, (255, 0, 0), 1)
        cv2.imshow("Detected contours", contour_vis)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= self.min_contour_points:
                ellipse = cv2.fitEllipse(cnt)

                # filter ellipses that are too eccentric
                e = ellipse[1]
                ecc1 = e[0]
                ecc2 = e[1]

                if ecc1 <= 0 or ecc2 <= 0:
                    continue

                ratio = ecc1/ecc2 if ecc1 > ecc2 else ecc2/ecc1
                if ratio <= self.ratio_thr and ecc1 < self.ecc_thr and ecc2 < self.ecc_thr:
                    elps.append(ellipse)

        vis = cv_image.copy()
        candidates = []

        # Check each ellipse individually
        for ellipse in elps:
            # display candidates
            cv2.ellipse(vis, ellipse, (255, 255, 0), 1)
            cv2.circle(vis, (int(ellipse[0][0]), int(ellipse[0][1])), 1, (255, 255, 0), -1)

            is_ring, inner_ellipse = self.ellipse_is_ring(depth, ellipse)

            if not is_ring:
                continue

            candidates.append((ellipse, inner_ellipse))

        print("Processing is done! found", len(candidates), "candidates for rings")

        # Plot the rings on the image
        for c in candidates:
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(vis, e1, (0, 255, 0), 2)
            cv2.ellipse(vis, e2, (0, 255, 0), 1)

            color_name = self.classify_ring_color(cv_image, e1)

            cx = int(e1[0][0])
            cy = int(e1[0][1])

            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(
                vis,
                color_name,
                (cx + 10, cy),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

            # save 2D detections for pointcloud callback
            self.rings_2d.append((e1, color_name))

        cv2.imshow("Detected rings", vis)
        cv2.waitKey(1)
        
    def get_ring_3d_point(self, pointcloud, ellipse):
        height, width, _ = pointcloud.shape

        ring_mask, _, _ = self.make_ring_mask_from_one_ellipse(
            (height, width), ellipse, self.inner_scale
        )

        ys, xs = np.where(ring_mask > 0)

        if len(xs) == 0:
            return None

        pts = pointcloud[ys, xs, :]
        pts = pts[np.isfinite(pts).all(axis=1)]

        if len(pts) == 0:
            return None

        # odstrani skoraj ničelne točke
        pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]
        if len(pts) == 0:
            return None

        # vzemi mediano po vseh točkah na obroču
        d = np.median(pts, axis=0)

        if not np.isfinite(d).all():
            return None

        return d

    def is_valid_depth(self, d):
        return np.isfinite(d) and d != 0 and self.min_valid_depth < d < self.max_valid_depth

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)
            return

        depth_image = depth_image.astype(np.float32)
        depth_image[depth_image == np.inf] = 0
        depth_image[~np.isfinite(depth_image)] = 0

        self.latest_depth = depth_image.copy()

        # Do the necessary conversion so we can visualize it in OpenCV
        image_viz = self.depth_to_gray(depth_image)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)

    def depth_to_gray(self, depth):
        d = depth.copy().astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        d[(d < self.min_valid_depth) | (d > self.max_valid_depth)] = 0.0

        out = np.zeros(d.shape, dtype=np.uint8)
        valid = d[d > 0]

        if len(valid) == 0:
            return out

        mn = np.min(valid)
        mx = np.max(valid)

        if mx - mn < 1e-6:
            return out

        norm = (d - mn) / (mx - mn)
        norm[d == 0] = 1.0
        out = (norm * 255).astype(np.uint8)
        return out

    def scale_ellipse(self, ellipse, scale):
        (cx, cy), (ax1, ax2), angle = ellipse
        return ((cx, cy), (ax1 * scale, ax2 * scale), angle)

    def make_filled_ellipse_mask(self, shape, ellipse):
        mask = np.zeros(shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, thickness=-1)
        return mask

    def make_ring_mask_from_one_ellipse(self, shape, ellipse, inner_scale):
        mask_outer = self.make_filled_ellipse_mask(shape, ellipse)

        inner_ellipse = self.scale_ellipse(ellipse, inner_scale)
        mask_inner = self.make_filled_ellipse_mask(shape, inner_ellipse)

        ring_mask = cv2.subtract(mask_outer, mask_inner)
        return ring_mask, mask_inner, inner_ellipse

    def ellipse_is_ring(self, depth, ellipse):
        ring_mask, inner_mask, inner_ellipse = self.make_ring_mask_from_one_ellipse(
            depth.shape, ellipse, self.inner_scale
        )

        ring_depths = depth[ring_mask > 0]
        ring_depths = ring_depths[np.isfinite(ring_depths)]
        ring_depths = ring_depths[
            (ring_depths > self.min_valid_depth) & (ring_depths < self.max_valid_depth)
        ]

        if len(ring_depths) < self.min_ring_depth_points:
            return False, inner_ellipse

        ring_depth = float(np.median(ring_depths))

        center_depths = depth[inner_mask > 0]
        center_depths = center_depths[np.isfinite(center_depths)]
        center_depths = center_depths[
            (center_depths > self.min_valid_depth) & (center_depths < self.max_valid_depth)
        ]

        if len(center_depths) < self.min_center_depth_points:
            return True, inner_ellipse

        center_depth = float(np.median(center_depths))

        if center_depth > ring_depth + self.depth_thr:
            return True, inner_ellipse

        return False, inner_ellipse

    def classify_ring_color(self, bgr_image, ellipse):
        ring_mask, _, _ = self.make_ring_mask_from_one_ellipse(
            bgr_image.shape, ellipse, self.inner_scale
        )

        hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        ring_pixels = hsv[ring_mask > 0]

        if len(ring_pixels) < 20:
            return "unknown"

        h = ring_pixels[:, 0]
        s = ring_pixels[:, 1]
        v = ring_pixels[:, 2]

        # remove almost gray / white / black pixels first
        colored = ring_pixels[s > 60]

        if len(colored) < 10:
            median_v = float(np.median(v))
            if median_v > 180:
                return "white"
            elif median_v < 60:
                return "black"
            else:
                return "gray"

        hue_vals = colored[:, 0]
        median_h = float(np.median(hue_vals))

        # H is in range 0-179
        if median_h < 10 or median_h >= 170:
            return "red"
        elif median_h < 25:
            return "orange"
        elif median_h < 35:
            return "yellow"
        elif median_h < 85:
            return "green"
        elif median_h < 130:
            return "blue"
        elif median_h < 170:
            return "purple"

        return "unknown"

    def checkRing_callback(self, data):
        height = data.height
        width = data.width

        a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        a = a.reshape((height, width, 3))

        for ellipse, color in self.rings_2d:
            if color == "unknown":
                continue

            d = self.get_ring_3d_point(a, ellipse)
            if d is None:
                continue

            detection = self.baseLink2Map(d)
            if detection is None:
                continue

            x = detection.point.x
            y = detection.point.y
            z = detection.point.z

            if not np.isfinite([x, y, z]).all():
                continue

            print("NEW DETECTION:", color, x, y, z)

            newRing = True

            for i, (ring_id, ring_pt, ring_color) in enumerate(self.coords):
                if ring_color != color:
                    continue

                print("  compare with:", ring_id, ring_color, ring_pt.x, ring_pt.y, ring_pt.z)

                if (
                    abs(ring_pt.x - x) < self.merge_distance_xy and
                    abs(ring_pt.y - y) < self.merge_distance_xy and
                    abs(ring_pt.z - z) < self.merge_distance_z
                ):
                    ring_pt.x = (ring_pt.x + x) / 2.0
                    ring_pt.y = (ring_pt.y + y) / 2.0
                    ring_pt.z = (ring_pt.z + z) / 2.0
                    self.coords[i] = (ring_id, ring_pt, ring_color)
                    print("MERGED INTO EXISTING RING", ring_id)
                    newRing = False
                    break

            if newRing:
                p = Point()
                p.x = x
                p.y = y
                p.z = z
                self.coords.append((self.nextRingId, p, color))
                print("ADDED NEW RING", self.nextRingId, color, x, y, z)
                self.nextRingId += 1

        self.rings_2d.clear()

    def publishRings_callback(self):
        pub = RingCoords()

        for ring_id, ring_pt, color in self.coords:
            if not np.isfinite([ring_pt.x, ring_pt.y, ring_pt.z]).all():
                continue
            pub.ids.append(ring_id)
            pub.points.append(ring_pt)
            pub.colors.append(color)

        self.coordPublisher.publish(pub)

    def baseLink2Map(self, d):
        p = PointStamped()
        p.header.frame_id = "oakd_rgb_camera_optical_frame"
        p.header.stamp = self.get_clock().now().to_msg()

        p.point.x = float(d[0])
        p.point.y = float(d[1])
        p.point.z = float(d[2])

        try:
            transform = self.tf_buffer.lookup_transform(
                "map",
                p.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
            return tfg.do_transform_point(p, transform)
        except Exception as e:
            print("TF transform failed:", e)
            return None


def main():
    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()