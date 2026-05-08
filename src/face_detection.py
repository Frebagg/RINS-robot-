#!/usr/bin/env python3
 
"""
Face detection node for TurtleBot4 with Gemini 355L camera (real world).
 
Architecture:
  - Subscribes to /gemini/color/image_raw  (RGB)
  - Subscribes to /gemini/depth/image_raw  (16-bit depth in mm, aligned to colour)
  - Subscribes to /gemini/depth/camera_info (intrinsics)
  - Uses ApproximateTimeSynchronizer to pair every RGB frame with its depth frame
  - Runs YOLOv8n (person class) on the RGB image
  - For each detection, samples a median depth from a patch around the bbox centre
  - Back-projects pixel + depth → 3-D point in camera frame using camera intrinsics
  - Transforms camera-frame point → map frame via tf2
  - Two-stage confirmation filter (pending → confirmed) to eliminate false positives
  - Merges duplicate detections that map to the same physical face
  - Publishes confirmed face coordinates on /face_coords
  - Displays annotated camera image in a window
"""
 
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
 
import message_filters
 
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
 
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped, Point
from rclpy.duration import Duration
import rclpy.time
 
from rins_robot.msg import FaceCoords
 
from ultralytics import YOLO
import torch
import os
 
 
# ---------------------------------------------------------------------------
# Tunable parameters (all in one place for easy adjustment)
# ---------------------------------------------------------------------------
 
# YOLO confidence threshold – increase to reduce false positives on printed faces
CONFIDENCE_THRESHOLD   = 0.7
 
# Depth validity range [metres].  Faces on walls are typically 0.4 – 3.5 m away.
DEPTH_MIN_M            = 0.30
DEPTH_MAX_M            = 1.50
 
# Median-depth sampling: half-size of the square patch around bbox centre [pixels]
DEPTH_PATCH_HALF       = 8    # samples a 17×17 patch
 
# Pending-stage: how many consistent hits needed before a detection is confirmed
MINHITS                = 4
 
# Matching radius for associating a new detection with an existing pending entry [m]
PENDING_XY_THRESHOLD   = 0.65
PENDING_Z_THRESHOLD    = 0.50
 
# Maximum age of a pending entry before it is discarded [nanoseconds]
PENDING_KEEPTIME_NS    = int(6e9)   # 6 seconds
 
# Matching radius for associating a new detection with a confirmed face [m]
MATCH_XY_THRESHOLD     = 0.75
MATCH_Z_THRESHOLD      = 0.60
 
# Two confirmed faces closer than these thresholds are merged into one [m]
MERGE_XY_THRESHOLD     = 0.40
MERGE_Z_THRESHOLD      = 0.50
 
# A face must have at least this many hits before it is published
PUBLISH_COUNT_THRESHOLD = 8
 
# ApproximateTimeSynchronizer queue/slop
SYNC_QUEUE_SIZE        = 10
SYNC_SLOP_S            = 0.08   # 80 ms – generous for real camera
 
 
# ---------------------------------------------------------------------------
 
class detect_faces(Node):
 
    def __init__(self):
        super().__init__('detect_faces')
 
        # ----- device selection -----
        self.declare_parameters(namespace='', parameters=[('device', '')])
        param_device = self.get_parameter('device').get_parameter_value().string_value
        if param_device != '':
            self.device = param_device
        elif torch.cuda.is_available():
            self.device = '0'
        else:
            self.device = 'cpu'
 
        # ----- ROS infrastructure -----
        self.bridge   = CvBridge()
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
 
        # Camera intrinsics – filled in once from /gemini/depth/camera_info
        self.fx = self.fy = self.cx_cam = self.cy_cam = None
        self.camera_frame = None
 
        # QoS that matches most real-sensor publishers
        sensor_qos = qos_profile_sensor_data
 
        self.info_sub = self.create_subscription(
            CameraInfo,
            '/gemini/depth/camera_info',
            self.cameraInfo_callback,
            sensor_qos,
        )
 
        # Synchronised RGB + depth subscriptions
        self.rgb_sub   = message_filters.Subscriber(self, Image, '/gemini/color/image_raw',   qos_profile=sensor_qos)
        self.depth_sub = message_filters.Subscriber(self, Image, '/gemini/depth/image_raw',   qos_profile=sensor_qos)
 
        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub],
            queue_size=SYNC_QUEUE_SIZE,
            slop=SYNC_SLOP_S,
        )
        self.sync.registerCallback(self.rgbdCallback)
 
        # ----- YOLO model -----
        model_path = os.path.expanduser("~/models/yolov8n-face-lindevs.pt")
        self.model = YOLO(model_path)
        if torch.cuda.is_available() and self.device != 'cpu':
            self.get_logger().info(f"YOLO inference on GPU {self.device} ({torch.cuda.get_device_name(0)})")
        else:
            self.get_logger().warn("No GPU detected – running YOLO on CPU (will be slow).")
 
        # ----- Publisher -----
        self.coordPublisher = self.create_publisher(FaceCoords, '/face_coords', 10)
        self.publishTimer   = self.create_timer(1.0 / 5.0, self.publishFaces_callback)
 
        # ----- Detection state -----
        # confirmed:  list of (faceId, Point[map], hit_count, last_seen_rclpy_time)
        # pending:    list of (Point[map], hit_count, last_seen_rclpy_time)
        self.coords        = []
        self.pendingCoords = []
        self.nextFaceId    = 1
 
        # Visualisation colours
        self.COL_DETECT  = (0, 255, 0)    # green  – raw YOLO box
        self.COL_PENDING = (0, 165, 255)  # orange – pending face
        self.COL_CONFIRM = (0, 0, 255)    # red    – confirmed face
 
        self.get_logger().info('Face detection node initialised – waiting for camera info…')
 
    # ------------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------------
 
    def cameraInfo_callback(self, msg: CameraInfo):
        if self.fx is not None:
            return  # already received
        k = msg.k  # row-major 3×3
        self.fx       = k[0]
        self.fy       = k[4]
        self.cx_cam   = k[2]
        self.cy_cam   = k[5]
        self.camera_frame = msg.header.frame_id
        self.get_logger().info(
            f'Camera info received: fx={self.fx:.1f} fy={self.fy:.1f} '
            f'cx={self.cx_cam:.1f} cy={self.cy_cam:.1f} frame={self.camera_frame}'
        )
        # Can destroy the subscription now – intrinsics won't change
        self.destroy_subscription(self.info_sub)
 
    # ------------------------------------------------------------------
    # Main synchronised RGB + depth callback
    # ------------------------------------------------------------------
 
    def rgbdCallback(self, rgb_msg: Image, depth_msg: Image):
        """Called once per synchronised (RGB, depth) pair."""
 
        # Drop frames until intrinsics are available
        if self.fx is None:
            self.get_logger().warn('Waiting for camera_info before processing frames…', throttle_duration_sec=5.0)
            return
 
        # ----- Decode images -----
        try:
            rgb   = self.bridge.imgmsg_to_cv2(rgb_msg,   desired_encoding='bgr8')
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge error: {e}')
            return
 
        # depth is uint16 in millimetres for the Gemini 355L
        if depth.dtype != np.uint16:
            depth = depth.astype(np.uint16)
 
        h_rgb, w_rgb = rgb.shape[:2]
        h_dep, w_dep = depth.shape[:2]
 
        # ----- YOLO inference -----
        results = self.model.predict(
            rgb,
            imgsz=(480, 640),   # native-ish resolution for real camera
            show=False,
            verbose=False,
            #classes=[0],        # person only
            device=self.device,
            conf=CONFIDENCE_THRESHOLD,
        )
 
        if not results or results[0].boxes is None or results[0].boxes.xyxy.nelement() == 0:
            cv2.imshow('Face detection', rgb)
            cv2.waitKey(1)
            return
 
        boxes = results[0].boxes
        now   = self.get_clock().now()
        stamp = rgb_msg.header.stamp
 
        display = rgb.copy()

        # Runtime counters for logging diagnostics
        boxes_processed = 0
        low_confidence_skips = 0
        depth_failures = 0
        tf_failures = 0
        confirmed_assocs = 0
        pending_matches = 0
        pending_creations = 0
        promotions = 0

        for i in range(len(boxes)):
            confidence = float(boxes.conf[i])
            boxes_processed += 1
            if confidence < CONFIDENCE_THRESHOLD:
                low_confidence_skips += 1
                self.get_logger().debug(f'Skip low conf: {confidence:.2f} < {CONFIDENCE_THRESHOLD:.2f}')
                continue

            verts = boxes.xyxy[i]
            x1, y1, x2, y2 = int(verts[0]), int(verts[1]), int(verts[2]), int(verts[3])
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Scale pixel centre to depth image coordinates (may differ in resolution)
            dcx = int(cx * w_dep / w_rgb)
            dcy = int(cy * h_dep / h_rgb)

            # ----- Robust depth sampling (median over a patch) -----
            depth_m = self._sampleDepth(depth, dcx, dcy)
            if depth_m is None:
                depth_failures += 1
                # Draw box in a different colour to show depth failed
                cv2.rectangle(display, (x1, y1), (x2, y2), (128, 128, 128), 1)
                continue

            # ----- Back-project pixel → 3-D in camera frame -----
            point_cam = self._backproject(cx, cy, depth_m)

            # ----- Transform to map frame -----
            map_pt = self._toMapFrame(point_cam, self.camera_frame, stamp)
            if map_pt is None:
                tf_failures += 1
                self.get_logger().debug('TF transform failed or unavailable for current stamp')
                continue

            mx, my, mz = map_pt.point.x, map_pt.point.y, map_pt.point.z

            # ----- Update tracking -----
            if self.updateConfirmed(mx, my, mz, now):
                confirmed_assocs += 1
            else:
                # updatePending returns None but logs promotions; track creates/matches here crudely
                before_pending = len(self.pendingCoords)
                self.updatePending(mx, my, mz, now)
                after_pending = len(self.pendingCoords)
                if after_pending > before_pending:
                    pending_creations += 1
                else:
                    pending_matches += 1

            # ----- Visualise on display -----
            label = f'{confidence:.2f}  d={depth_m:.2f}m'
            cv2.rectangle(display, (x1, y1), (x2, y2), self.COL_DETECT, 2)
            cv2.circle(display,   (cx, cy), 5, self.COL_DETECT, -1)
            cv2.putText(display, label, (x1, max(y1 - 6, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COL_DETECT, 1, cv2.LINE_AA)
 
        # ----- Housekeeping -----
        self.removePending(now)
        self.checkConfirmed(now)
 
        # ----- Overlay confirmed faces -----
        conf_count = sum(1 for _, _, c, _ in self.coords if c >= PUBLISH_COUNT_THRESHOLD)
        cv2.putText(display,
                    f'Confirmed faces: {conf_count}  |  Pending: {len(self.pendingCoords)}',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(display,
                    f'Confirmed faces: {conf_count}  |  Pending: {len(self.pendingCoords)}',
                    (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
 
        cv2.imshow('Face detection', display)
        key = cv2.waitKey(1)
        if key == 27:
            self.get_logger().info('ESC pressed – shutting down.')
            rclpy.shutdown()
 
        self.get_logger().debug(
            f'Confirmed: {len(self.coords)}, Pending: {len(self.pendingCoords)}'
        )
 
    # ------------------------------------------------------------------
    # Depth helpers
    # ------------------------------------------------------------------
 
    def _sampleDepth(self, depth_img: np.ndarray, cx: int, cy: int):
        """
        Returns median depth [metres] from a patch around (cx, cy).
        Returns None if no valid pixels exist or depth is out of range.
        """
        h, w = depth_img.shape[:2]
        r = DEPTH_PATCH_HALF
        y0, y1 = max(0, cy - r), min(h, cy + r + 1)
        x0, x1 = max(0, cx - r), min(w, cx + r + 1)
 
        if y1 <= y0 or x1 <= x0:
            self.get_logger().debug(f'_sampleDepth: patch out of bounds for cx={cx} cy={cy} (x0={x0} x1={x1} y0={y0} y1={y1})')
            return None
 
        patch = depth_img[y0:y1, x0:x1].astype(np.float32)
        patch_m = patch / 1000.0  # mm → m
 
        # Keep only physically plausible values
        valid = patch_m[(patch_m >= DEPTH_MIN_M) & (patch_m <= DEPTH_MAX_M)]
        if valid.size < 5:
            self.get_logger().debug(f'_sampleDepth: insufficient valid depth pixels ({valid.size}) at cx={cx} cy={cy}')
            return None

        return float(np.median(valid))
 
    def _backproject(self, u: int, v: int, depth_m: float) -> np.ndarray:
        """
        Back-project image pixel (u, v) with depth depth_m [m] to a 3-D
        point in the camera frame using the pinhole model.
        Returns np.array([X, Y, Z]) in metres.
        """
        X = (u - self.cx_cam) * depth_m / self.fx
        Y = (v - self.cy_cam) * depth_m / self.fy
        Z = depth_m
        return np.array([X, Y, Z], dtype=float)
 
    # ------------------------------------------------------------------
    # TF helper
    # ------------------------------------------------------------------
 
    def _toMapFrame(self, point_cam: np.ndarray, frame_id: str, stamp) -> PointStamped | None:
        """
        Transform a 3-D point from frame_id to the map frame.
        Handles future-extrapolation errors by falling back to the latest
        available transform.
        """
        ps = PointStamped()
        ps.header.frame_id = frame_id
        ps.header.stamp    = stamp
        ps.point.x = float(point_cam[0])
        ps.point.y = float(point_cam[1])
        ps.point.z = float(point_cam[2])
 
        timeout     = Duration(seconds=0.15)
        source_time = rclpy.time.Time.from_msg(stamp)
 
        try:
            trans     = self.tf_buffer.lookup_transform('map', frame_id, source_time, timeout)
            return tfg.do_transform_point(ps, trans)
        except TransformException as te:
            err = str(te).lower()
            if 'extrapolation into the future' in err or 'lookup would require extrapolation' in err:
                # Fallback: use the most recent available transform
                try:
                    trans = self.tf_buffer.lookup_transform('map', frame_id, rclpy.time.Time(), timeout)
                    return tfg.do_transform_point(ps, trans)
                except TransformException as te2:
                    self.get_logger().debug(f'Fallback TF failed: {te2}')
                    return None
            self.get_logger().debug(f'TF lookup failed: {te}')
            return None
 
    # ------------------------------------------------------------------
    # Tracking helpers
    # ------------------------------------------------------------------
 
    @staticmethod
    def _xyDist(ax, ay, bx, by) -> float:
        return float(np.hypot(ax - bx, ay - by))
 
    def updateConfirmed(self, x, y, z, now) -> bool:
        """
        Try to associate (x, y, z) with an existing confirmed face.
        If found, update its position with a weighted running mean and
        increment the hit count. Returns True on success.
        """
        bestIdx   = -1
        bestXyDist = float('inf')
 
        for i, (faceId, face, count, lastSeen) in enumerate(self.coords):
            xyDist = self._xyDist(face.x, face.y, x, y)
            dz     = abs(face.z - z)
            if xyDist <= MATCH_XY_THRESHOLD and dz <= MATCH_Z_THRESHOLD and xyDist < bestXyDist:
                bestXyDist = xyDist
                bestIdx    = i
 
        if bestIdx < 0:
            return False

        faceId, face, count, _ = self.coords[bestIdx]
        self.get_logger().debug(f'updateConfirmed: associating with confirmed face {faceId} (xyDist={bestXyDist:.3f} m, dz={abs(face.z - z):.3f} m)')
        # Weighted running mean – older observations contribute less over time
        w = 1.0 / (count + 1)
        face.x = face.x * (1.0 - w) + x * w
        face.y = face.y * (1.0 - w) + y * w
        face.z = face.z * (1.0 - w) + z * w
        self.coords[bestIdx] = (faceId, face, count + 1, now)
        return True
 
    def updatePending(self, x, y, z, now):
        """
        Associate (x, y, z) with an existing pending candidate or create a new one.
        When a pending candidate accumulates enough hits it is promoted to confirmed.
        """
        bestIdx    = -1
        bestXyDist = float('inf')
 
        for i, (face, count, lastSeen) in enumerate(self.pendingCoords):
            xyDist = self._xyDist(face.x, face.y, x, y)
            dz     = abs(face.z - z)
            if xyDist <= PENDING_XY_THRESHOLD and dz <= PENDING_Z_THRESHOLD and xyDist < bestXyDist:
                bestXyDist = xyDist
                bestIdx    = i
 
        if bestIdx >= 0:
            face, count, _ = self.pendingCoords[bestIdx]
            self.get_logger().debug(f'updatePending: matched pending idx={bestIdx} (count={count}) to new detection (x={x:.2f}, y={y:.2f}, z={z:.2f})')
            w = 1.0 / (count + 1)
            face.x = face.x * (1.0 - w) + x * w
            face.y = face.y * (1.0 - w) + y * w
            face.z = face.z * (1.0 - w) + z * w
            count += 1
            self.pendingCoords[bestIdx] = (face, count, now)

            if count >= MINHITS:
                self.get_logger().info(
                    f'New face confirmed! id={self.nextFaceId}  '
                    f'map=({face.x:.2f}, {face.y:.2f}, {face.z:.2f})'
                )
                self.coords.append((self.nextFaceId, face, count, now))
                self.nextFaceId += 1
                del self.pendingCoords[bestIdx]
                # track promotion
                self.get_logger().debug(f'updatePending: pending idx={bestIdx} promoted to confirmed id={self.nextFaceId - 1}')
            return
 
        # Brand-new candidate
        p = Point()
        p.x, p.y, p.z = float(x), float(y), float(z)
        self.pendingCoords.append((p, 1, now))
        self.get_logger().debug(f'updatePending: created new pending candidate at (x={p.x:.2f}, y={p.y:.2f}, z={p.z:.2f})')
 
    def removePending(self, now):
        """Discard pending candidates that haven't been seen recently."""
        keep = []
        for face, count, lastSeen in self.pendingCoords:
            age = (now - lastSeen).nanoseconds
            if age < 0:
                age = 0
            if age <= PENDING_KEEPTIME_NS:
                keep.append((face, count, lastSeen))
        self.pendingCoords = keep
 
    def checkConfirmed(self, now):
        """
        Merge any two confirmed faces that are suspiciously close together –
        this handles the case where the same physical face was confirmed twice
        from slightly different viewpoints.
        """
        if len(self.coords) < 2:
            return
 
        merged = []
        used   = set()
 
        for i in range(len(self.coords)):
            if i in used:
                continue
            faceIdI, faceI, countI, seenI = self.coords[i]
 
            for j in range(i + 1, len(self.coords)):
                if j in used:
                    continue
                faceIdJ, faceJ, countJ, seenJ = self.coords[j]
                xyDist = self._xyDist(faceI.x, faceI.y, faceJ.x, faceJ.y)
                dz     = abs(faceI.z - faceJ.z)
 
                if xyDist <= MERGE_XY_THRESHOLD and dz <= MERGE_Z_THRESHOLD:
                    total   = countI + countJ
                    faceI.x = (faceI.x * countI + faceJ.x * countJ) / total
                    faceI.y = (faceI.y * countI + faceJ.y * countJ) / total
                    faceI.z = (faceI.z * countI + faceJ.z * countJ) / total
                    countI  = total
                    seenI   = seenI if seenI > seenJ else seenJ
                    faceIdI = min(faceIdI, faceIdJ)  # keep lower ID
                    used.add(j)
                    self.get_logger().info(
                        f'Merged face {faceIdJ} into face {faceIdI} (xyDist={xyDist:.3f} m)'
                    )
 
            merged.append((faceIdI, faceI, countI, seenI))
 
        self.coords = merged
 
    # ------------------------------------------------------------------
    # Publisher
    # ------------------------------------------------------------------
 
    def publishFaces_callback(self):
        pub = FaceCoords()
        for faceId, face, count, _ in self.coords:
            if count >= PUBLISH_COUNT_THRESHOLD:
                pub.ids.append(faceId)
                pub.points.append(face)
        self.coordPublisher.publish(pub)
 
 
# ---------------------------------------------------------------------------
 
def main():
    rclpy.init(args=None)
    node = detect_faces()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
 
 
if __name__ == '__main__':
    main()