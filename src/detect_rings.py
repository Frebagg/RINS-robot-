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
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSHistoryPolicy, QoSReliabilityPolicy, qos_profile_sensor_data

import tf2_geometry_msgs as tfg
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from rclpy.duration import Duration


class RingDetector(Node):
    def __init__(self):
        super().__init__('ring_detector')

        # Parametri za detekcijo elips
        self.min_contour_points = 20   # min št. točk konture za fitanje elipse
        self.ratio_thr          = 1.5  # max razmerje osi (filtrira podolgovate elipse)
        self.ecc_thr            = 100  # max absolutna velikost osi [px]

        # Parametri za veljavnost depth vrednosti
        self.min_valid_depth = 0.05   # bliže od tega je šum [m]
        self.max_valid_depth = 5.0    # dlje od tega ne zaznamo [m]

        # Parametri za binarizacijo depth slike
        # Samo piksli v tem razponu globine postanejo beli v binarni sliki.
        # S tem izrežemo ozadje in tla, ki so preveč daleč ali preblizu.
        self.binary_depth_min = 0.5   # [m]
        self.binary_depth_max = 3.5   # [m]

        # Parametri za validacijo obroča z depth
        # inner_scale: notranja elipsa je inner_scale-krat manjša od zunanje.
        # Prostor znotraj notranje elipse = luknja obroča.
        self.inner_scale             = 0.45  # razmerje notranje elipse
        self.depth_thr               = 0.10  # luknja mora biti vsaj toliko dlje od obroča [m]
        self.min_ring_depth_points   = 12    # min veljavnih depth vzorcev na obroču
        self.min_center_depth_points = 8     # min veljavnih depth vzorcev v luknji

        # Parametri za združevanje zaznav v tabelo obročev
        # Če je nova zaznava bliže od teh pragov obstoječemu obroču,
        # jo združimo z njim (povprečenje pozicije) namesto da dodamo novega.
        self.merge_distance_xy = 0.5
        self.merge_distance_z  = 0.5

        # Parametri za stanje
        self.latest_depth = None  # zadnja depth slika (float32, metri)
        self.rings_2d     = []    # zaznave iz trenutnega frame-a: [(ellipse, color), ...]
        self.coords       = []    # tabela unikatnih obročev: [(id, Point, color), ...]
        self.next_ring_id = 1     # naraščajoči ID za nove obroče

        # ROS infrastruktura
        self.bridge = CvBridge()

        # TF za transformacijo točk iz kamere v map frame
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Naročnine (subscribers)
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)
        self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.pointcloud_callback, qos_profile_sensor_data)

        # Objava tabele obročev na /ring_coords vsakih 0.2 s
        self.coord_publisher = self.create_publisher(RingCoords, "/ring_coords", 10)
        self.create_timer(1 / 5, self.publish_rings_callback)

        # OpenCV okna za vizualizacijo
        cv2.namedWindow("Binary Image",    cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings",  cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window",    cv2.WINDOW_NORMAL)


    def image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # Počakamo da depth slika prispe pred prvo detekcijo
        if self.latest_depth is None:
            cv2.imshow("Detected rings", cv_image)
            cv2.waitKey(1)
            return

        depth = self.latest_depth.copy()

        # Varnostni check: RGB in depth morata imeti enako resolucijo
        # if cv_image.shape[:2] != depth.shape[:2]:
        #     self.get_logger().warn("RGB in depth resoluciji se ne ujemata!")
        #     return

        # Počistimo zaznave iz prejšnjega frame-a
        self.rings_2d.clear()

        # 1. Binarizacija depth slike
        # Naredimo binarno sliko kjer so beli samo piksli z globino v
        # želenem razponu [binary_depth_min, binary_depth_max].
        # S tem dobimo masko predmetov na primernih razdaljah.
        # valid = np.isfinite(depth) & (depth > self.min_valid_depth) & (depth < self.max_valid_depth)
        thresh = np.zeros(depth.shape, dtype=np.uint8)
        thresh[depth & (depth > self.binary_depth_min) & (depth < self.binary_depth_max)] = 255

        # Morphološke operacije: OPEN odstrani šum, CLOSE zapolni luknje v konturah
        kernel = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN,  kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        cv2.imshow("Binary Image", thresh)

        # 2. Iskanje kontur in fitanje elips
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        elps = []
        for cnt in contours:
            # Elipso lahko fitamo samo če ima kontura dovolj točk
            if cnt.shape[0] < self.min_contour_points:
                continue

            ellipse = cv2.fitEllipse(cnt)
            ax1, ax2 = ellipse[1]  # dolžini osi elipse

            if ax1 <= 0 or ax2 <= 0:
                continue

            # Filtriramo elipse ki so preveč podolgovate (razmerje osi > ratio_thr)
            # ali preveč velike (verjetno so ozadje, ne obroč)
            ratio = ax1 / ax2 if ax1 > ax2 else ax2 / ax1
            if ratio <= self.ratio_thr and ax1 < self.ecc_thr and ax2 < self.ecc_thr:
                elps.append(ellipse)

        # 3. Depth validacija vsake elipse
        vis = cv_image.copy()
        candidates = []

        for ellipse in elps:
            # Pokažemo vse kandidate z rumeno barvo
            cv2.ellipse(vis, ellipse, (255, 255, 0), 1)

            is_ring, inner_ellipse = self.ellipse_is_ring(depth, ellipse)
            if is_ring:
                candidates.append((ellipse, inner_ellipse))

        self.get_logger().info(f"Najdenih obročev: {len(candidates)}")

        # 4. Izris rezultatov in shranjevanje za pointcloud callback
        for ellipse, inner_ellipse in candidates:
            cv2.ellipse(vis, ellipse,       (0, 255, 0), 2)  # zunanja elipsa
            cv2.ellipse(vis, inner_ellipse, (0, 255, 0), 1)  # notranja (luknja)

            color_name = self.classify_ring_color(cv_image, ellipse)

            cx, cy = int(ellipse[0][0]), int(ellipse[0][1])
            cv2.circle(vis, (cx, cy), 3, (0, 0, 255), -1)
            cv2.putText(vis, color_name, (cx + 10, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Shranimo 2D zaznavo za pointcloud_callback, ki bo izračunal 3D pozicijo
            self.rings_2d.append((ellipse, color_name))

        cv2.imshow("Detected rings", vis)
        cv2.waitKey(1)

    def depth_callback(self, data):
        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        # Neveljavne vrednosti (inf, nan) postavimo na 0
        depth_image = depth_image.astype(np.float32)
        depth_image[~np.isfinite(depth_image)] = 0

        self.latest_depth = depth_image.copy()

        # Vizualizacija depth slike v sivini
        cv2.imshow("Depth window", self.depth_to_gray(depth_image))
        cv2.waitKey(1)

    def pointcloud_callback(self, data):
        """
        Sprejme pointcloud in za vsako 2D zaznavo obroča izračuna 3D pozicijo.
        Nato jo transformira v map frame in doda/združi v tabelo obročev.
        """
        # Pretvorimo PointCloud2 v numpy array oblike (height, width, 3)
        pts = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
        pts = pts.reshape((data.height, data.width, 3))

        for ellipse, color in self.rings_2d:
            if color == "unknown":
                continue

            # Izračunamo 3D pozicijo obroča iz pointclouda
            point_3d = self.get_ring_3d_point(pts, ellipse)
            if point_3d is None:
                continue

            # Transformiramo točko iz camera frame v map frame
            point_map = self.camera_to_map(point_3d)
            if point_map is None:
                continue

            x, y, z = point_map.point.x, point_map.point.y, point_map.point.z

            if not np.isfinite([x, y, z]).all():
                continue

            self.get_logger().info(f"Nova zaznava: {color}  x={x:.2f} y={y:.2f} z={z:.2f}")

            # Preverimo ali se zaznava ujema z že obstoječim obroč (merge)
            merged = False
            for i, (ring_id, ring_pt, ring_color) in enumerate(self.coords):
                if ring_color != color:
                    continue
                if (abs(ring_pt.x - x) < self.merge_distance_xy and abs(ring_pt.y - y) < self.merge_distance_xy and abs(ring_pt.z - z) < self.merge_distance_z):
                    # Povprečimo pozicijo z novo zaznavo (running average)
                    ring_pt.x = (ring_pt.x + x) / 2.0
                    ring_pt.y = (ring_pt.y + y) / 2.0
                    ring_pt.z = (ring_pt.z + z) / 2.0
                    self.coords[i] = (ring_id, ring_pt, ring_color)
                    self.get_logger().info(f"  → združeno z obročem #{ring_id}")
                    merged = True
                    break

            if not merged:
                # Dodamo nov obroč v tabelo
                p = Point()
                p.x, p.y, p.z = x, y, z
                self.coords.append((self.next_ring_id, p, color))
                self.get_logger().info(f"  → nov obroč #{self.next_ring_id}: {color}")
                self.next_ring_id += 1

        # Počistimo 2D zaznave — pointcloud_callback jih je že obdelal
        self.rings_2d.clear()

    def publish_rings_callback(self):
        """Periodično objavlja tabelo vseh znanih obročev na /ring_coords."""
        msg = RingCoords()
        for ring_id, ring_pt, color in self.coords:
            if not np.isfinite([ring_pt.x, ring_pt.y, ring_pt.z]).all():
                continue
            msg.ids.append(ring_id)
            msg.points.append(ring_pt)
            msg.colors.append(color)
        self.coord_publisher.publish(msg)

    def ellipse_is_ring(self, depth, ellipse):
        """
        Preveri ali je elipsa dejansko obroč z luknjo.

        Logika:
          1. Naredimo masko za pas obroča (med zunanjo in notranjo elipso).
          2. Naredimo masko za luknjo (notranjost notranje elipse).
          3. Če je mediana globine v luknji večja od mediane globine obroča
             za vsaj depth_thr, potem je to obroč (luknja je dlje = gledamo skozi).
          4. Če luknja nima dovolj veljavnih depth točk (morda je za robom slike),
             privzamemo da je obroč — boljše lažno pozitivno kot da bi ga zgrešili.

        Vrne: (is_ring: bool, inner_ellipse)
        """
        ring_mask, inner_mask, inner_ellipse = self.make_ring_mask(depth.shape, ellipse)

        # Depth vrednosti na pasu obroča (material)
        ring_depths = depth[ring_mask > 0]
        ring_depths = self.filter_valid_depths(ring_depths)
        if len(ring_depths) < self.min_ring_depth_points:
            return False, inner_ellipse
        ring_depth = float(np.median(ring_depths))

        # Depth vrednosti v luknji
        center_depths = depth[inner_mask > 0]
        center_depths = self.filter_valid_depths(center_depths)

        # Premalo točk v luknji → verjetno je luknja izven dosega senzorja,
        # kar pomeni da je globoka/prazna → štejemo kot obroč
        if len(center_depths) < self.min_center_depth_points:
            return True, inner_ellipse

        center_depth = float(np.median(center_depths))

        # Luknja mora biti dlje od materiala obroča
        is_ring = center_depth > ring_depth + self.depth_thr
        return is_ring, inner_ellipse

    def get_ring_3d_point(self, pointcloud, ellipse):
        """
        Iz pointclouda izračuna 3D koordinate obroča (mediana točk na pasu obroča).
        Vrne numpy array [x, y, z] v camera frame ali None če ni dovolj točk.
        """
        h, w, _ = pointcloud.shape
        ring_mask, _, _ = self.make_ring_mask((h, w), ellipse)

        ys, xs = np.where(ring_mask > 0)
        if len(xs) == 0:
            return None

        pts = pointcloud[ys, xs, :]
        # Ohranimo samo končne, neničelne točke
        pts = pts[np.isfinite(pts).all(axis=1)]
        pts = pts[np.linalg.norm(pts, axis=1) > 1e-6]

        if len(pts) == 0:
            return None

        median_pt = np.median(pts, axis=0)
        return median_pt if np.isfinite(median_pt).all() else None

    def camera_to_map(self, point_3d):
        """
        Transformira točko iz 'oakd_rgb_camera_optical_frame' v 'map' frame z TF.
        Vrne PointStamped v map frame ali None če transformacija ni na voljo.
        """
        p = PointStamped()
        p.header.frame_id = "oakd_rgb_camera_optical_frame"
        p.header.stamp    = self.get_clock().now().to_msg()
        p.point.x = float(point_3d[0])
        p.point.y = float(point_3d[1])
        p.point.z = float(point_3d[2])

        try:
            transform = self.tf_buffer.lookup_transform(
                "map", p.header.frame_id,
                rclpy.time.Time(),
                timeout=Duration(seconds=0.2)
            )
            return tfg.do_transform_point(p, transform)
        except Exception as e:
            self.get_logger().warn(f"TF transformacija neuspešna: {e}")
            return None

    def make_ring_mask(self, shape, ellipse):
        """
        Naredi dve maski iz ene elipse:
          - ring_mask:  pas med zunanjo in notranjo elipso (material obroča)
          - inner_mask: notranjost notranje elipse (luknja)

        Notranja elipsa je skalirana z inner_scale (npr. 0.45 = 45% velikosti).
        """
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

    def filter_valid_depths(self, depths):
        """Filtrira depth vrednosti: samo končne vrednosti v [min, max] razponu."""
        depths = depths[np.isfinite(depths)]
        return depths[(depths > self.min_valid_depth) & (depths < self.max_valid_depth)]

    def depth_to_gray(self, depth):
        """Pretvori depth sliko (float32, metri) v sivino za vizualizacijo."""
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
        norm[d == 0] = 1.0  # neveljavne točke → bele (daleč)
        return (norm * 255).astype(np.uint8)

    def classify_ring_color(self, bgr_image, ellipse):
        """
        Klasificira barvo obroča iz RGB slike z analizo HSV vrednosti.

        Postopek:
          1. Vzamemo piksle na pasu obroča (ring_mask).
          2. Pretvorimo v HSV prostor.
          3. Ločimo barvne piksle (S > 60) od sivih/belih/črnih.
          4. Iz mediane odtenka (H) določimo barvo.
        """
        ring_mask, _, _ = self.make_ring_mask(bgr_image.shape, ellipse)

        hsv         = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
        ring_pixels = hsv[ring_mask > 0]

        if len(ring_pixels) < 20:
            return "unknown"

        s = ring_pixels[:, 1]
        v = ring_pixels[:, 2]

        # Ločimo barvne piksle od akromatskih (siva, bela, črna)
        colored = ring_pixels[s > 60]

        if len(colored) < 10:
            # Pretežno akromatska barva → določimo po svetlosti
            median_v = float(np.median(v))
            if   median_v > 180: 
                return "white"
            elif median_v < 60:  
                return "black"
            else:                
                return "gray"

        # H je v razponu 0–179 (OpenCV konvencija)
        median_h = float(np.median(colored[:, 0]))

        if   median_h < 10 or median_h >= 170: 
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


def main():
    rclpy.init(args=None)
    rclpy.spin(RingDetector())
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
