#!/usr/bin/python3

import rclpy
from rclpy.node import Node
import cv2, math
import numpy as np
import tf2_ros

from sensor_msgs.msg import Image
from geometry_msgs.msg import PointStamped, Vector3, Pose
from cv_bridge import CvBridge, CvBridgeError
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy

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
        self.center_thr = 10
        
        # depth thresholds
        self.latest_depth = None
        self.depth_thr = 0.10   # metri
        #self.min_valid_depth = 0.05
        #self.max_valid_depth = 5.0

        # An object we use for converting images between ROS format and OpenCV format
        self.bridge = CvBridge()

        # Subscribe to the image and/or depth topic
        self.image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.image_callback, 1)
        self.depth_sub = self.create_subscription(Image, "/oakd/rgb/preview/depth", self.depth_callback, 1)

        cv2.namedWindow("Binary Image", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected contours", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Detected rings", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Depth window", cv2.WINDOW_NORMAL)        

    def image_callback(self, data, depth):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        # Tranform image to grayscale
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

        # Binarize the image
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 30)
        cv2.imshow("Binary Image", thresh)

        # Extract contours
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Example of how to draw the contours, only for visualization purposes
        cv2.drawContours(gray, contours, -1, (255, 0, 0), 1)
        cv2.imshow("Detected contours", gray)

        # Fit elipses to all extracted contours
        elps = []
        for cnt in contours:
            if cnt.shape[0] >= 20:
                ellipse = cv2.fitEllipse(cnt)

                # filter ellipses that are too eccentric
                e = ellipse[1]
                ecc1 = e[0]
                ecc2 = e[1]

                ratio = ecc1/ecc2 if ecc1>ecc2 else ecc2/ecc1
                if ratio<=self.ratio_thr and ecc1<self.ecc_thr and ecc2<self.ecc_thr:

                    elps.append(ellipse)

        # Find two elipses with same centers
        candidates1 = []
        for n in range(len(elps)):
            e1 = elps[n]

            # display candidates
            cv2.ellipse(cv_image, e1, (255, 255, 0), 1)
            cv2.circle(cv_image, (int(e1[0][0]), int(e1[0][1])), 1, (255, 255, 0), -1)

            for m in range(n + 1, len(elps)):
                # e[0] is the center of the ellipse (x,y), e[1] are the lengths of major and minor axis (major, minor), e[2] is the rotation in degrees
                
                e1 = elps[n]
                e2 = elps[m]
                dist = np.sqrt(((e1[0][0] - e2[0][0]) ** 2 + (e1[0][1] - e2[0][1]) ** 2))
                angle_diff = np.abs(e1[2] - e2[2])

                # The centers of the two elipses should be within 10 pixels of each other

                if dist >= self.center_thr:
                    continue

                e1_minor_axis = e1[1][0]
                e1_major_axis = e1[1][1]

                e2_minor_axis = e2[1][0]
                e2_major_axis = e2[1][1]

                if e1_major_axis>=e2_major_axis and e1_minor_axis>=e2_minor_axis: # the larger ellipse should have both axis larger
                    le = e1 # e1 is larger ellipse
                    se = e2 # e2 is smaller ellipse
                elif e2_major_axis>=e1_major_axis and e2_minor_axis>=e1_minor_axis:
                    le = e2 # e2 is larger ellipse
                    se = e1 # e1 is smaller ellipse
                else:
                    continue # if one ellipse does not contain the other, it is not a ring
                    
                candidates1.append((le,se)) #tuki je blo prej e1, e2 upam da to nc ne sfuka
        
        candidates2 = []
        for le, se in candidates1:
            
            if self.latest_depth is None:
                continue
            h, w = self.latest_depth.shape[:2]
            
            
            cx = le[0][0]
            cy = le[0][1]
            
            if not (0 <= cx < w and 0 <= cy < h):
                continue
            
            #center = self.latest_depth[int(cy), int(cx)] #((center_x, center_y), (axis1, axis2), angle) , numpy ma [y,x] #easy verzija
            
            patch = self.latest_depth[(cy-2):(cy+3),(cx-2):(cx+3)] #5x5 pixel del za sredino
            vals = patch[np.isfinite(patch)]
            #vals = vals[(vals > self.min_valid_depth) & (vals < self.max_valid_depth)]

            if len(vals) == 0:
                continue
            center = int(np.median(vals)) # avg?
            
            if self.is_valid_depth(center): # not?
                continue
            
            #pol osi elips
            outer_a = max(le[1][0], le[1][1]) / 2.0
            outer_b = min(le[1][0], le[1][1]) / 2.0

            inner_a = max(se[1][0], se[1][1]) / 2.0
            inner_b = min(se[1][0], se[1][1]) / 2.0
            
            ring_depths = []
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False): # 24 razlicnih tock v sredini ringa
                # polmer nekje med notranjo in zunanjo elipso
                a = (outer_a + inner_a) / 2.0
                b = (outer_b + inner_b) / 2.0

                x = int(cx + a * np.cos(angle))
                y = int(cy + b * np.sin(angle))

                if 0 <= x < w and 0 <= y < h:
                    d = self.latest_depth[y, x]
                    if self.is_valid_depth(d):
                        ring_depths.append(d)

            # if len(ring_depths) < 8: # myb
            #     continue

            ring_depth = float(np.median(ring_depths)) # average? al pa use?
            
            if center <= ring_depth + self.depth_thr:
                continue
            
            candidates2.append((le,se))
        
        print("Processing is done! found", len(candidates2), "candidates for rings")

        # Plot the rings on the image
        for c in candidates2:

            # the centers of the ellipses
            e1 = c[0]
            e2 = c[1]

            # drawing the ellipses on the image
            cv2.ellipse(cv_image, e1, (0, 255, 0), 2)
            cv2.ellipse(cv_image, e2, (0, 255, 0), 2)

            # Get a bounding box, around the first ellipse ('average' of both elipsis)
            size = (e1[1][0]+e1[1][1])/2
            center = (e1[0][1], e1[0][0])

            x1 = int(center[0] - size / 2)
            x2 = int(center[0] + size / 2)
            x_min = x1 if x1>0 else 0
            x_max = x2 if x2<cv_image.shape[0] else cv_image.shape[0]

            y1 = int(center[1] - size / 2)
            y2 = int(center[1] + size / 2)
            y_min = y1 if y1 > 0 else 0
            y_max = y2 if y2 < cv_image.shape[1] else cv_image.shape[1]

        cv2.imshow("Detected rings", cv_image)
        cv2.waitKey(1)

    def is_valid_depth(self, d):
        return not (np.isnan(d) or d == 0) # and self.min_valid_depth < d < self.max_valid_depth # za zdaj pustimo
    
    def depth_callback(self,data):

        try:
            depth_image = self.bridge.imgmsg_to_cv2(data, "32FC1")
        except CvBridgeError as e:
            print(e)

        depth_image[depth_image==np.inf] = 0
        
        self.latest_depth = depth_image.copy() 
        
        # Do the necessairy conversion so we can visuzalize it in OpenCV
        image_1 = depth_image / 65536.0 * 255
        image_1 = image_1/np.max(image_1)*255

        image_viz = np.array(image_1, dtype= np.uint8)

        cv2.imshow("Depth window", image_viz)
        cv2.waitKey(1)


def main():

    rclpy.init(args=None)
    rd_node = RingDetector()

    rclpy.spin(rd_node)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()