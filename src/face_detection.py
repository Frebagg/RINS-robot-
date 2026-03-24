#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data, QoSReliabilityPolicy

from sensor_msgs.msg import Image, PointCloud2
from sensor_msgs_py import point_cloud2 as pc2

from visualization_msgs.msg import Marker

from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import tf2_geometry_msgs as tfg
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from geometry_msgs.msg import PointStamped
from rclpy.duration import Duration
from RINS_robot.msg import FaceCoords
from geometry_msgs.msg import Point
import time


from ultralytics import YOLO

# from rclpy.parameter import Parameter
# from rcl_interfaces.msg import SetParametersResult

class detect_faces(Node):

	def __init__(self):
		super().__init__('detect_faces')

		self.declare_parameters(
			namespace='',
			parameters=[
				('device', ''),
		])
		
		self.detection_color = (0,0,255)
		self.device = self.get_parameter('device').get_parameter_value().string_value

		self.bridge = CvBridge()
		self.scan = None
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.yolo_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.checkFace_callback, qos_profile_sensor_data)
		self.model = YOLO("yolov8n.pt")

		self.coordPublisher = self.create_publisher(FaceCoords, "/face_coords", 10)
		self.publishTimer = self.create_timer(1/5,self.publishFaces_callback)


		self.faces = []
		self.coords = [] #tukaj notri sta Point() in stevilo detekcij zanj
		self.COUNTTHRESHOLD = 7 #za spreminjat,treba testirat
		self.CONFIDENCETHRESHOLD = 0.5
		self.lastSeen = []
		self.counter= 0 #za določanje kdaj se sprozi cleanFaceList
		#self.listProcessed = True

		self.get_logger().info(f"Face detection node initialized!")

	def yolo_callback(self, data):

		try:
			cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.get_logger().debug(f"Running inference on image...")
			res = self.model.predict(cv_image, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
			if len(res) == 0:
				return

			boxes = res[0].boxes
			if boxes.xyxy.nelement() == 0:
				return

			best_bbox = None
			best_center = None
			bestConf = 0.0
			for i in range(len(boxes)):
				confidence = boxes.conf[i]
				if confidence > self.CONFIDENCETHRESHOLD:
					vertices = boxes.xyxy[i]
					cx = int(((vertices[0]+vertices[2])/2))
					cy = int(((vertices[1]+vertices[3])/2))
					self.faces.append((cx,cy))
					if confidence > bestConf:
						bestConf = confidence
						best_bbox = vertices
						best_center = (cx, cy)
					
			#VIZUALIZACIJA NAJBOLJSE DETEKCIJE
			if best_bbox != None:
				cv_image = cv2.rectangle(cv_image, (int(best_bbox[0]), int(best_bbox[1])), (int(best_bbox[2]), int(best_bbox[3])), self.detection_color, 3)
				cv_image = cv2.circle(cv_image, best_center, 5, self.detection_color, -1)
				cv2.imshow("image", cv_image)

			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def checkFace_callback(self, data):

		# get point cloud attributes
		height = data.height
		width = data.width
		# get 3-channel representation of the point cloud in numpy format once per callback
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))

	
		for cx,cy in self.faces:
			if cx < 0 or cy < 0 or cx >= width or cy >= height:
				continue
			d = a[cy,cx,:]
			if np.isnan(d).any(): #ce kaksen NaN continue
				continue
			detection = self.baseLink2Map(d) #map koordinate
				
			if not (detection is None):
				x = detection.point.x
				y = detection.point.y
				z = detection.point.z
				newFace = True
				for i,(face,count) in enumerate(self.coords):
					xx = face.x
					yy = face.y
					zz = face.z
					if ( np.abs(xx-x) <0.5 and np.abs(yy-y)<0.5 and np.abs(zz-z) < 1 ):
						newFace = False
						face.x = (x+xx)/2 #povpreci lokacijo ponovno zaznanih
						face.y = (y+yy)/2
						face.z = (z+zz)/2
						self.coords[i] = (face, count+1)
						self.lastSeen[i] = self.get_clock().now()
				if newFace:
					self.coords.append((detection.point,1))
					self.lastSeen.append(self.get_clock().now())
			
		if self.counter >= 30:
			self.counter = 0
			self.cleanFaceList()
		self.counter += 1
		self.faces.clear() #ko obdelas vse detekcije sprazni list, drugace memleak
		

	def baseLink2Map(self, data):
		detection = PointStamped()
		detection.header.frame_id = "base_link"
		detection.header.stamp = self.get_clock().now().to_msg()
		detection.point.x = float(data[0])
		detection.point.y = float(data[1])
		detection.point.z = float(data[2])
		time_now = rclpy.time.Time()
		timeout = Duration(seconds=0.1)

		try:
			trans = self.tf_buffer.lookup_transform("map", "base_link", time_now, timeout)
			map_detection = tfg.do_transform_point(detection, trans)
		except TransformException as te:
			self.get_logger().info(f"Could not get the transform: {te}")
			return None
		return map_detection
		
	def cleanFaceList(self):
		if len(self.coords) != len(self.lastSeen):
			return
		ixList = []
		now = self.get_clock().now()
		for i in range(len(self.lastSeen)):
			if (now - self.lastSeen[i]).nanoseconds > 5000000000:  # 5s
				ixList.append(i)
		offset = 0
		for idx in ixList:
			del self.coords[idx - offset]
			del self.lastSeen[idx - offset]
			offset += 1


	def publishFaces_callback(self):
		pub = FaceCoords()
		for face,count in self.coords:
			if count >= self.COUNTTHRESHOLD:
				pub.points.append(face)
		self.coordPublisher.publish(pub)


def main():
	rclpy.init(args=None)
	node = detect_faces()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()