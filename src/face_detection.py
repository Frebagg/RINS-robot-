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
from rins_robot.msg import FaceCoords
from geometry_msgs.msg import Point
import time


from ultralytics import YOLO
import torch

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
		param_device = self.get_parameter('device').get_parameter_value().string_value
		if param_device != '':
			self.device = param_device
		elif torch.cuda.is_available():
			self.device = '0'
		else:
			self.device = 'cpu'

		self.bridge = CvBridge()
		self.scan = None
		self.tf_buffer = Buffer()
		self.tf_listener = TransformListener(self.tf_buffer, self)

		self.rgb_image_sub = self.create_subscription(Image, "/oakd/rgb/preview/image_raw", self.yolo_callback, qos_profile_sensor_data)
		self.pointcloud_sub = self.create_subscription(PointCloud2, "/oakd/rgb/preview/depth/points", self.checkFace_callback, qos_profile_sensor_data)
		self.model = YOLO("yolov8n.pt")
		if torch.cuda.is_available() and self.device != 'cpu':
			gpu_name = torch.cuda.get_device_name(0)
			self.get_logger().info(f"Prediction with {self.device} ({gpu_name})")
		else:
			self.get_logger().warn("No GPU, using CPU!")

		self.coordPublisher = self.create_publisher(FaceCoords, "/face_coords", 10)
		self.publishTimer = self.create_timer(1/5,self.publishFaces_callback)

		#---------------------------------------------------------------------------------
		#SPREMENLJIVKE ZA ZAZNAVANJE OBRAZOV
		self.faces = []
		self.coords = [] #tukaj notri so (id, Point(), stevilo detekcij, cas_zadnje_detekcije)
		self.pendingCoords = [] # kandidati pred potrditvijo: (Point(), stevilo_detekcij, cas_zadnje_detekcije)
		self.nextFaceId = 1
		self.COUNTTHRESHOLD = 10
		self.CONFIDENCETHRESHOLD = 0.7
		self.MATCH_XY_THRESHOLD = 1.2
		self.MATCH_Z_THRESHOLD = 1.2
		self.PENDING_XY_THRESHOLD = 0.9
		self.PENDING_Z_THRESHOLD = 1.0
		self.MINHITS = 4
		self.MERGE_XY_THRESHOLD = 0.45
		self.MERGE_Z_THRESHOLD = 0.8
		self.KEEPTIME = int(8e9)
		self.counter= 0 #za določanje kdaj se sprozi cleanFaceList
		#---------------------------------------------------------------------------------
		

		self.get_logger().info(f"Face detection node initialized!")

	def yolo_callback(self, data):

		try:
			cvImage = self.bridge.imgmsg_to_cv2(data, "bgr8")
			self.get_logger().debug(f"Running inference on image...")
			res = self.model.predict(cvImage, imgsz=(256, 320), show=False, verbose=False, classes=[0], device=self.device)
			if len(res) == 0:
				return

			boxes = res[0].boxes
			if boxes.xyxy.nelement() == 0:
				return

			bestBbox = None
			bestCenter = None
			bestConf = 0.0
			for i in range(len(boxes)):
				confidence = float(boxes.conf[i])
				if confidence > self.CONFIDENCETHRESHOLD:
					vertices = boxes.xyxy[i]
					cx = int(((vertices[0]+vertices[2])/2))
					cy = int(((vertices[1]+vertices[3])/2))
					self.faces.append((cx, cy, confidence))
					if confidence > bestConf:	
						bestConf = confidence
						bestBbox = vertices
						bestCenter = (cx, cy)
					
			#VIZUALIZACIJA NAJBOLJSE DETEKCIJE
			if bestBbox != None:
				cvImage = cv2.rectangle(cvImage, (int(bestBbox[0]), int(bestBbox[1])), (int(bestBbox[2]), int(bestBbox[3])), self.detection_color, 3)
				cvImage = cv2.circle(cvImage, bestCenter, 5, self.detection_color, -1)

			cv2.imshow("image",cvImage)

			key = cv2.waitKey(1)
			if key==27:
				print("exiting")
				exit()
			
		except CvBridgeError as e:
			print(e)

	def checkFace_callback(self, data):

		height = data.height
		width = data.width
		sourceFrame = data.header.frame_id
		sourceStamp = data.header.stamp
		# get 3-channel representation of the point cloud in numpy format once per callback
		a = pc2.read_points_numpy(data, field_names=("x", "y", "z"))
		a = a.reshape((height, width, 3))

	
		now = self.get_clock().now()

		for cx, cy, conf in self.faces:
			if cx < 0 or cy < 0 or cx >= width or cy >= height:
				continue
			d = a[cy,cx,:]
			if np.isnan(d).any(): #ce kaksen NaN continue
				continue
			detection = self.baseLink2Map(d, sourceFrame, sourceStamp) #map koordinate
				
			if not (detection is None):
				x = detection.point.x
				y = detection.point.y
				z = detection.point.z
				#ce ni ze zaznan obraz potem posodabljamo pending drugac ce je znan pa updatamo confirmed
				if not self.updateConfirmed(x, y, z, now): 
					self.updatePending(x, y, z, now)

		#ce se slucajno kdaj ponesrec zazna npr drevo za en frame potem ta funkcija odstrani
		#stare elemente iz pending seznama
		self.removePending(now) 
		#tega sem dodal ker je se zmeraj zajebavu, v bistvu v confirmed faces ce so preblizu tudi tam notri zdruzi
		self.checkConfirmed(now)
			
		"""if self.counter >= 30:
			self.counter = 0
			self.cleanFaceList()
		self.counter += 1"""
		self.faces.clear() #ko obdelas vse detekcije sprazni list, drugace memleak
		self.get_logger().debug(f"Tracked faces: {len(self.coords)}, pending: {len(self.pendingCoords)}")		

	def xyDist(self, ax, ay, bx, by): #vrne evklidsko razdaljo med tockama
		dx = ax - bx
		dy = ay - by
		return float(np.sqrt(dx * dx + dy * dy))

	#primerja detekcijo z vsemi "ziher" obrazi do zdaj
	def updateConfirmed(self, x, y, z, now): 
		bestIdx = -1
		bestXyDist = float('inf')
		for i, (faceId, face, count, lastSeen) in enumerate(self.coords):
			xyDist = self.xyDist(face.x, face.y, x, y)
			dz = np.abs(face.z - z)
			#ce se ujema z enim zaznanim potem tistega posodobimo in povprecimo koordinate
			if xyDist <= self.MATCH_XY_THRESHOLD and dz <= self.MATCH_Z_THRESHOLD and xyDist < bestXyDist:
				bestXyDist = xyDist
				bestIdx = i

		if bestIdx < 0:
			return False

		faceId, face, count, _ = self.coords[bestIdx]
		face.x = (x + face.x) / 2.0
		face.y = (y + face.y) / 2.0
		face.z = (z + face.z) / 2.0
		self.coords[bestIdx] = (faceId, face, count + 1, now)
		return True

	def updatePending(self, x, y, z, now):
		bestIdx = -1
		bestXyDist = float('inf')
		for i, (face, count, lastSeen) in enumerate(self.pendingCoords):
			xyDist = self.xyDist(face.x, face.y, x, y)
			dz = np.abs(face.z - z)
			if xyDist <= self.PENDING_XY_THRESHOLD and dz <= self.PENDING_Z_THRESHOLD and xyDist < bestXyDist:
				bestXyDist = xyDist
				bestIdx = i

		if bestIdx >= 0:
			face, count, _ = self.pendingCoords[bestIdx]
			face.x = (x + face.x) / 2.0
			face.y = (y + face.y) / 2.0
			face.z = (z + face.z) / 2.0
			count += 1
			self.pendingCoords[bestIdx] = (face, count, now)
			#ce je obraz zaznan dovoljkrat potem se ga premakne v self.coords (real obrazi)
			if count >= self.MINHITS:
				self.coords.append((self.nextFaceId, face, count, now))
				self.nextFaceId += 1
				del self.pendingCoords[bestIdx]
			return

		p = Point()
		p.x = float(x)
		p.y = float(y)
		p.z = float(z)
		self.pendingCoords.append((p, 1, now))

	def removePending(self, now):
		keep = []
		for face, count, lastSeen in self.pendingCoords:
			age = (now - lastSeen).nanoseconds
			if age < 0:
				age = 0
			if age <= self.KEEPTIME:
				keep.append((face, count, lastSeen))
		self.pendingCoords = keep

	def checkConfirmed(self, now):
		if len(self.coords) < 2:
			return

		merged = []
		used = []
		for i in range(len(self.coords)):
			if i in used:
				continue
			faceIdI, faceI, countI, seenI = self.coords[i]
			for j in range(i + 1, len(self.coords)):
				if j in used:
					continue
				faceIdJ, faceJ, countJ, seenJ = self.coords[j]
				xyDist = self.xyDist(faceI.x, faceI.y, faceJ.x, faceJ.y)
				dz = np.abs(faceI.z - faceJ.z)
				if xyDist <= self.MERGE_XY_THRESHOLD and dz <= self.MERGE_Z_THRESHOLD:
					total = countI + countJ
					faceI.x = (faceI.x * countI + faceJ.x * countJ) / total
					faceI.y = (faceI.y * countI + faceJ.y * countJ) / total
					faceI.z = (faceI.z * countI + faceJ.z * countJ) / total
					countI = total
					seenI = now if seenI < seenJ else seenI
					if faceIdJ < faceIdI:
						faceIdI = faceIdJ
					used.append(j)
			merged.append((faceIdI, faceI, countI, seenI))

		self.coords = merged

	def baseLink2Map(self, data, sourceFrame, sourceStamp):
		detection = PointStamped()
		detection.header.frame_id = sourceFrame
		detection.header.stamp = sourceStamp
		detection.point.x = float(data[0])
		detection.point.y = float(data[1])
		detection.point.z = float(data[2])
		sourceTime = rclpy.time.Time.from_msg(sourceStamp)
		timeout = Duration(seconds=0.1)

		try:
			trans = self.tf_buffer.lookup_transform("map", sourceFrame, sourceTime, timeout)
			map_detection = tfg.do_transform_point(detection, trans)
		except TransformException as te: #to je neki s stampi za cas ker lahko zajebava
			if "extrapolation into the future" in str(te).lower():
				try:
					latest = rclpy.time.Time()
					trans = self.tf_buffer.lookup_transform("map", sourceFrame, latest, timeout)
					map_detection = tfg.do_transform_point(detection, trans)
					return map_detection
				except TransformException as te2:
					self.get_logger().debug(f"Could not get fallback transform: {te2}")
					return None
			self.get_logger().debug(f"Could not get the transform: {te}")
			return None
		return map_detection
		
	"""def cleanFaceList(self):
		now = self.get_clock().now()
		keep = []
		for faceId, face, count, lastSeen in self.coords:
			ageNs = (now - lastSeen).nanoseconds
			if ageNs < 0:
				ageNs = 0
			if ageNs <= 5000000000:  # 5s
				keep.append((faceId, face, count, lastSeen))
		self.coords = keep
		"""


	def publishFaces_callback(self):
		pub = FaceCoords()
		for faceId,face,count,_ in self.coords:
			if count >= self.COUNTTHRESHOLD:
				pub.ids.append(faceId)
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