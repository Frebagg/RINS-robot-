#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy

from visualization_msgs.msg import Marker
import numpy as np
from rins_robot.msg import FaceCoords, RingCoords
from geometry_msgs.msg import Point


class visualizeMarkers(Node):

    def __init__(self):
        super().__init__("visualizeMarkers")


        self.faceCoordClient = self.create_subscription(FaceCoords,"/face_coords",self.manageFaceMarkers_callback,10)
        self.faceMarkerIds = []

        self.ringCoordClient = self.create_subscription(RingCoords,"/ring_coords",self.manageRingMarkers_callback,10)
        self.ringMarkerIds = []

        marker_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=100,
        )

        self.faceMarkerPublisher = self.create_publisher(Marker,"/face_marker",marker_qos)
        self.ringMarkerPublisher = self.create_publisher(Marker,"/ring_marker",marker_qos)
        self.markerOffset = 100

        self.get_logger().info(f"Visualisation node initialized!")



    def manageFaceMarkers_callback(self,msg):
        if len(msg.points) != len(msg.ids):
            self.get_logger().info("Length mismatch in face_coords!")
            return

        for face, id in zip(msg.points, msg.ids):
            if id in self.faceMarkerIds: #ce je ta obraz ze markiran ignoriraj
                 continue
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "faces"
            marker.id = id
            marker.type = Marker.SPHERE
            scale = 0.25
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.pose.position = face
            marker.color.r = 1.0
            marker.color.g = 0.
            marker.color.b = 0.
            marker.color.a = 1.0
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            self.faceMarkerIds.append(id)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "faces_label"
            text.id = id
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.x = face.x
            text.pose.position.y = face.y
            text.pose.position.z = face.z + 0.8
            text.scale.z = 0.5
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"Face {id}"
            text.pose.orientation.w = 1.0
            text.action = Marker.ADD

            self.faceMarkerPublisher.publish(text)
            self.faceMarkerPublisher.publish(marker)
            self.get_logger().info(f"Published face {id}!")
            #self.get_logger().info(f"P")


    def manageRingMarkers_callback(self,msg):
        if len(msg.points) != len(msg.ids) or len(msg.points) != len(msg.colors):
            self.get_logger().info("Length mismatch in ring_coords!")
            return

        for ring, ring_id, color in zip(msg.points, msg.ids, msg.colors):
            if ring_id in self.ringMarkerIds: #ce je ta ring ze markiran ignoriraj
                 continue
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rings"
            marker.id = ring_id + self.markerOffset
            marker.type = Marker.CUBE
            scale = 0.25
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale
            marker.pose.position = ring
            marker.color.r = 0.0
            marker.color.g = 0.
            marker.color.b = 1.0
            marker.color.a = 1.0
            marker.action = Marker.ADD
            marker.pose.orientation.w = 1.0
            self.ringMarkerIds.append(ring_id)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "rings_label"
            text.id = ring_id + self.markerOffset
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.x = ring.x
            text.pose.position.y = ring.y
            text.pose.position.z = ring.z + 0.8
            r, g, b = self.color_to_rgb(color)
            text.scale.z = 0.25
            text.color.r = r
            text.color.g = g
            text.color.b = b
            text.color.a = 1.0
            text.text = f"Ring {ring_id}"
            text.pose.orientation.w = 1.0
            text.action = Marker.ADD

            self.ringMarkerPublisher.publish(text)
            self.ringMarkerPublisher.publish(marker)
            self.get_logger().info(f"Published ring {ring_id}!")

    def color_to_rgb(self, color_name):
        if color_name == "red":
            return (1.0, 0.0, 0.0)
        elif color_name == "green":
            return (0.0, 1.0, 0.0)
        elif color_name == "blue":
            return (0.0, 0.0, 1.0)
        elif color_name == "yellow":
            return (1.0, 1.0, 0.0)
        elif color_name == "black":
            return (0.1, 0.1, 0.1)
        elif color_name == "white":
            return (1.0, 1.0, 1.0)
        elif color_name == "gray":
            return (0.5, 0.5, 0.5)
        elif color_name == "orange":
            return (1.0, 0.647, 1.0)
        elif color_name == "purple":
            return (0.5, 0.0, 0.5)
        return (0.5, 0.5, 0.5)

def main():
	print('Visualisation Node starting.')

	rclpy.init(args=None)
	node = visualizeMarkers()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
    main()