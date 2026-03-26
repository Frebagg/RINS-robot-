#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker
import numpy as np
from rins_robot.msg import FaceCoords
from geometry_msgs.msg import Point


class visualizeMarkers(Node):
    
    def __init__(self):
        super().__init__("visualizeMarkers")


        self.faceCoordClient = self.create_subscription(FaceCoords,"/face_coords",self.manageFaceMarkers_callback,10)
        self.faceMarkerIds = []

        self.ringCoordClient = self.create_subscription(FaceCoords,"/ring_coords",self.manageRingMarkers_callback,10)
        self.ringMarkerIds = []

        self.markerPublisher = self.create_publisher(Marker,"/face_marker",10)
        self.markerOffset = 100

        self.get_logger().info(f"Visualisation node initialized!")



    def manageFaceMarkers_callback(self,msg):
        if len(msg.points) != len(msg.ids):
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
            scale = 0.5
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
            text.pose.position.z = face.z + 0.5
            text.scale.z = 0.5
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"Face {id}"
            text.pose.orientation.w = 1.0
            text.action = Marker.ADD

            self.markerPublisher.publish(text)
            self.markerPublisher.publish(marker)

        
    def manageRingMarkers_callback(self,msg):
        if len(msg.points) != len(msg.ids):
            return

        for ring, id in zip(msg.points, msg.ids):
            if id in self.ringMarkerIds: #ce je ta obraz ze markiran ignoriraj
                 continue
            marker = Marker()
            marker.header.frame_id = "map"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "rings"
            marker.id = id + self.markerOffset
            marker.type = Marker.CUBE
            scale = 0.5
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
            self.ringMarkerIds.append(id)

            text = Marker()
            text.header.frame_id = "map"
            text.header.stamp = self.get_clock().now().to_msg()
            text.ns = "rings_label"
            text.id = id + self.markerOffset
            text.type = Marker.TEXT_VIEW_FACING
            text.pose.position.x = ring.x
            text.pose.position.y = ring.y
            text.pose.position.z = ring.z + 0.5
            text.scale.z = 0.5
            text.color.r = 1.0
            text.color.g = 1.0
            text.color.b = 1.0
            text.color.a = 1.0
            text.text = f"Ring {id}"
            text.pose.orientation.w = 1.0
            text.action = Marker.ADD

            self.markerPublisher.publish(text)
            self.markerPublisher.publish(marker)


def main():
	print('Visualisation Node starting.')

	rclpy.init(args=None)
	node = visualizeMarkers()
	rclpy.spin(node)
	node.destroy_node()
	rclpy.shutdown()

if __name__ == '__main__':
	main()