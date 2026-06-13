#!/usr/bin/env python3
import argparse

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class SharedCameraPublisher(Node):
    def __init__(self, args):
        super().__init__("shared_camera_publisher")
        self.args = args
        self.pub = self.create_publisher(Image, args.topic, 10)

        source_label = ""
        if args.gst:
            pipeline = (
                f"v4l2src device={args.device} ! "
                f"image/jpeg,width={args.width},height={args.height},framerate={args.fps}/1 ! "
                "jpegparse ! jpegdec ! videoconvert ! video/x-raw,format=BGR ! "
                "appsink drop=true sync=false max-buffers=1"
            )
            self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
            source_label = pipeline

            if not self.cap.isOpened():
                self.get_logger().warning("GStreamer camera open failed; trying V4L2 source")
                self.cap.release()
                self.cap = cv2.VideoCapture(args.source, cv2.CAP_V4L2)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
                self.cap.set(cv2.CAP_PROP_FPS, args.fps)
                source_label = str(args.source)
        else:
            self.cap = cv2.VideoCapture(args.source, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            self.cap.set(cv2.CAP_PROP_FPS, args.fps)
            source_label = str(args.source)

        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source_label}")

        self.timer = self.create_timer(1.0 / max(args.publish_hz, 1.0), self.step)
        self.get_logger().info(
            f"Publishing camera {source_label} -> {args.topic} "
            f"{args.width}x{args.height}@{args.publish_hz:.1f}"
        )

    def step(self):
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.get_logger().warning("No camera frame")
            return

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.args.frame_id
        msg.height = frame.shape[0]
        msg.width = frame.shape[1]
        msg.encoding = "bgr8"
        msg.is_bigendian = False
        msg.step = frame.shape[1] * 3
        msg.data = frame.tobytes()
        self.pub.publish(msg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", default="/camera/image_raw")
    parser.add_argument("--frame-id", default="camera")
    parser.add_argument("--gst", action="store_true")
    parser.add_argument("--device", default="/dev/video0")
    parser.add_argument("--source", type=int, default=0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--publish-hz", type=float, default=15.0)
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = SharedCameraPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        node.cap.release()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
