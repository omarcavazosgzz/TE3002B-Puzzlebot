#!/usr/bin/env python3
import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def letterbox(image, size):
    h, w = image.shape[:2]
    scale = min(size / float(w), size / float(h))
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx = (size - nw) // 2
    dy = (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(x1 + 1, min(int(round(x2)), w))
    y2 = max(y1 + 1, min(int(round(y2)), h))
    return x1, y1, x2, y2


def expand_box(box, frac, w, h):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    pad_x = bw * frac
    pad_y = bh * frac
    return clamp_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), w, h)


def nms(boxes, scores, conf, iou_thresh):
    if not boxes:
        return []
    idxs = cv2.dnn.NMSBoxes(
        [list(map(int, b)) for b in boxes],
        scores,
        conf,
        iou_thresh,
    )
    if len(idxs) == 0:
        return []
    return [int(i) for i in np.array(idxs).reshape(-1)]


class TrafficLightSignPublisher(Node):
    def __init__(self, args):
        super().__init__("traffic_light_sign_publisher")
        self.args = args
        self.pub = self.create_publisher(String, "/sign_detected", 10)
        self.net = cv2.dnn.readNetFromONNX(args.model)
        self.history = deque(maxlen=args.stable_window)
        self.last_pub = ""
        self.last_pub_time = 0.0
        self.frame_lock = threading.Lock()
        self.latest_frame = None

        self.cap = None
        if args.image_topic:
            self.image_sub = self.create_subscription(
                Image,
                args.image_topic,
                self.image_callback,
                rclpy.qos.qos_profile_sensor_data,
            )
            camera_source = args.image_topic
        else:
            self.image_sub = None
            self.cap = cv2.VideoCapture(args.camera, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            self.cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera {args.camera}")
            camera_source = str(args.camera)

        period = 1.0 / max(args.hz, 0.1)
        self.timer = self.create_timer(period, self.step)
        self.get_logger().info(
            f"Traffic light detector ready: model={args.model} camera={camera_source} hz={args.hz}"
        )

    def image_callback(self, msg):
        frame = self.image_msg_to_bgr(msg)
        if frame is None:
            self.get_logger().warning(f"Unsupported image encoding: {msg.encoding}")
            return
        with self.frame_lock:
            self.latest_frame = frame

    def image_msg_to_bgr(self, msg):
        if not msg.data or msg.height == 0 or msg.width == 0:
            return None

        h, w = int(msg.height), int(msg.width)
        data = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()

        try:
            if enc in ("bgr8", "8uc3"):
                return data.reshape((h, msg.step))[:, :w * 3].reshape((h, w, 3)).copy()
            if enc == "rgb8":
                rgb = data.reshape((h, msg.step))[:, :w * 3].reshape((h, w, 3))
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if enc in ("mono8", "8uc1"):
                gray = data.reshape((h, msg.step))[:, :w].reshape((h, w))
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            if enc == "bgra8":
                bgra = data.reshape((h, msg.step))[:, :w * 4].reshape((h, w, 4))
                return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            if enc == "rgba8":
                rgba = data.reshape((h, msg.step))[:, :w * 4].reshape((h, w, 4))
                return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
        except ValueError:
            return None

        return None

    def get_frame(self):
        if self.cap is not None:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                return None
            return frame

        with self.frame_lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def parse_yolov8(self, output, frame_w, frame_h, scale, dx, dy):
        out = np.squeeze(output)
        if out.ndim != 2:
            return []

        # YOLOv8 ONNX is commonly [channels, anchors]; transpose to [anchors, channels].
        if out.shape[0] < out.shape[1] and out.shape[0] <= 128:
            out = out.T

        boxes = []
        scores = []
        for row in out:
            if row.shape[0] < 5:
                continue
            class_scores = row[4:]
            score = float(np.max(class_scores)) if class_scores.size else float(row[4])
            if score < self.args.conf:
                continue

            cx, cy, bw, bh = map(float, row[:4])
            x1 = (cx - bw * 0.5 - dx) / scale
            y1 = (cy - bh * 0.5 - dy) / scale
            x2 = (cx + bw * 0.5 - dx) / scale
            y2 = (cy + bh * 0.5 - dy) / scale
            x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), frame_w, frame_h)
            boxes.append((x1, y1, x2 - x1, y2 - y1))
            scores.append(score)

        keep = nms(boxes, scores, self.args.conf, self.args.iou)
        abs_boxes = []
        for i in keep:
            x, y, w, h = boxes[i]
            abs_boxes.append((x, y, x + w, y + h, scores[i]))
        return abs_boxes

    def detect_traffic_light_box(self, frame):
        h, w = frame.shape[:2]
        roi = frame
        x_off = 0
        y_off = 0

        if self.args.roi:
            rx1, ry1, rx2, ry2 = self.args.roi
            x1 = int(rx1 * w)
            y1 = int(ry1 * h)
            x2 = int(rx2 * w)
            y2 = int(ry2 * h)
            x1, y1, x2, y2 = clamp_box((x1, y1, x2, y2), w, h)
            roi = frame[y1:y2, x1:x2]
            x_off = x1
            y_off = y1

        blob_img, scale, dx, dy = letterbox(roi, self.args.imgsz)
        blob = cv2.dnn.blobFromImage(
            blob_img,
            scalefactor=1.0 / 255.0,
            size=(self.args.imgsz, self.args.imgsz),
            swapRB=True,
            crop=False,
        )
        self.net.setInput(blob)
        output = self.net.forward(self.net.getUnconnectedOutLayersNames()[0])
        boxes = self.parse_yolov8(output, roi.shape[1], roi.shape[0], scale, dx, dy)
        if not boxes:
            return None

        # Yellow can split into two YOLO boxes. Union nearby detections into one lamp body.
        xs1 = [b[0] for b in boxes]
        ys1 = [b[1] for b in boxes]
        xs2 = [b[2] for b in boxes]
        ys2 = [b[3] for b in boxes]
        union = (
            min(xs1) + x_off,
            min(ys1) + y_off,
            max(xs2) + x_off,
            max(ys2) + y_off,
        )
        return expand_box(union, self.args.box_pad, w, h)

    def color_scores(self, crop):
        if crop.size == 0:
            return {"RED": 0.0, "YELLOW": 0.0, "GREEN": 0.0}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        b, g, r = cv2.split(crop)

        red_hsv = (
            cv2.inRange(hsv, (0, 45, 110), (12, 255, 255)) |
            cv2.inRange(hsv, (168, 45, 110), (180, 255, 255))
        )
        red_rgb = ((r > 135) & (r > g * 1.18) & (r > b * 1.18)).astype(np.uint8) * 255

        yellow_hsv = cv2.inRange(hsv, (14, 25, 130), (42, 255, 255))
        yellow_rgb = ((r > 135) & (g > 115) & (b < 170) & (r > b * 1.15)).astype(np.uint8) * 255

        green_hsv = cv2.inRange(hsv, (42, 35, 100), (95, 255, 255))
        green_rgb = ((g > 130) & (g > r * 1.12) & (g > b * 1.12)).astype(np.uint8) * 255

        masks = {
            "RED": red_hsv | red_rgb,
            "YELLOW": yellow_hsv | yellow_rgb,
            "GREEN": green_hsv | green_rgb,
        }
        return {name: float(cv2.countNonZero(mask)) for name, mask in masks.items()}

    def classify_color(self, frame, box):
        x1, y1, x2, y2 = box
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return "NONE", {}

        h = crop.shape[0]
        thirds = {
            "RED": crop[0:h // 3, :],
            "YELLOW": crop[h // 3:2 * h // 3, :],
            "GREEN": crop[2 * h // 3:h, :],
        }

        scores = {}
        for name, part in thirds.items():
            scores[name] = self.color_scores(part)[name]

        label = max(scores, key=scores.get)
        total_px = max(1.0, float(crop.shape[0] * crop.shape[1]))
        if scores[label] / total_px < self.args.min_color_ratio:
            return "NONE", scores
        return label, scores

    def stable_label(self, label):
        if label != "NONE":
            self.history.append(label)
        if not self.history:
            return "NONE"
        counts = {x: self.history.count(x) for x in set(self.history)}
        best = max(counts, key=counts.get)
        if counts[best] >= self.args.stable_votes:
            return best
        return "NONE"

    def maybe_publish(self, label):
        now = time.monotonic()
        if label == "NONE":
            return
        if label == self.last_pub and now - self.last_pub_time < self.args.republish_s:
            return
        msg = String()
        msg.data = label
        self.pub.publish(msg)
        self.last_pub = label
        self.last_pub_time = now
        self.get_logger().info(f"published /sign_detected: {label}")

    def step(self):
        frame = self.get_frame()
        if frame is None:
            self.get_logger().warning("No camera frame")
            return

        box = self.detect_traffic_light_box(frame)
        label = "NONE"
        scores = {}
        if box is not None:
            label, scores = self.classify_color(frame, box)
        stable = self.stable_label(label)
        self.maybe_publish(stable)

        if self.args.show:
            debug = frame.copy()
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 255, 255), 2)
                h = y2 - y1
                cv2.line(debug, (x1, y1 + h // 3), (x2, y1 + h // 3), (255, 255, 255), 1)
                cv2.line(debug, (x1, y1 + 2 * h // 3), (x2, y1 + 2 * h // 3), (255, 255, 255), 1)
            cv2.putText(
                debug,
                f"{label} stable={stable} scores={scores}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("traffic_light_sign_publisher", debug)
            cv2.waitKey(1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--image-topic", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--hz", type=float, default=2.0)
    parser.add_argument("--conf", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--box-pad", type=float, default=0.18)
    parser.add_argument("--min-color-ratio", type=float, default=0.008)
    parser.add_argument("--stable-window", type=int, default=5)
    parser.add_argument("--stable-votes", type=int, default=2)
    parser.add_argument("--republish-s", type=float, default=1.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument(
        "--roi",
        type=float,
        nargs=4,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Normalized detector ROI, e.g. --roi 0.0 0.0 0.55 0.85",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rclpy.init()
    node = TrafficLightSignPublisher(args)
    try:
        rclpy.spin(node)
    finally:
        if node.cap is not None:
            node.cap.release()
        if args.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
