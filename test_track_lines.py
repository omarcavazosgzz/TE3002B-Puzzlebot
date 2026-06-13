import cv2
import numpy as np
import argparse
import os
from pathlib import Path


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]


def make_trapezoid_roi(frame):
    h, w = frame.shape[:2]

    polygon = np.array([
        [int(w * 0.15), int(h * 0.95)],
        [int(w * 0.85), int(h * 0.95)],
        [int(w * 0.62), int(h * 0.45)],
        [int(w * 0.38), int(h * 0.45)]
    ], dtype=np.int32)

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)

    return mask, polygon


def detect_black_hsv(frame, roi_mask):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 120, 90])

    mask = cv2.inRange(hsv, lower_black, upper_black)
    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_black_otsu(frame, roi_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    roi_gray = cv2.bitwise_and(gray, gray, mask=roi_mask)

    _, mask = cv2.threshold(
        roi_gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_black_adaptive(frame, roi_mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mask = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        7
    )

    mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def get_line_center(mask, min_area=400):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > min_area:
            valid.append(c)

    if not valid:
        return None, None, None

    biggest = max(valid, key=cv2.contourArea)
    M = cv2.moments(biggest)

    if M["m00"] == 0:
        return None, biggest, None

    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    return (cx, cy), biggest, cv2.contourArea(biggest)


def detect_horizontal_crossing(mask):
    h, w = mask.shape[:2]

    y1 = int(h * 0.35)
    y2 = int(h * 0.55)

    band = mask[y1:y2, :]

    black_ratio = cv2.countNonZero(band) / band.size

    lines = cv2.HoughLinesP(
        band,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=int(w * 0.10),
        maxLineGap=20
    )

    horizontal_count = 0

    if lines is not None:
        for line in lines:
            x1, y1_l, x2, y2_l = line[0]
            dx = x2 - x1
            dy = y2_l - y1_l

            if dx == 0:
                continue

            angle = abs(np.degrees(np.arctan2(dy, dx)))

            if angle < 12:
                horizontal_count += 1

    crossing_detected = black_ratio > 0.08 and horizontal_count >= 2

    return crossing_detected, black_ratio, horizontal_count


def draw_debug(frame, polygon, mask, method_name):
    output = frame.copy()
    h, w = frame.shape[:2]

    cv2.polylines(output, [polygon], True, (255, 0, 0), 3)

    center, contour, area = get_line_center(mask)

    if contour is not None:
        cv2.drawContours(output, [contour], -1, (0, 255, 0), 3)

    if center is not None:
        cx, cy = center
        error = cx - (w // 2)

        cv2.circle(output, (cx, cy), 8, (0, 0, 255), -1)
        cv2.line(output, (w // 2, h), (w // 2, 0), (255, 255, 0), 2)

        cv2.putText(
            output,
            f"cx={cx} error={error}",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )
    else:
        cv2.putText(
            output,
            "LINE LOST",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    crossing, ratio, hcount = detect_horizontal_crossing(mask)

    cv2.putText(
        output,
        f"method={method_name}",
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"crossing={crossing} ratio={ratio:.3f} hlines={hcount}",
        (30, 130),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2
    )

    return output


def process_image(image_path, output_folder):
    frame = cv2.imread(str(image_path))

    if frame is None:
        print(f"[SKIP] Could not read: {image_path}")
        return

    image_name = image_path.stem
    image_out = output_folder / image_name
    image_out.mkdir(parents=True, exist_ok=True)

    roi_mask, polygon = make_trapezoid_roi(frame)

    hsv_mask = detect_black_hsv(frame, roi_mask)
    otsu_mask = detect_black_otsu(frame, roi_mask)
    adaptive_mask = detect_black_adaptive(frame, roi_mask)

    hsv_debug = draw_debug(frame, polygon, hsv_mask, "HSV black mask")
    otsu_debug = draw_debug(frame, polygon, otsu_mask, "Otsu")
    adaptive_debug = draw_debug(frame, polygon, adaptive_mask, "Adaptive")

    cv2.imwrite(str(image_out / "01_original.jpg"), frame)
    cv2.imwrite(str(image_out / "02_roi_mask.jpg"), roi_mask)
    cv2.imwrite(str(image_out / "03_hsv_mask.jpg"), hsv_mask)
    cv2.imwrite(str(image_out / "04_otsu_mask.jpg"), otsu_mask)
    cv2.imwrite(str(image_out / "05_adaptive_mask.jpg"), adaptive_mask)
    cv2.imwrite(str(image_out / "06_hsv_debug.jpg"), hsv_debug)
    cv2.imwrite(str(image_out / "07_otsu_debug.jpg"), otsu_debug)
    cv2.imwrite(str(image_out / "08_adaptive_debug.jpg"), adaptive_debug)

    print(f"[OK] Processed: {image_path.name}")


def collect_images(input_path):
    input_path = Path(input_path).expanduser()

    if input_path.is_file():
        return [input_path]

    if input_path.is_dir():
        images = []
        for ext in IMAGE_EXTS:
            images.extend(input_path.rglob(f"*{ext}"))
            images.extend(input_path.rglob(f"*{ext.upper()}"))
        return sorted(images)

    return []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image file or folder")
    parser.add_argument("--out", default="results_lines", help="Output folder")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser()
    output_folder = Path(args.out).expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)

    images = collect_images(input_path)

    if len(images) == 0:
        print("No images found.")
        return

    print(f"Found {len(images)} image(s).")

    for image_path in images:
        process_image(image_path, output_folder)

    print()
    print("Done.")
    print(f"Results saved in: {output_folder}")
    print("Check each image folder and compare:")
    print(" - 06_hsv_debug.jpg")
    print(" - 07_otsu_debug.jpg")
    print(" - 08_adaptive_debug.jpg")


if __name__ == "__main__":
    main()