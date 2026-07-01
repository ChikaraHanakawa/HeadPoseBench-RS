import cv2
import numpy as np
import pyrealsense2 as rs
import time
from sixdrepnet import SixDRepNet
from retinaface import RetinaFace

# =========================
# Utility
# =========================
def draw_text(
    image, text, x, y,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.6,
    color=(255, 255, 255),
    thickness=2,
    bg_color=(0, 0, 0)
):
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.putText(image, text, (x + 2, y - 4), font, font_scale, color, thickness)
    return image


def draw_axis(img, pitch, yaw, roll, tdx=None, tdy=None, size=100):
    pitch = np.radians(pitch)
    yaw   = -np.radians(yaw)
    roll  = np.radians(roll)

    if tdx is None or tdy is None:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    x1 = size * (np.cos(yaw) * np.cos(roll)) + tdx
    y1 = size * (np.cos(pitch) * np.sin(roll) + np.cos(roll) * np.sin(pitch) * np.sin(yaw)) + tdy

    x2 = size * (-np.cos(yaw) * np.sin(roll)) + tdx
    y2 = size * (np.cos(pitch) * np.cos(roll) - np.sin(pitch) * np.sin(yaw) * np.sin(roll)) + tdy

    x3 = size * (np.sin(yaw)) + tdx
    y3 = size * (-np.cos(yaw) * np.sin(pitch)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0, 0, 255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0, 255, 0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255, 0, 0), 2)

    return img


def get_stable_depth(depth_frame, cx, cy, radius=5):
    h = depth_frame.get_height()
    w = depth_frame.get_width()
    values = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < w and 0 <= ny < h:
                d = depth_frame.get_distance(nx, ny)
                if d > 0:
                    values.append(d)
    return float(np.median(values)) if values else None


def detect_faces_retinaface(frame, threshold=0.9):
    """
    RetinaFaceで顔検出し、(x1, y1, x2, y2) のリストを返す。
    RGB変換はRetinaFace内部で行われるためBGRのまま渡す。
    """
    detections = RetinaFace.detect_faces(frame, threshold=threshold)
    boxes = []
    if isinstance(detections, dict):
        for face_key in detections:
            facial_area = detections[face_key]["facial_area"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = facial_area
            # フレーム範囲にクリップ
            h, w = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w - 1, x2)
            y2 = min(h - 1, y2)
            boxes.append((x1, y1, x2, y2))
    return boxes


# =========================
# Parameters
# =========================
WIDTH  = 640
HEIGHT = 480
FPS    = 30

# RetinaFaceの検出間隔（毎フレーム実行すると重いので間引く）
DETECT_INTERVAL = 5

# =========================
# RealSense Pipeline Setup
# =========================
pipeline = rs.pipeline()
config   = rs.config()

config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, FPS)
config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16,  FPS)

pipeline_profile = pipeline.start(config)
align = rs.align(rs.stream.color)

depth_sensor = pipeline_profile.get_device().first_depth_sensor()
depth_sensor.set_option(rs.option.visual_preset, 3)

# =========================
# Head Pose Model
# =========================
model = SixDRepNet(gpu_id=-1)

# =========================
# Main Loop
# =========================
prev_time  = time.perf_counter()
frame_idx  = 0
fps_sum = 0
last_boxes = []

print("Press [q] or [Esc] to quit.")

try:
    while True:
        frames         = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())

        # --- FPS計測 ---
        cur_time  = time.perf_counter()
        fps       = 1.0 / (cur_time - prev_time + 1e-9)
        prev_time = cur_time

        # --- 顔検出（DETECT_INTERVALフレームに1回）---
        if frame_idx % DETECT_INTERVAL == 0:
            boxes = detect_faces_retinaface(color_image, threshold=0.9)
            if len(boxes) == 0 and len(last_boxes) > 0:
                boxes = last_boxes  # 検出失敗時は前回結果を使用
            last_boxes = boxes

        # --- 顔ごとに姿勢推定 ---
        for (x1, y1, x2, y2) in last_boxes:
            face_crop = color_image[y1:y2+1, x1:x2+1]
            if face_crop.size == 0:
                continue

            try:
                pitch, yaw, roll = model.predict(face_crop)
                pitch = float(pitch)
                yaw   = float(yaw)
                roll  = float(roll)
            except Exception as e:
                print("Head pose estimation error:", e)
                continue

            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            # 距離取得
            depth_m    = get_stable_depth(depth_frame, cx, cy, radius=5)
            dist_text  = f"{depth_m:.2f} m" if depth_m is not None else "-- m"
            pitch_text = f"pitch: {pitch:.1f} deg"

            cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            draw_text(color_image, pitch_text, x1, y1 - 35, color=(0, 255, 0))
            draw_text(color_image, dist_text,  x1, y1 - 10, color=(0, 255, 255))

            cv2.circle(color_image, (cx, cy), 5, (0, 0, 255), -1)
            draw_axis(color_image, pitch, yaw, roll, tdx=cx, tdy=cy, size=80)

        fps_sum += fps
        # --- FPS左上描画 ---
        draw_text(color_image, f"FPS: {fps:.1f}", 10, 30, color=(0, 255, 255))

        cv2.imshow("Head Pose + Depth (RealSense)", color_image)

        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            print(f"平均FPS: {frame_idx / fps_sum}")
            break

        frame_idx += 1

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print("Stopped.")
