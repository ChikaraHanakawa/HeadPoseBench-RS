import cv2
import numpy as np
import torch
import pyrealsense2 as rs
from network.network import Network
from torchvision import transforms
from PIL import Image
from utils import load_snapshot
from utils.camera_normalize import drawAxis
import time
import argparse


def parse_option():
    parser = argparse.ArgumentParser('RealSense Head Pose Estimator', add_help=False)
    parser.add_argument('--output', type=str, default="",
                        help='保存先パス（空文字なら保存しない）')
    args = parser.parse_args()
    return args


def scale_bbox(bbox, scale):
    w = max(bbox[2], bbox[3]) * scale
    x = max(bbox[0] + bbox[2] / 2 - w / 2, 0)
    y = max(bbox[1] + bbox[3] / 2 - w / 2, 0)
    return np.asarray([x, y, w, w], np.int64)


def get_stable_depth(depth_frame, cx, cy, radius=5):
    """顔中心付近の有効深度ピクセルの中央値を返す（単位: m）"""
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


def draw_label(frame, text, pos, color):
    """黒背景付きテキストを描画し、次ラベルへの縦オフセットを返す"""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2
    pad        = 4
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(frame, (x, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), cv2.FILLED)
    cv2.putText(frame, text, (x + pad // 2, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return th + pad * 2  # 次ラベルへのオフセット


# =========================
# Parameters
# =========================
WIDTH  = 640
HEIGHT = 480
FPS    = 30

# speaker1=cyan, speaker2=orange, speaker3=magenta (BGR)
SPEAKER_COLORS = [(255, 255, 0), (0, 165, 255), (255, 0, 255)]
SPEAKER_NAMES  = ["speaker1", "speaker2", "speaker3"]


def main():
    args = parse_option()

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
    depth_sensor.set_option(rs.option.visual_preset, 3)  # High Accuracy

    # =========================
    # Output VideoWriter
    # =========================
    outstream = None
    if args.output != "":
        outstream = cv2.VideoWriter(
            args.output,
            cv2.VideoWriter_fourcc(*'MJPG'),
            FPS, (WIDTH, HEIGHT)
        )

    # =========================
    # Face Detector
    # =========================
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

    # =========================
    # Head Pose Model
    # =========================
    pose_estimator = Network(bin_train=False)
    load_snapshot(pose_estimator, "./models/model-b66.pkl")
    pose_estimator = pose_estimator.to('cpu').eval()

    transform_test = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    count      = 0
    last_faces = None
    prev_time  = time.time()

    # =========================
    # Log file (CSV, append)
    # =========================
    log_file = open("output.log", "a")
    log_file.write("speaker1_dist,speaker1_pitch,speaker2_dist,speaker2_pitch,"
                   "speaker3_dist,speaker3_pitch\n")

    print("Press [q] or [Esc] to quit.")

    try:
        while True:
            # --- フレーム取得 ---
            frames         = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)

            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # BGR

            # --- FPS計測 ---
            cur_time  = time.time()
            fps       = 1.0 / (cur_time - prev_time + 1e-9)
            prev_time = cur_time

            # --- 顔検出 ---
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_img, 1.2)
            if len(faces) == 0 and last_faces is not None:
                faces = last_faces
            last_faces = faces

            # --- Phase 1: テンソル収集（描画なし）---
            face_tensors = []
            face_meta    = []  # (cx, cy, bx, by, x, y, w, h)

            for bbox in (last_faces if last_faces is not None else []):
                x, y, w, h = scale_bbox(bbox, 1.5)
                cx = int(x + w / 2)
                cy = int(y + h / 2)

                face_img = frame[y:y + h, x:x + w]
                if face_img.size == 0:
                    continue

                pil_img = Image.fromarray(
                    cv2.cvtColor(cv2.resize(face_img, (224, 224)), cv2.COLOR_BGR2RGB)
                )
                face_tensors.append(transform_test(pil_img)[None])
                face_meta.append((cx, cy, x, y, w, h))

            # --- Phase 2: 推論 + 深度取得 + ソート ---
            if len(face_tensors) > 0:
                with torch.no_grad():
                    start = time.time()
                    face_tensors_cat = torch.cat(face_tensors, dim=0)
                    roll, yaw, pitch = pose_estimator(face_tensors_cat)
                    inf_ms = (time.time() - start) / len(roll) * 1000
                    #print(f"inference time: {inf_ms:.3f} ms/face")

                face_data = []
                for i, (cx, cy, x, y, w, h) in enumerate(face_meta):
                    depth_m   = get_stable_depth(depth_frame, cx, cy, radius=5)
                    pitch_val = pitch[i].item() if hasattr(pitch[i], 'item') else float(pitch[i])
                    face_data.append({
                        'cx': cx, 'cy': cy,
                        'x': x, 'y': y, 'w': w, 'h': h,
                        'depth_m': depth_m,
                        'pitch':   pitch_val,
                        'roll':    roll[i],
                        'yaw':     yaw[i],
                    })

                # 距離でソート（近い順）、上位3人に絞る
                face_data.sort(
                    key=lambda d: d['depth_m'] if d['depth_m'] is not None else float('inf')
                )
                face_data = face_data[:3]

                # --- Phase 3: speaker ラベルで描画 ---
                for k, fd in enumerate(face_data):
                    color = SPEAKER_COLORS[k]
                    name  = SPEAKER_NAMES[k]
                    x, y, w, h = fd['x'], fd['y'], fd['w'], fd['h']

                    # 顔枠
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # 頭部姿勢軸
                    face_img = frame[y:y + h, x:x + w]
                    drawAxis(face_img, [fd['roll'], fd['yaw'], fd['pitch']], size=50)

                    # ラベル（speaker名 / 距離 / pitch）
                    dist_text  = f"{fd['depth_m']:.2f} m" if fd['depth_m'] is not None else "-- m"
                    pitch_text = f"pitch: {fd['pitch']:.1f} deg"

                    label_y = max(y - 4, 20)
                    offset  = draw_label(frame, name,       (x, label_y),          color)
                    offset += draw_label(frame, dist_text,  (x, label_y + offset), color)
                    draw_label(frame, pitch_text, (x, label_y + offset), color)

                # --- ログ書き出し ---
                parts = []
                for k in range(3):
                    if k < len(face_data) and face_data[k]['depth_m'] is not None:
                        parts.append(f"{face_data[k]['depth_m']:.3f}")
                        parts.append(f"{face_data[k]['pitch']:.1f}")
                    else:
                        parts += ["nan", "nan"]
                log_file.write(",".join(parts) + "\n")
                log_file.flush()

            else:
                # 顔なしフレーム
                log_file.write("nan,nan,nan,nan,nan,nan\n")
                log_file.flush()

            # --- FPS左上描画 ---
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Result", frame)

            if outstream is not None:
                outstream.write(frame)

            key = cv2.waitKey(1)
            if key == 27 or key == ord("q"):
                break

            count += 1

    finally:
        pipeline.stop()
        log_file.close()
        cv2.destroyAllWindows()
        if outstream is not None:
            outstream.release()
        print("Stopped.")


if __name__ == '__main__':
    main()
