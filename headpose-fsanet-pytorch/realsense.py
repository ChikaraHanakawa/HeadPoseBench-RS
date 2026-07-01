import time
import numpy as np
import cv2
import onnxruntime
import pyrealsense2 as rs
from pathlib import Path
# local imports
from face_detector import FaceDetector
from utils import draw_axis

root_path = str(Path(__file__).absolute().parent.parent)


def get_median_depth(depth_frame, cx, cy, half=5):
    """顔中心付近の中央値深度を返す (メートル)"""
    w = depth_frame.get_width()
    h = depth_frame.get_height()
    x0, x1 = max(cx - half, 0), min(cx + half, w - 1)
    y0, y1 = max(cy - half, 0), min(cy + half, h - 1)

    depths = [
        depth_frame.get_distance(x, y)
        for y in range(y0, y1 + 1)
        for x in range(x0, x1 + 1)
        if depth_frame.get_distance(x, y) > 0
    ]
    return float(np.median(depths)) if depths else 0.0


def draw_label(img, text, pos, color):
    """背景付きテキストを描画する"""
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness  = 2
    pad        = 4
    x, y = pos
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    cv2.rectangle(img, (x, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), cv2.FILLED)
    cv2.putText(img, text, (x + pad // 2, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return th + pad * 2  # 次のラベルへのオフセット


def _main():
    # ----- RealSense パイプライン -----
    pipeline = rs.pipeline()
    config   = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # ----- モデル・検出器 -----
    face_d = FaceDetector()
    sess  = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-1x1-iter-688590.onnx')
    sess2 = onnxruntime.InferenceSession(f'{root_path}/pretrained/fsanet-var-iter-688590.onnx')

    print('Processing frames, press q to exit application...')

    prev_time = time.time()

    try:
        while True:
            # ----- フレーム取得 -----
            frames      = pipeline.wait_for_frames()
            aligned     = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())  # (480, 640, 3) BGR

            # ----- FPS計測 -----
            cur_time  = time.time()
            fps       = 1.0 / (cur_time - prev_time + 1e-9)
            prev_time = cur_time

            # ----- 顔検出・姿勢推定 -----
            face_bb = face_d.get(frame)
            for (x1, y1, x2, y2) in face_bb:
                face_roi = frame[y1:y2+1, x1:x2+1]

                # 前処理
                face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = face_roi.transpose((2, 0, 1))
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = (face_roi - 127.5) / 128
                face_roi = face_roi.astype(np.float32)

                # 姿勢推定
                res1 = sess.run(["output"],  {"input": face_roi})[0]
                res2 = sess2.run(["output"], {"input": face_roi})[0]
                yaw, pitch, roll = np.mean(np.vstack((res1, res2)), axis=0)

                # 軸描画
                tdx = (x2 - x1) // 2 + x1
                tdy = (y2 - y1) // 2 + y1
                draw_axis(frame, yaw, pitch, roll, tdx=tdx, tdy=tdy, size=50)

                # 距離取得
                dist_m = get_median_depth(depth_frame, tdx, tdy, half=5)
                dist_label  = f"{dist_m:.2f} m" if dist_m > 0 else "-- m"
                pitch_label = f"pitch: {pitch:.1f} deg"

                # ラベル描画（顔矩形の上から順に）
                label_y = max(y1 - 4, 20)
                offset  = draw_label(frame, dist_label,  (x1, label_y),          (0, 255, 255))
                draw_label(frame, pitch_label, (x1, label_y + offset), (0, 255, 0))

            # ----- FPS左上描画 -----
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow('Frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    _main()
