import os
import cv2
import sys
import time
sys.path.append('..')
import numpy as np
from math import cos, sin
import pyrealsense2 as rs
from lib.FSANET_model import *
from keras import backend as K
from keras.layers import Average
from keras.models import Model
from keras.layers import Input


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size=80):
    pitch_rad = pitch * np.pi / 180
    yaw_rad   = -(yaw  * np.pi / 180)
    roll_rad  = roll  * np.pi / 180

    if tdx is not None and tdy is not None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width  / 2
        tdy = height / 2

    # X-Axis → red
    x1 = size * (cos(yaw_rad) * cos(roll_rad)) + tdx
    y1 = size * (cos(pitch_rad) * sin(roll_rad) + cos(roll_rad) * sin(pitch_rad) * sin(yaw_rad)) + tdy

    # Y-Axis ↓ green
    x2 = size * (-cos(yaw_rad) * sin(roll_rad)) + tdx
    y2 = size * (cos(pitch_rad) * cos(roll_rad) - sin(pitch_rad) * sin(yaw_rad) * sin(roll_rad)) + tdy

    # Z-Axis (screen out) blue
    x3 = size * (sin(yaw_rad)) + tdx
    y3 = size * (-cos(yaw_rad) * sin(pitch_rad)) + tdy

    cv2.line(img, (int(tdx), int(tdy)), (int(x1), int(y1)), (0,   0,   255), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x2), int(y2)), (0,   255,   0), 3)
    cv2.line(img, (int(tdx), int(tdy)), (int(x3), int(y3)), (255,   0,   0), 2)

    return img


def get_median_depth(depth_frame, cx, cy, half=5):
    """顔中心付近の中央値深度を返す (メートル)"""
    h = depth_frame.get_height()
    w = depth_frame.get_width()

    x0 = max(cx - half, 0)
    x1 = min(cx + half, w - 1)
    y0 = max(cy - half, 0)
    y1 = min(cy + half, h - 1)

    depths = []
    for y in range(y0, y1 + 1):
        for x in range(x0, x1 + 1):
            d = depth_frame.get_distance(x, y)
            if d > 0:
                depths.append(d)

    if len(depths) == 0:
        return 0.0
    return float(np.median(depths))


def draw_results(detected, color_img, depth_frame,
                 faces, ad, img_size, img_w, img_h, model):

    if len(detected) > 0:
        for i, (x, y, w, h) in enumerate(detected):
            x1, y1 = x,     y
            x2, y2 = x + w, y + h

            xw1 = max(int(x1 - ad * w), 0)
            yw1 = max(int(y1 - ad * h), 0)
            xw2 = min(int(x2 + ad * w), img_w - 1)
            yw2 = min(int(y2 + ad * h), img_h - 1)

            # --- 顔パッチを正規化してモデル入力 ---
            faces[i] = cv2.resize(
                color_img[yw1:yw2+1, xw1:xw2+1, :], (img_size, img_size))
            faces[i] = cv2.normalize(
                faces[i], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

            face_input = np.expand_dims(faces[i], axis=0)
            p_result   = model.predict(face_input)          # [yaw, pitch, roll]
            yaw, pitch, roll = p_result[0]

            # --- 軸描画 ---
            patch = draw_axis(
                color_img[yw1:yw2+1, xw1:xw2+1, :],
                yaw, pitch, roll)
            color_img[yw1:yw2+1, xw1:xw2+1, :] = patch

            # --- 顔中心座標 ---
            cx = (xw1 + xw2) // 2
            cy = (yw1 + yw2) // 2

            # --- 距離取得 (メートル) ---
            dist_m = get_median_depth(depth_frame, cx, cy, half=5)

            # --- ラベル描画 ---
            label_dist  = f"{dist_m:.2f} m"         if dist_m > 0 else "-- m"
            label_pitch = f"pitch: {pitch:.1f} deg"

            font       = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness  = 2
            pad        = 4

            # 距離ラベル (顔矩形の上)
            (tw, th), _ = cv2.getTextSize(label_dist, font, font_scale, thickness)
            ty = max(yw1 - pad, th + pad)
            cv2.rectangle(color_img,
                          (xw1, ty - th - pad),
                          (xw1 + tw + pad, ty + pad),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(color_img, label_dist,
                        (xw1 + pad//2, ty),
                        font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)

            # pitchラベル (距離ラベルのすぐ下)
            (tw2, th2), _ = cv2.getTextSize(label_pitch, font, font_scale, thickness)
            ty2 = ty + th2 + pad * 2
            cv2.rectangle(color_img,
                          (xw1, ty2 - th2 - pad),
                          (xw1 + tw2 + pad, ty2 + pad),
                          (0, 0, 0), cv2.FILLED)
            cv2.putText(color_img, label_pitch,
                        (xw1 + pad//2, ty2),
                        font, font_scale, (0, 255, 0), thickness, cv2.LINE_AA)

    return color_img


def main():
    try:
        os.mkdir('./img')
    except OSError:
        pass

    K.set_learning_phase(0)

    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_improved.xml')

    # ----- モデル設定 -----
    img_size     = 64
    num_capsule  = 3
    dim_capsule  = 16
    routings     = 2
    stage_num    = [3, 3, 3]
    lambda_d     = 1
    num_classes  = 3
    image_size   = 64
    num_primcaps = 7 * 3
    m_dim        = 5
    S_set        = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]

    model1 = FSA_net_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()
    model2 = FSA_net_Var_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    num_primcaps = 8 * 8 * 3
    S_set        = [num_capsule, dim_capsule, routings, num_primcaps, m_dim]
    model3 = FSA_net_noS_Capsule(image_size, num_classes, stage_num, lambda_d, S_set)()

    print('Loading models ...')
    model1.load_weights('../pre-trained/300W_LP_models/fsanet_capsule_3_16_2_21_5/fsanet_capsule_3_16_2_21_5.h5')
    print('Finished loading model 1.')
    model2.load_weights('../pre-trained/300W_LP_models/fsanet_var_capsule_3_16_2_21_5/fsanet_var_capsule_3_16_2_21_5.h5')
    print('Finished loading model 2.')
    model3.load_weights('../pre-trained/300W_LP_models/fsanet_noS_capsule_3_16_2_192_5/fsanet_noS_capsule_3_16_2_192_5.h5')
    print('Finished loading model 3.')

    inputs    = Input(shape=(64, 64, 3))
    avg_model = Average()([model1(inputs), model2(inputs), model3(inputs)])
    model     = Model(inputs=inputs, outputs=avg_model)

    # ----- RealSense パイプライン -----
    pipeline = rs.pipeline()
    config   = rs.config()

    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16,  30)

    pipeline.start(config)

    # カラー↔深度 位置合わせ
    align = rs.align(rs.stream.color)

    print('Start detecting pose ...')

    img_idx      = 0
    skip_frame   = 5
    ad           = 0.6
    detected     = []
    detected_pre = []
    faces        = np.empty((0, img_size, img_size, 3))
    prev_time    = time.time()
    fps          = 0.0

    try:
        while True:
            frames       = pipeline.wait_for_frames()
            aligned      = align.process(frames)
            color_frame  = aligned.get_color_frame()
            depth_frame  = aligned.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_img = np.asanyarray(color_frame.get_data())   # (480, 640, 3) BGR
            img_h, img_w, _ = color_img.shape
            img_idx += 1

            # FPS計測
            cur_time = time.time()
            fps      = 1.0 / (cur_time - prev_time + 1e-9)
            prev_time = cur_time

            if img_idx == 1 or img_idx % skip_frame == 0:
                gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
                detected = face_cascade.detectMultiScale(gray_img, 1.1)

                if len(detected_pre) > 0 and len(detected) == 0:
                    detected = detected_pre

                faces = np.empty((len(detected), img_size, img_size, 3))

                color_img = draw_results(
                    detected, color_img, depth_frame,
                    faces, ad, img_size, img_w, img_h, model)
                cv2.imwrite(f'img/{img_idx}.png', color_img)

            else:
                color_img = draw_results(
                    detected, color_img, depth_frame,
                    faces, ad, img_size, img_w, img_h, model)

            if len(detected) > len(detected_pre) or img_idx % (skip_frame * 3) == 0:
                detected_pre = detected

            # FPS左上描画
            fps_label = f"FPS: {fps:.1f}"
            cv2.putText(color_img, fps_label, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow("Head Pose + Distance", color_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
