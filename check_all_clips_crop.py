import cv2, json
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from pathlib import Path

model_path = Path('models/pose_landmarker_lite.task')
with open(model_path, 'rb') as f:
    model_bytes = f.read()

base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.15,
    min_pose_presence_confidence=0.15,
    min_tracking_confidence=0.15,
    num_poses=1,
)
detector = vision.PoseLandmarker.create_from_options(options)

with open('reports/motion_analysis.json') as f:
    motion_clips = json.load(f)

roi = (0.30, 0.20, 0.70, 0.88)
print("Evaluating Crease Crop ROI across all 12 clips (24 frames each):")
print("ROI: " + str(roi))
print("-" * 60)
rx1, ry1, rx2, ry2 = roi

results = []
for c in motion_clips:
    rel_path = c['relative_video_path']
    video_path = Path(rel_path)
    cap = cv2.VideoCapture(str(video_path))
    det_count = 0
    upright_count = 0
    for f in range(24):
        ret, frame_bgr = cap.read()
        if not ret: break
        H, W = frame_bgr.shape[:2]
        ymin, ymax = max(0, int(ry1 * H)), min(H, int(ry2 * H))
        xmin, xmax = max(0, int(rx1 * W)), min(W, int(rx2 * W))
        crop_bgr = frame_bgr[ymin:ymax, xmin:xmax]
        crop_rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        res = detector.detect(mp_img)
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            det_count += 1
            p = res.pose_landmarks[0]
            if p[0].y < min(p[23].y, p[24].y):
                upright_count += 1
    cap.release()
    c_name = c['clip_name']
    c_cls = c['class']
    print(f"{c_cls:<16} {c_name:<14}: {det_count:2d}/24 detected, {upright_count:2d}/24 upright")
    results.append((c_cls, c_name, det_count, upright_count))

print("-" * 60)
total_upright = sum(r[3] for r in results)
print(f"TOTAL: {total_upright}/{12*24} ({total_upright/(12*24)*100:.2f}%) valid upright batter detections")

