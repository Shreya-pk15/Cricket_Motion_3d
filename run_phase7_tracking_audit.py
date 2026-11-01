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
rx1, ry1, rx2, ry2 = roi

print("--- PER-CLIP AUDIT: DETECTIONS, CENTROIDS, SUSPECTED WRONG-PERSON OR MISSED ---")
clip_audit = []
for c in motion_clips:
    rel_path = c['relative_video_path']
    video_path = Path(rel_path)
    cap = cv2.VideoCapture(str(video_path))
    missed_frames = []
    suspected_wrong_person = []
    valid_batter_frames = []
    
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
            p = res.pose_landmarks[0]
            is_upright = p[0].y < min(p[23].y, p[24].y)
            cx_crop = (p[11].x + p[12].x + p[23].x + p[24].x) * 0.25
            cy_crop = (p[11].y + p[12].y + p[23].y + p[24].y) * 0.25
            cx_full = round(float(rx1 + cx_crop * (rx2 - rx1)), 3)
            cy_full = round(float(ry1 + cy_crop * (ry2 - ry1)), 3)
            
            body_h = None
            if res.pose_world_landmarks and len(res.pose_world_landmarks) > 0:
                w = res.pose_world_landmarks[0]
                body_h = round(float(abs(((w[11].y + w[12].y)*0.5) - ((w[27].y + w[28].y)*0.5))), 3)
                
            if is_upright:
                if body_h is not None and (body_h < 0.40 or body_h > 1.45):
                    suspected_wrong_person.append({"frame": f, "reason": "anomalous_stature", "body_h": body_h, "cx": cx_full, "cy": cy_full})
                else:
                    valid_batter_frames.append(f)
            else:
                suspected_wrong_person.append({"frame": f, "reason": "inverted/non-upright", "body_h": body_h, "cx": cx_full, "cy": cy_full})
        else:
            missed_frames.append(f)
    cap.release()
    c_res = {
        'class': c['class'],
        'clip_name': c['clip_name'],
        'valid_count': len(valid_batter_frames),
        'missed_count': len(missed_frames),
        'missed_frames': missed_frames,
        'suspected_wrong_person': suspected_wrong_person
    }
    clip_audit.append(c_res)
    c_cls = c['class']
    c_nm = c['clip_name']
    print(f"{c_cls:<15} {c_nm:<14} | Valid: {len(valid_batter_frames):2d}/24 | Missed: {missed_frames} | Suspected: {len(suspected_wrong_person)}")

with open('reports/phase7_clip_tracking_audit.json', 'w', encoding='utf-8') as f:
    json.dump(clip_audit, f, indent=2)
print("Saved to reports/phase7_clip_tracking_audit.json")

