"""
Cricket Kinematic & Joint-Angle Analysis from 3D World Landmarks.

Computes 3D joint angles using MediaPipe Tasks PoseLandmarker world landmarks (metric space):
1. Left Elbow (Shoulder 11 -> Elbow 13 -> Wrist 15)
2. Right Elbow (Shoulder 12 -> Elbow 14 -> Wrist 16)
3. Left Shoulder (Hip 23 -> Shoulder 11 -> Elbow 13)
4. Right Shoulder (Hip 24 -> Shoulder 12 -> Elbow 14)
5. Left Knee (Hip 23 -> Knee 25 -> Ankle 27)
6. Right Knee (Hip 24 -> Knee 26 -> Ankle 28)
7. Trunk Inclination: Angle between spinal line (Mid-Hip -> Mid-Shoulder) and the vertical gravity vector [0, 1, 0].

Safety & Validity:
- Angles are computed only if all supporting joint landmarks have visibility >= 0.5.
- If any joint landmark is below threshold or undetected, angle is set to None (null in JSON), never 0.0.

Evaluates:
- Full 24-frame reference trajectory.
- Uniform (6 frames), Motion-Aware (6 frames), and Hybrid (6 frames) representations.
- Mean Absolute Error (MAE) of nearest selected frame angles vs reference trajectory.
- Peak kinematic extrema preservation: error in capturing Min & Max flexion/extension angles.

Generates:
- reports/joint_angle_analysis.json
- outputs/kinematic_plots/angle_trajectory_*.png
"""

from pathlib import Path
import json
import time
import sys
import numpy as np
import cv2

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] Matplotlib is not installed.")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"[ERROR] MediaPipe Tasks import failed: {e}")
    sys.exit(1)


def create_pose_detector(model_path: Path):
    """Initializes MediaPipe Tasks PoseLandmarker on CPU."""
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.3,
        min_pose_presence_confidence=0.3,
        min_tracking_confidence=0.3,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def calculate_3d_angle(a: dict, b: dict, c: dict, min_vis=0.5):
    """
    Computes angle at vertex B formed by rays BA and BC in degrees.
    Formula:
      u = a - b, v = c - b
      cos(theta) = (u . v) / (||u|| * ||v||)
    Returns None if any landmark visibility < min_vis.
    """
    if a.get("vis", 0.0) < min_vis or b.get("vis", 0.0) < min_vis or c.get("vis", 0.0) < min_vis:
        return None

    pt_a = np.array([a["x"], a["y"], a["z"]], dtype=np.float64)
    pt_b = np.array([b["x"], b["y"], b["z"]], dtype=np.float64)
    pt_c = np.array([c["x"], c["y"], c["z"]], dtype=np.float64)

    u = pt_a - pt_b
    v = pt_c - pt_b

    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u < 1e-6 or norm_v < 1e-6:
        return None

    cosine = np.dot(u, v) / (norm_u * norm_v)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return round(float(np.degrees(angle_rad)), 2)


def calculate_trunk_inclination(left_hip, right_hip, left_shoulder, right_shoulder, min_vis=0.5):
    """
    Angle between spinal vector (Mid-Hip -> Mid-Shoulder) and vertical axis [0, 1, 0] in degrees.
    0 deg = perfectly upright vertical torso; larger angles = forward/lateral leaning.
    Returns None if any hip/shoulder has visibility < min_vis.
    """
    if (left_hip.get("vis", 0.0) < min_vis or right_hip.get("vis", 0.0) < min_vis or
            left_shoulder.get("vis", 0.0) < min_vis or right_shoulder.get("vis", 0.0) < min_vis):
        return None

    mid_hip = np.array([
        (left_hip["x"] + right_hip["x"]) * 0.5,
        (left_hip["y"] + right_hip["y"]) * 0.5,
        (left_hip["z"] + right_hip["z"]) * 0.5
    ], dtype=np.float64)

    mid_shoulder = np.array([
        (left_shoulder["x"] + right_shoulder["x"]) * 0.5,
        (left_shoulder["y"] + right_shoulder["y"]) * 0.5,
        (left_shoulder["z"] + right_shoulder["z"]) * 0.5
    ], dtype=np.float64)

    spine = mid_shoulder - mid_hip
    norm_spine = np.linalg.norm(spine)
    if norm_spine < 1e-6:
        return None

    # Upright vertical unit vector in inverted Y coordinate frame
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cosine = np.dot(spine, vertical) / norm_spine
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return round(float(np.degrees(angle_rad)), 2)


def extract_clip_frame_angles(video_path: Path, detector):
    """
    Decodes all 24 frames of video_path, extracts 3D world landmarks,
    and calculates joint angles for each frame.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_records = []
    for f_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            frame_records.append({"frame": f_idx, "detected": False, "angles": {}})
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        res = detector.detect(mp_img)

        if not (res.pose_world_landmarks and len(res.pose_world_landmarks) > 0):
            frame_records.append({
                "frame": f_idx,
                "detected": False,
                "angles": {
                    "left_elbow": None, "right_elbow": None,
                    "left_shoulder": None, "right_shoulder": None,
                    "left_knee": None, "right_knee": None,
                    "trunk_inclination": None
                }
            })
            continue

        # Invert Y so upright elevation is positive
        wlm = [
            {
                "x": float(p.x),
                "y": -float(p.y),
                "z": float(p.z),
                "vis": float(getattr(p, "visibility", 0.0))
            }
            for p in res.pose_world_landmarks[0]
        ]

        # Landmark Indices:
        # Shoulders: 11 (L), 12 (R)
        # Elbows: 13 (L), 14 (R)
        # Wrists: 15 (L), 16 (R)
        # Hips: 23 (L), 24 (R)
        # Knees: 25 (L), 26 (R)
        # Ankles: 27 (L), 28 (R)
        angles = {
            "left_elbow": calculate_3d_angle(wlm[11], wlm[13], wlm[15]),
            "right_elbow": calculate_3d_angle(wlm[12], wlm[14], wlm[16]),
            "left_shoulder": calculate_3d_angle(wlm[23], wlm[11], wlm[13]),
            "right_shoulder": calculate_3d_angle(wlm[24], wlm[12], wlm[14]),
            "left_knee": calculate_3d_angle(wlm[23], wlm[25], wlm[27]),
            "right_knee": calculate_3d_angle(wlm[24], wlm[26], wlm[28]),
            "trunk_inclination": calculate_trunk_inclination(wlm[23], wlm[24], wlm[11], wlm[12]),
        }

        frame_records.append({
            "frame": f_idx,
            "detected": True,
            "angles": angles
        })

    cap.release()
    return frame_records


def evaluate_method_kinematics(ref_records, selected_indices):
    """
    Evaluates kinematic trajectory representation for a chosen 6-frame subset.
    1. Angle Trajectory Error: Nearest-selected-frame Mean Absolute Error (MAE) against reference.
    2. Extrema Preservation: Absolute difference between reference Peak Max / Min angle
       and the peak Max / Min angle captured within the selected subset.
    """
    valid_selected = [s for s in selected_indices if ref_records[s]["detected"]]
    angle_keys = [
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_knee", "right_knee", "trunk_inclination"
    ]

    method_eval = {}

    for k in angle_keys:
        # Reference valid values
        ref_valid_frames = [
            (f_idx, rec["angles"][k])
            for f_idx, rec in enumerate(ref_records)
            if rec["detected"] and rec["angles"][k] is not None
        ]

        sel_valid_frames = [
            (s, ref_records[s]["angles"][k])
            for s in valid_selected
            if ref_records[s]["angles"][k] is not None
        ]

        if not ref_valid_frames or not sel_valid_frames:
            method_eval[k] = {
                "mae_deg": None,
                "ref_min_deg": None,
                "ref_max_deg": None,
                "sel_min_deg": None,
                "sel_max_deg": None,
                "min_error_deg": None,
                "max_error_deg": None,
                "evaluated_frames_count": 0
            }
            continue

        ref_vals = [val for _, val in ref_valid_frames]
        sel_vals = [val for _, val in sel_valid_frames]

        ref_min, ref_max = min(ref_vals), max(ref_vals)
        sel_min, sel_max = min(sel_vals), max(sel_vals)

        min_err = round(abs(sel_min - ref_min), 2)
        max_err = round(abs(sel_max - ref_max), 2)

        # Nearest-selected frame MAE
        frame_abs_errors = []
        for t, r_val in ref_valid_frames:
            nearest_s, nearest_s_val = min(sel_valid_frames, key=lambda pair: abs(t - pair[0]))
            frame_abs_errors.append(abs(r_val - nearest_s_val))

        mae = round(float(np.mean(frame_abs_errors)), 2)

        method_eval[k] = {
            "mae_deg": mae,
            "ref_min_deg": ref_min,
            "ref_max_deg": ref_max,
            "sel_min_deg": sel_min,
            "sel_max_deg": sel_max,
            "min_error_deg": min_err,
            "max_error_deg": max_err,
            "evaluated_frames_count": len(frame_abs_errors)
        }

    return method_eval


def plot_kinematic_trajectories(clip_name, shot_class, ref_records, u_idx, m_idx, h_idx, out_path: Path):
    """
    Renders 4-subplot kinematic trajectory plots:
    - Lead/Trailing Elbow Angles (Left & Right)
    - Lead/Trailing Knee Angles (Left & Right)
    - Shoulder Angles (Left & Right)
    - Trunk Inclination
    Highlights the selected frames of Uniform, Motion-Aware, and Hybrid.
    """
    frames = [r["frame"] for r in ref_records]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#141414")
    axes = axes.flatten()

    sub_configs = [
        ("Elbow Joint Angles", ["left_elbow", "right_elbow"], ["#FF5722", "#4CAF50"], ["Left Elbow", "Right Elbow"]),
        ("Knee Joint Angles", ["left_knee", "right_knee"], ["#E91E63", "#2196F3"], ["Left Knee", "Right Knee"]),
        ("Shoulder Joint Angles", ["left_shoulder", "right_shoulder"], ["#FF9800", "#8BC34A"], ["Left Shoulder", "Right Shoulder"]),
        ("Trunk Inclination (Spine vs. Vertical)", ["trunk_inclination"], ["#00ADB5"], ["Trunk Tilt"]),
    ]

    for ax, (title, keys, colors, labels) in zip(axes, sub_configs):
        ax.set_facecolor("#1A1D24")
        ax.grid(True, linestyle="--", alpha=0.25, color="#888888")

        for key, col, lbl in zip(keys, colors, labels):
            y_vals = [ref_records[f]["angles"].get(key) for f in frames]
            valid_x = [f for f, v in zip(frames, y_vals) if v is not None]
            valid_y = [v for v in y_vals if v is not None]

            if valid_x:
                ax.plot(valid_x, valid_y, color=col, linewidth=2.0, alpha=0.85, label=f"{lbl} (Ref Trajectory)")

        # Highlight method selections with vertical dashed bars or scatter markers
        # Uniform: blue dots; Motion: green squares; Hybrid: cyan diamonds
        sample_key = keys[0]
        for u in u_idx:
            val = ref_records[u]["angles"].get(sample_key)
            if val is not None:
                ax.scatter(u, val, color="#2196F3", marker="o", s=55, zorder=5, label="Uniform" if u == u_idx[0] else "")

        for m in m_idx:
            val = ref_records[m]["angles"].get(sample_key)
            if val is not None:
                ax.scatter(m, val, color="#4CAF50", marker="s", s=60, zorder=6, label="Motion-Aware" if m == m_idx[0] else "")

        for h in h_idx:
            val = ref_records[h]["angles"].get(sample_key)
            if val is not None:
                ax.scatter(h, val, color="#00ADB5", marker="D", s=70, edgecolors="#FFFFFF", linewidths=1.2, zorder=7, label="Hybrid" if h == h_idx[0] else "")

        ax.set_title(title, color="#FFFFFF", fontsize=11, weight="bold", pad=8)
        ax.set_xlabel("Video Frame Index (0-23)", color="#AAAAAA", fontsize=9)
        ax.set_ylabel("Angle (degrees)", color="#AAAAAA", fontsize=9)
        ax.tick_params(colors="#CCCCCC", labelsize=8)
        ax.legend(facecolor="#22252F", edgecolor="#444", labelcolor="#E0E0E0", fontsize=8, loc="upper right")

    header = (
        f"3D Kinematic Joint-Angle Trajectory  |  Shot: {shot_class} ({clip_name})\n"
        "[Estimated Monocular MediaPipe World Landmarks (m) — Not Optical Motion-Capture Ground Truth]"
    )
    fig.suptitle(header, color="#FFFFFF", fontsize=12, weight="bold", y=0.98)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(str(out_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    motion_json_path = base_dir / "reports" / "motion_analysis.json"
    hybrid_json_path = base_dir / "reports" / "hybrid_selection_evaluation.json"
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    reports_dir = base_dir / "reports"
    plots_dir = base_dir / "outputs" / "kinematic_plots"

    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not hybrid_json_path.exists():
        raise FileNotFoundError(f"Missing hybrid evaluation: {hybrid_json_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    with open(hybrid_json_path, "r", encoding="utf-8") as f:
        hybrid_data = json.load(f)

    with open(motion_json_path, "r", encoding="utf-8") as f:
        motion_data = json.load(f)

    clip_dict_motion = {c["clip_name"]: c for c in motion_data}

    print("=" * 96)
    print("CRICKET 3D KINEMATICS & JOINT-ANGLE ANALYSIS (CPU-ONLY)")
    print(f"Total Clips: {len(hybrid_data)} | Angles: 7 per frame | Evaluated Subsets: Uniform, Motion, Hybrid")
    print("=" * 96)

    detector = create_pose_detector(model_path)
    all_clip_results = []

    # Clips designated for publication trajectory plots: Cover Drive, Pull, Scoop
    plot_targets = [
        "Cover Drive/vid100_41.avi",
        "Pull/vid100_29.avi",
        "Scoop/vid100_28.avi"
    ]
    generated_plots = []

    try:
        for idx, clip in enumerate(hybrid_data):
            clip_name = clip["clip_name"]
            shot_class = clip["class"]
            rel_path = clip_dict_motion[clip_name]["relative_video_path"]
            video_file = base_dir / rel_path

            u_idx = clip["methods"]["uniform"]["indices"]
            m_idx = clip["methods"]["motion_aware"]["indices"]
            h_idx = clip["methods"]["hybrid"]["indices"]

            # 1. Extract angles across all 24 reference frames
            ref_records = extract_clip_frame_angles(video_file, detector)

            # 2. Evaluate kinematic representation for each selection method
            u_eval = evaluate_method_kinematics(ref_records, u_idx)
            m_eval = evaluate_method_kinematics(ref_records, m_idx)
            h_eval = evaluate_method_kinematics(ref_records, h_idx)

            clip_entry = {
                "clip_name": clip_name,
                "class": shot_class,
                "relative_video_path": rel_path,
                "total_frames": len(ref_records),
                "frame_angles_reference": ref_records,
                "methods_evaluation": {
                    "uniform": {"indices": u_idx, "kinematics": u_eval},
                    "motion_aware": {"indices": m_idx, "kinematics": m_eval},
                    "hybrid": {"indices": h_idx, "kinematics": h_eval},
                }
            }
            all_clip_results.append(clip_entry)

            # Check if this clip is a plotting target
            target_key = f"{shot_class}/{clip_name}"
            if target_key in plot_targets or any(target in rel_path for target in ["vid100_41", "vid100_29", "vid100_28"]):
                plot_name = f"angle_trajectory_{shot_class.replace(' ', '_')}_{Path(clip_name).stem}.png"
                plot_path = plots_dir / plot_name
                plot_kinematic_trajectories(clip_name, shot_class, ref_records, u_idx, m_idx, h_idx, plot_path)
                generated_plots.append(plot_name)
                print(f"  [PLOT GENERATED] {plot_name}")

            # Quick console progress
            u_mean_mae = np.mean([v["mae_deg"] for v in u_eval.values() if v["mae_deg"] is not None])
            m_mean_mae = np.mean([v["mae_deg"] for v in m_eval.values() if v["mae_deg"] is not None])
            h_mean_mae = np.mean([v["mae_deg"] for v in h_eval.values() if v["mae_deg"] is not None])

            print(f"[{idx+1:02d}/12] {shot_class:<20} {clip_name:<16} | Mean Angle MAE [U: {u_mean_mae:.2f}° | M: {m_mean_mae:.2f}° | H: {h_mean_mae:.2f}°]")

    finally:
        detector.close()
        print("MediaPipe PoseLandmarker closed cleanly.")

    # 3. Aggregate Kinematic Statistics Across All 12 Clips
    angle_keys = [
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_knee", "right_knee", "trunk_inclination"
    ]

    aggregate_summary = {
        "metadata": {
            "total_clips": len(all_clip_results),
            "total_frames_analyzed": len(all_clip_results) * 24,
            "disclaimer": "Kinematics calculated from MediaPipe Tasks pose_world_landmarks on CPU. Approximations for research comparison; not validated against multi-camera marker optical motion capture, nor certified for injury prediction."
        },
        "aggregate_angle_mae": {},
        "aggregate_extrema_error": {},
        "overall_summary": {}
    }

    for k in angle_keys:
        u_maes = [c["methods_evaluation"]["uniform"]["kinematics"][k]["mae_deg"] for c in all_clip_results if c["methods_evaluation"]["uniform"]["kinematics"][k]["mae_deg"] is not None]
        m_maes = [c["methods_evaluation"]["motion_aware"]["kinematics"][k]["mae_deg"] for c in all_clip_results if c["methods_evaluation"]["motion_aware"]["kinematics"][k]["mae_deg"] is not None]
        h_maes = [c["methods_evaluation"]["hybrid"]["kinematics"][k]["mae_deg"] for c in all_clip_results if c["methods_evaluation"]["hybrid"]["kinematics"][k]["mae_deg"] is not None]

        u_min_err = [c["methods_evaluation"]["uniform"]["kinematics"][k]["min_error_deg"] for c in all_clip_results if c["methods_evaluation"]["uniform"]["kinematics"][k]["min_error_deg"] is not None]
        m_min_err = [c["methods_evaluation"]["motion_aware"]["kinematics"][k]["min_error_deg"] for c in all_clip_results if c["methods_evaluation"]["motion_aware"]["kinematics"][k]["min_error_deg"] is not None]
        h_min_err = [c["methods_evaluation"]["hybrid"]["kinematics"][k]["min_error_deg"] for c in all_clip_results if c["methods_evaluation"]["hybrid"]["kinematics"][k]["min_error_deg"] is not None]

        u_max_err = [c["methods_evaluation"]["uniform"]["kinematics"][k]["max_error_deg"] for c in all_clip_results if c["methods_evaluation"]["uniform"]["kinematics"][k]["max_error_deg"] is not None]
        m_max_err = [c["methods_evaluation"]["motion_aware"]["kinematics"][k]["max_error_deg"] for c in all_clip_results if c["methods_evaluation"]["motion_aware"]["kinematics"][k]["max_error_deg"] is not None]
        h_max_err = [c["methods_evaluation"]["hybrid"]["kinematics"][k]["max_error_deg"] for c in all_clip_results if c["methods_evaluation"]["hybrid"]["kinematics"][k]["max_error_deg"] is not None]

        aggregate_summary["aggregate_angle_mae"][k] = {
            "uniform_mae_deg": round(float(np.mean(u_maes)), 2) if u_maes else None,
            "motion_mae_deg": round(float(np.mean(m_maes)), 2) if m_maes else None,
            "hybrid_mae_deg": round(float(np.mean(h_maes)), 2) if h_maes else None,
        }

        aggregate_summary["aggregate_extrema_error"][k] = {
            "min_angle_error_deg": {
                "uniform": round(float(np.mean(u_min_err)), 2) if u_min_err else None,
                "motion": round(float(np.mean(m_min_err)), 2) if m_min_err else None,
                "hybrid": round(float(np.mean(h_min_err)), 2) if h_min_err else None,
            },
            "max_angle_error_deg": {
                "uniform": round(float(np.mean(u_max_err)), 2) if u_max_err else None,
                "motion": round(float(np.mean(m_max_err)), 2) if m_max_err else None,
                "hybrid": round(float(np.mean(h_max_err)), 2) if h_max_err else None,
            }
        }

    # Grand averages
    all_u_mae = [v["uniform_mae_deg"] for v in aggregate_summary["aggregate_angle_mae"].values()]
    all_m_mae = [v["motion_mae_deg"] for v in aggregate_summary["aggregate_angle_mae"].values()]
    all_h_mae = [v["hybrid_mae_deg"] for v in aggregate_summary["aggregate_angle_mae"].values()]

    all_u_ext = [v["min_angle_error_deg"]["uniform"] + v["max_angle_error_deg"]["uniform"] for v in aggregate_summary["aggregate_extrema_error"].values()]
    all_m_ext = [v["min_angle_error_deg"]["motion"] + v["max_angle_error_deg"]["motion"] for v in aggregate_summary["aggregate_extrema_error"].values()]
    all_h_ext = [v["min_angle_error_deg"]["hybrid"] + v["max_angle_error_deg"]["hybrid"] for v in aggregate_summary["aggregate_extrema_error"].values()]

    aggregate_summary["overall_summary"] = {
        "mean_kinematic_mae_deg": {
            "uniform": round(float(np.mean(all_u_mae)), 2),
            "motion_aware": round(float(np.mean(all_m_mae)), 2),
            "hybrid": round(float(np.mean(all_h_mae)), 2),
        },
        "mean_extrema_gap_deg": {
            "uniform": round(float(np.mean(all_u_ext)) / 2.0, 2),
            "motion_aware": round(float(np.mean(all_m_ext)) / 2.0, 2),
            "hybrid": round(float(np.mean(all_h_ext)) / 2.0, 2),
        }
    }

    # Save to reports/joint_angle_analysis.json
    out_json = reports_dir / "joint_angle_analysis.json"
    full_output = {
        "summary": aggregate_summary,
        "per_clip_kinematics": all_clip_results
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 96)
    print("AGGREGATE KINEMATIC ANGLE TRAJECTORY MAE SUMMARY (12 CLIPS, 7 ANGLES)")
    print("=" * 96)
    print(f"{'Joint Angle':<26} {'Uniform MAE':<18} {'Motion MAE':<18} {'Hybrid MAE':<18} {'Best Method'}")
    print("-" * 96)
    for k in angle_keys:
        u_m = aggregate_summary["aggregate_angle_mae"][k]["uniform_mae_deg"]
        m_m = aggregate_summary["aggregate_angle_mae"][k]["motion_mae_deg"]
        h_m = aggregate_summary["aggregate_angle_mae"][k]["hybrid_mae_deg"]
        candidates = [("Uniform", u_m), ("Motion", m_m), ("Hybrid", h_m)]
        best = min(candidates, key=lambda x: x[1])[0]
        print(f"{k:<26} {u_m:<18.2f}° {m_m:<18.2f}° {h_m:<18.2f}° {best}")
    print("-" * 96)
    ov = aggregate_summary["overall_summary"]["mean_kinematic_mae_deg"]
    print(f"{'OVERALL MEAN ANGLE MAE':<26} {ov['uniform']:<18.2f}° {ov['motion_aware']:<18.2f}° {ov['hybrid']:<18.2f}° {'Hybrid' if ov['hybrid'] <= ov['uniform'] and ov['hybrid'] <= ov['motion_aware'] else ('Uniform' if ov['uniform'] <= ov['motion_aware'] else 'Motion')}")
    ex = aggregate_summary["overall_summary"]["mean_extrema_gap_deg"]
    print(f"{'PEAK EXTREMA PRESERVATION ERR':<26} {ex['uniform']:<18.2f}° {ex['motion_aware']:<18.2f}° {ex['hybrid']:<18.2f}° {'Hybrid' if ex['hybrid'] <= ex['uniform'] and ex['hybrid'] <= ex['motion_aware'] else ('Motion' if ex['motion_aware'] <= ex['uniform'] else 'Uniform')}")
    print("=" * 96)

    print(f"\n[OK] Results saved to: {out_json.relative_to(base_dir)}")
    print(f"[OK] Plots saved to: {plots_dir.relative_to(base_dir)}/ ({len(generated_plots)} figures)")


if __name__ == "__main__":
    main()

