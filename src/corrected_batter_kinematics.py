"""
Evaluates 3D Joint Kinematics across Uniform, Motion-Aware, and Hybrid Selection
using strictly Batter-Isolated Crease Crop Inference (CPU-Only).

Preserves original frame indices, calculates 3D joint angles with monocular occlusion
safeguards, and outputs corrected kinematic MAE and extrema preservation reports.

Outputs:
- reports/corrected_batter_joint_angle_analysis.json
- outputs/kinematic_plots/corrected_angle_trajectory_*.png
"""

from pathlib import Path
import json
import time
import sys
import numpy as np

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV (cv2) is not installed.")
    sys.exit(1)

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
except ImportError as e:
    print(f"[ERROR] MediaPipe Tasks import failed: {e}")
    sys.exit(1)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_crop_pose_detector(model_path: Path):
    with open(model_path, "rb") as f:
        model_bytes = f.read()

    base_options = mp_python.BaseOptions(model_asset_buffer=model_bytes)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.15,
        min_pose_presence_confidence=0.15,
        min_tracking_confidence=0.15,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)


def calculate_3d_angle(a: dict, b: dict, c: dict, min_vis=0.25):
    """Calculates 3D Euclidean angle formed by points A - B - C with vertex at B."""
    if a.get("vis", 0.0) < min_vis or b.get("vis", 0.0) < min_vis or c.get("vis", 0.0) < min_vis:
        return None

    ba = np.array([a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"]], dtype=np.float64)
    bc = np.array([c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"]], dtype=np.float64)

    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba < 1e-6 or norm_bc < 1e-6:
        return None

    cosine = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return round(float(np.degrees(angle_rad)), 2)


def calculate_trunk_inclination(left_hip, right_hip, left_shoulder, right_shoulder, min_vis=0.25):
    """Calculates 3D trunk inclination angle relative to vertical axis with side-on fallback."""
    has_l_hip = left_hip.get("vis", 0.0) >= min_vis
    has_r_hip = right_hip.get("vis", 0.0) >= min_vis
    has_l_sh = left_shoulder.get("vis", 0.0) >= min_vis
    has_r_sh = right_shoulder.get("vis", 0.0) >= min_vis

    if has_l_hip and has_r_hip and has_l_sh and has_r_sh:
        mid_hip = np.array([
            (left_hip["x"] + right_hip["x"]) * 0.5,
            (left_hip["y"] + right_hip["y"]) * 0.5,
            (left_hip["z"] + right_hip["z"]) * 0.5,
        ], dtype=np.float64)
        mid_sh = np.array([
            (left_shoulder["x"] + right_shoulder["x"]) * 0.5,
            (left_shoulder["y"] + right_shoulder["y"]) * 0.5,
            (left_shoulder["z"] + right_shoulder["z"]) * 0.5,
        ], dtype=np.float64)
    elif has_l_hip and has_l_sh:
        mid_hip = np.array([left_hip["x"], left_hip["y"], left_hip["z"]], dtype=np.float64)
        mid_sh = np.array([left_shoulder["x"], left_shoulder["y"], left_shoulder["z"]], dtype=np.float64)
    elif has_r_hip and has_r_sh:
        mid_hip = np.array([right_hip["x"], right_hip["y"], right_hip["z"]], dtype=np.float64)
        mid_sh = np.array([right_shoulder["x"], right_shoulder["y"], right_shoulder["z"]], dtype=np.float64)
    else:
        return None

    spine = mid_sh - mid_hip
    norm_spine = np.linalg.norm(spine)
    if norm_spine < 1e-6:
        return None

    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    cosine = np.dot(spine, vertical) / norm_spine
    cosine = np.clip(cosine, -1.0, 1.0)
    angle_rad = np.arccos(cosine)
    return round(float(np.degrees(angle_rad)), 2)


def extract_clip_batter_angles(video_path: Path, detector, roi_box=(0.30, 0.20, 0.70, 0.88)):
    """
    Decodes all 24 frames of video_path, crops to Crease ROI with np.ascontiguousarray,
    validates upright batter posture, and extracts 3D world landmarks for joint angles.
    """
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rx1, ry1, rx2, ry2 = roi_box

    frame_records = []
    for f_idx in range(total_frames):
        ret, frame_bgr = cap.read()
        if not ret or frame_bgr is None:
            frame_records.append({"frame": f_idx, "detected": False, "angles": {}})
            continue

        H, W = frame_bgr.shape[:2]
        ymin, ymax = max(0, int(ry1 * H)), min(H, int(ry2 * H))
        xmin, xmax = max(0, int(rx1 * W)), min(W, int(rx2 * W))

        crop_bgr = frame_bgr[ymin:ymax, xmin:xmax]
        crop_rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
        res = detector.detect(mp_crop)

        is_valid_batter = False
        if res.pose_landmarks and len(res.pose_landmarks) > 0:
            p = res.pose_landmarks[0]
            if p[0].y < min(p[23].y, p[24].y):
                is_valid_batter = True

        if not is_valid_batter or not res.pose_world_landmarks or len(res.pose_world_landmarks) == 0:
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

        wlm = [
            {
                "x": float(pt.x),
                "y": -float(pt.y),
                "z": float(pt.z),
                "vis": float(getattr(pt, "visibility", 0.0))
            }
            for pt in res.pose_world_landmarks[0]
        ]

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
    """
    frames = [r["frame"] for r in ref_records]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), facecolor="#141414")
    axes = axes.flatten()

    sub_configs = [
        ("Elbow Joint Angles", ["left_elbow", "right_elbow"], ["#FF5722", "#4CAF50"], ["Left Elbow", "Right Elbow"]),
        ("Knee Joint Angles", ["left_knee", "right_knee"], ["#E91E63", "#2196F3"], ["Left Knee", "Right Knee"]),
        ("Shoulder Joint Angles", ["left_shoulder", "right_shoulder"], ["#9C27B0", "#00BCD4"], ["Left Shoulder", "Right Shoulder"]),
        ("Trunk Inclination (Relative to Vertical)", ["trunk_inclination"], ["#FFC107"], ["Trunk Inclination"])
    ]

    for ax, (title, keys, colors, labels) in zip(axes, sub_configs):
        ax.set_facecolor("#1E1E1E")
        ax.set_title(title, color="#FFFFFF", fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Frame Index", color="#AAAAAA", fontsize=10)
        ax.set_ylabel("Angle (Degrees)", color="#AAAAAA", fontsize=10)
        ax.tick_params(colors="#CCCCCC", labelsize=9)
        ax.grid(True, linestyle="--", alpha=0.3, color="#555555")

        for k, c, lbl in zip(keys, colors, labels):
            y_vals = [ref_records[f]["angles"].get(k) for f in frames]
            valid_x = [f for f, v in zip(frames, y_vals) if v is not None]
            valid_y = [v for v in y_vals if v is not None]

            if valid_x:
                ax.plot(valid_x, valid_y, color=c, linewidth=2.0, label=f"Ref {lbl}", alpha=0.85)

        for u in u_idx:
            ax.axvline(x=u, color="#FFD700", linestyle=":", alpha=0.5, linewidth=1.2, label="Uniform" if u == u_idx[0] else "")
        for m in m_idx:
            ax.axvline(x=m, color="#FF5252", linestyle="--", alpha=0.5, linewidth=1.2, label="Motion-Aware" if m == m_idx[0] else "")
        for h in h_idx:
            ax.axvline(x=h, color="#00E676", linestyle="-", alpha=0.6, linewidth=1.5, label="Hybrid" if h == h_idx[0] else "")

        handles, plot_labels = ax.get_legend_handles_labels()
        by_label = dict(zip(plot_labels, handles))
        if by_label:
            ax.legend(by_label.values(), by_label.keys(), facecolor="#2A2A2A", edgecolor="#444444",
                      labelcolor="#EEEEEE", fontsize=8, loc="upper right")

    plt.suptitle(f"Corrected Batter Kinematics: {shot_class} ({clip_name})", color="#FFFFFF", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def main():
    base_dir = Path(__file__).resolve().parents[1]
    motion_json_path = base_dir / "reports" / "motion_analysis.json"
    corrected_sel_path = base_dir / "reports" / "corrected_batter_selection_evaluation.json"
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    reports_dir = base_dir / "reports"
    plots_dir = base_dir / "outputs" / "kinematic_plots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not corrected_sel_path.exists():
        raise FileNotFoundError(f"Missing corrected selection evaluation: {corrected_sel_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model: {model_path}")

    with open(corrected_sel_path, "r", encoding="utf-8") as f:
        selection_data = json.load(f)

    with open(motion_json_path, "r", encoding="utf-8") as f:
        motion_data = json.load(f)

    clip_dict_motion = {c["clip_name"]: c for c in motion_data}

    print("=" * 96)
    print("CORRECTED BATTER 3D KINEMATICS & JOINT-ANGLE ANALYSIS (CPU-ONLY)")
    print(f"Total Clips: {len(selection_data)} | Angles: 7 per frame | Evaluated Subsets: Uniform, Motion, Hybrid")
    print("Crease ROI Crop Isolation + Upright Posture Filter")
    print("=" * 96)

    detector = create_crop_pose_detector(model_path)
    all_clip_results = []
    generated_plots = []

    try:
        for idx, clip in enumerate(selection_data):
            clip_name = clip["clip_name"]
            shot_class = clip["class"]
            rel_path = clip_dict_motion[clip_name]["relative_video_path"]
            video_file = base_dir / rel_path

            u_idx = clip["methods"]["uniform"]["indices"]
            m_idx = clip["methods"]["motion_aware"]["indices"]
            h_idx = clip["methods"]["hybrid"]["indices"]

            ref_records = extract_clip_batter_angles(video_file, detector)

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

            if any(target in rel_path for target in ["vid100_41", "vid100_29", "vid100_28"]):
                plot_name = f"corrected_angle_trajectory_{shot_class.replace(' ', '_')}_{Path(clip_name).stem}.png"
                plot_path = plots_dir / plot_name
                plot_kinematic_trajectories(clip_name, shot_class, ref_records, u_idx, m_idx, h_idx, plot_path)
                generated_plots.append(plot_name)

            u_mean_mae = np.mean([v["mae_deg"] for v in u_eval.values() if v["mae_deg"] is not None])
            m_mean_mae = np.mean([v["mae_deg"] for v in m_eval.values() if v["mae_deg"] is not None])
            h_mean_mae = np.mean([v["mae_deg"] for v in h_eval.values() if v["mae_deg"] is not None])

            print(f"[{idx+1:02d}/12] {shot_class:<20} {clip_name:<16} | Mean Batter Angle MAE [U: {u_mean_mae:.2f}° | M: {m_mean_mae:.2f}° | H: {h_mean_mae:.2f}°]")

    finally:
        detector.close()

    angle_keys = [
        "left_elbow", "right_elbow", "left_shoulder", "right_shoulder",
        "left_knee", "right_knee", "trunk_inclination"
    ]

    aggregate_summary = {
        "metadata": {
            "total_clips": len(all_clip_results),
            "total_frames_analyzed": len(all_clip_results) * 24,
            "evaluation_type": "Corrected Batter-Specific 3D Kinematics (CPU-Only)",
            "disclaimer": "Kinematics calculated strictly from isolated batsman world landmarks in the batting crease. Not validated against optical mocap."
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

    all_u_mae = [v["uniform_mae_deg"] for v in aggregate_summary["aggregate_angle_mae"].values() if v["uniform_mae_deg"] is not None]
    all_m_mae = [v["motion_mae_deg"] for v in aggregate_summary["aggregate_angle_mae"].values() if v["motion_mae_deg"] is not None]
    all_h_mae = [v["hybrid_mae_deg"] for v in aggregate_summary["aggregate_angle_mae"].values() if v["hybrid_mae_deg"] is not None]

    all_u_ext = [v["min_angle_error_deg"]["uniform"] + v["max_angle_error_deg"]["uniform"] for v in aggregate_summary["aggregate_extrema_error"].values() if v["min_angle_error_deg"]["uniform"] is not None]
    all_m_ext = [v["min_angle_error_deg"]["motion"] + v["max_angle_error_deg"]["motion"] for v in aggregate_summary["aggregate_extrema_error"].values() if v["min_angle_error_deg"]["motion"] is not None]
    all_h_ext = [v["min_angle_error_deg"]["hybrid"] + v["max_angle_error_deg"]["hybrid"] for v in aggregate_summary["aggregate_extrema_error"].values() if v["min_angle_error_deg"]["hybrid"] is not None]

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

    out_json = reports_dir / "corrected_batter_joint_angle_analysis.json"
    full_output = {
        "summary": aggregate_summary,
        "per_clip_kinematics": all_clip_results
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2)

    print("\n" + "=" * 96)
    print("CORRECTED BATTER KINEMATIC ANGLE TRAJECTORY MAE SUMMARY (12 CLIPS, 7 ANGLES)")
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
    print(f"{'OVERALL MEAN ANGLE MAE':<26} {ov['uniform']:<18.2f}° {ov['motion_aware']:<18.2f}° {ov['hybrid']:<18.2f}°")
    ex = aggregate_summary["overall_summary"]["mean_extrema_gap_deg"]
    print(f"{'PEAK EXTREMA PRESERVATION ERR':<26} {ex['uniform']:<18.2f}° {ex['motion_aware']:<18.2f}° {ex['hybrid']:<18.2f}°")
    print("=" * 96)

    print(f"\n[OK] Corrected results saved to: {out_json.relative_to(base_dir)}")
    print(f"[OK] Corrected plots saved to: {plots_dir.relative_to(base_dir)}/ ({len(generated_plots)} figures)")


if __name__ == "__main__":
    main()

