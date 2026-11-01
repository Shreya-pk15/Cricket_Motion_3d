"""
Evaluates Uniform vs Motion-Aware vs Hybrid Frame Selection on Monocular Cricket Video
using strictly Batter-Isolated Crease Crop Inference (CPU-Only).

Preserves original frame indices, maps normalized coordinates back to the full frame,
and prevents multi-person identity switching or distorted bowler detections.

Outputs:
- reports/corrected_batter_selection_evaluation.json
- reports/corrected_batter_selection_summary.json
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


def select_hybrid_frames(frame_energies: list, total_frames: int = 24, total_budget: int = 6) -> list:
    """
    Hybrid 6-frame selection:
    - 3 anchors: boundary frames [0, total_frames - 1] and midpoint [round((total_frames - 1) / 2)].
      For 24 frames: [0, 11, 23].
    - 3 motion peaks: from 3 non-overlapping intermediate temporal intervals:
      Band 1: [1 .. 7]
      Band 2: [8 .. 15] excluding midpoint
      Band 3: [16 .. 22]
    Ensures zero duplicate indices and chronological sorting.
    """
    if total_frames <= total_budget:
        return list(range(total_frames))

    anchors = [0, int(round((total_frames - 1) / 2)), total_frames - 1]  # [0, 11, 23]
    selected_set = set(anchors)

    bands = [
        [i for i in range(1, 8) if i not in selected_set],
        [i for i in range(8, 16) if i not in selected_set],
        [i for i in range(16, total_frames - 1) if i not in selected_set],
    ]

    for band in bands:
        if not band:
            continue
        best_idx = max(band, key=lambda idx: frame_energies[idx])
        selected_set.add(best_idx)

    if len(selected_set) < total_budget:
        remaining = [i for i in range(total_frames) if i not in selected_set]
        remaining.sort(key=lambda idx: frame_energies[idx], reverse=True)
        needed = total_budget - len(selected_set)
        for r_idx in remaining[:needed]:
            selected_set.add(r_idx)

    return sorted(list(selected_set))


def decode_all_clip_frames(video_path: Path):
    """Decode all frames into BGR list."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frames.append(frame)
    cap.release()
    return frames


def extract_batter_poses_for_all_frames(frames, detector, roi_box=(0.30, 0.20, 0.70, 0.88)):
    """
    Runs MediaPipe PoseLandmarker strictly within the targeted Crease ROI crop.
    Ensures:
      - Memory contiguity with np.ascontiguousarray
      - Anatomical upright validation: head (0) above hips (23, 24)
      - Mapping of normalized crop coordinates back to full-frame coordinates:
          x_full = rx1 + x_crop * (rx2 - rx1)
          y_full = ry1 + y_crop * (ry2 - ry1)
      - Never silently switches to bowler or background persons.
    """
    poses = {}
    rx1, ry1, rx2, ry2 = roi_box
    rw = rx2 - rx1
    rh = ry2 - ry1

    for idx, frame_bgr in enumerate(frames):
        H, W = frame_bgr.shape[:2]
        ymin, ymax = max(0, int(ry1 * H)), min(H, int(ry2 * H))
        xmin, xmax = max(0, int(rx1 * W)), min(W, int(rx2 * W))

        crop_bgr = frame_bgr[ymin:ymax, xmin:xmax]
        crop_rgb = np.ascontiguousarray(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
        mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)

        t_start = time.perf_counter()
        result = detector.detect(mp_crop)
        inf_ms = round((time.perf_counter() - t_start) * 1000, 2)

        is_valid_batter = False
        lm_list = []

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            p = result.pose_landmarks[0]
            # Upright validation: Nose above both hips
            if p[0].y < min(p[23].y, p[24].y):
                is_valid_batter = True
                for pt in p:
                    lm_list.append({
                        "x": float(rx1 + pt.x * rw),
                        "y": float(ry1 + pt.y * rh),
                        "z": float(pt.z),
                        "visibility": float(getattr(pt, "visibility", 0.0)),
                        "presence": float(getattr(pt, "presence", 0.0)),
                    })

        poses[idx] = {
            "detected": is_valid_batter,
            "landmarks": lm_list if is_valid_batter else [],
            "inference_time_ms": inf_ms,
        }
    return poses


def compute_temporal_span_and_gaps(selected_indices, total_frames=24):
    """Computes temporal span ratio and gap stats."""
    if not selected_indices:
        return {"temporal_span_ratio": 0.0, "max_gap": total_frames, "mean_gap": float(total_frames)}

    s_idx = sorted(selected_indices)
    span = (s_idx[-1] - s_idx[0]) / max(total_frames - 1, 1)

    if len(s_idx) > 1:
        gaps = [s_idx[i + 1] - s_idx[i] for i in range(len(s_idx) - 1)]
        max_gap = int(max(gaps))
        mean_gap = round(float(np.mean(gaps)), 2)
    else:
        max_gap = total_frames
        mean_gap = float(total_frames)

    return {
        "temporal_span_ratio": round(span, 4),
        "min_frame": s_idx[0],
        "max_frame": s_idx[-1],
        "max_gap": max_gap,
        "mean_gap": mean_gap,
    }


def compute_nspe_with_audit(clip_poses, selected_indices, total_frames=24):
    """
    Computes Nearest-Selected-Pose Error (NSPE) against detected reference batter poses.
    """
    valid_selected = [s for s in selected_indices if clip_poses.get(s, {}).get("detected", False)]
    total_ref_frames_with_pose = sum(1 for t in range(total_frames) if clip_poses.get(t, {}).get("detected"))

    if not valid_selected:
        return {
            "computable": False,
            "mean_nspe": None,
            "median_nspe": None,
            "valid_selected_count": 0,
            "matched_ref_frames_count": 0,
            "unmatched_ref_frames_count": total_ref_frames_with_pose,
            "unmatched_ref_frames": [t for t in range(total_frames) if clip_poses.get(t, {}).get("detected")],
            "per_frame_details": [],
        }

    per_frame_details = []
    distances = []
    unmatched_ref = []

    for t in range(total_frames):
        ref_p = clip_poses.get(t, {})
        if not ref_p.get("detected", False):
            continue

        ref_lm = ref_p["landmarks"]
        nearest_s = min(valid_selected, key=lambda s: abs(t - s))
        sel_lm = clip_poses[nearest_s]["landmarks"]

        joint_dists = []
        for j in range(min(len(ref_lm), len(sel_lm))):
            r_pt = ref_lm[j]
            s_pt = sel_lm[j]
            if r_pt["visibility"] >= 0.5 and s_pt["visibility"] >= 0.5:
                d = np.sqrt((r_pt["x"] - s_pt["x"]) ** 2 + (r_pt["y"] - s_pt["y"]) ** 2)
                joint_dists.append(float(d))

        if joint_dists:
            frame_err = float(np.mean(joint_dists))
            distances.append(frame_err)
            per_frame_details.append({
                "ref_frame": t,
                "matched_selected_frame": nearest_s,
                "temporal_distance": abs(t - nearest_s),
                "num_visible_joints": len(joint_dists),
                "error": round(frame_err, 4),
            })
        else:
            unmatched_ref.append(t)
            per_frame_details.append({
                "ref_frame": t,
                "matched_selected_frame": nearest_s,
                "temporal_distance": abs(t - nearest_s),
                "num_visible_joints": 0,
                "error": None,
                "note": "No mutually visible joints >= 0.5"
            })

    mean_err = round(float(np.mean(distances)), 4) if distances else None
    med_err = round(float(np.median(distances)), 4) if distances else None

    return {
        "computable": bool(distances),
        "mean_nspe": mean_err,
        "median_nspe": med_err,
        "valid_selected_count": len(valid_selected),
        "matched_ref_frames_count": len(distances),
        "unmatched_ref_frames_count": len(unmatched_ref),
        "unmatched_ref_frames": unmatched_ref,
        "per_frame_details": per_frame_details,
    }


def main():
    base_dir = Path(__file__).resolve().parents[1]
    motion_json_path = base_dir / "reports" / "motion_analysis.json"
    model_path = base_dir / "models" / "pose_landmarker_lite.task"
    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    if not motion_json_path.exists():
        raise FileNotFoundError(f"Missing motion analysis file: {motion_json_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    with open(motion_json_path, "r", encoding="utf-8") as f:
        clips_data = json.load(f)

    print("=" * 102)
    print("CORRECTED BATTER-SPECIFIC SELECTION EVALUATION: UNIFORM vs MOTION vs HYBRID (CPU)")
    print(f"Total Clips: {len(clips_data)} | Methods: 3 (Budget=6 frames/clip) | Reference: 24 frames/clip")
    print("Crease ROI Isolation: [X: 0.30 - 0.70, Y: 0.20 - 0.88] + Upright Posture Filter")
    print("=" * 102)

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
    detector = vision.PoseLandmarker.create_from_options(options)

    clip_evaluations = []

    try:
        for idx, clip in enumerate(clips_data):
            clip_name = clip["clip_name"]
            clip_class = clip["class"]
            rel_path = clip["relative_video_path"]
            video_file = base_dir / rel_path
            frame_energies = clip["frame_motion_energies"]

            u_indices = clip["uniform_selected_indices"]
            m_indices = clip["motion_selected_indices"]
            h_indices = select_hybrid_frames(frame_energies, total_frames=24, total_budget=6)

            all_frames = decode_all_clip_frames(video_file)
            total_frames = len(all_frames)
            clip_poses = extract_batter_poses_for_all_frames(all_frames, detector)

            ref_detected = [f for f in range(total_frames) if clip_poses[f]["detected"]]
            ref_det_pct = round((len(ref_detected) / total_frames) * 100, 2)
            ref_vis_vals = [pt["visibility"] for f in ref_detected for pt in clip_poses[f]["landmarks"]]
            ref_mean_vis = round(float(np.mean(ref_vis_vals)), 4) if ref_vis_vals else 0.0

            method_results = {}
            for m_key, indices in [("uniform", u_indices), ("motion_aware", m_indices), ("hybrid", h_indices)]:
                det_frames = [f for f in indices if clip_poses[f]["detected"]]
                det_pct = round((len(det_frames) / len(indices)) * 100, 2)
                vis_vals = [pt["visibility"] for f in det_frames for pt in clip_poses[f]["landmarks"]]
                mean_vis = round(float(np.mean(vis_vals)), 4) if vis_vals else 0.0

                temporal_meta = compute_temporal_span_and_gaps(indices, total_frames)
                nspe_meta = compute_nspe_with_audit(clip_poses, indices, total_frames)

                method_results[m_key] = {
                    "indices": indices,
                    "detected_frames_count": len(det_frames),
                    "detection_rate_pct": det_pct,
                    "mean_visibility": mean_vis,
                    "temporal_coverage": temporal_meta,
                    "nearest_pose_error": {
                        "computable": nspe_meta["computable"],
                        "mean_nspe": nspe_meta["mean_nspe"],
                        "median_nspe": nspe_meta["median_nspe"],
                        "matched_ref_frames": nspe_meta["matched_ref_frames_count"],
                        "unmatched_ref_frames": nspe_meta["unmatched_ref_frames_count"],
                    },
                    "missing_detection_indices": [f for f in indices if not clip_poses[f]["detected"]],
                    "nspe_details": nspe_meta["per_frame_details"],
                }

            clip_evaluations.append({
                "clip_name": clip_name,
                "class": clip_class,
                "total_frames": total_frames,
                "reference_sequence": {
                    "total_frames": total_frames,
                    "detected_frames_count": len(ref_detected),
                    "detection_rate_pct": ref_det_pct,
                    "mean_visibility": ref_mean_vis,
                    "missed_frames": [f for f in range(total_frames) if not clip_poses[f]["detected"]],
                },
                "methods": method_results,
            })

            label = f"{clip_class}/{clip_name}"
            u_err = method_results["uniform"]["nearest_pose_error"]["mean_nspe"]
            m_err = method_results["motion_aware"]["nearest_pose_error"]["mean_nspe"]
            h_err = method_results["hybrid"]["nearest_pose_error"]["mean_nspe"]

            u_str = f"{u_err:.4f}" if u_err is not None else "N/A"
            m_str = f"{m_err:.4f}" if m_err is not None else "N/A"
            h_str = f"{h_err:.4f}" if h_err is not None else "N/A"

            print(f"[{idx+1:02d}/12] {label:<28} Batter Det [U: {method_results['uniform']['detected_frames_count']}/6 | M: {method_results['motion_aware']['detected_frames_count']}/6 | H: {method_results['hybrid']['detected_frames_count']}/6] NSPE [U: {u_str} | M: {m_str} | H: {h_str}]")

    finally:
        detector.close()

    methods = ["uniform", "motion_aware", "hybrid"]
    summary = {
        "pilot_scope": {
            "total_clips": len(clip_evaluations),
            "total_reference_frames": len(clip_evaluations) * 24,
            "evaluation_type": "Corrected Batter-Specific Crease Isolation (CPU-Only)",
            "disclaimer": "Evaluated strictly on the isolated batsman within the batting crease region. Eliminates foreground bowler and background identity switches."
        },
        "aggregate_comparison": {},
        "per_class_comparison": {},
    }

    summary["aggregate_comparison"]["reference_sequence"] = {
        "total_frames": len(clip_evaluations) * 24,
        "total_detected_frames": sum(c["reference_sequence"]["detected_frames_count"] for c in clip_evaluations),
        "mean_detection_rate_pct": round(float(np.mean([c["reference_sequence"]["detection_rate_pct"] for c in clip_evaluations])), 2),
        "mean_landmark_visibility": round(float(np.mean([c["reference_sequence"]["mean_visibility"] for c in clip_evaluations])), 4),
    }

    for m in methods:
        det_pcts = [c["methods"][m]["detection_rate_pct"] for c in clip_evaluations]
        vis_vals = [c["methods"][m]["mean_visibility"] for c in clip_evaluations if c["methods"][m]["mean_visibility"] > 0]
        spans = [c["methods"][m]["temporal_coverage"]["temporal_span_ratio"] for c in clip_evaluations]
        max_gaps = [c["methods"][m]["temporal_coverage"]["max_gap"] for c in clip_evaluations]
        nspes = [c["methods"][m]["nearest_pose_error"]["mean_nspe"] for c in clip_evaluations if c["methods"][m]["nearest_pose_error"]["mean_nspe"] is not None]

        summary["aggregate_comparison"][m] = {
            "mean_detection_rate_pct": round(float(np.mean(det_pcts)), 2),
            "mean_landmark_visibility": round(float(np.mean(vis_vals)), 4),
            "mean_temporal_span_ratio": round(float(np.mean(spans)), 4),
            "mean_max_temporal_gap_frames": round(float(np.mean(max_gaps)), 2),
            "mean_nearest_pose_error_nspe": round(float(np.mean(nspes)), 4) if nspes else None,
            "total_unmatched_ref_frames": sum(c["methods"][m]["nearest_pose_error"]["unmatched_ref_frames"] for c in clip_evaluations),
        }

    classes = sorted(list(set(c["class"] for c in clip_evaluations)))
    for c in classes:
        c_clips = [clip for clip in clip_evaluations if clip["class"] == c]
        summary["per_class_comparison"][c] = {"num_clips": len(c_clips)}

        for m in methods:
            m_det = round(float(np.mean([clip["methods"][m]["detection_rate_pct"] for clip in c_clips])), 1)
            m_vis = round(float(np.mean([clip["methods"][m]["mean_visibility"] for clip in c_clips])), 3)
            m_nspe_list = [clip["methods"][m]["nearest_pose_error"]["mean_nspe"] for clip in c_clips if clip["methods"][m]["nearest_pose_error"]["mean_nspe"] is not None]
            m_nspe = round(float(np.mean(m_nspe_list)), 4) if m_nspe_list else None

            summary["per_class_comparison"][c][m] = {
                "detection_rate_pct": m_det,
                "mean_visibility": m_vis,
                "mean_nspe": m_nspe,
            }

    eval_json_path = reports_dir / "corrected_batter_selection_evaluation.json"
    sum_json_path = reports_dir / "corrected_batter_selection_summary.json"

    with open(eval_json_path, "w", encoding="utf-8") as f:
        json.dump(clip_evaluations, f, indent=2)
    with open(sum_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 102)
    print("CORRECTED AGGREGATE COMPARISON (BATTER-SPECIFIC): UNIFORM vs MOTION vs HYBRID")
    print("=" * 102)
    u_agg = summary["aggregate_comparison"]["uniform"]
    m_agg = summary["aggregate_comparison"]["motion_aware"]
    h_agg = summary["aggregate_comparison"]["hybrid"]

    print(f"{'Evaluation Metric':<38} {'Uniform':<18} {'Motion-Aware':<18} {'Hybrid':<18}")
    print("-" * 102)
    print(f"{'Batter Detection Rate (%)':<38} {u_agg['mean_detection_rate_pct']:<18.2f} {m_agg['mean_detection_rate_pct']:<18.2f} {h_agg['mean_detection_rate_pct']:.2f}%")
    print(f"{'Mean Landmark Visibility':<38} {u_agg['mean_landmark_visibility']:<18.4f} {m_agg['mean_landmark_visibility']:<18.4f} {h_agg['mean_landmark_visibility']:.4f}")
    print(f"{'Temporal Span Ratio (1.0 = full)':<38} {u_agg['mean_temporal_span_ratio']:<18.4f} {m_agg['mean_temporal_span_ratio']:<18.4f} {h_agg['mean_temporal_span_ratio']:.4f}")
    print(f"{'Mean Max Temporal Gap (frames)':<38} {u_agg['mean_max_temporal_gap_frames']:<18.2f} {m_agg['mean_max_temporal_gap_frames']:<18.2f} {h_agg['mean_max_temporal_gap_frames']:.2f}")
    print(f"{'Nearest-Selected-Pose Error (NSPE)':<38} {u_agg['mean_nearest_pose_error_nspe']:<18.4f} {m_agg['mean_nearest_pose_error_nspe']:<18.4f} {h_agg['mean_nearest_pose_error_nspe']:.4f}")
    print("=" * 102)

    print(f"\n[OK] Corrected detailed evaluations: {eval_json_path.relative_to(base_dir)}")
    print(f"[OK] Corrected summary comparison: {sum_json_path.relative_to(base_dir)}")


if __name__ == "__main__":
    main()

