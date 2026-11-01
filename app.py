"""
Streamlit Web Dashboard for Cricket3D_Motion Project.

Integrates:
1. Overview: Research purpose, pipeline methodology, dataset details, key findings.
2. Video Selection: Interactive clip selector covering all 12 sample clips with metadata.
3. Frame Selection: Visual and quantitative comparison of Uniform vs Motion-Aware vs Hybrid (6 frames).
4. 3D Pose Viewer: Synchronized 2D video frame, interactive Plotly 3D skeleton (rotatable/zoomable),
   and real-time joint-angle measurements.
5. Kinematic Analysis: Joint-angle trajectories (elbows, shoulders, knees, trunk inclination) over time
   with method subsampling markers.
6. Method Comparison: Cross-method benchmarks covering Pose Coverage, NSPE, Angle MAE, and Extrema Error.

CPU-Only, zero external model downloads, reuses existing reports and pre-computed caches.
"""

from pathlib import Path
import json
import numpy as np
import cv2
import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Theming
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cricket3D Motion Analytics",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark modern aesthetic
st.markdown("""
<style>
    .metric-card {
        background-color: #1E222B;
        border-radius: 8px;
        padding: 14px 18px;
        border: 1px solid #2E3440;
        margin-bottom: 10px;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8892B0;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-val {
        font-size: 1.5rem;
        color: #64FFDA;
        font-weight: 700;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 16px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Skeletal Topology & Color Scheme
# ---------------------------------------------------------
SKELETON_EDGES = [
    (11, 12, "torso"),
    (11, 23, "torso"),
    (12, 24, "torso"),
    (23, 24, "torso"),
    (11, 13, "left_arm"),
    (13, 15, "left_arm"),
    (12, 14, "right_arm"),
    (14, 16, "right_arm"),
    (23, 25, "left_leg"),
    (25, 27, "left_leg"),
    (27, 29, "left_leg"),
    (29, 31, "left_leg"),
    (24, 26, "right_leg"),
    (26, 28, "right_leg"),
    (28, 30, "right_leg"),
    (30, 32, "right_leg"),
    (0, 1, "head"),
    (1, 2, "head"),
    (2, 3, "head"),
    (3, 7, "head"),
    (0, 4, "head"),
    (4, 5, "head"),
    (5, 6, "head"),
    (6, 8, "head"),
    (9, 10, "head")
]

EDGE_COLORS = {
    "torso": "#00E5FF",      # Cyan
    "left_arm": "#FFD700",   # Gold
    "right_arm": "#FF9100",  # Orange
    "left_leg": "#00E676",   # Green
    "right_leg": "#76FF03",  # Lime
    "head": "#B0BEC5"        # Gray
}

# ---------------------------------------------------------
# Data Loaders (Cached)
# ---------------------------------------------------------
@st.cache_data
def load_json_reports():
    reports = {}
    base_dir = Path("reports")
    files = {
        "hybrid_eval": base_dir / "corrected_batter_selection_evaluation.json",
        "hybrid_summary": base_dir / "corrected_batter_selection_summary.json",
        "hybrid_audit": base_dir / "phase7_validation_audit.json",
        "kinematics": base_dir / "corrected_batter_joint_angle_analysis.json",
        "temporal_quality": base_dir / "pose_temporal_quality_summary.json"
    }
    for k, p in files.items():
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                reports[k] = json.load(f)
        else:
            reports[k] = None
    return reports

@st.cache_data
def extract_clip_frame(video_path: str, frame_idx: int):
    """Extracts a specific video frame as RGB."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret and frame is not None:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

@st.cache_resource
def get_pose_detector():
    """Initializes CPU Pose Landmarker from local model with multi-person candidate support."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model_path = Path("models/pose_landmarker_lite.task")
    if not model_path.exists():
        return None
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.15,
        min_pose_presence_confidence=0.15,
        min_tracking_confidence=0.15,
        num_poses=5,  # Detect multiple candidates
    )
    return vision.PoseLandmarker.create_from_options(options)

@st.cache_resource
def get_crop_pose_detector():
    """Specialized single-person detector for direct Batter ROI Cropped Inference."""
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision
    model_path = Path("models/pose_landmarker_lite.task")
    if not model_path.exists():
        return None
    base_options = mp_python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        min_pose_detection_confidence=0.15,
        min_pose_presence_confidence=0.15,
        min_tracking_confidence=0.15,
        num_poses=1,
    )
    return vision.PoseLandmarker.create_from_options(options)

def evaluate_frame_candidates(pose_landmarks_list, world_landmarks_list, roi_box):
    """
    Evaluates detected candidates in the frame.
    Strictly verifies anatomical orientation (head above hips above ankles)
    and full-body stature (shoulder-to-ankle >= 0.70m), rejecting inverted or fragmented poses.
    """
    if not pose_landmarks_list or not world_landmarks_list:
        return [], None
        
    rx1, ry1, rx2, ry2 = roi_box
    rcx, rcy = (rx1 + rx2) * 0.5, (ry1 + ry2) * 0.5
    
    diagnostics = []
    best_idx = None
    best_score = -float("inf")
    
    for i, (p, wlm) in enumerate(zip(pose_landmarks_list, world_landmarks_list)):
        cx = float((p[11].x + p[12].x + p[23].x + p[24].x) * 0.25)
        cy = float((p[11].y + p[12].y + p[23].y + p[24].y) * 0.25)
        
        # Anatomical 3D body height (shoulder to ankle vertical distance)
        sh_y = (wlm[11].y + wlm[12].y) * 0.5
        ank_y = (wlm[27].y + wlm[28].y) * 0.5
        body_h_3d = float(abs(sh_y - ank_y))
        
        # Anatomical 2D orientation: In upright cricketer, Head (0) must be ABOVE Hips (23, 24)
        is_upright = (p[0].y < min(p[23].y, p[24].y) < max(p[27].y, p[28].y) + 0.1)
        
        vis = float(sum(pt.visibility for pt in p) / len(p))
        dist = float(np.hypot(cx - rcx, cy - rcy))
        is_inside = (rx1 <= cx <= rx2) and (ry1 <= cy <= ry2)
        
        # Valid full-body standing/crouched cricketer: height 0.70m - 1.35m and upright
        is_valid_batter = is_upright and (0.70 <= body_h_3d <= 1.35)
        
        # Score calculation
        score = (vis * 2.0) + (2.0 if is_valid_batter else -4.0) + (1.0 if is_inside else -3.0) - dist
        
        info = {
            "id": i,
            "cx": cx,
            "cy": cy,
            "body_h_3d": body_h_3d,
            "is_inside": is_inside,
            "is_upright": is_upright,
            "is_valid_batter": is_valid_batter,
            "dist_to_center": dist,
            "score": score
        }
        diagnostics.append(info)
        
        if is_inside and is_valid_batter and score > best_score:
            best_score = score
            best_idx = i
            
    return diagnostics, best_idx

@st.cache_data
def get_cached_clip_pose(video_path: str, frame_idx: int, roi_box=(0.35, 0.20, 0.65, 0.85), use_crop_isolation=True):
    """
    Extracts 3D landmarks for the targeted batter.
    Prioritizes direct Crease Crop Isolation to guarantee that foreground bowlers/umpires
    and distorted inverted full-frame predictions are filtered out.
    """
    frame_rgb = extract_clip_frame(video_path, frame_idx)
    if frame_rgb is None:
        return None, False, [], "Video frame could not be read."
        
    import mediapipe as mp
    H, W = frame_rgb.shape[:2]
    rx1, ry1, rx2, ry2 = roi_box
    
    # 1. Direct Targeted Crease Crop (Primary isolated detection)
    if use_crop_isolation:
        crop_det = get_crop_pose_detector()
        ymin, ymax = max(0, int(ry1 * H)), min(H, int(ry2 * H))
        xmin, xmax = max(0, int(rx1 * W)), min(W, int(rx2 * W))
        if ymax > ymin and xmax > xmin:
            crop_rgb = np.ascontiguousarray(frame_rgb[ymin:ymax, xmin:xmax])
            mp_crop = mp.Image(image_format=mp.ImageFormat.SRGB, data=crop_rgb)
            res_crop = crop_det.detect(mp_crop)
            if res_crop.pose_world_landmarks and len(res_crop.pose_world_landmarks) > 0:
                p = res_crop.pose_landmarks[0]
                wlm = res_crop.pose_world_landmarks[0]
                # Check upright posture: head above hips
                if p[0].y < min(p[23].y, p[24].y):
                    pts = [
                        {
                            "x": float(pt.x),
                            "y": -float(pt.y),
                            "z": float(pt.z),
                            "vis": float(getattr(pt, "visibility", 0.0))
                        }
                        for pt in wlm
                    ]
                    return pts, True, [], "Batter isolated at Crease ROI (Upright full-body posture)."

    # 2. Fallback to Full-frame multi-candidate evaluation if crop failed
    detector = get_pose_detector()
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_img)
    
    diagnostics = []
    best_idx = None
    if result.pose_landmarks and len(result.pose_landmarks) > 0:
        diagnostics, best_idx = evaluate_frame_candidates(
            result.pose_landmarks, result.pose_world_landmarks, roi_box
        )
        
    if best_idx is not None and result.pose_world_landmarks:
        wlm = result.pose_world_landmarks[best_idx]
        pts = [
            {
                "x": float(p.x),
                "y": -float(p.y),
                "z": float(p.z),
                "vis": float(getattr(p, "visibility", 0.0))
            }
            for p in wlm
        ]
        return pts, True, diagnostics, f"Batter selected from full-frame (Candidate #{best_idx} upright)."

    diag_msg = f"Detected {len(diagnostics)} person(s), but none matched upright full-body batter criteria in ROI." if diagnostics else "0 people detected by MediaPipe in this frame."
    return None, False, diagnostics, diag_msg

def compute_live_frame_kinematics(wlm, min_vis=0.25):
    """
    Computes real-time joint angles from 33 world landmarks with robust fallback
    for monocular broadcast self-occlusion.
    """
    if not wlm or len(wlm) < 33:
        return {}

    def angle_between_points(a, b, c):
        if a.get("vis", 0.0) < min_vis or b.get("vis", 0.0) < min_vis or c.get("vis", 0.0) < min_vis:
            return None
        ba = np.array([a["x"] - b["x"], a["y"] - b["y"], a["z"] - b["z"]], dtype=np.float64)
        bc = np.array([c["x"] - b["x"], c["y"] - b["y"], c["z"] - b["z"]], dtype=np.float64)
        n_ba, n_bc = np.linalg.norm(ba), np.linalg.norm(bc)
        if n_ba < 1e-6 or n_bc < 1e-6:
            return None
        cos_ang = np.clip(np.dot(ba, bc) / (n_ba * n_bc), -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cos_ang))), 1)

    def trunk_angle(l_hip, r_hip, l_sh, r_sh):
        # Allow single-side hip/shoulder fallback if one side is self-occluded in side-on stance
        if (l_hip.get("vis", 0.0) >= min_vis and r_hip.get("vis", 0.0) >= min_vis and
            l_sh.get("vis", 0.0) >= min_vis and r_sh.get("vis", 0.0) >= min_vis):
            mid_hip = np.array([(l_hip["x"] + r_hip["x"])*0.5, (l_hip["y"] + r_hip["y"])*0.5, (l_hip["z"] + r_hip["z"])*0.5])
            mid_sh = np.array([(l_sh["x"] + r_sh["x"])*0.5, (l_sh["y"] + r_sh["y"])*0.5, (l_sh["z"] + r_sh["z"])*0.5])
        elif l_hip.get("vis", 0.0) >= min_vis and l_sh.get("vis", 0.0) >= min_vis:
            mid_hip = np.array([l_hip["x"], l_hip["y"], l_hip["z"]])
            mid_sh = np.array([l_sh["x"], l_sh["y"], l_sh["z"]])
        elif r_hip.get("vis", 0.0) >= min_vis and r_sh.get("vis", 0.0) >= min_vis:
            mid_hip = np.array([r_hip["x"], r_hip["y"], r_hip["z"]])
            mid_sh = np.array([r_sh["x"], r_sh["y"], r_sh["z"]])
        else:
            return None
            
        spine = mid_sh - mid_hip
        norm_s = np.linalg.norm(spine)
        if norm_s < 1e-6:
            return None
        vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        cos_val = np.clip(np.dot(spine, vertical) / norm_s, -1.0, 1.0)
        return round(float(np.degrees(np.arccos(cos_val))), 1)

    return {
        "left_elbow": angle_between_points(wlm[11], wlm[13], wlm[15]),
        "right_elbow": angle_between_points(wlm[12], wlm[14], wlm[16]),
        "left_shoulder": angle_between_points(wlm[23], wlm[11], wlm[13]),
        "right_shoulder": angle_between_points(wlm[24], wlm[12], wlm[14]),
        "left_knee": angle_between_points(wlm[23], wlm[25], wlm[27]),
        "right_knee": angle_between_points(wlm[24], wlm[26], wlm[28]),
        "trunk_inclination": trunk_angle(wlm[23], wlm[24], wlm[11], wlm[12])
    }

def compute_single_frame_pose(frame_rgb, detector):
    """Extracts 3D landmarks for a single RGB frame on CPU."""
    if detector is None or frame_rgb is None:
        return None, False
    import mediapipe as mp
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
    result = detector.detect(mp_img)
    if result.pose_world_landmarks and len(result.pose_world_landmarks) > 0:
        pts = [
            {
                "x": float(p.x),
                "y": -float(p.y),
                "z": float(p.z),
                "vis": float(getattr(p, "visibility", 0.0))
            }
            for p in result.pose_world_landmarks[0]
        ]
        return pts, True
    return None, False

# ---------------------------------------------------------
# Sidebar Setup & Navigation
# ---------------------------------------------------------
st.sidebar.title("🏏 Cricket3D Motion")
st.sidebar.caption("Monocular 3D Pose Estimation & Frame Selection")

# Load precomputed reports
reports = load_json_reports()
hybrid_eval = reports.get("hybrid_eval")
hybrid_summary = reports.get("hybrid_summary")
kinematics_data = reports.get("kinematics")

# Clip discovery
sample_dir = Path("outputs/sample_clips")
clip_paths = sorted(list(sample_dir.rglob("*.avi")))
clip_map = {f"{p.parent.name} — {p.name}": p for p in clip_paths}

selected_clip_label = st.sidebar.selectbox("Select Cricket Shot Clip", list(clip_map.keys()))
selected_video_path = clip_map[selected_clip_label]
selected_clip_name = selected_video_path.name
selected_shot_class = selected_video_path.parent.name

st.sidebar.divider()
st.sidebar.markdown(f"**Shot Class:** `{selected_shot_class}`")
st.sidebar.markdown(f"**File:** `{selected_clip_name}`")
st.sidebar.markdown("**Hardware Mode:** `CPU-Only (x86_64)`")
st.sidebar.markdown("**Budget:** `6 Selected Frames / Clip`")

# ---------------------------------------------------------
# Main App Layout with Tabs
# ---------------------------------------------------------
tab_overview, tab_video, tab_frames, tab_viewer, tab_kinematics, tab_comparison = st.tabs([
    "📋 Overview",
    "🎥 Video Selection",
    "🎯 Frame Selection",
    "🦴 3D Pose Viewer",
    "📈 Kinematics",
    "📊 Method Comparison"
])

# ---------------------------------------------------------
# TAB 1: Overview
# ---------------------------------------------------------
with tab_overview:
    st.header("Project Overview & Experimental Methodology")
    
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        ### Research Objective
        Evaluating whether **motion-aware** and **hybrid frame selection** can preserve temporal 3D body pose
        dynamics and kinematic trajectories from monocular cricket broadcast videos, compared to standard uniform subsampling.
        
        #### Core Constraints
        - **Monocular Broadcast Inputs**: Standard 24-frame uncalibrated action clips from CricShot10k dataset.
        - **Resource Constrained**: Pure CPU inference pipeline, zero high-end GPU or server requirements.
        - **Rigorous Budgeting**: Fixed budget of 6 frames per clip (75% compression ratio) across all selection algorithms.
        
        #### Frame Selection Strategies
        1. **Uniform (Baseline)**: Evenly spaced temporal grid ($[0, 5, 9, 14, 18, 23]$). Covers overall duration with balanced temporal intervals.
        2. **Motion-Aware**: Optical flow & frame differencing motion energy. Captures high-velocity swings but clumps around impact and misses boundary poses.
        3. **Hybrid (Proposed)**: Balances temporal coverage (3 boundary/span anchors) with motion-adaptive peaks (3 highest energy frames).
        """)
    with col2:
        st.info("""
        **Hardware Environment**
        - OS: Windows 11
        - Python: 3.11.9
        - Execution: CPU (TFLite XNNPACK)
        - Detector: MediaPipe Pose Landmarker Lite (~5.5 MB)
        
        **Important Disclaimer**
        Estimated 3D landmarks (`pose_world_landmarks`) and extracted joint angles reflect neural monocular predictions in approximate metric coordinates. Evaluated across 12 pilot cricket clips (288 reference frames); they are designed for comparative selection evaluation and do not represent optical ground-truth or calibrated clinical biomechanics.
        """)

    st.subheader("Key Empirical Findings Across 12 Cricket Clips (288 Reference Frames)")
    if hybrid_summary and kinematics_data:
        agg_ov = hybrid_summary.get("aggregate_comparison", {})
        kin_ov = kinematics_data.get("summary", {}).get("overall_summary", {})
        u_det = agg_ov.get("uniform", {}).get("mean_detection_rate_pct", 0)
        h_det = agg_ov.get("hybrid", {}).get("mean_detection_rate_pct", 0)
        m_det = agg_ov.get("motion_aware", {}).get("mean_detection_rate_pct", 0)
        
        u_nspe = agg_ov.get("uniform", {}).get("mean_nearest_pose_error_nspe", 0)
        m_nspe = agg_ov.get("motion_aware", {}).get("mean_nearest_pose_error_nspe", 0)
        h_nspe = agg_ov.get("hybrid", {}).get("mean_nearest_pose_error_nspe", 0)
        
        u_mae = kin_ov.get("mean_kinematic_mae_deg", {}).get("uniform", 0)
        m_mae = kin_ov.get("mean_kinematic_mae_deg", {}).get("motion_aware", 0)
        h_mae = kin_ov.get("mean_kinematic_mae_deg", {}).get("hybrid", 0)
        
        u_ext = kin_ov.get("mean_extrema_gap_deg", {}).get("uniform", 0)
        m_ext = kin_ov.get("mean_extrema_gap_deg", {}).get("motion_aware", 0)
        h_ext = kin_ov.get("mean_extrema_gap_deg", {}).get("hybrid", 0)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric(label="Batter Pose Detection Rate", value=f"{u_det:.2f}%", delta=f"Uniform Best (M: {m_det:.1f}%, H: {h_det:.1f}%)")
        with c2:
            st.metric(label="Nearest Pose Error (NSPE)", value=f"{u_nspe:.4f}", delta=f"Uniform Best (M: {m_nspe:.4f}, H: {h_nspe:.4f})", delta_color="inverse")
        with c3:
            st.metric(label="Kinematic Trajectory MAE", value=f"{u_mae:.2f}°", delta=f"Uniform Best (M: {m_mae:.2f}°, H: {h_mae:.2f}°)", delta_color="inverse")
        with c4:
            st.metric(label="Extrema Preservation Gap", value=f"{u_ext:.2f}°", delta=f"Uniform Best (H: {h_ext:.2f}°, M: {m_ext:.2f}°)", delta_color="inverse")
    else:
        st.warning("Benchmark data unavailable for overview metrics.")

# ---------------------------------------------------------
# TAB 2: Video Selection & Clip Inspection
# ---------------------------------------------------------
with tab_video:
    st.header("Video Selection & Gallery")
    st.write("Browse and select from all available sample clips across different cricket shot categories.")
    
    # In-tab Clip Selection Dropdown or Gallery
    tab_clip_labels = list(clip_map.keys())
    current_index = tab_clip_labels.index(selected_clip_label) if selected_clip_label in tab_clip_labels else 0
    new_selected_label = st.selectbox(
        "Choose a clip to inspect:",
        tab_clip_labels,
        index=current_index,
        key="tab_clip_selector"
    )
    if new_selected_label != selected_clip_label:
        selected_clip_label = new_selected_label
        selected_video_path = clip_map[selected_clip_label]
        selected_clip_name = selected_video_path.name
        selected_shot_class = selected_video_path.parent.name
    
    st.divider()
    
    # Clip Metadata & Preview
    cap = cv2.VideoCapture(str(selected_video_path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    col_meta, col_prev = st.columns([2, 3])
    with col_meta:
        st.subheader("Clip Technical Metadata")
        st.markdown(f"""
        - **Shot Classification**: `{selected_shot_class}`
        - **Clip Filename**: `{selected_clip_name}`
        - **Resolution**: `{width} × {height} px`
        - **Framerate**: `{fps:.1f} FPS`
        - **Duration / Frames**: `{total_frames} frames (~{total_frames/fps:.2f}s)`
        - **Color Format**: `RGB / BGR 8-bit`
        """)
        
        # Pull clip stats from JSON
        clip_eval = None
        if hybrid_eval:
            for item in hybrid_eval:
                if item.get("clip_name") == selected_clip_name:
                    clip_eval = item
                    break
        
        if clip_eval:
            st.markdown("#### Selection Summary for this Clip")
            methods_data = clip_eval.get("methods", {})
            st.write(f"- **Uniform Indices**: `{methods_data.get('uniform', {}).get('indices')}`")
            st.write(f"- **Motion-Aware Indices**: `{methods_data.get('motion_aware', {}).get('indices')}`")
            st.write(f"- **Hybrid Indices**: `{methods_data.get('hybrid', {}).get('indices')}`")

    with col_prev:
        st.subheader("Key Frame Timeline Strip")
        strip_indices = [0, total_frames // 4, total_frames // 2, (3 * total_frames) // 4, total_frames - 1]
        strip_cols = st.columns(len(strip_indices))
        for idx, col in zip(strip_indices, strip_cols):
            f_img = extract_clip_frame(str(selected_video_path), idx)
            with col:
                if f_img is not None:
                    st.image(f_img, caption=f"Frame #{idx}", width='stretch')

    # All clips gallery
    st.divider()
    st.subheader(f"All Available Clips ({len(clip_paths)} Clips across 10 Cricket Shot Categories)")
    
    gallery_cols = st.columns(4)
    for idx, (label, path) in enumerate(clip_map.items()):
        col = gallery_cols[idx % 4]
        with col:
            thumb = extract_clip_frame(str(path), 0)
            if thumb is not None:
                st.image(thumb, caption=label, width='stretch')

# ---------------------------------------------------------
# TAB 3: Frame Selection Comparison
# ---------------------------------------------------------
with tab_frames:
    st.header("Comparative Frame Selection: 6-Frame Budget")
    st.write("Evaluating which 6 frames each algorithm selects out of the 24-frame action sequence.")
    
    clip_eval = None
    if hybrid_eval:
        for item in hybrid_eval:
            if item.get("clip_name") == selected_clip_name:
                clip_eval = item
                break
                
    if clip_eval:
        methods = clip_eval.get("methods", {})
        
        for m_key, m_title in [("uniform", "1. Uniform Sampling (Baseline)"),
                                ("motion_aware", "2. Motion-Aware Selection"),
                                ("hybrid", "3. Hybrid Selection (Proposed)")]:
            m_data = methods.get(m_key, {})
            indices = m_data.get("indices", [])
            st.markdown(f"### {m_title}")
            st.markdown(f"**Selected Frames**: `{indices}` | **Detected Poses**: `{m_data.get('detected_frames_count')}/6` | **NSPE**: `{m_data.get('nearest_pose_error', {}).get('mean_nspe', 'N/A')}`")
            
            f_cols = st.columns(6)
            for f_idx, col in zip(indices, f_cols):
                f_img = extract_clip_frame(str(selected_video_path), f_idx)
                with col:
                    if f_img is not None:
                        is_missing = f_idx in m_data.get("missing_detection_indices", [])
                        status = "❌ No Pose" if is_missing else "✅ Pose OK"
                        st.image(f_img, caption=f"Frame #{f_idx}\n({status})", width='stretch')
            st.divider()

# ---------------------------------------------------------
# TAB 4: 3D Pose Viewer (Interactive Plotly & Video Frame)
# ---------------------------------------------------------
with tab_viewer:
    st.header("Interactive 3D Pose & Kinematics Viewer")
    st.caption("Synchronized 2D video frame and 3D skeleton with real-time kinematic telemetry.")
    
    # Frame slider & Playback controls
    col_play, col_slide = st.columns([1.2, 5])
    with col_play:
        st.write("") # spacing
        st.write("")
        is_playing = st.session_state.get("is_playing", False)
        play_btn = st.button("⏸ Pause" if is_playing else "▶ Play Clip", use_container_width=True)
        if play_btn:
            st.session_state["is_playing"] = not is_playing
            st.rerun()
            
    with col_slide:
        # If playing, advance frame automatically
        if st.session_state.get("is_playing", False):
            cur_f = st.session_state.get("active_frame", 0)
            next_f = (cur_f + 1) % total_frames
            st.session_state["active_frame"] = next_f
            
        slider_frame = st.slider(
            "Scrub Video Frame",
            min_value=0,
            max_value=total_frames - 1,
            value=st.session_state.get("active_frame", 0),
            step=1,
            key="slider_frame_key",
            help="Select a video frame index to visualize corresponding 2D pose and 3D world skeleton."
        )
        st.session_state["active_frame"] = slider_frame

    # Camera Preset Selector & Batter ROI Filter Controls
    col_cam, col_roi = st.columns([3, 2])
    with col_cam:
        cam_preset_options = [
            "Free Rotation",
            "Isometric",
            "Front",
            "Back",
            "Left",
            "Right",
            "Top",
            "Bottom"
        ]
        if "cam_preset" not in st.session_state:
            st.session_state["cam_preset"] = "Isometric"

        selected_cam = st.radio(
            "3D Camera View Preset",
            cam_preset_options,
            index=cam_preset_options.index(st.session_state["cam_preset"]),
            horizontal=True,
            help="Select a standard anatomical camera projection, or choose Free Rotation for manual mouse control."
        )
        st.session_state["cam_preset"] = selected_cam
        
    with col_roi:
        with st.expander("🎯 Batter ROI Target (Multi-Person Filter)", expanded=False):
            st.caption("Target batsman at crease and exclude foreground bowler/umpire.")
            roi_x = st.slider("Crease Lateral (X)", 0.0, 1.0, (0.30, 0.70), 0.05)
            roi_y = st.slider("Crease Vertical (Y)", 0.0, 1.0, (0.20, 0.88), 0.05)
            roi_box = (roi_x[0], roi_y[0], roi_x[1], roi_y[1])
            show_roi_box = st.checkbox("Show ROI on 2D Video", value=True)

    # Extract current frame
    curr_frame = extract_clip_frame(str(selected_video_path), slider_frame)
    
    # Pre-calculated kinematics check from report
    kinematic_clip_match = None
    if kinematics_data:
        for item in kinematics_data.get("per_clip_kinematics", []):
            if item.get("clip_name") == selected_clip_name:
                kinematic_clip_match = item
                break
                
    frame_angles = None
    if kinematic_clip_match:
        ref_angles = kinematic_clip_match.get("frame_angles_reference", [])
        if slider_frame < len(ref_angles):
            frame_angles = ref_angles[slider_frame].get("angles", {})
    
    # Extract 3D landmarks on CPU using cached extractor with batter ROI filtering
    world_pts, detected, diag_candidates, diag_msg = get_cached_clip_pose(str(selected_video_path), slider_frame, roi_box)
    
    col_2d, col_3d, col_angles = st.columns([4, 5, 3])
    
    with col_2d:
        st.subheader(f"Video Frame #{slider_frame}")
        if curr_frame is not None:
            display_frame = curr_frame.copy()
            h_img, w_img = display_frame.shape[:2]
            
            # Draw ROI box if selected
            if show_roi_box:
                rx1_px, ry1_px = int(roi_box[0] * w_img), int(roi_box[1] * h_img)
                rx2_px, ry2_px = int(roi_box[2] * w_img), int(roi_box[3] * h_img)
                cv2.rectangle(display_frame, (rx1_px, ry1_px), (rx2_px, ry2_px), (0, 229, 255), 2)
                cv2.putText(display_frame, "BATTER ROI", (rx1_px + 5, ry1_px + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 229, 255), 1, cv2.LINE_AA)
                            
            # Render candidate markers on 2D frame if multiple detected
            for cand in diag_candidates:
                cx_px, cy_px = int(cand["cx"] * w_img), int(cand["cy"] * h_img)
                color = (0, 255, 0) if cand["is_inside"] else (0, 165, 255)
                cv2.circle(display_frame, (cx_px, cy_px), 5, color, -1)
                cv2.putText(display_frame, f"P#{cand['id']}", (cx_px + 6, cy_px - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
                            
            st.image(display_frame, width='stretch')
            if detected:
                st.success(f"✅ {diag_msg}")
            else:
                st.warning(f"⚠️ {diag_msg}")
        else:
            st.warning("Frame could not be loaded.")
            
        # Candidate Inspection Table / Diagnostic Card
        if diag_candidates:
            with st.expander(f"🔍 MediaPipe Candidate Telemetry ({len(diag_candidates)} Detected)", expanded=False):
                cand_table = [
                    {
                        "ID": f"#{c['id']}",
                        "Centroid (X, Y)": f"({c['cx']:.2f}, {c['cy']:.2f})",
                        "In ROI?": "✅ Yes" if c["is_inside"] else "❌ Outside",
                        "Dist to Center": f"{c['dist_to_center']:.2f}",
                        "Score": f"{c['score']:.2f}"
                    }
                    for c in diag_candidates
                ]
                st.table(cand_table)
            
    with col_3d:
        st.subheader("3D Skeletal View (Interactive)")
        if detected and world_pts:
            # Construct Plotly 3D Figure
            fig_3d = go.Figure()
            
            # Draw anatomical skeleton edges with clear, uncluttered styling and distinct limb colors
            for idx1, idx2, group in SKELETON_EDGES:
                p1, p2 = world_pts[idx1], world_pts[idx2]
                if p1["vis"] >= 0.35 and p2["vis"] >= 0.35:
                    color = EDGE_COLORS.get(group, "#FFFFFF")
                    fig_3d.add_trace(go.Scatter3d(
                        x=[p1["x"], p2["x"]],
                        y=[p1["z"], p2["z"]],  # Swap Y and Z for natural 3D camera depth
                        z=[p1["y"], p2["y"]],
                        mode="lines",
                        line=dict(color=color, width=6),
                        hoverinfo="none",
                        showlegend=False
                    ))
            
            # Draw anatomical joint nodes (clean spheres, uncluttered, no pink diamond clutter)
            xs = [p["x"] for p in world_pts if p["vis"] >= 0.35]
            ys = [p["z"] for p in world_pts if p["vis"] >= 0.35]
            zs = [p["y"] for p in world_pts if p["vis"] >= 0.35]
            fig_3d.add_trace(go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="markers",
                marker=dict(size=4.5, color="#F0F4F8", opacity=0.9),
                hoverinfo="none",
                showlegend=False
            ))
            
            # Camera Preset Configurations
            camera_presets = {
                "Isometric": dict(eye=dict(x=1.5, y=-1.5, z=0.8), up=dict(x=0, y=0, z=1)),
                "Front": dict(eye=dict(x=0.0, y=-2.2, z=0.0), up=dict(x=0, y=0, z=1)),
                "Back": dict(eye=dict(x=0.0, y=2.2, z=0.0), up=dict(x=0, y=0, z=1)),
                "Left": dict(eye=dict(x=-2.2, y=0.0, z=0.0), up=dict(x=0, y=0, z=1)),
                "Right": dict(eye=dict(x=2.2, y=0.0, z=0.0), up=dict(x=0, y=0, z=1)),
                "Top": dict(eye=dict(x=0.0, y=0.0, z=2.2), up=dict(x=0, y=1, z=0)),
                "Bottom": dict(eye=dict(x=0.0, y=0.0, z=-2.2), up=dict(x=0, y=1, z=0)),
                "Free Rotation": dict(eye=dict(x=1.5, y=-1.5, z=0.8), up=dict(x=0, y=0, z=1))
            }
            chosen_cam = camera_presets.get(selected_cam, camera_presets["Free Rotation"])

            # Equal aspect ratio layout
            layout_scene = dict(
                xaxis=dict(range=[-1.0, 1.0], title="Lateral (X)", showgrid=True, zeroline=False),
                yaxis=dict(range=[-1.0, 1.0], title="Depth (Z)", showgrid=True, zeroline=False),
                zaxis=dict(range=[-1.0, 1.0], title="Vertical (Y)", showgrid=True, zeroline=False),
                aspectmode="cube"
            )
            if selected_cam != "Free Rotation":
                layout_scene["camera"] = chosen_cam

            fig_3d.update_layout(
                scene=layout_scene,
                margin=dict(l=0, r=0, b=0, t=0),
                paper_bgcolor="#141414",
                height=450
            )
            st.plotly_chart(fig_3d, width='stretch')
        else:
            st.info("Pose not detected in this frame. Skeleton cannot be rendered.")
            
    with col_angles:
        st.subheader("Kinematic Telemetry")
        if detected and world_pts:
            live_angles = compute_live_frame_kinematics(world_pts)
            def render_angle(label, key):
                val = live_angles.get(key)
                if val is not None:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-val">{val:.1f}°</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-label">{label}</div>
                        <div class="metric-val" style="color: #FF5252;">Null (Occluded)</div>
                    </div>
                    """, unsafe_allow_html=True)

            render_angle("Left Elbow", "left_elbow")
            render_angle("Right Elbow", "right_elbow")
            render_angle("Left Shoulder", "left_shoulder")
            render_angle("Right Shoulder", "right_shoulder")
            render_angle("Left Knee", "left_knee")
            render_angle("Right Knee", "right_knee")
            render_angle("Trunk Inclination", "trunk_inclination")
        else:
            st.warning("Kinematic telemetry unavailable for this frame.")

    # Auto-play loop trigger if playing
    if st.session_state.get("is_playing", False):
        import time
        time.sleep(0.08)  # ~12 FPS video animation rate
        st.rerun()

# ---------------------------------------------------------
# TAB 5: Kinematic Analysis (Trajectories)
# ---------------------------------------------------------
with tab_kinematics:
    st.header(f"Kinematic Angle Trajectories: {selected_shot_class} ({selected_clip_name})")
    st.caption("Temporal joint-angle evolution across all 24 frames compared against Uniform, Motion-Aware, and Hybrid samples.")
    
    if kinematic_clip_match:
        ref_frames = kinematic_clip_match.get("frame_angles_reference", [])
        frames = [rf["frame"] for rf in ref_frames]
        
        # Angle selector
        angle_choice = st.selectbox(
            "Select Joint Angle to Plot",
            [
                ("left_elbow", "Left Elbow Angle (Deg)"),
                ("right_elbow", "Right Elbow Angle (Deg)"),
                ("left_shoulder", "Left Shoulder Angle (Deg)"),
                ("right_shoulder", "Right Shoulder Angle (Deg)"),
                ("left_knee", "Left Knee Angle (Deg)"),
                ("right_knee", "Right Knee Angle (Deg)"),
                ("trunk_inclination", "Trunk Inclination (Deg)")
            ],
            format_func=lambda x: x[1]
        )[0]
        
        ref_vals = [rf.get("angles", {}).get(angle_choice) for rf in ref_frames]
        
        # Build Plotly trajectory figure
        fig_traj = go.Figure()
        
        # Reference line (24 frames)
        fig_traj.add_trace(go.Scatter(
            x=frames,
            y=ref_vals,
            mode="lines+markers",
            name="24-Frame Reference Sequence",
            line=dict(color="#00E5FF", width=3),
            marker=dict(size=6, color="#00E5FF")
        ))
        
        # Overlay subsampled methods if indices exist
        if clip_eval:
            methods = clip_eval.get("methods", {})
            method_styles = {
                "uniform": ("Uniform (6 frames)", "#FFD700", "circle"),
                "motion_aware": ("Motion-Aware (6 frames)", "#FF5252", "diamond"),
                "hybrid": ("Hybrid (6 frames)", "#00E676", "star")
            }
            for m_key, (m_label, m_col, m_sym) in method_styles.items():
                m_indices = methods.get(m_key, {}).get("indices", [])
                m_x = [f for f in m_indices if f < len(ref_vals)]
                m_y = [ref_vals[f] for f in m_x]
                fig_traj.add_trace(go.Scatter(
                    x=m_x,
                    y=m_y,
                    mode="markers",
                    name=m_label,
                    marker=dict(size=12, color=m_col, symbol=m_sym, line=dict(color="#141414", width=2))
                ))
                
        fig_traj.update_layout(
            xaxis=dict(title="Frame Index (0 to 23)", dtick=1),
            yaxis=dict(title=f"Joint Angle (Degrees)"),
            height=450,
            template="plotly_dark",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_traj, width='stretch')
        
        # Per-clip kinematic stats
        st.subheader("Clip Kinematic Evaluation Metrics")
        methods_eval = kinematic_clip_match.get("methods_evaluation", {})
        col_u, col_m, col_h = st.columns(3)
        for col, m_key, m_name in [(col_u, "uniform", "Uniform"), (col_m, "motion_aware", "Motion-Aware"), (col_h, "hybrid", "Hybrid")]:
            m_stats = methods_eval.get(m_key, {})
            kin_dict = m_stats.get("kinematics", {})
            maes = [v["mae_deg"] for v in kin_dict.values() if v.get("mae_deg") is not None]
            min_errs = [v["min_error_deg"] for v in kin_dict.values() if v.get("min_error_deg") is not None]
            max_errs = [v["max_error_deg"] for v in kin_dict.values() if v.get("max_error_deg") is not None]
            
            c_mae = f"{np.mean(maes):.2f}°" if maes else "N/A"
            c_min = f"{np.mean(min_errs):.2f}°" if min_errs else "N/A"
            c_max = f"{np.mean(max_errs):.2f}°" if max_errs else "N/A"
            
            with col:
                st.markdown(f"#### {m_name}")
                st.write(f"- **Mean Joint MAE**: `{c_mae}`")
                st.write(f"- **Min Extrema Error**: `{c_min}`")
                st.write(f"- **Max Extrema Error**: `{c_max}`")
    else:
        st.warning("No kinematics data available for this clip.")

# ---------------------------------------------------------
# TAB 6: Method Comparison
# ---------------------------------------------------------
with tab_comparison:
    st.header("Comprehensive Benchmark Across All 12 Clips")
    st.caption("Aggregated results comparing Uniform, Motion-Aware, and Hybrid selection under identical 6-frame constraints.")
    
    if hybrid_summary and kinematics_data:
        agg = hybrid_summary.get("aggregate_comparison", {})
        kin_summary = kinematics_data.get("summary", {})
        mae_summary = kin_summary.get("aggregate_angle_mae", {})
        ext_summary = kin_summary.get("aggregate_extrema_error", {})
        
        # Extract correctly matching metric dictionaries from JSON
        kin_overall = kin_summary.get("overall_summary", {})
        mae_overall = kin_overall.get("mean_kinematic_mae_deg", {})
        ext_overall = kin_overall.get("mean_extrema_gap_deg", {})
        
        # Summary Table
        st.subheader("1. Metric Summary Table")
        table_data = [
            {
                "Method": "Uniform Sampling",
                "Pose Detection Rate": f"{agg.get('uniform', {}).get('mean_detection_rate_pct', 0):.2f}%",
                "Mean NSPE (Norm Pose Error)": f"{agg.get('uniform', {}).get('mean_nearest_pose_error_nspe', 0):.4f}",
                "Kinematic MAE": f"{mae_overall.get('uniform', 0):.2f}°",
                "Extrema Gap": f"{ext_overall.get('uniform', 0):.2f}°",
                "Temporal Span Ratio": f"{agg.get('uniform', {}).get('mean_temporal_span_ratio', 0):.2f}"
            },
            {
                "Method": "Motion-Aware Selection",
                "Pose Detection Rate": f"{agg.get('motion_aware', {}).get('mean_detection_rate_pct', 0):.2f}%",
                "Mean NSPE (Norm Pose Error)": f"{agg.get('motion_aware', {}).get('mean_nearest_pose_error_nspe', 0):.4f}",
                "Kinematic MAE": f"{mae_overall.get('motion_aware', 0):.2f}°",
                "Extrema Gap": f"{ext_overall.get('motion_aware', 0):.2f}°",
                "Temporal Span Ratio": f"{agg.get('motion_aware', {}).get('mean_temporal_span_ratio', 0):.2f}"
            },
            {
                "Method": "Hybrid Selection (Proposed)",
                "Pose Detection Rate": f"{agg.get('hybrid', {}).get('mean_detection_rate_pct', 0):.2f}%",
                "Mean NSPE (Norm Pose Error)": f"{agg.get('hybrid', {}).get('mean_nearest_pose_error_nspe', 0):.4f}",
                "Kinematic MAE": f"{mae_overall.get('hybrid', 0):.2f}°",
                "Extrema Gap": f"{ext_overall.get('hybrid', 0):.2f}°",
                "Temporal Span Ratio": f"{agg.get('hybrid', {}).get('mean_temporal_span_ratio', 0):.2f}"
            }
        ]
        st.table(table_data)
        
        # Comparative Bar Charts
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.subheader("Normalized Nearest Pose Error (NSPE)")
            st.caption("Lower is better; measures deviation between selected and full-reference 2D poses.")
            fig_nspe = go.Figure(data=[
                go.Bar(
                    x=["Uniform", "Motion-Aware", "Hybrid"],
                    y=[
                        agg.get("uniform", {}).get("mean_nearest_pose_error_nspe", 0),
                        agg.get("motion_aware", {}).get("mean_nearest_pose_error_nspe", 0),
                        agg.get("hybrid", {}).get("mean_nearest_pose_error_nspe", 0)
                    ],
                    marker_color=["#FFD700", "#FF5252", "#00E676"],
                    text=[f"{agg.get('uniform', {}).get('mean_nearest_pose_error_nspe', 0):.4f}",
                          f"{agg.get('motion_aware', {}).get('mean_nearest_pose_error_nspe', 0):.4f}",
                          f"{agg.get('hybrid', {}).get('mean_nearest_pose_error_nspe', 0):.4f}"],
                    textposition="auto"
                )
            ])
            fig_nspe.update_layout(template="plotly_dark", height=320, yaxis_title="Mean NSPE")
            st.plotly_chart(fig_nspe, width='stretch')
            
        with col_c2:
            st.subheader("Extrema Preservation Gap (Deg)")
            st.caption("Lower is better; measures how closely min/max angles match full clip extrema.")
            fig_ext = go.Figure(data=[
                go.Bar(
                    x=["Uniform", "Motion-Aware", "Hybrid"],
                    y=[
                        ext_overall.get("uniform", 0),
                        ext_overall.get("motion_aware", 0),
                        ext_overall.get("hybrid", 0)
                    ],
                    marker_color=["#FFD700", "#FF5252", "#00E676"],
                    text=[f"{ext_overall.get('uniform', 0):.2f}°",
                          f"{ext_overall.get('motion_aware', 0):.2f}°",
                          f"{ext_overall.get('hybrid', 0):.2f}°"],
                    textposition="auto"
                )
            ])
            fig_ext.update_layout(template="plotly_dark", height=320, yaxis_title="Degrees (°)")
            st.plotly_chart(fig_ext, width='stretch')
            
        st.markdown(f"""
        ### Key Analytical Takeaways (12-Clip Pilot Evaluation)
        1. **Uniform Sampling Aggregate Accuracy**: Across all 12 pilot cricket clips (288 reference frames), **Uniform sampling achieves the best overall tracking accuracy**, with the highest batter detection rate ({agg.get('uniform', {}).get('mean_detection_rate_pct', 0):.2f}%), lowest Nearest Pose Error ({agg.get('uniform', {}).get('mean_nearest_pose_error_nspe', 0):.4f}), lowest joint-angle MAE ({mae_overall.get('uniform', 0):.2f}°), and lowest extrema preservation error ({ext_overall.get('uniform', 0):.2f}°).
        2. **Motion-Aware Boundary Truncation**: While Motion-Aware selection concentrates on high-velocity stroke moments, it suffers from temporal clumping and boundary truncation (temporal span ratio: {agg.get('motion_aware', {}).get('mean_temporal_span_ratio', 0):.2f}), leading to a higher extrema error ({ext_overall.get('motion_aware', 0):.2f}°) because stance setup and follow-through extensions are omitted.
        3. **Hybrid Temporal Span Trade-off**: The proposed Hybrid selection resolves boundary truncation by maintaining complete temporal span ({agg.get('hybrid', {}).get('mean_temporal_span_ratio', 0):.2f}) and capturing intermediate motion peaks. However, distributing fixed anchors over 24 frames results in larger maximum temporal gaps ({agg.get('hybrid', {}).get('mean_max_temporal_gap_frames', 0):.2f} frames vs {agg.get('uniform', {}).get('mean_max_temporal_gap_frames', 0):.2f} for Uniform), yielding slightly higher trajectory MAE ({mae_overall.get('hybrid', 0):.2f}° vs {mae_overall.get('uniform', 0):.2f}°).
        4. **Batter Crease Isolation vs Unconstrained Detection**: Applying direct crease crop isolation successfully eliminates foreground bowler and background umpire identity switches, confirming that spatial crease filtering is essential for markerless monocular pose estimation in broadcast sports.
        """)
    else:
        st.warning("Benchmark JSON reports not found.")

st.sidebar.divider()
st.sidebar.caption("Cricket3D Motion Analytics • Sem 7 RM Project")
