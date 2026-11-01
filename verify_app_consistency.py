import json, sys
from pathlib import Path

# Load reports exactly as app.py does now
base_dir = Path("reports")
eval_data = json.load(open(base_dir / "corrected_batter_selection_evaluation.json", encoding="utf-8"))
sum_data = json.load(open(base_dir / "corrected_batter_selection_summary.json", encoding="utf-8"))
kin_data = json.load(open(base_dir / "corrected_batter_joint_angle_analysis.json", encoding="utf-8"))

print("=== CHECK 1: AGGREGATE SUMMARY FROM CORRECTED REPORTS ===")
agg = sum_data["aggregate_comparison"]
kin_ov = kin_data["summary"]["overall_summary"]
mae_ov = kin_ov["mean_kinematic_mae_deg"]
ext_ov = kin_ov["mean_extrema_gap_deg"]

expected = {
    "uniform": {"det": 97.22, "vis": 0.7991, "span": 1.0, "gap": 5.0, "nspe": 0.0414, "mae": 12.27, "ext": 10.94},
    "motion_aware": {"det": 93.06, "vis": 0.8070, "span": 0.8188, "gap": 5.58, "nspe": 0.0437, "mae": 12.36, "ext": 15.12},
    "hybrid": {"det": 91.67, "vis": 0.8046, "span": 1.0, "gap": 7.75, "nspe": 0.0466, "mae": 13.29, "ext": 13.95}
}

all_match = True
for m, exp in expected.items():
    act = {
        "det": agg[m]["mean_detection_rate_pct"],
        "vis": agg[m]["mean_landmark_visibility"],
        "span": agg[m]["mean_temporal_span_ratio"],
        "gap": agg[m]["mean_max_temporal_gap_frames"],
        "nspe": agg[m]["mean_nearest_pose_error_nspe"],
        "mae": mae_ov[m],
        "ext": ext_ov[m]
    }
    match = all(abs(act[k] - exp[k]) < 0.001 for k in exp)
    if not match: all_match = False
    print(f"[{'PASS' if match else 'FAIL'}] Method {m}:")
    print(f"       Act: {act}")
    print(f"       Exp: {exp}")

print(f"\nOverall Aggregate Values Validation: {'PASS' if all_match else 'FAIL'}")

print("\n=== CHECK 2: PER-CLIP KINEMATICS VALIDATION (TAB 5) ===")
clip0 = kin_data["per_clip_kinematics"][0]
print(f"Clip 0 Name: {clip0['clip_name']} ({clip0['class']})")
for m in ["uniform", "motion_aware", "hybrid"]:
    kin_dict = clip0["methods_evaluation"][m]["kinematics"]
    maes = [v["mae_deg"] for v in kin_dict.values() if v.get("mae_deg") is not None]
    min_errs = [v["min_error_deg"] for v in kin_dict.values() if v.get("min_error_deg") is not None]
    max_errs = [v["max_error_deg"] for v in kin_dict.values() if v.get("max_error_deg") is not None]
    import numpy as np
    c_mae = round(float(np.mean(maes)), 2) if maes else 0
    c_min = round(float(np.mean(min_errs)), 2) if min_errs else 0
    c_max = round(float(np.mean(max_errs)), 2) if max_errs else 0
    print(f"  {m:<14}: Mean MAE = {c_mae} deg, Min Err = {c_min} deg, Max Err = {c_max} deg")

print("\n=== CHECK 3: APP.PY SYNTAX & CONFIG VERIFICATION ===")
with open("app.py", encoding="utf-8") as f:
    src = f.read()

checks = [
    ("corrected_batter_selection_evaluation.json", "Report loading (hybrid_eval)"),
    ("corrected_batter_selection_summary.json", "Report loading (hybrid_summary)"),
    ("corrected_batter_joint_angle_analysis.json", "Report loading (kinematics)"),
    ("roi_x = st.slider(\"Crease Lateral (X)\", 0.0, 1.0, (0.30, 0.70), 0.05)", "ROI X default (0.30, 0.70)"),
    ("roi_y = st.slider(\"Crease Vertical (Y)\", 0.0, 1.0, (0.20, 0.88), 0.05)", "ROI Y default (0.20, 0.88)"),
    ("[0, 5, 9, 14, 18, 23]", "Uniform grid text"),
    ("Uniform sampling achieves the best overall tracking accuracy", "Analytical takeaway text"),
]

for s, desc in checks:
    present = s in src
    print(f"[{'PASS' if present else 'FAIL'}] {desc}")

