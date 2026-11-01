"""
Test script: validates batter detection across all 24 frames for Cover Drive, Pull, Scoop.
Imports only the helper functions directly, bypassing Streamlit module-level execution.
Run: python test_frames.py
"""
import sys, os, types, importlib.util
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# We import only the utility functions we need from app.py by grabbing them
# after patching streamlit + faking the module-level variables.
# ---------------------------------------------------------------------------

# --- minimal streamlit stub ---
st = types.ModuleType("streamlit")
def _noop(*a, **kw): return None
def _passthrough(f=None, **kw):
    if f is not None:
        return f
    return lambda g: g

for attr in dir(st):
    pass

# Set all the things app.py calls at module level
st.cache_data = _passthrough
st.cache_resource = _passthrough
st.set_page_config = _noop
st.session_state = {}

# Fake sidebar so selectbox works
st.sidebar = types.SimpleNamespace()
st.sidebar.selectbox = lambda label, opts, **kw: opts[0] if opts else ""
st.sidebar.title = _noop
st.sidebar.caption = _noop
st.sidebar.divider = _noop
st.sidebar.markdown = _noop
st.sidebar.slider = lambda *a, **kw: kw.get("value", 0)
st.sidebar.radio = lambda *a, **kw: (kw.get("options", [""])[0] if kw.get("options") else "")
st.sidebar.number_input = lambda *a, **kw: kw.get("value", 0)
st.sidebar.checkbox = lambda *a, **kw: False
st.sidebar.write = _noop

for attr in ["title","header","subheader","write","info","warning","error","success",
             "markdown","caption","divider","image","plotly_chart","dataframe","table",
             "metric","expander","container","empty","spinner","progress","button",
             "selectbox","slider","radio","number_input","checkbox","text_input",
             "multiselect","columns","tabs","rerun","stop","balloons","snow",
             "altair_chart","bar_chart","line_chart","area_chart","map","pyplot",
             "video","audio","file_uploader","download_button","color_picker",
             "date_input","time_input","text_area","form","form_submit_button",
             "status","toast","feedback","code","json","latex","html"]:
    if not hasattr(st, attr):
        setattr(st, attr, _noop)

sys.modules["streamlit"] = st

# --- We now exec only the top portion of app.py (functions + constants, NOT the UI) ---
# Read app.py, find where the UI starts (sidebar setup), and exec only through function defs
with open("app.py", encoding="utf-8") as f:
    full_src = f.read()

# Find the line "# Sidebar Setup" which is where module-level UI starts
import re
marker = "# ---------------------------------------------------------\n# Sidebar Setup"
cutoff = full_src.find(marker)
if cutoff == -1:
    # fallback: just exec everything with full stub
    cutoff = len(full_src)

src_functions_only = full_src[:cutoff]

# Build a namespace with required globals
g = {
    "__name__": "__main__",
    "__file__": "app.py",
    "np": np,
    "cv2": cv2,
}
exec(compile(src_functions_only, "app.py", "exec"), g)

get_cached_clip_pose = g["get_cached_clip_pose"]
compute_live_frame_kinematics = g["compute_live_frame_kinematics"]

# ---------------------------------------------------------------------------
def test_clip(clip_path, roi, label, bad_frames=None):
    bad_frames = bad_frames or []
    print(f"\n{'='*70}")
    print(f"  {label}")
    print(f"  ROI: {roi}")
    print(f"{'='*70}")
    print(f"{'F':>3} | {'Det':>3} | {'H(m)':>5} | {'Angles':>6} | {'Src':<4} | Note")
    print(f"{'-'*3}-+-{'-'*3}-+-{'-'*5}-+-{'-'*6}-+-{'-'*4}-+-{'-'*35}")
    detected = 0
    for f in range(24):
        pts, det, diags, msg = get_cached_clip_pose(clip_path, f, roi, True)
        h = 0.0
        n_angles = 0
        if det and pts:
            detected += 1
            sh_y = (pts[11]["y"] + pts[12]["y"]) * 0.5
            ank_y = (pts[27]["y"] + pts[28]["y"]) * 0.5
            h = abs(sh_y - ank_y)
            angles = compute_live_frame_kinematics(pts)
            n_angles = sum(1 for v in angles.values() if v is not None)
        tag = "CROP" if "Crease ROI" in msg else ("FULL" if "full-frame" in msg else "----")
        note = "<-- WAS BAD" if f in bad_frames and det else ("<-- STILL FAIL" if f in bad_frames and not det else "")
        print(f" {f:3d} | {str(det)[0]}   | {h:5.2f} | {n_angles:6d} | {tag:<4} | {note}")
    print(f"\n  >> {detected}/24 detected")
    return detected

# ---------------------------------------------------------------------------
r1 = test_clip(
    "outputs/sample_clips/Cover Drive/vid100_41.avi",
    (0.35, 0.20, 0.65, 0.85),
    "Cover Drive — vid100_41.avi",
    bad_frames=[9,10,11,12,13,15,20,21]
)

r2 = test_clip(
    "outputs/sample_clips/Pull/vid100_29.avi",
    (0.30, 0.20, 0.70, 0.85),
    "Pull — vid100_29.avi"
)

r3 = test_clip(
    "outputs/sample_clips/Scoop/vid100_28.avi",
    (0.25, 0.20, 0.75, 0.90),
    "Scoop — vid100_28.avi"
)

print(f"\n{'='*70}")
print(f"  SUMMARY:  Cover Drive {r1}/24  |  Pull {r2}/24  |  Scoop {r3}/24")
print(f"{'='*70}\n")
