# Cricket3D_Motion

A cricket-video analysis project that investigates how frame-selection
strategies affect batter-specific 3D pose estimation and kinematic
information preservation under a limited frame budget.

## 1. Project Overview

The project processes selected cricket batting clips, selects six
representative frames from each 24-frame clip, isolates the batter using
a crease-focused region of interest (ROI), estimates body pose using
MediaPipe Pose Landmarker Lite, and evaluates how well each selection
strategy preserves pose and joint-angle information compared with the
full-video reference sequence.

The project compares three frame-selection strategies:

-   **Uniform Sampling** -- selects frames at evenly distributed
    temporal positions.
-   **Motion-Aware Sampling** -- selects frames with high motion energy
    from temporal segments.
-   **Hybrid Sampling** -- combines temporal anchor frames with
    motion-selected frames.

The goal is not to claim that one method is universally best, but to
measure the trade-offs among pose detection, landmark visibility,
temporal coverage and kinematic preservation.

## 2. Research Focus

**Research question:** How does selecting six frames from a 24-frame
cricket batting clip affect the preservation of a batter's estimated 3D
pose and joint-angle motion?

The project focuses on:

1.  Batter-specific pose isolation in broadcast cricket footage.
2.  Comparison of three frame-selection methods under the same six-frame
    budget.
3.  Evaluation of pose visibility and temporal coverage.
4.  Evaluation of joint-angle trajectory and extrema preservation.
5.  An interactive dashboard for inspecting clips, selected frames, 3D
    poses and evaluation results.

## 3. Dataset

-   **Dataset:** CricShot10k.
-   **Evaluation subset:** 12 selected clips covering 10 batting-shot
    categories.
-   **Frames per clip:** 24.
-   **Frame resolution:** 896 × 540.
-   **Frame rate:** 30 FPS.
-   **Selection budget:** 6 frames per clip.

The selected clips are a pilot evaluation subset, not the complete
CricShot10k dataset.

Original dataset archives are retained separately. Do not overwrite or
modify the original archives.

## 4. Processing Pipeline

``` text
CricShot10k cricket clips
          |
          v
Video decoding and preprocessing
          |
          v
Grayscale frame-difference motion analysis
          |
          v
Frame selection (six frames)
   |          |          |
Uniform   Motion-Aware  Hybrid
   |          |          |
   +----------+----------+
              |
              v
Crease-region batter ROI isolation
              |
              v
MediaPipe Pose Landmarker Lite
              |
              v
3D pose landmarks (33 landmarks)
              |
              v
Seven joint-angle measurements
              |
              v
Evaluation against full-clip reference
              |
              v
Results, plots and Streamlit dashboard
```

## 5. Frame-Selection Methods

### 5.1 Uniform Sampling

Selects six frames distributed across the 24-frame clip. The implemented
frame indices are:

`[0, 5, 9, 14, 18, 23]`

### 5.2 Motion-Aware Sampling

Calculates motion energy using the mean absolute difference (MAD)
between consecutive grayscale frames. The clip is divided into temporal
segments, and a high-motion frame is selected from each segment.

### 5.3 Hybrid Sampling

Combines three temporal anchors (beginning, middle and end) with three
motion-selected frames from the remaining temporal bands. The final set
contains six unique frames.

## 6. Batter Isolation and Pose Estimation

A crease-focused ROI is applied to reduce interference from other people
appearing in broadcast footage.

Default normalized ROI:

`[0.30, 0.20, 0.70, 0.88]`

The ROI is applied before pose inference. Two-dimensional crop landmarks
are mapped back to full-frame coordinates for visualisation. MediaPipe
3D world landmarks are not transformed using this 2D coordinate mapping.

Pose estimation uses:

-   **Model:** MediaPipe Pose Landmarker Lite.
-   **Landmarks:** 33 body landmarks.
-   **Execution:** CPU inference.
-   **Kinematics:** Seven measurements:
    -   Left elbow angle
    -   Right elbow angle
    -   Left shoulder angle
    -   Right shoulder angle
    -   Left knee angle
    -   Right knee angle
    -   Trunk inclination

## 7. Evaluation Metrics

The project evaluates:

-   **Batter pose detection rate:** Percentage of evaluated frames with
    a valid batter pose.
-   **Mean landmark visibility:** Average visibility of detected body
    landmarks.
-   **Temporal span ratio:** Coverage of the clip's temporal range by
    selected frames.
-   **Mean maximum temporal gap:** Largest temporal gap between selected
    frames, averaged across clips.
-   **Normalized pose sequence error (NSPE):** Difference between
    selected and reference pose sequences using mutually visible
    landmarks.
-   **Kinematic trajectory MAE:** Mean absolute error between
    selected-frame and full-reference joint-angle trajectories.
-   **Extrema preservation gap:** Difference between reference and
    selected-sequence angle extrema.

The full 24-frame MediaPipe sequence is an estimated reference sequence.
It is **not** motion-capture ground truth.

For exact thresholds, denominators, missing-frame handling and formulas,
consult the implementation and corrected evaluation reports. Do not
infer or change metric definitions when preparing research results.

## 8. Corrected Aggregate Results

The following values are from the corrected batter-specific evaluation.
The reference column represents the full-clip MediaPipe estimated pose
sequence.

  Metric                        Full-clip reference   Uniform   Motion-Aware   Hybrid
  --------------------------- --------------------- --------- -------------- --------
  Batter pose detection                      95.14%    97.22%         93.06%   91.67%
  Mean landmark visibility                   0.8053    0.7991         0.8070   0.8046
  Temporal span ratio                        1.0000    1.0000         0.8188   1.0000
  Mean maximum temporal gap                    1.00      5.00           5.58     7.75
  NSPE                                     0.0000\*    0.0414         0.0437   0.0466
  Kinematic trajectory MAE                  0.00°\*    12.27°         12.36°   13.29°
  Extrema preservation gap                  0.00°\*    10.94°         15.12°   13.95°

`*` The zero values indicate the reference sequence's self-comparison
baseline, not independently measured ground-truth accuracy.

### Main observations

-   Uniform achieved the best aggregate NSPE, kinematic trajectory MAE
    and extrema preservation gap.
-   Motion-Aware achieved the highest mean landmark visibility, but its
    temporal span was lower.
-   Hybrid preserved the full temporal span, but had the largest mean
    maximum temporal gap and highest aggregate trajectory MAE.
-   These results describe the 12-clip pilot subset and should not be
    generalised to all cricket videos without further evaluation.

## 9. Project Structure

The project contains source code, reports, outputs and dashboard files.
Key files include:

``` text
Cricket3D_Motion/
├── app.py
├── src/
│   ├── inspect_videos.py
│   ├── motion_analysis.py
│   ├── hybrid_frame_selection.py
│   ├── corrected_batter_selection.py
│   ├── corrected_batter_kinematics.py
│   └── visualize_3d_pose.py
├── reports/
│   ├── corrected_batter_selection_evaluation.json
│   ├── corrected_batter_selection_summary.json
│   ├── corrected_batter_joint_angle_analysis.json
│   ├── phase7_validation_audit.json
│   ├── dashboard_consistency_audit.json
│   └── dashboard_consistency_summary.md
├── outputs/
│   ├── sample_clips/
│   ├── motion_comparisons/
│   └── kinematic_plots/
└── README.md
```

This is a guide to the important files, not a guarantee that every
directory is present in every checkout. Refer to the actual repository
for the complete structure.

## 10. Software and Environment

The recorded environment includes:

-   Python 3.11.9
-   OpenCV / opencv-contrib-python 5.0.0.93
-   NumPy 2.4.6
-   SciPy 1.17.1
-   ONNX Runtime 1.29.0 (CPU)
-   MediaPipe Tasks 1.2.0
-   Streamlit 1.63.0
-   Plotly 7.1.0
-   Matplotlib

The project was evaluated using CPU inference. Do not assume GPU
acceleration is enabled.

## 11. Running the Dashboard

From the project root, activate the environment containing the project
dependencies and run:

``` bash
streamlit run app.py
```

The dashboard provides these sections:

1.  Overview
2.  Video Selection
3.  Frame Selection
4.  3D Pose Viewer
5.  Kinematics
6.  Method Comparison

The 3D viewer provides interactive rotation and camera presets,
including front, back, left, right, top, bottom, isometric and free
rotation.

## 12. Important Evaluation Reports

Use the corrected reports as the primary sources for final research
results:

  ------------------------------------------------------------------------------------------
  File                                                   Purpose
  ------------------------------------------------------ -----------------------------------
  `reports/corrected_batter_selection_evaluation.json`   Corrected batter-specific
                                                         evaluation records

  `reports/corrected_batter_selection_summary.json`      Corrected aggregate selection
                                                         metrics

  `reports/corrected_batter_joint_angle_analysis.json`   Joint-angle and kinematic
                                                         evaluation

  `reports/phase7_validation_audit.json`                 Validation and metric consistency
                                                         audit

  `reports/dashboard_consistency_audit.json`             Dashboard/report consistency checks

  `reports/dashboard_consistency_summary.md`             Human-readable dashboard audit
                                                         summary
  ------------------------------------------------------------------------------------------

Older full-frame evaluation reports are historical and must not be
substituted for the corrected batter-specific results.

## 13. Research and Reporting Guidelines

When using this project for a research paper:

-   Report only experiments that were actually performed.
-   Use the corrected batter-specific reports for final results.
-   Keep the six-frame budget identical across the three methods.
-   Define every metric, threshold, denominator and missing-data rule.
-   Distinguish the full-clip estimated reference from motion-capture
    ground truth.
-   Do not claim statistical significance unless statistical tests were
    performed.
-   Do not claim that Hybrid is the most accurate method.
-   Do not describe the system as a new pose-estimation network.
-   Cite CricShot10k, MediaPipe and all external research appropriately.
-   Preserve the original dataset archives and historical reports.

## 14. Limitations

-   The evaluation uses 12 selected clips and is a pilot study.
-   The 3D pose is estimated from monocular RGB video.
-   The full-clip pose sequence is a reference generated by the same
    pose-estimation system, not physical ground truth.
-   Motion blur, occlusion and partial-body visibility can cause missed
    or incomplete detections.
-   Crease-region cropping depends on the batter appearing within the
    selected ROI.
-   CPU inference limits processing speed compared with an appropriately
    configured accelerated environment.

## 15. Future Work

Potential future work includes evaluation on a larger and more diverse
set of cricket videos, comparison against motion-capture or calibrated
multiview ground truth, improved handling of occlusion and partial-body
visibility, and investigation of adaptive frame budgets.

These are future directions and are not claimed as completed features.

## 16. Reproducibility and Data Integrity

-   Keep original dataset archives unchanged.
-   Do not overwrite corrected reports with historical results.
-   Preserve the selected clip list and frame budget.
-   Record software versions and experimental parameters when rerunning
    evaluations.
-   Verify aggregate values against per-clip records before updating
    paper tables.
-   Keep source code, plots and reports together so that paper claims
    can be traced to implementation evidence.
