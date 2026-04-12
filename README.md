# PhysScore

**Physics-based plausibility metrics for 3D human pose estimation.**

PhysScore evaluates whether estimated 3D human motion is physically plausible, complementing geometric metrics like MPJPE. Six sub-metrics computed directly from SMPL joint positions — no training, no ground truth, no physics simulator required.

> **Paper**: _Beyond MPJPE: A Physics-Based Audit of Monocular 3D Human Pose Estimation_
> PhysHuman Workshop @ CVPR 2026

**Current version: 0.2.0** — incorporates camera-ready reviewer fixes ([R3] widened jerk normalization range, [R6] `calibrate_ranges()` auto-recalibration tool).

## Install

```bash
pip install physscore
```

From source:

```bash
git clone https://github.com/mujtabahasan/physscore.git
cd physscore
pip install -e .
```

## Usage

```python
import numpy as np
from physscore import physscore

# joints: (T, 24, 3) numpy array, SMPL 24-joint format
# Y-up coordinate system, ground plane at y = 0, units in meters
joints = np.load("method_output.npy")

result = physscore(joints)

print(f"PhysScore: {result['physscore']:.3f}")   # 0 = perfect, 1 = severe
print(f"Foot skating: {result['raw']['foot_skating']:.3f} m/s")
print(f"Jerk:         {result['raw']['smoothness']:.0f} m/s³")
```

Verbose mode prints a per-metric breakdown:

```python
result = physscore(joints, verbose=True)
```

```
  Metric                          Raw   Normalized
  ──────────────────────────────────────────────────
  foot_skating                 1.5013        0.190
  ground_penetration           0.0011        0.143
  smoothness                   3876.1        0.078
  joint_limits                 0.0067        0.154
  self_penetration             0.1521        0.121
  com_stability                0.0597        0.376
  ──────────────────────────────────────────────────
  PHYSSCORE                                  0.190
```

## Sub-Metrics

| Metric             | What it measures                                        | Unit     | Lower |
| ------------------ | ------------------------------------------------------- | -------- | :---: |
| Foot skating       | Horizontal foot velocity during ground contact          | m/s      |   ✓   |
| Ground penetration | Joint depth below ground plane                          | m        |   ✓   |
| Smoothness (jerk)  | 3rd time-derivative of joint positions                  | m/s³     |   ✓   |
| Joint limits       | Fraction of frames violating anatomical ROM             | fraction |   ✓   |
| Self-penetration   | Fraction of non-adjacent bone pairs intersecting        | fraction |   ✓   |
| CoM stability      | Fraction of frames with CoM outside the support polygon | fraction |   ✓   |

## Auto-recalibration for new method cohorts

If you are comparing a new set of methods whose error distribution differs substantially from the paper's cohort, use `calibrate_ranges()` to recompute normalization ranges from the observed raw scores:

```python
from physscore import physscore, calibrate_ranges

# Step 1: collect raw sub-metric values across all your methods
raw_scores = {
    "foot_skating": [], "ground_penetration": [],
    "smoothness": [],   "joint_limits": [],
    "self_penetration": [], "com_stability": [],
}
for method_joints in all_method_outputs:
    result = physscore(method_joints)
    for k, v in result["raw"].items():
        raw_scores[k].append(v)

# Step 2: build dataset-specific normalization ranges (95th percentile by default)
new_ranges = calibrate_ranges(raw_scores, percentile=95)

# Step 3: re-run PhysScore with the adapted ranges
result = physscore(joints, ranges=new_ranges)
```

This is the recommended workflow when new state-of-the-art methods extend beyond the range of the original cohort — it prevents sigmoid saturation and preserves discriminative power.

## Individual metrics

```python
from physscore import foot_skating, smoothness, joint_limits

skating    = foot_skating(joints, fps=30.0)
jerk       = smoothness(joints, fps=30.0)
violations = joint_limits(joints)
```

## Input format

- **Shape**: `(T, 24, 3)` with T ≥ 10 frames
- **Joint order**: standard SMPL 24-joint (pelvis, left_hip, right_hip, spine1, …)
- **Coordinate system**: Y-up
- **Ground plane**: y = 0 (shift minimum foot height to 0 before calling)
- **Units**: meters

## Dependencies

- `numpy >= 1.20`
- `scipy >= 1.7`

No PyTorch, no GPU, no SMPL model files.

## Changelog

### v0.2.0 (camera-ready)

- **[R3]** Widened jerk (smoothness) normalization range from `(0, 15000)` to `(0, 50000)` to prevent sigmoid saturation on modern per-frame methods.
- **[R6]** Added `calibrate_ranges(all_raw_scores, percentile=95)` auto-recalibration tool for new method cohorts.
- Added regression tests for both fixes.
- Exposed `DEFAULT_RANGES`, `DEFAULT_WEIGHTS`, and `calibrate_ranges` at package level.

### v0.1.0

- Initial release.

## Citation

```bibtex
@inproceedings{physscore2026,
  title={Beyond {MPJPE}: A Physics-Based Audit of Monocular 3{D} Human Pose Estimation},
  author={Hasan, Mujtaba},
  booktitle={PhysHuman Workshop, CVPR},
  year={2026}
}
```

## License

MIT
