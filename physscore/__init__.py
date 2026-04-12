"""
PhysScore v0.2.0: Physics-based plausibility metrics for 3D human pose estimation.

Given (T, 24, 3) SMPL joint positions, computes 6 physics sub-metrics
and a weighted composite score. No training, no ground truth, no simulator.

Paper: "Beyond MPJPE: A Physics-Based Audit of Monocular 3D Human
Pose Estimation", PhysHuman Workshop @ CVPR 2026.
"""

import numpy as np
from collections import OrderedDict

__version__ = "0.2.0"

# ── SMPL 24-joint names and indices ──────────────────────────────────

JOINT_NAMES = [
    "pelvis", "left_hip", "right_hip", "spine1",
    "left_knee", "right_knee", "spine2",
    "left_ankle", "right_ankle", "spine3",
    "left_foot", "right_foot", "neck",
    "left_collar", "right_collar", "head",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hand", "right_hand",
]
_IDX = {name: i for i, name in enumerate(JOINT_NAMES)}
_FOOT_JOINTS = [_IDX["left_ankle"], _IDX["right_ankle"],
                _IDX["left_foot"],  _IDX["right_foot"]]


# ── Sub-metrics ──────────────────────────────────────────────────────

def foot_skating(joints, fps=30.0, contact_thresh=0.05):
    """Horizontal foot velocity during ground contact (m/s). Lower = better."""
    dt = 1.0 / fps
    foot_pos = joints[:, _FOOT_JOINTS, :]
    foot_vel = np.diff(foot_pos, axis=0) / dt
    foot_speed_xz = np.linalg.norm(foot_vel[:, :, [0, 2]], axis=-1)
    foot_heights = foot_pos[:-1, :, 1]
    ground_level = foot_heights.min()
    contact = foot_heights < (ground_level + contact_thresh)
    if contact.sum() == 0:
        return 0.0
    return float(np.mean(foot_speed_xz[contact]))


def ground_penetration(joints):
    """Average joint depth below ground plane (m). Lower = better."""
    min_y_per_frame = joints[:, :, 1].min(axis=1)
    return float(np.mean(np.maximum(0.0, -min_y_per_frame)))


def smoothness(joints, fps=30.0):
    """Mean jerk magnitude — 3rd derivative of joint positions (m/s³). Lower = smoother."""
    dt = 1.0 / fps
    vel = np.diff(joints, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.linalg.norm(jerk, axis=-1)))


def joint_limits(joints):
    """Fraction of (joint, frame) pairs violating anatomical ROM. Lower = better."""
    def _angle(j, parent, child):
        v1 = joints[:, parent] - joints[:, j]
        v2 = joints[:, child] - joints[:, j]
        cos_a = np.sum(v1 * v2, axis=-1) / (
            np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
        return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

    T = joints.shape[0]
    checks = [
        (_IDX["left_knee"],  _IDX["left_hip"],      _IDX["left_ankle"],  20, 175),
        (_IDX["right_knee"], _IDX["right_hip"],     _IDX["right_ankle"], 20, 175),
        (_IDX["left_elbow"],  _IDX["left_shoulder"], _IDX["left_wrist"],  10, 170),
        (_IDX["right_elbow"], _IDX["right_shoulder"],_IDX["right_wrist"], 10, 170),
    ]
    violations = np.zeros(T, dtype=bool)
    for (j, p, c, lo, hi) in checks:
        angles = _angle(j, p, c)
        violations |= (angles < lo) | (angles > hi)
    return float(np.mean(violations))


def self_penetration(joints, capsule_radius=0.04):
    """Fraction of non-adjacent bone pairs intersecting. Lower = better."""
    pairs = [
        (_IDX["left_hand"],  _IDX["right_hand"]),
        (_IDX["left_wrist"], _IDX["right_wrist"]),
        (_IDX["left_foot"],  _IDX["right_foot"]),
        (_IDX["left_ankle"], _IDX["right_ankle"]),
        (_IDX["pelvis"],     _IDX["left_foot"]),
        (_IDX["pelvis"],     _IDX["right_foot"]),
        (_IDX["head"],       _IDX["left_hand"]),
        (_IDX["head"],       _IDX["right_hand"]),
        (_IDX["spine2"],     _IDX["left_ankle"]),
        (_IDX["spine2"],     _IDX["right_ankle"]),
    ]
    T = joints.shape[0]
    pen_count, pen_total = 0, 0
    min_dist = capsule_radius * 2
    for (a, b) in pairs:
        dist = np.linalg.norm(joints[:, a] - joints[:, b], axis=-1)
        pen_count += int(np.sum(dist < min_dist))
        pen_total += T
    return float(pen_count / max(pen_total, 1))


def com_stability(joints):
    """Fraction of frames with CoM outside the support polygon. Lower = better."""
    from scipy.spatial import ConvexHull, Delaunay

    com = joints.mean(axis=1)
    T = joints.shape[0]
    outside = 0
    for t in range(T):
        support = joints[t, _FOOT_JOINTS][:, [0, 2]]
        com_xz = com[t, [0, 2]]
        try:
            unique_pts = np.unique(support, axis=0)
            if len(unique_pts) < 3:
                outside += 1
                continue
            hull = ConvexHull(unique_pts)
            tri = Delaunay(unique_pts[hull.vertices])
            if tri.find_simplex(com_xz) < 0:
                outside += 1
        except Exception:
            outside += 1
    return float(outside / T)


# ── Normalization [R3: jerk range widened 15000 → 50000] ─────────────

DEFAULT_RANGES = OrderedDict([
    ("foot_skating",       (0.0, 2.0)),
    ("ground_penetration", (0.0, 0.05)),
    ("smoothness",         (0.0, 50000.0)),   # [R3] widened from 15000
    ("joint_limits",       (0.0, 0.30)),
    ("self_penetration",   (0.0, 0.20)),
    ("com_stability",      (0.0, 0.80)),
])

DEFAULT_WEIGHTS = OrderedDict([
    ("foot_skating",       0.20),
    ("ground_penetration", 0.20),
    ("smoothness",         0.15),
    ("joint_limits",       0.15),
    ("self_penetration",   0.15),
    ("com_stability",      0.15),
])


def _normalize(value, low, high):
    return float(np.clip((value - low) / (high - low + 1e-8), 0, 1))


# ── [R6] Auto-recalibration tool ─────────────────────────────────────

def calibrate_ranges(all_raw_scores, percentile=95):
    """
    Auto-recalibrate normalization ranges from observed data.

    Enables researchers to adapt the sub-metric normalization to a new
    method cohort, addressing the dataset-dependence of the default ranges
    noted in the paper.

    Args:
        all_raw_scores: dict of {metric_name: list_of_raw_values_across_methods}
        percentile: upper percentile used as the normalization maximum (default 95)

    Returns:
        OrderedDict of (low, high) ranges suitable for passing as `ranges=`
        to the physscore() function.

    Example:
        >>> raw = {'foot_skating': [], 'ground_penetration': [], ...}
        >>> for method_joints in all_methods:
        ...     r = physscore(method_joints)
        ...     for k, v in r['raw'].items():
        ...         raw[k].append(v)
        >>> new_ranges = calibrate_ranges(raw)
        >>> result = physscore(joints, ranges=new_ranges)
    """
    ranges = OrderedDict()
    for k in DEFAULT_RANGES:
        if k not in all_raw_scores or len(all_raw_scores[k]) == 0:
            ranges[k] = DEFAULT_RANGES[k]
            continue
        values = np.asarray(all_raw_scores[k], dtype=np.float64)
        lo = 0.0
        hi = float(np.percentile(values, percentile))
        if hi <= lo:
            hi = float(np.max(values)) + 1e-8
        ranges[k] = (lo, hi)
    return ranges


# ── Main API ─────────────────────────────────────────────────────────

def physscore(joints, fps=30.0, weights=None, ranges=None, verbose=False):
    """
    Compute PhysScore: composite physics plausibility metric.

    Args:
        joints: np.ndarray of shape (T, 24, 3). SMPL 24-joint format,
                Y-up coordinate system, ground plane at y=0. Units: meters.
        fps: frame rate of the input sequence (default 30).
        weights: dict of sub-metric weights (default: paper weights).
        ranges: dict of (low, high) normalization ranges (default: DEFAULT_RANGES).
                Use calibrate_ranges() to generate custom ranges from observed data.
        verbose: if True, print per-metric breakdown.

    Returns:
        dict with keys:
            'physscore':  float, composite score in [0, 1]. 0 = physically perfect.
            'raw':        dict of raw sub-metric values in their native units.
            'normalized': dict of normalized sub-metric values in [0, 1].
    """
    joints = np.asarray(joints, dtype=np.float64)
    assert joints.ndim == 3 and joints.shape[1] == 24 and joints.shape[2] == 3, \
        f"Expected shape (T, 24, 3), got {joints.shape}"
    assert joints.shape[0] >= 10, f"Need >= 10 frames, got {joints.shape[0]}"

    w = weights if weights is not None else DEFAULT_WEIGHTS
    r = ranges if ranges is not None else DEFAULT_RANGES

    raw = OrderedDict([
        ("foot_skating",       foot_skating(joints, fps)),
        ("ground_penetration", ground_penetration(joints)),
        ("smoothness",         smoothness(joints, fps)),
        ("joint_limits",       joint_limits(joints)),
        ("self_penetration",   self_penetration(joints)),
        ("com_stability",      com_stability(joints)),
    ])

    normalized = OrderedDict()
    for k, v in raw.items():
        lo, hi = r[k]
        normalized[k] = _normalize(v, lo, hi)

    composite = sum(w[k] * normalized[k] for k in raw)

    if verbose:
        print(f"  {'Metric':<25} {'Raw':>12} {'Normalized':>12}")
        print(f"  {'-' * 50}")
        for k in raw:
            print(f"  {k:<25} {raw[k]:>12.4f} {normalized[k]:>12.3f}")
        print(f"  {'-' * 50}")
        print(f"  {'PHYSSCORE':<25} {'':>12} {composite:>12.3f}")

    return {
        "physscore": float(composite),
        "raw": dict(raw),
        "normalized": dict(normalized),
    }
