"""Tests for PhysScore — includes R3 (jerk range) and R6 (calibration) regressions."""

import numpy as np
from collections import OrderedDict
from physscore import (
    physscore, foot_skating, ground_penetration, smoothness, joint_limits,
    DEFAULT_RANGES, DEFAULT_WEIGHTS, calibrate_ranges, __version__,
)


def make_static(T=30):
    j = np.zeros((T, 24, 3))
    heights = [0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.6, 0.0, 0.0, 0.9,
               0.0, 0.0, 1.2, 1.1, 1.1, 1.4, 1.1, 1.1, 0.8, 0.8,
               0.5, 0.5, 0.4, 0.4]
    for idx, h in enumerate(heights):
        j[:, idx, 1] = h
    for idx in [1, 4, 7, 10]:
        j[:, idx, 0] = 0.1
    for idx in [2, 5, 8, 11]:
        j[:, idx, 0] = -0.1
    return j


def make_jittery(T=30):
    j = make_static(T)
    j += np.random.RandomState(42).randn(*j.shape) * 0.05
    j[:, [7, 8, 10, 11], 1] = np.maximum(j[:, [7, 8, 10, 11], 1], 0.0)
    return j


def test_version():
    assert __version__ == "0.2.0"


def test_output_format():
    r = physscore(make_static())
    assert set(r.keys()) == {"physscore", "raw", "normalized"}
    assert 0.0 <= r["physscore"] <= 1.0
    assert len(r["raw"]) == 6 and len(r["normalized"]) == 6


def test_static_low_score():
    assert physscore(make_static())["physscore"] < 0.5


def test_jittery_higher_jerk():
    assert smoothness(make_jittery()) > smoothness(make_static())


def test_ground_penetration():
    j = make_static()
    assert ground_penetration(j) == 0.0
    j[:, 10, 1] = -0.05
    assert ground_penetration(j) > 0.0


def test_shape_validation():
    for bad in [np.zeros((5, 24, 3)), np.zeros((30, 17, 3)), np.zeros((30, 24, 2))]:
        try:
            physscore(bad)
            assert False, f"should reject {bad.shape}"
        except AssertionError as e:
            if "should reject" in str(e):
                raise


def test_custom_weights():
    w = {k: 0.0 for k in DEFAULT_WEIGHTS}
    w["foot_skating"] = 1.0
    r = physscore(make_static(), weights=w)
    assert abs(r["physscore"] - r["normalized"]["foot_skating"]) < 1e-6


# ── [R3] Jerk normalization regression ──
def test_R3_jerk_range_widened():
    """The jerk range MUST be 50000 (not 15000). If this fails, R3 is regressed."""
    low, high = DEFAULT_RANGES["smoothness"]
    assert low == 0.0
    assert high == 50000.0, \
        f"R3 REGRESSED: jerk range is ({low}, {high}), expected (0, 50000)"


def test_R3_no_premature_saturation():
    """A moderately jittery sequence should not saturate at 1.0."""
    j = make_static()
    j += np.random.RandomState(7).randn(*j.shape) * 0.02
    r = physscore(j)
    assert r["normalized"]["smoothness"] < 0.99, \
        f"Jerk saturated at {r['normalized']['smoothness']:.3f}"


# ── [R6] Auto-recalibration regression ──
def test_R6_calibrate_ranges_exists():
    assert callable(calibrate_ranges)


def test_R6_calibrate_ranges_output():
    raw = {
        "foot_skating":       [0.1, 0.5, 1.0, 1.5, 2.5],
        "ground_penetration": [0.0, 0.01, 0.02, 0.03, 0.05],
        "smoothness":         [100, 500, 1000, 5000, 45000],
        "joint_limits":       [0.0, 0.05, 0.1, 0.2, 0.3],
        "self_penetration":   [0.0, 0.01, 0.05, 0.1, 0.15],
        "com_stability":      [0.01, 0.05, 0.1, 0.3, 0.6],
    }
    new_ranges = calibrate_ranges(raw, percentile=95)
    assert isinstance(new_ranges, OrderedDict)
    assert set(new_ranges.keys()) == set(DEFAULT_RANGES.keys())
    for k, (lo, hi) in new_ranges.items():
        assert lo == 0.0 and hi > 0.0, f"{k}: bad range ({lo}, {hi})"


def test_R6_calibrate_ranges_usable():
    raw = {k: [0.1, 0.2, 0.3] for k in DEFAULT_RANGES}
    new_ranges = calibrate_ranges(raw)
    r = physscore(make_static(), ranges=new_ranges)
    assert 0.0 <= r["physscore"] <= 1.0


def test_R6_calibrate_handles_missing_keys():
    raw = {"foot_skating": [0.1, 0.5]}
    new_ranges = calibrate_ranges(raw)
    assert new_ranges["ground_penetration"] == DEFAULT_RANGES["ground_penetration"]


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed != len(tests):
        raise SystemExit(1)
