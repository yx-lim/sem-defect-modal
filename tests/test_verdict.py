import numpy as np

from sem.verdict import verdict_for_groups
from sem.kpi import ARTIFACT_KPIS


def _ref(n=7, base=None):
    rng = np.random.default_rng(1)
    base = base or {}
    return {f"r{i}": {"area_frac_void": float(rng.normal(0.05, 0.005)),
                     "curtaining_score": float(rng.normal(0.3, 0.01)),
                     "brightness_mean": float(rng.normal(150, 1)),
                     **base} for i in range(n)}


def test_abstain_few_ref():
    v = verdict_for_groups({"t": {"area_frac_void": 0.5}}, {"r": {"area_frac_void": 0.05}},
                           ["x"], "all", [])
    assert v.verdict == "abstain"


def test_within_bounds():
    ref = _ref()
    test = {"t1": {"area_frac_void": 0.051, "curtaining_score": 0.30,
                   "brightness_mean": 150.2}}
    v = verdict_for_groups(test, ref, ["x"], "all", list(ref))
    assert v.verdict == "within_bounds", v.reasons


def test_outside_bounds():
    ref = _ref()
    test = {"t1": {"area_frac_void": 0.5, "curtaining_score": 0.30,
                   "brightness_mean": 150.0}}
    v = verdict_for_groups(test, ref, ["x"], "all", list(ref))
    # n_test=1 <3 so |z|>=3 alone suffices
    assert v.verdict == "outside_bounds"
    assert any("area_frac_void" in r for r in v.reasons)


def test_investigate_artifact_only():
    ref = _ref()
    test = {"t1": {"area_frac_void": 0.05, "curtaining_score": 0.9,
                   "brightness_mean": 150.0}}
    v = verdict_for_groups(test, ref, ["x"], "all", list(ref))
    assert v.verdict == "investigate"
    assert any("acquisition difference may confound" in r for r in v.reasons)


def test_artifact_kpis_excluded_from_family():
    ref = _ref()
    test = {"t1": {"area_frac_void": 0.05, "curtaining_score": 0.9,
                   "brightness_mean": 190.0}}
    v = verdict_for_groups(test, ref, ["x"], "all", list(ref))
    # artifact KPIs get z but no holm_p (excluded from correction family)
    curt = [r for r in v.per_kpi if r.kpi == "curtaining_score"][0]
    assert curt.robust_z is not None and curt.holm_p is None
    assert v.verdict == "investigate"  # artifact z>=3 -> investigate, not outside
