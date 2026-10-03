"""Verdict rules on KPI tables (SPEC §4). No accept/reject wording."""

from __future__ import annotations

import numpy as np

from .contract import KPIVerdict, KpiRow
from .kpi import ARTIFACT_KPIS, DEFECT_KPIS

MIN_REF_GROUPS = 5
Z_OUT = 3.0
Z_INVESTIGATE = 2.0
Z_ARTIFACT = 3.0


def _median_mad(vals):
    med = float(np.median(vals))
    mad = float(np.median(np.abs(np.asarray(vals) - med)))
    return med, mad


def robust_z(test_mean: float, ref_vals: np.ndarray):
    med, mad = _median_mad(ref_vals)
    if mad == 0:
        sd = float(np.std(ref_vals))
        if sd == 0:
            return med, mad, None, True  # degenerate
        return med, mad, (test_mean - med) / sd, False
    return med, mad, (test_mean - med) / (1.4826 * mad), False


def bootstrap_ci(test: np.ndarray, ref: np.ndarray, n: int = 2000, seed: int = 0):
    """CI of (test mean - ref mean) resampling GROUPS (rows = group means)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        t = rng.choice(test, size=len(test), replace=True).mean()
        r = rng.choice(ref, size=len(ref), replace=True).mean()
        diffs.append(t - r)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_p(test: np.ndarray, ref: np.ndarray, n: int = 5000, seed: int = 0):
    """Two-sided permutation test on group labels."""
    rng = np.random.default_rng(seed)
    obs = abs(test.mean() - ref.mean())
    pooled = np.concatenate([test, ref])
    nt = len(test)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        d = abs(pooled[:nt].mean() - pooled[nt:].mean())
        if d >= obs:
            count += 1
    return (count + 1) / (n + 1)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm correction over the provided family only."""
    items = sorted(((k, v) for k, v in pvals.items() if v is not None),
                   key=lambda kv: kv[1])
    m = len(items)
    out = {}
    running = 0.0
    for i, (k, p) in enumerate(items):
        adj = max(running, min(1.0, (m - i) * p))
        running = adj
        out[k] = adj
    return out


def verdict_for_groups(
    test_group_kpis: dict[str, dict[str, float]],
    ref_group_kpis: dict[str, dict[str, float]],
    image_ids: list[str],
    reference_spec: str,
    ref_image_ids: list[str],
    n_vlm_only_labels: int = 0,
    px_calibrated: bool = False,
) -> KPIVerdict:
    """test_group_kpis / ref_group_kpis: group_id -> {kpi: value}.
    Group aggregate = mean over that group's BSE images (already done upstream;
    here each entry is the group's image KPI dict)."""
    n_test, n_ref = len(test_group_kpis), len(ref_group_kpis)
    reasons: list[str] = []
    limitations = [
        "no engineering thresholds configured; no accept/reject mapping possible",
        f"labels partly VLM-only (n={n_vlm_only_labels})",
    ]
    if not px_calibrated:
        limitations.append("pixel-size calibration not found in TIFF tags; KPIs in px units")
    limitations.append("tiles are pseudo-replicates; unit of evaluation is the group")

    base = {
        "image_ids": image_ids,
        "reference": {"spec": reference_spec, "image_ids": ref_image_ids,
                      "n_groups": n_ref},
        "per_image": [{"image_id": iid, "kpis": test_group_kpis.get(gid, {})}
                      for iid, gid in zip(image_ids, test_group_kpis)],
        "engineering_thresholds": None,
        "limitations": limitations,
    }
    if n_ref < MIN_REF_GROUPS or n_test < 1:
        return KPIVerdict(
            **base,
            per_kpi=[],
            verdict="abstain",
            reasons=[f"insufficient reference groups (n_ref={n_ref} < {MIN_REF_GROUPS})"]
            if n_test >= 1 else ["no test groups"],
        )

    kpis = sorted({k for g in ref_group_kpis.values() for k in g
                   if g.get(k) is not None})
    rows: list[KpiRow] = []
    family_p: dict[str, float] = {}
    for kpi in kpis:
        ref_vals = np.array([g[kpi] for g in ref_group_kpis.values()
                             if g.get(kpi) is not None], dtype=float)
        test_vals = np.array([g[kpi] for g in test_group_kpis.values()
                              if g.get(kpi) is not None], dtype=float)
        if len(ref_vals) == 0 or len(test_vals) == 0:
            continue
        tm = float(test_vals.mean())
        med, mad, z, degen = robust_z(tm, ref_vals)
        lo_hi = bootstrap_ci(test_vals, ref_vals)
        perm = permutation_p(test_vals, ref_vals)
        rows.append(KpiRow(kpi=kpi, ref_median=med, ref_mad=mad, test_mean=tm,
                           robust_z=z, boot_ci95=lo_hi, perm_p=perm,
                           degenerate=degen))
        if kpi in DEFECT_KPIS and perm is not None and not degen:
            family_p[kpi] = perm
    holm_p = holm(family_p)
    for r in rows:
        if r.kpi in holm_p:
            r.holm_p = holm_p[r.kpi]

    # rules
    outside, investigate = [], []
    for r in rows:
        if r.robust_z is None or r.degenerate:
            continue
        in_family = r.kpi in DEFECT_KPIS
        if in_family and abs(r.robust_z) >= Z_OUT:
            hp = r.holm_p
            if n_test >= 3:
                if hp is not None and hp < 0.05:
                    outside.append(r.kpi)
            else:
                outside.append(r.kpi)
        if in_family and abs(r.robust_z) >= Z_INVESTIGATE:
            investigate.append(r.kpi)
        if r.kpi in ARTIFACT_KPIS and abs(r.robust_z) >= Z_ARTIFACT:
            investigate.append(r.kpi)

    for r in rows:
        if r.kpi in outside:
            reasons.append(f"{r.kpi}: |robust_z|={abs(r.robust_z):.2f}>=3"
                           + (f" holm_p={r.holm_p:.4f}<0.05" if n_test >= 3 and r.holm_p is not None else ""))
        elif r.kpi in investigate:
            if r.kpi in ARTIFACT_KPIS:
                reasons.append(f"{r.kpi}: |robust_z|={abs(r.robust_z):.2f}>=3 — acquisition difference may confound")
            else:
                reasons.append(f"{r.kpi}: |robust_z|={abs(r.robust_z):.2f}>=2")

    if outside:
        v = "outside_bounds"
    elif investigate:
        v = "investigate"
    else:
        v = "within_bounds"
    return KPIVerdict(**base, per_kpi=rows, verdict=v, reasons=reasons)
