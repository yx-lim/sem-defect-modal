import numpy as np

from sem.anomaly import group_coreset, logo_bank


def test_group_coreset_deterministic_and_size():
    rng = np.random.default_rng(1)
    feats = rng.normal(size=(500, 32)).astype(np.float32)
    i1 = group_coreset(feats, ratio=0.05, proj_dim=16)
    i2 = group_coreset(feats, ratio=0.05, proj_dim=16)
    assert np.array_equal(i1, i2)  # deterministic
    assert len(i1) == 25  # 5% of 500
    assert len(set(i1.tolist())) == 25
    assert i1.min() >= 0 and i1.max() < 500


def test_logo_bank_excludes_scored_group():
    rng = np.random.default_rng(2)
    coresets = {}
    for g in ("g1", "g2", "g3"):
        f = rng.normal(size=(200, 8)).astype(np.float32)
        idx = group_coreset(f, ratio=0.1, proj_dim=4)
        coresets[g] = f[idx]  # 20 pts each
    bank = logo_bank(coresets, "g2")
    assert bank.shape == (40, 8)  # g1+g3 only
    # every bank row comes from g1 or g3 coresets
    rest = np.concatenate([coresets["g1"], coresets["g3"]])
    assert np.allclose(np.sort(bank, axis=0), np.sort(rest, axis=0))
    bank_all = logo_bank(coresets, "not_a_group")
    assert bank_all.shape == (60, 8)
