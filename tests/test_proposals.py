import numpy as np

from sem.proposals import dedupe, propose_for_image


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    img = rng.normal(180, 8, (800, 1200)).clip(0, 255).astype(np.uint8)
    # thin dark diagonal "crack" (~200px long, 2px wide)
    for i in range(200):
        y, x = 200 + i // 2, 300 + i
        img[max(0, y - 1):y + 1, x] = 40
    # big dark disk "void"
    yy, xx = np.ogrid[:800, :1200]
    img[(yy - 500) ** 2 + (xx - 800) ** 2 <= 40 ** 2] = 30
    return img


def test_proposals_find_crack_and_void():
    img = synthetic()
    props = propose_for_image(img, "B/g/BSE", "g", "B", "BSE", "run1")
    srcs = {p.source for p in props}
    assert "tophat_crack" in srcs or "anomaly_peak" in srcs
    assert "dark_void" in srcs
    crack = [p for p in props if p.source == "tophat_crack"]
    if crack:
        b = crack[0].bbox
        cx = (b[0] + b[2]) / 2
        assert 300 <= cx <= 600  # crack x range
    void = [p for p in props if p.source == "dark_void"]
    assert any((p.bbox[2] - p.bbox[0]) >= 60 for p in void)  # disk diameter 80


def test_dedupe_priority():
    box = (100, 100, 200, 200)
    cands = [
        (box, 0.5, "random"),
        ((110, 110, 210, 210), 0.9, "anomaly_peak"),
        ((105, 105, 205, 205), 0.7, "dark_void"),
        ((500, 500, 600, 600), 0.1, "random"),
    ]
    kept = dedupe(cands)
    kept_box0 = [k for k in kept if k[0][0] < 300]
    assert len(kept_box0) == 1 and kept_box0[0][2] == "anomaly_peak"
    assert any(k[2] == "random" and k[0][0] == 500 for k in kept)


def test_randoms_survive_cap():
    # >60 non-random candidates; all 15 non-overlapping randoms must survive
    cands = [((i * 5 % 700, 0, i * 5 % 700 + 50, 50), float(i), "tophat_crack")
             for i in range(80)]
    from sem.proposals import random_boxes
    cands += [(b, s, "random") for b, s in random_boxes((800, 1200), n=15, seed=0)]
    kept = dedupe(cands)
    n_rand = sum(1 for k in kept if k[2] == "random")
    n_nonrand = sum(1 for k in kept if k[2] != "random")
    assert n_nonrand <= 60
    assert n_rand == 15  # none overlap the kept band boxes (IoU<0.5)
    assert len(kept) <= 75


def test_random_source_present():
    img = np.full((500, 800), 200, np.uint8)
    props = propose_for_image(img, "B/g/BSE", "g", "B", "BSE", "run1")
    assert sum(1 for p in props if p.source == "random") == 15
