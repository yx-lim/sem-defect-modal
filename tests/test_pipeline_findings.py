import numpy as np
import pytest
import tifffile

from sem.pipeline import (_ref_scores, check_name, feature_path,
                          coreset_path, ingest, kpi_verdict)


def test_check_name():
    assert check_name(None, "model_version") is None
    assert check_name("v1", "label_version") == "v1"
    assert check_name("all", "reference_spec") == "all"
    assert check_name("batch:Batch_1", "reference_spec") == "batch:Batch_1"
    for bad in ("../x", "a/b"):
        with pytest.raises(ValueError):
            check_name(bad, "model_version")
    with pytest.raises(ValueError):
        check_name("batch:../x", "reference_spec")
    with pytest.raises(ValueError):
        check_name("Batch_1", "reference_spec")


def test_ref_scores_filters_groups():
    scored = {
        "i1": ({"group_id": "g1"}, np.array([1.0])),
        "i2": ({"group_id": "g2"}, np.array([2.0])),
        "i3": ({"group_id": "g3"}, np.array([3.0])),
    }
    out = _ref_scores(scored, {"g1", "g3"})
    assert [float(s[0]) for s in out] == [1.0, 3.0]


def _mk_src(tmp_path, pixels):
    src_dir = tmp_path / "src"
    (src_dir / "Batch_1").mkdir(parents=True)
    tifffile.imwrite(src_dir / "Batch_1" / "img_g1_BSE.tif", pixels)
    return src_dir


def test_ingest_changed_sha_relinks_and_invalidates(tmp_path):
    from sem.io import sha256_file

    src = _mk_src(tmp_path, np.zeros((64, 64), np.uint8))
    root = tmp_path / "work"
    r1 = ingest(root, src)
    assert r1["added"] == 1
    fp = feature_path(root, "Batch_1/g1/BSE")
    cp = coreset_path(root, "Batch_1/g1/BSE")
    for p in (fp, cp):
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.zeros(4))
    # change the source content -> different sha
    tifffile.imwrite(src / "Batch_1" / "img_g1_BSE.tif", np.ones((64, 64), np.uint8))
    r2 = ingest(root, src)
    assert r2["added"] == 1
    e = next(i for i in __import__("sem.pipeline", fromlist=["x"])
             .load_inventory(root)["images"])
    dst = __import__("pathlib").Path(e["path"])
    assert sha256_file(dst) == sha256_file(src / "Batch_1" / "img_g1_BSE.tif")
    assert not fp.exists() and not cp.exists()


def test_kpi_verdict_missing_pred_abstains(tmp_path):
    # inventory with two BSE entries; no pred pngs -> abstain
    inv = {"images": [
        {"image_id": "Batch_1/g1/BSE", "path": "x", "group_id": "g1",
         "batch": "Batch_1", "detector": "BSE", "sha256": "a",
         "px_size_nm": None, "shape": [1, 1]},
        {"image_id": "Batch_1/g2/BSE", "path": "x", "group_id": "g2",
         "batch": "Batch_1", "detector": "BSE", "sha256": "b",
         "px_size_nm": None, "shape": [1, 1]},
    ], "n_images": 2}
    (tmp_path / "meta").mkdir()
    (tmp_path / "meta" / "inventory.json").write_text(
        __import__("json").dumps(inv))
    v = kpi_verdict(tmp_path, ["Batch_1/g1/BSE"], model_version="mv1")
    assert v["verdict"] == "abstain"
    assert "Batch_1/g1/BSE" in v["reasons"][0]
    assert "Batch_1/g2/BSE" in v["reasons"][0]
