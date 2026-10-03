import cv2
import numpy as np

from sem.pipeline import ANOMALY, results_for, write_preview


def test_results_for_preview_and_status(tmp_path):
    d = tmp_path / ANOMALY / "run1"
    d.mkdir(parents=True)
    png = np.random.default_rng(0).integers(0, 255, (1000, 4000, 3),
                                            dtype=np.uint8)
    cv2.imwrite(str(d / "Batch_1_x_BSE_heat.png"), png)

    heat, ovl, kpis, status = results_for(tmp_path, "Batch_1/x/BSE")
    assert heat.endswith("_preview.jpg")
    img = cv2.imread(heat)
    h, w = img.shape[:2]
    assert w <= 1600
    assert abs(w / h - 4000 / 1000) * min(h, w) <= 1 or abs(h - 1000 * w / 4000) <= 1
    assert ovl is None and kpis == {}
    assert "No predicted mask yet" in status
    assert "No KPIs yet" in status
    # idempotent
    assert write_preview(d / "Batch_1_x_BSE_heat.png").stat().st_mtime > 0


def test_results_for_unknown_image(tmp_path):
    (tmp_path / ANOMALY).mkdir(parents=True)
    heat, ovl, kpis, status = results_for(tmp_path, "Batch_9/zzz/BSE")
    assert heat is None and "No anomaly heatmap" in status
