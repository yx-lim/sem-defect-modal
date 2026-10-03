from sem.ui import sort_pending, vlm_preselect, vlm_info_text


def test_queue_sort_uncertain_first_then_conf_asc():
    items = [
        {"proposal_id": "a", "vlm_suggestion": {"label": "crack_inter", "confidence": 0.9}},
        {"proposal_id": "b", "vlm_suggestion": {"label": "uncertain", "confidence": 0.7}},
        {"proposal_id": "c", "vlm_suggestion": {"label": "void", "confidence": 0.2}},
        {"proposal_id": "d", "vlm_suggestion": {"label": "uncertain", "confidence": 0.3}},
        {"proposal_id": "e", "vlm_suggestion": {"label": "crack_intra", "confidence": 0.5}},
        {"proposal_id": "f", "vlm_suggestion": None},
    ]
    order = [p["proposal_id"] for p in sort_pending(items)]
    assert order[:2] == ["d", "b"]          # uncertain first, conf asc
    assert order[2:] == ["c", "e", "a", "f"]  # rest conf asc, missing conf last


def test_vlm_preselect_mapping():
    assert vlm_preselect({"label": "background"}) == "normal"
    assert vlm_preselect({"label": "uncertain"}) is None
    assert vlm_preselect({"label": "crack_intra"}) == "crack_intra"
    assert vlm_preselect(None) is None
    assert "conf=0.5" in vlm_info_text({"label": "void", "confidence": 0.5,
                                       "is_artifact": False, "rationale": "r"})
