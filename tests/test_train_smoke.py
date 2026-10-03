import numpy as np
import pytest
import torch.nn as nn

from sem.train import build_mask, train, group_split
from sem.contract import Label, Proposal, CLASS_INDEX


def _prop(pid, bbox):
    return Proposal(proposal_id=pid, image_id="B/g/BSE", group_id="g",
                    batch="B", detector="BSE", bbox=bbox, mask_rle=None,
                    source="dark_void", score=1.0, run_id="r")


def _lab(pid, label, status="accepted_vlm_only"):
    return Label(label_id="l" + pid, proposal_id=pid, label=label, confidence=0.9,
                 rationale="x", is_artifact=False, source="vlm", model_id="m",
                 prompt_version="v1", reviewer_id=None, vlm_suggestion=None,
                 status=status, created_at="t", label_version="v1")


def test_build_mask():
    rng = np.random.default_rng(0)
    img = rng.normal(180, 5, (400, 400)).clip(0, 255).astype(np.uint8)
    img[100:160, 100:160] = 30  # dark square
    p = _prop("p1", (100, 100, 160, 160))
    mask = build_mask(img, [p], [_lab("p1", "void")])
    assert (mask != 255).any()
    assert mask[130, 130] == CLASS_INDEX["void"]
    assert mask[0, 0] == 255


def test_group_split_disjoint():
    groups = [f"g{i}" for i in range(10)]
    batches = {g: "Batch_1" if i < 5 else "Batch_2" for i, g in enumerate(groups)}
    tr, va = group_split(groups, batches, seed=0)
    assert not set(tr) & set(va) and len(tr) + len(va) == 10


class StubBackbone(nn.Module):
    """Stand-in for dinov2_vitb14 in CPU tests: fixed random conv 'patch tokens'."""
    embed_dim = 8

    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 8, 1)
        torch_mod = __import__("torch")
        g = torch_mod.Generator().manual_seed(0)
        self.proj.weight.data.normal_(0, 0.1, generator=g)

    def forward_features(self, x):
        import torch
        B = x.shape[0]
        tok = self.proj(x.mean((2, 3), keepdim=True).expand(-1, -1, 4, 4))
        tok = tok.flatten(2).transpose(1, 2)  # [B,16,8] -> 4x4 "grid"
        return {"x_norm_patchtokens": tok}

    def parameters(self, recurse=True):
        return iter([])


@pytest.mark.slow
def test_train_smoke_dinov2_head():
    from sem.train import DinoV2Head

    rng = np.random.default_rng(0)
    data = []
    for i in range(2):
        img = rng.normal(180, 8, (256, 256)).clip(0, 255).astype(np.uint8)
        img[50:110, 50:110] = 30
        mask = np.full((256, 256), 255, np.uint8)
        mask[50:110, 50:110] = CLASS_INDEX["void"]
        mask[150:180, 150:180] = CLASS_INDEX["background"]
        data.append((img, mask))
    model = DinoV2Head(StubBackbone(), n_classes=len(CLASS_INDEX))
    metrics = train(model, data, data, epochs=2, crop=128, device="cpu")
    assert "iou" in metrics and "dice" in metrics
