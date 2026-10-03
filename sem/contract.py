"""Taxonomy and pydantic v2 contracts (SPEC §1-2)."""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

CLASSES = [
    "background",
    "crack_intra",
    "crack_inter",
    "void",
    "agglomerate",
    "curtaining",
    "edge_bloom",
    "other_anomaly",
]  # index = mask value
IGNORE = 255
ARTIFACT_CLASSES = {"curtaining", "edge_bloom"}
# extra answers the label agent may return
EXTRA_LABELS = ["normal", "uncertain"]

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

PROPOSAL_SOURCES = {
    "tophat_crack",
    "dark_void",
    "fft_curtain",
    "edge_band",
    "anomaly_peak",
    "microsam",
    "random",
}


class Proposal(BaseModel):
    proposal_id: str  # sha1(image_id+bbox+source)[:12]
    image_id: str
    group_id: str
    batch: str
    detector: str
    bbox: tuple[int, int, int, int]  # [x0,y0,x1,y1] full-res
    mask_rle: Optional[dict] = None  # COCO-style RLE within bbox
    source: Literal[
        "tophat_crack",
        "dark_void",
        "fft_curtain",
        "edge_band",
        "anomaly_peak",
        "microsam",
        "random",
    ]
    score: float
    run_id: str


class VlmSuggestion(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    is_artifact: bool
    rationale: str


class Label(BaseModel):
    label_id: str
    proposal_id: str
    label: str  # CLASSES + "normal" | "uncertain"
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str
    is_artifact: bool
    source: Literal["vlm", "human"]
    model_id: Optional[str] = None
    prompt_version: str
    reviewer_id: Optional[str] = None
    vlm_suggestion: Optional[VlmSuggestion] = None
    status: Literal[
        "accepted_vlm_only", "accepted_human", "rejected", "pending_review"
    ]
    created_at: str
    label_version: str


class FileHash(BaseModel):
    path: str
    sha256: str


class ModelRef(BaseModel):
    name: str
    weights_sha256: Optional[str] = None
    hub_id: Optional[str] = None
    revision: Optional[str] = None


class ModalInfo(BaseModel):
    function_call_id: Optional[str] = None
    gpu: Optional[str] = None
    est_cost_usd: Optional[float] = None


class AuditRecord(BaseModel):
    run_id: str
    function: str
    started_at: str
    ended_at: str
    wall_s: float
    git_commit: str
    git_dirty: bool
    config: dict
    config_sha256: str
    inputs: list[FileHash] = []
    outputs: list[FileHash] = []
    model: Optional[ModelRef] = None
    label_version: Optional[str] = None
    modal: Optional[ModalInfo] = None
    metrics: dict = {}
    verdict: Optional[dict] = None
    limitations: list[str] = []
    prev_hash: Optional[str] = None
    record_hash: Optional[str] = None


class KpiRow(BaseModel):
    kpi: str
    ref_median: Optional[float] = None
    ref_mad: Optional[float] = None
    test_mean: Optional[float] = None
    robust_z: Optional[float] = None
    boot_ci95: Optional[tuple[float, float]] = None
    perm_p: Optional[float] = None
    holm_p: Optional[float] = None
    degenerate: bool = False


class ReferenceInfo(BaseModel):
    spec: str
    image_ids: list[str]
    n_groups: int


class KPIVerdict(BaseModel):
    image_ids: list[str]
    reference: ReferenceInfo
    per_image: list[dict]
    per_kpi: list[KpiRow]
    verdict: Literal["within_bounds", "investigate", "outside_bounds", "abstain"]
    reasons: list[str]
    engineering_thresholds: Optional[dict] = None
    limitations: list[str]


ALLOWED_LABELS = set(CLASSES) | set(EXTRA_LABELS)
