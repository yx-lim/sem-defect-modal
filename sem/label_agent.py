"""Label agent: crop rendering, PROMPT v1 (verbatim), strict JSON parse,
injectable client, first-5 accepted_vlm_only rule (SPEC §3 label_agent, §5)."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Protocol

import numpy as np

from .contract import ALLOWED_LABELS, ARTIFACT_CLASSES, Label, Proposal, VlmSuggestion

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """You are assisting a materials scientist labelling FIB-SEM cross-section images of lithium-ion battery electrodes (backscattered-electron detector, grayscale). Bright regions are usually active-material particles, dark regions are pores/binder. Normal electrodes contain many pores between particles; ordinary porosity is NOT a defect. You classify one proposed region. Be conservative: if you cannot tell, answer "uncertain". Respond with a single JSON object and nothing else."""

USER_PROMPT = """Image 1 is a close-up of the proposed region. Image 2 shows wider context with the region outlined in red.
Proposal source: {source} (a heuristic detector; it is often wrong).
Classify the outlined region as exactly one of:
- "crack_intra": thin dark linear fracture running through the inside of a particle.
- "crack_inter": separation or fracture along a particle boundary, between particle and binder, or at a layer interface.
- "void": an enclosed cavity or bubble clearly larger or rounder than the surrounding normal porosity.
- "agglomerate": a clump of fine particles or binder/carbon material distinct from the surrounding microstructure.
- "curtaining": vertical streaks or stripes caused by ion-beam milling (imaging artefact, not material).
- "edge_bloom": abnormally bright band or halo at an edge or surface caused by charging/edge effects (imaging artefact).
- "other_anomaly": clearly unusual structure that fits none of the above.
- "normal": typical electrode microstructure, including ordinary pores.
- "uncertain": cannot decide from these images.
Return: {"label": <one of the above>, "confidence": <0-1>, "is_artifact": <true if curtaining/edge_bloom or other imaging artefact>, "rationale": <one sentence describing the visual evidence>}"""


def _resize_long_side(img: np.ndarray, long_side: int) -> np.ndarray:
    import cv2

    h, w = img.shape[:2]
    scale = long_side / max(h, w)
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)


def render_crop(img: np.ndarray, bbox: tuple[int, int, int, int],
                min_size: int = 128, out_long: int = 384) -> np.ndarray:
    """Crop bbox padded to >= min_size, resized to `out_long` long side. RGB uint8."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    pad = max(0, (min_size - max(w, h)) // 2 + (0 if max(w, h) >= min_size else 1))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(img.shape[1], x1 + pad), min(img.shape[0], y1 + pad)
    crop = img[y0:y1, x0:x1]
    crop = _resize_long_side(crop, out_long)
    return np.stack([crop] * 3, axis=-1)


def render_context(img: np.ndarray, bbox: tuple[int, int, int, int],
                   area_mult: float = 4.0, out_long: int = 768) -> np.ndarray:
    """Context = ~4x bbox area around it, red rectangle around bbox, resized."""
    import cv2

    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    scale = np.sqrt(area_mult)
    cw, ch = w * scale, h * scale
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    X0, Y0 = int(max(0, cx - cw / 2)), int(max(0, cy - ch / 2))
    X1, Y1 = int(min(img.shape[1], cx + cw / 2)), int(min(img.shape[0], cy + ch / 2))
    ctx = np.stack([img[Y0:Y1, X0:X1]] * 3, axis=-1)
    rx0, ry0, rx1, ry1 = x0 - X0, y0 - Y0, x1 - X0, y1 - Y0
    thick = max(2, int(round(min(ctx.shape[:2]) * 0.008)))
    cv2.rectangle(ctx, (rx0, ry0), (rx1 - 1, ry1 - 1), (255, 0, 0), thick)
    return _resize_long_side(ctx, out_long)


def parse_vlm_json(text: str) -> VlmSuggestion | None:
    """Strict parse: single JSON object with required fields and allowed label."""
    try:
        text = text.strip()
        # tolerate code fences around the object
        if text.startswith("```"):
            text = text.strip("`")
            if text.lower().startswith("json"):
                text = text[4:]
        d = json.loads(text)
        if not isinstance(d, dict):
            return None
        label = d["label"]
        if label not in ALLOWED_LABELS:
            return None
        conf = float(d["confidence"])
        if not (0.0 <= conf <= 1.0):
            return None
        return VlmSuggestion(
            label=label,
            confidence=conf,
            is_artifact=bool(d["is_artifact"]),
            rationale=str(d["rationale"]),
        )
    except Exception:
        return None


class VlmClient(Protocol):
    def classify(self, system: str, user: str, crop_png: bytes, context_png: bytes) -> str:
        """Return raw text of the model response."""
        ...


def _png_bytes(rgb: np.ndarray) -> bytes:
    import cv2

    ok, buf = cv2.imencode(".png", rgb[:, :, ::-1])
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def stratified_order(proposals: list[Proposal], seed: int = 0) -> list[int]:
    """Stratified shuffle across batch x source (stable within strata)."""
    import random as _r

    rng = _r.Random(seed)
    strata: dict[tuple[str, str], list[int]] = {}
    for i, p in enumerate(proposals):
        strata.setdefault((p.batch, p.source), []).append(i)
    for idxs in strata.values():
        rng.shuffle(idxs)
    # round-robin across sorted strata keys
    order = []
    keys = sorted(strata)
    while any(strata[k] for k in keys):
        for k in keys:
            if strata[k]:
                order.append(strata[k].pop())
    return order


def label_proposals(
    proposals: list[Proposal],
    images: dict[str, np.ndarray],
    client: VlmClient,
    label_version: str,
    model_id: str | None = None,
    vlm_only_first_n: int = 5,
    seed: int = 0,
    max_retries: int = 1,
) -> list[Label]:
    """Label proposals in stratified order. First `vlm_only_first_n` successful
    labels get accepted_vlm_only; all others pending_review with vlm_suggestion.
    'uncertain' is always pending_review."""
    labels: list[Label] = []
    accepted = 0
    for i in stratified_order(proposals, seed=seed):
        p = proposals[i]
        img = images[p.image_id]
        crop_png = _png_bytes(render_crop(img, tuple(p.bbox)))
        ctx_png = _png_bytes(render_context(img, tuple(p.bbox)))
        user = USER_PROMPT.replace("{source}", p.source)
        sug = None
        for _ in range(max_retries + 1):
            raw = client.classify(SYSTEM_PROMPT, user, crop_png, ctx_png)
            sug = parse_vlm_json(raw)
            if sug is not None:
                break
        if sug is None:
            sug = VlmSuggestion(label="uncertain", confidence=0.0,
                                is_artifact=False, rationale="unparseable VLM response")
        label_val = "background" if sug.label == "normal" else sug.label
        if sug.label == "uncertain":
            status = "pending_review"
        elif accepted < vlm_only_first_n:
            status = "accepted_vlm_only"
            accepted += 1
        else:
            status = "pending_review"
        labels.append(
            Label(
                label_id=uuid.uuid4().hex[:12],
                proposal_id=p.proposal_id,
                label=label_val,
                confidence=sug.confidence,
                rationale=sug.rationale,
                is_artifact=sug.is_artifact or sug.label in ARTIFACT_CLASSES,
                source="vlm",
                model_id=model_id,
                prompt_version=PROMPT_VERSION,
                reviewer_id=None,
                vlm_suggestion=sug,
                status=status,
                created_at=datetime.now(timezone.utc).isoformat(),
                label_version=label_version,
            )
        )
    return labels


def make_anthropic_client(model_id: str, api_key: str | None = None) -> VlmClient:
    """Real Anthropic-backed client (used inside Modal). Lazy import."""
    import anthropic, base64

    cli = anthropic.Anthropic(api_key=api_key)

    class _C:
        def classify(self, system: str, user: str, crop_png: bytes, context_png: bytes) -> str:
            def blk(png):
                return {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": base64.b64encode(png).decode(),
                    },
                }

            msg = cli.messages.create(
                model=model_id,
                max_tokens=400,
                temperature=0,
                system=system,
                messages=[
                    {
                        "role": "user",
                        "content": [blk(crop_png), blk(context_png),
                                    {"type": "text", "text": user}],
                    }
                ],
            )
            return "".join(b.text for b in msg.content if b.type == "text")

    return _C()
