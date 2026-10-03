"""Gradio review UI mounted on FastAPI (SPEC §3 ui).
Pure-gradio factory; Modal-independent. Backends are injected callables so the
same UI works locally and inside @modal.asgi_app."""

from .contract import CLASSES

REVIEW_CHOICES = [c for c in CLASSES if c != "background"] + ["normal", "reject", "skip"]
QUEUE_HEADERS = ["proposal_id", "image_id", "source", "vlm", "conf"]


def sort_pending(items: list[dict]) -> list[dict]:
    """uncertain first, then confidence ascending, then the rest."""
    def key(it):
        sug = it.get("vlm_suggestion") or {}
        unc = sug.get("label") == "uncertain"
        return (0 if unc else 1, float(sug.get("confidence", 1.0)))

    return sorted(items, key=key)


def vlm_preselect(sug: dict | None):
    """Radio value for a VLM suggestion; background/normal -> normal;
    None for uncertain or missing."""
    if not sug:
        return None
    lab = sug.get("label")
    if lab in ("background", "normal"):
        return "normal"
    if lab in REVIEW_CHOICES:
        return lab
    return None


def vlm_info_text(sug: dict | None) -> str:
    if not sug:
        return ""
    return (f"label={sug.get('label')} conf={sug.get('confidence')} "
            f"is_artifact={sug.get('is_artifact')} — {sug.get('rationale')}")


def build_app(
    list_pending,      # () -> list[dict] pending_review items w/ vlm_suggestion
    get_crop_ctx,      # (proposal_id) -> (crop_path, ctx_path)
    submit_review,     # (proposal_id, label, reviewer_id) -> None
    list_images,       # () -> list[str]
    get_results,       # (image_id) -> (heatmap_path, overlay_path, kpis dict)
    get_audit_table,   # () -> list[dict]
    verify_chain,      # () -> dict
):
    import gradio as gr

    def _pending_sorted():
        return sort_pending(list_pending())

    def _queue():
        return [[p.get("proposal_id"), p.get("image_id"), p.get("source"),
                 (p.get("vlm_suggestion") or {}).get("label"),
                 (p.get("vlm_suggestion") or {}).get("confidence")]
                for p in _pending_sorted()]

    def _sug_for(pid):
        for p in _pending_sorted():
            if p.get("proposal_id") == pid:
                return p.get("vlm_suggestion")
        return None

    def _load(pid):
        if not pid:
            return None, None, "", gr.update(value=None)
        crop, ctx = get_crop_ctx(pid)
        sug = _sug_for(pid)
        return crop, ctx, vlm_info_text(sug), gr.update(value=vlm_preselect(sug))

    def _row_select(df, evt: gr.SelectData):
        pid = None
        try:
            pid = evt.row_value[0]
        except Exception:
            pass
        if not pid:
            try:
                pid = df[evt.index[0]][0]
            except Exception:
                pid = None
        return (pid,) + _load(pid)

    def _submit(pid, choice, reviewer):
        if not reviewer or not reviewer.strip():
            return ("reviewer_id required", _queue(), pid, *(_load(pid)))
        if choice == "skip" or not pid:
            return ("skipped", _queue(), pid, *(_load(pid)))
        lab = ("rejected" if choice == "reject"
               else ("background" if choice == "normal" else choice))
        submit_review(pid, lab, reviewer.strip())
        rows = _queue()
        nxt = rows[0][0] if rows else None
        n = len(rows)
        crop, ctx, info, radio = _load(nxt)
        return (f"saved {pid} as {lab} ({n} remaining)", rows, nxt, crop, ctx,
                info, radio)

    with gr.Blocks(title="SEM defect review") as demo:
        with gr.Tab("Review"):
            q = gr.Dataframe(headers=QUEUE_HEADERS, value=_queue(),
                             interactive=False)
            pid = gr.Textbox(label="proposal_id")
            with gr.Row():
                crop_img = gr.Image(label="crop")
                ctx_img = gr.Image(label="context")
            vlm_info = gr.Textbox(label="VLM suggestion", interactive=False)
            rev = gr.Textbox(label="reviewer_id")
            choice = gr.Radio(REVIEW_CHOICES, label="label")
            out = gr.Textbox(label="status")
            gr.Button("Refresh queue").click(_queue, outputs=q)
            gr.Button("Load").click(_load, inputs=pid,
                                    outputs=[crop_img, ctx_img, vlm_info, choice])
            q.select(_row_select, inputs=q,
                     outputs=[pid, crop_img, ctx_img, vlm_info, choice])
            gr.Button("Submit").click(
                _submit, inputs=[pid, choice, rev],
                outputs=[out, q, pid, crop_img, ctx_img, vlm_info, choice])
        with gr.Tab("Results"):
            iid = gr.Dropdown(choices=list_images(), label="image_id")
            with gr.Row():
                heat = gr.Image(label="anomaly heatmap")
                ovl = gr.Image(label="predicted mask overlay")
            kpi_out = gr.JSON(label="KPIs")
            status = gr.Markdown()
            gr.Button("Show").click(
                lambda i: get_results(i), inputs=iid,
                outputs=[heat, ovl, kpi_out, status])
        with gr.Tab("Audit"):
            tbl = gr.Dataframe(headers=["run_id", "function", "ended_at", "record_hash"],
                               value=[[r.get("run_id"), r.get("function"),
                                       r.get("ended_at"), r.get("record_hash")]
                                      for r in get_audit_table()])
            ver = gr.JSON(label="verify_chain")
            gr.Button("Refresh").click(lambda: [[r.get("run_id"), r.get("function"),
                                                r.get("ended_at"), r.get("record_hash")]
                                               for r in get_audit_table()], outputs=tbl)
            gr.Button("Verify chain").click(verify_chain, outputs=ver)
    return demo


def mount_fastapi(demo, path: str = "/", allowed_paths: list[str] | None = None):
    """Mount a Blocks on a FastAPI app; allowed_paths passed to gradio so
    local image files are servable."""
    from fastapi import FastAPI
    import gradio as gr

    app = FastAPI()
    return gr.mount_gradio_app(app, demo, path=path, allowed_paths=allowed_paths)
