"""Gradio review UI mounted on FastAPI (SPEC §3 ui).
Pure-gradio factory; Modal-independent. Backends are injected callables so the
same UI works locally and inside @modal.asgi_app."""

from __future__ import annotations

from .contract import CLASSES


def build_app(
    list_pending,      # () -> list[dict] pending_review labels+proposals
    get_crop_ctx,      # (proposal_id) -> (crop_path, ctx_path)
    submit_review,     # (proposal_id, label, reviewer_id) -> None
    list_images,       # () -> list[str]
    get_results,       # (image_id) -> (heatmap_path, overlay_path, kpis dict)
    get_audit_table,   # () -> list[dict]
    verify_chain,      # () -> dict
):
    import gradio as gr

    REVIEW_CHOICES = [c for c in CLASSES if c != "background"] + ["normal", "reject", "skip"]

    def _queue():
        return [[p.get("proposal_id"), p.get("image_id"), p.get("source"),
                 (p.get("vlm_suggestion") or {}).get("label")] for p in list_pending()]

    def _pick(pid):
        crop, ctx = get_crop_ctx(pid)
        return crop, ctx

    def _submit(pid, choice, reviewer):
        if choice == "skip" or not pid:
            return "skipped"
        lab = "rejected" if choice == "reject" else ("background" if choice == "normal" else choice)
        submit_review(pid, lab, reviewer or "anonymous")
        return f"saved {choice} for {pid}"

    with gr.Blocks(title="SEM defect review") as demo:
        with gr.Tab("Review"):
            q = gr.Dataframe(headers=["proposal_id", "image_id", "source", "vlm"],
                             value=_queue(), interactive=False)
            pid = gr.Textbox(label="proposal_id")
            with gr.Row():
                crop_img = gr.Image(label="crop")
                ctx_img = gr.Image(label="context")
            rev = gr.Textbox(label="reviewer_id")
            choice = gr.Radio(REVIEW_CHOICES, label="label")
            out = gr.Textbox(label="status")
            gr.Button("Refresh queue").click(_queue, outputs=q)
            gr.Button("Load").click(_pick, inputs=pid, outputs=[crop_img, ctx_img])
            gr.Button("Submit").click(_submit, inputs=[pid, choice, rev], outputs=out)
        with gr.Tab("Results"):
            iid = gr.Dropdown(choices=list_images(), label="image_id")
            with gr.Row():
                heat = gr.Image(label="anomaly heatmap")
                ovl = gr.Image(label="predicted mask overlay")
            kpi_out = gr.JSON(label="KPIs")
            gr.Button("Show").click(
                lambda i: get_results(i), inputs=iid, outputs=[heat, ovl, kpi_out])
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


def mount_fastapi(demo, path: str = "/"):
    """Mount a Blocks on a FastAPI app."""
    from fastapi import FastAPI
    import gradio as gr

    app = FastAPI()
    return gr.mount_gradio_app(app, demo, path=path)
