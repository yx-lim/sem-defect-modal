# sem84 pipeline (Batch 3 SEM labelling)

84-feature, six-layer draft labelling pipeline for FIB-SEM battery-electrode images
(`sem84/`), plus the Batch 3 coordinator (`workflow/batch3_workflow.py`).
Outputs are AI-reviewed draft labels, not expert ground truth.

- Run one stage: `python -m sem84.cli run --image-id <id> --src <tif> --out <dir>/v001 --stage pass1|pass2|final [--review review.json] [--reference-src <BSE tif> --reference-id <BSE id>]`
- Validate: `python -m sem84.cli validate <stage_dir>`; package: `python -m sem84.cli package --version-dir <dir> --out <pkg.tar.gz>`
- Models (`models/`, not committed): MobileSAM ONNX files listed with SHA-256 in `model_sources.json`.
- Deps: numpy Pillow scipy matplotlib contourpy onnxruntime scikit-image tifffile.

Source recovered from the command log of the original run (v0.1.2 code, used for the 32 v001 drafts).
