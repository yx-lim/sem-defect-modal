# sem84 pipeline (Batch 3 SEM labelling)

84-feature, six-layer draft labelling pipeline for FIB-SEM battery-electrode images
(`sem84/`), plus the Batch 3 coordinator (`workflow/batch3_workflow.py`).
Outputs are AI-reviewed draft labels, not expert ground truth.

- Run one stage: `python -m sem84.cli run --image-id <id> --src <tif> --out <dir>/v001 --stage pass1|pass2|final [--review review.json] [--reference-src <BSE tif> --reference-id <BSE id>]`
- Validate: `python -m sem84.cli validate <stage_dir>`; package: `python -m sem84.cli package --version-dir <dir> --out <pkg.tar.gz>`
- Models (`models/`, not committed): MobileSAM ONNX files listed with SHA-256 in `model_sources.json`.
- Deps: numpy Pillow scipy matplotlib contourpy onnxruntime scikit-image tifffile.

Source recovered from the command log of the original run (v0.1.2 code, used for the 32 v001 drafts).

## Cheaper review flow (pilot)
`python -m sem84.crop_review review --stage-dir <v001>/pass1 --src <tif> --queue-dir <v001>/.verify/pass1 --out <v001>/review_pass2.json --reviewer crop-verifier-<id>`
builds a flagged-crop queue (<=25 raw+overlay panels), verifies each crop with Claude (needs `ANTHROPIC_API_KEY`), and writes a review file.
After pass2, rerun with `--diff <v001>/pass2/pass_diff.json` to re-verify only changed objects. `unsure`/low-confidence verdicts are left unchanged.
