# SEM defect pipeline on Modal — BUILD SPEC (lead-authored, authoritative)

Repo: /home/ubuntu/repos/sem-defect-modal (git init locally; remote yx-lim/sem-defect-modal will be added later).
Python 3.11. Package name `sem`. Copy this file to docs/SPEC.md.

## 0. Data facts
- Local data: /home/ubuntu/data/sem/Batch_{1,2,3}/img_<gid>_<DET>.tif, DET in {BSE, ETD, Inlens}. ~93 files, 8-bit, ~7000x2000.
- group_id = <gid>; batch = folder name; image_id = f"{batch}/{gid}/{DET}".
- Unit of evaluation = group (all detectors, tiles, crops of a group stay in the same split). Tiles are pseudo-replicates.
- Primary detector for all modelling = BSE. ETD/Inlens only used as optional context/covariates.
- Pixel size: try to parse from TIFF tags (Zeiss CZ_SEM tag 34118 / FEI tag 34682 / ImageDescription). If found, store px_size_nm; else null and report KPIs in px units. Never invent.
- Original files are never modified. Derived data lives on the Modal Volume.

## 1. Taxonomy (sem/contract.py)
CLASSES = ["background","crack_intra","crack_inter","void","agglomerate","curtaining","edge_bloom","other_anomaly"]  # index = mask value
IGNORE = 255
ARTIFACT_CLASSES = {"curtaining","edge_bloom"}
Label agent may additionally answer "normal" (-> background) or "uncertain" (-> never used for training, always sent to human).

## 2. Contracts (pydantic v2 models in sem/contract.py; JSON on disk)
Proposal: proposal_id (sha1 of image_id+bbox+source, 12 hex), image_id, group_id, batch, detector, bbox [x0,y0,x1,y1] full-res ints,
  mask_rle (COCO-style RLE within bbox, optional), source in {"tophat_crack","dark_void","fft_curtain","edge_band","anomaly_peak","microsam","random"},
  score float, run_id.
Label: label_id, proposal_id, label (CLASSES + "normal"|"uncertain"), confidence [0,1], rationale str, is_artifact bool,
  source in {"vlm","human"}, model_id (vlm model or null), prompt_version, reviewer_id (null for vlm), vlm_suggestion (Label-lite dict or null),
  status in {"accepted_vlm_only","accepted_human","rejected","pending_review"}, created_at ISO, label_version.
AuditRecord: run_id (uuid4), function, started_at, ended_at, wall_s, git_commit, git_dirty bool, config (dict), config_sha256,
  inputs [{path, sha256}], outputs [{path, sha256}], model {name, weights_sha256 or hub id+revision}, label_version|null,
  modal {function_call_id (modal.current_function_call_id()), gpu, est_cost_usd}, metrics dict, verdict dict|null, limitations [str],
  prev_hash, record_hash = sha256(canonical json (sort_keys, separators=(",",":")) of record without record_hash).
KPIVerdict: image_ids, reference {spec, image_ids, n_groups}, per_image [{image_id, kpis{...}}], per_kpi [{kpi, ref_median, ref_mad,
  test_mean, robust_z, boot_ci95 [lo,hi], perm_p, holm_p}], verdict in {"within_bounds","investigate","outside_bounds","abstain"},
  reasons [str], engineering_thresholds null, limitations [str].
Wording rule (enforced in verdict text + UI): never "accept/reject/defective batch/supplier fault". Only map to accept/reject if
  config.engineering_thresholds is non-null (it is null now).

## 3. Modal layout (modal_app.py)
app = modal.App("sem-defects"); vol = modal.Volume.from_name("sem-data", create_if_missing=True) mounted at /vol.
Volume paths: /vol/raw/<batch>/<file>, /vol/meta/inventory.json, /vol/features/<model>/<image_id>.npy, /vol/anomaly/<run_id>/,
  /vol/proposals/<run_id>.jsonl, /vol/crops/<proposal_id>_{crop,context}.png, /vol/labels/<label_version>/labels.jsonl,
  /vol/models/<model_version>/{weights.pt,metrics.json,config.json}, /vol/pred/<model_version>/<image_id>.png, /vol/kpi/<run_id>.json,
  /vol/audit/chain.jsonl, /vol/audit/freeze.json.
Images: base image debian_slim("3.11").uv_pip_install(torch, torchvision, tifffile, scikit-image, opencv-python-headless, numpy, scipy,
  pydantic, faiss-cpu, anthropic, gradio, fastapi, segmentation-models-pytorch, pillow, pycocotools).add_local_python_source("sem").
  Pin versions in requirements.lock / pyproject; record lockfile sha in audit config.
Secret: modal.Secret.from_name("anthropic") (key ANTHROPIC_API_KEY) on label_agent only.
GPU: "L4" for features/anomaly/train/detect; CPU for everything else. Every function: explicit timeout, retries=1 where idempotent,
  max_containers<=8 on mapped functions. Every function that writes calls vol.commit(); every function that reads calls vol.reload() first.
  Every function ends by appending an AuditRecord (sem/audit.py). Cost = wall_h * {L4:0.80, A10:1.10, L40S:1.95, cpu:0.05}.
Functions:
  upload (local entrypoint): vol.batch_upload raw TIFFs + inventory (sha256 per file, group, batch, detector, shape, px_size_nm).
  Embedder (@app.cls gpu L4, @modal.enter loads DINOv2 ViT-B/14 once from torch.hub facebookresearch/dinov2 dinov2_vitb14, pin revision/sha):
    .embed(image_id) -> per-tile patch features. Tiling: 518x518, stride 448, reflect-pad edges, gray->3ch, ImageNet norm, fp16.
    Store features + tile coords. 14px patch => 37x37 grid per tile.
  anomaly_scan(reference_spec, run_id) (L4): PatchCore. For each BSE group g: memory bank = patch features of reference images excluding g
    (leave-one-group-out), greedy coreset 10%, faiss L2, score = 1-NN distance. Stitch to full-res heatmap (max over overlaps, bilinear).
    Calibrate: threshold = 99th percentile of LOGO patch scores of reference images. Outputs heatmap .npy (downsampled 4x) + png overlay,
    per-image stats {p99, max, frac_above_thr}. Default reference_spec = "all" (every BSE group, LOGO); also allow "batch:Batch_1".
  propose(run_id) (CPU, .map over images): classical generators on BSE full-res + anomaly peaks:
    tophat_crack: gaussian sigma=1, black_tophat disk r=7, thr = 99.5th pct, components with skeleton length>=30px and major/minor>=4.
    dark_void: multi-Otsu 3 classes, lowest class components, area>= 400px AND area > 99th pct of that image's dark-component areas
      (normal porosity is expected; only unusually large dark regions are candidates).
    fft_curtain: per 512 tile, ratio of spectral energy in |fy|<2 band (vertical stripes) excluding DC; tiles > 95th pct over corpus.
    edge_band: brightness of outer 64px bands vs interior; flag bands > interior mean + 3*MAD-sigma.
    anomaly_peak: top 20 local maxima per image of heatmap above threshold, box 112x112.
    random: 15 random 224x224 boxes per image (seeded) — provides "normal" examples; required, not optional.
    Dedupe IoU>0.5 keeping priority order anomaly_peak>tophat_crack>dark_void>edge_band>fft_curtain>random. Cap 60 per image.
    microsam: OPTIONAL stretch in a separate micromamba image (conda-forge micro_sam, model vit_b_em_organelles or vit_b_lm whichever EM
      generalist exists) — only after everything else works.
  label_agent(proposal_ids, label_version) (CPU, secret): for each proposal render crop (bbox padded to >=128px, resized to 384 long side)
    and context (4x bbox area, resized to 768 long side, red rectangle around bbox). Call Claude with PROMPT below, temperature 0,
    parse strict JSON (retry once on parse failure, else label="uncertain").
    Ordering: proposals stratified-shuffled (seed 0) across batch x source. The FIRST 5 labelled get status "accepted_vlm_only"
      (config vlm_only_first_n=5). ALL others get status "pending_review" with vlm_suggestion filled; they are only training-eligible
      after human review. "uncertain" is always pending_review.
    Model id: config vlm_model; verify the id exists via the Anthropic models list API before first run; prompt_version="v1".
  ui (@modal.asgi_app, gradio mounted on FastAPI): tabs Review (queue of pending_review, shows crop+context+VLM suggestion, buttons per
    class + normal + reject + skip, reviewer_id text box; writes Label status accepted_human, commit), Results (image picker -> heatmap +
    predicted mask overlay + KPIs), Audit (chain table + verify_chain button).
  train_supervised(label_version, arch in {"dinov2_head","micronet_unet"}, seed) (L4):
    Build masks: accepted labels only. Region mask = proposal mask_rle if present else: for crack labels, tophat threshold within bbox;
      for void, dark-class pixels within bbox; for curtaining/edge_bloom/agglomerate/other_anomaly/normal, full bbox. Pixels outside
      labelled regions = IGNORE. normal -> background(0).
    Split: GroupKFold-style by group_id, stratified by batch: ~20% groups val (seed). Report counts per class per split.
    dinov2_head: frozen dinov2_vitb14; head = concat(patch tokens upsampled x4) + shallow conv stem on raw image at 1/2 res -> 2 conv
      blocks -> 1x1 to 8 classes -> bilinear to full res. micronet_unet: segmentation_models_pytorch Unet, encoder resnet50 with
      NASA pretrained-microscopy-models "micronet" weights (verify the loader; if weights unavailable, STOP and report, do not substitute).
      Encoder frozen first.
    Loss: class-weighted CE (ignore_index=255) + soft Dice over present classes. AdamW lr 1e-3 head / 1e-4, 40 epochs, aug: flips, 90°
      rot, brightness/contrast jitter. Metrics on val: per-class IoU, Dice, crack object-level F1 (IoU>=0.3 matching), confusion matrix.
    Save weights sha256, metrics.json, config.json; audit.
  detect(image_id, model_version) (L4, .map): sliding-window inference 518/448, softmax avg in overlaps; writes class-index PNG + overlay.
  kpi_verdict(image_ids, reference_spec, model_version, anomaly_run_id) (CPU): see §4.
  verdict_api (@modal.fastapi_endpoint(method="POST")): body {image_ids, reference_spec, model_version, anomaly_run_id} -> KPIVerdict JSON.
  freeze(model_version, anomaly_run_id) : writes /vol/audit/freeze.json {config_sha256, weights_sha256, thresholds, git_commit} + audit.
  evaluate_holdout(image_ids): refuses unless current config/weights hashes == freeze.json; results written to new path, never overwrite.
  verify_chain(): recompute every record_hash and prev_hash link; return first broken index or OK.

## 4. KPIs + verdict (sem/kpi.py, sem/verdict.py)
Per BSE image from predicted mask (+ heatmap + raw):
  area_frac_<cls> for crack_intra, crack_inter, void, agglomerate, other_anomaly; crack_length_density = skeleton px of crack classes /
  solid-phase px (solid = multi-Otsu classes 2-3); dark_phase_frac (multi-Otsu lowest class); anomaly_prevalence = frac heatmap > thr;
  curtaining_score (mean FFT ratio); edge_bloom_score; focus = var(Laplacian); brightness mean, std.
  Units: px unless px_size_nm known.
Aggregate per group (mean over that group's BSE image; one image per group so == image).
Reference = groups in reference_spec excluding test groups. Per KPI: robust_z of test group mean vs ref median/(1.4826*MAD) (MAD floor
  1e-9 -> if MAD==0 use std; if both 0 mark kpi "degenerate"); bootstrap (2000, seed 0) CI of test-mean minus ref-mean resampling GROUPS;
  permutation test (5000) on group labels; Holm correction across KPIs (artifact/covariate KPIs focus, brightness, curtaining, edge_bloom
  reported but EXCLUDED from verdict family; they are confounder flags).
Rules: abstain if n_ref_groups < 5 or n_test_groups < 1. outside_bounds if any defect KPI has |robust_z|>=3 AND holm_p<0.05 (when
  n_test>=3) — with n_test<3 use |z|>=3 alone. investigate if any defect KPI |z|>=2 or any artifact KPI |z|>=3 (reason: "acquisition
  difference may confound"). else within_bounds. reasons list every triggering KPI. limitations always include: no engineering
  thresholds; labels partly VLM-only (n); pixel units if uncalibrated; tiles are pseudo-replicates.

## 5. Label-agent PROMPT v1 (use verbatim; system + user text; images: crop then context)
SYSTEM:
You are assisting a materials scientist labelling FIB-SEM cross-section images of lithium-ion battery electrodes (backscattered-electron
detector, grayscale). Bright regions are usually active-material particles, dark regions are pores/binder. Normal electrodes contain
many pores between particles; ordinary porosity is NOT a defect. You classify one proposed region. Be conservative: if you cannot tell,
answer "uncertain". Respond with a single JSON object and nothing else.
USER:
Image 1 is a close-up of the proposed region. Image 2 shows wider context with the region outlined in red.
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
Return: {"label": <one of the above>, "confidence": <0-1>, "is_artifact": <true if curtaining/edge_bloom or other imaging artefact>,
"rationale": <one sentence describing the visual evidence>}
