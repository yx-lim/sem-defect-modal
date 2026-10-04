"""Batch 3 SEM 84-feature labelling: deterministic coordinator.

One shared-VM worker per source image, at most 5 active, at most 2 retries per image.
The coordinator (this script) is the sole ledger writer; it mirrors the ledger to Drive after every update.
STAGE = 'smoke' runs the 3 images of one field stem (BSE+ETD+Inlens) and stops; 'full' runs the rest.
"""
import asyncio
import json
import pathlib
import subprocess
import sys

BASE = pathlib.Path('/home/ubuntu/sem_b3')
ROOT = BASE / 'pipeline'
sys.path.insert(0, str(ROOT))
from sem84.ledger import Ledger  # noqa: E402

STAGE = 'full'
BATCH = 'Batch_3'
SMOKE_STEM = '9luzk4jm'
MAX_ACTIVE = 5
MAX_RETRIES = 2
THREADS = 3
BUDGET_TOTAL, BUDGET_PER_WORKER = 1000, 25
RCLONE = '/home/ubuntu/.local/bin/rclone'
REMOTE = 'gdrive:'           # rclone remote rooted at the user's Drive folder 1h5kEoojAA63WU5isKZKpN0S_OcWe9oz5
LEDGER = BASE / 'ledger' / 'ledger.json'
OUTROOT = BASE / 'outputs'

RESULT_SCHEMA = {
    'type': 'object',
    'properties': {
        'image_id': {'type': 'string'},
        'status': {'type': 'string', 'enum': ['completed_draft', 'failed_retryable', 'blocked']},
        'artifact_uri': {'type': 'string'},
        'package_sha256': {'type': 'string'},
        'final_sha256sums_sha256': {'type': 'string'},
        'feature_state_counts_json': {'type': 'string'},
        'review_summary': {'type': 'string'},
        'systemic_issues': {'type': 'string'},
        'error': {'type': 'string'},
    },
    'required': ['image_id', 'status', 'artifact_uri', 'package_sha256', 'feature_state_counts_json', 'review_summary', 'systemic_issues'],
}

WORKER_PROMPT = """You are an image worker for a conservative SEM electrode-labelling campaign (Batch 3). Process exactly ONE image: {image_id}.
You run on a SHARED machine with up to 4 other workers. Write ONLY inside {out} and {pkg}* and the Drive path {remote_dir}.
Do not touch other images' directories, the ledger ({ledger}), the pipeline source, git, or other shells/processes. Do not open PRs. Do not install packages.

ENVIRONMENT (already set up and verified; do not re-download anything)
- Pipeline: {root} (sem84 CLI). Python: {base}/.venv/bin/python. Models: {root}/models (hash-checked by the CLI).
- Source TIFF: {src} (manifest sha256 {sha256}; the CLI verifies hash and native size).
{reference_block}- Drive: rclone binary {rclone}; remote "{remote}" is rooted at the user's output Drive folder. Ignore rclone's "shared client_id is being retired" NOTICE.

SCIENTIFIC RULES (non-negotiable)
- Native pixel units only; the ~25 nm/px TIFF tag is unverified. Fresh destructively prepared section: no cycling-damage/failure inference.
- BSE is the canonical composition-contrast view; ETD/Inlens/SE are surface/relief views. Never copy masks across detectors unless the pipeline's registration AND local validation pass.
- Do not label Si, SiOx, graphite, binder, carbon black, copper, contamination or oxidation state as confirmed. Dark != pore. Touching != electrical contact. 2D cannot prove 3D connectivity/tortuosity/enclosure.
- fissure_candidate is a 2D morphology proposal only. A dark region inside a particle mask may be a boundary gap the mask leaked over, an edge shadow, a cavity or a prep fracture: keep fissure_candidate only for thin slits clearly surrounded by one particle at native resolution; otherwise reclassify (void_like_region / interfacial_gap / unresolved_dark).
- Unknown/unreviewed pixels stay ignore (255). Model scores are not accuracy. Draft labelling, not expert ground truth.

PROCEDURE (cd {root}; CLI="{base}/.venv/bin/python -m sem84.cli"; OUT={out}; COMMON="--image-id {image_id} --src {src} --out $OUT --threads {threads} {ref_args}")
If a stage directory already exists and `$CLI validate $OUT/<stage>` prints ok=true, keep it and continue from the next step (resume). If a pass1/pass2 stage is invalid, rerun it with --force. Never modify $OUT/final once it validates.
1. Pass 1: $CLI run $COMMON --stage pass1
2. Pass-1 review: view EVERY $OUT/pass1/review_tiles/*_raw.jpg and matching *_annotated.jpg (10 tiles) and the QA crops in $OUT/pass1/qa/, using your image viewer.
   At native pixels look for: merged or false-positive particle masks, missed particles (give seeds), dark regions mis-classed (void vs fissure vs interfacial gap vs unresolved), artifacts (curtaining = vertical streaks, charging, smearing, scratches, redeposition), seams, exterior/embedding, and every not_assessable decision. Mark curtaining/charging regions explicitly in artifact_regions.
   Write $OUT/review_pass2.json (outside the stage dirs) with keys: reviewer ("worker-{image_id}"), notes, inspected_regions [{{bbox_xyxy, what}}] (only regions you actually looked at),
   remove_instances [{{id, reason}}], relabel_instances [{{id, class, reason}}] (classes: particle_contrast_A, particle_contrast_B [BSE only], unknown_inclusion, particle_unclassified_contrast),
   reclassify_dark [{{id, class, reason}}] (void_like_region, fissure_candidate, interfacial_gap, unresolved_dark), add_seeds [{{xy:[x,y], reason}}],
   artifact_regions / ignore_regions / verified_exterior [{{bbox_xyxy, reason}}], verified_references {{}} (only with explicit evidence in the image),
   feature_overrides {{FID: {{state, rationale, confidence}}}}, confirm_not_assessable [FIDs]. IDs come from $OUT/pass1/annotation.json.
   Pipeline behaviour: reviewer removals and ignore_regions/verified_exterior boxes suppress automatic re-proposal in later passes; seeds are rejected (with a warning in the stage log) if the SAM mask exceeds 15% of the 1024px window or has solidity < 0.7, so check the log and re-seed closer to the particle centre if needed; brightness fall-off is flattened before dark thresholds (see qc/annotation intensity_model.illumination_correction).
   If review removes every fissure candidate, D01/D05 fall back to unreviewed: set an explicit feature_overrides entry (not_observed_in_valid_view only with >=50% inspected coverage, else not_assessable) with a rationale.
   Keep temporary review helpers only in $OUT/.scratch and delete it before packaging.
   Prefer removing/ignoring over guessing. Never claim not_observed_in_valid_view unless you inspected >=50% of the valid area. Chemistry features stay requires_external_evidence.
3. Pass 2: $CLI run $COMMON --stage pass2 --review $OUT/review_pass2.json   (independently re-proposes small structures; writes pass_diff.json)
4. Pass-2 review: inspect $OUT/pass2/review_tiles, qa/, pass_diff.json, the dashboard PNG and every not_assessable/unreviewed feature in feature_presence.csv. Write $OUT/review_final.json (same format; only additional corrections, reviewer "worker-{image_id}").
5. Final: $CLI run $COMMON --stage final --review $OUT/review_final.json ; then $CLI validate $OUT/final must print ok=true.
6. Package: $CLI package --version-dir $OUT --out {pkg} --arcname {batch}/{image_id}/v001 ; write {pkg}.sha256 (sha256sum format) if the CLI did not.
7. Durable upload (labelled layout {remote_dir}/{{pass1,pass2,final,review_*.json,checkpoint.json}} + {image_id}_v001.tar.gz + .sha256):
   {rclone} copy --immutable --exclude ".scratch/**" $OUT {remote}{remote_dir}
   {rclone} copy --immutable {pkg} {remote}{remote_dir} ; {rclone} copy --immutable {pkg}.sha256 {remote}{remote_dir}
   The tarball and .sha256 MUST sit inside {remote_dir}/ (next to pass1/pass2/final), not one level up.
   Verify: {rclone} check --one-way --exclude ".scratch/**" $OUT {remote}{remote_dir} must report 0 differences, and `{rclone} lsl {remote}{remote_dir}/{image_id}_v001.tar.gz` size must equal the local size.
   --immutable refuses to overwrite: if it reports an existing file with different content, stop and report blocked.
8. Report structured output: image_id; status completed_draft only if final validated AND upload verified, failed_retryable for transient/infra errors, blocked for scientific/data problems;
   artifact_uri ("{remote}{remote_dir}"); package_sha256; final_sha256sums_sha256 (sha256 of $OUT/final/SHA256SUMS);
   feature_state_counts_json (state counts from $OUT/final/result.json, as JSON); review_summary (what you changed in each pass and why, <=1200 chars);
   systemic_issues (pipeline problems likely to affect other images, or "none"); error ("" if none).
"""


def mirror_ledger():
    try:
        subprocess.run([RCLONE, 'copy', str(LEDGER.parent), f'{REMOTE}{BATCH}/_ledger', '-q'], timeout=300, check=False)
    except Exception as e:  # mirror failure must not stop the run; the local ledger is authoritative
        log(f'ledger mirror failed: {e}')


def reference_for(job, jobs):
    if job['detector'] == 'BSE':
        return None
    for j in jobs.values():
        if j['field_stem'] == job['field_stem'] and j['detector'] == 'BSE':
            return j
    return None


def build_prompt(job, jobs, manifest):
    rec = manifest[job['image_id']]
    ref = reference_for(job, jobs)
    if ref:
        rr = manifest[ref['image_id']]
        rb = f"- Reference (canonical BSE of the same field stem, for registration only): {BASE}/src/{rr['filename']} (sha256 {rr['sha256']}).\n"
        ra = f"--reference-src {BASE}/src/{rr['filename']} --reference-id {rr['image_id']}"
    else:
        rb = '- No reference view (this image is the canonical BSE frame, or the stem has no BSE).\n'
        ra = ''
    iid = job['image_id']
    return WORKER_PROMPT.format(image_id=iid, out=OUTROOT / BATCH / iid / 'v001', pkg=OUTROOT / 'packages' / f'{iid}_v001.tar.gz',
                                remote=REMOTE, remote_dir=f'{BATCH}/{iid}/v001', ledger=LEDGER, root=ROOT, base=BASE, rclone=RCLONE,
                                src=BASE / 'src' / rec['filename'], sha256=rec['sha256'], reference_block=rb,
                                threads=THREADS, ref_args=ra, batch=BATCH)


async def main():
    manifest = {f['image_id']: f for f in json.loads((ROOT / 'manifest.json').read_text())['files']}
    L = Ledger(LEDGER)
    if not LEDGER.exists():
        L.init(BATCH, BUDGET_TOTAL, BUDGET_PER_WORKER)
    led = L.load()
    jobs = led['jobs']
    order = sorted(jobs, key=lambda k: (jobs[k]['field_stem'], jobs[k]['detector']))
    smoke = [k for k in order if jobs[k]['field_stem'] == SMOKE_STEM][:3]
    targets = smoke if STAGE == 'smoke' else [k for k in order if k not in smoke] + smoke
    targets = [k for k in targets if jobs[k]['status'] not in ('completed_draft', 'qc_pending', 'blocked')]
    await register_workflow({
        'name': f'sem84-batch3-{STAGE}',
        'description': f'Batch 3 SEM 84-feature two-pass labelling ({STAGE}): one worker per image, max {MAX_ACTIVE} active, {MAX_RETRIES} retries',
        'product': 'SEM electrode microstructure labelling (sem84 pipeline)',
        'soft_time_limit_minutes': 45,
        'phases': [{'title': 'image', 'detail': 'pass1, review, pass2, review, final, validate, package, Drive upload',
                    'labels': targets}],
    })
    mirror_ledger()
    sem = asyncio.Semaphore(MAX_ACTIVE)
    (OUTROOT / 'packages').mkdir(parents=True, exist_ok=True)

    async def one(image_id):
        job = jobs[image_id]
        prompt = build_prompt(job, jobs, manifest)
        last = None
        for attempt in range(MAX_RETRIES + 1):
            async with sem:
                L.update(image_id, 'running', worker_id=f'shared-{image_id}-a{attempt + 1}')
                mirror_ledger()
                log(f'{image_id}: attempt {attempt + 1} started')
                try:
                    p = prompt if attempt == 0 else prompt + f'\nRETRY {attempt}: previous attempt failed with: {last}\nResume from validated stages as described above.\n'
                    # vm_mode="shared", reason 2: the Drive write token and the verified sources/models exist only on this machine.
                    res = await agent(p, phase='image', schema=RESULT_SCHEMA, label=image_id, vm_mode='shared')
                except WorkflowAgentError as e:  # noqa: F821 (provided by runtime)
                    last = f'agent error: {e}'
                    L.update(image_id, 'failed_retryable', error=last)
                    mirror_ledger()
                    continue
            st = res.get('status')
            outputs = {k: res.get(k) for k in ('artifact_uri', 'package_sha256', 'final_sha256sums_sha256',
                                               'feature_state_counts_json', 'review_summary', 'systemic_issues')}
            if st == 'completed_draft' and res.get('artifact_uri') and res.get('package_sha256'):
                L.update(image_id, 'qc_pending', outputs=outputs)
                mirror_ledger()
                log(f'{image_id}: worker done -> qc_pending ({res.get("artifact_uri")})')
                return res
            last = res.get('error') or f'status {st}'
            L.update(image_id, 'blocked' if st == 'blocked' else 'failed_retryable', outputs=outputs, error=last)
            mirror_ledger()
            if st == 'blocked':
                return res
        L.update(image_id, 'blocked', error=f'retry limit exceeded: {last}')
        mirror_ledger()
        return None

    results = await asyncio.gather(*(one(k) for k in targets))
    done = [r for r in results if r and r.get('status') == 'completed_draft']
    log(f'{STAGE}: {len(done)}/{len(targets)} worker drafts; ledger summary {json.dumps(L.load()["summary"])}')
    for r in results:
        if r and r.get('systemic_issues') not in (None, '', 'none'):
            log(f"systemic: {r['image_id']}: {r['systemic_issues']}")


asyncio.run(main())
