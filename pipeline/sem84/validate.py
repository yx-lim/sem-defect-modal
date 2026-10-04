"""Schema/consistency validator for one stage directory (pass1, pass2 or final)."""
import csv
import hashlib
import json
import pathlib

import numpy as np
from PIL import Image

from .analysis import CLASS_MAP
from .catalogue import CHEM_LIMITED, EXTERNAL, STATES, load_catalogue

Image.MAX_IMAGE_PIXELS = None
LAYERS = ['A_particles', 'B_pores_damage', 'C_contacts_interfaces', 'D_coating_heterogeneity', 'E_chemistry_conditional_si', 'F_prep_acquisition_qc']
REQUIRED = ['annotation.json', 'feature_presence.csv', 'measurements.csv', 'masks/instance_mask.png',
            'masks/semantic_mask.png', 'masks/qc_flags.png', 'masks/class_map.json', 'vectors_coco.json',
            'overview.png', 'dashboard.png', 'charts.png', 'qa/index.json', 'layers.html', 'source_preview.jpg'] + \
           [f'overlays/{n}.png' for n in LAYERS]
PIPELINE_CLASSES = {'particle_contrast_A', 'particle_contrast_B', 'unknown_inclusion', 'particle_unclassified_contrast'}  # pipeline's own neutral classes ('inclusion' contains 'sio')
FORBIDDEN_CLASS_WORDS = ('silicon', 'graphite', 'binder', 'carbon_black', 'sio', 'copper', 'background', 'healthy')


def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def validate_stage(d, check_sums=True, final_files=True):
    d = pathlib.Path(d)
    err, warn = [], []
    req = REQUIRED + (['qc.json', 'result.json', 'SHA256SUMS'] if final_files else [])
    if d.name.startswith('final') or '.final.' in d.name:
        req += [f'layers/{n}.png' for n in LAYERS]
    for r in req:
        if not (d / r).exists():
            err.append(f'missing {r}')
    if err:
        return {'ok': False, 'errors': err, 'warnings': warn}
    ann = json.loads((d / 'annotation.json').read_text())
    cat = [f['id'] for f in load_catalogue()]
    ids = [f['id'] for f in ann['features']]
    if ids != cat:
        err.append('annotation features != 84 catalogue IDs in order')
    for f in ann['features']:
        if f['state'] not in STATES:
            err.append(f"{f['id']}: invalid state {f['state']}")
        if f['id'] in CHEM_LIMITED and f['state'] not in ('requires_external_evidence', 'candidate_inference'):
            err.append(f"{f['id']}: chemistry/3D-limited feature in state {f['state']}")
        if f['state'] == 'requires_external_evidence' and (f['value'] is not None or not f.get('evidence_needed')):
            err.append(f"{f['id']}: requires_external_evidence must have null value and evidence_needed")
        if f['id'] in EXTERNAL and f['state'] == 'requires_external_evidence' and f.get('evidence_needed') != EXTERNAL[f['id']]:
            warn.append(f"{f['id']}: evidence text differs from catalogue rule")
        if f['state'] == 'not_observed_in_valid_view' and 'reviewer' not in (f.get('review_status') or '') and 'overridden' not in (f.get('review_status') or ''):
            err.append(f"{f['id']}: absence claim without reviewer")
        if not f.get('rationale'):
            err.append(f"{f['id']}: missing rationale")
    with open(d / 'feature_presence.csv') as fh:
        rows = list(csv.DictReader(fh))
    if [r['feature_id'] for r in rows] != cat:
        err.append('feature_presence.csv must list exactly the 84 IDs in catalogue order')
    with open(d / 'measurements.csv') as fh:
        for k, r in enumerate(csv.DictReader(fh)):
            u = (r.get('unit') or '').lower()
            if any(x in u for x in ('nm', 'um', 'µm', 'mm')):
                err.append(f'measurements row {k}: physical unit {u} without verified calibration')
                break
    src = ann['source']
    W, H = src['width'], src['height']
    inst = np.asarray(Image.open(d / 'masks/instance_mask.png'))
    sem = np.asarray(Image.open(d / 'masks/semantic_mask.png'))
    for name, a in (('instance', inst), ('semantic', sem)):
        if a.shape != (H, W):
            err.append(f'{name} mask shape {a.shape} != source {(H, W)}')
    vals = set(np.unique(sem).tolist())
    if not vals <= set(CLASS_MAP):
        err.append(f'semantic values outside class map: {sorted(vals - set(CLASS_MAP))}')
    if 0 in vals:
        err.append('semantic value 0 used (no background class allowed)')
    labs = set(np.unique(inst).tolist()) - {0}
    if labs != {i['label'] for i in ann['instances']}:
        err.append('instance mask labels != annotation instance labels')
    cm = json.loads((d / 'masks/class_map.json').read_text())
    if cm.get('ignore_value') != 255:
        err.append('class_map ignore_value must be 255')
    for i in ann['instances']:
        if i['class'] not in PIPELINE_CLASSES and any(w in i['class'].lower() for w in FORBIDDEN_CLASS_WORDS):
            err.append(f"instance {i['id']}: forbidden class {i['class']}")
        if not i.get('polygon_xy'):
            warn.append(f"instance {i['id']}: empty polygon")
    if ann.get('units', {}).get('length') != 'px':
        err.append('annotation units must be px')
    if ann['source'].get('calibration_verified'):
        warn.append('calibration_verified true: check')
    if check_sums and (d / 'SHA256SUMS').exists():
        listed = set()
        for line in (d / 'SHA256SUMS').read_text().splitlines():
            h, p = line.split('  ', 1)
            listed.add(p)
            if not (d / p).exists() or sha256(d / p) != h:
                err.append(f'checksum mismatch {p}')
        actual = {str(p.relative_to(d)) for p in d.rglob('*') if p.is_file() and p.name != 'SHA256SUMS'}
        if actual - listed:
            err.append(f'files not in SHA256SUMS: {sorted(actual - listed)[:5]}')
    return {'ok': not err, 'errors': err, 'warnings': warn[:50], 'n_warnings': len(warn)}
