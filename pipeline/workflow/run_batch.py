"""Run the remaining Batch 3 images with the automatic verifier flow and upload to Drive.

Destination: gdrive:Test Batch/Batch_3/<image_id>/v001 (where the latest v001 outputs were moved).
Ledger: local copy of Training Batch/Batch_3/_ledger, mirrored back after each update.
BSE of each stem runs first (canonical reference for ETD/Inlens/SE). Existing Drive files are never overwritten (--immutable).
"""
import concurrent.futures as cf
import json
import pathlib
import subprocess
import sys
import threading
import time

BASE = pathlib.Path('/home/ubuntu/sem_b3')
sys.path.insert(0, str(BASE / 'pipeline'))
from sem84.ledger import Ledger  # noqa: E402

RCLONE = '/home/ubuntu/.local/bin/rclone'
DEST = 'gdrive:Test Batch/Batch_3'
LEDGER_REMOTE = 'gdrive:Training Batch/Batch_3/_ledger'
OUT = BASE / 'outputs' / 'Batch_3'
SRC = BASE / 'src'
PY = str(BASE / '.venv' / 'bin' / 'python')
MAX_ACTIVE = int(sys.argv[1]) if len(sys.argv) > 1 else 3
led = Ledger(BASE / 'ledger_drive' / 'ledger.json')
lock = threading.Lock()


def sh(*a, **kw):
    return subprocess.run(list(a), capture_output=True, text=True, **kw)


def upd(iid, status, **kw):
    with lock:  # Ledger uses a non-blocking flock, so serialise writers in-process
        led.update(iid, status, worker_id='devin-1578a91e-auto-verifier', **kw)
        sh(RCLONE, 'copy', str(led.path.parent), LEDGER_REMOTE, '--exclude', '.ledger.lock', '-q')


def upload(iid, vd):
    pkg = vd.parent / f'{iid}_v001.tar.gz'
    r = sh(PY, '-m', 'sem84.cli', 'package', '--version-dir', str(vd), '--out', str(pkg), '--arcname', 'v001', cwd=BASE / 'pipeline')
    if r.returncode:
        raise RuntimeError('package: ' + r.stderr[-500:])
    rd = f'{DEST}/{iid}/v001'
    for a in ([str(vd), rd, '--exclude', '.scratch/**'], [str(pkg), rd], [str(pkg) + '.sha256', rd]):
        r = sh(RCLONE, 'copy', '--immutable', *a)
        if r.returncode:
            raise RuntimeError('upload: ' + r.stderr[-500:])
    r = sh(RCLONE, 'check', '--one-way', '--exclude', '.scratch/**', str(vd), rd)
    if r.returncode:
        raise RuntimeError('check: ' + r.stderr[-500:])
    return {'artifact_uri': rd, 'package_sha256': pathlib.Path(str(pkg) + '.sha256').read_text().split()[0]}


def run_one(job, ref):
    iid = job['image_id']
    vd = OUT / iid / 'v001'
    t0 = time.time()
    upd(iid, 'running')
    args = [str(BASE / 'drive_image.sh'), iid, str(SRC / job['filename']), str(vd)]
    if ref:
        args += [str(SRC / ref['filename']), ref['image_id']]
    log = OUT / f'{iid}.log'
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, 'w') as f:
        r = subprocess.run(args, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode or not json.loads((vd / 'validate.json').read_text())['ok']:
        upd(iid, 'failed_retryable', error=f'drive_image exit {r.returncode}; see {log}')
        return iid, 'failed'
    out = upload(iid, vd)
    usd = sum(json.loads((vd / '.verify' / s / 'summary.json').read_text())['usd'] for s in ('pass1', 'pass2'))
    upd(iid, 'qc_pending', outputs=dict(out, api_usd=round(usd, 3), wall_s=round(time.time() - t0)),
        note='automatic tile+crop+policy verifier review; awaiting human QC')
    return iid, 'ok'


def main():
    recs = {r['image_id']: r for r in json.load(open(BASE / 'remaining.json'))}
    jobs = led.load()['jobs']
    todo = [i for i in recs if jobs[i]['status'] in ('queued', 'running', 'failed_retryable')
            and i not in ('Batch_3__img_tuy3zymq_ETD', 'Batch_3__img_ufdvpb81_BSE')]  # those two finished on the old VM
    stems = {}
    for i in todo:
        stems.setdefault(recs[i]['field_stem'], []).append(i)
    bse = {r['field_stem']: r for r in json.load(open(BASE / 'pipeline' / 'manifest.json'))['files'] if r['batch'] == 'Batch_3' and r['detector'] == 'BSE'}

    def stem_chain(stem):
        ids = sorted(stems[stem], key=lambda i: recs[i]['detector'] != 'BSE')
        res = []
        b = bse[stem]
        for i in ids:
            ref = None if recs[i]['detector'] == 'BSE' else b
            try:
                res.append(run_one(recs[i], ref))
            except Exception as e:
                upd(i, 'failed_retryable', error=str(e)[-500:])
                res.append((i, f'failed: {e}'))
        return res

    with cf.ThreadPoolExecutor(MAX_ACTIVE) as ex:
        for r in ex.map(stem_chain, sorted(stems)):
            print(r, flush=True)


if __name__ == '__main__':
    main()
