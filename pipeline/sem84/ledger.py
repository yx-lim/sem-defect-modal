"""Single-writer central ledger (coordinator only). Atomic JSON writes + append-only event log."""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import pathlib
import sys

from .catalogue import ROOT

STATES = ['queued', 'running', 'pass1_saved', 'pass2_saved', 'qc_pending', 'completed_draft', 'failed_retryable', 'blocked']
MAX_RETRIES = 2


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')


def split_for(stem, seed='sem84-b3'):
    h = int(hashlib.sha256(f'{seed}:{stem}'.encode()).hexdigest(), 16) % 100
    return 'train' if h < 70 else ('validation' if h < 85 else 'test')


class Ledger:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.events = self.path.with_name('ledger_events.jsonl')
        self.lock = self.path.with_name('.ledger.lock')

    def _locked(self):
        fh = open(self.lock, 'w')
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('ledger is locked by another writer (coordinator is the sole writer)')
        return fh

    def load(self):
        return json.loads(self.path.read_text())

    def _write(self, led, event):
        led['updated_at'] = now()
        led['summary'] = summarise(led)
        tmp = self.path.with_suffix('.tmp')
        tmp.write_text(json.dumps(led, indent=1))
        tmp.replace(self.path)
        with open(self.events, 'a') as f:
            f.write(json.dumps(dict(event, at=led['updated_at'])) + '\n')

    def init(self, batch, budget_total=None, budget_per_worker=None):
        if self.path.exists():
            raise SystemExit(f'{self.path} exists; refusing to overwrite')
        m = json.loads((ROOT / 'manifest.json').read_text())
        jobs = {}
        for f in sorted((f for f in m['files'] if f['batch'] == batch), key=lambda f: (f['field_stem'], f['detector'])):
            jobs[f['image_id']] = {'image_id': f['image_id'], 'filename': f['filename'], 'field_stem': f['field_stem'],
                                   'detector': f['detector'], 'source_sha256': f['sha256'], 'split': split_for(f['field_stem']),
                                   'status': 'queued', 'attempts': 0, 'worker_ids': [], 'acu': 0.0, 'history': [],
                                   'outputs': {}, 'last_error': None}
        led = {'schema_version': 'sem84.ledger.v1', 'batch': batch, 'created_at': now(), 'version': 'v001',
               'budget': {'total_acu': budget_total, 'per_worker_acu': budget_per_worker, 'consumed_acu': 0.0},
               'max_active_workers': 5, 'max_retries': MAX_RETRIES, 'jobs': jobs,
               'split_policy': 'all detector views of a field stem share one split (sha256(seed:stem) % 100: <70 train, <85 validation, else test)'}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            self._write(led, {'event': 'init', 'batch': batch, 'n_jobs': len(jobs)})
        return led

    def update(self, image_id, status, **kw):
        if status not in STATES:
            raise SystemExit(f'bad status {status}')
        with self._locked():
            led = self.load()
            j = led['jobs'][image_id]
            if j['status'] == 'completed_draft' and status != 'completed_draft':
                raise SystemExit(f'{image_id} is completed_draft; immutable')
            if status == 'running':
                j['attempts'] += 1
                if j['attempts'] > MAX_RETRIES + 1:
                    status = 'blocked'
                    kw['error'] = 'retry limit exceeded'
            if kw.get('worker_id') and kw['worker_id'] not in j['worker_ids']:
                j['worker_ids'].append(kw['worker_id'])
            if kw.get('acu') is not None:
                led['budget']['consumed_acu'] = round(led['budget']['consumed_acu'] - j['acu'] + float(kw['acu']), 3)
                j['acu'] = float(kw['acu'])
            if kw.get('outputs'):
                j['outputs'].update(kw['outputs'])
            if kw.get('error'):
                j['last_error'] = kw['error']
            j['history'].append({'at': now(), 'from': j['status'], 'to': status, **{k: v for k, v in kw.items() if k != 'outputs'}})
            j['status'] = status
            self._write(led, {'event': 'update', 'image_id': image_id, 'status': status, **{k: v for k, v in kw.items() if k != 'outputs'}})
        return j


def summarise(led):
    c = {s: 0 for s in STATES}
    for j in led['jobs'].values():
        c[j['status']] += 1
    rem = [k for k, j in led['jobs'].items() if j['status'] not in ('completed_draft', 'blocked')]
    return {'counts': c, 'remaining': len(rem), 'consumed_acu': led['budget']['consumed_acu']}


def main(argv=None):
    ap = argparse.ArgumentParser(prog='sem84.ledger')
    ap.add_argument('--ledger', required=True)
    sp = ap.add_subparsers(dest='cmd', required=True)
    i = sp.add_parser('init')
    i.add_argument('--batch', default='Batch_3')
    i.add_argument('--budget-total', type=float)
    i.add_argument('--budget-per-worker', type=float)
    u = sp.add_parser('update')
    u.add_argument('image_id')
    u.add_argument('status', choices=STATES)
    u.add_argument('--worker-id')
    u.add_argument('--acu', type=float)
    u.add_argument('--error')
    u.add_argument('--outputs-json')
    sp.add_parser('summary')
    a = ap.parse_args(argv)
    L = Ledger(a.ledger)
    if a.cmd == 'init':
        print(json.dumps(L.init(a.batch, a.budget_total, a.budget_per_worker)['summary']))
    elif a.cmd == 'update':
        j = L.update(a.image_id, a.status, worker_id=a.worker_id, acu=a.acu, error=a.error,
                     outputs=json.loads(a.outputs_json) if a.outputs_json else None)
        print(json.dumps({'image_id': a.image_id, 'status': j['status'], 'attempts': j['attempts']}))
    else:
        print(json.dumps(L.load()['summary'], indent=1))


if __name__ == '__main__':
    sys.exit(main())
