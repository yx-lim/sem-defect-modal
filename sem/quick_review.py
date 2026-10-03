"""One-click grid review page: plain HTML + JSON API on the same FastAPI app
as the Gradio UI. All thumbnails are small cached JPEGs so the whole queue
renders at once. Backends are injected callables (local and Modal)."""

import json
import re
from pathlib import Path

from .ui import REVIEW_CHOICES, sort_pending, vlm_preselect

THUMB_W = {"crop": 320, "context": 480}
GRID_CHOICES = [c for c in REVIEW_CHOICES if c != "skip"]
_PID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
RISK_TIERS = ["Critical", "High", "Medium", "Low"]


def load_risk(path: Path | None) -> dict:
    """{proposal_id: {tier, failure_mode, why}} from a risk_tiers.json file
    (review priority only, not a verdict); {} if absent."""
    if path is None or not Path(path).exists():
        return {}
    d = json.loads(Path(path).read_text())
    return {k: v for k, v in d.get("tiers", {}).items()
            if v.get("tier") in RISK_TIERS}


def to_label(choice: str) -> str:
    """Grid button -> stored label (same mapping as the Review tab)."""
    if choice == "reject":
        return "rejected"
    if choice == "normal":
        return "background"
    if choice in GRID_CHOICES:
        return choice
    raise ValueError(f"bad choice: {choice!r}")


def grid_items(pending: list[dict], risk: dict | None = None) -> list[dict]:
    """Queue order (uncertain first, conf asc), then within each of the
    uncertain / non-uncertain blocks: risk tier Critical..Low, unranked last."""
    risk = risk or {}

    def key(p):
        unc = (p.get("vlm_suggestion") or {}).get("label") == "uncertain"
        t = (risk.get(p.get("proposal_id")) or {}).get("tier")
        return (0 if unc else 1,
                RISK_TIERS.index(t) if t in RISK_TIERS else len(RISK_TIERS))

    out = []
    for p in sorted(sort_pending(pending), key=key):
        sug = p.get("vlm_suggestion") or {}
        rk = risk.get(p.get("proposal_id")) or {}
        out.append({
            "proposal_id": p.get("proposal_id"),
            "image_id": p.get("image_id"),
            "source": p.get("source"),
            "vlm_label": sug.get("label"),
            "confidence": sug.get("confidence"),
            "is_artifact": sug.get("is_artifact"),
            "rationale": sug.get("rationale"),
            "accept": vlm_preselect(sug),
            "risk_tier": rk.get("tier"),
            "risk_mode": rk.get("failure_mode"),
            "risk_why": rk.get("why"),
        })
    return out


def prewarm_thumbs(crop_dir: Path, pids) -> int:
    from .pipeline import write_preview

    n = 0
    for pid in pids:
        for kind, w in THUMB_W.items():
            src = crop_dir / f"{pid}_{kind}.png"
            if src.exists():
                write_preview(src, w, suffix="_thumb")
                n += 1
    return n


def quick_router(list_pending, crop_dir: Path, submit_review,
                 risk_path: Path | None = None):
    """submit_review(pid, label, reviewer, revise=False). risk_path is
    re-read per request so tier edits apply without a restart."""
    from fastapi import APIRouter, HTTPException
    from fastapi.responses import FileResponse, HTMLResponse
    from pydantic import BaseModel

    from .pipeline import write_preview

    r = APIRouter()

    class Review(BaseModel):
        proposal_id: str
        choice: str
        reviewer_id: str
        revise: bool = False

    class Batch(BaseModel):
        proposal_ids: list[str]
        reviewer_id: str

    def _reviewer(s: str) -> str:
        s = (s or "").strip()
        if not s:
            raise HTTPException(400, "reviewer_id required")
        return s

    def _src(pid: str, kind: str) -> Path:
        if not _PID.match(pid) or kind not in THUMB_W:
            raise HTTPException(400, "bad id")
        src = Path(crop_dir) / f"{pid}_{kind}.png"
        if not src.exists():
            raise HTTPException(404, "no crop")
        return src

    @r.get("/quick", response_class=HTMLResponse)
    def page():
        return PAGE

    @r.get("/quick/api/items")
    def items():
        return {"items": grid_items(list_pending(), load_risk(risk_path)),
                "choices": GRID_CHOICES, "tiers": RISK_TIERS}

    @r.get("/quick/thumb/{pid}/{kind}.jpg")
    def thumb(pid: str, kind: str):
        dst = write_preview(_src(pid, kind), THUMB_W[kind], suffix="_thumb")
        return FileResponse(dst, media_type="image/jpeg",
                            headers={"Cache-Control": "max-age=86400"})

    @r.get("/quick/full/{pid}/{kind}.png")
    def full(pid: str, kind: str):
        return FileResponse(_src(pid, kind), media_type="image/png",
                            headers={"Cache-Control": "max-age=86400"})

    @r.post("/quick/api/review")
    def review(b: Review):
        rev = _reviewer(b.reviewer_id)
        if not _PID.match(b.proposal_id):
            raise HTTPException(400, "bad id")
        try:
            lab = to_label(b.choice)
        except ValueError as e:
            raise HTTPException(400, str(e))
        submit_review(b.proposal_id, lab, rev, revise=b.revise)
        return {"ok": True, "proposal_id": b.proposal_id, "label": lab}

    @r.post("/quick/api/accept")
    def accept(b: Batch):
        rev = _reviewer(b.reviewer_id)
        pend = {it["proposal_id"]: it for it in grid_items(list_pending())}
        done, skipped = {}, []
        for pid in b.proposal_ids:
            it = pend.get(pid)
            if it is None or it["accept"] is None:
                skipped.append(pid)
                continue
            lab = to_label(it["accept"])
            submit_review(pid, lab, rev)
            done[pid] = lab
        return {"accepted": done, "skipped": skipped}

    return r


PAGE = r"""<!doctype html>
<html><head><meta charset="utf-8"><title>SEM quick review</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{--bg:#f6f7f9;--card:#fff;--b:#d9dce1;--t:#1d2330;--m:#677085;--ok:#1a7f37;--warn:#9a6700;--bad:#cf222e;--acc:#0969da}
*{box-sizing:border-box}body{margin:0;font:14px/1.4 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--t)}
header{position:sticky;top:0;z-index:5;background:#fff;border-bottom:1px solid var(--b);padding:10px 16px;display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:16px;margin:0 8px 0 0}
input[type=text]{padding:6px 8px;border:1px solid var(--b);border-radius:6px;width:150px}
.chips{display:flex;flex-wrap:wrap;gap:6px}
.chip{border:1px solid var(--b);background:#fff;border-radius:14px;padding:3px 10px;cursor:pointer;font-size:13px}
.chip.on{background:var(--acc);border-color:var(--acc);color:#fff}
.spacer{flex:1}
button{font:inherit;cursor:pointer;border-radius:6px;border:1px solid var(--b);background:#fff;padding:5px 10px}
button.primary{background:var(--ok);border-color:var(--ok);color:#fff;font-weight:600}
button:disabled{opacity:.45;cursor:not-allowed}
#stats{color:var(--m)}
main{padding:14px 16px;display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:12px}
.card{background:var(--card);border:1px solid var(--b);border-radius:10px;overflow:hidden;display:flex;flex-direction:column;transition:opacity .15s}
.card.done{opacity:.55;border-color:var(--ok)}.card.done.rej{border-color:var(--bad)}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:2px;background:#000}
.imgs img{width:100%;height:170px;object-fit:contain;display:block;cursor:zoom-in;background:#111}
.body{padding:8px 10px;display:flex;flex-direction:column;gap:6px}
.top{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.badge{font-weight:600;padding:1px 7px;border-radius:10px;background:#eef1f5}
.badge.uncertain{background:#fff3cd;color:var(--warn)}
.conf{color:var(--m);font-size:12px}.meta{color:var(--m);font-size:11px;margin-left:auto}
.why{color:var(--m);font-size:12px;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.acts{display:flex;flex-wrap:wrap;gap:4px}
.acts button{padding:3px 7px;font-size:12px}
.acts button.sel{background:var(--acc);border-color:var(--acc);color:#fff}
.acts button.primary{font-size:13px;padding:5px 10px;flex-basis:100%}
.saved{font-size:12px;font-weight:600;color:var(--ok)}
#lb{position:fixed;inset:0;background:rgba(0,0,0,.85);display:none;align-items:center;justify-content:center;gap:12px;z-index:10;padding:20px}
#lb img{max-width:48vw;max-height:90vh;object-fit:contain;background:#111}
#toast{position:fixed;bottom:16px;left:50%;transform:translateX(-50%);background:#1d2330;color:#fff;padding:8px 14px;border-radius:8px;display:none;z-index:20}
.risk{font-size:11px;font-weight:700;padding:1px 7px;border-radius:10px;color:#fff;cursor:help}
.r-Critical{background:#a40e26}.r-High{background:#d1242f}.r-Medium{background:#bf8700}.r-Low{background:#6e7781}
.chip.rk{border-style:dashed}
.empty{grid-column:1/-1;text-align:center;color:var(--m);padding:40px}
</style></head><body>
<header>
  <h1>SEM quick review</h1>
  <label>reviewer_id <input id="rev" type="text" placeholder="your name"></label>
  <div class="chips" id="chips"></div>
  <span class="spacer"></span>
  <span id="stats"></span>
  <button id="clear">Hide done</button>
  <button id="acceptAll" class="primary">Accept all shown</button>
</header>
<main id="grid"><div class="empty">Loading…</div></main>
<div id="lb"></div><div id="toast"></div>
<script>
const BASE = location.pathname.replace(/\/quick\/?$/, '');
const SHORT = {crack_intra:'crack intra',crack_inter:'crack inter',other_anomaly:'other',edge_bloom:'edge bloom'};
let items = [], choices = [], tiers = [], filter = 'all', done = {};
const $ = s => document.querySelector(s);
const esc = s => String(s==null?'':s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const rev = $('#rev'); rev.value = localStorage.getItem('sem_reviewer') || '';
rev.oninput = () => localStorage.setItem('sem_reviewer', rev.value.trim());
function toast(m, ms=1800){const t=$('#toast');t.textContent=m;t.style.display='block';clearTimeout(t._h);t._h=setTimeout(()=>t.style.display='none',ms);}
function reviewer(){const r=rev.value.trim(); if(!r){toast('Enter a reviewer_id first');rev.focus();} return r;}
async function post(url, body){
  const r = await fetch(BASE+url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  if(!r.ok){let d=''; try{d=(await r.json()).detail}catch(e){} throw new Error(d||r.status);}
  return r.json();
}
function shown(){return items.filter(it => filter==='all' ||
  (filter.startsWith('risk:') ? it.risk_tier===filter.slice(5) : it.vlm_label===filter));}
function chips(){
  const c = {}; items.forEach(it => c[it.vlm_label] = (c[it.vlm_label]||0)+1);
  const keys = ['all', ...Object.keys(c).sort((a,b)=>c[b]-c[a])];
  const r = {}; items.forEach(it => {if(it.risk_tier) r[it.risk_tier] = (r[it.risk_tier]||0)+1;});
  $('#chips').innerHTML = keys.map(k => `<span class="chip ${k===filter?'on':''}" data-k="${k}">${k} (${k==='all'?items.length:c[k]})</span>`).join('')
    + tiers.filter(t => r[t]).map(t => `<span class="chip rk ${'risk:'+t===filter?'on':''}" data-k="risk:${t}">risk: ${t} (${r[t]})</span>`).join('');
  document.querySelectorAll('.chip').forEach(e => e.onclick = () => {filter=e.dataset.k; render();});
}
function stats(){
  const n = Object.keys(done).length, s = shown();
  const left = items.filter(it=>!done[it.proposal_id]).length;
  const acc = s.filter(it=>!done[it.proposal_id] && it.accept).length;
  $('#stats').textContent = `${left} pending · ${n} reviewed this session`;
  const b = $('#acceptAll'); b.textContent = `Accept Claude's label for all shown (${acc})`; b.disabled = !acc;
}
function card(it){
  const pid = it.proposal_id, d = done[pid];
  const el = document.createElement('div');
  el.className = 'card' + (d ? ' done' + (d==='rejected'?' rej':'') : ''); el.id = 'c_'+pid;
  const conf = it.confidence==null ? '' : `conf ${(+it.confidence).toFixed(2)}`;
  const acc = it.accept;
  const sel = d ? (d==='background'?'normal':(d==='rejected'?'reject':d)) : null;
  el.innerHTML = `
    <div class="imgs">
      <img loading="eager" decoding="async" src="${BASE}/quick/thumb/${pid}/crop.jpg" data-pid="${pid}" alt="crop">
      <img loading="eager" decoding="async" src="${BASE}/quick/thumb/${pid}/context.jpg" data-pid="${pid}" alt="context">
    </div>
    <div class="body">
      <div class="top"><span class="badge ${it.vlm_label==='uncertain'?'uncertain':''}">${it.vlm_label||'—'}</span>
        ${it.risk_tier ? `<span class="risk r-${it.risk_tier}" title="${esc(it.risk_mode)} — ${esc(it.risk_why)}">${it.risk_tier} risk</span>` : ''}
        <span class="conf">${conf}${it.is_artifact?' · artifact':''}</span>
        <span class="meta">${it.image_id} · ${it.source}</span></div>
      ${it.risk_tier ? `<div class="why"><b>Risk:</b> ${esc(it.risk_mode)} — ${esc(it.risk_why)}</div>` : ''}
      <div class="why" title="${esc(it.rationale)}">${esc(it.rationale)}</div>
      ${d ? `<div class="saved">✓ saved as ${d} — click another label to change</div>` : ''}
      <div class="acts">
        <button class="primary" data-c="${acc||''}" ${acc && !d ? '' : 'disabled'}>${acc ? '✓ Accept: '+(SHORT[acc]||acc) : 'No suggestion — pick a label'}</button>
        ${choices.map(c=>`<button data-c="${c}" class="${c===sel?'sel':''}">${SHORT[c]||c}</button>`).join('')}
      </div>
    </div>`;
  el.querySelectorAll('.acts button').forEach(b => b.onclick = () => decide(it, b.dataset.c));
  el.querySelectorAll('.imgs img').forEach(i => i.onclick = () => lightbox(pid));
  return el;
}
function render(){
  chips(); stats();
  const g = $('#grid'); g.innerHTML = '';
  const s = shown();
  if(!s.length){g.innerHTML='<div class="empty">Nothing to review here.</div>'; return;}
  const f = document.createDocumentFragment(); s.forEach(it => f.appendChild(card(it))); g.appendChild(f);
}
function refreshCard(it){const o=document.getElementById('c_'+it.proposal_id); if(o) o.replaceWith(card(it)); stats();}
async function decide(it, choice){
  if(!choice) return; const r = reviewer(); if(!r) return;
  const revise = !!done[it.proposal_id];
  try{
    const res = await post('/quick/api/review',{proposal_id:it.proposal_id,choice,reviewer_id:r,revise});
    done[it.proposal_id] = res.label; refreshCard(it);
  }catch(e){toast('Save failed: '+e.message, 3500);}
}
$('#acceptAll').onclick = async () => {
  const r = reviewer(); if(!r) return;
  const ids = shown().filter(it=>!done[it.proposal_id] && it.accept).map(it=>it.proposal_id);
  if(!ids.length || !confirm(`Accept Claude's suggested label for ${ids.length} crops?`)) return;
  try{
    const res = await post('/quick/api/accept',{proposal_ids:ids,reviewer_id:r});
    Object.assign(done, res.accepted); render();
    toast(`Accepted ${Object.keys(res.accepted).length}` + (res.skipped.length?`, skipped ${res.skipped.length}`:''));
  }catch(e){toast('Save failed: '+e.message, 3500);}
};
$('#clear').onclick = () => {items = items.filter(it=>!done[it.proposal_id]); done = {}; render();};
function lightbox(pid){
  const lb = $('#lb');
  lb.innerHTML = `<img src="${BASE}/quick/full/${pid}/crop.png"><img src="${BASE}/quick/full/${pid}/context.png">`;
  lb.style.display = 'flex'; lb.onclick = () => lb.style.display = 'none';
}
document.addEventListener('keydown', e => {if(e.key==='Escape') $('#lb').style.display='none';});
(async () => {
  try{
    const r = await fetch(BASE+'/quick/api/items'); const d = await r.json();
    items = d.items; choices = d.choices; tiers = d.tiers || []; render();
  }catch(e){$('#grid').innerHTML='<div class="empty">Failed to load: '+e.message+'</div>';}
})();
</script></body></html>
"""
