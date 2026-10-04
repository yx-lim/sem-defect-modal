"""Six clean visualization layers plus overview, charts, dashboard, QA crops and review tiles."""
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage as ndi
from skimage.segmentation import find_boundaries

FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
COL = {
    'particle': (35, 231, 213), 'fine': (255, 224, 0), 'truncated': (255, 159, 28),
    'void': (61, 139, 255), 'fissure': (255, 61, 210), 'gap': (255, 140, 0), 'crack': (255, 46, 46),
    'unresolved': (154, 167, 184), 'skeleton': (127, 209, 255),
    'touch': (57, 255, 20), 'sep_gap': (255, 140, 0), 'isolated': (255, 46, 46), 'matrix': (190, 120, 255),
    'contrastB': (255, 210, 63), 'cluster': (255, 255, 255), 'transferred': (255, 170, 60),
    'clip_low': (255, 0, 0), 'clip_high': (255, 0, 255), 'blur': (255, 255, 0), 'curtain': (0, 255, 255),
    'scan': (255, 0, 255), 'ignore': (40, 60, 90), 'rev_art': (255, 120, 0),
}
STATE_COL = {'observed': '#2ca02c', 'measured_2d': '#1f77b4', 'candidate_inference': '#ff7f0e',
             'not_observed_in_valid_view': '#9edae5', 'not_assessable': '#7f7f7f',
             'requires_external_evidence': '#9467bd', 'unreviewed': '#d62728'}


def font(sz, bold=False):
    return ImageFont.truetype(FONTB if bold else FONT, sz)


def _thick(b, n=1):
    return ndi.binary_dilation(b, iterations=n) if n else b


def _paint(ov, mask, color, alpha=255):
    ov[mask, :3] = color
    ov[mask, 3] = alpha


class Renderer:
    def __init__(self, g, ctx):
        self.g = g
        self.H, self.W = g.shape
        self.ctx = ctx
        yy, xx = np.indices((self.H, self.W))
        self.dash = ((xx // 8 + yy // 8) % 2 == 0)
        self.hatch = ((xx + yy) % 10 == 0)

    def blank(self):
        return np.zeros((self.H, self.W, 4), np.uint8)

    # -- layers ---------------------------------------------------------------
    def layer_A(self):
        c = self.ctx
        ov = self.blank()
        canvas = c['masks']['canvas']
        b = find_boundaries(canvas, mode='inner') & (canvas > 0)
        b = _thick(b) & (canvas > 0)
        lut = np.zeros((int(canvas.max()) + 1, 3), np.uint8)
        kind = np.zeros(int(canvas.max()) + 1, np.uint8)
        for i in c['inst']:
            if i['frame_truncated']:
                kind[i['label']] = 2
            elif 'fine_relative_lower_decile' in i['tags']:
                kind[i['label']] = 1
        lut[kind == 0] = COL['particle']
        lut[kind == 1] = COL['fine']
        lut[kind == 2] = COL['truncated']
        sel = b & ~((kind[canvas] == 2) & ~self.dash)
        ov[sel, :3] = lut[canvas[sel]]
        ov[sel, 3] = 255
        return ov, [('particle instance', COL['particle'], 'solid'), ('fine (lower-decile area)', COL['fine'], 'solid'),
                    ('frame-truncated', COL['truncated'], 'dashed')]

    def _dark_outline(self, cls_set, tag=None):
        lab = self.ctx['masks']['dark_lab']
        sel = np.zeros(int(lab.max()) + 1, bool)
        for d in self.ctx['dark']:
            if d['class'] in cls_set and (tag is None or tag in d['tags']):
                sel[d['label']] = True
        m = sel[lab]
        return find_boundaries(m, mode='outer') & ~m, m

    def layer_B(self):
        ov = self.blank()
        bv, _ = self._dark_outline({'void_like_region'})
        _paint(ov, bv, COL['void'])
        _, fgm = self._dark_outline({'fissure_candidate', 'interfacial_gap'})
        _paint(ov, self.ctx['masks']['skeleton'] & fgm, COL['skeleton'])
        bg, _ = self._dark_outline({'interfacial_gap'})
        _paint(ov, _thick(bg) & ~self.ctx['masks']['supported_dark'], COL['gap'])
        bf, _ = self._dark_outline({'fissure_candidate'})
        _paint(ov, _thick(bf) & ~self.ctx['masks']['supported_dark'], COL['fissure'])
        bc, _ = self._dark_outline({'void_like_region', 'interfacial_gap', 'fissure_candidate'}, 'interparticle_crack_candidate')
        _paint(ov, _thick(bc, 2) & self.dash & ~self.ctx['masks']['supported_dark'], COL['crack'])
        _paint(ov, self.ctx['masks']['unresolved_dark'] & self.hatch, COL['unresolved'])
        return ov, [('void-like (boundary-supported)', COL['void'], 'solid'), ('fissure/gap skeleton', COL['skeleton'], 'solid'),
                    ('fissure candidate (host-linked)', COL['fissure'], 'solid'), ('interfacial gap', COL['gap'], 'solid'),
                    ('interparticle crack candidate', COL['crack'], 'dashed'), ('unresolved dark (ignore)', COL['unresolved'], 'hatched')]

    def _disc(self, ov, xy, r, color, ring=False):
        x, y = int(round(xy[0])), int(round(xy[1]))
        y0, y1, x0, x1 = max(0, y - r), min(self.H, y + r + 1), max(0, x - r), min(self.W, x + r + 1)
        yy, xx = np.ogrid[y0:y1, x0:x1]
        d = np.hypot(yy - y, xx - x)
        m = (d <= r) & ((d >= r - 2) if ring else True)
        sub = ov[y0:y1, x0:x1]
        sub[m, :3] = color
        sub[m, 3] = 255

    def layer_C(self):
        ov = self.blank()
        mx = self.ctx['masks']['matrix']
        _paint(ov, mx & self.hatch, COL['matrix'])
        for r in self.ctx['rels']:
            if r['type'] == 'touches':
                self._disc(ov, r['point_xy'], 4, COL['touch'])
            elif r['type'] == 'separated_by_gap':
                self._disc(ov, r['point_xy'], 5, COL['sep_gap'], ring=True)
        for i in self.ctx['inst']:
            if i['touch_count'] == 0 and not i['frame_truncated']:
                self._disc(ov, i['centroid_xy'], 10, COL['isolated'], ring=True)
        return ov, [('apparent touch (not electrical proof)', COL['touch'], 'point'), ('separated by gap', COL['sep_gap'], 'ring'),
                    ('isolated-looking particle', COL['isolated'], 'ring'), ('unresolved fine-matrix candidate', COL['matrix'], 'hatched')]

    def layer_E(self):
        ov = self.blank()
        c = self.ctx
        canvas = c['masks']['canvas']
        lutB = np.zeros(int(canvas.max()) + 1, bool)
        for i in c['inst']:
            if i['class'] == 'particle_contrast_B':
                lutB[i['label']] = True
        mB = lutB[canvas]
        _paint(ov, _thick(find_boundaries(mB, mode='outer') & ~mB), COL['contrastB'])
        img = Image.fromarray(ov, 'RGBA')
        dr = ImageDraw.Draw(img)
        for cl in c['clusters']:
            pts = [tuple(p) for p in cl['envelope_xy_visual_only']]
            if len(pts) >= 3:
                pts.append(pts[0])
                for k in range(len(pts) - 1):
                    (xa, ya), (xb, yb) = pts[k], pts[k + 1]
                    n = max(1, int(math.dist(pts[k], pts[k + 1]) // 14))
                    for s in range(0, n, 2):
                        t0, t1 = s / n, min(1, (s + 1) / n)
                        dr.line([(xa + (xb - xa) * t0, ya + (yb - ya) * t0), (xa + (xb - xa) * t1, ya + (yb - ya) * t1)],
                                fill=COL['cluster'] + (255,), width=2)
        ov = np.asarray(img).copy()
        tm = c['masks'].get('transferred_bright')
        if tm is not None:
            _paint(ov, find_boundaries(tm, mode='outer') & ~tm & self.dash, COL['transferred'])
        leg = [('contrast_B / bright-phase candidate', COL['contrastB'], 'solid'),
               ('proximity cluster envelope (visual only)', COL['cluster'], 'dashed')]
        if tm is not None:
            leg.append(('BSE bright candidate via registration', COL['transferred'], 'dashed'))
        return ov, leg

    def layer_F(self):
        ov = self.blank()
        m = self.ctx['masks']
        _paint(ov, m['semantic'] == 255, COL['ignore'], 70)
        _paint(ov, m['unresolved_dark'] & self.hatch, COL['unresolved'], 220)
        _paint(ov, m['clip_low'], COL['clip_low'], 150)
        _paint(ov, m['clip_high'], COL['clip_high'], 150)
        img = Image.fromarray(ov, 'RGBA')
        dr = ImageDraw.Draw(img)
        qc = self.ctx['qc']
        for b in qc['blur']['low_sharpness_tiles']:
            dr.rectangle(b, outline=COL['blur'] + (255,), width=3)
        for b in qc['directional_streaks']['aligned_tiles']:
            dr.rectangle([b[0] + 4, b[1] + 4, b[2] - 4, b[3] - 4], outline=COL['curtain'] + (255,), width=2)
        for r in qc['scan_line_rows']:
            dr.line([(0, r), (self.W, r)], fill=COL['scan'] + (200,), width=1)
        for c_ in qc['seam_columns']:
            dr.line([(c_, 0), (c_, self.H)], fill=COL['scan'] + (200,), width=1)
        for reg in self.ctx.get('artifact_regions', []):
            dr.rectangle(reg['bbox_xyxy'], outline=COL['rev_art'] + (255,), width=3)
        return np.asarray(img).copy(), [('ignore / unknown (not background)', COL['ignore'], 'tint'), ('unresolved dark', COL['unresolved'], 'hatched'),
                                        ('clipped low (0)', COL['clip_low'], 'fill'), ('clipped high (255)', COL['clip_high'], 'fill'),
                                        ('low-sharpness tile', COL['blur'], 'box'), ('aligned streak tile', COL['curtain'], 'box'),
                                        ('scan-line / seam jump', COL['scan'], 'line'), ('reviewer artifact region', COL['rev_art'], 'box')]

    # -- composition ------------------------------------------------------------
    def composite(self, ov):
        base = np.repeat(self.g[..., None], 3, axis=2).astype(np.float32)
        a = ov[..., 3:4].astype(np.float32) / 255
        return (base * (1 - a) + ov[..., :3] * a).astype(np.uint8)

    def footer(self, title, legend, status_lines, width=None):
        W = width or self.W
        h = 70 + 38 * max(math.ceil(len(legend) / 4), 1) + 32 * len(status_lines)
        im = Image.new('RGB', (W, h), (18, 18, 22))
        d = ImageDraw.Draw(im)
        d.text((20, 12), title, fill=(255, 255, 255), font=font(30, True))
        colw = W // 4
        for k, (name, col, style) in enumerate(legend):
            x, y = 20 + (k % 4) * colw, 60 + (k // 4) * 38
            if style in ('dashed',):
                for s in range(0, 44, 12):
                    d.line([(x + s, y + 14), (x + s + 7, y + 14)], fill=col, width=4)
            elif style in ('point', 'ring'):
                d.ellipse([x + 12, y + 4, x + 32, y + 24], outline=col, width=3, fill=col if style == 'point' else None)
            elif style == 'hatched':
                for s in range(0, 44, 6):
                    d.line([(x + s, y + 26), (x + s + 10, y + 2)], fill=col, width=1)
            else:
                d.rectangle([x, y + 4, x + 44, y + 24], outline=col, width=3, fill=col if style in ('fill', 'tint') else None)
            d.text((x + 56, y + 2), name, fill=(230, 230, 230), font=font(22))
        y = 60 + 38 * max(math.ceil(len(legend) / 4), 1)
        for line in status_lines:
            d.text((20, y), line, fill=(200, 200, 200), font=font(22))
            y += 32
        return im

    def save_layer(self, path, ov, title, legend, status, sidecar=None):
        img = Image.fromarray(self.composite(ov))
        parts = [img] + ([sidecar] if sidecar is not None else []) + [self.footer(title, legend, status)]
        out = Image.new('RGB', (self.W, sum(p.height for p in parts)))
        y = 0
        for p in parts:
            out.paste(p, (0, y))
            y += p.height
        a = np.asarray(out)
        packed = (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2]
        u = np.unique(packed)
        if len(u) <= 256:  # lossless palette encoding
            idx = np.searchsorted(u, packed).astype(np.uint8)
            pim = Image.fromarray(idx, 'P')
            pal = np.stack([(u >> 16) & 255, (u >> 8) & 255, u & 255], 1).astype(np.uint8).ravel().tolist()
            pim.putpalette(pal + [0] * (768 - len(pal)))
            pim.save(path, compress_level=6)
        else:
            out.save(path, optimize=False, compress_level=6)

    def sidecar_D(self, grid, profile):
        hh = max(240, self.H // 6)
        vmax = np.nanmax(grid) if np.isfinite(grid).any() else 1
        cmap = plt.get_cmap('viridis')
        rgba = cmap(np.nan_to_num(grid / max(vmax, 1e-6), nan=0))
        rgba[np.isnan(grid)] = (0.3, 0.3, 0.3, 1)
        hm = Image.fromarray((rgba[..., :3] * 255).astype(np.uint8)).resize((self.W, hh), Image.Resampling.NEAREST)
        d = ImageDraw.Draw(hm)
        d.text((12, 8), f'Sidecar: windowed void-like fraction (256 px windows, grey = <80% valid); max {vmax:.3f}. Orientation: image x right, y down; coating direction unverified.',
               fill=(255, 255, 255), font=font(22))
        return hm

    def write_all(self, outdir, feats_by_id, title_prefix, composites=True):
        c = self.ctx
        layers = {}
        ovdir = outdir / 'overlays'
        ldir = outdir / 'layers'
        ovdir.mkdir(exist_ok=True)
        ldir.mkdir(exist_ok=True)

        def st(ids):
            return '  '.join(f"{i}:{feats_by_id[i]['state']}" for i in ids)
        cov = c['coverage']
        covline = f"Labelled (non-ignore) coverage of valid area: {cov:.1%}. Units: pixels (TIFF ~25 nm/px tag unverified). Draft labelling, not expert ground truth."
        spec = [
            ('A_particles', self.layer_A, 'A. PARTICLES', [st(['P01', 'P02', 'P03', 'P04', 'P05', 'P07', 'Q10'])]),
            ('B_pores_damage', self.layer_B, 'B. PORES AND DAMAGE', [st(['V01', 'V02', 'V04', 'V08', 'D01', 'D02', 'D03', 'D07'])]),
            ('C_contacts_interfaces', self.layer_C, 'C. CONTACTS AND INTERFACES', [st(['I01', 'I02', 'I03', 'I06', 'I07', 'I08']), 'Collector interface: not visible/verified in this field.']),
            ('E_chemistry_conditional_si', self.layer_E, 'E. CHEMISTRY AND CONDITIONAL SILICON — CHEMISTRY UNCONFIRMED',
             [st(['S01', 'S02', 'S03', 'C01', 'P08']), ('BSE contrast_B = bright-phase candidate only; no Si-confirmed colour without registered EDS.' if c['is_bse'] else
                                                         'Non-BSE view: composition contrast not assessed here' + ('; dashed = BSE candidates via validated registration.' if c['masks'].get('transferred_bright') is not None else '.'))]),
            ('F_prep_acquisition_qc', self.layer_F, 'F. PREPARATION / ACQUISITION QC (QA copy, not a defect map)', [st(['Q01', 'Q03', 'Q05', 'Q07', 'Q09', 'Q10'])]),
        ]
        counts = {}
        for name, fn, title, status in spec:
            ov, leg = fn()
            Image.fromarray(ov, 'RGBA').save(ovdir / f'{name}.png', compress_level=6)
            if composites:
                self.save_layer(ldir / f'{name}.png', ov, f'{title_prefix}  {title}', leg, status + [covline])
            (ovdir / f'{name}.legend.json').write_text(json.dumps({'title': title, 'legend': [[n, list(c), st_] for n, c, st_ in leg], 'status': status + [covline]}))
            counts[name] = int((ov[..., 3] > 0).sum())
            layers[name] = ov
        # D: no overlay on the micrograph; sidecar heatmap + footer
        ovD = self.blank()
        Image.fromarray(ovD, 'RGBA').save(ovdir / 'D_coating_heterogeneity.png', compress_level=6)
        side = self.sidecar_D(c['stats']['grids'][256], c['stats']['void_profile_x_256'])
        side.save(ovdir / 'D_coating_heterogeneity_sidecar.png')
        if composites:
          self.save_layer(ldir / 'D_coating_heterogeneity.png', ovD, f'{title_prefix}  D. COATING-SCALE HETEROGENEITY', [],
                        [st(['H01', 'H02', 'H03', 'H04', 'H05', 'H08', 'V09']),
                         'Not overlaid: free surface / collector / coating direction unverified in this field (skin, bands, gradients, thickness not assessable).', covline],
                        sidecar=side)
        counts['D_coating_heterogeneity'] = int(np.isfinite(c['stats']['grids'][256]).sum())
        Image.fromarray(self.g).save(outdir / 'source_preview.jpg', quality=92)
        self.html(outdir)
        self.overview(outdir / 'overview.png', layers, title_prefix)
        return counts, layers

    def html(self, outdir):
        names = ['A_particles', 'B_pores_damage', 'C_contacts_interfaces', 'D_coating_heterogeneity', 'E_chemistry_conditional_si', 'F_prep_acquisition_qc']
        boxes = ''.join(f'<label><input type="checkbox" onchange="t(\'{n}\',this.checked)" {"checked" if n == "A_particles" else ""}> {n}</label> ' for n in names)
        imgs = ''.join(f'<img id="{n}" src="overlays/{n}.png" style="display:{"block" if n == "A_particles" else "none"}">' for n in names)
        (outdir / 'layers.html').write_text(f'''<!doctype html><meta charset="utf-8"><title>layer toggle</title>
<style>body{{background:#111;color:#ddd;font:14px sans-serif}}#v{{position:relative;width:100%}}#v img{{position:absolute;left:0;top:0;width:100%}}#v img.base{{position:relative}}</style>
<div>{boxes} — legend/status: see layers/*.png footers; CHEMISTRY UNCONFIRMED</div>
<div id="v"><img class="base" src="source_preview.jpg">{imgs}</div>
<script>function t(n,s){{document.getElementById(n).style.display=s?'block':'none'}}</script>''')

    def overview(self, path, layers, title):
        f = 1600 / self.W
        size = (1600, max(1, round(self.H * f)))
        base = Image.fromarray(self.g).resize(size, Image.Resampling.LANCZOS).convert('RGB')
        canvas = Image.fromarray(self.ctx['masks']['canvas'].astype(np.int32)).resize(size, Image.Resampling.NEAREST)
        cs = np.asarray(canvas)
        b = find_boundaries(cs, mode='inner') & (cs > 0)
        arr = np.asarray(base).copy()
        arr[b] = COL['particle']
        fis = self.ctx['masks']['fissure_only']
        fs = np.asarray(Image.fromarray(fis.astype(np.uint8) * 255).resize(size, Image.Resampling.BOX)) > 0
        arr[fs] = COL['fissure']
        img = Image.fromarray(arr)
        foot = self.footer(title + '  overview', [('particle instance', COL['particle'], 'solid'), ('fissure candidate', COL['fissure'], 'fill')],
                           ['Core classes only. Full layers in layers/. CHEMISTRY UNCONFIRMED.'], width=1600)
        out = Image.new('RGB', (1600, img.height + foot.height))
        out.paste(img, (0, 0))
        out.paste(foot, (0, img.height))
        out.save(path)

    def qa_crops(self, outdir, layers, items, size=384, limit=24):
        qd = outdir / 'qa'
        qd.mkdir(exist_ok=True)
        comb = layers['A_particles'].copy()
        for k in ('B_pores_damage',):
            m = layers[k][..., 3] > 0
            comb[m] = layers[k][m]
        comp = self.composite(comb)
        index = []
        for n, (kind, oid, xy, why) in enumerate(items[:limit]):
            x, y = int(xy[0]), int(xy[1])
            x0, y0 = max(0, min(self.W - size, x - size // 2)), max(0, min(self.H - size, y - size // 2))
            raw = Image.fromarray(self.g[y0:y0 + size, x0:x0 + size]).convert('RGB')
            ann = Image.fromarray(comp[y0:y0 + size, x0:x0 + size])
            panel = Image.new('RGB', (2 * size + 10, size + 40), (18, 18, 22))
            panel.paste(raw, (0, 40))
            panel.paste(ann, (size + 10, 40))
            ImageDraw.Draw(panel).text((6, 8), f'{kind} {oid} @({x},{y}) native px — {why}'[:90], fill=(255, 255, 255), font=font(16))
            fn = f'qa_{n + 1:02d}_{kind}_{oid}.png'
            panel.save(qd / fn)
            index.append({'file': f'qa/{fn}', 'kind': kind, 'object_id': oid, 'xy': [x, y], 'crop_xyxy': [x0, y0, x0 + size, y0 + size], 'reason': why})
        (qd / 'index.json').write_text(json.dumps(index, indent=1))
        return index

    def review_tiles(self, outdir, layers, inst, nx=5, ny=2):
        rd = outdir / 'review_tiles'
        rd.mkdir(exist_ok=True)
        comb = layers['A_particles'].copy()
        for k in ('B_pores_damage', 'E_chemistry_conditional_si'):
            m = layers[k][..., 3] > 0
            comb[m] = layers[k][m]
        comp = Image.fromarray(self.composite(comb))
        d = ImageDraw.Draw(comp)
        fnt = font(18)
        for i in inst:
            d.text((i['centroid_xy'][0] - 20, i['centroid_xy'][1] - 9), i['id'][3:], fill=(255, 255, 0), font=fnt, stroke_width=2, stroke_fill=(0, 0, 0))
        tw, th = math.ceil(self.W / nx), math.ceil(self.H / ny)
        tiles = []
        for r in range(ny):
            for c_ in range(nx):
                box = (c_ * tw, r * th, min(self.W, (c_ + 1) * tw), min(self.H, (r + 1) * th))
                comp.crop(box).save(rd / f'tile_r{r}c{c_}_annotated.jpg', quality=88)
                Image.fromarray(self.g).crop(box).save(rd / f'tile_r{r}c{c_}_raw.jpg', quality=92)
                tiles.append({'tile': f'r{r}c{c_}', 'bbox_xyxy': list(box)})
        (rd / 'index.json').write_text(json.dumps(tiles, indent=1))
        return tiles


def charts(path, ctx):
    inst = [i for i in ctx['inst'] if not i['frame_truncated']]
    dark = [d for d in ctx['dark'] if d['class'] != 'unresolved_dark']
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(231)
    if inst:
        ax.hist(np.log10([i['equivalent_diameter_px'] for i in inst]), bins=30, color='#23a7a0')
    ax.set_title('particle eq. diameter (log10 px), 2D section')
    ax = fig.add_subplot(232)
    if inst:
        ax.hist([i['aspect_ratio'] for i in inst], bins=30, color='#555')
    ax.set_title('particle aspect ratio')
    ax = fig.add_subplot(233, projection='polar')
    if inst:
        a = np.deg2rad([i['orientation_deg'] for i in inst if i['aspect_ratio'] > 1.5])
        a = np.concatenate([a, a + np.pi])
        ax.hist(a, bins=36, color='#23a7a0')
    ax.set_title('particle long-axis orientation (axial)')
    ax = fig.add_subplot(234)
    v = [d['equivalent_diameter_px'] for d in dark if d['class'] == 'void_like_region']
    if v:
        ax.hist(np.log10(v), bins=30, color='#3d8bff')
    ax.set_title('void-like eq. diameter (log10 px)')
    ax = fig.add_subplot(235)
    w = [d['width_profile_px']['median'] for d in dark if d['class'] in ('fissure_candidate', 'interfacial_gap') and d['width_profile_px']['median']]
    if w:
        ax.hist(w, bins=20, color='#ff3dd2')
    ax.set_title('fissure/gap median width (px)')
    ax = fig.add_subplot(236)
    p = [np.nan if x is None else x for x in ctx['stats']['void_profile_x_256']]
    ax.plot(np.arange(len(p)) * 256, p, color='#3d8bff')
    ax.set_title('void fraction vs x (256 px windows)')
    fig.suptitle(f"{ctx['image_id']} — draft 2D statistics in pixel units (calibration unverified)")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def dashboard(path, feats, title):
    fig, ax = plt.subplots(figsize=(15, 8))
    for k, f in enumerate(feats):
        r, c = divmod(k, 12)
        ax.add_patch(plt.Rectangle((c, -r), 0.95, 0.9, color=STATE_COL[f['state']]))
        ax.text(c + 0.47, -r + 0.45, f['id'], ha='center', va='center', color='white', fontsize=10, weight='bold')
    ax.set_xlim(0, 12)
    ax.set_ylim(-7, 1)
    ax.axis('off')
    counts = {}
    for f in feats:
        counts[f['state']] = counts.get(f['state'], 0) + 1
    handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COL[s]) for s in STATE_COL]
    ax.legend(handles, [f'{s} ({counts.get(s, 0)})' for s in STATE_COL], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.12))
    ax.set_title(title + ' — 84-feature status (CHEMISTRY UNCONFIRMED)')
    fig.savefig(path, dpi=80, bbox_inches='tight')
    plt.close(fig)
