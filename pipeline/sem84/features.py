"""Rules mapping analysis outputs onto all 84 catalogue records."""
import numpy as np

from .analysis import q, orientation_stats, windowed, clark_evans, chord_lengths
from .catalogue import EXTERNAL, COATING_REF, CHEM_LIMITED, STATES, load_catalogue

M_INST = 'MobileSAM point-grid proposals + hole fill + geometric screening (sem84 analysis.instance_geometry)'
M_DARK = 'gaussian(1px) < multi-Otsu void threshold, boundary-gradient support >= 1.2x median, 8-connected components'


def build_stats(inst, dark, rels, clusters, encl, masks, qc, tm, is_bse):
    valid = masks['valid']
    vpx = int(valid.sum())
    sup, unres, canvas = masks['supported_dark'], masks['unresolved_dark'], masks['canvas']
    ok = [i for i in inst if not i['frame_truncated']]
    s = {'valid_area_px': vpx, 'n_instances': len(inst), 'n_complete_instances': len(ok)}
    s['void_fraction'] = {'lower_supported': float(sup[valid].mean()),
                          'upper_including_unresolved_dark': float((sup | unres)[valid].mean())}
    s['windowed_void'] = {}
    grids = {}
    for w in (128, 256, 512, 1024):
        s['windowed_void'][w], grids[w] = windowed(sup, valid, w)
    s['windowed_particle'], grids['particle512'] = windowed(canvas > 0, valid, 512)
    if is_bse:
        s['windowed_contrast_B'], grids['b512'] = windowed(masks['contrast_B'], valid, 512)
    col = np.nanmean(grids[256], axis=0)
    row = np.nanmean(grids[256], axis=1)
    s['void_profile_x_256'] = [None if np.isnan(v) else float(v) for v in col]
    s['void_profile_y_256'] = [None if np.isnan(v) else float(v) for v in row]
    s['euler_number_8conn'] = int(__import__('skimage').measure.euler_number(sup, connectivity=2))
    s['chords'] = chord_lengths(sup & valid)
    s['grids'] = grids
    return s


def _rec(f, state, value=None, method=None, confidence=None, rationale='', evidence=None, ovi=None, unit='px'):
    assert state in STATES
    return {'id': f['id'], 'name': f['name'], 'group': f['group'], 'observability': f['observability'],
            'state': state, 'value': value, 'value_is_null': value is None, 'unit': unit if value is not None else None,
            'method': method, 'confidence': confidence, 'rationale': rationale, 'evidence_needed': evidence,
            'observed_vs_inferred': ovi or ('observed_2d' if state in ('observed', 'measured_2d') else
                                            'inferred_candidate' if state == 'candidate_inference' else 'none')}


def assess(ctx):
    inst, dark, rels, st, qc = ctx['inst'], ctx['dark'], ctx['rels'], ctx['stats'], ctx['qc']
    is_bse = ctx['is_bse']
    cat = load_catalogue()
    F = {f['id']: f for f in cat}
    out = {}
    ok = [i for i in inst if not i['frame_truncated']]
    B = [i for i in inst if i['class'] == 'particle_contrast_B']
    fiss = [d for d in dark if d['class'] == 'fissure_candidate']
    gaps = [d for d in dark if d['class'] == 'interfacial_gap']
    voids = [d for d in dark if d['class'] == 'void_like_region']
    cracks = [d for d in dark if 'interparticle_crack_candidate' in d['tags']]
    sup = [d for d in dark if d['class'] != 'unresolved_dark']
    touches = [r for r in rels if r['type'] == 'touches']
    res = qc['effective_resolution_limit_px']
    vpx = st['valid_area_px']
    trunc_note = f"{len(inst) - len(ok)} frame-truncated instances excluded from size/shape statistics"

    def put(i, *a, **k):
        out[i] = _rec(F[i], *a, **k)

    # ---- particles
    if inst:
        cls = {}
        for i in inst:
            cls[i['class']] = cls.get(i['class'], 0) + 1
        put('P01', 'observed', {'n_instances': len(inst), 'by_class': cls,
                                'labelled_particle_area_fraction': float(sum(i['area_px'] for i in inst) / vpx)},
            M_INST, 'medium', 'Instance masks from model proposals screened by geometry/contrast; classes are neutral contrast labels, not chemistry.', unit='count')
        put('P02', 'measured_2d', {'area_px2': q([i['area_px'] for i in ok]), 'equivalent_diameter_px': q([i['equivalent_diameter_px'] for i in ok]),
                                   'feret_max_px': q([i['feret_max_px'] for i in ok]), 'feret_min_px': q([i['feret_min_px'] for i in ok]),
                                   'oversize_tail_n_above_p95': int(sum(i['area_px'] > np.percentile([j['area_px'] for j in ok], 95) for i in ok)) if ok else 0},
            M_INST, 'medium', f'2D section-profile sizes (biased; not a 3D powder distribution). {trunc_note}.')
        put('P03', 'measured_2d', {'aspect_ratio': q([i['aspect_ratio'] for i in ok]), 'minor_axis_px': q([i['minor_axis_px'] for i in ok]),
                                   'major_axis_px': q([i['major_axis_px'] for i in ok])},
            'regionprops second-moment axes', 'medium', 'Section-intercept widths; true thickness not assigned because section/particle orientation is unknown.')
        put('P04', 'measured_2d', {'circularity': q([i['circularity'] for i in ok]), 'solidity': q([i['solidity'] for i in ok]),
                                   'convexity': q([i['convexity'] for i in ok])}, '4*pi*A/P^2; area/convex area; hull perimeter/perimeter', 'medium',
            'Shape descriptors from reviewed-pass geometry; perimeter is resolution-dependent.', unit='ratio')
        elong = [i for i in ok if i['aspect_ratio'] > 1.5]
        put('P05', 'measured_2d' if elong else 'unreviewed', orientation_stats([i['orientation_deg'] for i in elong]),
            'axial statistics of major-axis angles (aspect > 1.5)', 'medium', 'Projected 2D orientation in the section plane only.', unit='deg')
        put('P06', 'measured_2d', {'boundary_roughness_P_over_hullP': q([i['boundary_roughness'] for i in ok]),
                                   'median_facet_length_px': q([i['median_facet_length_px'] for i in ok]),
                                   'facet_count': q([i['facet_count'] for i in ok])},
            'Douglas-Peucker (3px) facets; perimeter/convex perimeter', 'low', f'Roughness below ~{res:.1f}px effective resolution is not resolved.', unit='ratio')
        fines = [i for i in inst if 'fine_relative_lower_decile' in i['tags']]
        put('P07', 'candidate_inference' if fines else 'unreviewed',
            {'n_fines_relative': len(fines), 'fine_area_fraction_of_particles': float(sum(i['area_px'] for i in fines) / max(1, sum(i['area_px'] for i in inst))),
             'cutoff': 'within-image lower decile of instance area (relative, not universal)', 'ids': [i['id'] for i in fines][:200]},
            'within-image area decile', 'low', 'Fragment vs fine particle cannot be separated from one 2D section; preparation chips also possible.', unit='count')
        nn = clark_evans(np.array([i['centroid_xy'] for i in inst]), vpx)
        put('P09', 'measured_2d', {'nearest_neighbour_centroid': nn, 'touch_coordination': q([i['touch_count'] for i in inst])},
            'centroid kd-tree; 2D touch graph', 'medium', 'Section-biased in-plane spacing/coordination; not 3D coordination number.', unit='px')
        tex = [i for i in inst if i.get('texture')]
        R = qc['directional_streaks']['cross_particle_alignment_R']
        put('P10', 'candidate_inference' if tex else 'unreviewed',
            {'median_texture_coherence': float(np.median([i['texture']['coherence'] for i in tex])) if tex else None,
             'n_measured': len(tex), 'cross_particle_alignment_R': R},
            'structure tensor of high-pass interior texture', 'low',
            'Internal texture/lamellae ambiguous with BSE channelling and milling streaks' + (' (texture aligned across particles: likely preparation/scan streaks)' if R > 0.6 else '') + '.', unit='ratio')
    else:
        for k in ('P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P09', 'P10'):
            put(k, 'unreviewed', rationale='No accepted particle instances; requires review.')
    if is_bse:
        cl = ctx['clusters']
        put('P08', 'candidate_inference' if cl else 'unreviewed',
            {'n_cluster_candidates': len(cl), 'sizes': [len(c['members']) for c in cl]} if cl else None,
            'single-linkage of contrast_B centroids (2x median eq. diameter)', 'low',
            'Proximity clusters only; nearby particles do not prove an agglomerate. Envelopes are visual only.', unit='count')
    else:
        put('P08', 'not_assessable', rationale='Composition-contrast clusters are assessed in the canonical BSE view of this stem.')

    # ---- pores and gaps
    vf = st['void_fraction']
    put('V01', 'measured_2d' if sup else 'unreviewed', vf, M_DARK, 'medium',
        'Apparent 2D dark void-like area / valid area; upper bound adds dark regions without boundary support (pore, shadow, milling, matrix ambiguous).', unit='fraction')
    put('V02', 'measured_2d' if voids else 'unreviewed', {'equivalent_diameter_px': q([d['equivalent_diameter_px'] for d in voids]),
                                                          'max_inscribed_diameter_px': q([d['max_inscribed_diameter_px'] for d in voids])},
        M_DARK + '; distance transform', 'medium', f'2D pore-body sizes; bodies near {res:.1f}px resolution limit unresolved.')
    sc = {}
    for d in sup:
        sc[d['shape_class']] = sc.get(d['shape_class'], 0) + 1
    put('V03', 'measured_2d' if sup else 'unreviewed', {'aspect_ratio': q([d['aspect_ratio'] for d in sup]), 'shape_classes': sc,
                                                        'orientation': orientation_stats([d['orientation_deg'] for d in sup if d['aspect_ratio'] > 1.5])},
        'regionprops; shape class rules (slit: AR>4 & width<=6px; rounded: solidity>0.85 & AR<2)', 'medium', 'Shape/orientation in section plane.', unit='ratio')
    loc = {}
    for d in sup:
        loc[d['location_2d']] = loc.get(d['location_2d'], 0) + 1
    put('V04', 'measured_2d' if sup else 'unreviewed', loc, 'fraction of dark component inside hole-filled instance masks; ring neighbours', 'medium',
        'Location relative to segmented 2D boundaries; depends on particle segmentation.', unit='count')
    thr = [d['width_profile_px']['min_throat'] / max(d['max_inscribed_diameter_px'], 1e-6) for d in sup if d['width_profile_px']['min_throat']]
    put('V05', 'measured_2d' if thr else 'unreviewed', {'throat_to_body_ratio': q(thr), 'min_throat_px': q([d['width_profile_px']['min_throat'] for d in sup])},
        '2x distance transform along skeleton', 'low', 'In-plane apertures only; throats below resolution are not detected.')
    ep, bp = sum(d['endpoints'] for d in sup), sum(d['branch_points'] for d in sup)
    put('V06', 'measured_2d' if sup else 'unreviewed', {'endpoints': ep, 'branch_points': bp, 'endpoint_density_per_Mpx': ep / vpx * 1e6,
                                                        'branch_density_per_Mpx': bp / vpx * 1e6, 'skeleton_length_px': int(sum(d['skeleton_length_px'] for d in sup))},
        'skimage skeletonize, 8-neighbour counts', 'low', '2D skeleton of this section only.', unit='count')
    inner = [d for d in sup if 'frame_truncated' not in d['tags']]
    put('V07', 'measured_2d' if sup else 'unreviewed', {'n_components': len(sup), 'n_not_touching_frame': len(inner),
                                                        'n_single_endpoint_or_less': sum(d['endpoints'] <= 1 for d in sup)},
        'connected components (8-conn)', 'low', 'Isolated-looking/dead-end-looking in this 2D section; may connect out of plane.', unit='count')
    if voids:
        areas = np.array([d['area_px'] for d in voids])
        p99 = float(np.percentile(areas, 99))
        put('V08', 'measured_2d', {'area_px2': q(areas, (50, 90, 95, 99)), 'n_above_within_image_p99': int((areas > p99).sum()),
                                   'largest_ids': [d['id'] for d in sorted(voids, key=lambda d: -d['area_px'])[:10]]},
            M_DARK, 'medium', 'Upper-tail cavities by within-image quantile; no universal cavity threshold.')
    else:
        put('V08', 'unreviewed', rationale='No supported void-like regions.')
    put('V09', 'measured_2d' if sup else 'unreviewed', {'window_px': 256, 'profile_x': st['void_profile_x_256'], 'profile_y': st['void_profile_y_256'],
                                                        'band_asserted': False},
        'windowed void fraction profiles', 'low', 'Directional profiles only; no dense/pore-rich band asserted without verified coating direction.', unit='fraction')
    ring = [i['ring_void_fraction'] for i in inst if i.get('ring_void_fraction') is not None]
    put('V11', 'measured_2d' if ring else 'unreviewed', {'ring_6px_void_fraction': q(ring)}, 'void fraction in 2-8px ring around each instance', 'low',
        'Halo/clearance in 2D section.', unit='fraction')
    put('V12', 'measured_2d' if sup else 'unreviewed', {'euler_number_8conn': st['euler_number_8conn'], 'chord_lengths': st['chords']},
        'skimage euler_number; run-length chords along x and y', 'low', '2D topology of the resolved void mask only.')

    # ---- discontinuities
    put('D01', 'candidate_inference' if fiss else 'unreviewed',
        {'n_fissure_candidates': len(fiss), 'n_hosts': len({d['host_instance'] for d in fiss}), 'ids': [d['id'] for d in fiss][:300]} if fiss else None,
        'thin elongated dark components >=70% inside one hole-filled instance', 'low',
        'Intraparticle fissure candidates linked to host; cause unknown (preparation fracture/lamellar edge alternatives retained); not cycling damage.', unit='count')
    put('D02', 'candidate_inference' if cracks else 'unreviewed',
        {'n_crack_candidates': len(cracks), 'ids': [d['id'] for d in cracks]} if cracks else None,
        'elongated dark components spanning >=3 instances, length >= 3x median particle eq. diameter', 'low',
        'Extended discontinuities across several boundaries; preparation fracture not excluded.', unit='count')
    fc = fiss + gaps + cracks
    put('D03', 'measured_2d' if fc else 'unreviewed', {'median_width_px': q([d['width_profile_px']['median'] for d in fc]),
                                                       'p90_width_px': q([d['width_profile_px']['p90'] for d in fc])} if fc else None,
        '2x distance transform along centreline', 'low', f'Apertures near {res:.1f}px resolution limit are unreliable.')
    put('D04', 'measured_2d' if fc else 'unreviewed', {'length_px': q([d['skeleton_length_px'] for d in fc]),
                                                       'length_density_per_Mpx': float(sum(d['skeleton_length_px'] for d in fc) / vpx * 1e6),
                                                       'branch_points': int(sum(d['branch_points'] for d in fc)), 'tips': int(sum(d['endpoints'] for d in fc)),
                                                       'orientation': orientation_stats([d['orientation_deg'] for d in fc])} if fc else None,
        'skeleton pixel count (approx. length), branch/endpoint counts', 'low', 'Centreline length from candidate fissures/gaps/cracks.')
    hosts = {}
    for d in fiss:
        hosts.setdefault(d['host_instance'], []).append(d['orientation_deg'])
    cleave = [h for h, a in hosts.items() if len(a) >= 2 and (max(a) - min(a) < 15 or max(a) - min(a) > 165)]
    put('D05', 'candidate_inference' if cleave else 'unreviewed', {'hosts_with_parallel_fissures': cleave} if cleave else None,
        '>=2 fissure candidates in one host with orientation spread < 15 deg', 'low', 'Exfoliation-like separation candidate only; channelling/streaks can mimic lamellae.', unit='count')
    big = {i['id'] for i in inst if 'fine_relative_lower_decile' not in i['tags']}
    fine_ids = {i['id'] for i in inst if 'fine_relative_lower_decile' in i['tags']}
    chips = sorted({r['a'] if r['a'] in fine_ids else r['b'] for r in touches if (r['a'] in fine_ids and r['b'] in big) or (r['b'] in fine_ids and r['a'] in big)})
    put('D06', 'candidate_inference' if chips else 'unreviewed', {'fines_touching_larger_particles': len(chips), 'ids': chips[:200]} if chips else None,
        'fine instances touching non-fine instances', 'low', 'Chip/fragment candidates; preparation debris not excluded.', unit='count')
    cav = [d for d in dark if 'particle_shaped_cavity_candidate' in d['tags']]
    put('D07', 'candidate_inference' if cav else 'unreviewed', {'n_particle_shaped_cavities': len(cav), 'ids': [d['id'] for d in cav]} if cav else None,
        'compact void (solidity>0.85) >= median particle area outside instances', 'low',
        'Particle-shaped cavity candidate; mechanical pull-out interpretation requires preparation evidence.', unit='count')
    if is_bse and B and ctx['masks']['fissure_gap'].any():
        dt = ctx['masks']['dist_to_fissure_gap']
        dd = [float(dt[int(i['centroid_xy'][1]), int(i['centroid_xy'][0])]) for i in B]
        put('D08', 'measured_2d', {'contrast_B_centroid_to_nearest_fissure_or_gap_px': q(dd)}, 'Euclidean distance transform of fissure+gap mask', 'low',
            'Distances only; correlation or causation not asserted; contrast_B chemistry unconfirmed.')
    else:
        put('D08', 'not_assessable' if not is_bse else 'unreviewed',
            rationale='Inclusion-centred distances computed in canonical BSE view only.' if not is_bse else 'No contrast_B candidates or fissure/gap candidates to relate.')

    # ---- contacts and interfaces
    if inst:
        put('I01', 'measured_2d' if touches else 'unreviewed', {'n_touch_relations': len(touches),
                                                                'apparent_contact_length_px': q([r['apparent_contact_length_px'] for r in touches]),
                                                                'coordination': q([i['touch_count'] for i in inst])},
            'boundary pixels within 2px of another instance', 'low', 'Apparent touching length; touch is not proof of electrical contact.')
        iso = [i['id'] for i in ok if i['touch_count'] == 0]
        put('I02', 'candidate_inference' if iso else 'unreviewed', {'n_isolated_looking': len(iso), 'ids': iso[:300]},
            'complete instances with zero touch relations', 'low', 'Apparently isolated in this 2D section; may be connected out of plane or via unresolved matrix.', unit='count')
        mf = [i['matrix_adjacent_perimeter_fraction'] for i in inst if i.get('matrix_adjacent_perimeter_fraction') is not None]
        put('I03', 'candidate_inference' if any(v > 0 for v in mf) else 'unreviewed', {'matrix_adjacent_perimeter_fraction': q(mf)},
            'ring adjacency to unresolved fine-matrix candidate pixels', 'low', 'Fine-matrix identity (binder/carbon/fines) unknown; textured unresolved region only.', unit='fraction')
        tri = 0
        adj = {}
        for r in touches:
            adj.setdefault(r['a'], set()).add(r['b'])
            adj.setdefault(r['b'], set()).add(r['a'])
        for a in adj:
            for b in adj[a]:
                tri += len(adj[a] & adj.get(b, set()))
        tri //= 6
        cfrac = []
        clen = {}
        for r in touches:
            clen[r['a']] = clen.get(r['a'], 0) + r['apparent_contact_length_px']
            clen[r['b']] = clen.get(r['b'], 0) + r['apparent_contact_length_px']
        for i in ok:
            cfrac.append(clen.get(i['id'], 0) / max(i['perimeter_px'], 1))
        put('I08', 'measured_2d', {'perimeter_px': q([i['perimeter_px'] for i in ok]), 'contact_perimeter_fraction': q(cfrac),
                                   'triple_junctions_touch_graph_triangles': tri},
            'perimeter, contact length / perimeter, 3-cliques of touch graph', 'low', 'Interface perimeter geometry in 2D section.')
    else:
        for k in ('I01', 'I02', 'I03', 'I08'):
            put(k, 'unreviewed', rationale='No accepted particle instances.')

    # ---- heterogeneity
    h1 = {'particle_area_fraction_512': st['windowed_particle']}
    if is_bse:
        h1['contrast_B_area_fraction_512'] = st['windowed_contrast_B']
    put('H01', 'measured_2d' if inst else 'unreviewed', h1, 'windowed area fractions (windows with >=80% valid pixels)', 'low',
        'Local packing/contrast-phase fractions; contrast phases are not chemistry.' + ('' if is_bse else ' Contrast phases not assessed in non-BSE view.'), unit='fraction')
    h2 = {'all_instances': clark_evans(np.array([i['centroid_xy'] for i in inst]), vpx) if inst else None}
    if is_bse:
        h2['contrast_B'] = clark_evans(np.array([i['centroid_xy'] for i in B]), vpx) if B else None
    put('H02', 'measured_2d' if inst else 'unreviewed', h2, 'Clark-Evans nearest-neighbour ratio', 'low', 'Spatial clustering in 2D section, no edge correction.', unit='ratio')
    put('H08', 'measured_2d' if sup else 'unreviewed', {str(k): v for k, v in st['windowed_void'].items()},
        'windowed void fraction at 128/256/512/1024 px', 'low', 'Multi-scale variability within this field only.', unit='fraction')
    unk = [i['id'] for i in inst if i['class'] == 'unknown_inclusion']
    put('H09', 'candidate_inference' if unk else 'unreviewed', {'n_unknown_inclusion_candidates': len(unk), 'ids': unk} if unk else None,
        'fibre-like shape or reviewer relabel', 'low', 'Unknown inclusion candidates; identity requires EDS.', unit='count')
    ext = qc['exterior_candidates']
    verified = ctx.get('verified_references', {})
    for k in sorted(COATING_REF):
        if verified.get('free_surface') or verified.get('collector'):
            put(k, 'unreviewed', rationale='Reviewer marked a reference as verified; quantitative coating-scale measurement requires manual measurement and was not automated.')
        else:
            put(k, 'not_assessable', rationale='No verified free surface, collector or coating direction in this field' +
                (f' ({len(ext)} large low-texture exterior candidate region(s) need review)' if ext else '') + '.')

    # ---- chemistry / 3D / functional
    for k, ev in EXTERNAL.items():
        note = ''
        if k in ('S01', 'S02') and is_bse:
            note = f' {len(B)} BSE bright-phase (contrast_B) candidates are listed in the E layer as contrast candidates only.'
        if k == 'S03' and is_bse:
            enc = ctx['enclosures']
            note = f" 2D boundary states of contrast_B candidates recorded as candidate relations ({sum(e['enclosure_2d_state']=='closed_2d_candidate' for e in enc)} closed-looking)."
        put(k, 'requires_external_evidence', None, None, None, 'Ordinary SEM morphology/contrast cannot resolve this entry.' + note, ev)

    # ---- QC
    ds = qc['directional_streaks']
    put('Q01', 'candidate_inference' if ds['aligned_tiles'] else 'unreviewed',
        {'aligned_tiles': len(ds['aligned_tiles']), 'streak_direction_deg': ds['global_streak_direction_deg'],
         'global_coherence': ds['global_coherence'], 'cross_particle_alignment_R': ds['cross_particle_alignment_R']},
        ds['method'], 'low', 'Directional streak candidate mask; curtaining vs scan streak vs texture requires visual review.', unit='count')
    put('Q02', 'unreviewed', rationale='Redeposition/smearing needs visual review at native resolution.')
    put('Q03', 'candidate_inference' if ext else 'unreviewed', {'exterior_candidates': ext} if ext else None,
        'large low-texture regions touching frame', 'low', 'Exterior/embedding candidate requires review.' if ext else 'No exterior candidate detected automatically; absence not yet reviewed.')
    put('Q04', 'unreviewed', {'related_candidates': {'D06': len(chips), 'D07': len(cav)}},
        None, None, 'Section fracture/debris/pull-out needs visual review; related morphology candidates referenced.', unit='count')
    hi = qc['clipped_high_fraction']
    put('Q05', 'candidate_inference' if (not is_bse and hi > 1e-3) else 'unreviewed', {'clipped_high_fraction': hi, 'detector': ctx['detector']},
        'saturated-pixel fraction', 'low', 'Charging/detector-dependent contrast candidate if saturation clusters; edge brightness in ETD/Inlens/SE is expected relief contrast.', unit='fraction')
    mid = ctx['masks'].get('showthrough_fraction')
    put('Q06', 'candidate_inference' if mid and mid > 0.2 else 'unreviewed', {'void_pixels_with_intermediate_grey_fraction': mid},
        'fraction of supported void pixels with grey > 0.6x void threshold', 'low', 'Possible subsurface show-through/pore shadow inside dark regions.', unit='fraction')
    put('Q07', 'measured_2d', {'low_sharpness_tiles': len(qc['blur']['low_sharpness_tiles']), 'scan_line_rows': len(qc['scan_line_rows']),
                               'seam_columns': len(qc['seam_columns'])}, qc['blur']['method'] + '; row/column mean jumps > 6 MAD', 'medium',
        'Acquisition QC metrics; flagged tiles/lines are in the F layer.', unit='count')
    put('Q09', 'measured_2d', {'clipped_low_fraction': qc['clipped_low_fraction'], 'clipped_high_fraction': hi,
                               'effective_resolution_limit_px': res}, qc['effective_resolution_method'], 'medium',
        'Clipping and approximate effective resolution; TIFF pitch (~25 nm/px tag) is unverified and not used.', unit='fraction')
    rej = ctx.get('reject_counts', {})
    put('Q10', 'measured_2d', {'frame_truncated_instances': len(inst) - len(ok), 'frame_truncated_fraction': (len(inst) - len(ok)) / max(1, len(inst)),
                               'tile_clipped_rejections': {k: v.get('crop_clipped_tile_truncation', 0) for k, v in rej.items()}},
        'bbox/invalid-region contact; internal crop-edge rejections', 'high', 'Frame-truncated objects are flagged and excluded from size statistics.', unit='count')
    assert len(out) == 84, sorted(set(F) - set(out))
    return [out[f['id']] for f in cat]


def apply_overrides(records, review_list, review_coverage):
    """Apply reviewer overrides in order. Returns warnings for rejected overrides."""
    warn = []
    by = {r['id']: r for r in records}
    for rv in review_list:
        for fid, ov in (rv.get('feature_overrides') or {}).items():
            r = by.get(fid)
            st = ov.get('state')
            if r is None or st not in STATES:
                warn.append(f'override {fid}: invalid id/state {st}')
                continue
            if fid in CHEM_LIMITED and st not in ('requires_external_evidence', 'candidate_inference'):
                warn.append(f'override {fid}: {st} not allowed without independent evidence')
                continue
            if st == 'not_observed_in_valid_view' and review_coverage < 0.5:
                warn.append(f'override {fid}: absence needs >=50% inspected valid area (have {review_coverage:.2f})')
                continue
            if not ov.get('rationale'):
                warn.append(f'override {fid}: missing rationale')
                continue
            r['auto_state'] = r.get('auto_state', r['state'])
            r['state'] = st
            if 'value' in ov:
                r['value'] = ov['value']
                r['value_is_null'] = ov['value'] is None
            r['rationale'] = ov['rationale'] + ' [auto: ' + r['rationale'] + ']'
            if ov.get('confidence'):
                r['confidence'] = ov['confidence']
            r['review_status'] = f"overridden_by_{rv.get('reviewer', 'reviewer')}_{rv.get('stage')}"
        for fid in rv.get('confirm_not_assessable') or []:
            r = by.get(fid)
            if r and r['state'] in ('not_assessable', 'requires_external_evidence'):
                r['review_status'] = f"confirmed_by_{rv.get('reviewer', 'reviewer')}_{rv.get('stage')}"
    for r in records:
        r.setdefault('review_status', 'auto_unreviewed')
    return warn
