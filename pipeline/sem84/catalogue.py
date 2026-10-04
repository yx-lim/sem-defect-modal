import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
STATES = ['observed', 'measured_2d', 'candidate_inference', 'not_observed_in_valid_view',
          'not_assessable', 'requires_external_evidence', 'unreviewed']
POSITIVE = {'observed', 'measured_2d', 'candidate_inference'}

# Features that ordinary SEM morphology in this dataset cannot resolve.
EXTERNAL = {
    'I04': 'Registered EDS/EELS maps (e.g. F for PVDF, Na for CMC) or calibrated low-kV ESB contrast to separate carbon-binder domain from active particles.',
    'I05': 'Registered EDS/EELS maps or calibrated ESB contrast identifying binder/carbon-black domains; morphology alone cannot identify binder.',
    'S01': 'Registered EDS (Si K) map of this field to confirm which BSE bright-phase candidates contain Si.',
    'S02': 'Registered EDS (Si K) map plus reviewed cluster envelopes; nearby bright candidates alone do not prove a Si agglomerate.',
    'S03': 'Registered EDS to identify Si, plus serial-section/FIB-SEM tomography for 3D enclosure; 2D boundary state kept only as a candidate relation.',
    'S04': 'Registered EDS for Si identity and FIB-SEM/X-ray tomography for 3D contact/free volume.',
    'S05': 'Higher-resolution imaging (TEM or high-resolution SEM) plus EDS on Si-confirmed particles.',
    'S06': 'EDS/EELS line scans and TEM/high-resolution imaging to establish core-shell chemistry and shell thickness.',
    'C01': 'EDS for elemental Si/inclusions; XPS/EELS/XANES for oxidation state; Raman/calibrated ESB for carbon phases.',
    'C02': 'Registered EDS (binder markers), Raman mapping for carbon black, or stained/contrast-calibrated imaging.',
    'C03': 'EBSD, Raman polarisation mapping or XRD texture; BSE channelling contrast is not a calibrated orientation map.',
    'C04': 'Raman (D/G ratio), XRD or TEM for graphitic disorder and grain structure.',
    'C05': 'EDS/XPS of the deposit or inclusion; SEM contrast cannot identify contamination.',
    'C06': 'XPS/EELS/XANES for surface oxide chemistry and TEM for thickness.',
    'C07': 'FTIR/Raman, TGA/DSC and nanoindentation or other mechanical tests of the binder.',
    'C08': 'TGA/Karl Fischer titration or XPS for residual moisture/solvent and surface condition.',
    'C09': 'Contact-angle or electrolyte wetting measurements.',
    'C10': 'Not applicable to fresh samples; needs formed/cycled samples with air-free XPS, cryo-imaging or operando methods.',
    'T01': 'FIB-SEM/X-ray tomography or serial sections, with open-to-which-boundary specified.',
    'T02': '3D pore network from tomography (body/throat extraction).',
    'T03': '3D reconstruction plus transport simulation or experimental tortuosity (e.g. EIS symmetric cell).',
    'T04': '3D reconstruction plus chemical phase identification for accessible surface/contact areas.',
    'T05': '3D reconstruction plus electronic conductivity/contact-resistance measurement.',
    'T06': 'Electrolyte filling experiments or operando imaging; fresh dry sections cannot show filling.',
    'T07': 'Registered multi-state imaging with DIC/DVC and a constitutive model.',
    'T08': 'Peel/scratch/pull-off adhesion and cohesion tests.',
    'T09': 'Conductivity/diffusivity/thermal measurements or 3D-model homogenisation.',
    'T10': 'Operando/post-mortem electrochemical mapping and validated reaction/plating models.',
    'Q08': 'Repeated-dose imaging comparison or acquisition/preparation records (beam current, dwell, dose).',
}
COATING_REF = {'H03', 'H04', 'H05', 'H06', 'H07', 'H10', 'I06', 'I07', 'V10'}
# Features whose positive state may only be a candidate without independent evidence.
CHEM_LIMITED = set(EXTERNAL)


def load_catalogue(path=None):
    path = pathlib.Path(path or ROOT / 'microstructure_feature_catalogue.json')
    data = json.loads(path.read_text())
    feats = data['features']
    assert len(feats) == 84 and len({f['id'] for f in feats}) == 84
    return feats
