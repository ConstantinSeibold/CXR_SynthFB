# Install once (commented to avoid reinstall):
# !pip install cxas opencv-python matplotlib pycocotools

import os, json, math, random, uuid, warnings
from pathlib import Path

import numpy as np
import cv2
import torch
from PIL import Image
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
RNG_SEED = 0
random.seed(RNG_SEED); np.random.seed(RNG_SEED); torch.manual_seed(RNG_SEED)

# All paths + run params overridable via env vars (for generate_synthfb.sh).
# Defaults are relative — set MIMIC_CLEAN to your MIMIC images root before running.
MIMIC_ROOT       = Path(os.environ.get('MIMIC_CLEAN',       './mimic_data/images'))
MIMIC_SUBSET_CSV = Path(os.environ.get('MIMIC_SUBSET_CSV',  './mimic_fb_free_train.csv'))
CUTOUTS_ROOT     = Path(os.environ.get('CUTOUTS_ROOT',      './cutouts/labels'))
OUT_DIR          = Path(os.environ.get('SYNTHFB_OUT',       './synthfb_out'))
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_IMAGES        = int(os.environ.get('SYNTHFB_N',          5))     # bump for paper-scale runs
MAX_ANNOTATIONS = int(os.environ.get('SYNTHFB_MAX_ANN',    12))
CUTPASTE_PROB   = float(os.environ.get('SYNTHFB_CUTPASTE_PROB', 0.25))
DEVICE_PROB     = float(os.environ.get('SYNTHFB_DEVICE_PROB',   0.25))
RNG_SEED        = int(os.environ.get('SYNTHFB_SEED',       RNG_SEED))
# Ablation gate for anatomy-guided PLACEMENT (rib-occlusion / physics rendering keep the real masks -> appearance
# held constant). SYNTHFB_PLACEMENT: 'anatomy' (default), 'fallback' (zero masks -> each rule's geometric default),
# or 'random' (whole anatomy stack rigidly shifted a random amount PER DEVICE -> coherent multi-part devices but
# placement decorrelated from the real patient anatomy, full positional diversity). SYNTHFB_USE_ANATOMY=0 kept as
# an alias for 'fallback' (back-compat).
USE_ANATOMY     = int(os.environ.get('SYNTHFB_USE_ANATOMY', 1)) == 1
PLACEMENT_MODE  = os.environ.get('SYNTHFB_PLACEMENT', 'anatomy' if USE_ANATOMY else 'fallback')

# ---- speed flags (production: set all 3 to skip galleries/viz, use JPG) ----
SKIP_VIZ        = int(os.environ.get('SYNTHFB_SKIP_VIZ',     0)) == 1
SKIP_GALLERY    = int(os.environ.get('SYNTHFB_SKIP_GALLERY', 0)) == 1
JPG_QUALITY     = int(os.environ.get('SYNTHFB_JPG_QUALITY',  0))   # 0 = PNG. 85-95 = JPG quality.
PNG_COMPRESS    = int(os.environ.get('SYNTHFB_PNG_COMPRESS', 1))   # 0=fastest/largest, 9=slowest/smallest

# ---- sim-to-real DEVICE realism post-processing (default OFF; opt-in via SYNTHFB_REALISM=1) ----
# INSERTION-ONLY: only the composited device / foreign-body regions are modified;
# the REAL MIMIC base is left UNTOUCHED (re-toning a real base is not realism and
# corrupts real anatomy). Within the feathered device mask it pulls each device
# toward its local background to cut the over-contrast "pop" synth devices show vs
# real ones, adds matched detector grain, and feathers the boundary -- no global
# filtering, no base blur. Randomized per image (domain randomization). Geometry
# (masks / boxes / keypoints) is untouched. Tune against a real-vs-synth
# device-crop metric (e.g. KID on device crops), NOT a blur that games a single
# global statistic.
REALISM_ENABLE   = int(os.environ.get('SYNTHFB_REALISM', 0)) == 1
REALISM_PROB     = float(os.environ.get('SYNTHFB_REALISM_PROB', 0.9))
REALISM_CONTRAST = (float(os.environ.get('SYNTHFB_REALISM_CONTRAST_MIN', 0.28)),
                    float(os.environ.get('SYNTHFB_REALISM_CONTRAST_MAX', 0.40)))
REALISM_GRAIN    = (float(os.environ.get('SYNTHFB_REALISM_GRAIN_MIN', 4.0)),
                    float(os.environ.get('SYNTHFB_REALISM_GRAIN_MAX', 8.0)))
REALISM_FEATHER  = float(os.environ.get('SYNTHFB_REALISM_FEATHER', 2.0))
REALISM_SOFT     = (float(os.environ.get('SYNTHFB_REALISM_SOFT_MIN', 2.5)),
                    float(os.environ.get('SYNTHFB_REALISM_SOFT_MAX', 3.3)))
# #5 visibility fix: real catheters are FAINT (device-vs-bg contrast ~4 vs synth ~32-55) and
# vary along their length / between instances. DEV_FAINT replaces the single contrast scalar
# with a smooth low-freq per-pixel RETAIN field skewed low (most of the device near-threshold,
# occasional more-visible runs). Still insertion-only (applied within the device mask). 0=old op.
DEV_FAINT        = int(os.environ.get('SYNTHFB_DEV_FAINT', 1)) == 1
DEV_RETAIN       = (float(os.environ.get('SYNTHFB_DEV_RETAIN_MIN', 0.012)),   # base = CVC level (real ~4)
                    float(os.environ.get('SYNTHFB_DEV_RETAIN_MAX', 0.045)))
# Per-class faint LEVEL as a per-INSTANCE (lo,hi) range, SAMPLED per device so a class's instances
# VARY in visibility like real foreign bodies (not all-faint / all-bright). Multiplies the base
# DEV_RETAIN field. Real visibility order (scorecard): NGT faintest, CVC mid, ETT most visible.
DEV_RETAIN_SCALE = {'ng_tube': (0.25, 0.70), 'et_tube_shaft': (1.6, 3.6)}
DEV_SCALE_DEFAULT = (0.55, 1.55)                # CVC/Swan/other lines: mean ~1.0, per-instance varied
# Wide straight tubes the area/bbox<0.15 thinness gate misses but that still need fading (else
# they stay at full un-faded contrast -- scorecard caught ETT at contrast 33).
DEV_WIDE_TUBES   = {'et_tube_shaft', 'dl_et_shaft'}
# Foreign bodies also appear on LATERAL CXRs (the base pool is ~68% lateral). Per-view multiplier
# on the faint level (tunable; lateral devices project through heavier tissue superposition).
DEV_VIEW_SCALE   = {'frontal': 1.0, 'lateral': 1.0, 'unknown': 1.0}
# Catheter/line WIDTH (px @512) + intrinsic CONTRAST (drawn gray), env-tunable. Real RANZCR
# catheters are ~5.6px wide @512 (synth thin-line classes were 1-2px -> the model never saw
# real-width lines); and real catheters vary faint->bright between instances. Widening the line
# and varying the DRAWN gray (NOT the all-faint REALISM post-process, which lost to plain on
# synth->real) keeps the bright high-contrast signal while adding faint-line robustness so faint
# real catheters get traced (recall). numpy randint: high is EXCLUSIVE.
CATH_W    = (int(os.environ.get('SYNTHFB_CATH_W_MIN', 3)),   int(os.environ.get('SYNTHFB_CATH_W_MAX', 8)))    # 3-7px, median ~5 ~= real
CATH_GRAY = (int(os.environ.get('SYNTHFB_CATH_GRAY_MIN', 110)), int(os.environ.get('SYNTHFB_CATH_GRAY_MAX', 256)))  # faint(110) -> bright(255)


def _wind_range(name, lo, hi):
    """Parse a 'lo,hi' env override for a winding param (else default). amp_frac
    (lo,hi)=(0,0) -> straight path (shape change OFF), for regen-validation baselines."""
    v = os.environ.get(name)
    if v:
        try:
            a, b = v.split(','); return (float(a), float(b))
        except Exception:
            pass
    return (lo, hi)


# Swan/CVC path-winding ranges (env-tunable so the regen-validation harness can
# sweep + toggle them). Set *_AF to "0,0" for a straight baseline.
WIND_SWAN_NW = _wind_range('SYNTHFB_WIND_SWAN_NW', 2.0, 2.6)
# AF 0.45-0.60 (was 0.28-0.40): validated optimum from a real-renderer sweep.
# The old default gave swan
# tortuosity ~1.6 (real ~2.40); 0.45-0.60 reaches ~2.16 AND lowers BOTH device-crop
# KID and general-stat dist vs the old default (clean dual-metric win). Higher amp
# (0.60-0.80) saturates tortuosity but regresses general-stat; more waves (NW>2.6)
# overshoots tortuosity and worsens KID -- so bump AF only, leave NW.
WIND_SWAN_AF = _wind_range('SYNTHFB_WIND_SWAN_AF', 0.45, 0.60)
WIND_CVC_NW = _wind_range('SYNTHFB_WIND_CVC_NW', 0.9, 1.4)
WIND_CVC_AF = _wind_range('SYNTHFB_WIND_CVC_AF', 0.12, 0.18)
# Fraction of catheter_tube ("CVC") instances generated as a long non-port central
# line (IJ/subclavian venous entry -> cavoatrial tip) vs a short chest-wall port.
# Real RANZCR CVCs are mostly the former; the port-only default was too short
# (length/diag ~0.20 vs real ~0.47). Env-tunable so regen-validation can A/B it (0=old).
CVC_CENTRAL_P = float(os.environ.get('SYNTHFB_CVC_CENTRAL_P', '0.7'))


def _retain_field(shape, rng, lo, hi):
    """Smooth low-frequency per-pixel 'retain' map skewed toward `lo`: device contrast varies
    along each line (fade in/out) and between devices (per-location), most of it near-threshold
    with occasional more-visible runs -- matching real catheters. Applied ONLY in the device mask."""
    H, W = shape
    sm = np.random.default_rng(rng.randrange(2 ** 31)).random((max(2, H // 48), max(2, W // 48))).astype(np.float32)
    f = np.clip(cv2.resize(sm, (W, H), interpolation=cv2.INTER_CUBIC), 0.0, 1.0) ** 2
    return lo + (hi - lo) * f


def _soft_grain(dev, rng, grain_scale=1.0, soft=True):
    if soft:                                                          # device-local MTF softness
        s = rng.uniform(*REALISM_SOFT)                               # NB: skip on the faint pass -- blurring a
        if s > 0.05:                                                 # faint line pulls the bright SURROUND back
            dev = cv2.GaussianBlur(dev, (0, 0), s)                   # onto it and re-brightens it (floors contrast)
    grain = rng.uniform(*REALISM_GRAIN) * grain_scale                # faint lines: gentler grain (not buried)
    if grain > 0:
        nrng = np.random.default_rng(rng.randrange(2 ** 31))
        dev = dev + nrng.standard_normal(dev.shape).astype(np.float32) * grain
    return dev


def _feather(mask):
    hard = (mask > 0).astype(np.float32)                              # full effect at the core
    if REALISM_FEATHER > 0:                                           # + a soft halo OUTSIDE only
        return np.clip(np.maximum(hard, cv2.GaussianBlur(hard, (0, 0), REALISM_FEATHER)), 0.0, 1.0)
    return hard


def realism_postprocess(rgb, dev_mask, faint_scale, rng):
    """INSERTION-ONLY device realism. The real base is untouched; only device pixels change.
    Opaque devices (pacemakers/wires/valves/ports) get a MILD pop reduction toward a wide local
    blur -- they are genuinely bright on real CXRs. Thin LINES/catheters (faint_scale>0) additionally
    get DEV_FAINT: contrast crushed toward the INPAINTED tissue UNDER the line (a blur of g would
    include the line's own brightness and floor the contrast, so it could never go faint), via a
    low + spatially-varying retain field scaled PER CLASS (faint_scale, e.g. NGT fainter, ETT more
    visible) -> real catheters are near-threshold and fade along their length. No-op when
    SYNTHFB_REALISM is off / no devices / prob (1 - REALISM_PROB)."""
    if not REALISM_ENABLE or dev_mask is None or not dev_mask.any() or rng.random() > REALISM_PROB:
        return rgb
    g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    local_bg = cv2.GaussianBlur(g, (0, 0), 15.0)
    dev = _soft_grain(local_bg + rng.uniform(*REALISM_CONTRAST) * (g - local_bg), rng)
    w = _feather(dev_mask)
    out = g * (1.0 - w) + dev * w
    fm = (faint_scale > 0) if faint_scale is not None else None
    if DEV_FAINT and fm is not None and fm.any():
        m = cv2.dilate(fm.astype(np.uint8), np.ones((3, 3), np.uint8))
        tissue = cv2.inpaint(np.clip(g, 0, 255).astype(np.uint8), m, 4, cv2.INPAINT_TELEA).astype(np.float32)
        field = np.clip(_retain_field(g.shape, rng, *DEV_RETAIN) * faint_scale, 0.0, 0.9)   # per-class retain
        _anat = float(os.environ.get('SYNTHFB_DEV_ANAT_CONTRAST', '0'))   # pop over lung, fade over dense tissue
        if _anat > 0:
            bg = cv2.GaussianBlur(g, (0, 0), 25.0)
            field = np.clip(field * np.clip(1.0 + _anat * (0.5 - bg / 255.0), 0.3, 2.0), 0.0, 0.9)
        dev_f = _soft_grain(tissue + field * (g - tissue), rng, grain_scale=0.22, soft=False)
        _fsoft = float(os.environ.get('SYNTHFB_DEV_FAINT_SOFT', '0'))   # detector-PSF blur (sharpness axis)
        if _fsoft > 0:
            dev_f = cv2.GaussianBlur(dev_f, (0, 0), _fsoft)
        wf = _feather(fm.astype(np.uint8))
        out = out * (1.0 - wf) + dev_f * wf
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

# Source subset built by prepare_mimic_subset.py:
#   filter dataset_name==MIMIC, Mode==train, Support_Devices==0
#   -> ~4.7k FB-free training CXRs (matches paper's 4,769 selection)
print(f'MIMIC_ROOT={MIMIC_ROOT}')
print(f'SUBSET_CSV={MIMIC_SUBSET_CSV} (exists={MIMIC_SUBSET_CSV.exists()})')
print(f'OUT_DIR={OUT_DIR}  N_IMAGES={N_IMAGES}  MAX_ANN={MAX_ANNOTATIONS}  SEED={RNG_SEED}')

# torch>=2.6 defaults weights_only=True for torch.load. cxas checkpoint contains
# argparse.Namespace -> UnpicklingError. Patch back to weights_only=False (cxas
# weights are from official upstream gdrive, safe to load).
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from cxas import CXAS
from cxas import label_mapper

cxas_model = CXAS(gpus='cpu' if not torch.cuda.is_available() else '0')

import hashlib

# Cache for cxas anatomy masks. Same source CXR → same (159, H, W) mask, so we
# only need to run cxas once per unique image path. Critical when generating with
# replacement (same source reused many times) or iterating on synthesis params
# without changing the underlying anatomy. Big speedup, especially on CPU.
#
# Storage: per-image .npz with packbits-compressed masks + uint8 RGB (~250 KB each).
# Path: $SYNTHFB_CXAS_CACHE_DIR (default: <SynthFB.py dir>/.cache_cxas — shared
# across all OUT_DIRs since the underlying MIMIC images don't change).
# Disable by setting SYNTHFB_CXAS_CACHE=0.
_DEFAULT_CXAS_CACHE = Path(__file__).resolve().parent / '.cache_cxas'
_CXAS_CACHE_DIR = Path(os.environ.get('SYNTHFB_CXAS_CACHE_DIR', str(_DEFAULT_CXAS_CACHE)))
_CXAS_CACHE_ENABLED = int(os.environ.get('SYNTHFB_CXAS_CACHE', '1')) == 1
if _CXAS_CACHE_ENABLED:
    _CXAS_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_CXAS_CACHE_HITS = 0
_CXAS_CACHE_MISSES = 0


def _cxas_cache_path(image_path) -> Path:
    """Stable cache filename derived from absolute image path."""
    abs_p = str(Path(image_path).resolve())
    h = hashlib.sha1(abs_p.encode('utf-8')).hexdigest()[:20]
    return _CXAS_CACHE_DIR / f'{h}.npz'


def get_anatomy(model, image_path):
    """Return (img RGB uint8 HxWx3, masks bool (159,H,W)). Cached on disk."""
    global _CXAS_CACHE_HITS, _CXAS_CACHE_MISSES
    if _CXAS_CACHE_ENABLED:
        cache_p = _cxas_cache_path(image_path)
        if cache_p.exists():
            try:
                data = np.load(str(cache_p))
                img = data['img']
                packed = data['masks_packed']
                shape = tuple(int(s) for s in data['shape'])
                bits = np.unpackbits(packed, count=int(np.prod(shape)))
                masks = bits.reshape(shape).astype(bool)
                _CXAS_CACHE_HITS += 1
                if (_CXAS_CACHE_HITS + _CXAS_CACHE_MISSES) % 50 == 0:
                    total = _CXAS_CACHE_HITS + _CXAS_CACHE_MISSES
                    print(f'  [cxas cache] {_CXAS_CACHE_HITS}/{total} hits '
                          f'({100*_CXAS_CACHE_HITS/total:.0f}%)')
                return img, masks
            except Exception as e:
                print(f'  [cxas cache] failed to load {cache_p}: {e} — re-running cxas')

    fd = model.fileloader.load_file(str(image_path))
    fd['filename']  = [fd['filename']]
    fd['file_size'] = [fd['file_size']]
    with torch.no_grad():
        pr = model.model(fd)
    masks = pr['segmentation_preds'][0].cpu().numpy().astype(bool)
    img   = pr['orig_data'].transpose(1, 2, 0).astype(np.uint8)

    if _CXAS_CACHE_ENABLED:
        _CXAS_CACHE_MISSES += 1
        try:
            packed = np.packbits(masks.astype(np.uint8))
            np.savez_compressed(
                str(_cxas_cache_path(image_path)),
                img=img.astype(np.uint8),
                masks_packed=packed,
                shape=np.array(masks.shape, dtype=np.int32),
            )
        except Exception as e:
            print(f'  [cxas cache] failed to save: {e}')
    return img, masks

# label name -> id
label2id = {v: int(k) for k, v in label_mapper.id2label_dict.items()}

def group_ids(names):
    return [label2id[n] for n in names if n in label2id]

def safe_attr(mod, name):
    v = getattr(mod, name, None)
    return list(v) if isinstance(v, (list, tuple, set)) else []

ANAT_GROUPS = {
    'vessels':  group_ids(safe_attr(label_mapper, 'vessels')),
    'trachea':  group_ids(safe_attr(label_mapper, 'trachea')),
    'heart':    group_ids(safe_attr(label_mapper, 'heart')),
    'humerus':  group_ids(safe_attr(label_mapper, 'humerus')),
    'lungs':    group_ids(safe_attr(label_mapper, 'lung_halves')),
    'aorta':    group_ids(['aortic arch', 'ascending aorta', 'descending aorta']),
    'ribs':     group_ids(safe_attr(label_mapper, 'rib')),
    'clavicle': group_ids(safe_attr(label_mapper, 'clavicle_set')),
}
{k: len(v) for k, v in ANAT_GROUPS.items()}

def union_mask(masks, ids):
    if not ids:
        return np.zeros(masks.shape[1:], bool)
    valid = [i for i in ids if i < masks.shape[0]]
    if not valid:
        return np.zeros(masks.shape[1:], bool)
    return np.any(masks[valid], axis=0)

def sample_point_in_mask(mask, rng):
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    i = rng.randrange(len(ys))
    return int(xs[i]), int(ys[i])


_ZERO_ANAT_CACHE = {}


def _ablate_anat(anat_masks, rng=None):
    """Placement-ablation transform of the anatomy masks fed to one shape/device rule (called per instance/device):
    'anatomy' -> unchanged; 'fallback' -> all-zero (cached) twin so the rule hits its geometric default; 'random'
    -> the whole stack rigidly shifted a random (dx, dy) (<=35% of each dim, zero-filled) so the anatomy stays
    internally coherent (multi-part devices intact) but placement is decorrelated from the real patient."""
    if PLACEMENT_MODE == 'anatomy':
        return anat_masks
    if PLACEMENT_MODE == 'fallback':
        z = _ZERO_ANAT_CACHE.get(anat_masks.shape)
        if z is None:
            z = np.zeros_like(anat_masks)
            _ZERO_ANAT_CACHE[anat_masks.shape] = z
        return z
    # random_body (mild relocation -> stays on-patient) / random_general (large -> anywhere in the frame)
    frac = {'random_body': 0.10, 'random_general': 0.40, 'random': 0.10}.get(PLACEMENT_MODE, 0.10)
    _, H, W = anat_masks.shape
    r = rng if rng is not None else random
    dy = r.randint(-int(frac * H), int(frac * H)); dx = r.randint(-int(frac * W), int(frac * W))
    out = np.zeros_like(anat_masks)
    out[:, max(0, dy):min(H, H + dy), max(0, dx):min(W, W + dx)] = \
        anat_masks[:, max(0, -dy):min(H, H - dy), max(0, -dx):min(W, W - dx)]
    return out


_CLASS_FRAC = float(os.environ.get('SYNTHFB_CLASS_FRAC', 1.0))


def _subset_rules(shapes, devices):
    """Generation-diversity ablation: keep a deterministic (seeded) SYNTHFB_CLASS_FRAC of the shape+device rule
    TYPES (per-image instance count unchanged) -> isolates how many DISTINCT generation types exist."""
    if _CLASS_FRAC >= 1.0:
        return shapes, devices
    allnames = sorted(set(shapes) | set(devices))
    keep = set(random.Random(RNG_SEED + 12345).sample(allnames, max(1, int(round(len(allnames) * _CLASS_FRAC)))))
    return type(shapes)(n for n in shapes if n in keep), type(devices)(n for n in devices if n in keep)

import pandas as pd

def load_subset_paths(csv_path, img_root, n, seed=0, split=None):
    """Read filtered CSV; return list of n absolute image paths.

    - If `split` is given AND CSV has 'split' column, filter to that split first.
    - If n > pool_size, sample WITH replacement (each source image used multiple
      times with different random insertion outcomes).
    - If n <= pool_size, shuffle + take first n (no replacement).
    """
    df = pd.read_csv(csv_path)
    if 'full_path' not in df.columns:
        df['full_path'] = df['img_path'].apply(lambda p: os.path.join(str(img_root), p))
    if split is not None and 'split' in df.columns:
        df = df[df['split'] == split]
        if len(df) == 0:
            raise SystemExit(f"split='{split}' has 0 rows in {csv_path}")
    # Verify files on disk first (skip missing)
    pool = [Path(p) for p in df['full_path'] if os.path.isfile(p)]
    if not pool:
        raise SystemExit(f'no files on disk for split={split}')
    rng_local = random.Random(seed)
    if n <= len(pool):
        rng_local.shuffle(pool)
        return pool[:n]
    # n > pool: sample with replacement
    return [rng_local.choice(pool) for _ in range(n)]

# Optional split filter from env (set by generate_synthfb.sh --split arg)
_SPLIT = os.environ.get('SYNTHFB_SPLIT') or None
if _SPLIT == '':
    _SPLIT = None

if MIMIC_SUBSET_CSV.exists():
    sample_paths = load_subset_paths(MIMIC_SUBSET_CSV, MIMIC_ROOT, N_IMAGES,
                                     seed=RNG_SEED, split=_SPLIT)
    print(f'loaded {len(sample_paths)} paths from subset CSV ({MIMIC_SUBSET_CSV.name})'
          + (f" [split={_SPLIT}]" if _SPLIT else ''))
elif int(os.environ.get('SYNTHFB_ALLOW_GLOB_FALLBACK', '0')) == 1:
    # Opt-in fallback: glob any MIMIC image (NO FB filtering — strictly debug/smoke use).
    # Off by default because the random sample bypasses every cleaning step (NLP gate,
    # image-side purity classifier, view gate) and silently breaks reproducibility.
    def list_mimic_images(root, n, seed=0):
        rng = random.Random(seed)
        paths = list(root.glob('p*/s*/*.jpg'))
        rng.shuffle(paths)
        return paths[:n]
    sample_paths = list_mimic_images(MIMIC_ROOT, N_IMAGES, seed=RNG_SEED)
    print(f'WARNING: SYNTHFB_ALLOW_GLOB_FALLBACK=1 — sampling {len(sample_paths)} '
          f'random MIMIC images, NO FB filtering applied.')
else:
    raise SystemExit(
        f'ERROR: subset CSV not found at {MIMIC_SUBSET_CSV}\n'
        f'  Either:\n'
        f'    - run prepare_mimic_subset.py to build it, or\n'
        f'    - point MIMIC_SUBSET_CSV / generate_synthfb.sh --subset-csv at an existing one, or\n'
        f'    - set SYNTHFB_ALLOW_GLOB_FALLBACK=1 to sample random MIMIC (smoke test only).'
    )

demo_img, demo_anat = get_anatomy(cxas_model, sample_paths[0])
print('image:', demo_img.shape, demo_img.dtype, '| anatomy masks:', demo_anat.shape, demo_anat.dtype)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(demo_img); axes[0].set_title('CXR'); axes[0].axis('off')
for ax, group in zip(axes[1:], ['heart', 'trachea', 'vessels']):
    m = union_mask(demo_anat, ANAT_GROUPS[group])
    ax.imshow(demo_img); ax.imshow(m, alpha=0.5, cmap='Reds')
    ax.set_title(group); ax.axis('off')
plt.tight_layout(); plt.show()

def bezier_points(p0, p1, p2, p3, n=128):
    t = np.linspace(0, 1, n)[:, None]
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return pts.astype(np.int32)


def jitter_path(pts, rng, amp_px=0.8, wavelength_pts=14):
    """Add smooth low-frequency noise perpendicular to the path direction.
    Real catheters/wires aren't math-perfect curves — small kinks from anatomic
    contact, surgeon-twist, and acquisition geometry give them organic irregularity.
    `amp_px`: peak displacement in pixels. `wavelength_pts`: smoothness (higher = smoother).
    """
    if len(pts) < 4 or amp_px <= 0:
        return pts.astype(np.int32)
    n = len(pts)
    # smooth gaussian noise via cumulative sum
    raw = np.array([rng.gauss(0, 1) for _ in range(n)], dtype=float)
    k = max(3, wavelength_pts)
    kernel = np.ones(k) / k
    smooth = np.convolve(raw, kernel, mode='same')
    smooth = smooth * (amp_px / (np.abs(smooth).max() + 1e-6))
    # perpendicular unit vector at each point
    tan = np.gradient(pts.astype(float), axis=0)
    norm = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    norm /= (np.linalg.norm(norm, axis=1, keepdims=True) + 1e-6)
    return (pts.astype(float) + norm * smooth[:, None]).astype(np.int32)


def winding_path(p0, p3, rng, n=128, n_waves=1.5, amp_frac=0.18, jitter=0.6):
    """Tortuous path p0->p3: end-tapered sinusoidal perpendicular waves + fine jitter.
    n_waves / amp_frac raise tortuosity (path length / endpoint distance) to mimic the
    winding course of heart-traversing catheters (Swan-Ganz, CVC) -- a single bezier
    stays near-straight (tortuosity ~1, the synth-vs-real shape gap)."""
    p0 = np.asarray(p0, float); p3 = np.asarray(p3, float)
    d = p3 - p0; L = float(np.hypot(d[0], d[1])) + 1e-6
    t = np.linspace(0.0, 1.0, n)
    base = p0[None, :] + t[:, None] * d[None, :]
    perp = np.array([-d[1], d[0]]) / L
    wave = np.sin(t * np.pi * 2.0 * n_waves + rng.uniform(0, np.pi)) * np.sin(t * np.pi)  # 0 at both ends
    amp = amp_frac * L * rng.uniform(0.85, 1.15)
    pts = (base + (wave[:, None] * amp) * perp[None, :]).astype(np.int32)
    return jitter_path(pts, rng, amp_px=jitter, wavelength_pts=12)


def extend_tip_oob(pts, H, W, rng, direction='bottom', length_px=None):
    """Append a tail past the existing tip pts[-1] so the tube exits the image
    edge. Used to model the (very common on chest CXRs) case where the NG/PEG
    tip is below the diaphragm but cropped out of frame, or a misplaced PICC/
    chest_tube projects past the top edge.

    Side effect on COCO output: tip kpt (last point) is OOB so
    _flatten_keypoints automatically sets its visibility=0 — no separate
    OOB-handling needed downstream.

    `direction`: 'bottom' | 'top' | 'left' | 'right'
    """
    if length_px is None:
        length_px = rng.randint(40, 110)
    last = pts[-1].astype(float)
    if direction == 'bottom':
        target = np.array([last[0] + rng.uniform(-15, 15), H + length_px], float)
    elif direction == 'top':
        target = np.array([last[0] + rng.uniform(-15, 15), -length_px], float)
    elif direction == 'left':
        target = np.array([-length_px, last[1] + rng.uniform(-15, 15)], float)
    else:  # right
        target = np.array([W + length_px, last[1] + rng.uniform(-15, 15)], float)
    n_extra = max(6, length_px // 8)
    tail = np.linspace(last, target, num=n_extra)[1:]
    return np.concatenate([pts, tail.astype(pts.dtype)], axis=0)


def extend_tube_externally(pts, H, W, rng, direction='top', length_px=None,
                           lateral_drift=0.0):
    """Prepend an external-portion segment beyond the image edge to show the
    proximal portion of a tube/line (e.g. ET tube above mouth, NG tube from nostril,
    CVL from skin entry). Real CXRs almost always show a few cm of catheter outside
    the body — synth devices that start at an internal anchor look truncated.

    `direction`:
      'top'           — exits straight up (mouth/oral ET tube)
      'top_nostril'   — exits up + slight lateral offset (NG/orogastric tube)
      'lateral_left'  / 'lateral_right' — exits side (chest tube, dialysis CVC)
      'top_lateral'   — exits upper corner (subclavian/IJ CVL skin entry)
    `length_px`: how far past image edge to extend. Default H/4.
    Returns concatenated path: [external_segment ... original_pts].
    """
    if length_px is None:
        length_px = H // 4
    p_in = pts[0].astype(float)
    if direction == 'top':
        p_out = np.array([p_in[0] + lateral_drift * rng.uniform(-1, 1), -length_px])
    elif direction == 'top_nostril':
        p_out = np.array([p_in[0] + rng.uniform(15, 45) * rng.choice([-1, 1]),
                          -length_px])
    elif direction == 'lateral_left':
        p_out = np.array([-length_px, p_in[1] + rng.uniform(-30, 30)])
    elif direction == 'lateral_right':
        p_out = np.array([W + length_px, p_in[1] + rng.uniform(-30, 30)])
    elif direction == 'top_lateral_left':
        p_out = np.array([rng.uniform(0, W * 0.2), -length_px])
    elif direction == 'top_lateral_right':
        p_out = np.array([rng.uniform(W * 0.8, W), -length_px])
    else:
        return pts.astype(np.int32)
    n_seg = max(8, length_px // 6)
    ext = np.linspace(p_out, p_in, num=n_seg, endpoint=False).astype(np.int32)
    return np.vstack([ext, pts.astype(np.int32)])


def render_3d_cuff(mask, color_map, cx, cy, rx, ry, *, angle=0,
                   rim_intensity=240, interior_intensity=130, rim_thickness=2):
    """Render a tube cuff as bright rim + slightly darker air-filled interior.
    Real ET-tube / trach cuff is inflated — appears as a wider segment on X-ray with
    bright radiopaque cuff walls and lower-density air-filled interior. Previous code
    drew a flat filled ellipse with uniform color, which reads as 'pasted'.
    Mutates mask + color_map in place.
    """
    rim_intensity = int(np.clip(rim_intensity, 0, 255))
    interior_intensity = int(np.clip(interior_intensity, 0, 255))
    cv2.ellipse(mask,      (cx, cy), (rx, ry), angle, 0, 360, 255, -1, cv2.LINE_AA)
    cv2.ellipse(color_map, (cx, cy), (rx, ry), angle, 0, 360, interior_intensity,
                -1, cv2.LINE_AA)
    cv2.ellipse(color_map, (cx, cy), (rx, ry), angle, 0, 360, rim_intensity,
                rim_thickness, cv2.LINE_AA)

def random_outside_point(H, W, rng):
    side = rng.choice(['l', 'r', 't', 'b'])
    if side == 'l': return (-rng.randint(0, 20), rng.randint(0, H - 1))
    if side == 'r': return (W + rng.randint(0, 20), rng.randint(0, H - 1))
    if side == 't': return (rng.randint(0, W - 1), -rng.randint(0, 20))
    return (rng.randint(0, W - 1), H + rng.randint(0, 20))

def _sample_text_angle(rng):
    """Per-instance text rotation: 60% axis-aligned, 30% slight tilt ±15°, 10% strong ±90°."""
    r = rng.random()
    if r < 0.60: return 0.0
    if r < 0.90: return rng.uniform(-15, 15)
    return rng.uniform(-90, 90)

def _render_rotated_text(chars, font, scale, thickness, angle):
    """Rasterize text into a small mask, rotate via warpAffine, crop to non-zero bbox.
    Returns (mask uint8, w, h). Empty (zeros) on failure."""
    (tw, th), baseline = cv2.getTextSize(chars, font, scale, thickness)
    pad = max(4, int(0.2 * max(tw, th)))
    canvas_w, canvas_h = tw + 2 * pad, th + baseline + 2 * pad
    canvas = np.zeros((canvas_h, canvas_w), np.uint8)
    cv2.putText(canvas, chars, (pad, pad + th), font, scale, 255, thickness, cv2.LINE_AA)
    if angle != 0.0:
        # rotate w/ bounding canvas resize so chars aren't clipped
        cos_a = abs(math.cos(math.radians(angle)))
        sin_a = abs(math.sin(math.radians(angle)))
        new_w = int(canvas_h * sin_a + canvas_w * cos_a) + 2
        new_h = int(canvas_h * cos_a + canvas_w * sin_a) + 2
        M = cv2.getRotationMatrix2D((canvas_w / 2, canvas_h / 2), angle, 1.0)
        M[0, 2] += (new_w - canvas_w) / 2
        M[1, 2] += (new_h - canvas_h) / 2
        canvas = cv2.warpAffine(canvas, M, (new_w, new_h), flags=cv2.INTER_LINEAR,
                                borderValue=0)
    ys, xs = np.where(canvas > 0)
    if len(ys) == 0:
        return canvas, canvas.shape[1], canvas.shape[0]
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    return canvas[y0:y1, x0:x1].copy(), x1 - x0, y1 - y0

# --- 1. Textual ---
def plot_textual(img, anat_masks, rng):
    H, W = img.shape[:2]
    out, mask = img.copy(), np.zeros((H, W), np.uint8)
    chars = ''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-+', k=rng.randint(1, 6)))
    scale = rng.uniform(0.6, 2.2); thickness = rng.randint(1, 3)
    font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX,
                       cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX])
    angle = _sample_text_angle(rng)
    glyph, tw, th = _render_rotated_text(chars, font, scale, thickness, angle)
    if tw == 0 or th == 0 or tw >= W or th >= H:
        return out, mask
    x = rng.randint(0, max(W - tw, 1)); y = rng.randint(0, max(H - th, 1))
    mask[y:y + th, x:x + tw] = np.maximum(mask[y:y + th, x:x + tw], glyph)
    mode = rng.choice(['dark', 'bright', 'transparent'])
    if mode == 'dark':      out[mask > 0] = 0
    elif mode == 'bright':  out[mask > 0] = 255
    else:
        a = rng.uniform(0.2, 0.5)
        blended = out[mask > 0].astype(np.float32) * (1 - a) + 255 * a
        out[mask > 0] = np.clip(blended, 0, 255).astype(np.uint8)
    return out, mask

# --- 2. Circular ---
def plot_circular(img, anat_masks, rng):
    H, W = img.shape[:2]; out, mask = img.copy(), np.zeros((H, W), np.uint8)
    target = union_mask(anat_masks, ANAT_GROUPS['heart'] + ANAT_GROUPS['humerus'])
    pt = sample_point_in_mask(target, rng) or (rng.randint(0, W - 1), rng.randint(0, H - 1))
    ax = rng.randint(8, 40); by = rng.randint(8, 40); angle = rng.uniform(0, 360)
    gray = rng.randint(180, 255)
    cv2.ellipse(out,  pt, (ax, by), angle, 0, 360, (gray,) * 3, -1)
    cv2.ellipse(mask, pt, (ax, by), angle, 0, 360, 255, -1)
    return out, mask

# --- 3. Ring ---
def plot_ring(img, anat_masks, rng):
    H, W = img.shape[:2]; out, mask = img.copy(), np.zeros((H, W), np.uint8)
    target = union_mask(anat_masks, ANAT_GROUPS['heart'] + ANAT_GROUPS['aorta'])
    pt = sample_point_in_mask(target, rng) or (W // 2, H // 2)
    ax = rng.randint(15, 50); by = rng.randint(15, 50)
    angle = rng.uniform(0, 360); t = rng.randint(2, 5); gray = rng.randint(200, 255)
    cv2.ellipse(out,  pt, (ax, by), angle, 0, 360, (gray,) * 3, t)
    cv2.ellipse(mask, pt, (ax, by), angle, 0, 360, 255, t)
    return out, mask

# --- 4. Rectangular ---
def plot_rect(img, anat_masks, rng):
    H, W = img.shape[:2]; out, mask = img.copy(), np.zeros((H, W), np.uint8)
    w = rng.randint(15, 60); h = rng.randint(15, 60)
    x = rng.randint(0, W - w); y = rng.randint(0, H - h)
    gray = rng.randint(180, 255); opacity = rng.uniform(0.55, 1.0)
    overlay = out.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (gray,) * 3, -1)
    out = cv2.addWeighted(overlay, opacity, out, 1 - opacity, 0)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return out, mask

# --- 5. Clip ---
def plot_clip(img, anat_masks, rng):
    H, W = img.shape[:2]; out, mask = img.copy(), np.zeros((H, W), np.uint8)
    rib_u = union_mask(anat_masks, ANAT_GROUPS['ribs'])
    target = rib_u if rib_u.any() else anat_masks.any(axis=0)
    pt = sample_point_in_mask(target, rng) or (rng.randint(0, W - 1), rng.randint(0, H - 1))
    L = rng.randint(8, 22); t = rng.randint(2, 4); a = rng.uniform(0, math.pi)
    dx, dy = int(L * math.cos(a)), int(L * math.sin(a))
    p1 = (pt[0] - dx, pt[1] - dy); p2 = (pt[0] + dx, pt[1] + dy)
    cv2.line(out,  p1, p2, (240, 240, 240), t, cv2.LINE_AA)
    cv2.line(mask, p1, p2, 255, t, cv2.LINE_AA)
    return out, mask

# --- 6. Grid ---
def plot_grid(img, anat_masks, rng):
    H, W = img.shape[:2]; out, mask = img.copy(), np.zeros((H, W), np.uint8)
    u = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    if not u.any(): u = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    if not u.any(): return out, mask
    ys, xs = np.where(u)
    y0, y1 = int(ys.min()), int(ys.max()); x0, x1 = int(xs.min()), int(xs.max())
    sub_h = rng.randint(20, max(30, (y1 - y0) // 2 + 30))
    sub_w = rng.randint(10, max(20, (x1 - x0) // 2 + 20))
    cy = rng.randint(y0, y1); cx = rng.randint(x0, x1)
    yA = max(0, cy - sub_h // 2); yB = min(H, yA + sub_h)
    xA = max(0, cx - sub_w // 2); xB = min(W, xA + sub_w)
    cell = rng.randint(4, 8)
    for yi in range(yA, yB, cell):
        for xi in range(xA, xB, cell):
            if yi >= H or xi >= W or not u[yi, xi]: continue
            g = rng.randint(180, 255)
            if rng.random() < 0.7:
                xj = min(xi + cell, xB - 1)
                cv2.line(out,  (xi, yi), (xj, yi), (g, g, g), 1)
                cv2.line(mask, (xi, yi), (xj, yi), 255, 1)
            if rng.random() < 0.7:
                yj = min(yi + cell, yB - 1)
                cv2.line(out,  (xi, yi), (xi, yj), (g, g, g), 1)
                cv2.line(mask, (xi, yi), (xi, yj), 255, 1)
    return out, mask

# --- 7. Line ---
def plot_line(img, anat_masks, rng):
    H, W = img.shape[:2]; out, mask = img.copy(), np.zeros((H, W), np.uint8)
    p0 = np.array(random_outside_point(H, W, rng), float)
    target = union_mask(anat_masks, ANAT_GROUPS['vessels'] + ANAT_GROUPS['heart'])
    end = sample_point_in_mask(target, rng) or (rng.randint(0, W - 1), rng.randint(0, H - 1))
    p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-80, 80), rng.randint(-80, 80)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-80, 80), rng.randint(-80, 80)], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    t = rng.randint(1, 3); gray = rng.randint(200, 255)
    cv2.polylines(out,  [pts], False, (gray,) * 3, t, cv2.LINE_AA)
    cv2.polylines(mask, [pts], False, 255, t, cv2.LINE_AA)
    return out, mask

# --- 8. Parallel ---
def plot_parallel(img, anat_masks, rng):
    H, W = img.shape[:2]; out, mask = img.copy(), np.zeros((H, W), np.uint8)
    target = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    end = sample_point_in_mask(target, rng) if target.any() else (W // 2, H // 2)
    p0 = np.array(random_outside_point(H, W, rng), float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-40, 40), rng.randint(-40, 40)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-40, 40), rng.randint(-40, 40)], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    w = rng.randint(4, 10)
    tangents = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left  = (pts[:-1] + norms * w).astype(np.int32)
    right = (pts[:-1] - norms * w).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    inner = rng.randint(40, 120)
    cv2.fillPoly(out, [poly], (inner,) * 3); cv2.fillPoly(mask, [poly], 255)
    edge = rng.randint(200, 255)
    cv2.polylines(out, [left], False, (edge,) * 3, 1, cv2.LINE_AA)
    cv2.polylines(out, [right], False, (edge,) * 3, 1, cv2.LINE_AA)
    return out, mask

STRUCT_FNS = {
    'textual': plot_textual, 'circular': plot_circular, 'ring': plot_ring,
    'rect': plot_rect, 'clip': plot_clip, 'grid': plot_grid,
    'line': plot_line, 'parallel': plot_parallel,
}
print('paper primitives:', list(STRUCT_FNS))

demo_rng = random.Random(42)
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for ax, (name, fn) in zip(axes.flat, STRUCT_FNS.items()):
    out, m = fn(demo_img.copy(), demo_anat, demo_rng)
    ax.imshow(out); ax.imshow(m, alpha=0.4, cmap='Reds')
    ax.set_title(name); ax.axis('off')
plt.suptitle('Paper baseline — 8 primitives (overlay rendering)'); plt.tight_layout(); plt.show()

def load_cutouts(root):
    out = {}
    if not root.exists(): return out
    for cat in sorted(os.listdir(root)):
        d = root / cat
        if not d.is_dir(): continue
        files = sorted([p for p in d.glob('*.png') if not p.name.startswith('.')])
        if files: out[cat] = [str(p) for p in files]
    return out

# Per-cutout metadata, keyed by absolute PNG path. Loaded from the optional
# cutouts/book_manifest.json (schema documented in cutouts/labels/README.md).
# When a cutpaste annotation lands and its source path matches an entry here,
# we override the generic 'cutpaste' label with the canonical SynthFB class
# and attach the entry's provenance fields to the COCO output.
def load_cutout_metadata(manifest_path):
    out = {}
    if not manifest_path.exists():
        return out
    import json as _json
    with open(manifest_path, encoding='utf-8') as f:
        manifest = _json.load(f)
    base = manifest_path.parent
    for entry in manifest:
        abs_path = str((base / entry['rel_path']).resolve())
        out[abs_path] = entry
    return out

CUTOUTS = load_cutouts(CUTOUTS_ROOT)
CUTOUT_META = load_cutout_metadata(CUTOUTS_ROOT.parent / 'book_manifest.json')
print('cutout categories:', {k: len(v) for k, v in CUTOUTS.items()})
print(f'cutout metadata entries: {len(CUTOUT_META)}')

def load_rgba(path):
    im = Image.open(path).convert('RGBA')
    return np.array(im, dtype=np.uint8)

def weak_augment_rgba(rgba, rng):
    h, w = rgba.shape[:2]
    s = rng.uniform(0.6, 1.4)
    nh, nw = max(8, int(h * s)), max(8, int(w * s))
    img = cv2.resize(rgba, (nw, nh), interpolation=cv2.INTER_LINEAR)
    if rng.random() < 0.5: img = cv2.flip(img, 1)
    angle = rng.uniform(-30, 30)
    M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (nw, nh), borderValue=(0, 0, 0, 0))
    b = rng.uniform(0.8, 1.2)
    img[..., :3] = np.clip(img[..., :3].astype(float) * b, 0, 255).astype(np.uint8)
    return img

_EDGE_GAMMA = 1.15  # >1 sharpens AA edge. 1.15 keeps a touch of focal-spot penumbra (realism)
                    # without ghost-halo. Earlier value 1.4 was too aggressive — synth objects
                    # looked pasted with hyper-crisp edges that don't match real X-ray geometry.

def render_overlay(img, mask, color, rng):
    """Alpha-aware composite w/ gamma-sharpened edges.

    Plain `alpha = mask/255` produces a soft halo around inserts (AA fragments composite
    at 0.3-0.7 alpha → visible blur ring). Real CXR wires are sharp 1-2px bright lines
    with no fuzzy rim. Gamma 1.8 collapses partial-alpha pixels toward 0, keeping high-
    alpha interior at ~1.0. Result: crisp edges, no ghost halo, no jagged stair steps.
    """
    if not mask.any():
        return img.copy()
    alpha = ((mask.astype(np.float32) / 255.0) ** _EDGE_GAMMA)[..., None]
    if isinstance(color, np.ndarray) and color.ndim == 2:
        layer = np.stack([color, color, color], axis=-1).astype(np.float32)
        layer = np.clip(layer, 0, 255)
    else:
        c = int(np.clip(color, 0, 255))
        layer = np.full_like(img, c, dtype=np.uint8).astype(np.float32)
    out = alpha * layer + (1.0 - alpha) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def render_soft_edge(img, mask, color, rng):
    """Gaussian-feathered alpha composite w/ gamma sharpening.

    Sigma kept small (0.3-1.2) so soft_edge mode adds minimal blur — real CXR objects
    are sharp. Gamma collapses partial-alpha pixels toward 0 to kill ghost halo.
    """
    sigma = rng.uniform(0.3, 1.2)
    src_alpha = mask.astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(src_alpha, (0, 0), sigma)
    soft = np.clip(soft, 0.0, 1.0) ** _EDGE_GAMMA
    soft = soft[..., None]
    if isinstance(color, np.ndarray) and color.ndim == 2:
        layer = np.clip(color, 0, 255).astype(np.float32)
        layer = np.stack([layer, layer, layer], axis=-1)
    else:
        layer = np.full_like(img, int(np.clip(color, 0, 255)), dtype=np.uint8).astype(np.float32)
    out = soft * layer + (1.0 - soft) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def render_beer_lambert(img, mask, color, rng):
    """I_out = I_in * exp(-mu * thickness). Thickness from distance transform.
    Multiplies final effect by mask alpha (mask/255) so AA edges blend cleanly.
    """
    if not mask.any(): return img.copy()
    binm = (mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(binm, cv2.DIST_L2, 3)
    thickness = (dt / (dt.max() + 1e-6)) ** 0.5
    mu = rng.uniform(0.5, 3.5)
    # Accept either scalar or per-pixel 2D color map (e.g. textured wire). For the
    # `c > 128` branch we collapse to mean — only affects the attenuation polarity.
    if isinstance(color, np.ndarray) and color.ndim == 2:
        c_arr = np.clip(color, 0, 255).astype(np.float32)
        c_for_effect = c_arr
        c_for_branch = float(c_arr[mask > 0].mean()) if (mask > 0).any() else 128.0
    else:
        c_scalar = float(np.clip(color, 0, 255))
        c_for_effect = c_scalar
        c_for_branch = c_scalar
    src_alpha = ((mask.astype(np.float32) / 255.0) ** _EDGE_GAMMA)[..., None]
    if c_for_branch > 128:
        atten = np.exp(-mu * thickness)[..., None]
    else:
        atten = np.exp(-mu * 2.0 * thickness)[..., None]
    if isinstance(c_for_effect, np.ndarray):
        c_layer = np.stack([c_for_effect, c_for_effect, c_for_effect], axis=-1)
    else:
        c_layer = c_for_effect
    effect = img.astype(np.float32) * atten + (1.0 - atten) * c_layer
    out = src_alpha * effect + (1.0 - src_alpha) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def render_occluded(img, mask, color, rng, anat_masks=None):
    """Soft-edge composite, then darken FB pixels behind bone (alpha-weighted to avoid hard step)."""
    out = render_soft_edge(img, mask, color, rng)
    if anat_masks is None: return out
    occ = union_mask(anat_masks, ANAT_GROUPS['ribs'] + ANAT_GROUPS['clavicle'])
    if not occ.any(): return out
    f = rng.uniform(0.55, 0.85)
    # weight occlusion darkening by mask alpha so it fades at edges instead of stepping
    occ_alpha = (occ.astype(np.float32) * mask.astype(np.float32) / 255.0)[..., None]
    out_f = out.astype(np.float32)
    out_f = out_f * (1.0 - occ_alpha * (1.0 - f))
    return np.clip(out_f, 0, 255).astype(np.uint8)

def render_translucent_implant(img, mask, color, rng, alpha_core=None):
    """Translucent implant body: bright rim + see-through interior.

    Distance-transform-derived alpha: 1.0 at rim, alpha_core (~0.35-0.75) at deepest interior.
    Anatomy bleeds through the body while contour stays sharp/bright — matches modern CXR
    appearance of pacemakers / ICDs / implants (Image #3 reference).
    `color` may be scalar (uniform fill) or per-pixel 2D map (textured body).
    """
    if not mask.any():
        return img.copy()
    if alpha_core is None:
        # Real pacemakers/ICDs on portable/AP CXRs are notably more OPAQUE than the previous
        # 0.20-0.55 range produced — they read as "metallic body, not glass". Bumping to
        # 0.45-0.80 matches reference real-CXR appearance (compare /tmp/compare_pacemaker.png).
        alpha_core = rng.uniform(0.45, 0.80)
    soft_mask = (mask.astype(np.float32) / 255.0) ** _EDGE_GAMMA
    binm = (mask >= 128).astype(np.uint8)
    if not binm.any():
        binm = (mask > 0).astype(np.uint8)
    dt = cv2.distanceTransform(binm, cv2.DIST_L2, 3)
    dmax = float(dt.max()) + 1e-6
    rim_w = 1.0 - (dt / dmax)
    alpha = (alpha_core + (1.0 - alpha_core) * rim_w).astype(np.float32)
    alpha = alpha * soft_mask
    alpha = np.clip(alpha, 0.0, 1.0)[..., None]
    if isinstance(color, np.ndarray) and color.ndim == 2:
        layer = np.clip(color, 0, 255).astype(np.float32)
        layer = np.stack([layer, layer, layer], axis=-1)
    else:
        layer = np.full_like(img, int(np.clip(color, 0, 255)), dtype=np.uint8).astype(np.float32)
    # mild interior dimming so circuit/battery pops (no rim_boost — caused saturation)
    interior = (1.0 - rim_w)[..., None]
    layer = layer - interior * 6.0
    layer = np.clip(layer, 0, 255)
    out = alpha * layer + (1.0 - alpha) * img.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)

def render_screen_blend(img, mask, color, rng):
    """Screen (lighten-only) blend: output = 1 - (1-bg)(1-fg*alpha).
    Models radio-opaque additive attenuation: object PLUS background, brighter where
    both contribute, never darker than either. This is closer to physical X-ray summation
    than the previous overlay/alpha-replace, which made implants look pasted-on top.
    """
    if not mask.any():
        return img.copy()
    alpha = ((mask.astype(np.float32) / 255.0) ** _EDGE_GAMMA)[..., None]
    if isinstance(color, np.ndarray) and color.ndim == 2:
        fg = np.stack([color, color, color], axis=-1).astype(np.float32) / 255.0
    else:
        c = float(np.clip(color, 0, 255)) / 255.0
        fg = np.full_like(img, int(c * 255), dtype=np.uint8).astype(np.float32) / 255.0
    bg = img.astype(np.float32) / 255.0
    # Screen blend, weighted by alpha
    blended = 1.0 - (1.0 - bg) * (1.0 - fg * alpha)
    out = blended * 255.0
    return np.clip(out, 0, 255).astype(np.uint8)


def render_xray_summed(img, mask, color, rng):
    """Physically-motivated additive attenuation for radio-opaque solids.
    On real X-ray, object attenuation ADDS to anatomy attenuation. We approximate:
        output = clip(bg + alpha * (color - bg/2), 0, 255)
    where the (color - bg/2) factor brightens proportionally to how dark the bg already is.
    Net: object pops over bone (already bright) less than over lung (already dark),
    matching the way real implants overlay anatomy.
    """
    if not mask.any():
        return img.copy()
    alpha = ((mask.astype(np.float32) / 255.0) ** _EDGE_GAMMA)[..., None]
    if isinstance(color, np.ndarray) and color.ndim == 2:
        fg = np.stack([color, color, color], axis=-1).astype(np.float32)
    else:
        c = float(np.clip(color, 0, 255))
        fg = np.full_like(img, int(c), dtype=np.uint8).astype(np.float32)
    bg = img.astype(np.float32)
    # additive with bg-aware boost
    contribution = alpha * (fg - bg * 0.5)
    out = bg + contribution
    return np.clip(out, 0, 255).astype(np.uint8)


PHYSICS_MODES = ['overlay', 'soft_edge', 'beer_lambert', 'occluded']

def render_with_physics(img, mask, color, anat_masks, rng, mode=None):
    if mode is None:
        mode = rng.choice(PHYSICS_MODES)
    if mode == 'overlay':             return render_overlay(img, mask, color, rng), mode
    if mode == 'soft_edge':           return render_soft_edge(img, mask, color, rng), mode
    if mode == 'beer_lambert':        return render_beer_lambert(img, mask, color, rng), mode
    if mode == 'occluded':            return render_occluded(img, mask, color, rng, anat_masks), mode
    if mode == 'translucent_implant': return render_translucent_implant(img, mask, color, rng), mode
    if mode == 'screen':              return render_screen_blend(img, mask, color, rng), mode
    if mode == 'xray_summed':         return render_xray_summed(img, mask, color, rng), mode
    raise ValueError(mode)

def lumen_tube_mask(pts, H, W, half_w, *, marker='solid',
                    wall_color=220, fill_color=None, marker_color=245,
                    tip_radius=0, tip_color=250, murphy_eye=False, rng=None):
    """Build (mask, per-pixel color_map) for radio-opaque tube along centerline `pts`.

    On a real CXR, a catheter / tube is NOT a uniform bright stripe — only the radio-opaque
    walls + any embedded marker stripe show. This builder bakes that structure into mask + color.

    marker:
      'paired_walls'      — only the two wall lines (no interior fill). Catheters, leads.
      'solid'             — filled body + bright walls. Chest tube, ECMO cannula.
      'continuous_stripe' — body faint + bright single-line stripe along center. NG tube.
      'dashed'            — periodic short bright segments along center. Swan-Ganz, ETT.
      'tip_only'          — body very faint, bright tip dot only. PICC line.
    `tip_radius` > 0 stamps a bright circle at last pt (catheter tip / electrode tip).
    `murphy_eye` cuts a small wall gap ~5px before tip (ETT/chest tube standard feature).
    Returns mask uint8 (255 inside) and color_map uint8 (per-pixel intensity).
    """
    mask = np.zeros((H, W), np.uint8)
    color_map = np.zeros((H, W), np.uint8)
    pts32 = pts.astype(np.int32)
    if len(pts32) < 2:
        return mask, color_map

    # defensive clamp — avoid uint8 wrap if caller passes overshooting rng values
    wall_color   = int(np.clip(wall_color,   0, 255))
    marker_color = int(np.clip(marker_color, 0, 255))
    tip_color    = int(np.clip(tip_color,    0, 255))
    if fill_color is not None:
        fill_color = int(np.clip(fill_color, 0, 255))

    if half_w >= 2:
        tangents = np.diff(pts, axis=0).astype(float)
        norms = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
        norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
        left  = (pts[:-1] + norms * half_w).astype(np.int32)
        right = (pts[:-1] - norms * half_w).astype(np.int32)
        # apply Murphy eye: drop last few wall pts on one side
        if murphy_eye and len(left) > 8:
            n_gap = 4
            gap_start = len(left) - 6
            left  = np.concatenate([left[:gap_start],  left[gap_start + n_gap:]])
            right = np.concatenate([right[:gap_start], right[gap_start + n_gap:]])
        if marker == 'solid' or fill_color is not None:
            try:
                poly = np.concatenate([left, right[::-1]], axis=0)
                # FILL = semi-transparent (mask alpha ~70) so background lung bleeds through tube
                # interior. Walls drawn after at full alpha 255 = opaque bright contour.
                # Keeps instance segmentation intact (mask > 0 still covers fill region).
                cv2.fillPoly(mask, [poly], 70)
                if fill_color is None:
                    fill_color = max(60, wall_color - 80)
                cv2.fillPoly(color_map, [poly], int(fill_color))
            except ValueError:
                pass
        # Walls drawn at 2px so the mask reads as a real tube on downsample/JPEG —
        # 1px lines vanish under bilinear/JPEG and the model can't learn from them.
        cv2.polylines(mask,      [left],  False, 255, 2, cv2.LINE_AA)
        cv2.polylines(mask,      [right], False, 255, 2, cv2.LINE_AA)
        cv2.polylines(color_map, [left],  False, int(wall_color), 2, cv2.LINE_AA)
        cv2.polylines(color_map, [right], False, int(wall_color), 2, cv2.LINE_AA)
    else:
        # thin (catheter): single centerline at the wall_color intensity. 2px
        # so the line survives downstream resize/JPEG.
        body_intensity = int(wall_color if marker in ('paired_walls', 'tip_only') else wall_color)
        cv2.polylines(mask,      [pts32], False, 255, 2, cv2.LINE_AA)
        cv2.polylines(color_map, [pts32], False, body_intensity, 2, cv2.LINE_AA)

    if marker == 'continuous_stripe':
        cv2.polylines(mask,      [pts32], False, 255, 2, cv2.LINE_AA)
        cv2.polylines(color_map, [pts32], False, int(marker_color), 2, cv2.LINE_AA)
    elif marker == 'dashed':
        step = max(8, len(pts32) // 12)
        dash_len = max(2, step // 3)
        for i in range(0, len(pts32) - 1, step):
            seg = pts32[i:i + dash_len + 1]
            if len(seg) > 1:
                cv2.polylines(mask,      [seg], False, 255, 2, cv2.LINE_AA)
                cv2.polylines(color_map, [seg], False, int(marker_color), 2, cv2.LINE_AA)

    if tip_radius > 0:
        tip = (int(pts32[-1][0]), int(pts32[-1][1]))
        cv2.circle(mask,      tip, tip_radius, 255, -1)
        cv2.circle(color_map, tip, tip_radius, int(tip_color), -1)
    return mask, color_map


def shape_textual(H, W, anat_masks, rng):
    mask = np.zeros((H, W), np.uint8)
    chars = ''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-+', k=rng.randint(1, 6)))
    scale = rng.uniform(0.6, 2.2); thickness = rng.randint(1, 3)
    font = rng.choice([cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_COMPLEX,
                       cv2.FONT_HERSHEY_PLAIN, cv2.FONT_HERSHEY_DUPLEX])
    angle = _sample_text_angle(rng)
    glyph, tw, th = _render_rotated_text(chars, font, scale, thickness, angle)
    color = rng.choice([0, 255, rng.randint(160, 230)])
    if tw == 0 or th == 0 or tw >= W or th >= H:
        return mask, color
    x = rng.randint(0, max(W - tw, 1)); y = rng.randint(0, max(H - th, 1))
    mask[y:y + th, x:x + tw] = np.maximum(mask[y:y + th, x:x + tw], glyph)
    return mask, color

def shape_circular(H, W, anat_masks, rng):
    target = union_mask(anat_masks, ANAT_GROUPS['heart'] + ANAT_GROUPS['humerus'])
    pt = sample_point_in_mask(target, rng) or (rng.randint(0, W - 1), rng.randint(0, H - 1))
    ax = rng.randint(8, 40); by = rng.randint(8, 40); angle = rng.uniform(0, 360)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, pt, (ax, by), angle, 0, 360, 255, -1)
    return mask, rng.randint(180, 255)

def shape_ring(H, W, anat_masks, rng):
    target = union_mask(anat_masks, ANAT_GROUPS['heart'] + ANAT_GROUPS['aorta'])
    pt = sample_point_in_mask(target, rng) or (W // 2, H // 2)
    ax = rng.randint(15, 50); by = rng.randint(15, 50)
    angle = rng.uniform(0, 360); t = rng.randint(2, 5)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, pt, (ax, by), angle, 0, 360, 255, t)
    return mask, rng.randint(200, 255)

def shape_rect(H, W, anat_masks, rng):
    w = rng.randint(15, 60); h = rng.randint(15, 60)
    x = rng.randint(0, W - w); y = rng.randint(0, H - h)
    mask = np.zeros((H, W), np.uint8)
    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask, rng.randint(180, 255)

def shape_clip(H, W, anat_masks, rng):
    rib_u = union_mask(anat_masks, ANAT_GROUPS['ribs'])
    target = rib_u if rib_u.any() else anat_masks.any(axis=0)
    pt = sample_point_in_mask(target, rng) or (rng.randint(0, W - 1), rng.randint(0, H - 1))
    L = rng.randint(8, 22); t = rng.randint(2, 4); a = rng.uniform(0, math.pi)
    dx, dy = int(L * math.cos(a)), int(L * math.sin(a))
    p1 = (pt[0] - dx, pt[1] - dy); p2 = (pt[0] + dx, pt[1] + dy)
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, p1, p2, 255, t, cv2.LINE_AA)
    return mask, 240

def shape_grid(H, W, anat_masks, rng):
    mask = np.zeros((H, W), np.uint8)
    u = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    if not u.any(): u = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    if not u.any(): return mask, 220
    ys, xs = np.where(u)
    y0, y1 = int(ys.min()), int(ys.max()); x0, x1 = int(xs.min()), int(xs.max())
    sub_h = rng.randint(20, max(30, (y1 - y0) // 2 + 30))
    sub_w = rng.randint(10, max(20, (x1 - x0) // 2 + 20))
    cy = rng.randint(y0, y1); cx = rng.randint(x0, x1)
    yA = max(0, cy - sub_h // 2); yB = min(H, yA + sub_h)
    xA = max(0, cx - sub_w // 2); xB = min(W, xA + sub_w)
    cell = rng.randint(4, 8)
    for yi in range(yA, yB, cell):
        for xi in range(xA, xB, cell):
            if yi >= H or xi >= W or not u[yi, xi]: continue
            if rng.random() < 0.7:
                xj = min(xi + cell, xB - 1); cv2.line(mask, (xi, yi), (xj, yi), 255, 1)
            if rng.random() < 0.7:
                yj = min(yi + cell, yB - 1); cv2.line(mask, (xi, yi), (xi, yj), 255, 1)
    return mask, rng.randint(200, 255)

def shape_line(H, W, anat_masks, rng):
    p0 = np.array(random_outside_point(H, W, rng), float)
    target = union_mask(anat_masks, ANAT_GROUPS['vessels'] + ANAT_GROUPS['heart'])
    end = sample_point_in_mask(target, rng) or (rng.randint(0, W - 1), rng.randint(0, H - 1))
    p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-80, 80), rng.randint(-80, 80)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-80, 80), rng.randint(-80, 80)], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, rng.randint(1, 3), cv2.LINE_AA)
    kpts = {'start': (int(p0[0]), int(p0[1]), 2),
            'end': (int(p3[0]), int(p3[1]), 2)}
    return mask, rng.randint(200, 255), kpts

def shape_parallel(H, W, anat_masks, rng):
    target = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    end = sample_point_in_mask(target, rng) if target.any() else (W // 2, H // 2)
    p0 = np.array(random_outside_point(H, W, rng), float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-40, 40), rng.randint(-40, 40)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-40, 40), rng.randint(-40, 40)], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    w = rng.randint(4, 10)
    tangents = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left  = (pts[:-1] + norms * w).astype(np.int32)
    right = (pts[:-1] - norms * w).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    return mask, rng.randint(80, 160)

def shape_blob(H, W, anat_masks, rng):
    """Irregular closed contour via Fourier polar series — ingested objects, unknown FBs."""
    body = anat_masks.any(axis=0)
    pt = sample_point_in_mask(body, rng) or (W // 2, H // 2)
    cx, cy = pt
    base_r = rng.randint(15, 50)
    n_pts = 64
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    radii = np.full(n_pts, base_r, dtype=float)
    n_harm = rng.randint(2, 6)
    for k in range(1, n_harm + 1):
        amp = base_r * rng.uniform(0.05, 0.25) / k
        phase = rng.uniform(0, 2 * np.pi)
        radii += amp * np.cos(k * theta + phase)
    xs = (cx + radii * np.cos(theta)).astype(np.int32)
    ys = (cy + radii * np.sin(theta)).astype(np.int32)
    pts = np.stack([xs, ys], axis=1)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask, rng.randint(180, 255)

# --- SDF primitives + smooth union ---
def _sdf_rect(P, c, half):
    d = np.abs(P - c) - half
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.max(d, axis=-1), 0)

def _sdf_capsule(P, a, b, r):
    pa = P - a; ba = b - a
    bb = float(np.dot(ba, ba)) + 1e-6
    h = np.clip(np.sum(pa * ba, -1) / bb, 0, 1)
    return np.linalg.norm(pa - h[..., None] * ba, axis=-1) - r

def _smooth_union(d1, d2, k):
    h = np.clip(0.5 + 0.5 * (d2 - d1) / k, 0, 1)
    return d2 + (d1 - d2) * h - k * h * (1 - h)

def shape_sdf_compound(H, W, anat_masks, rng):
    """Rect body + N capsule arms via smooth-union — generates legged/T/U/cross shapes."""
    body = anat_masks.any(axis=0)
    pt = sample_point_in_mask(body, rng) or (W // 2, H // 2)
    cx, cy = pt
    bw, bh = rng.randint(20, 60), rng.randint(20, 60)
    span = max(bw, bh) + 30
    y0 = max(0, cy - span); y1 = min(H, cy + span)
    x0 = max(0, cx - span); x1 = min(W, cx + span)
    mask = np.zeros((H, W), np.uint8)
    if y1 <= y0 or x1 <= x0: return mask, rng.randint(180, 255)
    yy, xx = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
    P = np.stack([xx, yy], -1).astype(np.float32)
    c = np.array([cx, cy], np.float32)
    d = _sdf_rect(P, c, np.array([bw / 2, bh / 2], np.float32))
    n_legs = rng.randint(1, 4)
    for _ in range(n_legs):
        angle = rng.uniform(0, 2 * np.pi)
        L = rng.randint(20, 60)
        end = c + L * np.array([np.cos(angle), np.sin(angle)], np.float32)
        r = rng.uniform(2, 5)
        d = _smooth_union(d, _sdf_capsule(P, c, end, r), k=4.0)
    mask[y0:y1, x0:x1] = ((d <= 0).astype(np.uint8) * 255)
    return mask, rng.randint(180, 255)

SHAPE_FNS = {
    'textual':  shape_textual,
    'circular': shape_circular,
    'ring':     shape_ring,
    'rect':     shape_rect,
    'clip':     shape_clip,
    'grid':     shape_grid,
    'line':     shape_line,
    'parallel': shape_parallel,
    'blob':     shape_blob,
    'sdf':      shape_sdf_compound,
}
print('all shape types:', list(SHAPE_FNS))

def shape_branching_tree(H, W, anat_masks, rng):
    """Recursive Bezier branches — multi-port catheter / ablation lead set."""
    mask = np.zeros((H, W), np.uint8)
    p0 = np.array(random_outside_point(H, W, rng), float)
    target = union_mask(anat_masks, ANAT_GROUPS['heart'] + ANAT_GROUPS['vessels'])
    end = sample_point_in_mask(target, rng) or (W // 2, H // 2)
    p3 = np.array(end, float)

    def draw_branch(a, b, depth, thickness):
        if depth == 0:
            return
        off = np.array([rng.randint(-30, 30), rng.randint(-30, 30)], float)
        p1 = a + (b - a) * 0.33 + off
        p2 = a + (b - a) * 0.66 + off
        pts = bezier_points(a, p1, p2, b, n=64)
        cv2.polylines(mask, [pts], False, 255, thickness, cv2.LINE_AA)
        if depth > 1 and rng.random() < 0.55:
            branch_start = pts[rng.randint(20, 50)].astype(float)
            branch_end = branch_start + np.array(
                [rng.randint(-70, 70), rng.randint(-70, 70)], float)
            draw_branch(branch_start, branch_end, depth - 1, max(1, thickness - 1))

    draw_branch(p0, p3, depth=rng.randint(2, 4), thickness=rng.randint(1, 3))
    return mask, rng.randint(200, 255)


def shape_scratch(H, W, anat_masks, rng):
    """Long jittered straight line — film scratch / dust streak across the radiograph."""
    mask = np.zeros((H, W), np.uint8)
    angle = rng.uniform(0, math.pi)
    L = rng.randint(W // 3, W)
    cx, cy = rng.randint(0, W), rng.randint(0, H)
    dx = int(L / 2 * math.cos(angle)); dy = int(L / 2 * math.sin(angle))
    p1 = (cx - dx, cy - dy); p2 = (cx + dx, cy + dy)
    nseg = 12
    pts = []
    for i in range(nseg + 1):
        t = i / nseg
        x = int(p1[0] + (p2[0] - p1[0]) * t + rng.randint(-2, 2))
        y = int(p1[1] + (p2[1] - p1[1]) * t + rng.randint(-2, 2))
        pts.append([x, y])
    cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    kpts = {'start': (int(p1[0]), int(p1[1]), 2),
            'end': (int(p2[0]), int(p2[1]), 2)}
    return mask, rng.randint(200, 255), kpts


def shape_dotted_screws(H, W, anat_masks, rng):
    """Small bright circles on bone — orthopedic screws / bullets."""
    bone = union_mask(anat_masks,
                      ANAT_GROUPS['ribs'] + ANAT_GROUPS['humerus'] + ANAT_GROUPS['clavicle'])
    if not bone.any():
        bone = anat_masks.any(axis=0)
    mask = np.zeros((H, W), np.uint8)
    n = rng.randint(2, 6)
    for _ in range(n):
        pt = sample_point_in_mask(bone, rng)
        if pt is None:
            continue
        r = rng.randint(2, 5)
        cv2.circle(mask, pt, r, 255, -1)
    return mask, rng.randint(220, 255)


def shape_staple_line(H, W, anat_masks, rng):
    """Small parallel ticks along a Bezier — surgical staple line."""
    body = anat_masks.any(axis=0)
    start = sample_point_in_mask(body, rng) or (W // 2, H // 2)
    p0 = np.array(start, float)
    p3 = p0 + np.array([rng.randint(-120, 120), rng.randint(-60, 60)], float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-30, 30), rng.randint(-30, 30)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-30, 30), rng.randint(-30, 30)], float)
    pts = bezier_points(p0, p1, p2, p3, n=64)
    tan = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    half = rng.randint(3, 6); step = rng.randint(4, 8)
    mask = np.zeros((H, W), np.uint8)
    for i in range(0, len(pts) - 1, step):
        c = pts[i]; n = norms[i]
        a = (c + n * half).astype(np.int32); b = (c - n * half).astype(np.int32)
        cv2.line(mask, tuple(a), tuple(b), 255, 1, cv2.LINE_AA)
    return mask, rng.randint(220, 255)


def shape_ring_chain(H, W, anat_masks, rng):
    """Chain of hollow circles along an arc — necklace, cerclage chain."""
    p0 = (rng.randint(W // 4, W // 2), rng.randint(H // 6, H // 3))
    p1 = (rng.randint(W // 2, 3 * W // 4), rng.randint(H // 6, H // 3))
    n_links = rng.randint(8, 16); r = rng.randint(3, 6)
    mid_y = (p0[1] + p1[1]) // 2 + rng.randint(-20, 30)
    mid_x = (p0[0] + p1[0]) // 2
    mask = np.zeros((H, W), np.uint8)
    for t in np.linspace(0, 1, n_links):
        x = int((1 - t) ** 2 * p0[0] + 2 * (1 - t) * t * mid_x + t ** 2 * p1[0])
        y = int((1 - t) ** 2 * p0[1] + 2 * (1 - t) * t * mid_y + t ** 2 * p1[1])
        cv2.circle(mask, (x, y), r, 255, 1)
    return mask, rng.randint(200, 255)


def shape_pellet_cluster(H, W, anat_masks, rng):
    """Small dense circles clustered around a body point — shotgun pellets."""
    body = anat_masks.any(axis=0)
    center = sample_point_in_mask(body, rng) or (W // 2, H // 2)
    n = rng.randint(3, 10)
    mask = np.zeros((H, W), np.uint8)
    for _ in range(n):
        dx = rng.randint(-30, 30); dy = rng.randint(-30, 30)
        r = rng.randint(2, 5)
        cv2.circle(mask, (center[0] + dx, center[1] + dy), r, 255, -1)
    return mask, rng.randint(240, 255)


def shape_ekg_bundle(H, W, anat_masks, rng):
    """N parallel Bezier curves entering the body — EKG / monitoring cable bundle."""
    mask = np.zeros((H, W), np.uint8)
    n_wires = rng.randint(2, 5)
    p0 = np.array(random_outside_point(H, W, rng), float)
    chest = anat_masks.any(axis=0)
    end_pt = sample_point_in_mask(chest, rng) or (W // 2, H // 2)
    base_end = np.array(end_pt, float)
    for i in range(n_wires):
        off = np.array([rng.randint(-5, 5) * i, rng.randint(-5, 5) * i], float)
        p3 = base_end + off
        p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-60, 60), rng.randint(-60, 60)], float)
        p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-60, 60), rng.randint(-60, 60)], float)
        pts = bezier_points(p0 + off, p1, p2, p3, n=96)
        cv2.polylines(mask, [pts], False, 255, 2, cv2.LINE_AA)
    return mask, rng.randint(180, 240)


def shape_offframe_cables(H, W, anat_masks, rng):
    """Overlying monitoring/IV cables drifting across upper chest and exiting frame.
    Real ICU portable CXRs almost always show 3-5 thin radiopaque cable strands
    crossing the upper image and going off-edge (EKG telemetry, IV poles, pulse-ox).
    They mostly DON'T terminate on the body — they pass through and exit.
    """
    mask = np.zeros((H, W), np.uint8)
    n = rng.randint(3, 6)
    for _ in range(n):
        # both endpoints outside the frame, but path traverses inside
        side_a = rng.choice(['top', 'left', 'right'])
        side_b = rng.choice([s for s in ['top', 'left', 'right'] if s != side_a])
        def edge_pt(side):
            if side == 'top':   return (rng.randint(0, W), -rng.randint(5, 25))
            if side == 'left':  return (-rng.randint(5, 25), rng.randint(0, H // 2))
            if side == 'right': return (W + rng.randint(5, 25), rng.randint(0, H // 2))
            return (W // 2, 0)
        p0 = np.array(edge_pt(side_a), float)
        p3 = np.array(edge_pt(side_b), float)
        # control points pull the cable through the upper chest area
        mid = np.array([rng.randint(W // 4, 3 * W // 4),
                        rng.randint(H // 8, H // 3)], float)
        p1 = (p0 + mid) / 2 + np.array([rng.randint(-30, 30), rng.randint(-20, 20)], float)
        p2 = (mid + p3) / 2 + np.array([rng.randint(-30, 30), rng.randint(-20, 20)], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        pts = jitter_path(pts, rng, amp_px=0.5)
        cv2.polylines(mask, [pts], False, 255, 2, cv2.LINE_AA)
    return mask, rng.randint(170, 220)


def shape_acquisition_burn_in(H, W, anat_masks, rng):
    """Burn-in technique label as you see on real portable/AP CXRs.
    Real AP/portable studies always have PORTABLE / AP / SUPINE / UPRIGHT / SEMI-ERECT
    burned into a corner by the technologist. Synth backgrounds (cleaned PA) lack
    these, which is a strong giveaway in any synth-vs-real discriminator.
    """
    mask = np.zeros((H, W), np.uint8)
    label = rng.choice([
        'PORTABLE', 'AP', 'AP PORTABLE', 'PORTABLE SUPINE', 'SEMI-ERECT',
        'PORTABLE ERECT', 'SUPINE', 'AP UPRIGHT', 'PORTABLE\nSUPINE',
    ])
    scale = rng.uniform(0.55, 0.95)
    thickness = rng.choice([1, 2, 2])
    font = cv2.FONT_HERSHEY_SIMPLEX
    # pick a corner
    corner = rng.choice(['tl', 'tr', 'bl', 'br'])
    margin = rng.randint(10, 35)
    lines = label.split('\n')
    line_h = int(28 * scale)
    block_h = line_h * len(lines)
    for i, ln in enumerate(lines):
        (tw, th), _ = cv2.getTextSize(ln, font, scale, thickness)
        if corner == 'tl':
            x = margin; y = margin + (i+1) * line_h
        elif corner == 'tr':
            x = W - tw - margin; y = margin + (i+1) * line_h
        elif corner == 'bl':
            x = margin; y = H - block_h + (i+1) * line_h - margin
        else:
            x = W - tw - margin; y = H - block_h + (i+1) * line_h - margin
        cv2.putText(mask, ln, (x, y), font, scale, 255, thickness, cv2.LINE_AA)
    return mask, rng.randint(225, 250)


def shape_vent_hose(H, W, anat_masks, rng):
    """Ventilator circuit / O2 cannula silhouette: one thick faint curve overlying
    upper chest, often paired with ET tube. Lower opacity than catheters (it's plastic
    tubing several cm thick — only its silhouette shows on a portable X-ray)."""
    mask = np.zeros((H, W), np.uint8)
    # enter from top, sweep across upper chest, exit top or side
    p0 = np.array([rng.randint(W // 4, 3 * W // 4), -rng.randint(5, 20)], float)
    side = rng.choice([-1, +1])
    exit_x = 0 - rng.randint(10, 30) if side < 0 else W + rng.randint(10, 30)
    p3 = np.array([exit_x, rng.randint(H // 8, H // 4)], float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-40, 40), rng.randint(20, 80)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-40, 40), rng.randint(0, 50)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    pts = jitter_path(pts, rng, amp_px=1.0, wavelength_pts=20)
    # vent hose is thick (radiopaque silhouette of the corrugated tube)
    half = rng.randint(8, 14)
    tangents = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left  = (pts[:-1] + norms * half).astype(np.int32)
    right = (pts[:-1] - norms * half).astype(np.int32)
    try:
        poly = np.concatenate([left, right[::-1]], axis=0)
        cv2.fillPoly(mask, [poly], 110)              # faint translucent fill
        cv2.polylines(mask, [left],  False, 200, 1, cv2.LINE_AA)
        cv2.polylines(mask, [right], False, 200, 1, cv2.LINE_AA)
    except ValueError:
        pass
    return mask, rng.randint(180, 220)


def shape_mesh_stent(H, W, anat_masks, rng):
    """Diamond mesh pattern clipped by aorta / vessel mask — denser/realistic stent."""
    target = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    if not target.any():
        target = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    mask = np.zeros((H, W), np.uint8)
    if not target.any():
        return mask, 220
    ys, xs = np.where(target)
    y0, y1 = int(ys.min()), int(ys.max())
    cy = rng.randint(y0, y1)
    sub_h = rng.randint(20, max(30, (y1 - y0) // 2 + 30))
    yA = max(0, cy - sub_h // 2); yB = min(H, yA + sub_h)
    cell = rng.randint(5, 10)
    cx0 = int(np.median(xs))
    for off in range(-sub_h - 30, sub_h + 30, cell):
        for sign in (+1, -1):
            pts = []
            for y in range(yA, yB):
                x = cx0 + off + sign * (y - yA)
                if 0 <= x < W and target[y, x]:
                    pts.append([x, y])
            if len(pts) > 2:
                cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, 1, cv2.LINE_AA)
    return mask, rng.randint(180, 255)


# extend the SHAPE_FNS dict (defined in §5.3) with the new primitives
SHAPE_FNS.update({
    'branching_tree': shape_branching_tree,
    'scratch':        shape_scratch,
    'dotted_screws':  shape_dotted_screws,
    'staple_line':    shape_staple_line,
    'ring_chain':     shape_ring_chain,
    'pellet_cluster': shape_pellet_cluster,
    'ekg_bundle':     shape_ekg_bundle,
    'offframe_cables': shape_offframe_cables,
    'vent_hose':      shape_vent_hose,
    'acquisition_burn_in': shape_acquisition_burn_in,
    'mesh_stent':     shape_mesh_stent,
})
print(f'total shape types: {len(SHAPE_FNS)}')
print(list(SHAPE_FNS))

def device_pacemaker(H, W, anat_masks, rng):
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        i = rng.randrange(len(ys))
        cx = int(xs[i]); cy = int(ys[i]) + rng.randint(20, 60)
    else:
        cx, cy = W // 4, H // 4
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    parts = []
    bw, bh = rng.randint(22, 38), rng.randint(28, 48)
    body_mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(body_mask, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
    parts.append(('pacemaker_body', body_mask, rng.randint(220, 255)))
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    n_leads = rng.randint(1, 3)
    for _ in range(n_leads):
        p0 = np.array([cx, cy + bh // 2], float)
        end = sample_point_in_mask(heart_u, rng) if heart_u.any() else (cx, min(H - 1, cy + 150))
        p3 = np.array(end, float)
        p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-25, 25), rng.randint(-15, 30)], float)
        p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-35, 35), rng.randint(-15, 30)], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        lead = np.zeros((H, W), np.uint8)
        cv2.polylines(lead, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
        parts.append(('pacemaker_lead', lead, rng.randint(200, 245)))
    return parts

def device_port_catheter(H, W, anat_masks, rng):
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        i = rng.randrange(len(ys))
        cx = int(xs[i]); cy = int(ys[i]) + rng.randint(30, 80)
    else:
        cx, cy = 3 * W // 4, H // 4
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    r = rng.randint(10, 18)
    res = np.zeros((H, W), np.uint8)
    cv2.circle(res, (cx, cy), r, 255, -1)
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    end = sample_point_in_mask(vess, rng) if vess.any() else (W // 2, min(H - 1, cy + 120))
    p0 = np.array([cx, cy + r], float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-30, 30), rng.randint(-15, 30)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-40, 40), rng.randint(-15, 30)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    tube = np.zeros((H, W), np.uint8)
    cv2.polylines(tube, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    return [
        ('port_reservoir', res,  rng.randint(215, 255)),
        ('catheter_tube',  tube, rng.randint(190, 240)),
    ]

def device_sternotomy_wires(H, W, anat_masks, rng):
    """Cerclage wires along sternum midline. Real cerclage is FAINT, with thin variable
    cross-section, intensity drop-outs from rib occlusion, and never a clean uniform
    bright line — that's the threshold-separable look I'm trying to avoid.

    Build each wire as a per-pixel color map (not solid color) so length-wise intensity
    noise + thickness variation render naturally. With `screen` blend the wire reads as
    overlying anatomy, not as pasted.
    """
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        cx = int(np.median(xs))
        ymin, ymax = int(ys.min()) + 20, int(ys.max()) - 20
    else:
        cx, ymin, ymax = W // 2, H // 4, 3 * H // 4
    n = rng.randint(4, 7)
    spacing = max(20, (ymax - ymin) // (n + 1))
    parts = []
    # rib mask for partial-occlusion shading
    ribs = union_mask(anat_masks, ANAT_GROUPS['ribs'])
    for i in range(n):
        y = ymin + (i + 1) * spacing + rng.randint(-4, 4)
        x = cx + rng.randint(-5, 5)
        variant = rng.choices(['loop', 'transverse', 'figure8'], weights=[0.6, 0.3, 0.1])[0]
        wire_mask = np.zeros((H, W), np.uint8)
        # Real cerclage is ~1mm dia, on 512px CXR that's 2px stroke. Was 1px before
        # (read as a "drawn line"). Use 2px with mid-alpha so it has visible thickness
        # without screaming for attention.
        mask_alpha = rng.randint(150, 195)
        stroke_w = 2
        if variant == 'loop':
            # Real cerclage encircles the sternum HORIZONTALLY. On PA view we see
            # the loop end-on / oblique: a HORIZONTAL OVAL across midline, not a vertical one.
            rx = rng.randint(8, 14)            # wide axis (across sternum)
            ry = rng.randint(2, 4)             # short axis (along sternum)
            angle = rng.uniform(-8, 8)
            cv2.ellipse(wire_mask, (x, y), (rx, ry), angle, 0, 360, mask_alpha, stroke_w, cv2.LINE_AA)
            if rng.random() < 0.4:
                # twist knot to one side of the cerclage
                knot_x = x + (rx if rng.random() < 0.5 else -rx)
                cv2.circle(wire_mask, (knot_x, y), rng.randint(2, 3), max(60, mask_alpha - 20),
                           stroke_w, cv2.LINE_AA)
            kpts = {'left_end': (x - rx, y, 2), 'right_end': (x + rx, y, 2)}
        elif variant == 'transverse':
            half = rng.randint(10, 16)
            pts = np.array([
                [x - half, y + rng.randint(-2, 2)],
                [x,        y + rng.randint(-3, 3)],
                [x + half, y + rng.randint(-2, 2)],
            ], np.int32)
            cv2.polylines(wire_mask, [pts], False, mask_alpha, stroke_w, cv2.LINE_AA)
            kpts = {'left_end':  (int(pts[0][0]),  int(pts[0][1]),  2),
                    'right_end': (int(pts[-1][0]), int(pts[-1][1]), 2)}
        else:  # figure-8 (twisted-pair cerclage) — also horizontal
            rx = rng.randint(5, 9); ry = rng.randint(2, 4)
            cv2.ellipse(wire_mask, (x - rx, y), (rx, ry), 0, 0, 360, mask_alpha, stroke_w, cv2.LINE_AA)
            cv2.ellipse(wire_mask, (x + rx, y), (rx, ry), 0, 0, 360, mask_alpha, stroke_w, cv2.LINE_AA)
            kpts = {'left_end': (x - 2 * rx, y, 2), 'right_end': (x + 2 * rx, y, 2)}
        # Per-pixel color: low base intensity (80-130, was 110-165) so the wire matches
        # real CXR appearance — faint twisted-metal hint, not a sharp bright stroke.
        wire_color = np.zeros((H, W), np.uint8)
        wire_pixels = wire_mask > 0
        if wire_pixels.any():
            base = rng.randint(80, 130)
            noise = np.random.RandomState(rng.randint(0, 2**31)).randint(-25, 25, size=wire_mask.shape)
            intens = np.clip(base + noise, 50, 175).astype(np.uint8)
            # heavier rib occlusion: 35% dim (vs prior 55%) so the wire all but disappears
            # behind cortical bone, matching real radiographic occlusion
            occ = ribs & wire_pixels if ribs is not None and ribs.any() else None
            if occ is not None and occ.any():
                intens[occ] = (intens[occ].astype(int) * 0.35).clip(25, 150).astype(np.uint8)
            wire_color[wire_pixels] = intens[wire_pixels]
        parts.append(('sternotomy_wire', wire_mask, wire_color, kpts))
    return parts

def device_et_tube(H, W, anat_masks, rng):
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        top_idx = int(np.argmin(ys)); bot_idx = int(np.argmax(ys))
        entry = (int(xs[top_idx]) + rng.randint(-10, 10), int(ys[top_idx]) - rng.randint(30, 60))
        tip   = (int(xs[bot_idx]) + rng.randint(-10, 10), int(ys[bot_idx]) - rng.randint(0, 20))
    else:
        entry = (W // 2, 20); tip = (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-15, 15), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-15, 15), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    w = rng.randint(5, 9)
    tangents = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left  = (pts[:-1] + norms * w).astype(np.int32)
    right = (pts[:-1] - norms * w).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    shaft = np.zeros((H, W), np.uint8); cv2.fillPoly(shaft, [poly], 255)
    cuff = np.zeros((H, W), np.uint8)
    cv2.ellipse(cuff, (int(tip[0]), int(tip[1])), (w + 4, w + 2), 0, 0, 360, 255, -1)
    return [
        ('et_tube_shaft', shaft, rng.randint(90, 160)),
        ('et_tube_cuff',  cuff,  rng.randint(180, 230)),
    ]

DEVICE_FNS = {
    'pacemaker':        device_pacemaker,
    'port_catheter':    device_port_catheter,
    'sternotomy_wires': device_sternotomy_wires,
    'et_tube':          device_et_tube,
}
print('devices:', list(DEVICE_FNS))

# extend anatomy groups with spine (used by spinal rods)
ANAT_GROUPS.setdefault('spine', group_ids(safe_attr(label_mapper, 'thoracic_spine')))


# ===== Sprint 1 SHAPES =====

def shape_ng_tube(H, W, anat_masks, rng):
    """Thin Bezier from nostril (top-center) through trachea/esophagus to stomach (below diaphragm)."""
    entry = (W // 2 + rng.randint(-15, 15), rng.randint(0, 30))
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, _ = np.where(lungs)
        tip_y = min(H - 10, int(ys.max()) + rng.randint(10, 60))
    else:
        tip_y = H - 30
    tip = (W // 2 + rng.randint(-30, 30), tip_y)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-20, 20), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, rng.randint(*CATH_W), cv2.LINE_AA)
    return mask, rng.randint(*CATH_GRAY)


def shape_picc_line(H, W, anat_masks, rng):
    """Bezier from clavicle/humerus into SVC tip, with radio-opaque tip dot."""
    side = rng.choice([-1, +1])
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'] + ANAT_GROUPS['humerus'])
    if clav.any():
        ys, xs = np.where(clav)
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        if sel.any():
            i = rng.randrange(int(sel.sum()))
            entry = (int(xs[sel][i]), int(ys[sel][i]))
        else:
            entry = (W // 2 + side * W // 4, rng.randint(20, H // 3))
    else:
        entry = (W // 2 + side * W // 4, rng.randint(20, H // 3))
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    if vess.any():
        ys, xs = np.where(vess)
        top_idx = int(np.argmin(ys))
        tip = (int(xs[top_idx]) + rng.randint(-5, 5), int(ys[top_idx]) + rng.randint(-5, 5))
    else:
        tip = (W // 2, H // 3)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-30, 30), rng.randint(-15, 15)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-30, 30), rng.randint(-15, 15)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, rng.randint(*CATH_W), cv2.LINE_AA)
    cv2.circle(mask, (int(tip[0]), int(tip[1])), rng.randint(2, 4), 255, -1)
    return mask, rng.randint(*CATH_GRAY)


def shape_annuloplasty_ring(H, W, anat_masks, rng):
    """D-shaped partial ellipse at mitral position (heart center)."""
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    pt = sample_point_in_mask(heart, rng) or (W // 2, H // 2)
    cx, cy = pt
    rx = rng.randint(12, 22); ry = rng.randint(10, 18)
    angle = rng.uniform(0, 180)
    mask = np.zeros((H, W), np.uint8)
    arc_start = rng.uniform(-30, 20); arc_end = rng.uniform(180, 220)
    cv2.ellipse(mask, (cx, cy), (rx, ry), angle, arc_start, arc_end, 255, 2)
    return mask, rng.randint(220, 255)


def shape_chest_tube(H, W, anat_masks, rng):
    """Thick tube entering lateral chest with dotted radio-opaque marker stripe."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        side = rng.choice([-1, +1])
        ymid = int(np.median(ys))
        xedge = int(xs.max()) if side > 0 else int(xs.min())
        entry = (xedge + side * rng.randint(20, 40), ymid + rng.randint(-30, 30))
        end = sample_point_in_mask(lungs, rng) or (W // 2, H // 2)
    else:
        side = rng.choice([-1, +1])
        entry = (rng.randint(0, 30) if side < 0 else rng.randint(W - 30, W), H // 2)
        end = (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-20, 20), rng.randint(-20, 20)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-20, 20), rng.randint(-20, 20)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    w = rng.randint(4, 8)
    tangents = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left = (pts[:-1] + norms * w).astype(np.int32)
    right = (pts[:-1] - norms * w).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    for i in range(0, len(pts), 4):
        cv2.circle(mask, tuple(pts[i].tolist()), 1, 255, -1)
    return mask, rng.randint(80, 160), _bezier_kpts(pts, prox_name='skin_entry')  # darker tube body


# ===== Sprint 1 DEVICES =====

def device_icd(H, W, anat_masks, rng):
    """ICD = bigger pacemaker body + shock-coil lead (alternating thick/thin segments)."""
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        i = rng.randrange(len(ys))
        cx = int(xs[i]); cy = int(ys[i]) + rng.randint(20, 60)
    else:
        cx, cy = W // 4, H // 4
    cx = int(np.clip(cx, 30, W - 30)); cy = int(np.clip(cy, 30, H - 30))
    parts = []
    bw, bh = rng.randint(38, 55), rng.randint(45, 60)
    body = np.zeros((H, W), np.uint8)
    cv2.ellipse(body, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
    parts.append(('icd_body', body, rng.randint(220, 255), {'center': (cx, cy, 2)}))
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    n_leads = rng.randint(1, 3)
    for _ in range(n_leads):
        p0 = np.array([cx, cy + bh // 2], float)
        end = sample_point_in_mask(heart_u, rng) if heart_u.any() else (cx, min(H - 1, cy + 150))
        p3 = np.array(end, float)
        p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-25, 25), rng.randint(-15, 30)], float)
        p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-35, 35), rng.randint(-15, 30)], float)
        pts = bezier_points(p0, p1, p2, p3, n=128)
        lead = np.zeros((H, W), np.uint8)
        # alternating shock-coil (thick) + regular (thin) segments
        seg_len = max(8, len(pts) // rng.randint(6, 10))
        thick_t = rng.randint(3, 5)
        for s in range(0, len(pts) - 1, seg_len):
            t = thick_t if (s // seg_len) % 2 == 1 else 1
            seg = pts[s:s + seg_len + 1]
            if len(seg) > 1:
                cv2.polylines(lead, [seg], False, 255, t, cv2.LINE_AA)
        parts.append(('icd_lead', lead, rng.randint(210, 250), _bezier_kpts(pts)))
    return parts


def device_tracheostomy(H, W, anat_masks, rng):
    """Short curved tube entering anterior neck → proximal trachea (different from ET tube entry)."""
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        top_idx = int(np.argmin(ys))
        tip = (int(xs[top_idx]) + rng.randint(-5, 5), int(ys[top_idx]) + rng.randint(5, 30))
    else:
        tip = (W // 2, H // 4)
    entry = (tip[0] + rng.randint(-20, 20), tip[1] - rng.randint(20, 40))
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.5 + np.array([rng.randint(-5, 5), 0], float)
    p2 = p0 + (p3 - p0) * 0.8 + np.array([rng.randint(-5, 5), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=64)
    w = rng.randint(5, 9)
    tangents = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left = (pts[:-1] + norms * w).astype(np.int32)
    right = (pts[:-1] - norms * w).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    shaft = np.zeros((H, W), np.uint8); cv2.fillPoly(shaft, [poly], 255)
    cuff = np.zeros((H, W), np.uint8)
    cv2.ellipse(cuff, (int(tip[0]), int(tip[1])), (w + 4, w + 2), 0, 0, 360, 255, -1)
    shaft_kpts = _bezier_kpts(pts)
    cuff_kpts = {'center': (int(tip[0]), int(tip[1]), 2)}
    return [
        ('trach_shaft', shaft, rng.randint(90, 160), shaft_kpts),
        ('trach_cuff',  cuff,  rng.randint(180, 230), cuff_kpts),
    ]


def device_mechanical_valve(H, W, anat_masks, rng):
    """Ring + 2 perpendicular strut lines at heart-aorta junction."""
    aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    junction = aorta & heart
    if junction.any():
        pt = sample_point_in_mask(junction, rng)
    elif heart.any():
        pt = sample_point_in_mask(heart, rng)
    else:
        pt = (W // 2, H // 2)
    cx, cy = pt
    r = rng.randint(8, 14)
    angle = rng.uniform(0, 180)
    ring = np.zeros((H, W), np.uint8)
    cv2.ellipse(ring, (cx, cy), (r, max(4, int(r * 0.85))), angle, 0, 360, 255, 2)
    struts = np.zeros((H, W), np.uint8)
    for off_deg in (0, 90):
        a = math.radians(angle + off_deg)
        dx = int(r * math.cos(a)); dy = int(r * math.sin(a))
        cv2.line(struts, (cx - dx, cy - dy), (cx + dx, cy + dy), 255, 1, cv2.LINE_AA)
    return [
        ('valve_ring',   ring,   rng.randint(230, 255)),
        ('valve_struts', struts, rng.randint(210, 245)),
    ]


def device_spinal_rods(H, W, anat_masks, rng):
    """2 rods straddling spine centerline + pedicle screws. Rods now track per-Y centroid
    of the spine mask, so they curve with thoracic kyphosis on lateral views and with any
    scoliosis on frontal — no more straight pasted-on vertical lines on lateral CXRs.
    """
    spine = union_mask(anat_masks, ANAT_GROUPS['spine'])
    if not spine.any():
        lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
        if lungs.any():
            ys, xs = np.where(lungs)
            cx = int(np.median(xs))
            ymin, ymax = int(ys.min()), int(ys.max())
            spine_track = None
        else:
            cx, ymin, ymax = W // 2, H // 4, 3 * H // 4
            spine_track = None
    else:
        ys, xs = np.where(spine)
        ymin, ymax = int(ys.min()), int(ys.max())
        # Sparse anchor points (every ~15px) through spine mask, then Bezier-smooth.
        # Per-Y centroid with per-Y jitter produced stairstep artifacts. Sampling sparsely
        # and Catmull-Rom-interpolating gives a clean smooth curve that still tracks
        # scoliosis/kyphosis.
        step = 15
        anchor_ys = list(range(ymin, ymax + 1, step))
        if anchor_ys[-1] != ymax:
            anchor_ys.append(ymax)
        anchor_xs = []
        for ay in anchor_ys:
            xs_at = xs[ys == ay]
            if len(xs_at):
                anchor_xs.append(int(np.mean(xs_at)))
            else:
                # search nearest scanline
                for d in range(1, 20):
                    xs_above = xs[ys == ay - d]
                    if len(xs_above):
                        anchor_xs.append(int(np.mean(xs_above))); break
                else:
                    anchor_xs.append(int(np.median(xs)))
        # interp + smooth
        from scipy.ndimage import gaussian_filter1d  # noqa: E402
        spine_track = np.interp(np.arange(H), anchor_ys, anchor_xs)
        spine_track = gaussian_filter1d(spine_track, sigma=8.0).astype(np.int32)
        cx = int(np.median(xs))

    rod_off = rng.randint(4, 8)
    parts = []
    # Real fixation usually spans 4-8 vertebral levels, NOT the entire visible spine.
    # Pick a sub-span and let the two rods have slightly different start/end (asymmetry).
    spine_len = ymax - ymin
    span_frac = rng.uniform(0.35, 0.85)                  # how much of spine to cover
    span_len = int(spine_len * span_frac)
    span_start = ymin + rng.randint(0, max(1, spine_len - span_len))
    span_end = span_start + span_len
    # per-rod jitter on endpoints (asymmetric extents — real surgery often is)
    rod_segs = []
    for sign in (-1, +1):
        s0 = span_start + rng.randint(-15, 15)
        s1 = span_end + rng.randint(-15, 15)
        s0 = max(ymin, min(s0, span_end - 30))
        s1 = min(ymax, max(s1, s0 + 30))
        rod_segs.append((sign, s0, s1))

    rod_extents = {}
    for sign, s0, s1 in rod_segs:
        rod = np.zeros((H, W), np.uint8)
        pts = []
        for y in range(s0, s1 + 1):
            tx = (int(spine_track[y]) if spine_track is not None else cx) + sign * rod_off
            pts.append((int(np.clip(tx, 0, W - 1)), y))
        pts = np.array(pts, dtype=np.int32)
        if len(pts) >= 2:
            cv2.polylines(rod, [pts], False, 255, 2, cv2.LINE_AA)
        rod_kpts = {'top': (int(pts[0][0]), int(pts[0][1]), 2),
                    'bottom': (int(pts[-1][0]), int(pts[-1][1]), 2)}
        parts.append(('spinal_rod', rod, rng.randint(220, 250), rod_kpts))
        rod_extents[sign] = (s0, s1)

    # Pedicle screws: only span the OVERLAP region between the two rods (where they
    # share levels). Real screws don't extend past their rod.
    screw_y0 = max(rod_extents[-1][0], rod_extents[+1][0])
    screw_y1 = min(rod_extents[-1][1], rod_extents[+1][1])
    if screw_y1 - screw_y0 > 30:
        n_screws = max(2, (screw_y1 - screw_y0) // rng.randint(25, 35))
        for i in range(n_screws):
            # don't put screw at every level — real ones have some skipped levels
            if rng.random() < 0.15:
                continue
            y = screw_y0 + int((i + 0.5) * (screw_y1 - screw_y0) / max(1, n_screws)) + rng.randint(-3, 3)
            if spine_track is not None and 0 <= y < H:
                scx = int(spine_track[y])
            else:
                scx = cx
            screw = np.zeros((H, W), np.uint8)
            half = rng.randint(8, 14)
            angle = rng.uniform(-8, 8)
            dx = int(half * np.cos(np.radians(angle)))
            dy = int(half * np.sin(np.radians(angle)))
            cv2.line(screw, (scx - dx, y - dy), (scx + dx, y + dy), 255, 2, cv2.LINE_AA)
            parts.append(('pedicle_screw', screw, rng.randint(220, 250), {'center': (scx, y, 2)}))
    return parts


# extend SHAPE_FNS + DEVICE_FNS
SHAPE_FNS.update({
    'ng_tube':            shape_ng_tube,
    'picc_line':          shape_picc_line,
    'annuloplasty_ring':  shape_annuloplasty_ring,
    'chest_tube':         shape_chest_tube,
})
DEVICE_FNS.update({
    'icd':                device_icd,
    'tracheostomy':       device_tracheostomy,
    'mechanical_valve':   device_mechanical_valve,
    'spinal_rods':        device_spinal_rods,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')
print('new shapes:',  ['ng_tube', 'picc_line', 'annuloplasty_ring', 'chest_tube'])
print('new devices:', ['icd', 'tracheostomy', 'mechanical_valve', 'spinal_rods'])

# ===== Sprint 2 SHAPES =====

def shape_tevar(H, W, anat_masks, rng):
    """Endovascular aortic stent-graft: dense mesh clipped to full aorta union (arch + descending)."""
    target = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    mask = np.zeros((H, W), np.uint8)
    if not target.any():
        return mask, 220
    ys, xs = np.where(target)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    cx0 = int(np.median(xs))
    span = max(y1 - y0, x1 - x0) + 40
    cell = rng.randint(4, 8)
    for off in range(-span, span, cell):
        for sign in (+1, -1):
            pts = []
            for y in range(y0, y1):
                x = cx0 + off + sign * (y - y0)
                if 0 <= x < W and target[y, x]:
                    pts.append([x, y])
            if len(pts) > 2:
                cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, 2, cv2.LINE_AA)
    return mask, rng.randint(190, 240)


def shape_swan_ganz(H, W, anat_masks, rng):
    """Long looping Bezier from clavicle → pulmonary artery; balloon ellipse at tip."""
    side = rng.choice([-1, +1])
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        if sel.any():
            i = rng.randrange(int(sel.sum()))
            entry = (int(xs[sel][i]), int(ys[sel][i]))
        else:
            entry = (W // 2 + side * W // 4, H // 4)
    else:
        entry = (W // 2 + side * W // 4, H // 4)
    pa = union_mask(anat_masks, group_ids(['pulmonary artery']))
    if not pa.any():
        pa = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    tip = sample_point_in_mask(pa, rng) or (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    mid_off = np.array([rng.randint(-80, 80), rng.randint(60, 140)], float)
    p1 = p0 + (p3 - p0) * 0.25 + mid_off
    p2 = p0 + (p3 - p0) * 0.55 + np.array([rng.randint(-40, 40), rng.randint(-80, -20)], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, rng.randint(*CATH_W), cv2.LINE_AA)
    cv2.ellipse(mask, (int(tip[0]), int(tip[1])), (rng.randint(3, 6), rng.randint(2, 4)),
                rng.uniform(0, 180), 0, 360, 255, -1)
    return mask, rng.randint(*CATH_GRAY), _bezier_kpts(pts)


def shape_breast_implant(H, W, anat_masks, rng):
    """Large mid-gray ellipse overlaying lower lung field (per side)."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if not lungs.any():
        cx, cy = W // 2, 2 * H // 3
    else:
        ys, xs = np.where(lungs)
        side = rng.choice([-1, +1])
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        if sel.any():
            cx = int(np.mean(xs[sel]))
            cy = int(np.percentile(ys[sel], 75)) + rng.randint(-20, 20)
        else:
            cx, cy = W // 2, 2 * H // 3
    rx = rng.randint(40, 70); ry = rng.randint(30, 55)
    angle = rng.uniform(-25, 25)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
    return mask, rng.randint(140, 200)


# ===== Sprint 2 DEVICES =====

def device_lvad(H, W, anat_masks, rng):
    """Big body @ left ventricle + outflow cannula → aorta + driveline exit."""
    parts = []
    lv = union_mask(anat_masks, group_ids(['heart ventricle left']))
    if not lv.any():
        lv = union_mask(anat_masks, ANAT_GROUPS['heart'])
    body_pt = sample_point_in_mask(lv, rng) or (W // 2, H // 2)
    cx, cy = body_pt
    bw, bh = rng.randint(65, 85), rng.randint(65, 90)
    angle = rng.uniform(-20, 20)
    body, body_color = _textured_body(H, W, cx, cy, bw, bh, rng, angle=angle)
    parts.append(('lvad_body', body, body_color, {'center': (cx, cy, 2)}))

    aorta = union_mask(anat_masks, group_ids(['ascending aorta', 'aortic arch']))
    if not aorta.any():
        aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    aorta_pt = sample_point_in_mask(aorta, rng) if aorta.any() else (cx, max(0, cy - 100))
    a0 = np.array([cx + rng.randint(-10, 10), cy - bh // 2], float)
    a1 = np.array(aorta_pt, float)
    p1 = a0 + (a1 - a0) * 0.5 + np.array([rng.randint(-10, 10), 0], float)
    p2 = a0 + (a1 - a0) * 0.8
    pts = bezier_points(a0, p1, p2, a1, n=64)
    width = rng.randint(5, 8)
    tan = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left = (pts[:-1] + norms * width).astype(np.int32)
    right = (pts[:-1] - norms * width).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    outflow = np.zeros((H, W), np.uint8); cv2.fillPoly(outflow, [poly], 255)
    parts.append(('lvad_outflow', outflow, rng.randint(210, 245), _bezier_kpts(pts)))

    side = rng.choice([-1, +1])
    d0 = np.array([cx + side * (bw // 2 - 2), cy + bh // 4], float)
    d3 = np.array([int(np.clip(cx + side * (W // 4 + rng.randint(20, 60)), 0, W - 1)),
                   min(H - 5, cy + rng.randint(40, 120))], float)
    dp1 = d0 + (d3 - d0) * 0.4 + np.array([rng.randint(-20, 20), rng.randint(-10, 20)], float)
    dp2 = d0 + (d3 - d0) * 0.7 + np.array([rng.randint(-20, 20), rng.randint(-10, 20)], float)
    dpts = bezier_points(d0, dp1, dp2, d3, n=64)
    driveline = np.zeros((H, W), np.uint8)
    cv2.polylines(driveline, [dpts], False, 255, rng.randint(2, 3), cv2.LINE_AA)
    parts.append(('lvad_driveline', driveline, rng.randint(200, 240), _bezier_kpts(dpts)))
    return parts


def device_tunneled_cvc(H, W, anat_masks, rng):
    """Tunneled central catheter (Hickman/Broviac): skin-exit dot + Bezier tube → SVC tip."""
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        i = rng.randrange(len(ys))
        cx = int(xs[i]); cy = int(ys[i]) + rng.randint(40, 80)
    else:
        cx, cy = 3 * W // 4, H // 4
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    exit_dot = np.zeros((H, W), np.uint8)
    cv2.circle(exit_dot, (cx, cy), rng.randint(3, 5), 255, -1)
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    if vess.any():
        ys, xs = np.where(vess)
        top_idx = int(np.argmin(ys))
        tip = (int(xs[top_idx]) + rng.randint(-5, 5), int(ys[top_idx]) + rng.randint(0, 10))
    else:
        tip = (W // 2, H // 4)
    p0 = np.array([cx, cy], float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-25, 25), rng.randint(-15, 15)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-25, 25), rng.randint(-15, 15)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    tube = np.zeros((H, W), np.uint8)
    cv2.polylines(tube, [pts], False, 255, rng.randint(*CATH_W), cv2.LINE_AA)
    return [
        ('tunneled_exit', exit_dot, rng.randint(220, 255), {'center': (cx, cy, 2)}),
        ('tunneled_tube', tube,     rng.randint(*CATH_GRAY), _bezier_kpts(pts)),
    ]


SHAPE_FNS.update({
    'tevar':          shape_tevar,
    'swan_ganz':      shape_swan_ganz,
    'breast_implant': shape_breast_implant,
})
DEVICE_FNS.update({
    'lvad':          device_lvad,
    'tunneled_cvc':  device_tunneled_cvc,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')
print('new shapes:',  ['tevar', 'swan_ganz', 'breast_implant'])
print('new devices:', ['lvad', 'tunneled_cvc'])

# ===== Sprint 3 SHAPES =====

def shape_gossypiboma(H, W, anat_masks, rng):
    """Sin-wave serpentine line — radio-opaque thread of retained surgical sponge."""
    body = anat_masks.any(axis=0)
    cx, cy = sample_point_in_mask(body, rng) or (W // 2, H // 2)
    length = rng.randint(60, 120)
    angle = rng.uniform(0, 2 * np.pi)
    n_pts = 64
    t = np.linspace(0, 1, n_pts)
    bx = cx + (t - 0.5) * length * np.cos(angle)
    by = cy + (t - 0.5) * length * np.sin(angle)
    amp = rng.randint(8, 18); freq = rng.uniform(3, 7)
    perp_x = -np.sin(angle); perp_y = np.cos(angle)
    sx = bx + amp * np.sin(freq * 2 * np.pi * t) * perp_x
    sy = by + amp * np.sin(freq * 2 * np.pi * t) * perp_y
    pts = np.stack([sx, sy], axis=1).astype(np.int32)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    kpts = {'start': (int(pts[0][0]), int(pts[0][1]), 2),
            'end':   (int(pts[-1][0]), int(pts[-1][1]), 2)}
    return mask, rng.randint(220, 255), kpts


def shape_ivc_filter(H, W, anat_masks, rng):
    """Radial-spoke cone umbrella, apex up, below diaphragm midline (IVC anatomic position)."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        cx = int(np.median(xs)) + rng.randint(-10, 25)  # IVC slightly right of midline
        cy = int(ys.max()) + rng.randint(10, 40)
    else:
        cx, cy = W // 2, 3 * H // 4
    cx = int(np.clip(cx, 30, W - 30)); cy = int(np.clip(cy, 30, H - 30))
    n_spokes = rng.randint(6, 10)
    spoke_len = rng.randint(15, 25)
    mask = np.zeros((H, W), np.uint8)
    for i in range(n_spokes):
        a = -math.pi / 2 + (i / max(1, n_spokes - 1) - 0.5) * math.pi * 0.6
        dx = int(spoke_len * math.cos(a)); dy = int(spoke_len * math.sin(a))
        cv2.line(mask, (cx, cy), (cx + dx, cy - dy), 255, 1, cv2.LINE_AA)
    kpts = {'apex': (cx, cy - spoke_len, 2), 'base': (cx, cy, 2)}
    return mask, rng.randint(220, 255), kpts


def shape_vertebroplasty(H, W, anat_masks, rng):
    """Dense fill of one thoracic vertebra mask (T1-T12)."""
    mask = np.zeros((H, W), np.uint8)
    t_names = [f'vertebrae T{k}' for k in range(1, 13)]
    t_ids = group_ids(t_names)
    if not t_ids:
        return mask, 240
    rng.shuffle(t_ids)
    for vid in t_ids:
        if vid < anat_masks.shape[0] and anat_masks[vid].any():
            v = anat_masks[vid].astype(np.uint8) * 255
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            v = cv2.dilate(v, k)
            mask = np.maximum(mask, v)
            break
    return mask, rng.randint(230, 255)


def shape_watchman(H, W, anat_masks, rng):
    """Small mesh umbrella at left atrial appendage edge."""
    la = union_mask(anat_masks, group_ids(['heart atrium left']))
    if not la.any():
        la = union_mask(anat_masks, ANAT_GROUPS['heart'])
    pt = sample_point_in_mask(la, rng) or (W // 2, H // 2)
    cx, cy = pt
    r = rng.randint(8, 14)
    n_spokes = rng.randint(6, 10)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (cx, cy), (r, max(3, int(r * 0.6))), 0, 0, 180, 255, 1, cv2.LINE_AA)
    apex = (cx, cy - max(3, int(r * 0.6)))
    for i in range(n_spokes):
        a = -math.pi / 2 + (i / max(1, n_spokes - 1) - 0.5) * math.pi
        dx = int(r * math.cos(a)); dy = int(r * math.sin(a))
        cv2.line(mask, apex, (cx + dx, cy + dy), 255, 1, cv2.LINE_AA)
    return mask, rng.randint(220, 255)


def shape_cardiomems(H, W, anat_masks, rng):
    """Tiny rotated rectangle inside pulmonary artery."""
    pa = union_mask(anat_masks, group_ids(['pulmonary artery']))
    if not pa.any():
        pa = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    pt = sample_point_in_mask(pa, rng) or (W // 2, H // 2)
    cx, cy = pt
    w = rng.randint(4, 7); h = rng.randint(2, 4)
    angle = rng.uniform(0, 180)
    mask = np.zeros((H, W), np.uint8)
    box = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(angle))).astype(np.int32)
    cv2.fillPoly(mask, [box], 255)
    return mask, rng.randint(230, 255)


SHAPE_FNS.update({
    'gossypiboma':    shape_gossypiboma,
    'ivc_filter':     shape_ivc_filter,
    'vertebroplasty': shape_vertebroplasty,
    'watchman':       shape_watchman,
    'cardiomems':     shape_cardiomems,
})
print(f'shapes: {len(SHAPE_FNS)} | new: gossypiboma, ivc_filter, vertebroplasty, watchman, cardiomems')

# ===== Sprint 4 SHAPES =====

def shape_lumpectomy_clips(H, W, anat_masks, rng):
    """Tight cluster of 3-6 tiny clips marking lumpectomy site."""
    body = anat_masks.any(axis=0)
    pt = sample_point_in_mask(body, rng) or (W // 2, H // 2)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    n = rng.randint(3, 6)
    for _ in range(n):
        dx = rng.randint(-15, 15); dy = rng.randint(-15, 15)
        length = rng.randint(4, 8); thick = rng.randint(1, 2)
        a = rng.uniform(0, math.pi)
        ddx = int(length * math.cos(a)); ddy = int(length * math.sin(a))
        cv2.line(mask, (cx + dx - ddx, cy + dy - ddy), (cx + dx + ddx, cy + dy + ddy), 255, thick, cv2.LINE_AA)
    return mask, rng.randint(230, 255)


def shape_cabg_clips(H, W, anat_masks, rng):
    """Series of small bright clips along Bezier route from sternum → coronary territory."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        sternum_x = int(np.median(xs))
        sternum_y_top = int(ys.min()) + rng.randint(20, 60)
    else:
        sternum_x = W // 2; sternum_y_top = H // 4
    p0 = np.array([sternum_x + rng.randint(-10, 10), sternum_y_top], float)
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    end = sample_point_in_mask(heart, rng) or (sternum_x, sternum_y_top + 100)
    p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-30, 30), rng.randint(-10, 30)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-30, 30), rng.randint(-10, 30)], float)
    pts = bezier_points(p0, p1, p2, p3, n=64)
    mask = np.zeros((H, W), np.uint8)
    n_clips = rng.randint(3, 8)
    step = max(1, len(pts) // n_clips)
    tan = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    for i in range(0, len(pts) - 1, step):
        c = pts[i]
        n = norms[min(i, len(norms) - 1)]
        half = rng.randint(2, 4)
        a = (c + n * half).astype(np.int32); b = (c - n * half).astype(np.int32)
        cv2.line(mask, tuple(a), tuple(b), 255, rng.randint(1, 2), cv2.LINE_AA)
    return mask, rng.randint(230, 255)


# ===== Sprint 4 DEVICES =====

def device_vp_shunt(H, W, anat_masks, rng):
    """Reservoir near top of frame (cranium proxy) + long tube down chest wall to abdomen."""
    rx = rng.randint(W // 3, 2 * W // 3); ry = rng.randint(5, 25)
    r = rng.randint(5, 10)
    res = np.zeros((H, W), np.uint8)
    cv2.circle(res, (rx, ry), r, 255, -1)
    side = rng.choice([-1, +1])
    p0 = np.array([rx, ry + r], float)
    p3 = np.array([int(np.clip(rx + side * rng.randint(40, 100), 10, W - 10)),
                   H - rng.randint(5, 50)], float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([side * rng.randint(10, 30), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([side * rng.randint(0, 20), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    tube = np.zeros((H, W), np.uint8)
    cv2.polylines(tube, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    return [
        ('vp_reservoir', res,  rng.randint(220, 255), {'center': (rx, ry, 2)}),
        ('vp_tube',      tube, rng.randint(200, 245), _bezier_kpts(pts)),
    ]


def device_shoulder_prosthesis(H, W, anat_masks, rng):
    """Ball head (circle) + tapered humeral stem (polygon) inside humerus mask."""
    humerus = union_mask(anat_masks, ANAT_GROUPS['humerus'])
    if humerus.any():
        ys, xs = np.where(humerus)
        side = rng.choice([-1, +1])
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        if sel.any():
            ys_s = ys[sel]; xs_s = xs[sel]
            cx = int(np.median(xs_s))
            ymin = int(ys_s.min()); ymax = int(ys_s.max())
        else:
            cx = rng.choice([W // 8, 7 * W // 8])
            ymin, ymax = H // 8, H // 3
    else:
        cx = rng.choice([W // 8, 7 * W // 8])
        ymin, ymax = H // 8, H // 3
    cx = int(np.clip(cx, 15, W - 15))
    r_head = rng.randint(10, 18)
    head_y = max(r_head, ymin)
    stem_top = head_y + r_head
    stem_bot = min(H - 5, stem_top + rng.randint(50, 120))
    head = np.zeros((H, W), np.uint8)
    cv2.circle(head, (cx, head_y), r_head, 255, -1)
    stem = np.zeros((H, W), np.uint8)
    poly = np.array([
        [cx - 6, stem_top], [cx + 6, stem_top],
        [cx + 3, stem_bot], [cx - 3, stem_bot]
    ], np.int32)
    cv2.fillPoly(stem, [poly], 255)
    return [
        ('shoulder_head', head, rng.randint(230, 255), {'center': (cx, head_y, 2)}),
        ('shoulder_stem', stem, rng.randint(225, 255),
         {'top': (cx, stem_top, 2), 'bottom': (cx, stem_bot, 2)}),
    ]


def device_tissue_expander(H, W, anat_masks, rng):
    """Large oval body + small radio-opaque magnet port disc at center."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        side = rng.choice([-1, +1])
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        if sel.any():
            cx = int(np.mean(xs[sel]))
            cy = int(np.percentile(ys[sel], 70)) + rng.randint(-15, 15)
        else:
            cx, cy = W // 2, 2 * H // 3
    else:
        cx, cy = W // 2, 2 * H // 3
    rx = rng.randint(40, 65); ry = rng.randint(30, 50)
    angle = rng.uniform(-25, 25)
    body = np.zeros((H, W), np.uint8)
    cv2.ellipse(body, (cx, cy), (rx, ry), angle, 0, 360, 255, -1)
    disc = np.zeros((H, W), np.uint8)
    cv2.circle(disc, (cx + rng.randint(-5, 5), cy + rng.randint(-5, 5)), rng.randint(3, 6), 255, -1)
    return [
        ('expander_body',      body, rng.randint(140, 200)),
        ('expander_port_disc', disc, rng.randint(230, 255)),
    ]


SHAPE_FNS.update({
    'lumpectomy_clips': shape_lumpectomy_clips,
    'cabg_clips':       shape_cabg_clips,
})
DEVICE_FNS.update({
    'vp_shunt':            device_vp_shunt,
    'shoulder_prosthesis': device_shoulder_prosthesis,
    'tissue_expander':     device_tissue_expander,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')
print('new shapes:',  ['lumpectomy_clips', 'cabg_clips'])
print('new devices:', ['vp_shunt', 'shoulder_prosthesis', 'tissue_expander'])

# ===== Sprint 5 SHAPES =====

def shape_tracheal_stent(H, W, anat_masks, rng):
    """Tracheal stent — cylindrical mesh with proximal/distal flared ends + repeating
    diamond struts. Real tracheal stents have characteristic radiopaque markers at
    both ends (highlighted in bright) + regular diamond mesh body in between.
    """
    target = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    mask = np.zeros((H, W), np.uint8)
    if not target.any():
        return mask, 220, {}
    ys, xs = np.where(target)
    y0 = int(ys.min()) + rng.randint(0, 15)
    y1 = min(y0 + rng.randint(40, 80), int(ys.max()))   # finite stent length
    cx0 = int(np.median(xs))
    # half-width: stent fits inside trachea
    half = max(4, rng.randint(5, 9))
    # 1) proximal + distal "flared end" marker rings (more radiopaque than body)
    for y_marker in (y0, y1):
        cv2.line(mask, (cx0 - half - 1, y_marker), (cx0 + half + 1, y_marker), 255, 2, cv2.LINE_AA)
    # 2) outer walls (slightly less bright than markers) — 2px so the stent
    # walls survive downstream resize/JPEG.
    cv2.line(mask, (cx0 - half, y0), (cx0 - half, y1), 200, 2, cv2.LINE_AA)
    cv2.line(mask, (cx0 + half, y0), (cx0 + half, y1), 200, 2, cv2.LINE_AA)
    # 3) regular diamond struts inside the body
    n_diamonds = max(3, (y1 - y0) // rng.randint(8, 14))
    for i in range(n_diamonds):
        cy = y0 + int((i + 0.5) * (y1 - y0) / n_diamonds)
        d_h = max(4, (y1 - y0) // (n_diamonds * 2) - 1)
        diamond = np.array([
            [cx0, cy - d_h], [cx0 + half, cy],
            [cx0, cy + d_h], [cx0 - half, cy],
        ], np.int32)
        cv2.polylines(mask, [diamond], True, 180, 1, cv2.LINE_AA)
    kpts = {'proximal_end': (cx0, y0, 2), 'tip': (cx0, y1, 2)}
    return mask, rng.randint(200, 245), kpts


def shape_aspirated_fb(H, W, anat_masks, rng):
    """Small dense object in lungs / bronchi — aspirated foreign body."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    target = lungs if lungs.any() else anat_masks.any(axis=0)
    pt = sample_point_in_mask(target, rng) or (W // 2, H // 2)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    kind = rng.choice(['circle', 'rect', 'irregular'])
    if kind == 'circle':
        cv2.circle(mask, (cx, cy), rng.randint(4, 10), 255, -1)
    elif kind == 'rect':
        w_ = rng.randint(6, 14); h_ = rng.randint(4, 10)
        cv2.rectangle(mask, (cx - w_ // 2, cy - h_ // 2), (cx + w_ // 2, cy + h_ // 2), 255, -1)
    else:
        n_pts = rng.randint(5, 8)
        pts = []
        for i in range(n_pts):
            a = 2 * math.pi * i / n_pts
            r_ = rng.randint(4, 8)
            pts.append([cx + int(r_ * math.cos(a)), cy + int(r_ * math.sin(a))])
        cv2.fillPoly(mask, [np.array(pts, np.int32)], 255)
    return mask, rng.randint(230, 255)


# ===== Sprint 5 DEVICES =====

def device_ecmo(H, W, anat_masks, rng):
    """2 thick cannulas entering from upper frame (jugular/subclavian) into great vessels / heart."""
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    target = vess if vess.any() else heart
    parts = []
    for kind, side_choice in (('ecmo_return', -1), ('ecmo_drain', +1)):
        entry = (W // 2 + side_choice * rng.randint(20, 60), rng.randint(0, 30))
        tip = sample_point_in_mask(target, rng) if target.any() else (W // 2, H // 2)
        p0 = np.array(entry, float); p3 = np.array(tip, float)
        p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-20, 20), 0], float)
        p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-20, 20), 0], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        w = rng.randint(7, 11)
        tan = np.diff(pts, axis=0).astype(float)
        norms = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
        norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
        left = (pts[:-1] + norms * w).astype(np.int32)
        right = (pts[:-1] - norms * w).astype(np.int32)
        poly = np.concatenate([left, right[::-1]], axis=0)
        mask = np.zeros((H, W), np.uint8)
        cv2.fillPoly(mask, [poly], 255)
        cv2.polylines(mask, [left], False, 255, 1)
        cv2.polylines(mask, [right], False, 255, 1)
        parts.append((kind, mask, rng.randint(180, 230), _bezier_kpts(pts)))
    return parts


def device_bronchial_y_stent(H, W, anat_masks, rng):
    """Y-shape mesh: trunk in distal trachea + 2 branches into main bronchi at carina."""
    parts = []
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        bot_idx = int(np.argmax(ys))
        carina = (int(xs[bot_idx]), int(ys[bot_idx]))
    else:
        carina = (W // 2, H // 2)
    cx, cy = carina
    half_w = rng.randint(5, 8)
    trunk_top = (cx, max(0, cy - rng.randint(40, 60)))
    trunk = np.zeros((H, W), np.uint8)
    cv2.line(trunk, (trunk_top[0] - half_w, trunk_top[1]), (cx - half_w, cy), 255, 2, cv2.LINE_AA)
    cv2.line(trunk, (trunk_top[0] + half_w, trunk_top[1]), (cx + half_w, cy), 255, 2, cv2.LINE_AA)
    for k in range(trunk_top[1], cy, 5):
        cv2.line(trunk, (cx - half_w, k), (cx + half_w, k + 3), 255, 1, cv2.LINE_AA)
    trunk_kpts = {'proximal_end': (int(trunk_top[0]), int(trunk_top[1]), 2),
                  'tip': (cx, cy, 2)}
    parts.append(('bronchial_y_trunk', trunk, rng.randint(190, 240), trunk_kpts))
    for side in (-1, +1):
        be = (cx + side * rng.randint(40, 70), min(H - 1, cy + rng.randint(40, 80)))
        branch = np.zeros((H, W), np.uint8)
        cv2.line(branch, (cx - half_w, cy), (be[0] - half_w, be[1]), 255, 2, cv2.LINE_AA)
        cv2.line(branch, (cx + half_w, cy), (be[0] + half_w, be[1]), 255, 2, cv2.LINE_AA)
        steps = 8
        for k in range(steps):
            t = k / steps
            x0 = int(cx - half_w + t * (be[0] - half_w - (cx - half_w)))
            y0 = int(cy + t * (be[1] - cy))
            x1 = int(cx + half_w + t * (be[0] + half_w - (cx + half_w)))
            cv2.line(branch, (x0, y0), (x1, y0 + 3), 255, 1, cv2.LINE_AA)
        branch_kpts = {'proximal_end': (cx, cy, 2),
                       'tip': (int(be[0]), int(be[1]), 2)}
        parts.append(('bronchial_y_branch', branch, rng.randint(190, 240), branch_kpts))
    return parts


def device_crt_pacemaker(H, W, anat_masks, rng):
    """CRT biventricular pacemaker: body + 3 leads (RA + RV + LV via coronary sinus)."""
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        i = rng.randrange(len(ys))
        cx = int(xs[i]); cy = int(ys[i]) + rng.randint(20, 60)
    else:
        cx, cy = W // 4, H // 4
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    parts = []
    bw, bh = rng.randint(28, 42), rng.randint(35, 52)
    body = np.zeros((H, W), np.uint8)
    cv2.ellipse(body, (cx, cy), (bw // 2, bh // 2), 0, 0, 360, 255, -1)
    parts.append(('crt_body', body, rng.randint(220, 255), {'center': (cx, cy, 2)}))
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    lead_targets = [
        ('crt_lead_ra', ['heart atrium right']),
        ('crt_lead_rv', ['heart ventricle right']),
        ('crt_lead_lv', ['heart ventricle left']),
    ]
    for label, names in lead_targets:
        chamber = union_mask(anat_masks, group_ids(names))
        if not chamber.any():
            chamber = heart_u
        end = sample_point_in_mask(chamber, rng) if chamber.any() else (cx, min(H - 1, cy + 150))
        p0 = np.array([cx, cy + bh // 2], float)
        p3 = np.array(end, float)
        p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-25, 25), rng.randint(-15, 30)], float)
        p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-35, 35), rng.randint(-15, 30)], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        lead = np.zeros((H, W), np.uint8)
        cv2.polylines(lead, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
        parts.append((label, lead, rng.randint(200, 245), _bezier_kpts(pts)))
    return parts


SHAPE_FNS.update({
    'tracheal_stent': shape_tracheal_stent,
    'aspirated_fb':   shape_aspirated_fb,
})
DEVICE_FNS.update({
    'ecmo':              device_ecmo,
    'bronchial_y_stent': device_bronchial_y_stent,
    'crt_pacemaker':     device_crt_pacemaker,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')

# ===== Sprint 6 SHAPES =====

def _midline_stomach_pt(H, W, anat_masks, rng):
    """Approximate esophagus/stomach point: midline below lung lower edge."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        cx = int(np.median(xs)) + rng.randint(-15, 15)
        cy = int(np.percentile(ys, 95)) + rng.randint(0, 40)
    else:
        cx, cy = W // 2, 3 * H // 4
    return int(np.clip(cx, 10, W - 10)), int(np.clip(cy, 10, H - 10))


def shape_coin_ingested(H, W, anat_masks, rng):
    cx, cy = _midline_stomach_pt(H, W, anat_masks, rng)
    r = rng.randint(6, 14)
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    return mask, rng.randint(240, 255)


def shape_button_battery(H, W, anat_masks, rng):
    """Filled circle + concentric inner halo (button battery double-rim signature)."""
    cx, cy = _midline_stomach_pt(H, W, anat_masks, rng)
    r = rng.randint(6, 12)
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    cv2.circle(mask, (cx, cy), max(2, r - 3), 255, 1)
    return mask, rng.randint(240, 255)


def shape_magnet_pair(H, W, anat_masks, rng):
    cx, cy = _midline_stomach_pt(H, W, anat_masks, rng)
    r = rng.randint(4, 8)
    mask = np.zeros((H, W), np.uint8)
    cv2.circle(mask, (cx - r, cy), r, 255, -1)
    cv2.circle(mask, (cx + r, cy), r, 255, -1)
    return mask, rng.randint(240, 255)


def shape_defib_pads(H, W, anat_masks, rng):
    """1-2 large low-opacity rectangles on chest skin (defib pad placement)."""
    mask = np.zeros((H, W), np.uint8)
    n = rng.randint(1, 3)
    for _ in range(n):
        w = rng.randint(50, 90); h = rng.randint(40, 70)
        x = rng.randint(0, max(W - w, 1))
        y = rng.randint(0, max(H - h, 1))
        cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
    return mask, rng.randint(100, 160)


def shape_pacing_wire(H, W, anat_masks, rng):
    """Thin near-straight wire from lateral chest wall to heart."""
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    end = sample_point_in_mask(heart, rng) or (W // 2, H // 2)
    side = rng.choice([-1, +1])
    entry = (int(np.clip(W // 2 + side * rng.randint(60, 120), 0, W - 1)), rng.randint(20, 80))
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-8, 8), rng.randint(-8, 8)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-8, 8), rng.randint(-8, 8)], float)
    pts = bezier_points(p0, p1, p2, p3, n=64)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, 1, cv2.LINE_AA)
    return mask, rng.randint(220, 255)


SHAPE_FNS.update({
    'coin_ingested':  shape_coin_ingested,
    'button_battery': shape_button_battery,
    'magnet_pair':    shape_magnet_pair,
    'defib_pads':     shape_defib_pads,
    'pacing_wire':    shape_pacing_wire,
})
print(f'shapes: {len(SHAPE_FNS)}')

# ===== Sprint 7 SHAPES =====

def shape_mediastinal_drain(H, W, anat_masks, rng):
    """Thick tube near sternum centerline w/ side-hole gaps."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        cx = int(np.median(xs))
        ymin = int(ys.min()) + 30
        ymax = min(H - 5, int(ys.max()))
    else:
        cx, ymin, ymax = W // 2, H // 4, 3 * H // 4
    entry = (cx + rng.randint(-10, 10), ymin - rng.randint(20, 40))
    end_y = rng.randint(ymin, max(ymin + 10, (ymax - ymin) // 2 + ymin))
    end = (cx + rng.randint(-15, 15), end_y)
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.5 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.8 + np.array([rng.randint(-10, 10), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    w = rng.randint(3, 6)
    tan = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left = (pts[:-1] + norms * w).astype(np.int32)
    right = (pts[:-1] - norms * w).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    mask = np.zeros((H, W), np.uint8)
    cv2.fillPoly(mask, [poly], 255)
    # side-hole gaps
    step = max(1, len(pts) // 6)
    for i in range(step, len(pts), step):
        if i < len(pts):
            cv2.circle(mask, tuple(pts[i].tolist()), 2, 0, -1)
    return mask, rng.randint(80, 160), _bezier_kpts(pts)


def shape_embolization_coils(H, W, anat_masks, rng):
    """Cluster of tightly-wound Archimedean spirals in vessels."""
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    pt = sample_point_in_mask(vess, rng) if vess.any() else (W // 2, H // 2)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    n_coils = rng.randint(2, 5)
    for _ in range(n_coils):
        ccx = cx + rng.randint(-12, 12)
        ccy = cy + rng.randint(-12, 12)
        angle0 = rng.uniform(0, 2 * math.pi)
        a, b = 0.5, 0.4
        pts = []
        for t in np.linspace(0, 4 * math.pi, 40):
            r = a + b * t
            pts.append([int(ccx + r * math.cos(t + angle0)),
                        int(ccy + r * math.sin(t + angle0))])
        cv2.polylines(mask, [np.array(pts, np.int32)], False, 255, 1, cv2.LINE_AA)
    return mask, rng.randint(230, 255)


def shape_esophageal_stent(H, W, anat_masks, rng):
    """Esophageal stent — wider tube (larger than tracheal), flared ends, diamond mesh
    pattern. Positioned in posterior mediastinum (midline, slightly behind trachea)."""
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        cx = int(np.median(xs)) + rng.randint(-5, 5)
        y0 = int(ys.min()) + rng.randint(20, 50)        # below thoracic inlet
        y1 = min(H - 5, y0 + rng.randint(60, 120))      # finite stent
    else:
        cx, y0, y1 = W // 2, H // 3, 3 * H // 4
    mask = np.zeros((H, W), np.uint8)
    half = rng.randint(7, 12)                            # esophageal > tracheal
    # proximal + distal flared end markers
    flare = 3
    for ym in (y0, y1):
        cv2.line(mask, (cx - half - flare, ym), (cx + half + flare, ym), 255, 2, cv2.LINE_AA)
    # walls
    cv2.line(mask, (cx - half, y0), (cx - half, y1), 200, 1, cv2.LINE_AA)
    cv2.line(mask, (cx + half, y0), (cx + half, y1), 200, 1, cv2.LINE_AA)
    # diamond struts
    n_diamonds = max(4, (y1 - y0) // rng.randint(10, 18))
    for i in range(n_diamonds):
        cy = y0 + int((i + 0.5) * (y1 - y0) / n_diamonds)
        d_h = max(5, (y1 - y0) // (n_diamonds * 2) - 1)
        diamond = np.array([
            [cx, cy - d_h], [cx + half, cy],
            [cx, cy + d_h], [cx - half, cy],
        ], np.int32)
        cv2.polylines(mask, [diamond], True, 180, 1, cv2.LINE_AA)
    kpts = {'proximal_end': (cx, y0, 2), 'tip': (cx, y1, 2)}
    return mask, rng.randint(200, 245), kpts


# ===== Sprint 7 DEVICES =====

def device_halo_traction(H, W, anat_masks, rng):
    """External cervical halo ring at top + 4 support posts descending."""
    parts = []
    cx = W // 2 + rng.randint(-10, 10)
    cy = rng.randint(15, 40)
    rx = rng.randint(min(W, H) // 4, min(W, H) // 3)
    ry = max(rx // 4, 10)
    ring = np.zeros((H, W), np.uint8)
    cv2.ellipse(ring, (cx, cy), (rx, ry), 0, 0, 360, 255, 3)
    parts.append(('halo_ring', ring, rng.randint(230, 255), {'center': (cx, cy, 2)}))
    for fraction in (-0.7, -0.3, 0.3, 0.7):
        post_top = (int(np.clip(cx + fraction * rx, 0, W - 1)), cy + ry)
        post_bot = (int(np.clip(cx + fraction * rx * 1.2, 0, W - 1)),
                    min(H - 1, cy + ry + rng.randint(60, 120)))
        post = np.zeros((H, W), np.uint8)
        cv2.line(post, post_top, post_bot, 255, 2, cv2.LINE_AA)
        post_kpts = {'top': (int(post_top[0]), int(post_top[1]), 2),
                     'bottom': (int(post_bot[0]), int(post_bot[1]), 2)}
        parts.append(('halo_post', post, rng.randint(220, 250), post_kpts))
    return parts


def device_sengstaken(H, W, anat_masks, rng):
    """Tube + esophageal balloon + gastric balloon (Sengstaken-Blakemore variceal tamponade)."""
    parts = []
    entry = (W // 2 + rng.randint(-15, 15), rng.randint(0, 30))
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, _ = np.where(lungs)
        end_y = min(H - 10, int(ys.max()) + rng.randint(30, 70))
    else:
        end_y = H - 30
    end = (W // 2 + rng.randint(-15, 15), end_y)
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-10, 10), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    tube = np.zeros((H, W), np.uint8)
    cv2.polylines(tube, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    parts.append(('sb_tube', tube, rng.randint(200, 245), _bezier_kpts(pts)))
    eso_pt = pts[int(0.55 * len(pts))]
    eso_bal = np.zeros((H, W), np.uint8)
    cv2.ellipse(eso_bal, tuple(int(v) for v in eso_pt),
                (rng.randint(6, 10), rng.randint(12, 20)), 0, 0, 360, 255, -1)
    parts.append(('sb_esophageal_balloon', eso_bal, rng.randint(160, 210),
                  {'center': (int(eso_pt[0]), int(eso_pt[1]), 2)}))
    gas_bal = np.zeros((H, W), np.uint8)
    cv2.ellipse(gas_bal, (int(end[0]), int(end[1])),
                (rng.randint(10, 18), rng.randint(12, 22)), 0, 0, 360, 255, -1)
    parts.append(('sb_gastric_balloon', gas_bal, rng.randint(170, 220),
                  {'center': (int(end[0]), int(end[1]), 2)}))
    return parts


SHAPE_FNS.update({
    'mediastinal_drain':  shape_mediastinal_drain,
    'embolization_coils': shape_embolization_coils,
    'esophageal_stent':   shape_esophageal_stent,
})
DEVICE_FNS.update({
    'halo_traction': device_halo_traction,
    'sengstaken':    device_sengstaken,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')

# ===== Sprint 8 SHAPES =====

def shape_iabp(H, W, anat_masks, rng):
    """IABP: catheter from femoral entry through descending aorta + elongated balloon at arch level."""
    aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    mask = np.zeros((H, W), np.uint8)
    if not aorta.any():
        return mask, rng.randint(180, 230), {}
    ys, xs = np.where(aorta)
    y0, y1 = int(ys.min()), int(ys.max())
    cx = int(np.median(xs))
    entry = (cx + rng.randint(-15, 15), min(H - 1, y1 + rng.randint(20, 60)))
    tip = (cx + rng.randint(-15, 15), max(0, y0 + rng.randint(0, 15)))
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-8, 8), 0], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-8, 8), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    cv2.polylines(mask, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    # balloon segment along middle 35-70% of catheter
    mid_start = int(0.35 * len(pts))
    mid_end = int(0.7 * len(pts))
    bal_pts = pts[mid_start:mid_end]
    if len(bal_pts) >= 2:
        tan = np.diff(bal_pts, axis=0).astype(float)
        norms = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
        norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
        w_b = rng.randint(4, 7)
        left = (bal_pts[:-1] + norms * w_b).astype(np.int32)
        right = (bal_pts[:-1] - norms * w_b).astype(np.int32)
        poly = np.concatenate([left, right[::-1]], axis=0)
        cv2.fillPoly(mask, [poly], 255)
    return mask, rng.randint(180, 230), _bezier_kpts(pts)


def shape_mitra_clip(H, W, anat_masks, rng):
    """2 tiny adjacent X-clips at mitral position (heart center)."""
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    pt = sample_point_in_mask(heart, rng) or (W // 2, H // 2)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    offset = rng.randint(3, 6)
    for sign in (-1, +1):
        ccx = cx + sign * offset
        half = rng.randint(2, 4)
        cv2.line(mask, (ccx - half, cy - half), (ccx + half, cy + half), 255, 1, cv2.LINE_AA)
        cv2.line(mask, (ccx - half, cy + half), (ccx + half, cy - half), 255, 1, cv2.LINE_AA)
    return mask, rng.randint(230, 255)


# ===== Sprint 8 DEVICES =====

def device_dialysis_catheter(H, W, anat_masks, rng):
    """Large-bore tunneled CVC w/ skin exit + thick tube + Y-split 2-lumen tip."""
    parts = []
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        i = rng.randrange(len(ys))
        cx = int(xs[i]); cy = int(ys[i]) + rng.randint(40, 90)
    else:
        cx, cy = 3 * W // 4, H // 4
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    exit_dot = np.zeros((H, W), np.uint8)
    cv2.circle(exit_dot, (cx, cy), rng.randint(4, 7), 255, -1)
    parts.append(('dialysis_exit', exit_dot, rng.randint(220, 255), {'center': (cx, cy, 2)}))
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    if vess.any():
        ys, xs = np.where(vess)
        top_idx = int(np.argmin(ys))
        tip = (int(xs[top_idx]) + rng.randint(-5, 5), int(ys[top_idx]) + rng.randint(0, 15))
    else:
        tip = (W // 2, H // 4)
    p0 = np.array([cx, cy], float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-20, 20), rng.randint(-15, 15)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-20, 20), rng.randint(-15, 15)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    tube = np.zeros((H, W), np.uint8)
    cv2.polylines(tube, [pts], False, 255, rng.randint(*CATH_W), cv2.LINE_AA)
    parts.append(('dialysis_tube', tube, rng.randint(*CATH_GRAY), _bezier_kpts(pts)))
    tip_arr = np.array(tip, float)
    pre_idx = max(0, len(pts) - 6)
    tan = (tip_arr - pts[pre_idx].astype(float))
    norm_t = tan / (np.linalg.norm(tan) + 1e-6)
    perp = np.array([-norm_t[1], norm_t[0]])
    split_len = rng.randint(10, 18)
    y_split = np.zeros((H, W), np.uint8)
    arms = []
    for sign in (-1, +1):
        endp = tip_arr + norm_t * 5 + perp * sign * split_len
        arms.append((int(endp[0]), int(endp[1])))
        cv2.line(y_split,
                 (int(tip_arr[0]), int(tip_arr[1])),
                 (int(endp[0]), int(endp[1])),
                 255, rng.randint(1, 2), cv2.LINE_AA)
    y_kpts = {'junction': (int(tip_arr[0]), int(tip_arr[1]), 2),
              'arm_a': (arms[0][0], arms[0][1], 2),
              'arm_b': (arms[1][0], arms[1][1], 2)}
    parts.append(('dialysis_y_split', y_split, rng.randint(210, 250), y_kpts))
    return parts


def device_gtube(H, W, anat_masks, rng):
    """G-tube/PEG: skin disc at lower abdomen + retention balloon ring inside stomach."""
    parts = []
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        cx = int(np.median(xs)) + rng.randint(-30, 30)
        cy = min(H - 15, int(ys.max()) + rng.randint(30, 70))
    else:
        cx, cy = W // 2, 4 * H // 5
    cx = int(np.clip(cx, 30, W - 30)); cy = int(np.clip(cy, 30, H - 30))
    disc = np.zeros((H, W), np.uint8)
    cv2.circle(disc, (cx, cy), rng.randint(6, 12), 255, -1)
    parts.append(('gtube_disc', disc, rng.randint(220, 255)))
    ring = np.zeros((H, W), np.uint8)
    by = min(H - 5, cy + rng.randint(8, 15))
    cv2.ellipse(ring, (cx, by),
                (rng.randint(8, 14), rng.randint(6, 10)), 0, 0, 360, 255, 1)
    parts.append(('gtube_balloon', ring, rng.randint(180, 220)))
    return parts


def device_double_lumen_et(H, W, anat_masks, rng):
    """ET tube w/ 2 cuffs: proximal main-airway + distal bronchial (one side)."""
    parts = []
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        top_idx = int(np.argmin(ys)); bot_idx = int(np.argmax(ys))
        entry = (int(xs[top_idx]) + rng.randint(-10, 10), int(ys[top_idx]) - rng.randint(30, 60))
        side = rng.choice([-1, +1])
        tip = (int(xs[bot_idx]) + side * rng.randint(15, 40), int(ys[bot_idx]) + rng.randint(10, 30))
    else:
        entry = (W // 2, 20); tip = (W // 2 + 20, H // 2)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-15, 15), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    w_t = rng.randint(5, 9)
    tan = np.diff(pts, axis=0).astype(float)
    norms = np.stack([-tan[:, 1], tan[:, 0]], axis=1)
    norms /= (np.linalg.norm(norms, axis=1, keepdims=True) + 1e-6)
    left = (pts[:-1] + norms * w_t).astype(np.int32)
    right = (pts[:-1] - norms * w_t).astype(np.int32)
    poly = np.concatenate([left, right[::-1]], axis=0)
    shaft = np.zeros((H, W), np.uint8); cv2.fillPoly(shaft, [poly], 255)
    parts.append(('dl_et_shaft', shaft, rng.randint(90, 160), _bezier_kpts(pts)))
    prox_idx = int(0.6 * len(pts))
    prox_pt = pts[prox_idx]
    cuff_prox = np.zeros((H, W), np.uint8)
    cv2.ellipse(cuff_prox, tuple(int(v) for v in prox_pt),
                (w_t + 4, w_t + 2), 0, 0, 360, 255, -1)
    parts.append(('dl_et_proximal_cuff', cuff_prox, rng.randint(180, 230),
                  {'center': (int(prox_pt[0]), int(prox_pt[1]), 2)}))
    cuff_dist = np.zeros((H, W), np.uint8)
    cv2.ellipse(cuff_dist, (int(tip[0]), int(tip[1])),
                (w_t + 3, w_t + 2), 0, 0, 360, 255, -1)
    parts.append(('dl_et_distal_cuff', cuff_dist, rng.randint(180, 230),
                  {'center': (int(tip[0]), int(tip[1]), 2)}))
    return parts


SHAPE_FNS.update({
    'iabp':       shape_iabp,
    'mitra_clip': shape_mitra_clip,
})
DEVICE_FNS.update({
    'dialysis_catheter': device_dialysis_catheter,
    'gtube':             device_gtube,
    'double_lumen_et':   device_double_lumen_et,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')

# ===== Sprint 9 SHAPES =====

def shape_pericardial_drain(H, W, anat_masks, rng):
    """Pigtail catheter draining pericardial space — curl at tip."""
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    side = rng.choice([-1, +1])
    entry = (int(np.clip(W // 2 + side * rng.randint(80, 130), 0, W - 1)),
             rng.randint(H // 2, 3 * H // 4))
    tip = sample_point_in_mask(heart, rng) or (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-20, 20), rng.randint(-15, 15)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-15, 15), rng.randint(-15, 15)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    # pigtail curl
    curl_r = rng.randint(5, 9)
    cx_t, cy_t = int(tip[0]), int(tip[1])
    angles = np.linspace(0, 1.5 * np.pi, 32)
    cx_off = cx_t - curl_r
    pts_curl = np.stack([cx_off + curl_r * np.cos(angles),
                         cy_t + curl_r * np.sin(angles)], axis=1).astype(np.int32)
    cv2.polylines(mask, [pts_curl], False, 255, 1, cv2.LINE_AA)
    return mask, rng.randint(220, 250), _bezier_kpts(pts)


def shape_uac_uvc(H, W, anat_masks, rng):
    """Pediatric umbilical catheters: 2 Bezier paths from inferior frame edge → aorta + heart."""
    mask = np.zeros((H, W), np.uint8)
    entry_x = W // 2 + rng.randint(-15, 15)
    entry_y = H - rng.randint(5, 20)
    aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    targets = []
    targets.append(sample_point_in_mask(aorta, rng) if aorta.any() else (W // 2, H // 2))
    targets.append(sample_point_in_mask(heart, rng) if heart.any() else (W // 2, H // 3))
    for tip in targets:
        p0 = np.array([entry_x + rng.randint(-5, 5), entry_y], float)
        p3 = np.array(tip, float)
        p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-10, 10), 0], float)
        p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-15, 15), 0], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        cv2.polylines(mask, [pts], False, 255, 1, cv2.LINE_AA)
    return mask, rng.randint(220, 255)


def shape_brachytherapy_seeds(H, W, anat_masks, rng):
    """Grid-like cluster of tiny radioactive seeds (regular pattern, smaller than pellet)."""
    body = anat_masks.any(axis=0)
    pt = sample_point_in_mask(body, rng) or (W // 2, H // 2)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    n = rng.randint(8, 20)
    spread = rng.randint(15, 30)
    side = int(np.ceil(np.sqrt(n)))
    placed = 0
    for i in range(side):
        for j in range(side):
            if placed >= n:
                break
            dx = int((i / max(1, side - 1) - 0.5) * spread + rng.randint(-2, 2))
            dy = int((j / max(1, side - 1) - 0.5) * spread + rng.randint(-2, 2))
            sx, sy = cx + dx, cy + dy
            cv2.line(mask, (sx - 1, sy), (sx + 1, sy), 255, 1)
            placed += 1
    return mask, rng.randint(240, 255)


def shape_bra_underwire(H, W, anat_masks, rng):
    """Curved thin half-ellipse wire at lower chest border (one side)."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        y_low = int(np.percentile(ys, 95))
        side = rng.choice([-1, +1])
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        x_center = int(np.mean(xs[sel])) if sel.any() else W // 2
    else:
        x_center = rng.choice([W // 4, 3 * W // 4])
        y_low = 2 * H // 3
    rx = rng.randint(35, 60); ry = rng.randint(20, 35)
    cy = y_low + rng.randint(-5, 15)
    angle = rng.uniform(-10, 10)
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (x_center, cy), (rx, ry), angle, 180, 360, 255, rng.randint(1, 2), cv2.LINE_AA)
    return mask, rng.randint(200, 240)


# ===== Sprint 9 DEVICES =====

def device_five_lead_monitor(H, W, anat_masks, rng):
    """5 distinct dots (RA, LA, RL, LL, V1) + 5 wires exiting one frame edge."""
    parts = []
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        x_min, x_max = int(xs.min()), int(xs.max())
        y_top = int(ys.min()); y_bot = int(ys.max())
    else:
        x_min, x_max, y_top, y_bot = W // 4, 3 * W // 4, H // 4, 3 * H // 4
    positions = [
        (x_min + rng.randint(-20, 5), y_top + rng.randint(-10, 10)),
        (x_max + rng.randint(-5, 20), y_top + rng.randint(-10, 10)),
        (x_min + rng.randint(-20, 5), y_bot + rng.randint(-10, 20)),
        (x_max + rng.randint(-5, 20), y_bot + rng.randint(-10, 20)),
        ((x_min + x_max) // 2 + rng.randint(-20, 20), (y_top + y_bot) // 2),
    ]
    dots = np.zeros((H, W), np.uint8)
    for px, py in positions:
        px = int(np.clip(px, 0, W - 1)); py = int(np.clip(py, 0, H - 1))
        cv2.circle(dots, (px, py), rng.randint(3, 5), 255, -1)
    dot_centroid = (int(np.mean([p[0] for p in positions])),
                    int(np.mean([p[1] for p in positions])))
    parts.append(('monitor_lead_dots', dots, rng.randint(220, 255),
                  {'center': (dot_centroid[0], dot_centroid[1], 2)}))

    side = rng.choice([-1, +1])
    edge_x = 0 if side < 0 else W - 1
    wires = np.zeros((H, W), np.uint8)
    for px, py in positions:
        px = int(np.clip(px, 0, W - 1)); py = int(np.clip(py, 0, H - 1))
        end = (int(np.clip(edge_x + side * rng.randint(0, 20), 0, W - 1)),
               int(np.clip(py + rng.randint(-30, 30), 0, H - 1)))
        p0 = np.array([px, py], float); p3 = np.array(end, float)
        p1 = p0 + (p3 - p0) * 0.4 + np.array([0, rng.randint(-20, 20)], float)
        p2 = p0 + (p3 - p0) * 0.7 + np.array([0, rng.randint(-20, 20)], float)
        pts = bezier_points(p0, p1, p2, p3, n=48)
        cv2.polylines(wires, [pts], False, 255, 1, cv2.LINE_AA)
    # use centroid of dots as proximal_end; edge as tip
    wires_kpts = {'proximal_end': (dot_centroid[0], dot_centroid[1], 2),
                  'tip': (int(np.clip(edge_x + side * 10, 0, W - 1)), dot_centroid[1], 2)}
    parts.append(('monitor_lead_wires', wires, rng.randint(180, 220), wires_kpts))
    return parts


SHAPE_FNS.update({
    'pericardial_drain':   shape_pericardial_drain,
    'uac_uvc':             shape_uac_uvc,
    'brachytherapy_seeds': shape_brachytherapy_seeds,
    'bra_underwire':       shape_bra_underwire,
})
DEVICE_FNS.update({
    'five_lead_monitor': device_five_lead_monitor,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')

# ===== Sprint 10 SHAPES =====

def shape_cervical_plate(H, W, anat_masks, rng):
    """Anterior cervical plate + perpendicular screws (vertical plate at C-spine)."""
    cspine_ids = group_ids(safe_attr(label_mapper, 'cervical_spine'))
    cspine = union_mask(anat_masks, cspine_ids) if cspine_ids else None
    if cspine is not None and cspine.any():
        ys, xs = np.where(cspine)
        cx = int(np.median(xs))
        cy = int(np.percentile(ys, 50))
    else:
        cx, cy = W // 2, rng.randint(20, H // 6)
    mask = np.zeros((H, W), np.uint8)
    pw = rng.randint(3, 6); ph = rng.randint(25, 45)
    cv2.rectangle(mask, (cx - pw, cy - ph // 2), (cx + pw, cy + ph // 2), 255, -1)
    n_screws = rng.randint(2, 4)
    for i in range(n_screws):
        y = cy - ph // 2 + (i + 1) * ph // (n_screws + 1)
        cv2.line(mask, (cx - pw - 4, y), (cx + pw + 4, y), 255, 1, cv2.LINE_AA)
    return mask, rng.randint(230, 255)


def shape_endobronchial_valve(H, W, anat_masks, rng):
    """Small duckbill valve in peripheral lung bronchus."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    pt = sample_point_in_mask(lungs, rng) if lungs.any() else (W // 2, H // 2)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    rx = rng.randint(4, 7); ry = rng.randint(3, 5)
    cv2.ellipse(mask, (cx, cy), (rx, ry), 0, 0, 360, 255, 1)
    cv2.line(mask, (cx - rx, cy), (cx - rx - 3, cy + rng.randint(3, 6)), 255, 1, cv2.LINE_AA)
    cv2.line(mask, (cx + rx, cy), (cx + rx + 3, cy + rng.randint(3, 6)), 255, 1, cv2.LINE_AA)
    return mask, rng.randint(220, 250)


def shape_safety_pin(H, W, anat_masks, rng):
    """Bent safety pin (ingested) — bar + spring coil + clasp loop."""
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    pt = sample_point_in_mask(trachea, rng) or (W // 2, H // 3)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    L = rng.randint(20, 35)
    angle = rng.uniform(-math.pi / 6, math.pi / 6) + math.pi / 2
    dx, dy = int(L * math.cos(angle)), int(L * math.sin(angle))
    p1 = (cx - dx, cy - dy); p2 = (cx + dx, cy + dy)
    cv2.line(mask, p1, p2, 255, 1, cv2.LINE_AA)
    cv2.circle(mask, p1, rng.randint(3, 5), 255, 1)
    cv2.ellipse(mask, p2, (4, 3), math.degrees(angle), 90, 270, 255, 1)
    return mask, rng.randint(230, 255)


# ===== Sprint 10 DEVICES =====

def device_evd(H, W, anat_masks, rng):
    """External ventricular drain: tube from skull (top frame) → lateral exit + collection bag rect."""
    parts = []
    entry = (W // 2 + rng.randint(-15, 15), rng.randint(2, 20))
    side = rng.choice([-1, +1])
    bag_x = 0 if side < 0 else W - 1
    bag_y = rng.randint(H // 3, 2 * H // 3)
    p0 = np.array(entry, float)
    p3 = np.array([int(np.clip(bag_x + side * rng.randint(5, 20), 0, W - 1)), bag_y], float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([0, rng.randint(20, 60)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([0, rng.randint(0, 40)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    tube = np.zeros((H, W), np.uint8)
    cv2.polylines(tube, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    parts.append(('evd_tube', tube, rng.randint(200, 240),
                  _bezier_kpts(pts, prox_name='skin_entry')))
    bag = np.zeros((H, W), np.uint8)
    bw, bh = rng.randint(15, 25), rng.randint(20, 35)
    bx = int(np.clip(p3[0] - bw // 2, 0, W - bw))
    by = int(np.clip(p3[1] - bh // 2, 0, H - bh))
    cv2.rectangle(bag, (bx, by), (bx + bw, by + bh), 255, 1)
    parts.append(('evd_bag', bag, rng.randint(180, 220),
                  {'center': (bx + bw // 2, by + bh // 2, 2)}))
    return parts


def device_spinal_cord_stim(H, W, anat_masks, rng):
    """Pulse generator (soft tissue, lateral to spine) + lead along thoracic spine."""
    spine = union_mask(anat_masks, ANAT_GROUPS['spine'])
    if spine.any():
        ys, xs = np.where(spine)
        cx_spine = int(np.median(xs))
        ymin = int(ys.min()); ymax = int(ys.max())
    else:
        cx_spine = W // 2; ymin, ymax = H // 4, 3 * H // 4
    side = rng.choice([-1, +1])
    gen_cx = int(np.clip(cx_spine + side * rng.randint(60, 110), 30, W - 30))
    gen_cy = int(np.clip(ymax + rng.randint(-30, 30), 30, H - 30))
    parts = []
    bw, bh = rng.randint(25, 35), rng.randint(20, 30)
    body, body_color = _textured_body(H, W, gen_cx, gen_cy, bw, bh, rng)
    parts.append(('scs_generator', body, body_color, {'center': (gen_cx, gen_cy, 2)}))
    lead_end_y = ymin + rng.randint(20, 80)
    p0 = np.array([gen_cx, gen_cy - bh // 2], float)
    p3 = np.array([cx_spine + rng.randint(-3, 3), lead_end_y], float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-15, 15), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-10, 10), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    lead = np.zeros((H, W), np.uint8)
    cv2.polylines(lead, [pts], False, 255, rng.randint(1, 2), cv2.LINE_AA)
    parts.append(('scs_lead', lead, rng.randint(210, 250), _bezier_kpts(pts)))
    return parts


SHAPE_FNS.update({
    'cervical_plate':      shape_cervical_plate,
    'endobronchial_valve': shape_endobronchial_valve,
    'safety_pin':          shape_safety_pin,
})
DEVICE_FNS.update({
    'evd':              device_evd,
    'spinal_cord_stim': device_spinal_cord_stim,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')

# ===== Sprint 11 SHAPES =====

def shape_pleurx(H, W, anat_masks, rng):
    """Tunneled pleural catheter — subQ Bezier from chest wall + tube to pleural space w/ marker stripe."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        side = rng.choice([-1, +1])
        ymid = int(np.median(ys))
        xedge = int(xs.max()) if side > 0 else int(xs.min())
        entry = (xedge + side * rng.randint(40, 70), ymid + rng.randint(-20, 20))
        end = sample_point_in_mask(lungs, rng) or (W // 2, H // 2)
    else:
        side = rng.choice([-1, +1])
        entry = (rng.randint(0, 30) if side < 0 else rng.randint(W - 30, W), H // 2)
        end = (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-25, 25), rng.randint(-20, 20)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-20, 20), rng.randint(-15, 15)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, rng.randint(2, 3), cv2.LINE_AA)
    # subQ portion = bright marker stripe on outer 30%
    for i in range(0, int(0.3 * len(pts)), 3):
        cv2.circle(mask, tuple(pts[i].tolist()), 1, 255, -1)
    return mask, rng.randint(200, 240), _bezier_kpts(pts)


def shape_impella(H, W, anat_masks, rng):
    """Impella micro-pump: thin catheter across aortic valve, body in LV w/ inlet/outlet markers."""
    lv = union_mask(anat_masks, group_ids(['heart ventricle left']))
    aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    if lv.any():
        lv_pt = sample_point_in_mask(lv, rng)
    else:
        lv_pt = (W // 2, H // 2)
    if aorta.any():
        ao_pt = sample_point_in_mask(aorta, rng)
    else:
        ao_pt = (lv_pt[0], max(0, lv_pt[1] - 80))
    # catheter from descending aorta upward through arch into LV (one Bezier)
    side = rng.choice([-1, +1])
    entry = (int(np.clip(W // 2 + side * rng.randint(20, 50), 0, W - 1)), H - rng.randint(10, 40))
    p0 = np.array(entry, float); p3 = np.array(lv_pt, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-15, 15), 0], float)
    p2 = p0 + (p3 - p0) * 0.75 + np.array([rng.randint(-10, 10), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    mask = np.zeros((H, W), np.uint8)
    cv2.polylines(mask, [pts], False, 255, 1, cv2.LINE_AA)
    # pump body marker — small bright capsule near LV tip
    pump_pt = pts[int(0.92 * len(pts))]
    cv2.circle(mask, tuple(pump_pt.tolist()), rng.randint(3, 5), 255, -1)
    # inlet marker just before pump (small ring)
    inlet_pt = pts[int(0.85 * len(pts))]
    cv2.circle(mask, tuple(inlet_pt.tolist()), 2, 255, 1)
    return mask, rng.randint(220, 250), _bezier_kpts(pts)


def shape_ablation_antenna(H, W, anat_masks, rng):
    """Thick straight needle into peripheral lung lesion + bright tip enhancement."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        tip = sample_point_in_mask(lungs, rng) or (W // 2, H // 2)
    else:
        tip = (W // 2, H // 2)
    side = rng.choice([-1, +1])
    entry = (int(np.clip(tip[0] + side * rng.randint(60, 120), 0, W - 1)),
             int(np.clip(tip[1] + rng.randint(-40, 40), 0, H - 1)))
    mask = np.zeros((H, W), np.uint8)
    cv2.line(mask, entry, tip, 255, rng.randint(2, 3), cv2.LINE_AA)
    # bright enhanced tip (ablation zone marker)
    cv2.circle(mask, tip, rng.randint(3, 6), 255, -1)
    kpts = {'proximal_end': (int(entry[0]), int(entry[1]), 2),
            'tip': (int(tip[0]), int(tip[1]), 2)}
    return mask, rng.randint(230, 255), kpts


def shape_hairpin(H, W, anat_masks, rng):
    """Bent thin U-shape wire — ingested hairpin / bobby pin."""
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    pt = sample_point_in_mask(trachea, rng) or (W // 2, H // 3)
    cx, cy = pt
    mask = np.zeros((H, W), np.uint8)
    L = rng.randint(15, 28)
    angle = rng.uniform(0, math.pi)
    # 2 parallel legs
    perp_x = -math.sin(angle); perp_y = math.cos(angle)
    half_gap = rng.randint(2, 4)
    leg_dx = int(L * math.cos(angle)); leg_dy = int(L * math.sin(angle))
    for sign in (-1, +1):
        p_start = (cx + int(sign * half_gap * perp_x), cy + int(sign * half_gap * perp_y))
        p_end = (p_start[0] + leg_dx, p_start[1] + leg_dy)
        cv2.line(mask, p_start, p_end, 255, 1, cv2.LINE_AA)
    # U bend connecting tops
    bend_center = (cx, cy)
    cv2.ellipse(mask, bend_center, (half_gap + 1, half_gap + 1),
                math.degrees(angle), 180, 360, 255, 1, cv2.LINE_AA)
    return mask, rng.randint(230, 255)


def shape_cochlear(H, W, anat_masks, rng):
    """Cochlear implant: magnet disc + spiral electrode array at upper skull base."""
    # position: top of frame, slightly lateral
    side = rng.choice([-1, +1])
    cx = int(np.clip(W // 2 + side * rng.randint(60, 110), 30, W - 30))
    cy = rng.randint(5, 30)
    mask = np.zeros((H, W), np.uint8)
    # magnet disc
    cv2.circle(mask, (cx, cy + 5), rng.randint(6, 10), 255, -1)
    # spiral electrode array (cochlear cochlea-shape) below disc
    spiral_cx = cx + side * rng.randint(5, 12)
    spiral_cy = cy + rng.randint(15, 25)
    angles = np.linspace(0, 3.5 * np.pi, 40)
    rs = 1 + 0.6 * angles
    pts = np.stack([spiral_cx + rs * np.cos(angles + rng.uniform(0, 2 * np.pi)),
                    spiral_cy + rs * np.sin(angles + rng.uniform(0, 2 * np.pi))], axis=1).astype(np.int32)
    cv2.polylines(mask, [pts], False, 255, 1, cv2.LINE_AA)
    return mask, rng.randint(230, 255)


SHAPE_FNS.update({
    'pleurx':           shape_pleurx,
    'impella':          shape_impella,
    'ablation_antenna': shape_ablation_antenna,
    'hairpin':          shape_hairpin,
    'cochlear':         shape_cochlear,
})
print(f'shapes: {len(SHAPE_FNS)} | devices: {len(DEVICE_FNS)}')

def warp_rgba(rgba, rng):
    """Elastic deformation + optional morphological dilate/erode on alpha."""
    h, w = rgba.shape[:2]
    sigma = rng.uniform(6, 14); alpha_d = rng.uniform(3, 8)
    rs = np.random.RandomState(rng.randint(0, 2**31 - 1))
    dx = cv2.GaussianBlur((rs.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha_d
    dy = cv2.GaussianBlur((rs.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha_d
    map_x, map_y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = map_x + dx; map_y = map_y + dy
    warped = cv2.remap(rgba, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    if rng.random() < 0.5:
        kr = rng.randint(2, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kr, kr))
        op = rng.choice([cv2.MORPH_DILATE, cv2.MORPH_ERODE])
        warped[..., 3] = cv2.morphologyEx(warped[..., 3], op, kernel)
    return warped

# Anatomic anchor overrides for cutpaste — for classes where random body-mask
# placement would be obviously wrong (mitraclip in the elbow, breast implant
# in the neck, sternotomy wire on the lung). Each entry is a list of CXAS
# anatomy group names from ANAT_GROUPS; the union is the anchor region.
# Looked up by synthfb_category at paste time.
CUTPASTE_ANATOMIC_ANCHORS: dict[str, list[str]] = {
    'sternotomy_wire': ['sternum'],
    'mitraclip':       ['heart'],
    'breast_implant':  ['lungs'],   # breast tissue overlay — lower lung region
    'trach_shaft':     ['trachea'],
    'vent_hose':       ['trachea'],
    'event_recorder':  ['clavicle'],  # pectoral region, near clavicle
}


def cut_paste(img, rgba, anat_masks, rng, blend='poisson_normal', use_warp=True,
              anchor_group: list[str] | None = None):
    H, W = img.shape[:2]
    if use_warp and rng.random() < 0.5:
        rgba = warp_rgba(rgba, rng)
    rgba = weak_augment_rgba(rgba, rng)
    ch, cw = rgba.shape[:2]
    if ch >= H or cw >= W:
        s = min(H, W) / max(ch, cw) * 0.5
        rgba = cv2.resize(rgba, (max(8, int(cw * s)), max(8, int(ch * s))))
        ch, cw = rgba.shape[:2]
    # Prefer anatomic anchor when caller provides one (caters to classes with
    # strong position priors). Falls back to whole-body sampling when no
    # anchor is given or the requested anatomy is missing in this CXR.
    anchor_mask = None
    if anchor_group:
        try:
            anchor_mask = union_mask(anat_masks, group_ids(anchor_group))
        except Exception:
            anchor_mask = None
    if anchor_mask is None or not anchor_mask.any():
        anchor_mask = anat_masks.any(axis=0)
    pt = sample_point_in_mask(anchor_mask, rng) or (W // 2, H // 2)
    cx, cy = pt
    x0 = int(np.clip(cx - cw // 2, 0, W - cw))
    y0 = int(np.clip(cy - ch // 2, 0, H - ch))
    rgb = rgba[..., :3]; alpha = rgba[..., 3]
    insert_mask = (alpha > 10).astype(np.uint8) * 255
    full_mask = np.zeros((H, W), np.uint8)
    full_mask[y0:y0 + ch, x0:x0 + cw] = insert_mask
    if blend in ('poisson_normal', 'poisson_mixed') and insert_mask.any():
        try:
            flag = cv2.NORMAL_CLONE if blend == 'poisson_normal' else cv2.MIXED_CLONE
            center = (x0 + cw // 2, y0 + ch // 2)
            out = cv2.seamlessClone(rgb, img, insert_mask, center, flag)
            return out, full_mask
        except cv2.error:
            pass
    out = img.copy()
    af = (alpha[..., None] / 255.0)
    crop = out[y0:y0 + ch, x0:x0 + cw].astype(np.float32)
    blended = af * rgb.astype(np.float32) + (1 - af) * crop
    out[y0:y0 + ch, x0:x0 + cw] = np.clip(blended, 0, 255).astype(np.uint8)
    return out, full_mask

# Same circular shape, 4 physics modes side-by-side
demo_rng3 = random.Random(11)
mask_demo, color_demo = shape_circular(*demo_img.shape[:2], demo_anat, demo_rng3)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, mode in zip(axes, PHYSICS_MODES):
    out, _ = render_with_physics(demo_img.copy(), mask_demo, color_demo, demo_anat, demo_rng3, mode=mode)
    ax.imshow(out); ax.imshow(mask_demo, alpha=0.25, cmap='Reds')
    ax.set_title(f'circular | {mode}'); ax.axis('off')
plt.suptitle('Physics modes: same shape, 4 rendering styles'); plt.tight_layout(); plt.show()

# Composite devices
demo_rng4 = random.Random(23)
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, (name, fn) in zip(axes, DEVICE_FNS.items()):
    parts = fn(demo_img.shape[0], demo_img.shape[1], demo_anat, demo_rng4)
    out = demo_img.copy()
    mode = demo_rng4.choice(PHYSICS_MODES)
    overlay = np.zeros(demo_img.shape[:2], np.uint8)
    for _part in parts:
        m, c = _part[1], _part[2]
        out, _ = render_with_physics(out, m, c, demo_anat, demo_rng4, mode=mode)
        overlay = np.maximum(overlay, m)
    ax.imshow(out); ax.imshow(overlay, alpha=0.3, cmap='Reds')
    ax.set_title(f'{name} | {mode}'); ax.axis('off')
plt.suptitle('Composite devices (each with one random physics mode)'); plt.tight_layout(); plt.show()

# New shapes (blob, sdf)
demo_rng5 = random.Random(31)
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
for j, name in enumerate(['blob', 'sdf']):
    for k, mode in enumerate(PHYSICS_MODES):
        _ret = SHAPE_FNS[name](demo_img.shape[0], demo_img.shape[1], demo_anat, demo_rng5)
        m, c = _ret[0], _ret[1]
        out, _ = render_with_physics(demo_img.copy(), m, c, demo_anat, demo_rng5, mode=mode)
        ax = axes[j, k]
        ax.imshow(out); ax.imshow(m, alpha=0.3, cmap='Reds')
        ax.set_title(f'{name} | {mode}'); ax.axis('off')
plt.suptitle('New shapes × physics modes'); plt.tight_layout(); plt.show()

# Cutout warp demo
if CUTOUTS:
    cat = next(iter(CUTOUTS)); rgba_demo = load_rgba(CUTOUTS[cat][0])
    demo_rng6 = random.Random(53)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(rgba_demo); axes[0].set_title('original cutout'); axes[0].axis('off')
    for ax in axes[1:]:
        w = warp_rgba(rgba_demo, demo_rng6)
        ax.imshow(w); ax.set_title('elastic-warped'); ax.axis('off')
    plt.suptitle('Cutout warping — same source, 3 random warps'); plt.tight_layout(); plt.show()

DEVICE_PART_CATEGORIES = sorted({
    # paper baseline composites
    'pacemaker_body', 'pacemaker_lead',
    'port_reservoir', 'port_septum', 'catheter_tube',
    'sternotomy_wire',
    'et_tube_shaft', 'et_tube_cuff',
    # Sprint 1
    'icd_body', 'icd_lead',
    'trach_shaft', 'trach_cuff',
    'valve_ring', 'valve_struts',
    'spinal_rod', 'pedicle_screw',
    # Sprint 2
    'lvad_body', 'lvad_outflow', 'lvad_driveline',
    'tunneled_exit', 'tunneled_tube',
    # Sprint 4
    'vp_reservoir', 'vp_tube',
    'shoulder_head', 'shoulder_stem',
    'expander_body', 'expander_port_disc',
    # Sprint 5
    'ecmo_return', 'ecmo_drain',
    'bronchial_y_trunk', 'bronchial_y_branch',
    'crt_body', 'crt_lead_ra', 'crt_lead_rv', 'crt_lead_lv',
    # Sprint 7
    'halo_ring', 'halo_post',
    'sb_tube', 'sb_esophageal_balloon', 'sb_gastric_balloon',
    # Sprint 8
    'dialysis_exit', 'dialysis_tube', 'dialysis_y_split',
    'gtube_disc', 'gtube_balloon',
    'dl_et_shaft', 'dl_et_proximal_cuff', 'dl_et_distal_cuff',
    # Sprint 9
    'monitor_lead_dots', 'monitor_lead_wires',
    # Sprint 10
    'evd_tube', 'evd_bag',
    'scs_generator', 'scs_lead',
})
SHAPE_CATEGORIES = list(SHAPE_FNS.keys())
SPLIT_INSTANCE_CATS = [
    'cabg_clip', 'dotted_screw', 'surgical_staple',
    'monitor_lead_dot', 'monitor_lead_wire',
]
CATEGORIES = SHAPE_CATEGORIES + DEVICE_PART_CATEGORIES + SPLIT_INSTANCE_CATS + ['cutpaste']
CAT2ID = {c: i + 1 for i, c in enumerate(CATEGORIES)}
print(f'{len(CATEGORIES)} categories')

# ===== Keypoint schema (COCO format) =====
# Per-category list of keypoint names. Empty/missing list -> no keypoints.
# Multi-kpt categories must emit explicit kpts in their device/shape function;
# single-kpt {'center'} categories fall back to mask centroid automatically.

KEYPOINT_SCHEMA = {
    # cutpaste + simple symmetric -> centroid fallback
    'cutpaste':           ['center'],
    'coin_ingested':      ['center'],
    'button_battery':     ['center'],
    'pellet_cluster':     ['center'],
    'magnet_pair':        ['center'],
    'safety_pin':         ['center'],
    'hairpin':            ['center'],
    'aspirated_fb':       ['center'],
    'embolization_coils': ['center'],
    'brachy_seeds':       ['center'],
    'lumpectomy_clips':   ['center'],
    'breast_implant':     ['center'],
    'cardiomems':         ['center'],
    'watchman':           ['center'],
    'mitraclip':          ['center'],
    'tavr':               ['center'],
    'gossypiboma':        ['start', 'end'],
    'cabg_clips':         ['center'],
    'mv_struts':          ['center'],
    'cochlear_implant':   ['center'],
    'vertebroplasty':     ['center'],
    'event_recorder':     ['center'],
    'defib_pads':         ['center'],
    'bra_underwire':      ['center'],
    'jewelry':            ['center'],
    'piercing':           ['center'],
    'text_label':         ['center'],
    'ring_chain':         ['center'],
    'textual':            ['center'],
    'blob':               ['center'],
    'sdf':                ['center'],
    'circular':           ['center'],
    'clip':               ['center'],
    'endobronchial_valve':['center'],
    'ring':               ['center'],
    'rect':               ['center'],
    'grid':               ['center'],
    'parallel':           ['center'],
    'branching_tree':     ['center'],
    'dotted_screws':      ['center'],
    'staple_line':        ['center'],
    'ekg_bundle':         ['center'],
    'offframe_cables':    ['center'],
    'vent_hose':          ['center'],
    'acquisition_burn_in':['center'],
    'mesh_stent':         ['center'],
    'annuloplasty_ring':  ['center'],
    'tevar':              ['center'],
    'pacing_wire':        ['center'],
    'mitra_clip':         ['center'],
    'uac_uvc':            ['center'],
    'cochlear':           ['center'],
    'brachytherapy_seeds':['center'],

    # device parts: bodies/discs/cuffs/balloons -> centroid
    'pacemaker_body':     ['center'],
    'icd_body':           ['center'],
    'crt_body':           ['center'],
    'lvad_body':          ['center'],
    'port_reservoir':     ['center'],
    'tunneled_exit':      ['center'],
    'dialysis_exit':      ['center'],
    'et_tube_cuff':       ['center'],
    'trach_cuff':         ['center'],
    'dl_et_proximal_cuff':['center'],
    'dl_et_distal_cuff':  ['center'],
    'sb_esophageal_balloon':['center'],
    'sb_gastric_balloon': ['center'],
    'gtube_disc':         ['center'],
    'gtube_balloon':      ['center'],
    'vp_reservoir':       ['center'],
    'expander_body':      ['center'],
    'expander_port_disc': ['center'],
    'halo_ring':          ['center'],
    'evd_bag':            ['center'],
    'scs_generator':      ['center'],
    'shoulder_head':      ['center'],
    'pedicle_screw':      ['center'],
    'monitor_lead_dots':  ['center'],
    'valve_ring':         ['center'],
    'valve_struts':       ['center'],

    # tubes/leads with explicit endpoints (require kpt emission in device fn)
    'pacemaker_lead':    ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'icd_lead':          ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'crt_lead_ra':       ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'crt_lead_rv':       ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'crt_lead_lv':       ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'lvad_outflow':      ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'lvad_driveline':    ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'catheter_tube':     ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'tunneled_tube':     ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'dialysis_tube':     ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'dialysis_y_split':   ['junction', 'arm_a', 'arm_b'],
    'et_tube_shaft':     ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'trach_shaft':       ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'dl_et_shaft':       ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'picc_line':         ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'ng_tube':           ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'swan_ganz':         ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'chest_tube':        ['skin_entry', 'mid_a', 'mid_b', 'tip'],
    'mediastinal_drain': ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'pleurx':            ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'pericardial_drain': ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'iabp':              ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'impella':           ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'ecmo_return':        ['proximal_end', 'tip'],
    'ecmo_drain':         ['proximal_end', 'tip'],
    'bronchial_y_trunk':  ['proximal_end', 'tip'],
    'bronchial_y_branch': ['proximal_end', 'tip'],
    'tracheal_stent':     ['proximal_end', 'tip'],
    'esophageal_stent':   ['proximal_end', 'tip'],
    'sb_tube':           ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'monitor_lead_wires': ['proximal_end', 'tip'],
    'evd_tube':          ['skin_entry', 'mid_a', 'mid_b', 'tip'],
    'uac':                ['umbilicus', 'tip'],
    'uvc':                ['umbilicus', 'tip'],
    'ablation_antenna':   ['proximal_end', 'tip'],
    'sternotomy_wire':    ['left_end', 'right_end'],
    'surgical_staples':   ['proximal_end', 'distal_end'],
    'spinal_rod':         ['top', 'bottom'],
    'shoulder_stem':      ['top', 'bottom'],
    'cervical_plate':     ['top', 'bottom'],
    'scs_lead':          ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'vp_tube':           ['proximal_end', 'mid_a', 'mid_b', 'tip'],
    'halo_post':          ['top', 'bottom'],
    'film_scratch':       ['start', 'end'],
    'line':               ['start', 'end'],
    'scratch':            ['start', 'end'],
    'ivc_filter':         ['apex', 'base'],
}


def _bezier_kpts(pts, prox_name='proximal_end'):
    """Sample 4 kpts along Bezier path: prox, mid_a (~33%), mid_b (~66%), tip."""
    n = len(pts)
    return {
        prox_name: (int(pts[0][0]), int(pts[0][1]), 2),
        'mid_a':   (int(pts[n // 3][0]),     int(pts[n // 3][1]),     2),
        'mid_b':   (int(pts[2 * n // 3][0]), int(pts[2 * n // 3][1]), 2),
        'tip':     (int(pts[-1][0]), int(pts[-1][1]), 2),
    }


def _default_skeleton(n_kpts):
    """Connect consecutive keypoints (1-indexed COCO)."""
    if n_kpts < 2:
        return []
    return [[i + 1, i + 2] for i in range(n_kpts - 1)]


def _centroid_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return (int(np.mean(xs)), int(np.mean(ys)), 2)


def _inside_centroid(mask):
    """Return (x, y, 2) for a point GUARANTEED to lie inside the mask — the
    distance-transform peak (farthest point from any background pixel).
    Geometric centroid can fall outside for curved/concave shapes (e.g. a
    U-shaped chest tube), which corrupts COCO keypoint training. Falls back
    to centroid for degenerate masks where the DT is uniform."""
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return None
    dt = cv2.distanceTransform(bin_mask, cv2.DIST_L2, 3)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return (int(x), int(y), 2)


def _mask_axis_landmarks(mask, names):
    """Derive landmarks for an N-keypoint schema by projecting the mask onto
    its principal axis (PCA). Returns {name: (x, y, 2)} where names map to
    fractional positions along the axis:
        'proximal_end' / 'start' / 'p1'        → 0%
        'mid_a' / 'mid' (with mid_b absent)    → 50%
        'mid_a'  (when both)                   → 33%
        'mid_b'                                → 66%
        'tip' / 'end' / 'pN'                   → 100%
    For unrecognized names, distributes them evenly between 0 and 100%.
    All returned points are snapped to the nearest in-mask pixel so they
    visibly overlap the mask in viz overlays.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 5:
        c = _inside_centroid(mask)
        return {n: c for n in names} if c else {}
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mean = pts.mean(axis=0)
    centered = pts - mean
    # Principal axis: eigenvector of covariance with largest eigenvalue
    cov = np.cov(centered, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, -1]                # x,y unit vector
    projections = centered @ axis
    # Map name → fractional position along the principal axis.
    # left_end/right_end is the same idea as proximal/tip for transverse bars
    # (sternotomy wires, cervical plates) — picks whichever PCA extreme is
    # spatially more left/right after axis projection.
    pos_by_role = {
        'proximal_end': 0.0, 'start': 0.0, 'top': 0.0, 'p1': 0.0,
        'skin_entry':   0.0, 'left_end': 0.0,
        'mid_a': 1/3, 'mid': 0.5, 'mid_b': 2/3,
        'tip': 1.0, 'end': 1.0, 'bottom': 1.0, 'right_end': 1.0,
    }
    # If only 'mid_a' present (no mid_b), shift it to 50%
    has_b = 'mid_b' in names
    fracs = []
    unrecognized = []
    for i, name in enumerate(names):
        if name == 'mid_a' and not has_b:
            fracs.append(0.5)
        elif name in pos_by_role:
            fracs.append(pos_by_role[name])
        elif name.startswith('p') and name[1:].isdigit():
            # p1..pN evenly distributed across [0,1]
            idx = int(name[1:]) - 1
            fracs.append(idx / max(1, len(names) - 1))
        else:
            fracs.append(None)
            unrecognized.append(i)
    for idx in unrecognized:
        fracs[idx] = idx / max(1, len(names) - 1)
    # Sample positions on the axis
    p_min, p_max = projections.min(), projections.max()
    # For transverse bars (left_end / right_end pair), order PCA extremes by
    # screen-x so 'left_end' is always the visually leftmost point.
    if {'left_end', 'right_end'} <= set(names):
        # If the projection direction has negative x component, swap fractions:
        # the principal axis can point either way, so 'fracs' as-is may map
        # right_end to the visually-leftmost extreme.
        if axis[0] < 0:
            new_fracs = []
            for n, f in zip(names, fracs):
                if n == 'left_end':  new_fracs.append(1.0)
                elif n == 'right_end': new_fracs.append(0.0)
                else: new_fracs.append(f)
            fracs = new_fracs
    out = {}
    for name, frac in zip(names, fracs):
        target_proj = p_min + (p_max - p_min) * frac
        # Snap to nearest in-mask pixel along the axis
        idx = int(np.argmin(np.abs(projections - target_proj)))
        out[name] = (int(xs[idx]), int(ys[idx]), 2)
    return out


# Cutpaste-derived schemas where the PCA axis order needs to match anatomic
# convention. The heuristic: on a frontal CXR, "tip" is the more inferior end
# (larger image y) for catheter/tube classes; "skin_entry" is the more lateral
# end for chest_tube; "left_end"/"right_end" follow image-x.
_KPT_VERTICAL_TIP = {
    'pacemaker_lead', 'icd_lead', 'picc_line', 'ng_tube', 'swan_ganz',
    'mediastinal_drain', 'pleurx', 'tunneled_tube', 'dialysis_tube',
    'pericardial_drain',
}
_KPT_HORIZONTAL_LR = {'sternotomy_wire'}
_KPT_VERTICAL_TB = {'cervical_plate'}


def orient_kpts_by_axis(kpts_dict, cat_name):
    """For cutpaste-derived multi-kpt schemas, ensure tip/proximal (or
    left/right, top/bottom) follow anatomic convention. PCA gives an axis but
    not a direction so the raw landmark assignment is 50/50; this fn swaps
    paired slots when the convention is reversed.
    """
    if not kpts_dict:
        return kpts_dict
    def _swap(a_name, b_name):
        a, b = kpts_dict.get(a_name), kpts_dict.get(b_name)
        if a is None or b is None:
            return
        kpts_dict[a_name], kpts_dict[b_name] = b, a

    if cat_name in _KPT_VERTICAL_TIP:
        prox = kpts_dict.get('proximal_end')
        tip = kpts_dict.get('tip')
        if prox is not None and tip is not None and tip[1] < prox[1]:
            _swap('proximal_end', 'tip')
            _swap('mid_a', 'mid_b')   # preserve along-axis ordering
    elif cat_name == 'chest_tube':
        skin = kpts_dict.get('skin_entry')
        tip = kpts_dict.get('tip')
        # chest tube skin entry is the more lateral end (further from image center x)
        if skin is not None and tip is not None:
            cx = kpts_dict.get('mid_a', tip)[0]  # rough image center proxy: assume mid is more central
            # If skin is closer to center than tip, swap
            if abs(skin[0] - cx) < abs(tip[0] - cx):
                _swap('skin_entry', 'tip')
                _swap('mid_a', 'mid_b')
    elif cat_name in _KPT_HORIZONTAL_LR:
        left = kpts_dict.get('left_end')
        right = kpts_dict.get('right_end')
        if left is not None and right is not None and left[0] > right[0]:
            _swap('left_end', 'right_end')
    elif cat_name in _KPT_VERTICAL_TB:
        top = kpts_dict.get('top')
        bot = kpts_dict.get('bottom')
        if top is not None and bot is not None and top[1] > bot[1]:
            _swap('top', 'bottom')
    return kpts_dict


def _fallback_kpts(cat_name, mask):
    """Derive keypoints from a mask when the shape/device fn didn't emit any.
    - Single-'center' schemas → distance-transform peak (always inside mask).
    - Multi-kpt schemas → PCA-axis landmarks (also snapped inside).
    This is what cutpaste annotations use to populate their schema, and what
    any shape fn that returns kpts=None falls back to.

    For known directional schemas (leads, tubes, transverse bars), the raw
    PCA-axis assignment is post-processed by orient_kpts_by_axis so that
    'tip', 'left_end', etc. follow anatomic convention.
    """
    schema = KEYPOINT_SCHEMA.get(cat_name, [])
    if not schema:
        return {}
    if len(schema) == 1 and schema[0] == 'center':
        c = _inside_centroid(mask)
        return {'center': c} if c else {}
    landmarks = _mask_axis_landmarks(mask, schema)
    return orient_kpts_by_axis(landmarks, cat_name)


def snap_kpts_to_mask(kpts_dict, mask):
    """Force every visible (v>0) kpt to land inside the mask. If a kpt is
    already in-mask, leave it. If outside (e.g. anatomic-landmark kpts placed
    by a device fn at a point the rendered mask didn't reach), snap to the
    nearest in-mask pixel. Explicitly out-of-image kpts (negative or beyond
    bounds) keep their (x, y) so _flatten_keypoints can set visibility=0.
    """
    if not kpts_dict:
        return kpts_dict
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return kpts_dict
    H, W = bin_mask.shape
    ys, xs = np.where(bin_mask)
    out = {}
    for name, kp in kpts_dict.items():
        if kp is None:
            out[name] = None
            continue
        x, y, v = kp
        if v <= 0:
            out[name] = kp
            continue
        # Out-of-image kpts: leave alone — _flatten_keypoints will mark v=0
        if x < 0 or x >= W or y < 0 or y >= H:
            out[name] = kp
            continue
        # In-image but maybe outside mask — snap if needed
        if bin_mask[y, x]:
            out[name] = kp
        else:
            d = (xs - x) ** 2 + (ys - y) ** 2
            i = int(np.argmin(d))
            out[name] = (int(xs[i]), int(ys[i]), v)
    return out


def _flatten_keypoints(cat_name, kpts_dict, img_w, img_h):
    """Return (flat_list, num_visible) in schema order. Out-of-frame -> v=0."""
    schema = KEYPOINT_SCHEMA.get(cat_name, [])
    out = []
    n_vis = 0
    for kname in schema:
        kp = kpts_dict.get(kname) if kpts_dict else None
        if kp is None:
            out.extend([0, 0, 0])
        else:
            x, y, v = int(kp[0]), int(kp[1]), int(kp[2])
            if not (0 <= x < img_w and 0 <= y < img_h):
                v = 0
            out.extend([x, y, v])
            if v > 0:
                n_vis += 1
    return out, n_vis


print(f'keypoint schema: {len(KEYPOINT_SCHEMA)} categories')
print(f'  multi-kpt:  {sum(1 for v in KEYPOINT_SCHEMA.values() if len(v) > 1)}')
print(f'  single-kpt: {sum(1 for v in KEYPOINT_SCHEMA.values() if len(v) == 1)}')

# ===== Multi-instance splitting =====
SPLIT_INTO_INSTANCES = {
    'cabg_clips':         'cabg_clip',
    'dotted_screws':      'dotted_screw',
    'staple_line':        'surgical_staple',
    'monitor_lead_dots':  'monitor_lead_dot',
    'monitor_lead_wires': 'monitor_lead_wire',
}

# Tight-cluster cats: stay 1 instance, but emit per-component kpts (p1..pN_max).
# Visible kpts = number of components present; rest padded v=0.
MULTI_KPT_CLUSTERS = {
    'ring_chain':           16,
    'brachytherapy_seeds':  20,
    'pellet_cluster':       10,
    'embolization_coils':    5,
    'lumpectomy_clips':      6,
    'magnet_pair':           2,
}

for _cat, _n_max in MULTI_KPT_CLUSTERS.items():
    KEYPOINT_SCHEMA[_cat] = [f'p{i + 1}' for i in range(_n_max)]


def _cluster_kpts(mask, n_max):
    """Return {p1..pN_max} from connected-component centroids. Pad missing with v=0."""
    bin_mask = (mask > 0).astype(np.uint8)
    out = {f'p{i + 1}': (0, 0, 0) for i in range(n_max)}
    if bin_mask.sum() == 0:
        return out
    n, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    centers = []
    for i in range(1, n):
        ys, xs = np.where(labels == i)
        if len(xs):
            centers.append((int(np.mean(xs)), int(np.mean(ys))))
    centers.sort(key=lambda c: (c[1], c[0]))  # y then x for stable ordering
    for i, (cx, cy) in enumerate(centers[:n_max]):
        out[f'p{i + 1}'] = (cx, cy, 2)
    return out


# Custom skeleton overrides (clusters have no meaningful inter-point connectivity)
SKELETONS = {cat: [] for cat in MULTI_KPT_CLUSTERS}

for _plural, _singular in SPLIT_INTO_INSTANCES.items():
    if _singular == 'monitor_lead_wire':
        KEYPOINT_SCHEMA[_singular] = ['proximal_end', 'tip']
    else:
        KEYPOINT_SCHEMA[_singular] = ['center']


def _split_mask_instances(mask, cat_singular):
    """Connected-components split. Returns list of (cat_singular, sub_mask, kpts_dict)."""
    bin_mask = (mask > 0).astype(np.uint8)
    if bin_mask.sum() == 0:
        return []
    n, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    out = []
    for i in range(1, n):
        sub = ((labels == i).astype(np.uint8)) * 255
        ys, xs = np.where(sub > 0)
        if len(xs) == 0:
            continue
        cx, cy = int(np.mean(xs)), int(np.mean(ys))
        out.append((cat_singular, sub, {'center': (cx, cy, 2)}))
    return out


print(f'split-into-instances: {len(SPLIT_INTO_INSTANCES)} plural -> singular cats')


# ---- View classifier from subset CSV ----
# ViewDetailed coding (empirically verified from MIMIC samples):
#   0 = PA frontal, 3 = AP frontal
#   1 = lateral,   5 = lateral
#   -2 = unlabeled
FRONTAL_CODES = {0, 3}
LATERAL_CODES = {1, 5}

_view_lookup = {}
try:
    _vdf = pd.read_csv(MIMIC_SUBSET_CSV) if MIMIC_SUBSET_CSV.exists() else None
    if _vdf is not None and 'ViewDetailed' in _vdf.columns:
        if 'full_path' not in _vdf.columns:
            # CSV ships img_path only; resolve against the images root
            # (mirrors load_subset_paths).
            _vdf['full_path'] = _vdf['img_path'].apply(
                lambda p: os.path.join(str(MIMIC_ROOT), p))
        _view_lookup = dict(zip(_vdf['full_path'].astype(str), _vdf['ViewDetailed']))
except Exception as _e:
    print('view lookup unavailable:', _e)

def classify_view(image_path):
    code = _view_lookup.get(str(image_path), None)
    if code is None:
        return 'unknown'
    try:
        c = int(code)
    except (ValueError, TypeError):
        return 'unknown'
    if c in FRONTAL_CODES: return 'frontal'
    if c in LATERAL_CODES: return 'lateral'
    return 'unknown'

# ---- View restrictions ----
FRONTAL_ONLY_SHAPES = {
    'sdf', 'cardiomems',
    # implant / hardware shapes — geometry assumes AP/PA projection
    'tracheal_stent', 'tevar', 'vertebroplasty', 'mesh_stent',
    'breast_implant', 'bra_underwire', 'cabg_clips',
    'mitra_clip', 'embolization_coils', 'annuloplasty_ring',
    'watchman', 'impella',
    # TUBES / LINES — venous, tracheal, and pleural geometries are all built for
    # frontal projection. On lateral CXR these tubes either look anatomically wrong
    # (wrong path) or shouldn't be visible (e.g. NG behind the heart). Real CXRs
    # with tubes are nearly always AP/portable; restrict to frontal so the synth
    # device matches the view's projection.
    'picc_line', 'ng_tube', 'swan_ganz', 'chest_tube',
    'pleurx', 'mediastinal_drain', 'pericardial_drain',
    'pacing_wire', 'uac_uvc', 'iabp',
    'esophageal_stent',
}
FRONTAL_ONLY_DEVICES = {
    'shoulder_prosthesis', 'bronchial_y_stent',
    'mechanical_valve', 'crt_pacemaker',
    'five_lead_monitor', 'tissue_expander',
    'pacemaker', 'icd', 'lvad',
    'sternotomy_wires',
    # TUBES (composite devices) — same rationale as shapes above
    'et_tube', 'double_lumen_et', 'tracheostomy',
    'port_catheter', 'tunneled_cvc', 'dialysis_catheter',
    'ecmo', 'sengstaken', 'gtube', 'bronchial_y_stent',
}

# Large opaque solids: real implants/bodies fully block X-rays at their pixels.
# Must NOT use `occluded` (paints rib darkening INSIDE — produces grey patches inside body)
# or `beer_lambert` (lets underlying anatomy bleed through edges of large solid).
SOLID_OPAQUE = {
    # shape primitives — solid radiopaque objects
    'breast_implant', 'circular', 'rect', 'blob', 'sdf',
    'coin_ingested', 'button_battery', 'magnet_pair',
    'pellet_cluster', 'lumpectomy_clips', 'dotted_screws',
    'aspirated_fb', 'brachytherapy_seeds',
    'mitra_clip', 'watchman', 'cardiomems',
    # device PART names — solid bodies
    'pacemaker_body', 'icd_body', 'crt_body',
    'lvad_body',
    'expander_body', 'expander_port_disc',
    'gtube_disc', 'gtube_balloon',
    'shoulder_head',
    'scs_generator', 'evd_bag',
    'port_reservoir', 'tunneled_exit', 'vp_reservoir', 'dialysis_exit',
    'monitor_lead_dots',
    # small balloons (radiopaque cuff edges) — treat as solid
    'et_tube_cuff', 'trach_cuff',
    'dl_et_proximal_cuff', 'dl_et_distal_cuff',
    'sb_esophageal_balloon', 'sb_gastric_balloon',
    # valve struts + ring — small radiopaque metal
    'valve_ring', 'valve_struts',
    # halo external ring
    'halo_ring',
    # thick opaque cannulas/tubes (move from TUBE_ITEMS to avoid grey-patch artifact)
    'ecmo_return', 'ecmo_drain',
    'lvad_outflow',
    'et_tube_shaft', 'dl_et_shaft', 'trach_shaft',
    'sb_tube', 'iabp',
    'dialysis_tube',
    # radio-opaque stents/grafts (metal mesh appears bright + uniform on X-ray)
    'tracheal_stent', 'esophageal_stent', 'mesh_stent', 'tevar',
    'bronchial_y_trunk', 'bronchial_y_branch',
    'impella',
}

# Thin elongated tubes / wires / leads — can use beer_lambert/occluded (rib bands look correct)
TUBE_ITEMS = {
    # shape primitives that are tubes
    'chest_tube', 'parallel', 'pleurx',
    'mediastinal_drain', 'esophageal_stent',
    'ng_tube', 'picc_line', 'swan_ganz', 'pacing_wire',
    'cabg_clips', 'staple_line',
    # device part names — tubes/wires/leads
    'pacemaker_lead', 'icd_lead',
    'crt_lead_ra', 'crt_lead_rv', 'crt_lead_lv',
    'scs_lead', 'spinal_rod', 'pedicle_screw', 'shoulder_stem',
    'catheter_tube', 'lvad_driveline',
    'tunneled_tube', 'evd_tube', 'vp_tube',
    'monitor_lead_wires', 'sternotomy_wire',
    'dialysis_y_split', 'halo_post',
}

SKIN_SURFACE = {'defib_pads'}

# Burned-in technique labels (PORTABLE/AP/SUPINE etc) — always overlay/soft_edge
# at bright intensity. Never use occluded (would darken behind ribs) or beer_lambert
# (per-pixel attenuation makes the text look broken).
TEXT_BURN_IN = {'acquisition_burn_in'}

# Thin metal/wire leads — beer_lambert produces dashed look (per-pixel attenuation variance).
# Restrict to overlay/soft_edge for uniform brightness.
THIN_LEADS = {
    'pacemaker_lead', 'icd_lead',
    'crt_lead_ra', 'crt_lead_rv', 'crt_lead_lv',
    'scs_lead',
    'pedicle_screw', 'spinal_rod',
    'sternotomy_wire',
    'pacing_wire',
    'halo_post',
    'monitor_lead_wires', 'monitor_lead_wire',
    'uac_uvc',
    # overlying-equipment cables / vent hose: thin radiopaque silhouettes
    'offframe_cables', 'vent_hose',
}

SOLID_TEXTURED = {'pacemaker_body', 'icd_body', 'crt_body', 'lvad_body', 'scs_generator'}

# Real implants are partially radiolucent: anatomy visible through body, bright contour rim.
# Always renders via `translucent_implant` (distance-transform alpha, alpha_core sampled per-instance).
TRANSLUCENT_IMPLANT = {
    'pacemaker_body', 'icd_body', 'crt_body', 'lvad_body',
    'expander_body', 'breast_implant',
    'port_reservoir',
}

# Faint cerclage wires — soft_edge gives the low-contrast smeared look from real CXRs.
FAINT_WIRES = {'sternotomy_wire'}

# Lumen-tube cats built via `lumen_tube_mask` carry per-pixel color maps (walls vs markers
# vs fill at different intensities). Must render via overlay/soft_edge — beer_lambert's
# `if color > 128` branch would crash on the ndarray color.
LUMEN_TUBES = {
    # shapes built via lumen_tube_mask
    'ng_tube', 'picc_line', 'swan_ganz', 'chest_tube',
    'pleurx', 'mediastinal_drain', 'pericardial_drain',
    'pacing_wire', 'uac_uvc', 'iabp', 'impella',
    # device parts built via lumen_tube_mask
    'et_tube_shaft', 'trach_shaft', 'dl_et_shaft',
    'dialysis_tube', 'tunneled_tube', 'catheter_tube',
    'lvad_outflow', 'lvad_driveline',
    'ecmo_return', 'ecmo_drain',
    'sb_tube',
}

def pick_physics_mode(category_name, rng):
    if category_name in TEXT_BURN_IN:
        return 'overlay'                          # bright crisp text, no occluded/beer_lambert
    if category_name in TRANSLUCENT_IMPLANT:
        # Mix translucent_implant (rim/core distance-transform) with screen blend so
        # anatomy bleeds through the body rather than being replaced — looks integrated.
        return rng.choice(['translucent_implant', 'screen'])
    if category_name in LUMEN_TUBES:
        # Add 'screen' so tubes additively brighten over rib shadows instead of replacing.
        return rng.choice(['overlay', 'soft_edge', 'screen'])
    if category_name in FAINT_WIRES:
        # Sternotomy cerclage: additive over sternum + ribs reads as integrated cabling.
        return rng.choice(['soft_edge', 'screen'])
    if category_name in SKIN_SURFACE:
        return 'soft_edge'
    if category_name in SOLID_TEXTURED:
        return 'overlay'  # textured bodies render only via overlay (preserves per-pixel color map)
    if category_name in SOLID_OPAQUE:
        return rng.choice(['overlay', 'soft_edge', 'screen'])
    if category_name in THIN_LEADS:
        # Leads/wires/screws: screen blend so they don't look pasted over ribs.
        return rng.choice(['overlay', 'soft_edge', 'screen'])
    if category_name in TUBE_ITEMS:
        return rng.choice(['beer_lambert', 'occluded'])
    return rng.choice(PHYSICS_MODES)

print('view-conditional config loaded')
print(f'frontal-only shapes: {len(FRONTAL_ONLY_SHAPES)} | frontal-only devices: {len(FRONTAL_ONLY_DEVICES)}')
print(f'solid opaque: {len(SOLID_OPAQUE)} | tubes: {len(TUBE_ITEMS)} | thin leads: {len(THIN_LEADS)} | skin: {len(SKIN_SURFACE)}')
print(f'view lookup populated: {len(_view_lookup)} entries')

# ---- Lateralization helpers ----
# Radiograph convention: viewer faces patient → patient-left = image-right.
def pick_anchor_with_side(mask, rng, side_bias=None):
    """Sample pixel in mask; side_bias='image_left' or 'image_right' biases the half."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    if side_bias is not None:
        med = np.median(xs)
        sel = (xs > med) if side_bias == 'image_right' else (xs < med)
        if sel.any():
            ys, xs = ys[sel], xs[sel]
    i = rng.randrange(len(ys))
    return int(xs[i]), int(ys[i])

def _patient_to_image_side(patient_side):
    if patient_side == 'left':  return 'image_right'
    if patient_side == 'right': return 'image_left'
    return None

LATERALIZATION = {
    'pacemaker':         ('left',  0.95),
    'icd':               ('left',  0.95),
    'crt_pacemaker':     ('left',  0.95),
    'port_catheter':     ('right', 0.80),
    'tunneled_cvc':      ('right', 0.70),
    'dialysis_catheter': ('right', 0.70),
}

def sample_side_bias(category, rng):
    if category not in LATERALIZATION:
        return None
    side, prob = LATERALIZATION[category]
    chosen = side if rng.random() < prob else ('right' if side == 'left' else 'left')
    return _patient_to_image_side(chosen)


# ---- Tip-position helpers ----
def carina_point(anat_masks, rng):
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if not trachea.any():
        return None
    ys, xs = np.where(trachea)
    bot_idx = int(np.argmax(ys))
    return (int(xs[bot_idx]), int(ys[bot_idx]))

def cavoatrial_junction_point(anat_masks, rng):
    """Top of right atrium / SVC entry (cavoatrial junction)."""
    ra = union_mask(anat_masks, group_ids(['heart atrium right']))
    if ra.any():
        ys, xs = np.where(ra)
        top_idx = int(np.argmin(ys))
        return (int(xs[top_idx]) + rng.randint(-3, 3),
                int(ys[top_idx]) + rng.randint(0, 10))
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    if vess.any():
        ys, xs = np.where(vess)
        top_idx = int(np.argmin(ys))
        return (int(xs[top_idx]), int(ys[top_idx]) + 20)
    return None


def cvc_venous_entry_point(anat_masks, H, W, rng):
    """High lateral venous entry (right/left IJ or subclavian) a central line descends
    from: upper chest, lateral of midline, near/above the clavicle. Kept far from the
    cavoatrial tip so the projected line is long+fairly straight (real CVC length/diag
    ~0.47, tortuosity ~1.2) -- not an up-then-down detour (that would spike tortuosity)."""
    side = rng.choice([-1, 1])                       # patient L/R; RANZCR is balanced
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    midx = W // 2
    if clav.any():
        ys, xs = np.where(clav)
        top_y = int(ys.min())
        sel = (xs > midx) if side > 0 else (xs < midx)
        ex = int(np.median(xs[sel])) if sel.any() else midx + side * (W // 6)
    else:
        top_y = int(0.16 * H)
        ex = midx + side * rng.randint(W // 8, W // 5)
    ey = int(np.clip(top_y - rng.randint(0, int(0.06 * H)), int(0.04 * H), int(0.24 * H)))
    ex = int(np.clip(ex + rng.randint(-int(0.04 * W), int(0.04 * W)), 8, W - 8))
    return (ex, ey)


def _cvc_anatomical_path(entry, tip, anat_masks, H, W, rng):
    """IJ/subclavian lateral entry -> medial sweep into the SVC (just right of the trachea,
    upper mediastinum) -> vertical descent to the CAJ. The characteristic lateral->medial->
    vertical course generic winding misses."""
    trach = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trach.any():
        ys, xs = np.where(trach)
        svc = np.array([float(xs.max()) + rng.uniform(2, 12), float(np.percentile(ys, 60))])
    else:
        svc = np.array([W * 0.55, H * 0.35])
    return _smooth_through(np.array([np.array(entry, float), svc, np.array(tip, float)]), 96, rng)


def _cvc_central_line(H, W, anat_masks, rng):
    """Non-tunneled central venous catheter (IJ/subclavian): venous/skin entry high in
    the upper chest, monotonic descent through the SVC to the cavoatrial junction, plus
    a short external skin-entry segment past the top edge. The dominant real CVC -- long
    and fairly straight -- unlike a chest-wall port (short). No reservoir; catheter_tube
    only. Fixes the CVC routing gap (port reservoir->CAJ was only ~0.20 length/diag)."""
    tip = cavoatrial_junction_point(anat_masks, rng) or (W // 2, int(0.48 * H))
    entry = cvc_venous_entry_point(anat_masks, H, W, rng)
    # WIDE: far-lateral subclavian/axillary entry -> large horizontal sweep so the device
    # bbox is wide-to-square (real CVC median h/w ~0.8), not the tall vertical default (~2.2).
    _wide = os.environ.get('SYNTHFB_CVC_WIDE') == '1'
    if _wide:
        _sd = rng.choice([-1, 1])
        entry = (W // 2 + _sd * rng.randint(int(0.28 * W), int(0.42 * W)), rng.randint(int(0.10 * H), int(0.28 * H)))
    if os.environ.get('SYNTHFB_CVC_PATH') == 'anatomical':
        pts = _cvc_anatomical_path(entry, tip, anat_masks, H, W, rng)   # lateral->medial SVC->CAJ
    else:
        pts = winding_path(np.array(entry, float), np.array(tip, float), rng,
                           n=96, n_waves=rng.uniform(*WIND_CVC_NW), amp_frac=rng.uniform(*WIND_CVC_AF))
    side = 'top_lateral_right' if entry[0] > W // 2 else 'top_lateral_left'
    pts = extend_tube_externally(pts, H, W, rng, direction=side,
                                 length_px=rng.randint(H // 14, H // 8) if _wide else rng.randint(H // 6, H // 3))
    # Appearance env-tunable (device-crop fb-TSTR sweep): real CVC is thin/faint/smooth.
    _cwall = _wind_range('SYNTHFB_CVC_WALL', 190, 225)
    _chw = int(os.environ.get('SYNTHFB_CVC_HALFW', '2'))
    _ctip = float(os.environ.get('SYNTHFB_CVC_TIP', '1.0'))
    tube_mask, tube_color = lumen_tube_mask(
        pts, H, W, half_w=_chw, marker=os.environ.get('SYNTHFB_CVC_MARKER_STYLE', 'paired_walls'),
        wall_color=rng.randint(int(_cwall[0]), int(_cwall[1])),
        tip_radius=0 if _ctip <= 0 else max(1, int(rng.randint(2, 3) * _ctip)),
        tip_color=rng.randint(230, 250),
    )
    return [('catheter_tube', tube_mask, tube_color, _bezier_kpts(pts))]


def below_diaphragm_midline(H, W, anat_masks, rng):
    """Gastric fundus proxy: midline, just below lower lung edge, slightly patient-left."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        cx = int(np.median(xs)) + rng.randint(-5, 25)  # gastric fundus = patient left of midline
        cy = min(H - 10, int(ys.max()) + rng.randint(15, 70))
    else:
        cx, cy = W // 2, 4 * H // 5
    return (int(np.clip(cx, 10, W - 10)), int(np.clip(cy, 10, H - 10)))


# ---- Per-category count distribution ----
COUNT_RANGE = {
    # devices — usually 1 per patient
    'pacemaker': (1, 1), 'icd': (1, 1), 'crt_pacemaker': (1, 1),
    'lvad': (1, 1), 'et_tube': (1, 1), 'double_lumen_et': (1, 1),
    'tracheostomy': (1, 1), 'port_catheter': (1, 1),
    'tunneled_cvc': (1, 1), 'dialysis_catheter': (1, 1),
    'mechanical_valve': (1, 2), 'ecmo': (1, 1), 'evd': (1, 1),
    'vp_shunt': (1, 1), 'shoulder_prosthesis': (1, 1),
    'tissue_expander': (1, 2), 'five_lead_monitor': (1, 1),
    'spinal_rods': (1, 1), 'spinal_cord_stim': (1, 1),
    'bronchial_y_stent': (1, 1), 'sengstaken': (1, 1),
    'gtube': (1, 1), 'sternotomy_wires': (1, 1),
    'halo_traction': (1, 1),
    # shapes — distinct emission events (each can be multi-element internally)
    'pellet_cluster': (1, 1), 'lumpectomy_clips': (1, 1),
    'dotted_screws': (1, 1), 'cabg_clips': (1, 1),
    'staple_line': (1, 2), 'mitra_clip': (1, 1),
    'watchman': (1, 1), 'cardiomems': (1, 1),
    'vertebroplasty': (1, 3),
    'tracheal_stent': (1, 1), 'tevar': (1, 1),
    'mesh_stent': (1, 2),
    'ng_tube': (1, 1), 'picc_line': (1, 1),
    'swan_ganz': (1, 1), 'pacing_wire': (1, 2),
    'iabp': (1, 1), 'pleurx': (1, 1),
    'chest_tube': (1, 2), 'mediastinal_drain': (1, 2),
    'parallel': (1, 1), 'esophageal_stent': (1, 1),
    'ablation_antenna': (1, 2), 'impella': (1, 1),
    'pericardial_drain': (1, 1), 'uac_uvc': (1, 1),
    'textual': (1, 4), 'circular': (1, 3), 'ring': (1, 3),
    'rect': (1, 3), 'clip': (1, 5), 'grid': (1, 2),
    'line': (1, 3), 'blob': (1, 2), 'sdf': (1, 2),
    'branching_tree': (1, 2), 'scratch': (1, 3),
    'ring_chain': (1, 2), 'ekg_bundle': (1, 1),
    'offframe_cables': (1, 1), 'vent_hose': (1, 1),
    'acquisition_burn_in': (1, 1),
    'gossypiboma': (1, 2), 'ivc_filter': (1, 1),
    'breast_implant': (1, 2), 'bra_underwire': (1, 2),
    'aspirated_fb': (1, 2), 'coin_ingested': (1, 1),
    'button_battery': (1, 1), 'magnet_pair': (1, 2),
    'safety_pin': (1, 1), 'hairpin': (1, 1),
    'cervical_plate': (1, 1),
    'endobronchial_valve': (1, 5),
    'cochlear': (1, 2), 'embolization_coils': (1, 2),
    'defib_pads': (1, 1), 'brachytherapy_seeds': (1, 1),
}

def sample_count(category, rng):
    lo, hi = COUNT_RANGE.get(category, (1, 1))
    return rng.randint(lo, hi)


# ---- Patient archetypes ----
ARCHETYPES = {
    'no_finding':         {'required': [], 'optional': [], 'extra': (0, 0), 'weight': 0.25},
    'minimal_labels':     {'required': [('shape', 'textual')], 'optional': [('shape', 'ring_chain'), ('shape', 'scratch')], 'extra': (0, 1), 'weight': 0.18},
    'chronic_pacemaker':  {'required': [('device', 'pacemaker')], 'optional': [('device', 'sternotomy_wires')], 'extra': (0, 1), 'weight': 0.05},
    'icd_patient':        {'required': [('device', 'icd')], 'optional': [], 'extra': (0, 1), 'weight': 0.03},
    'crt_patient':        {'required': [('device', 'crt_pacemaker')], 'optional': [], 'extra': (0, 1), 'weight': 0.02},
    'lvad_patient':       {'required': [('device', 'lvad')], 'optional': [('device', 'icd')], 'extra': (0, 2), 'weight': 0.01},
    'post_cabg':          {'required': [('device', 'sternotomy_wires')], 'optional': [('shape', 'chest_tube'), ('shape', 'mediastinal_drain'), ('device', 'et_tube'), ('device', 'tunneled_cvc'), ('shape', 'cabg_clips')], 'extra': (0, 2), 'weight': 0.05},
    'arrest_intubated':   {'required': [('device', 'et_tube'), ('device', 'five_lead_monitor'), ('shape', 'offframe_cables'), ('shape', 'acquisition_burn_in')], 'optional': [('shape', 'defib_pads'), ('device', 'tunneled_cvc'), ('shape', 'vent_hose')], 'extra': (0, 2), 'weight': 0.04},
    'tavr_patient':       {'required': [('shape', 'mesh_stent')], 'optional': [('device', 'pacemaker')], 'extra': (0, 1), 'weight': 0.02},
    'port_chemo':         {'required': [('device', 'port_catheter')], 'optional': [('shape', 'textual')], 'extra': (0, 2), 'weight': 0.05},
    'dialysis_patient':   {'required': [('device', 'dialysis_catheter')], 'optional': [], 'extra': (0, 2), 'weight': 0.04},
    'icu_complex':        {'required': [('device', 'et_tube'), ('device', 'tunneled_cvc'), ('device', 'five_lead_monitor'), ('shape', 'offframe_cables'), ('shape', 'acquisition_burn_in')], 'optional': [('shape', 'ng_tube'), ('shape', 'chest_tube'), ('shape', 'vent_hose')], 'extra': (1, 3), 'weight': 0.06},
    'ecmo_patient':       {'required': [('device', 'ecmo'), ('device', 'et_tube'), ('shape', 'offframe_cables'), ('shape', 'acquisition_burn_in')], 'optional': [('shape', 'ng_tube'), ('shape', 'vent_hose')], 'extra': (0, 3), 'weight': 0.02},
    'oncology_brachy':    {'required': [('shape', 'brachytherapy_seeds')], 'optional': [('shape', 'lumpectomy_clips')], 'extra': (0, 2), 'weight': 0.03},
    'spine_surgery':      {'required': [('device', 'spinal_rods')], 'optional': [('shape', 'staple_line')], 'extra': (0, 2), 'weight': 0.03},
    'trauma':             {'required': [('shape', 'pellet_cluster')], 'optional': [('shape', 'aspirated_fb'), ('shape', 'scratch')], 'extra': (0, 3), 'weight': 0.02},
    'pediatric_ingest':   {'required': [], 'optional': [('shape', 'coin_ingested'), ('shape', 'button_battery'), ('shape', 'magnet_pair'), ('shape', 'safety_pin')], 'extra': (1, 2), 'weight': 0.03},
    'breast_recon':       {'required': [('shape', 'breast_implant')], 'optional': [('device', 'tissue_expander'), ('shape', 'lumpectomy_clips')], 'extra': (0, 2), 'weight': 0.02},
    'cancer_lung':        {'required': [('shape', 'ablation_antenna')], 'optional': [('shape', 'aspirated_fb')], 'extra': (0, 1), 'weight': 0.01},
    'neuro_shunt':        {'required': [('device', 'vp_shunt')], 'optional': [('shape', 'cervical_plate')], 'extra': (0, 1), 'weight': 0.02},
    'iabp_patient':       {'required': [('shape', 'iabp')], 'optional': [('device', 'et_tube'), ('device', 'tunneled_cvc')], 'extra': (0, 2), 'weight': 0.01},
    'pulmonary_emboli':   {'required': [('shape', 'embolization_coils')], 'optional': [], 'extra': (0, 1), 'weight': 0.01},
    'pleural_effusion':   {'required': [('shape', 'chest_tube')], 'optional': [('shape', 'pleurx')], 'extra': (0, 2), 'weight': 0.02},
}
_total_w = sum(a['weight'] for a in ARCHETYPES.values())
ARCHETYPE_WEIGHTS = {k: v['weight'] / _total_w for k, v in ARCHETYPES.items()}

def pick_archetype(rng):
    r = rng.random()
    acc = 0.0
    for name, w in ARCHETYPE_WEIGHTS.items():
        acc += w
        if r <= acc:
            return name
    return next(iter(ARCHETYPES))

print(f'archetypes: {len(ARCHETYPES)} | normalized weight sum: {sum(ARCHETYPE_WEIGHTS.values()):.3f}')
print(f'count_range entries: {len(COUNT_RANGE)}')
print(f'lateralization rules: {len(LATERALIZATION)}')

# ---- v2 device/shape fns with lateralization + tip priors ----

def _textured_body(H, W, cx, cy, bw, bh, rng, angle=0):
    """Build (mask, 2D color_map) for pacemaker/ICD/CRT/LVAD-style implant body.

    Realism details:
      - antialiased outer shell (LINE_AA, smooth boundary)
      - large rounded battery cell (lower-mid, darker than shell, faint rim)
      - circuit-board mosaic (upper, grid of small dark squares)
      - bright header rectangle at top w/ 3-4 dark connector pins
      - faint serial / model marking line
    All structural elements are drawn anti-aliased; intensity range varies per instance.
    """
    body_mask = np.zeros((H, W), np.uint8)
    color_map = np.zeros((H, W), np.uint8)
    cv2.ellipse(body_mask, (cx, cy), (bw // 2, bh // 2), angle, 0, 360, 255, -1, cv2.LINE_AA)
    # Reduced base brightness — was 210-240 (too bright; bodies popped off the chest).
    # Real PM/ICD bodies sit closer to mid-grey, ~170-205, with rim highlighting.
    base = rng.randint(170, 205)
    color_map[body_mask > 0] = base

    # ---- battery cell: rounded rect (drawn as ellipse), occupies lower ~55% width, ~40% height ----
    batt_w = max(8, int(bw * rng.uniform(0.45, 0.6)))
    batt_h = max(6, int(bh * rng.uniform(0.32, 0.42)))
    batt_cx = cx + rng.randint(-2, 2)
    batt_cy = cy + int(bh * rng.uniform(0.05, 0.18))
    batt_color = max(120, base - rng.randint(25, 55))
    cv2.ellipse(color_map, (batt_cx, batt_cy), (batt_w // 2, batt_h // 2),
                0, 0, 360, batt_color, -1, cv2.LINE_AA)
    cv2.ellipse(color_map, (batt_cx, batt_cy), (batt_w // 2, batt_h // 2),
                0, 0, 360, max(60, batt_color - 35), 1, cv2.LINE_AA)

    # ---- circuit-board mosaic: grid of small dark squares above battery ----
    cb_w = max(6, int(bw * rng.uniform(0.30, 0.45)))
    cb_h = max(4, int(bh * rng.uniform(0.14, 0.22)))
    cb_cy = cy - int(bh * rng.uniform(0.15, 0.25))
    cb_x0 = cx - cb_w // 2
    cb_y0 = cb_cy - cb_h // 2
    n_cols = rng.randint(3, 5); n_rows = rng.randint(2, 3)
    chip_dark = max(50, base - rng.randint(110, 150))
    for i in range(n_cols):
        for j in range(n_rows):
            sx = cb_x0 + i * (cb_w // max(1, n_cols)) + rng.randint(-1, 1)
            sy = cb_y0 + j * (cb_h // max(1, n_rows)) + rng.randint(-1, 1)
            sz = rng.randint(2, 4)
            if 0 <= sx < W - sz and 0 <= sy < H - sz:
                cv2.rectangle(color_map, (sx, sy), (sx + sz, sy + sz), chip_dark, -1)

    # ---- header: bright rectangle at top edge w/ 3-4 connector pins ----
    hw = max(10, int(bw * rng.uniform(0.40, 0.55)))
    hh = max(5, int(bh * rng.uniform(0.10, 0.16)))
    hx = cx - hw // 2
    hy = cy - bh // 2 - hh + 3
    if 0 <= hx and hx + hw < W and 0 <= hy and hy + hh < H:
        cv2.rectangle(body_mask, (hx, hy), (hx + hw, hy + hh), 255, -1, cv2.LINE_AA)
        # Header bar was darker but its DELTA from body was high enough that it
        # popped as a separate rectangle on top of the body. Make it closer to body
        # intensity (smaller delta) and add a soft rim instead of a hard fill.
        header_c = max(110, base - rng.randint(8, 20))
        cv2.rectangle(color_map, (hx, hy), (hx + hw, hy + hh), header_c, -1)
        # softer pin contrast: 70 (was 35) + smaller radius
        n_pins = rng.randint(3, 4)
        for i in range(n_pins):
            px = hx + (i + 1) * hw // (n_pins + 1)
            py = hy + hh // 2
            cv2.circle(color_map, (px, py), 1, max(60, base - 80), -1, cv2.LINE_AA)

    # ---- faint serial / model marking line ----
    if rng.random() < 0.6:
        text_y = cy + int(bh * rng.uniform(0.30, 0.40))
        text_w = int(bw * rng.uniform(0.20, 0.35))
        cv2.line(color_map, (cx - text_w // 2, text_y), (cx + text_w // 2, text_y),
                 max(80, base - 65), 1, cv2.LINE_AA)

    return body_mask, color_map


def device_pacemaker_v2(H, W, anat_masks, rng):
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('pacemaker', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = 3 * W // 4, H // 4  # patient-left fallback
    else:
        cx, cy = pt[0], pt[1] + rng.randint(20, 60)
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    parts = []
    bw, bh = rng.randint(30, 45), rng.randint(38, 55)
    body_mask, body_color = _textured_body(H, W, cx, cy, bw, bh, rng)
    parts.append(('pacemaker_body', body_mask, body_color, {'center': (cx, cy, 2)}))
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    n_leads = rng.randint(1, 3)
    for _ in range(n_leads):
        parts.append(_build_pacing_lead(H, W, cx, cy, bh, heart_u, rng,
                                         lead_cat='pacemaker_lead'))
    return parts


def _build_pacing_lead(H, W, gen_cx, gen_cy, gen_bh, heart_mask, rng, lead_cat='pacemaker_lead'):
    """Construct a realistic pacing-lead path. After v20 added an explicit midline
    crossover anchor, leads looked zigzag/unnatural. Reverted to:

      1) Service loop near generator (12-22px radius, downward arc) — slack for arm motion.
      2) Single smooth Bezier from loop exit to heart tip, with a MEDIAL TANGENT (control
         points biased toward chest midline) so the lead curves through subclavian + SVC
         rather than going straight diagonal to the tip.
    """
    p_gen = np.array([gen_cx, gen_cy + gen_bh // 2], float)
    end = sample_point_in_mask(heart_mask, rng) if heart_mask.any() else (gen_cx, min(H - 1, gen_cy + 150))
    p_tip = np.array(end, float)

    # 1) Service loop — small downward arc near the generator
    loop_r = rng.randint(12, 22)
    loop_cx = gen_cx + rng.randint(-3, 3)
    loop_cy = gen_cy + gen_bh // 2 + rng.randint(6, 12)
    n_loop = 28
    # arc that starts pointing down-right out of the generator, sweeps clockwise ~200°
    angles = np.linspace(np.pi * 1.3, np.pi * 1.3 + np.pi * 1.4, n_loop)
    loop_pts = np.stack([
        loop_cx + loop_r * np.cos(angles),
        loop_cy + loop_r * np.sin(angles),
    ], axis=1)

    # 2) Smooth Bezier from loop exit to heart tip. Control points are biased toward
    # the body midline so the path curves through subclavian → SVC, matching real anatomy.
    p_loop_exit = np.array(loop_pts[-1])
    body_midline_x = W // 2
    # control 1: pull toward midline at upper-third of descent
    c1 = p_loop_exit + np.array([
        0.4 * (body_midline_x - p_loop_exit[0]) + rng.uniform(-8, 8),
        0.35 * (p_tip[1] - p_loop_exit[1]) + rng.uniform(-5, 5),
    ], float)
    # control 2: near tip, slight overshoot opposite side for natural S
    c2 = p_tip + np.array([
        rng.uniform(-15, 15),
        -0.25 * (p_tip[1] - p_loop_exit[1]) + rng.uniform(-5, 5),
    ], float)
    descent = bezier_points(p_loop_exit, c1, c2, p_tip, n=96)

    pts = np.concatenate([loop_pts, descent], axis=0).astype(np.int32)
    pts = jitter_path(pts, rng, amp_px=0.4)

    lead = np.zeros((H, W), np.uint8)
    color_map = np.zeros((H, W), np.uint8)
    body_c = rng.randint(180, 210)
    cv2.polylines(lead,      [pts], False, 255,  1, cv2.LINE_AA)
    cv2.polylines(color_map, [pts], False, body_c, 1, cv2.LINE_AA)
    tip = (int(pts[-1][0]), int(pts[-1][1]))
    cv2.circle(lead,      tip, 2, 255, -1)
    cv2.circle(color_map, tip, 2, rng.randint(240, 255), -1)
    return (lead_cat, lead, color_map, _bezier_kpts(pts))


def device_icd_v2(H, W, anat_masks, rng):
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('icd', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = 3 * W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(20, 60)
    cx = int(np.clip(cx, 30, W - 30)); cy = int(np.clip(cy, 30, H - 30))
    parts = []
    bw, bh = rng.randint(48, 65), rng.randint(55, 72)
    body, body_color = _textured_body(H, W, cx, cy, bw, bh, rng)
    parts.append(('icd_body', body, body_color, {'center': (cx, cy, 2)}))
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    n_leads = rng.randint(1, 3)
    for _ in range(n_leads):
        # Reuse the pacing-lead builder (service loop + midline crossover + curve).
        # Then overlay the dashed-segment shock coil texture that's specific to ICD leads.
        _, lead_mask, lead_color, lead_kpts = _build_pacing_lead(
            H, W, cx, cy, bh, heart_u, rng, lead_cat='icd_lead')
        # ICD leads carry a SHOCK COIL — a thicker dashed segment about 70-95% along the
        # path. Apply that as the visible thickening segment.
        # Reconstruct the path from non-zero pixels of lead_mask (rough)
        ys, xs = np.where(lead_mask > 0)
        if len(ys) > 30:
            # use kpts ordering: prox -> mid_a -> mid_b -> tip
            kp = lead_kpts
            tail_x, tail_y = kp['tip'][0], kp['tip'][1]
            mid_x, mid_y = kp['mid_b'][0], kp['mid_b'][1]
            # draw thicker line between mid_b and tip (~25% of path) — the shock coil
            cv2.line(lead_mask, (mid_x, mid_y), (tail_x, tail_y), 255, 3, cv2.LINE_AA)
            cv2.line(lead_color, (mid_x, mid_y), (tail_x, tail_y),
                     rng.randint(195, 225), 3, cv2.LINE_AA)
        parts.append(('icd_lead', lead_mask, lead_color, lead_kpts))
    return parts


def device_crt_pacemaker_v2(H, W, anat_masks, rng):
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('crt_pacemaker', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = 3 * W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(20, 60)
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    parts = []
    bw, bh = rng.randint(35, 50), rng.randint(45, 60)
    body, body_color = _textured_body(H, W, cx, cy, bw, bh, rng)
    parts.append(('crt_body', body, body_color, {'center': (cx, cy, 2)}))
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    for label, names in [
        ('crt_lead_ra', ['heart atrium right']),
        ('crt_lead_rv', ['heart ventricle right']),
        ('crt_lead_lv', ['heart ventricle left']),
    ]:
        chamber = union_mask(anat_masks, group_ids(names))
        if not chamber.any():
            chamber = heart_u
        end = sample_point_in_mask(chamber, rng) if chamber.any() else (cx, min(H - 1, cy + 150))
        p0 = np.array([cx, cy + bh // 2], float)
        p3 = np.array(end, float)
        p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-25, 25), rng.randint(-15, 30)], float)
        p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-35, 35), rng.randint(-15, 30)], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        lead = np.zeros((H, W), np.uint8)
        color_map = np.zeros((H, W), np.uint8)
        body_c = rng.randint(180, 210)
        cv2.polylines(lead,      [pts], False, 255,  1, cv2.LINE_AA)
        cv2.polylines(color_map, [pts], False, body_c, 1, cv2.LINE_AA)
        tip = (int(pts[-1][0]), int(pts[-1][1]))
        cv2.circle(lead,      tip, 2, 255, -1)
        cv2.circle(color_map, tip, 2, rng.randint(240, 255), -1)
        parts.append((label, lead, color_map, _bezier_kpts(pts)))
    return parts


def device_port_catheter_v2(H, W, anat_masks, rng):
    """Port-a-cath: translucent reservoir w/ bright central septum dot, tube exits at edge."""
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('port_catheter', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(30, 80)
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    r = rng.randint(10, 18)
    res = np.zeros((H, W), np.uint8)
    cv2.circle(res, (cx, cy), r, 255, -1)
    septum = np.zeros((H, W), np.uint8)
    cv2.circle(septum, (cx, cy), max(2, r // 4), 255, -1)
    tip = cavoatrial_junction_point(anat_masks, rng) or (W // 2, H // 2)
    # tube exits laterally (toward midline) not bottom-center
    side_sgn = +1 if cx < W // 2 else -1
    exit_pt = (cx + side_sgn * r, cy)
    p0 = np.array(exit_pt, float); p3 = np.array(tip, float)
    pts = winding_path(p0, p3, rng, n=96, n_waves=rng.uniform(*WIND_CVC_NW), amp_frac=rng.uniform(*WIND_CVC_AF))  # CVCs curve into the SVC (real tortuosity ~1.2)
    tube_mask, tube_color = lumen_tube_mask(
        pts, H, W, half_w=2, marker='paired_walls',
        wall_color=rng.randint(190, 225),
        tip_radius=rng.randint(2, 3), tip_color=rng.randint(230, 250),
    )
    return [
        ('port_reservoir', res,    rng.randint(200, 235), {'center': (cx, cy, 2)}),
        ('port_septum',    septum, rng.randint(235, 255), {'center': (cx, cy, 2)}),
        ('catheter_tube',  tube_mask, tube_color, _bezier_kpts(pts)),
    ]


def device_tunneled_cvc_v2(H, W, anat_masks, rng):
    """Tunneled CVC (Hickman/Broviac): skin exit (low, lateral chest) → subQ tunnel
    coursing up over clavicle → venous entry → SVC tip. The tunnel + Dacron cuff are
    what distinguish a tunneled CVC from a regular CVC. Adds the cuff as a small
    radio-opaque bead in the middle of the tunnel segment.
    """
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('tunneled_cvc', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(60, 110)        # lower exit (chest wall)
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    exit_dot = np.zeros((H, W), np.uint8)
    cv2.circle(exit_dot, (cx, cy), rng.randint(4, 6), 255, -1)

    # SUBQ TUNNEL: course up + medial to clavicle entry. This is a distinct anatomic
    # segment from the venous portion.
    tunnel_x = cx + rng.randint(-15, 15)
    tunnel_y = cy - rng.randint(40, 65)
    tip = cavoatrial_junction_point(anat_masks, rng) or (W // 2, H // 4)
    p_exit = np.array([cx, cy], float)
    p_tunnel = np.array([tunnel_x, tunnel_y], float)
    p_tip = np.array(tip, float)
    # 2-segment Bezier: exit → tunnel landmark → tip
    seg1 = bezier_points(
        p_exit,
        p_exit + np.array([rng.uniform(-10, 10), -rng.uniform(15, 30)], float),
        p_tunnel + np.array([rng.uniform(-8, 8), rng.uniform(5, 15)], float),
        p_tunnel, n=32)
    seg2 = bezier_points(
        p_tunnel,
        p_tunnel + np.array([rng.uniform(-15, 15), rng.uniform(10, 25)], float),
        p_tip + np.array([rng.uniform(-10, 10), -rng.uniform(15, 30)], float),
        p_tip, n=64)
    pts = np.concatenate([seg1, seg2], axis=0).astype(np.int32)
    pts = jitter_path(pts, rng, amp_px=0.4)

    tube_mask, tube_color = lumen_tube_mask(
        pts, H, W, half_w=2, marker='paired_walls',
        wall_color=rng.randint(210, 240),
        tip_radius=rng.randint(3, 5), tip_color=rng.randint(245, 255),
    )
    # DACRON CUFF: small radio-opaque bead ~1/3 of the way along the tunnel segment.
    # Helps the tube tissue-anchor; on CXR it shows as a small bright thickening.
    if len(seg1) >= 12:
        cuff_idx = len(seg1) // 2
        cv2.circle(tube_mask,  tuple(int(v) for v in seg1[cuff_idx]), 4, 255, -1, cv2.LINE_AA)
        cv2.circle(tube_color, tuple(int(v) for v in seg1[cuff_idx]), 4,
                   rng.randint(240, 255), -1, cv2.LINE_AA)
    return [
        ('tunneled_exit', exit_dot, rng.randint(225, 250), {'center': (cx, cy, 2)}),
        ('tunneled_tube', tube_mask, tube_color, _bezier_kpts(pts)),
    ]


def device_dialysis_catheter_v2(H, W, anat_masks, rng):
    """Dialysis catheter: large skin-exit hub + 2 sub-port dots + thick paired-wall tube + Y-split."""
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('dialysis_catheter', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(40, 90)
    cx = int(np.clip(cx, 20, W - 20)); cy = int(np.clip(cy, 20, H - 20))
    # Y-HUB at skin exit: bigger + brighter, with two distinguishable ports separated
    # by a clear gap (arterial + venous limbs).
    hub_r = rng.randint(10, 16)
    exit_dot = np.zeros((H, W), np.uint8)
    # outer hub disc
    cv2.circle(exit_dot, (cx, cy), hub_r, 255, -1, cv2.LINE_AA)
    # two distinct port openings (brighter rim showing as bright "donuts")
    port_off = hub_r // 2
    cv2.circle(exit_dot, (cx - port_off, cy), 3, 0, -1, cv2.LINE_AA)
    cv2.circle(exit_dot, (cx + port_off, cy), 3, 0, -1, cv2.LINE_AA)
    # small connecting "neck" between the two ports
    cv2.line(exit_dot, (cx - port_off, cy), (cx + port_off, cy), 255, 1, cv2.LINE_AA)
    tip = cavoatrial_junction_point(anat_masks, rng) or (W // 2, H // 4)
    p0 = np.array([cx, cy + hub_r], float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-20, 20), rng.randint(-15, 15)], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-20, 20), rng.randint(-15, 15)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    # ~15% of dialysis catheters project the IJ/subclavian entry above the
    # image top, especially when the camera framing is tight.
    if rng.random() < 0.15:
        # Flip the path so tip is at the top, extend past top edge
        pts = pts[::-1]
        pts = extend_tip_oob(pts, H, W, rng, direction='top', length_px=rng.randint(30, 80))
    tube, tube_color = lumen_tube_mask(
        pts, H, W, half_w=rng.randint(5, 7),                # thicker (real dialysis ~12 Fr)
        marker='paired_walls',
        wall_color=rng.randint(215, 245),
        fill_color=rng.randint(115, 155),
        tip_radius=0,
    )
    # Dacron cuff bead about 1/3 down the tube
    if len(pts) >= 18:
        cuff_idx = len(pts) // 3
        cv2.circle(tube,       tuple(int(v) for v in pts[cuff_idx]), 4, 255, -1, cv2.LINE_AA)
        cv2.circle(tube_color, tuple(int(v) for v in pts[cuff_idx]), 4,
                   rng.randint(240, 255), -1, cv2.LINE_AA)
    tip_arr = np.array(tip, float)
    pre_idx = max(0, len(pts) - 6)
    tan = (tip_arr - pts[pre_idx].astype(float))
    norm_t = tan / (np.linalg.norm(tan) + 1e-6)
    perp = np.array([-norm_t[1], norm_t[0]])
    split_len = rng.randint(10, 18)
    y_split = np.zeros((H, W), np.uint8)
    for sign in (-1, +1):
        endp = tip_arr + norm_t * 5 + perp * sign * split_len
        cv2.line(y_split, (int(tip_arr[0]), int(tip_arr[1])),
                 (int(endp[0]), int(endp[1])), 255, rng.randint(1, 2), cv2.LINE_AA)
    # y-split endpoints
    arms = []
    for sign in (-1, +1):
        endp = tip_arr + norm_t * 5 + perp * sign * split_len
        arms.append((int(endp[0]), int(endp[1])))
    return [
        ('dialysis_exit',   exit_dot,  rng.randint(220, 250), {'center': (cx, cy, 2)}),
        ('dialysis_tube',   tube,      tube_color,            _bezier_kpts(pts)),
        ('dialysis_y_split', y_split, rng.randint(210, 245),
         {'junction': (int(tip_arr[0]), int(tip_arr[1]), 2),
          'arm_a': (arms[0][0], arms[0][1], 2),
          'arm_b': (arms[1][0], arms[1][1], 2)}),
    ]


def device_et_tube_v2(H, W, anat_masks, rng):
    """ET tube: tip 2-7cm ABOVE carina. Shaft = paired walls + dashed radio-opaque centerline +
    Murphy eye near tip. Cuff = inflated 3D-rim ellipse below tip.
    Now extends the external/oral portion above mouth (~20% of tube length leaves frame top)."""
    car = carina_point(anat_masks, rng) or (W // 2, H // 3)
    cx_c, cy_c = car
    tip = (cx_c + rng.randint(-5, 5), max(0, cy_c - rng.randint(30, 100)))
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        top_idx = int(np.argmin(ys))
        entry = (int(xs[top_idx]) + rng.randint(-10, 10), int(ys[top_idx]) - rng.randint(30, 60))
    else:
        entry = (W // 2, 20)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-15, 15), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    pts = jitter_path(pts, rng, amp_px=0.6)
    pts = extend_tube_externally(pts, H, W, rng, direction='top',
                                  length_px=H // 3, lateral_drift=10)
    # Appearance env-tunable (device-crop fb-TSTR sweep): ETT likes FULL-CONTRAST (faint hurts).
    # SYNTHFB_ETT_HALFW/WALL='lo,hi', SYNTHFB_ETT_CUFF=scale (0=no cuff).
    _ehw = _wind_range('SYNTHFB_ETT_HALFW', 6, 10)
    _ewall = _wind_range('SYNTHFB_ETT_WALL', 225, 250)
    half = rng.randint(int(_ehw[0]), int(_ehw[1]))
    shaft_mask, shaft_color = lumen_tube_mask(
        pts, H, W, half_w=half, marker='dashed',
        wall_color=rng.randint(int(_ewall[0]), int(_ewall[1])),
        fill_color=rng.randint(95, 145),
        marker_color=rng.randint(245, 255),
        murphy_eye=True,
    )
    # Distal tip MARKER — small bright transverse line just proximal to the cuff.
    # Real ETT has this radio-opaque "depth line" so the radiologist can locate the tip.
    if len(pts) >= 10:
        tip_idx = max(2, len(pts) - 4)
        tipp = pts[tip_idx]
        # perpendicular bar 2-3 px each side
        tangent = pts[tip_idx] - pts[tip_idx - 2]
        norm = np.array([-tangent[1], tangent[0]], dtype=float)
        norm = norm / (np.linalg.norm(norm) + 1e-6)
        bar_half = half + 1
        x1 = int(tipp[0] + norm[0] * bar_half); y1 = int(tipp[1] + norm[1] * bar_half)
        x2 = int(tipp[0] - norm[0] * bar_half); y2 = int(tipp[1] - norm[1] * bar_half)
        cv2.line(shaft_mask,  (x1, y1), (x2, y2), 255, 2, cv2.LINE_AA)
        cv2.line(shaft_color, (x1, y1), (x2, y2), rng.randint(245, 255), 2, cv2.LINE_AA)
    # Proximal CONNECTOR — Y-adapter visible above the mouth. Plug a wider rectangle
    # at the external-most point. This breaks the "thin line above mouth" look.
    if len(pts) >= 4:
        p_ext = pts[0]
        next_p = pts[2]
        ang = np.arctan2(next_p[1] - p_ext[1], next_p[0] - p_ext[0])
        conn_len = rng.randint(8, 14)
        conn_w = half + rng.randint(3, 6)
        # rectangle aligned with tube axis at p_ext
        cv2.line(shaft_mask,
                 (int(p_ext[0]), int(p_ext[1])),
                 (int(p_ext[0] - conn_len * np.cos(ang)),
                  int(p_ext[1] - conn_len * np.sin(ang))),
                 255, conn_w * 2, cv2.LINE_AA)
        cv2.line(shaft_color,
                 (int(p_ext[0]), int(p_ext[1])),
                 (int(p_ext[0] - conn_len * np.cos(ang)),
                  int(p_ext[1] - conn_len * np.sin(ang))),
                 rng.randint(200, 235), conn_w * 2, cv2.LINE_AA)
    _ecuff = float(os.environ.get('SYNTHFB_ETT_CUFF', '1.0'))
    parts = [('et_tube_shaft', shaft_mask, shaft_color, _bezier_kpts(pts))]
    if _ecuff > 0:
        cuff = np.zeros((H, W), np.uint8)
        cuff_color = np.zeros((H, W), np.uint8)
        render_3d_cuff(cuff, cuff_color, int(tip[0]), int(tip[1]),
                       max(2, int((half + rng.randint(5, 8)) * _ecuff)),
                       max(2, int((half + rng.randint(4, 6)) * _ecuff)),
                       rim_intensity=rng.randint(238, 255),
                       interior_intensity=rng.randint(135, 175))
        parts.append(('et_tube_cuff', cuff, cuff_color, {'center': (int(tip[0]), int(tip[1]), 2)}))
    return parts


def device_tracheostomy_v2(H, W, anat_masks, rng):
    """Tracheostomy: short paired-wall tube w/ dashed centerline, 3D-rim cuff at tip.
    External hub extends slightly anteriorly above tracheal entry."""
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        top_idx = int(np.argmin(ys))
        tip = (int(xs[top_idx]) + rng.randint(-5, 5), int(ys[top_idx]) + rng.randint(5, 30))
    else:
        tip = (W // 2, H // 4)
    entry = (tip[0] + rng.randint(-20, 20), tip[1] - rng.randint(20, 40))
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.5 + np.array([rng.randint(-5, 5), 0], float)
    p2 = p0 + (p3 - p0) * 0.8 + np.array([rng.randint(-5, 5), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=64)
    pts = jitter_path(pts, rng, amp_px=0.4)                 # short tube, small jitter
    pts = extend_tube_externally(pts, H, W, rng, direction='top',
                                  length_px=H // 8)          # small external hub
    half = rng.randint(5, 8)
    shaft_mask, shaft_color = lumen_tube_mask(
        pts, H, W, half_w=half, marker='dashed',
        wall_color=rng.randint(180, 215),
        fill_color=rng.randint(70, 120),
        marker_color=rng.randint(225, 245),
    )
    cuff = np.zeros((H, W), np.uint8)
    cuff_color = np.zeros((H, W), np.uint8)
    render_3d_cuff(cuff, cuff_color, int(tip[0]), int(tip[1]),
                   half + rng.randint(3, 5), half + rng.randint(2, 3),
                   rim_intensity=rng.randint(220, 245),
                   interior_intensity=rng.randint(120, 160))
    return [
        ('trach_shaft', shaft_mask, shaft_color, _bezier_kpts(pts)),
        ('trach_cuff',  cuff,       cuff_color,
         {'center': (int(tip[0]), int(tip[1]), 2)}),
    ]


def shape_ng_tube_v2(H, W, anat_masks, rng):
    """NG tube. Real NG appearance: faint thin tube body PLUS very bright continuous
    radio-opaque CENTERLINE STRIPE (the stripe is what radiologists actually see at the
    tube's location). Previously the body/stripe contrast was too low — bump the marker
    color into the 240-255 range so the stripe really pops, with body at 130-170."""
    entry = (W // 2 + rng.randint(-15, 15), rng.randint(0, 30))
    tip = below_diaphragm_midline(H, W, anat_masks, rng)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-20, 20), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    pts = jitter_path(pts, rng, amp_px=0.7)
    pts = extend_tube_externally(pts, H, W, rng, direction='top_nostril',
                                  length_px=H // 5)
    # ~25% of chest CXRs crop the upper abdomen tightly so the gastric NG tip
    # is below the bottom edge. Extending past the bottom flips the tip kpt to
    # v=0 automatically.
    tip_oob = rng.random() < 0.25
    if tip_oob:
        pts = extend_tip_oob(pts, H, W, rng, direction='bottom')
    # Appearance env-tunable (device-crop fb-TSTR sweep): SYNTHFB_NGT_WALL/MARKER='lo,hi',
    # NGT_HALFW=int, NGT_TIP=scale (0 = no gastric tip dot/side-holes), NGT_MARKER_STYLE.
    _nwall = _wind_range('SYNTHFB_NGT_WALL', 130, 165)
    _nmark = _wind_range('SYNTHFB_NGT_MARKER', 245, 255)
    _nhw = int(os.environ.get('SYNTHFB_NGT_HALFW', '1'))
    _ntip = float(os.environ.get('SYNTHFB_NGT_TIP', '1.0'))
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=_nhw, marker=os.environ.get('SYNTHFB_NGT_MARKER_STYLE', 'continuous_stripe'),
        wall_color=rng.randint(int(_nwall[0]), int(_nwall[1])),
        marker_color=rng.randint(int(_nmark[0]), int(_nmark[1])),
        tip_radius=0 if (tip_oob or _ntip <= 0) else max(1, int(rng.randint(3, 5) * _ntip)),
        tip_color=rng.randint(248, 255),
    )
    # SIDE HOLES near tip: 2 small bright dots offset 4-6 px proximal to the tip,
    # representing the radio-opaque markers that bracket side ports on real NG tubes.
    # Use the LAST IN-FRAME pt as the side-port anchor when the tip is OOB.
    if len(pts) >= 10 and _ntip > 0:
        # find last in-frame index
        last_in = len(pts) - 1
        while last_in > 0 and not (0 <= pts[last_in][0] < W and 0 <= pts[last_in][1] < H):
            last_in -= 1
        for back in (4, 8):
            i = max(2, last_in - back)
            cv2.circle(mask,      (int(pts[i][0]), int(pts[i][1])), 2, 255, -1)
            cv2.circle(color_map, (int(pts[i][0]), int(pts[i][1])), 2, rng.randint(245, 255), -1)
    return mask, color_map, _bezier_kpts(pts)


def shape_picc_line_v2(H, W, anat_masks, rng):
    """PICC: enters at upper-arm basilic/cephalic vein, courses through axillary →
    subclavian → SVC, terminates at cavoatrial junction. Path is multi-segment
    (not a single Bezier) so it has the characteristic medial curving sweep:
        upper-arm entry (very lateral) → axillary (medial+down) → subclavian
        (sharp medial turn) → SVC (vertical down) → cavoatrial junction tip.
    Line body is very faint (real PICC is thin); tip is the only reliably
    radiopaque part. Lateral entry is OUTSIDE the chest (arm region).
    """
    side = rng.choice([-1, +1])

    # 1) Upper-arm entry point — well lateral of clavicle (outside thorax)
    arm_x = W // 2 + side * rng.randint(int(W * 0.32), int(W * 0.45))
    arm_y = rng.randint(int(H * 0.18), int(H * 0.32))
    p_arm = np.array([arm_x, arm_y], float)

    # 2) Axillary vein anchor — closer to clavicle head, slightly inferior
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        if sel.any():
            i = rng.randrange(int(sel.sum()))
            ax_x = int(xs[sel][i])
            ax_y = int(ys[sel][i]) + rng.randint(8, 18)
        else:
            ax_x = W // 2 + side * W // 5
            ax_y = arm_y + rng.randint(15, 30)
    else:
        ax_x = W // 2 + side * W // 5
        ax_y = arm_y + rng.randint(15, 30)
    p_ax = np.array([ax_x, ax_y], float)

    # 3) Subclavian inflection — turns sharply medial toward SVC entry. This is the
    # CHARACTERISTIC PICC bend at the angle between arm and chest.
    sc_x = W // 2 + side * rng.randint(int(W * 0.05), int(W * 0.12))
    sc_y = ax_y + rng.randint(15, 30)
    p_sc = np.array([sc_x, sc_y], float)

    # 4) SVC descending to cavoatrial junction
    tip = cavoatrial_junction_point(anat_masks, rng) or (W // 2, H // 2)
    p_tip = np.array(tip, float)

    # Build path as 3 Bezier segments joined at vein landmarks
    # arm → axillary: gentle medial-and-down curve
    c1 = p_arm + np.array([(p_ax[0] - p_arm[0]) * 0.4, rng.uniform(5, 18)], float)
    c2 = p_ax + np.array([rng.uniform(-8, 8), -rng.uniform(2, 10)], float)
    seg1 = bezier_points(p_arm, c1, c2, p_ax, n=32)
    # axillary → subclavian: sharper medial sweep (characteristic kink)
    c3 = p_ax + np.array([(p_sc[0] - p_ax[0]) * 0.6, rng.uniform(2, 10)], float)
    c4 = p_sc + np.array([rng.uniform(-6, 6), -rng.uniform(2, 8)], float)
    seg2 = bezier_points(p_ax, c3, c4, p_sc, n=24)
    # subclavian → cavoatrial junction (close to vertical)
    c5 = p_sc + np.array([rng.uniform(-8, 8), rng.uniform(15, 30)], float)
    c6 = p_tip + np.array([rng.uniform(-4, 4), -rng.uniform(10, 20)], float)
    seg3 = bezier_points(p_sc, c5, c6, p_tip, n=64)

    pts = np.concatenate([seg1, seg2, seg3], axis=0).astype(np.int32)
    pts = jitter_path(pts, rng, amp_px=0.4)
    pts = extend_tube_externally(
        pts, H, W, rng,
        direction='lateral_left' if side < 0 else 'lateral_right',
        length_px=W // 8,
    )
    # PICC body should be VISIBLE but faint — real PICCs do show a thin radiopaque
    # line, not invisible. Bumped wall_color to 175-210 and added a small bright
    # bead at the tip (in addition to tip_radius), since real PICC tips are the most
    # consistently visible part.
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=1, marker='tip_only',
        wall_color=rng.randint(175, 210),
        tip_radius=rng.randint(4, 6), tip_color=rng.randint(245, 255),
    )
    return mask, color_map, _bezier_kpts(pts)


def _mask_centroid(anat_masks, name):
    m = union_mask(anat_masks, group_ids([name]))
    if not m.any():
        return None
    ys, xs = np.where(m)
    return np.array([float(xs.mean()), float(ys.mean())])


def _smooth_through(wps, n, rng):
    """Smooth spline THROUGH ordered waypoints (passes through each; s=0)."""
    keep = [0]
    for i in range(1, len(wps)):
        if np.hypot(*(wps[i] - wps[keep[-1]])) > 1.0:
            keep.append(i)
    wps = wps[keep]
    if len(wps) < 3:
        return wps
    from scipy.interpolate import splev, splprep
    try:
        tck, _ = splprep([wps[:, 0], wps[:, 1]], k=min(3, len(wps) - 1), s=0)
        uu = np.linspace(0.0, 1.0, n)
        xs, ys = splev(uu, tck)
        return np.stack([xs, ys], axis=1)
    except Exception:
        return wps


def _swan_anatomical_path(entry, tip, anat_masks, H, W, rng):
    """Real Swan course SVC->RA->RV->PA: dips inferiorly into the right ventricle, then
    curves back up to the pulmonary artery -- the characteristic loop generic winding misses."""
    ra = _mask_centroid(anat_masks, 'heart atrium right')
    rv = _mask_centroid(anat_masks, 'heart ventricle right')
    wps = [np.array(entry, float)]
    if ra is not None:
        wps.append(ra + np.array([rng.uniform(-12, 12), rng.uniform(-8, 8)]))
    if rv is not None:
        wps.append(rv + np.array([rng.uniform(-12, 12), rng.uniform(0.06 * H, 0.14 * H)]))  # dip to apex
    wps.append(np.array(tip, float))
    return _smooth_through(np.array(wps, float), 128, rng)


def shape_swan_ganz_v2(H, W, anat_masks, rng):
    """Swan-Ganz: dashed radio-opaque markers every ~10cm + balloon ellipse at tip."""
    side = rng.choice([-1, +1])
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    if clav.any():
        ys, xs = np.where(clav)
        med = np.median(xs)
        sel = (xs > med) if side > 0 else (xs < med)
        if sel.any():
            i = rng.randrange(int(sel.sum()))
            entry = (int(xs[sel][i]), int(ys[sel][i]))
        else:
            entry = (W // 2 + side * W // 4, H // 4)
    else:
        entry = (W // 2 + side * W // 4, H // 4)
    pa = union_mask(anat_masks, group_ids(['pulmonary artery']))
    if not pa.any():
        pa = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    tip = sample_point_in_mask(pa, rng) or (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    mid_off = np.array([rng.randint(-80, 80), rng.randint(60, 140)], float)
    p1 = p0 + (p3 - p0) * 0.25 + mid_off
    p2 = p0 + (p3 - p0) * 0.55 + np.array([rng.randint(-40, 40), rng.randint(-80, -20)], float)
    if os.environ.get('SYNTHFB_SWAN_PATH') == 'anatomical':
        pts = _swan_anatomical_path(entry, tip, anat_masks, H, W, rng)   # SVC->RA->RV->PA loop
    else:
        pts = winding_path(p0, p3, rng, n=128, n_waves=rng.uniform(*WIND_SWAN_NW), amp_frac=rng.uniform(*WIND_SWAN_AF))  # real Swan winds RA->RV->PA (real tortuosity ~2.4)
    pts = extend_tube_externally(
        pts, H, W, rng,
        direction='top_lateral_left' if side < 0 else 'top_lateral_right',
        length_px=H // 5,
    )
    # Appearance env-tunable (device-crop fb-TSTR sweep): real Swan is a FAINT THIN smooth
    # catheter, not the default bright bold dashed line. SYNTHFB_SWAN_WALL/MARKER='lo,hi',
    # SYNTHFB_SWAN_HALFW=int, SYNTHFB_SWAN_BALLOON=scale (0 = no balloon).
    _wall = _wind_range('SYNTHFB_SWAN_WALL', 195, 225)
    _mark = _wind_range('SYNTHFB_SWAN_MARKER', 248, 255)
    _hw = int(os.environ.get('SYNTHFB_SWAN_HALFW', '2'))
    _bsc = float(os.environ.get('SYNTHFB_SWAN_BALLOON', '1.0'))
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=_hw, marker=os.environ.get('SYNTHFB_SWAN_MARKER_STYLE', 'dashed'),
        wall_color=rng.randint(int(_wall[0]), int(_wall[1])),
        marker_color=rng.randint(int(_mark[0]), int(_mark[1])),
        tip_radius=0,
    )
    if _bsc > 0:
        _bshape = os.environ.get('SYNTHFB_SWAN_BALLOON_SHAPE', 'ellipse')
        ti = (int(tip[0]), int(tip[1]))
        bc = rng.randint(238, 255)
        if _bshape == 'sausage':  # elongated ~1cm balloon oriented ALONG the catheter tangent
            d = pts[-1] - pts[-4] if len(pts) >= 4 else np.array([1.0, 0.0])
            ang = float(np.degrees(np.arctan2(d[1], d[0])))
            br = (max(2, int(rng.randint(9, 14) * _bsc)), max(1, int(rng.randint(3, 5) * _bsc)))
            cv2.ellipse(mask, ti, br, ang, 0, 360, 255, -1)
            cv2.ellipse(color_map, ti, br, ang, 0, 360, bc, -1)
        elif _bshape == 'ring':   # faint outlined balloon (catheter visible through it)
            br = (max(2, int(rng.randint(6, 10) * _bsc)), max(2, int(rng.randint(5, 8) * _bsc)))
            ang = rng.uniform(0, 180)
            cv2.ellipse(mask, ti, br, ang, 0, 360, 255, 2)
            cv2.ellipse(color_map, ti, br, ang, 0, 360, bc, 2)
        else:                     # solid ellipse (default)
            br = (max(1, int(rng.randint(6, 10) * _bsc)), max(1, int(rng.randint(4, 7) * _bsc)))
            ang = rng.uniform(0, 180)
            cv2.ellipse(mask, ti, br, ang, 0, 360, 255, -1)
            cv2.ellipse(color_map, ti, br, ang, 0, 360, bc, -1)
            cv2.ellipse(mask, ti, br, ang, 0, 360, 255, 1, cv2.LINE_AA)
            cv2.ellipse(color_map, ti, br, ang, 0, 360, 255, 1, cv2.LINE_AA)
    return mask, color_map, _bezier_kpts(pts)


def shape_chest_tube_v2(H, W, anat_masks, rng):
    """Chest tube: thick paired walls w/ moderate fill + dashed radio-opaque stripe along center
    + Murphy eye gap near tip. Tube enters laterally through chest wall.
    """
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        side = rng.choice([-1, +1])
        ymid = int(np.median(ys))
        xedge = int(xs.max()) if side > 0 else int(xs.min())
        entry = (xedge + side * rng.randint(20, 40), ymid + rng.randint(-30, 30))
        end = sample_point_in_mask(lungs, rng) or (W // 2, H // 2)
    else:
        side = rng.choice([-1, +1])
        entry = (rng.randint(0, 30) if side < 0 else rng.randint(W - 30, W), H // 2)
        end = (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-20, 20), rng.randint(-20, 20)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-20, 20), rng.randint(-20, 20)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    pts = jitter_path(pts, rng, amp_px=0.7)
    pts = extend_tube_externally(
        pts, H, W, rng,
        direction='lateral_left' if side < 0 else 'lateral_right',
        length_px=W // 6,
    )
    # ~10% of chest tubes (apically pointing for pneumothorax) project off the
    # top edge of the image — rare but real.
    tip_oob = rng.random() < 0.10
    if tip_oob:
        pts = extend_tip_oob(pts, H, W, rng, direction='top', length_px=rng.randint(30, 70))
    # Chest tube is wider than ET/NG (drainage tube, ~7-8mm OD). Bump half to 7-10
    # and wall_color to brighter range so the tube is unambiguously visible.
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=rng.randint(7, 10),
        marker='dashed',
        wall_color=rng.randint(228, 252),
        fill_color=rng.randint(100, 155),
        marker_color=rng.randint(242, 255),
        murphy_eye=True,
    )
    # 2 additional side ports near tip (real chest tubes have 2-3 holes total)
    # rendered as small wall gaps. Use last in-frame pt when tip is OOB.
    if len(pts) >= 15:
        last_in = len(pts) - 1
        while last_in > 0 and not (0 <= pts[last_in][0] < W and 0 <= pts[last_in][1] < H):
            last_in -= 1
        for back in (5, 9):
            i = max(2, last_in - back)
            cv2.circle(mask,      (int(pts[i][0]), int(pts[i][1])), 2, 0, -1)
            cv2.circle(color_map, (int(pts[i][0]), int(pts[i][1])), 2, 0, -1)
    return mask, color_map, _bezier_kpts(pts, prox_name='skin_entry')


# ===== v2 retrofits for remaining tubes/lines (uses lumen_tube_mask) =====

def device_double_lumen_et_v2(H, W, anat_masks, rng):
    """Double-lumen ET: long shaft w/ dashed centerline + 2 cuffs (proximal main, distal bronchial)."""
    parts = []
    trachea = union_mask(anat_masks, ANAT_GROUPS['trachea'])
    if trachea.any():
        ys, xs = np.where(trachea)
        top_idx = int(np.argmin(ys)); bot_idx = int(np.argmax(ys))
        entry = (int(xs[top_idx]) + rng.randint(-10, 10), int(ys[top_idx]) - rng.randint(30, 60))
        side = rng.choice([-1, +1])
        tip = (int(xs[bot_idx]) + side * rng.randint(15, 40), int(ys[bot_idx]) + rng.randint(10, 30))
    else:
        entry = (W // 2, 20); tip = (W // 2 + 20, H // 2)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-15, 15), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    pts = jitter_path(pts, rng, amp_px=0.6)
    pts = extend_tube_externally(pts, H, W, rng, direction='top',
                                  length_px=H // 4, lateral_drift=10)
    half = rng.randint(5, 9)
    shaft, shaft_color = lumen_tube_mask(
        pts, H, W, half_w=half, marker='dashed',
        wall_color=rng.randint(180, 215),
        fill_color=rng.randint(70, 120),
        marker_color=rng.randint(225, 250),
    )
    parts.append(('dl_et_shaft', shaft, shaft_color, _bezier_kpts(pts)))
    prox_idx = int(0.78 * len(pts))                  # shifted to account for prepended external segment
    prox_pt = pts[prox_idx]
    cuff_prox = np.zeros((H, W), np.uint8)
    cuff_prox_color = np.zeros((H, W), np.uint8)
    render_3d_cuff(cuff_prox, cuff_prox_color, int(prox_pt[0]), int(prox_pt[1]),
                   half + rng.randint(3, 5), half + rng.randint(2, 3),
                   rim_intensity=rng.randint(220, 245),
                   interior_intensity=rng.randint(120, 160))
    parts.append(('dl_et_proximal_cuff', cuff_prox, cuff_prox_color,
                  {'center': (int(prox_pt[0]), int(prox_pt[1]), 2)}))
    cuff_dist = np.zeros((H, W), np.uint8)
    cuff_dist_color = np.zeros((H, W), np.uint8)
    render_3d_cuff(cuff_dist, cuff_dist_color, int(tip[0]), int(tip[1]),
                   half + rng.randint(2, 4), half + rng.randint(2, 3),
                   rim_intensity=rng.randint(220, 245),
                   interior_intensity=rng.randint(120, 160))
    parts.append(('dl_et_distal_cuff', cuff_dist, cuff_dist_color,
                  {'center': (int(tip[0]), int(tip[1]), 2)}))
    return parts


def device_ecmo_v2(H, W, anat_masks, rng):
    """ECMO cannulas: 2 thick paired-wall tubes from upper frame into great vessels/heart."""
    vess = union_mask(anat_masks, ANAT_GROUPS['vessels'])
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    target = vess if vess.any() else heart
    parts = []
    for kind, side_choice in (('ecmo_return', -1), ('ecmo_drain', +1)):
        entry = (W // 2 + side_choice * rng.randint(20, 60), rng.randint(0, 30))
        tip = sample_point_in_mask(target, rng) if target.any() else (W // 2, H // 2)
        p0 = np.array(entry, float); p3 = np.array(tip, float)
        p1 = p0 + (p3 - p0) * 0.3 + np.array([rng.randint(-20, 20), 0], float)
        p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-20, 20), 0], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        pts = jitter_path(pts, rng, amp_px=0.8)
        pts = extend_tube_externally(
            pts, H, W, rng,
            direction='top_lateral_left' if side_choice < 0 else 'top_lateral_right',
            length_px=H // 5,
        )
        mask, color_map = lumen_tube_mask(
            pts, H, W, half_w=rng.randint(9, 13),                # thicker (real ECMO 15-29 Fr)
            marker='solid',
            wall_color=rng.randint(215, 245),                    # brighter walls
            fill_color=rng.randint(120, 170),
        )
        # SPIRAL REINFORCEMENT: real ECMO cannulas have anti-kink spiral wire wrap.
        # On X-ray this shows as a regular series of bright bands crossing the tube.
        # Approximate as dashes perpendicular to tangent at fixed intervals.
        step = 6
        for i in range(step, len(pts) - 2, step):
            tan = pts[i] - pts[max(0, i - 2)]
            tan = tan.astype(float) / (np.linalg.norm(tan) + 1e-6)
            norm = np.array([-tan[1], tan[0]])
            half = 11
            x1 = int(pts[i][0] + norm[0] * half); y1 = int(pts[i][1] + norm[1] * half)
            x2 = int(pts[i][0] - norm[0] * half); y2 = int(pts[i][1] - norm[1] * half)
            cv2.line(mask,      (x1, y1), (x2, y2), 255, 1, cv2.LINE_AA)
            cv2.line(color_map, (x1, y1), (x2, y2), rng.randint(245, 255), 1, cv2.LINE_AA)
        parts.append((kind, mask, color_map, _bezier_kpts(pts)))
    return parts


def device_lvad_v2(H, W, anat_masks, rng):
    """LVAD: textured body @ LV + outflow cannula (paired walls) → aorta + driveline exit (thin walls)."""
    parts = []
    lv = union_mask(anat_masks, group_ids(['heart ventricle left']))
    if not lv.any():
        lv = union_mask(anat_masks, ANAT_GROUPS['heart'])
    body_pt = sample_point_in_mask(lv, rng) or (W // 2, H // 2)
    cx, cy = body_pt
    bw, bh = rng.randint(65, 85), rng.randint(65, 90)
    angle = rng.uniform(-20, 20)
    body, body_color = _textured_body(H, W, cx, cy, bw, bh, rng, angle=angle)
    parts.append(('lvad_body', body, body_color, {'center': (cx, cy, 2)}))
    aorta = union_mask(anat_masks, group_ids(['ascending aorta', 'aortic arch']))
    if not aorta.any():
        aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    aorta_pt = sample_point_in_mask(aorta, rng) if aorta.any() else (cx, max(0, cy - 100))
    a0 = np.array([cx + rng.randint(-10, 10), cy - bh // 2], float)
    a1 = np.array(aorta_pt, float)
    p1 = a0 + (a1 - a0) * 0.5 + np.array([rng.randint(-10, 10), 0], float)
    p2 = a0 + (a1 - a0) * 0.8
    pts = bezier_points(a0, p1, p2, a1, n=64)
    outflow, outflow_color = lumen_tube_mask(
        pts, H, W, half_w=rng.randint(5, 8),
        marker='paired_walls',
        wall_color=rng.randint(200, 235),
        fill_color=rng.randint(110, 150),
    )
    parts.append(('lvad_outflow', outflow, outflow_color, _bezier_kpts(pts)))
    side = rng.choice([-1, +1])
    d0 = np.array([cx + side * (bw // 2 - 2), cy + bh // 4], float)
    d3 = np.array([int(np.clip(cx + side * (W // 4 + rng.randint(20, 60)), 0, W - 1)),
                   min(H - 5, cy + rng.randint(40, 120))], float)
    dp1 = d0 + (d3 - d0) * 0.4 + np.array([rng.randint(-20, 20), rng.randint(-10, 20)], float)
    dp2 = d0 + (d3 - d0) * 0.7 + np.array([rng.randint(-20, 20), rng.randint(-10, 20)], float)
    dpts = bezier_points(d0, dp1, dp2, d3, n=64)
    drv, drv_color = lumen_tube_mask(
        dpts, H, W, half_w=2, marker='paired_walls',
        wall_color=rng.randint(190, 225),
    )
    parts.append(('lvad_driveline', drv, drv_color, _bezier_kpts(dpts)))
    return parts


def device_sengstaken_v2(H, W, anat_masks, rng):
    """Sengstaken-Blakemore: thin marker-stripe tube + esophageal balloon + gastric balloon."""
    parts = []
    entry = (W // 2 + rng.randint(-15, 15), rng.randint(0, 30))
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, _ = np.where(lungs)
        end_y = min(H - 10, int(ys.max()) + rng.randint(30, 70))
    else:
        end_y = H - 30
    end = (W // 2 + rng.randint(-15, 15), end_y)
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-10, 10), 0], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-10, 10), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    pts = jitter_path(pts, rng, amp_px=0.7)
    pts = extend_tube_externally(pts, H, W, rng, direction='top_nostril',
                                  length_px=H // 5)
    tube_mask, tube_color = lumen_tube_mask(
        pts, H, W, half_w=1, marker='continuous_stripe',
        wall_color=rng.randint(160, 195), marker_color=rng.randint(225, 250),
    )
    parts.append(('sb_tube', tube_mask, tube_color, _bezier_kpts(pts)))
    eso_pt = pts[int(0.7 * len(pts))]                 # shifted past prepended external
    eso_bal = np.zeros((H, W), np.uint8)
    eso_color = np.zeros((H, W), np.uint8)
    render_3d_cuff(eso_bal, eso_color, int(eso_pt[0]), int(eso_pt[1]),
                   rng.randint(6, 10), rng.randint(12, 20),
                   rim_intensity=rng.randint(190, 220), interior_intensity=rng.randint(110, 150))
    parts.append(('sb_esophageal_balloon', eso_bal, eso_color,
                  {'center': (int(eso_pt[0]), int(eso_pt[1]), 2)}))
    gas_bal = np.zeros((H, W), np.uint8)
    gas_color = np.zeros((H, W), np.uint8)
    render_3d_cuff(gas_bal, gas_color, int(end[0]), int(end[1]),
                   rng.randint(10, 18), rng.randint(12, 22),
                   rim_intensity=rng.randint(200, 230), interior_intensity=rng.randint(120, 160))
    parts.append(('sb_gastric_balloon', gas_bal, gas_color,
                  {'center': (int(end[0]), int(end[1]), 2)}))
    return parts


def shape_pleurx_v2(H, W, anat_masks, rng):
    """PleurX tunneled pleural catheter. Distinguishing features (vs chest tube):
       * EXTERNAL CAP / valve mechanism at skin exit (small radio-opaque disc)
       * SUBCUTANEOUS TUNNEL segment running laterally before pleural entry
       * Smaller body than chest tube (PleurX is 15-16 Fr ~5mm)
    """
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        side = rng.choice([-1, +1])
        ymid = int(np.median(ys))
        xedge = int(xs.max()) if side > 0 else int(xs.min())
        # external skin exit — lateral chest wall, with cap visible
        cap_x = xedge + side * rng.randint(50, 80)
        cap_y = ymid + rng.randint(-10, 25)
        # subQ tunnel runs back toward thorax (along chest wall under skin)
        tunnel_end_x = xedge + side * rng.randint(15, 30)
        tunnel_end_y = cap_y - rng.randint(5, 15)
        # pleural-cavity tip
        pleural_tip = sample_point_in_mask(lungs, rng) or (W // 2, H // 2)
    else:
        side = rng.choice([-1, +1])
        cap_x = rng.randint(0, 30) if side < 0 else rng.randint(W - 30, W)
        cap_y = H // 2
        tunnel_end_x = cap_x - side * 30
        tunnel_end_y = cap_y
        pleural_tip = (W // 2, H // 2)

    p_cap = np.array([cap_x, cap_y], float)
    p_tunnel_end = np.array([tunnel_end_x, tunnel_end_y], float)
    p_tip = np.array(pleural_tip, float)
    # subQ tunnel — short Bezier
    seg1 = bezier_points(
        p_cap,
        p_cap + np.array([-(cap_x - tunnel_end_x) * 0.4, rng.uniform(-5, 5)], float),
        p_tunnel_end + np.array([rng.uniform(-3, 3), rng.uniform(-3, 3)], float),
        p_tunnel_end, n=24)
    # pleural segment
    seg2 = bezier_points(
        p_tunnel_end,
        p_tunnel_end + np.array([(p_tip[0] - p_tunnel_end[0]) * 0.4 + rng.uniform(-10, 10),
                                  rng.uniform(-15, 15)], float),
        p_tip + np.array([rng.uniform(-10, 10), -rng.uniform(5, 15)], float),
        p_tip, n=64)
    pts = np.concatenate([seg1, seg2], axis=0).astype(np.int32)
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=rng.randint(2, 3), marker='continuous_stripe',
        wall_color=rng.randint(190, 220), marker_color=rng.randint(238, 255),
        tip_radius=rng.randint(2, 3), tip_color=rng.randint(240, 255),
    )
    # external cap — small bright disc at the skin exit
    cv2.circle(mask,      (int(cap_x), int(cap_y)), rng.randint(4, 7), 255, -1, cv2.LINE_AA)
    cv2.circle(color_map, (int(cap_x), int(cap_y)), rng.randint(4, 7),
               rng.randint(235, 255), -1, cv2.LINE_AA)
    return mask, color_map, _bezier_kpts(pts)


def shape_mediastinal_drain_v2(H, W, anat_masks, rng):
    """Mediastinal drain: thicker tube curving from anterior chest wall (xiphoid area)
    into mediastinum. Now with curved-not-straight path + multiple side ports near tip."""
    lungs = union_mask(anat_masks, ANAT_GROUPS['lungs'])
    if lungs.any():
        ys, xs = np.where(lungs)
        cx = int(np.median(xs)) + rng.randint(-15, 15)
        # entry from below xiphoid (not from upper mediastinum); end inside mediastinum
        y0 = min(H - 5, int(ys.max()) + rng.randint(20, 60))    # entry at lower edge
        y1 = int(ys.min()) + rng.randint(30, 90)                # tip in upper mediastinum
    else:
        cx, y0, y1 = W // 2, 3 * H // 4, H // 3
    # path is bottom-to-top with lateral curve
    p0 = np.array([cx + rng.randint(-5, 15), y0], float)         # entry slightly off midline
    p3 = np.array([cx + rng.randint(-12, 12), y1], float)
    p1 = p0 + (p3 - p0) * 0.35 + np.array([rng.randint(-25, 25), 0], float)
    p2 = p0 + (p3 - p0) * 0.65 + np.array([rng.randint(-15, 15), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    pts = jitter_path(pts, rng, amp_px=0.5)
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=rng.randint(4, 7),                     # thicker
        marker='dashed',
        wall_color=rng.randint(210, 240),
        fill_color=rng.randint(95, 140),
        marker_color=rng.randint(235, 252),
        murphy_eye=True,
    )
    # 2 extra side ports near tip
    if len(pts) >= 12:
        for back in (5, 9):
            i = max(2, len(pts) - back)
            cv2.circle(mask,      (int(pts[i][0]), int(pts[i][1])), 2, 0, -1)
            cv2.circle(color_map, (int(pts[i][0]), int(pts[i][1])), 2, 0, -1)
    return mask, color_map, _bezier_kpts(pts)


def shape_pericardial_drain_v2(H, W, anat_masks, rng):
    """Pigtail pericardial drain: paired walls + curl at tip."""
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    side = rng.choice([-1, +1])
    entry = (int(np.clip(W // 2 + side * rng.randint(80, 130), 0, W - 1)),
             rng.randint(H // 2, 3 * H // 4))
    tip = sample_point_in_mask(heart, rng) or (W // 2, H // 2)
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-20, 20), rng.randint(-15, 15)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-15, 15), rng.randint(-15, 15)], float)
    pts = bezier_points(p0, p1, p2, p3, n=96)
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=2, marker='paired_walls',
        wall_color=rng.randint(195, 230),
    )
    # pigtail curl
    curl_r = rng.randint(5, 9)
    cx_t, cy_t = int(tip[0]), int(tip[1])
    angles = np.linspace(0, 1.5 * np.pi, 32)
    cx_off = cx_t - curl_r
    pts_curl = np.stack([cx_off + curl_r * np.cos(angles),
                         cy_t + curl_r * np.sin(angles)], axis=1).astype(np.int32)
    cv2.polylines(mask,      [pts_curl], False, 255, 1, cv2.LINE_AA)
    cv2.polylines(color_map, [pts_curl], False, rng.randint(220, 245), 1, cv2.LINE_AA)
    return mask, color_map, _bezier_kpts(pts)


def shape_pacing_wire_v2(H, W, anat_masks, rng):
    """Pacing wire: thin faint body + bright distal electrode tip."""
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    end = sample_point_in_mask(heart, rng) or (W // 2, H // 2)
    side = rng.choice([-1, +1])
    entry = (int(np.clip(W // 2 + side * rng.randint(60, 120), 0, W - 1)), rng.randint(20, 80))
    p0 = np.array(entry, float); p3 = np.array(end, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-8, 8), rng.randint(-8, 8)], float)
    p2 = p0 + (p3 - p0) * 0.7 + np.array([rng.randint(-8, 8), rng.randint(-8, 8)], float)
    pts = bezier_points(p0, p1, p2, p3, n=64)
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=1, marker='tip_only',
        wall_color=rng.randint(180, 215),
        tip_radius=rng.randint(2, 4), tip_color=rng.randint(240, 255),
    )
    return mask, color_map, _bezier_kpts(pts)


def shape_uac_uvc_v2(H, W, anat_masks, rng):
    """Umbilical catheters (UAC/UVC): 2 thin faint lines from inferior frame edge."""
    full_mask = np.zeros((H, W), np.uint8)
    full_color = np.zeros((H, W), np.uint8)
    entry_x = W // 2 + rng.randint(-15, 15)
    entry_y = H - rng.randint(5, 20)
    aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    heart = union_mask(anat_masks, ANAT_GROUPS['heart'])
    targets = [
        sample_point_in_mask(aorta, rng) if aorta.any() else (W // 2, H // 2),
        sample_point_in_mask(heart, rng) if heart.any() else (W // 2, H // 3),
    ]
    last_pts = None
    for tip in targets:
        p0 = np.array([entry_x + rng.randint(-5, 5), entry_y], float)
        p3 = np.array(tip, float)
        p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-10, 10), 0], float)
        p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-15, 15), 0], float)
        pts = bezier_points(p0, p1, p2, p3, n=96)
        m, cm = lumen_tube_mask(
            pts, H, W, half_w=1, marker='tip_only',
            wall_color=rng.randint(170, 210),
            tip_radius=rng.randint(2, 3), tip_color=rng.randint(235, 252),
        )
        full_mask = np.maximum(full_mask, m)
        full_color = np.maximum(full_color, cm)
        last_pts = pts
    return full_mask, full_color, _bezier_kpts(last_pts) if last_pts is not None else {}


def shape_iabp_v2(H, W, anat_masks, rng):
    """IABP: thin catheter through descending aorta + elongated balloon segment at arch level."""
    aorta = union_mask(anat_masks, ANAT_GROUPS['aorta'])
    if not aorta.any():
        return np.zeros((H, W), np.uint8), np.zeros((H, W), np.uint8), {}
    ys, xs = np.where(aorta)
    y0, y1 = int(ys.min()), int(ys.max())
    cx = int(np.median(xs))
    entry = (cx + rng.randint(-15, 15), min(H - 1, y1 + rng.randint(20, 60)))
    tip = (cx + rng.randint(-15, 15), max(0, y0 + rng.randint(0, 15)))
    p0 = np.array(entry, float); p3 = np.array(tip, float)
    p1 = p0 + (p3 - p0) * 0.33 + np.array([rng.randint(-8, 8), 0], float)
    p2 = p0 + (p3 - p0) * 0.66 + np.array([rng.randint(-8, 8), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=1, marker='continuous_stripe',
        wall_color=rng.randint(170, 205), marker_color=rng.randint(220, 245),
    )
    # balloon segment along middle 35-70% of catheter
    mid_start = int(0.35 * len(pts)); mid_end = int(0.7 * len(pts))
    bal_pts = pts[mid_start:mid_end]
    if len(bal_pts) >= 2:
        bal_mask, bal_color = lumen_tube_mask(
            bal_pts, H, W, half_w=rng.randint(4, 7),
            marker='solid',
            wall_color=rng.randint(195, 230),
            fill_color=rng.randint(130, 175),
        )
        mask = np.maximum(mask, bal_mask)
        color_map = np.maximum(color_map, bal_color)
    return mask, color_map, _bezier_kpts(pts)


def shape_impella_v2(H, W, anat_masks, rng):
    """Impella micropump: thin dashed catheter through aorta + bright pump capsule + inlet ring."""
    lv = union_mask(anat_masks, group_ids(['heart ventricle left']))
    if lv.any():
        lv_pt = sample_point_in_mask(lv, rng)
    else:
        lv_pt = (W // 2, H // 2)
    side = rng.choice([-1, +1])
    entry = (int(np.clip(W // 2 + side * rng.randint(20, 50), 0, W - 1)), H - rng.randint(10, 40))
    p0 = np.array(entry, float); p3 = np.array(lv_pt, float)
    p1 = p0 + (p3 - p0) * 0.4 + np.array([rng.randint(-15, 15), 0], float)
    p2 = p0 + (p3 - p0) * 0.75 + np.array([rng.randint(-10, 10), 0], float)
    pts = bezier_points(p0, p1, p2, p3, n=128)
    mask, color_map = lumen_tube_mask(
        pts, H, W, half_w=1, marker='dashed',
        wall_color=rng.randint(180, 215), marker_color=rng.randint(225, 250),
    )
    pump_pt = pts[int(0.92 * len(pts))]
    cv2.circle(mask,      tuple(pump_pt.tolist()), rng.randint(3, 5), 255, -1)
    cv2.circle(color_map, tuple(pump_pt.tolist()), rng.randint(3, 5), rng.randint(235, 252), -1)
    inlet_pt = pts[int(0.85 * len(pts))]
    cv2.circle(mask,      tuple(inlet_pt.tolist()), 2, 255, 1)
    cv2.circle(color_map, tuple(inlet_pt.tolist()), 2, rng.randint(225, 245), 1)
    return mask, color_map, _bezier_kpts(pts)


# ===== Hybrid generation: book-cutout body + procedural lead =====
# The visually-hardest part of pacemaker/ICD/CRT bodies (rounded squares, battery
# cells, header connectors, model markings) is replaced with a real cutout from
# the book manifest when one is available. The leads / catheters are still drawn
# procedurally because their course is highly anatomy-constrained and benefits
# from the existing routing logic.

def _hybrid_cutouts_for(synthfb_cat):
    """Return list of cutout paths whose manifest synthfb_category matches.
    Prefers ECCV24 cutouts when available — they're hand-cleaned vs the
    polygon-derived book cutouts whose masks are sometimes imprecise.
    Falls back to book pool only when ECCV is empty for this class.
    Empty list if the manifest is empty or no matches exist.
    """
    eccv_paths, book_paths = [], []
    for path, meta in CUTOUT_META.items():
        if meta.get('synthfb_category') != synthfb_cat:
            continue
        if meta.get('source') == 'eccv24':
            eccv_paths.append(path)
        else:
            book_paths.append(path)
    return eccv_paths if eccv_paths else book_paths


def _paste_book_body_cutout(H, W, cx, cy, candidate_paths, rng,
                             max_dim_frac=0.18):
    """Pick one cutout from candidates, weak-augment it, and place it centered
    on (cx, cy). Returns (mask, color_map, source_meta) where mask is uint8
    H,W (255 inside body, 0 outside) and color_map is the cutout's intensity
    pasted at the same location.
    Returns None if no candidates or paste would be degenerate.
    """
    if not candidate_paths:
        return None
    src_path = rng.choice(candidate_paths)
    rgba = load_rgba(src_path)
    rgba = weak_augment_rgba(rgba, rng)
    ch, cw = rgba.shape[:2]
    # Scale so the cutout's longer side fits in max_dim_frac of image dim
    target_max = int(min(H, W) * max_dim_frac)
    cur_max = max(ch, cw)
    if cur_max > target_max:
        s = target_max / cur_max
        rgba = cv2.resize(rgba, (max(8, int(cw * s)), max(8, int(ch * s))))
        ch, cw = rgba.shape[:2]
    if ch < 8 or cw < 8:
        return None
    x0 = int(np.clip(cx - cw // 2, 0, W - cw))
    y0 = int(np.clip(cy - ch // 2, 0, H - ch))
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    insert = alpha > 10
    if not insert.any():
        return None
    gray = rgb.mean(axis=2).astype(np.uint8)
    mask = np.zeros((H, W), np.uint8)
    color_map = np.zeros((H, W), np.uint8)
    region_mask = mask[y0:y0 + ch, x0:x0 + cw]
    region_color = color_map[y0:y0 + ch, x0:x0 + cw]
    region_mask[insert] = 255
    region_color[insert] = gray[insert]
    return mask, color_map, dict(CUTOUT_META.get(str(Path(src_path).resolve()), {}))


def _lead_exit_from_body_mask(body_mask):
    """Pick a sensible lead-exit point on a pacemaker/ICD body mask. Real
    pacemakers have the header (where the lead screws in) at the inferior
    border. Returns (x, y) at the lower-center of the body bbox, snapped to
    an in-mask pixel."""
    ys, xs = np.where(body_mask > 0)
    if len(ys) == 0:
        return None
    y_max = int(ys.max())
    # Among bottom-row pixels of the mask, take the median x — robust to
    # outliers (jagged cutout edges).
    cols_at_bot = xs[ys >= y_max - 2]
    x_med = int(np.median(cols_at_bot))
    return (x_med, y_max)


def device_pacemaker_hybrid(H, W, anat_masks, rng):
    """Hybrid pacemaker: book cutout body + procedural lead(s).
    Falls back to device_pacemaker_v2 when no body cutout is in the manifest.
    """
    candidates = _hybrid_cutouts_for('pacemaker_body')
    if not candidates:
        return device_pacemaker_v2(H, W, anat_masks, rng)
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('pacemaker', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = 3 * W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(20, 60)
    cx = int(np.clip(cx, 40, W - 40)); cy = int(np.clip(cy, 40, H - 40))

    paste = _paste_book_body_cutout(H, W, cx, cy, candidates, rng)
    if paste is None:
        return device_pacemaker_v2(H, W, anat_masks, rng)
    body_mask, body_color, body_meta = paste
    body_center = _inside_centroid(body_mask)
    exit_pt = _lead_exit_from_body_mask(body_mask)
    if body_center is None or exit_pt is None:
        return device_pacemaker_v2(H, W, anat_masks, rng)
    # Body part — emit with provenance meta so the cutout's book caption / page
    # travels through to the COCO output, same as cutpaste pastes do.
    parts = [('pacemaker_body', body_mask, body_color,
              {'center': body_center}, body_meta)]
    # Procedural leads exiting from the body's bottom-medial corner
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    n_leads = rng.randint(1, 2)
    for _ in range(n_leads):
        # Reuse _build_pacing_lead with bh=0 so p_gen = (exit_x, exit_y)
        parts.append(_build_pacing_lead(H, W, exit_pt[0], exit_pt[1], 0,
                                         heart_u, rng, lead_cat='pacemaker_lead'))
    return parts


def device_icd_hybrid(H, W, anat_masks, rng):
    """Hybrid ICD: book cutout body + procedural lead with shock-coil thickening
    near the tip. Falls back to device_icd_v2 if no icd_body cutouts available.
    """
    candidates = _hybrid_cutouts_for('icd_body')
    if not candidates:
        return device_icd_v2(H, W, anat_masks, rng)
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('icd', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = 3 * W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(20, 60)
    cx = int(np.clip(cx, 40, W - 40)); cy = int(np.clip(cy, 40, H - 40))
    # ICDs are bigger than pacemakers — allow slightly larger paste
    paste = _paste_book_body_cutout(H, W, cx, cy, candidates, rng, max_dim_frac=0.22)
    if paste is None:
        return device_icd_v2(H, W, anat_masks, rng)
    body_mask, body_color, body_meta = paste
    body_center = _inside_centroid(body_mask)
    exit_pt = _lead_exit_from_body_mask(body_mask)
    if body_center is None or exit_pt is None:
        return device_icd_v2(H, W, anat_masks, rng)
    parts = [('icd_body', body_mask, body_color,
              {'center': body_center}, body_meta)]
    heart_u = union_mask(anat_masks, ANAT_GROUPS['heart'])
    # Reuse pacing-lead builder, then thicken last 30% to add the shock-coil
    _, lead_mask, lead_color, lead_kpts = _build_pacing_lead(
        H, W, exit_pt[0], exit_pt[1], 0, heart_u, rng, lead_cat='icd_lead')
    kp = lead_kpts
    tx, ty = kp['tip'][0], kp['tip'][1]
    mx, my = kp['mid_b'][0], kp['mid_b'][1]
    cv2.line(lead_mask,  (mx, my), (tx, ty), 255, 3, cv2.LINE_AA)
    cv2.line(lead_color, (mx, my), (tx, ty), rng.randint(195, 225), 3, cv2.LINE_AA)
    parts.append(('icd_lead', lead_mask, lead_color, lead_kpts))
    return parts


def device_port_catheter_hybrid(H, W, anat_masks, rng):
    """Hybrid port-a-cath: book cutout reservoir + procedural catheter to SVC.
    Falls back to device_port_catheter_v2 if no port_reservoir cutouts available.
    """
    # Most real CVCs (RANZCR "CVC") are non-port IJ/subclavian central lines: long,
    # fairly straight, entry high in the chest -> tip at the cavoatrial junction.
    # Generate that the majority of the time; the chest-wall port is the minority.
    # This is the CVC routing fix (catheter_tube length/diag ~0.20 -> ~0.47).
    if rng.random() < CVC_CENTRAL_P:
        return _cvc_central_line(H, W, anat_masks, rng)
    candidates = _hybrid_cutouts_for('port_reservoir')
    if not candidates:
        return device_port_catheter_v2(H, W, anat_masks, rng)
    clav = union_mask(anat_masks, ANAT_GROUPS['clavicle'])
    side_bias = sample_side_bias('port_catheter', rng)
    pt = pick_anchor_with_side(clav, rng, side_bias)
    if pt is None:
        cx, cy = W // 4, H // 4
    else:
        cx, cy = pt[0], pt[1] + rng.randint(30, 80)
    cx = int(np.clip(cx, 30, W - 30)); cy = int(np.clip(cy, 30, H - 30))
    # Ports are smaller than pacemakers
    paste = _paste_book_body_cutout(H, W, cx, cy, candidates, rng, max_dim_frac=0.12)
    if paste is None:
        return device_port_catheter_v2(H, W, anat_masks, rng)
    res_mask, res_color, body_meta = paste
    res_center = _inside_centroid(res_mask)
    if res_center is None:
        return device_port_catheter_v2(H, W, anat_masks, rng)
    # Procedural catheter from reservoir-side to cavoatrial junction.
    # Port catheters exit medially, not bottom-center — re-derive exit by
    # taking the side of the reservoir facing midline.
    ys, xs = np.where(res_mask > 0)
    medial_x = xs.min() if res_center[0] > W // 2 else xs.max()
    y_at_medial = int(np.median(ys[xs == medial_x])) if (xs == medial_x).any() else res_center[1]
    exit_pt = (int(medial_x), int(y_at_medial))
    tip = cavoatrial_junction_point(anat_masks, rng) or (W // 2, H // 2)
    p0 = np.array(exit_pt, float); p3 = np.array(tip, float)
    pts = winding_path(p0, p3, rng, n=96, n_waves=rng.uniform(*WIND_CVC_NW), amp_frac=rng.uniform(*WIND_CVC_AF))  # CVCs curve into the SVC (real tortuosity ~1.2)
    tube_mask, tube_color = lumen_tube_mask(
        pts, H, W, half_w=2, marker='paired_walls',
        wall_color=rng.randint(190, 225),
        tip_radius=rng.randint(2, 3), tip_color=rng.randint(230, 250),
    )
    return [
        ('port_reservoir', res_mask, res_color, {'center': res_center}, body_meta),
        ('catheter_tube', tube_mask, tube_color, _bezier_kpts(pts)),
    ]


# install overrides
DEVICE_FNS.update({
    'pacemaker':         device_pacemaker_hybrid,
    'icd':               device_icd_hybrid,
    'crt_pacemaker':     device_crt_pacemaker_v2,
    'port_catheter':     device_port_catheter_hybrid,
    'tunneled_cvc':      device_tunneled_cvc_v2,
    'dialysis_catheter': device_dialysis_catheter_v2,
    'et_tube':           device_et_tube_v2,
    'tracheostomy':      device_tracheostomy_v2,
    'double_lumen_et':   device_double_lumen_et_v2,
    'ecmo':              device_ecmo_v2,
    'lvad':              device_lvad_v2,
    'sengstaken':        device_sengstaken_v2,
})
SHAPE_FNS.update({
    'ng_tube':            shape_ng_tube_v2,
    'picc_line':          shape_picc_line_v2,
    'swan_ganz':          shape_swan_ganz_v2,
    'chest_tube':         shape_chest_tube_v2,
    'pleurx':             shape_pleurx_v2,
    'mediastinal_drain':  shape_mediastinal_drain_v2,
    'pericardial_drain':  shape_pericardial_drain_v2,
    'pacing_wire':        shape_pacing_wire_v2,
    'uac_uvc':            shape_uac_uvc_v2,
    'iabp':               shape_iabp_v2,
    'impella':            shape_impella_v2,
})
print(f'installed v2 overrides: 12 devices + 11 shapes')


# ---- Archetype-driven generation ----
def generate_one_archetype(img, anat_masks, cutouts, rng,
                           max_annotations=MAX_ANNOTATIONS,
                           view='unknown',
                           p_archetype=0.85):
    out = img.copy()
    annots = []
    H, W = img.shape[:2]

    if view == 'lateral':
        avail_shapes  = set(SHAPE_FNS)  - FRONTAL_ONLY_SHAPES
        avail_devices = set(DEVICE_FNS) - FRONTAL_ONLY_DEVICES
    else:
        avail_shapes  = set(SHAPE_FNS)
        avail_devices = set(DEVICE_FNS)
    avail_shapes, avail_devices = _subset_rules(avail_shapes, avail_devices)

    def emit(kind, name):
        nonlocal out
        if kind == 'shape':
            if name not in avail_shapes:
                return
            for _ in range(sample_count(name, rng)):
                ret = SHAPE_FNS[name](H, W, _ablate_anat(anat_masks, rng), rng)
                if len(ret) == 3:
                    mask, color, kpts = ret
                else:
                    mask, color = ret
                    kpts = None
                if not mask.any():
                    continue
                mode = pick_physics_mode(name, rng)
                out, _ = render_with_physics(out, mask, color, anat_masks, rng, mode=mode)
                if name in SPLIT_INTO_INSTANCES:
                    for sub_cat, sub_mask, sub_kpts in _split_mask_instances(mask, SPLIT_INTO_INSTANCES[name]):
                        annots.append((sub_cat, sub_mask, mode, sub_kpts))
                elif name in MULTI_KPT_CLUSTERS:
                    annots.append((name, mask, mode, _cluster_kpts(mask, MULTI_KPT_CLUSTERS[name])))
                else:
                    annots.append((name, mask, mode, kpts))
        elif kind == 'device':
            if name not in avail_devices:
                return
            for _ in range(sample_count(name, rng)):
                # Per-device instance id so downstream consumers can recover
                # the parts→device grouping after MFB-family eval merges them.
                dev_inst_id = f'{name}_{uuid.uuid4().hex[:8]}'
                parts = DEVICE_FNS[name](H, W, _ablate_anat(anat_masks, rng), rng)
                for part in parts:
                    # parts can be 3-tuple, 4-tuple (+ kpts), or 5-tuple
                    # (+ source_meta for hybrid book-cutout bodies)
                    part_name = part[0]
                    part_mask = part[1]
                    color     = part[2]
                    kpts      = part[3] if len(part) >= 4 else None
                    src_meta  = part[4] if len(part) >= 5 else None
                    if not part_mask.any():
                        continue
                    # Attach instance-grouping metadata to every device part
                    meta = dict(src_meta) if src_meta else {}
                    meta.setdefault('device_instance_id', dev_inst_id)
                    meta.setdefault('device_family', name)
                    mode = pick_physics_mode(part_name, rng)
                    out, _ = render_with_physics(out, part_mask, color, anat_masks, rng, mode=mode)
                    if part_name in SPLIT_INTO_INSTANCES:
                        for sub_cat, sub_mask, sub_kpts in _split_mask_instances(part_mask, SPLIT_INTO_INSTANCES[part_name]):
                            annots.append((sub_cat, sub_mask, mode, sub_kpts, dict(meta)))
                    elif part_name in MULTI_KPT_CLUSTERS:
                        annots.append((part_name, part_mask, mode, _cluster_kpts(part_mask, MULTI_KPT_CLUSTERS[part_name]), meta))
                    else:
                        annots.append((part_name, part_mask, mode, kpts, meta))
        elif kind == 'cutpaste' and cutouts:
            cat = rng.choice(list(cutouts.keys()))
            src_path = rng.choice(cutouts[cat])
            rgba = load_rgba(src_path)
            blend = rng.choice(['poisson_normal', 'poisson_mixed', 'normal'])
            # Use anatomic anchor for classes with strong position priors
            meta = CUTOUT_META.get(str(Path(src_path).resolve()))
            anchor = CUTPASTE_ANATOMIC_ANCHORS.get(
                meta.get('synthfb_category') if meta else None)
            out, m = cut_paste(out, rgba, anat_masks, rng, blend=blend,
                               anchor_group=anchor)
            if m.any():
                # synthfb_category may be missing (no manifest entry) or None
                # (pool-only entries). In both cases fall back to 'cutpaste'.
                annot_cat = (meta.get('synthfb_category') if meta else None) or 'cutpaste'
                # Defensive: if the manifest names a class not in CATEGORIES
                # (e.g. drift between book_split.BOOK_TO_SYNTHFB and SynthFB's
                # registered categories), gracefully degrade to 'cutpaste'
                # instead of failing the run mid-generation.
                if annot_cat not in CAT2ID:
                    annot_cat = 'cutpaste'
                annots.append((annot_cat, m, blend, None, meta))

    # ISOLATED single-device mode (fb-TSTR "no co-occurrence" training data): place
    # exactly ONE device family on the clean base -- no archetype, no clutter -- so a
    # class's transfer AUC is attributable purely to that class's generation rules.
    # Env SYNTHFB_ISOLATE = a DEVICE_FNS or SHAPE_FNS key (et_tube|ng_tube|port_catheter|
    # swan_ganz|...). Default unset = original behaviour.
    _iso = os.environ.get('SYNTHFB_ISOLATE')
    if _iso:
        if _iso in ('clean', 'none'):                 # clean base, no device (no-line negative)
            return out, annots, 'isolate:clean'
        if _iso.startswith('cut:'):                    # paste ONE real device crop of a class
            want = _iso[4:]
            cand = [p for p, mm in CUTOUT_META.items()
                    if mm.get('synthfb_category') == want and os.path.exists(p)]
            if not cand:
                raise ValueError(f'no cutouts with synthfb_category={want}')
            src = cand[rng.randrange(len(cand))]
            blend = rng.choice(['poisson_normal', 'poisson_mixed', 'normal'])
            anchor = CUTPASTE_ANATOMIC_ANCHORS.get(want)
            out, m = cut_paste(out, load_rgba(src), anat_masks, rng, blend=blend, anchor_group=anchor)
            if m.any():
                annots.append((want if want in CAT2ID else 'cutpaste', m, blend, None, CUTOUT_META.get(src)))
            return out, annots, f'isolate:{_iso}'
        kind = 'device' if _iso in DEVICE_FNS else ('shape' if _iso in SHAPE_FNS else None)
        if kind is None:
            raise ValueError(f'SYNTHFB_ISOLATE={_iso} not in DEVICE_FNS/SHAPE_FNS')
        emit(kind, _iso)
        return out, annots, f'isolate:{_iso}'

    arch_name = None
    if rng.random() < p_archetype:
        arch_name = pick_archetype(rng)
        arch = ARCHETYPES[arch_name]
        for kind, cat in arch['required']:
            emit(kind, cat)
        for kind, cat in arch.get('optional', []):
            if rng.random() < 0.5:
                emit(kind, cat)
        n_extra = rng.randint(*arch.get('extra', (0, 0)))
    else:
        n_extra = rng.randint(1, max_annotations)

    # Uniform sample from full shape+device pool (removes paper-baseline bias).
    # Cutpaste fires occasionally (10%), rest split 50/50 shape vs device.
    avail_shapes_l = list(avail_shapes)
    avail_devices_l = list(avail_devices)
    for _ in range(n_extra):
        r = rng.random()
        if cutouts and r < 0.10:
            emit('cutpaste', None)
            continue
        if avail_shapes_l and (not avail_devices_l or rng.random() < 0.5):
            emit('shape', rng.choice(avail_shapes_l))
        elif avail_devices_l:
            emit('device', rng.choice(avail_devices_l))

    return out, annots, arch_name

# switch demo_img to a frontal sample if available (gallery readability)
_demo_view = classify_view(sample_paths[0])
if _demo_view != 'frontal':
    _frontal_paths = [p for p in sample_paths if classify_view(p) == 'frontal']
    if _frontal_paths:
        demo_img, demo_anat = get_anatomy(cxas_model, _frontal_paths[0])
        print(f'switched demo to frontal: {_frontal_paths[0].name}')
    else:
        print(f'no frontal in sample (keeping {_demo_view} demo)')
else:
    print(f'demo already frontal: {sample_paths[0].name}')

def generate_one(img, anat_masks, cutouts, rng,
                 max_annotations=MAX_ANNOTATIONS,
                 cutpaste_prob=CUTPASTE_PROB, device_prob=DEVICE_PROB,
                 view='unknown'):
    out = img.copy()
    annots = []  # list of (cat_name, mask, render_mode)
    H, W = img.shape[:2]

    # filter category pool by view
    if view == 'lateral':
        avail_shapes  = [k for k in SHAPE_FNS  if k not in FRONTAL_ONLY_SHAPES]
        avail_devices = [k for k in DEVICE_FNS if k not in FRONTAL_ONLY_DEVICES]
    else:
        avail_shapes  = list(SHAPE_FNS.keys())
        avail_devices = list(DEVICE_FNS.keys())
    avail_shapes, avail_devices = _subset_rules(avail_shapes, avail_devices)

    n = rng.randint(1, max_annotations)
    for _ in range(n):
        r = rng.random()
        if cutouts and r < cutpaste_prob:
            cat = rng.choice(list(cutouts.keys()))
            rgba = load_rgba(rng.choice(cutouts[cat]))
            blend = rng.choice(['poisson_normal', 'poisson_mixed', 'normal'])
            out, m = cut_paste(out, rgba, anat_masks, rng, blend=blend)
            if m.any():
                annots.append(('cutpaste', m, blend, None))
        elif r < cutpaste_prob + device_prob and avail_devices:
            dev_name = rng.choice(avail_devices)
            dev_inst_id = f'{dev_name}_{uuid.uuid4().hex[:8]}'
            parts = DEVICE_FNS[dev_name](H, W, _ablate_anat(anat_masks, rng), rng)
            for part in parts:
                part_name = part[0]
                part_mask = part[1]
                color     = part[2]
                kpts      = part[3] if len(part) >= 4 else None
                src_meta  = part[4] if len(part) >= 5 else None
                if not part_mask.any():
                    continue
                meta = dict(src_meta) if src_meta else {}
                meta.setdefault('device_instance_id', dev_inst_id)
                meta.setdefault('device_family', dev_name)
                mode = pick_physics_mode(part_name, rng)
                out, _ = render_with_physics(out, part_mask, color, anat_masks, rng, mode=mode)
                if part_name in SPLIT_INTO_INSTANCES:
                    for sub_cat, sub_mask, sub_kpts in _split_mask_instances(part_mask, SPLIT_INTO_INSTANCES[part_name]):
                        annots.append((sub_cat, sub_mask, mode, sub_kpts, dict(meta)))
                elif part_name in MULTI_KPT_CLUSTERS:
                    annots.append((part_name, part_mask, mode, _cluster_kpts(part_mask, MULTI_KPT_CLUSTERS[part_name]), meta))
                else:
                    annots.append((part_name, part_mask, mode, kpts, meta))
        elif avail_shapes:
            name = rng.choice(avail_shapes)
            ret = SHAPE_FNS[name](H, W, _ablate_anat(anat_masks, rng), rng)
            if len(ret) == 3:
                mask, color, kpts = ret
            else:
                mask, color = ret
                kpts = None
            if not mask.any():
                continue
            mode = pick_physics_mode(name, rng)
            out, _ = render_with_physics(out, mask, color, anat_masks, rng, mode=mode)
            if name in SPLIT_INTO_INSTANCES:
                for sub_cat, sub_mask, sub_kpts in _split_mask_instances(mask, SPLIT_INTO_INSTANCES[name]):
                    annots.append((sub_cat, sub_mask, mode, sub_kpts))
            elif name in MULTI_KPT_CLUSTERS:
                annots.append((name, mask, mode, _cluster_kpts(mask, MULTI_KPT_CLUSTERS[name])))
            else:
                annots.append((name, mask, mode, kpts))
    return out, annots

from pycocotools import mask as mask_utils
from collections import Counter

def mask_to_ann(mask, ann_id, image_id, cat_id, cat_name=None, render_mode=None,
                kpts_dict=None, img_w=None, img_h=None, source_meta=None):
    if (mask > 0).sum() == 0:
        return None
    bin_mask = (mask > 0).astype(np.uint8)
    ys, xs = np.where(bin_mask)
    x0, y0 = int(xs.min()), int(ys.min())
    x1, y1 = int(xs.max()), int(ys.max())
    rle = mask_utils.encode(np.asfortranarray(bin_mask))
    rle['counts'] = rle['counts'].decode('ascii')
    ann = dict(
        id=ann_id, image_id=image_id, category_id=cat_id,
        bbox=[x0, y0, x1 - x0 + 1, y1 - y0 + 1],
        area=int(bin_mask.sum()), iscrowd=0,
        segmentation=rle,
    )
    if render_mode is not None:
        ann['render_mode'] = render_mode
    # keypoints
    if cat_name is not None and KEYPOINT_SCHEMA.get(cat_name):
        if not kpts_dict:
            kpts_dict = _fallback_kpts(cat_name, bin_mask)
        # Universal in-mask snap for any explicit kpt that drifted off-mask
        kpts_dict = snap_kpts_to_mask(kpts_dict, bin_mask)
        H = img_h if img_h is not None else bin_mask.shape[0]
        W = img_w if img_w is not None else bin_mask.shape[1]
        flat, n_vis = _flatten_keypoints(cat_name, kpts_dict, W, H)
        ann['keypoints'] = flat
        ann['num_keypoints'] = n_vis
    # source provenance — present only for cutpaste annots that resolved
    # against the book manifest. lets downstream consumers trace each pasted
    # instance back to its source CXR + figure caption.
    if source_meta is not None:
        # Book provenance (cutpaste + hybrid bodies)
        if source_meta.get('source_book_image'):
            ann['source_book_image'] = source_meta['source_book_image']
            ann['book_pdf_page']     = source_meta.get('book_pdf_page')
            ann['book_category']     = source_meta.get('book_category')
            if source_meta.get('caption'):
                ann['book_caption'] = source_meta['caption']
        # ECCV24 cutout provenance — different source, no captions
        if source_meta.get('source') == 'eccv24':
            ann['cutout_source']     = 'eccv24'
            ann['source_eccv_dir']   = source_meta.get('source_eccv_dir')
        # Device-instance grouping — every part of one logical device shares
        # the same instance id, so downstream eval can collapse parts into a
        # family-level annotation without ambiguity.
        if source_meta.get('device_instance_id'):
            ann['device_instance_id'] = source_meta['device_instance_id']
        if source_meta.get('device_family'):
            ann['device_family'] = source_meta['device_family']
    return ann

def _save_viz(out_path, img, viz_items):
    """Save horizontal concat: [clean image | image + mask overlay + kpts]."""
    H, W = img.shape[:2]
    overlay = img.copy()
    # blend a red mask
    mask_union = np.zeros((H, W), bool)
    for _cat, m, _ in viz_items:
        mask_union |= (m > 0)
    if mask_union.any():
        red = np.zeros_like(overlay); red[..., 0] = 255
        alpha = 0.45
        blended = (1 - alpha) * overlay[mask_union].astype(np.float32) + alpha * red[mask_union].astype(np.float32)
        overlay[mask_union] = np.clip(blended, 0, 255).astype(np.uint8)
    # draw kpts (yellow dot, red ring) + skeleton (cyan)
    for cat_name, _m, kpts in viz_items:
        if not kpts:
            continue
        schema = KEYPOINT_SCHEMA.get(cat_name, [])
        coords = []
        for kname in schema:
            kp = kpts.get(kname)
            if kp and kp[2] > 0 and 0 <= kp[0] < W and 0 <= kp[1] < H:
                coords.append((int(kp[0]), int(kp[1])))
            else:
                coords.append(None)
        # skeleton lines (consecutive)
        for i in range(len(coords) - 1):
            if coords[i] and coords[i + 1]:
                cv2.line(overlay, coords[i], coords[i + 1], (0, 255, 255), 2, cv2.LINE_AA)
        # kpt dots
        for c in coords:
            if c is None:
                continue
            cv2.circle(overlay, c, 5, (255, 0, 0), 2, cv2.LINE_AA)
            cv2.circle(overlay, c, 3, (255, 255, 0), -1, cv2.LINE_AA)
    concat = np.concatenate([img, overlay], axis=1)
    cv2.imwrite(str(out_path), cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))


def _offscan_black_mask(gray, thresh=10, min_frac=0.004):
    """Pure-black NON-SCAN regions (collimation / letterbox / off-detector black):
    near-zero intensity in large connected components touching the image border.
    EXCLUDES the proper darker-but-nonzero scan background (lung fields etc.).
    Devices must never render here -- a device on off-scan black is a dead giveaway.
    Returns bool HxW."""
    blk = (gray <= thresh).astype(np.uint8)
    if blk.sum() == 0:
        return np.zeros(gray.shape, dtype=bool)
    _n, lbl = cv2.connectedComponents(blk)
    H, W = gray.shape
    border = set(np.concatenate([lbl[0, :], lbl[-1, :], lbl[:, 0], lbl[:, -1]]).tolist())
    out = np.zeros(gray.shape, dtype=bool)
    for c in border:
        if c != 0 and (lbl == c).sum() >= min_frac * H * W:
            out |= (lbl == c)
    return out


def run_pipeline(image_paths, cxas_model, cutouts, out_dir,
                 max_annotations=MAX_ANNOTATIONS, seed=RNG_SEED):
    rng = random.Random(seed)
    out_dir = Path(out_dir)
    (out_dir / 'images').mkdir(parents=True, exist_ok=True)
    (out_dir / 'masks').mkdir(parents=True, exist_ok=True)
    if not SKIP_VIZ:
        (out_dir / 'viz').mkdir(parents=True, exist_ok=True)
    img_ext = '.jpg' if JPG_QUALITY > 0 else '.png'
    img_write_params = ([cv2.IMWRITE_JPEG_QUALITY, JPG_QUALITY] if JPG_QUALITY > 0
                       else [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESS])
    mask_write_params = [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESS]
    coco = dict(
        info=dict(description='SynthFB-mini reimplementation + physics/devices + view-aware + archetypes', version='1.3'),
        licenses=[], images=[], annotations=[],
        categories=[
            dict(id=CAT2ID[c], name=c, supercategory='foreign_object',
                 keypoints=KEYPOINT_SCHEMA.get(c, []),
                 skeleton=SKELETONS.get(c, _default_skeleton(len(KEYPOINT_SCHEMA.get(c, [])))))
            for c in CATEGORIES
        ],
    )
    ann_id = 1
    view_counts = Counter()
    arch_counts = Counter()
    for img_id, p in enumerate(image_paths, start=1):
        img, anat = get_anatomy(cxas_model, p)
        view = classify_view(p)
        view_counts[view] += 1
        synth, annots, arch_name = generate_one_archetype(
            img, anat, cutouts, rng,
            max_annotations=max_annotations, view=view,
        )
        arch_counts[arch_name or 'uniform'] += 1
        # Don't render devices over pure-black NON-SCAN regions (collimation / letterbox /
        # off-detector black) -- a device floating on off-scan black is a dead giveaway.
        # Restore those pixels to the base and clip device masks to the scanned area;
        # mask_to_ann then recomputes bbox/seg + snaps keypoints onto the clipped mask.
        _offscan = _offscan_black_mask(img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY))
        if _offscan.any():
            _restore = np.zeros(_offscan.shape, dtype=bool)
            _kept = []
            for _ann in annots:
                _orig = _ann[1]
                _clipped = _orig.copy(); _clipped[_offscan] = 0
                if (_clipped > 0).sum() == 0:
                    _restore |= (_orig > 0)                 # device entirely off-scan -> drop
                    continue
                _restore |= (_orig > 0) & _offscan
                _kept.append((_ann[0], _clipped, *_ann[2:]))
            annots = _kept
            if _restore.any():
                synth = synth.copy(); synth[_restore] = img[_restore]
        H, W = synth.shape[:2]
        _op_rng = random.Random(RNG_SEED * 1000003 + img_id)   # deterministic PER-IMAGE op RNG (so the op
        _vscale = DEV_VIEW_SCALE.get(view, 1.0)                 # is reproducible + faint/bright A/B fully matches)
        _dev_mask = np.zeros((H, W), np.uint8)              # union of device/FB masks
        _faint_scale = np.zeros((H, W), np.float32)         # per-INSTANCE retain scale (0 = not faded)
        for _ann in annots:
            name = _ann[0]; mm = _ann[1] > 0
            _dev_mask = np.maximum(_dev_mask, mm.astype(np.uint8))
            a = int(mm.sum())
            if a == 0:
                continue
            ys, xs = np.where(mm)
            bb = (ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1)
            if (bb > 0 and a / bb < 0.15) or name in DEV_WIDE_TUBES:   # thin line OR named wide tube (ETT)
                lo, hi = DEV_RETAIN_SCALE.get(name, DEV_SCALE_DEFAULT)
                _faint_scale[mm] = _op_rng.uniform(lo, hi) * _vscale   # per-instance level; blobs stay opaque
        synth = realism_postprocess(synth, _dev_mask, _faint_scale, _op_rng)  # INSERTION-ONLY, per-image RNG
        img_fname  = f'{img_id:06d}{img_ext}'
        mask_fname = f'{img_id:06d}.png'   # masks always PNG (lossless required)
        cv2.imwrite(str(out_dir / 'images' / img_fname),
                    cv2.cvtColor(synth, cv2.COLOR_RGB2BGR), img_write_params)
        inst_mask = np.zeros((H, W), np.uint16)
        for i, ann in enumerate(annots, start=1):
            m = ann[1]
            inst_mask = np.where(m > 0, i, inst_mask)
        cv2.imwrite(str(out_dir / 'masks' / mask_fname),
                    inst_mask.astype(np.uint16), mask_write_params)
        coco['images'].append(dict(
            id=img_id, file_name=img_fname, width=int(W), height=int(H),
            source=str(p), view=view, archetype=arch_name,
        ))
        # build per-image kpt list for viz
        viz_items = []
        for annot in annots:
            # tuples are (cat, mask, mode), (cat, mask, mode, kpts),
            # or (cat, mask, mode, kpts, source_meta) for book-cutout pastes
            cat_name, m, mode = annot[0], annot[1], annot[2]
            kpts = annot[3] if len(annot) >= 4 else None
            meta = annot[4] if len(annot) >= 5 else None
            a = mask_to_ann(m, ann_id, img_id, CAT2ID[cat_name],
                            cat_name=cat_name, render_mode=mode, kpts_dict=kpts,
                            img_w=W, img_h=H, source_meta=meta)
            if a:
                coco['annotations'].append(a)
                ann_id += 1
                if not kpts:
                    kpts = _fallback_kpts(cat_name, m)
                viz_items.append((cat_name, m, kpts))
        if not SKIP_VIZ:
            _save_viz(out_dir / 'viz' / mask_fname, synth, viz_items)
    with open(out_dir / 'annotations.json', 'w') as f:
        json.dump(coco, f)
    print(f'view counts: {dict(view_counts)}')
    print(f'archetype counts: {dict(arch_counts)}')
    return coco

coco = run_pipeline(sample_paths, cxas_model, CUTOUTS, OUT_DIR)
print(f"wrote {len(coco['images'])} images, {len(coco['annotations'])} annotations -> {OUT_DIR}")
from collections import Counter
id2cat = {c['id']: c['name'] for c in coco['categories']}
print('category counts:', Counter(id2cat[a['category_id']] for a in coco['annotations']))
print('render-mode counts:',
      Counter(a.get('render_mode') for a in coco['annotations'] if a.get('render_mode') is not None))

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils

coco_api = COCO(str(OUT_DIR / 'annotations.json'))
img_ids = coco_api.getImgIds()
n_show = min(len(img_ids), 4)
fig, axes = plt.subplots(1, n_show, figsize=(5 * n_show, 5))
if n_show == 1: axes = [axes]
for ax, iid in zip(axes, img_ids[:n_show]):
    info = coco_api.loadImgs(iid)[0]
    img = cv2.cvtColor(cv2.imread(str(OUT_DIR / 'images' / info['file_name'])), cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    for a in coco_api.loadAnns(coco_api.getAnnIds(imgIds=iid)):
        m = mask_utils.decode(a['segmentation'])
        ax.imshow(np.ma.masked_where(m == 0, m), alpha=0.45, cmap='spring')
        x, y, w, h = a['bbox']
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=1.0))
        label = id2cat[a['category_id']]
        if a.get('render_mode'):
            label = f"{label}/{a['render_mode'][:4]}"
        ax.text(x, max(y - 2, 0), label, color='lime', fontsize=6,
                bbox=dict(facecolor='black', alpha=0.5, pad=0.5, edgecolor='none'))
    ax.set_title(f"image {iid} ({len(coco_api.getAnnIds(imgIds=iid))} ann)"); ax.axis('off')
plt.tight_layout(); plt.show()

def _try_nonempty(produce_fn, max_tries=5):
    """Retry produce_fn until it returns a non-empty mask, up to max_tries."""
    for _ in range(max_tries):
        ret = produce_fn()
        out, overlay = ret[0], ret[1]
        if overlay.any():
            return ret
    return ret


if SKIP_GALLERY:
    print('SYNTHFB_SKIP_GALLERY=1 — skipping per-category gallery + examples export')
    import sys as _sys
    _sys.exit(0)

gallery_rng = random.Random(99)
all_items = []
for name, fn in SHAPE_FNS.items():
    all_items.append(('shape', name, fn))
for name, fn in DEVICE_FNS.items():
    all_items.append(('device', name, fn))
if CUTOUTS:
    for cat in CUTOUTS.keys():
        all_items.append(('cutpaste', cat, None))


def render_shape(name, fn, rng_):
    ret = fn(demo_img.shape[0], demo_img.shape[1], demo_anat, rng_)
    m, c = ret[0], ret[1]
    kpts = ret[2] if len(ret) >= 3 else None
    mode = pick_physics_mode(name, rng_)
    out_img, _ = render_with_physics(demo_img.copy(), m, c, demo_anat, rng_, mode=mode)
    if name in SPLIT_INTO_INSTANCES:
        items = _split_mask_instances(m, SPLIT_INTO_INSTANCES[name])
        kpts_list = [(cat, k) for cat, _sub, k in items]
        return out_img, m, kpts_list
    if name in MULTI_KPT_CLUSTERS:
        kpts = _cluster_kpts(m, MULTI_KPT_CLUSTERS[name])
        return out_img, m, [(name, kpts)]
    if not kpts:
        kpts = _fallback_kpts(name, m)
    return out_img, m, [(name, kpts)]


def render_device(name, fn, rng_):
    parts = fn(demo_img.shape[0], demo_img.shape[1], demo_anat, rng_)
    out_img = demo_img.copy()
    combined = np.zeros(demo_img.shape[:2], np.uint8)
    kpts_list = []
    for part in parts:
        part_name, m, c = part[0], part[1], part[2]
        kpts = part[3] if len(part) >= 4 else None
        mode = pick_physics_mode(part_name, rng_)
        out_img, _ = render_with_physics(out_img, m, c, demo_anat, rng_, mode=mode)
        combined = np.maximum(combined, m)
        if part_name in SPLIT_INTO_INSTANCES:
            for sub_cat, _sub_mask, sub_kpts in _split_mask_instances(m, SPLIT_INTO_INSTANCES[part_name]):
                kpts_list.append((sub_cat, sub_kpts))
        elif part_name in MULTI_KPT_CLUSTERS:
            kpts_list.append((part_name, _cluster_kpts(m, MULTI_KPT_CLUSTERS[part_name])))
        else:
            if not kpts:
                kpts = _fallback_kpts(part_name, m)
            kpts_list.append((part_name, kpts))
    return out_img, combined, kpts_list


def render_cutpaste(cat, rng_):
    rgba = load_rgba(rng_.choice(CUTOUTS[cat]))
    blend = rng_.choice(['poisson_normal', 'poisson_mixed', 'normal'])
    out_img, m = cut_paste(demo_img.copy(), rgba, demo_anat, rng_, blend=blend)
    kpts = _fallback_kpts('cutpaste', m)
    return out_img, m, [('cutpaste', kpts)]



def _draw_kpts(ax, kpts_list, marker_size=40):
    """Draw keypoints + skeleton lines on matplotlib axis."""
    for part_name, kpts in kpts_list:
        if not kpts:
            continue
        schema = KEYPOINT_SCHEMA.get(part_name, [])
        coords = []
        for kname in schema:
            kp = kpts.get(kname)
            if kp and kp[2] > 0:
                coords.append((kp[0], kp[1]))
                ax.scatter(kp[0], kp[1], s=marker_size, c='yellow', edgecolors='red',
                           linewidths=1.5, zorder=10)
            else:
                coords.append(None)
        # skeleton (consecutive)
        for i in range(len(coords) - 1):
            if coords[i] and coords[i+1]:
                ax.plot([coords[i][0], coords[i+1][0]], [coords[i][1], coords[i+1][1]],
                        '-', color='cyan', linewidth=1.5, zorder=9)

# render once, cache (img, overlay) per category so both gallery + per-file use same
rendered = []
for kind, name, fn in all_items:
    if kind == 'shape':
        out_img, overlay, kpts_list = _try_nonempty(lambda: render_shape(name, fn, gallery_rng))
    elif kind == 'device':
        out_img, overlay, kpts_list = _try_nonempty(lambda: render_device(name, fn, gallery_rng))
    else:
        out_img, overlay, kpts_list = _try_nonempty(lambda: render_cutpaste(name, gallery_rng))
    rendered.append((kind, name, out_img, overlay, kpts_list))

n = len(rendered)
ncols = 6
nrows = (n + ncols - 1) // ncols

# gallery WITH mask overlay (for labeling inspection)
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
axes_flat = np.array(axes).reshape(-1)
for ax, (kind, name, out_img, overlay, kpts_list) in zip(axes_flat, rendered):
    ax.imshow(out_img)
    ax.imshow(np.ma.masked_where(overlay == 0, overlay), alpha=0.45, cmap='spring')
    _draw_kpts(ax, kpts_list, marker_size=25)
    ax.set_title(f'{kind[0].upper()} {name}', fontsize=8)
    ax.axis('off')
for ax in axes_flat[n:]:
    ax.axis('off')
plt.tight_layout()
plt.savefig('category_gallery.png', dpi=120, bbox_inches='tight')
plt.show()

# gallery CLEAN (for realism inspection — what model actually sees)
fig2, axes2 = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 3))
axes2_flat = np.array(axes2).reshape(-1)
for ax, (kind, name, out_img, overlay, kpts_list) in zip(axes2_flat, rendered):
    ax.imshow(out_img)
    ax.set_title(f'{kind[0].upper()} {name}', fontsize=8)
    ax.axis('off')
for ax in axes2_flat[n:]:
    ax.axis('off')
plt.tight_layout()
plt.savefig('category_gallery_clean.png', dpi=120, bbox_inches='tight')
plt.show()

print(f'gallery: {n} items -> category_gallery.png + category_gallery_clean.png')

# export each category as side-by-side PNG: [clean | mask + kpts]
import os
os.makedirs('category_examples', exist_ok=True)
for kind, name, out_img, overlay, kpts_list in rendered:
    fig_, (axL, axR) = plt.subplots(1, 2, figsize=(10, 5))
    axL.imshow(out_img); axL.set_title(f'{kind} | {name} (clean)', fontsize=10); axL.axis('off')
    axR.imshow(out_img)
    axR.imshow(np.ma.masked_where(overlay == 0, overlay), alpha=0.4, cmap='spring')
    _draw_kpts(axR, kpts_list, marker_size=80)
    axR.set_title(f'{kind} | {name} (mask + kpts)', fontsize=10); axR.axis('off')
    plt.tight_layout()
    fig_.savefig(f'category_examples/{kind}_{name}.png', dpi=100, bbox_inches='tight')
    plt.close(fig_)
print('exported', len(rendered), 'category examples to category_examples/ + category_examples_clean/')
