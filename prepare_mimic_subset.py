"""Filter a MIMIC metadata CSV → MIMIC train, FB/device-free subset.

Filter (in order):
- dataset_name == 'MIMIC'
- Mode == 'train'
- Support_Devices in {0,-1,-2} by default (CheXpert: 0=neg, -1=uncertain,
  -2=unlabeled, 1=pos). Override via --support-devices-keep.
- No_Finding == 1        (extra tightening; sibling device cols are all -2 in MIMIC,
                         so they can't help)
- Report NLP gate: split report into sentences; reject the image if ANY sentence
  mentions a device noun (incl. CABG/sternotomy history, surgical clips/staples/
  coils, orthopedic plates/screws, ports, valves, leads, lines, etc.) without
  a co-located negation/removal cue. Catches cases CheXpert mislabels or
  unlabels (silent leads narrated in adjacent sentences, prior hardware not
  coded as Support_Devices, etc.).
- View gate: keep ViewDetailed in {0=PA, 1=Lateral, 5=LL}; drop AP (=3) by
  default because portable/ICU studies almost always have monitoring leads.

Writes mimic_fb_free_train.csv with columns:
  img_path, full_path, Mode, No_Finding, Support_Devices, ViewDetailed, split

The 'split' column assigns each source image to train/val/test (default 90/5/5)
via deterministic seed. Source images are disjoint across splits.
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEVICE_NOUN_RE = re.compile(
    r'\b('
    # cardiac / electrical
    r'pacemakers?|pacers?|icds?|aicds?|defibrillators?|'
    r'(?:pacing|pacer|transvenous|epicardial|electrode|dual|single|triple|biv|bi-?ventricular)\s+leads?|'
    # bare "lead(s)" only when followed by a positional/state verb — catches
    # "leads are in unchanged position" without matching e.g. "this leads to atelectasis"
    r'\bleads?\b(?=\s+(?:are|is|in|terminat|extend|position|placed|seen|visible|noted|identified))|'
    # vascular access — bare "line(s)" qualified by route, or followed by positional verb
    r'(?:central\s+(?:venous\s+)?(?:access\s+)?|peripheral\s+|midline\s+|subclavian\s+|jugular\s+|femoral\s+|tunneled\s+|arterial\s+|picc\s+|port\s+)lines?|'
    r'\blines?\b(?=\s+(?:tip|terminat|projects|in\s+(?:place|good\s+position)))|'
    r'piccs?|cvcs?|swan-?ganz|port-?a-?caths?|mediports?|catheters?|'
    # surgical hardware
    r'stents?|(?:artificial|prosthetic|mechanical|bioprosthetic|aortic|mitral|tricuspid|pulmonary)\s+valves?|'
    r'sternotomy|sternotom(?:ies|y)|cabg|(?:sternal|cerclage|pacing|sternotomy|guide)\s+wires?|'
    # airway / GI
    r'tracheostomy|trach\s+tubes?|etts?|endotracheal\s+tubes?|'
    r'chest\s+tubes?|nasogastric|ng\s+tubes?|ngts?|enteric\s+tubes?|'
    # generic devices
    r'drains?|electrodes?|prosthes[ie]s|prosthetics?|implants?|hardware|metalware|'
    # surgical clips / staples / coils — standalone, not just qualified
    r'(?:surgical\s+|metallic\s+)?clips?|'
    r'(?:surgical\s+|skin\s+)?staples?|'
    r'(?:embolization\s+)?coils?|'
    # orthopedic plates/screws/rods — REQUIRE qualifier so "plate densities"
    # (lung atelectasis pattern) doesn't false-positive
    r'(?:orthopedic|metal|metallic|fixation|fusion|spinal?|cervical|thoracic|lumbar|sternal|surgical|pedicle|locking|compression|hip|knee|shoulder)\s+(?:plates?|screws?|rods?)|'
    # joint replacement implants (bare, no s/p prefix required)
    r'arthroplas(?:ty|ties)|'
    # ekg / monitor leads (extremely common silent device in ICU/portable CXRs)
    r'(?:ekg|ecg)(?:\s+(?:leads?|electrodes?|stickers?|wires?|monitors?))?|'
    r'(?:cardiac\s+|external\s+)?monitor(?:s|ing)?\s+(?:leads?|wires?|stickers?|electrodes?)|'
    r'monitoring\s+(?:leads?|wires?|electrodes?|stickers?)|'
    # history-of-surgery markers
    r'(?:status[-\s]post|s/p|prior|previous|h/o|history\s+of)\s+'
    r'(?:cabg|sternotomy|avr|mvr|valve\s+replacement|aortic\s+valve|mitral\s+valve|'
    r'mastectomy|lobectomy|pneumonectomy|thoracotomy|wedge\s+resection)'
    r')\b',
    re.IGNORECASE,
)

# Negation: only count as negation when the cue is structurally tied to a device
# (e.g. "removal of the chest tube", "no longer visible", "no pacemaker"). Bare
# "no/not/without/absent" on its own previously rescued sentences like
# "port-a-cath terminates in the mid svc, without evidence of pneumothorax" —
# the "without" negates pneumothorax, not the device.
NEGATION_RE = re.compile(
    r'\b('
    r'removed|extracted|withdrawn|discontinued|d/c(?:\'?d)?|'
    r'(?:interval|s/p|recent|prior|previous|h/o|history\s+of)?\s*'
    r'(?:removal|extraction)\s+of|'
    r'no\s+longer\s+(?:present|seen|visible|visualized|visualised|in\s+place|noted|identified|appreciated)|'
    r'(?:exists?|remains?)\s+removed|'
    # device-adjacent "no X" / "no evidence of X" — only valid if X itself is a device noun
    r'no\s+(?:evidence\s+of\s+)?(?:central\s+|peripheral\s+|subclavian\s+|jugular\s+|chest\s+|endotracheal\s+|enteric\s+|nasogastric\s+|surgical\s+|pacing\s+|cardiac\s+|monitoring\s+)?'
    r'(?:pacemakers?|icds?|aicds?|leads?|lines?|catheters?|tubes?|wires?|drains?|stents?|valves?|hardware|implants?|clips?|staples?|coils?|electrodes?|ports?|piccs?|cvcs?)'
    r')\b',
    re.IGNORECASE,
)

SENT_SPLIT_RE = re.compile(r'[.;\n]|\|')

# Any-mention nouns. The fact that a report references the noun AT ALL is enough
# to disqualify the image — even if it says "removed". Examples we caught:
#   "removed wires" → image still shows sternotomy cerclage from prior CABG.
#   "removed catheter" → image still shows pacing leads or other prior hardware.
#   "removed support device" → ambiguous, but suggests recent device-bearing state.
# Better to reject these and lose the row than admit contaminated training data.
PERMANENT_DEVICE_RE = re.compile(
    r'\b('
    # cardiac surgery
    r'sternotomy|sternotom(?:ies|y)|cabg|coronary\s+artery\s+bypass|'
    # past surgery markers
    r'(?:status[-\s]post|s/p|prior|previous|h/o|history\s+of)\s+'
    r'(?:cabg|sternotomy|avr|mvr|valve\s+replacement|aortic\s+valve\s+replacement|'
    r'mitral\s+valve\s+replacement|mastectomy|lobectomy|pneumonectomy|thoracotomy|'
    r'wedge\s+resection|coronary\s+artery\s+bypass|device|implant|hardware)|'
    # implanted hardware
    r'(?:artificial|prosthetic|mechanical|bioprosthetic)\s+valves?|'
    r'arthroplas(?:ty|ties)|'
    r'(?:hip|knee|shoulder|total)\s+(?:replacement|arthroplasty|prosthes[ie]s)|'
    # spinal hardware
    r'spinal?\s+fusion|fusion\s+hardware|'
    r'(?:cervical|thoracic|lumbar|spinal|spine)\s+(?:plates?|screws?|rods?|fusion|hardware|fixation)|'
    # stents, valves, coils
    r'(?:coronary\s+|tracheal\s+|esophageal\s+|aortic\s+)?stents?|'
    r'annuloplasty|watchman|mitra-?clip|tavr|'
    r'(?:embolization\s+)?coils?|'
    # surgical clips/staples — even if "removed" they leave RIs
    r'(?:surgical\s+|metallic\s+)?clips?|(?:surgical\s+|skin\s+)?staples?|'
    # generic permanent implants
    r'prosthes[ie]s|prosthetics?|metalware|orthopedic\s+hardware|hardware|'
    # any device generators — even "removed" leaves leads or sternotomy
    r'pacemakers?|pacers?|icds?|aicds?|defibrillators?|'
    # bare "wires" — too ambiguous: could be EKG (removable) but usually sternotomy
    # cerclage from past CABG. The user has confirmed visible wires after "removed
    # wires" reports. Rejecting unconditionally.
    r'wires?|'
    # bare "leads" — pacing leads usually stay even when generator removed
    r'leads?|'
    # bare "device(s)" — even "monitoring device removed" patient was recently
    # device-bearing and may have residual cabling
    r'devices?|monitoring|monitor|'
    # bare "support" — covers "support device removed"
    r'support\s+(?:device|equipment)|'
    # explicit removal language — patient was recently bearing
    r'removed\s+(?:wires?|catheters?|tubes?|lines?|leads?|devices?|drains?|stents?|'
    r'pacemakers?|hardware|implants?|monitoring|support)|'
    r'(?:wires?|catheters?|tubes?|lines?|leads?|devices?|drains?)\s+(?:are|were|appear|have\s+been|has\s+been)\s+removed'
    r')\b',
    re.IGNORECASE,
)


def report_mentions_device(report: str) -> bool:
    """Return True if the report indicates a foreign body is (or was) present.

    Two passes:
      1) PERMANENT_DEVICE_RE: any mention rejects immediately, no negation override.
         These are post-surgical artifacts that don't disappear even when the
         report says 'removed' (sternotomy cerclage, valve replacement, fusion
         hardware, retained clips, etc.).
      2) DEVICE_NOUN_RE: regular device nouns with sentence-level negation logic
         (removable items: catheters, tubes, lines, drains).
    """
    if not isinstance(report, str) or not report.strip():
        return False
    # Pass 1: permanent hardware mentions, anywhere in report — unconditional reject
    if PERMANENT_DEVICE_RE.search(report):
        return True
    # Pass 2: sentence-level negation-aware check for removable devices
    for sent in SENT_SPLIT_RE.split(report):
        sent = sent.strip()
        if not sent:
            continue
        if DEVICE_NOUN_RE.search(sent) and not NEGATION_RE.search(sent):
            return True
    return False

DEFAULT_SRC_CSV = os.environ.get('MIMIC_META_CSV', './mimic_metadata.csv')
DEFAULT_IMG_ROOT = os.environ.get('MIMIC_CLEAN', './mimic_data/images')
DEFAULT_OUT = './mimic_fb_free_train.csv'


def build(src_csv: str, img_root: str, out_csv: str, max_n: int | None, seed: int, verify: bool,
          train_frac: float, val_frac: float, split_seed: int,
          view_keep: set[int] | None = None,
          ap_allowlist: set[str] | None = None,
          support_devices_keep: set[int] | None = None,
          include_pathology: bool = False):
    print(f'Reading {src_csv}')
    df = pd.read_csv(src_csv, low_memory=False)
    print(f'Total rows: {len(df):,}')

    mim_train = df[(df['dataset_name'] == 'MIMIC') & (df['Mode'] == 'train')]
    print(f'MIMIC train: {len(mim_train):,}')

    if support_devices_keep is None:
        support_devices_keep = {0}
    clean = mim_train[mim_train['Support_Devices'].isin(support_devices_keep)].copy()
    print(f'After Support_Devices in {sorted(support_devices_keep)}: {len(clean):,}')

    # No_Finding gate. By default we keep only normals (==1) — the paper's "FB-free
    # normals" pool. With --include-pathology, we admit pathological cases too, relying
    # on Support_Devices + NLP gate + downstream image classifier to keep them FB-free.
    # Rationale: deployed CXRs that actually get devices almost always have pathology,
    # so a pathology-bearing background distribution matches real test-time better.
    if 'No_Finding' in clean.columns and not include_pathology:
        clean = clean[clean['No_Finding'] == 1].copy()
        print(f'After No_Finding==1: {len(clean):,}')
    elif include_pathology:
        print(f'(skipping No_Finding gate — admitting pathology)')

    device_hit = clean['report'].apply(report_mentions_device)
    n_rejected = int(device_hit.sum())
    clean = clean[~device_hit].copy()
    print(f'After report NLP gate: {len(clean):,}  (rejected {n_rejected:,})')

    # View gate: keep PA + Lateral + LL by default (ViewDetailed: 0=PA, 1=LAT, 5=LL).
    # AP (==3) is dropped because it heavily correlates with portable/ICU studies
    # where silent ECG leads / monitoring electrodes are common and not narrated.
    # Optional allowlist re-admits hand-vetted AP images.
    if 'ViewDetailed' in clean.columns and view_keep is not None:
        before = len(clean)
        kept_views = clean[clean['ViewDetailed'].isin(view_keep)].copy()
        if ap_allowlist:
            ap_admit = clean[
                (clean['ViewDetailed'] == 3)
                & (clean['img_path'].isin(ap_allowlist))
            ].copy()
            print(f'AP allowlist admit: {len(ap_admit):,} of {len(ap_allowlist):,} requested')
            kept_views = pd.concat([kept_views, ap_admit], ignore_index=False)
        clean = kept_views
        print(f'After ViewDetailed gate (keep={sorted(view_keep)}): '
              f'{len(clean):,}  (dropped {before - len(clean):,})')

    if verify:
        print('Verifying file existence (may take a moment)...')
        clean['full_path'] = clean['img_path'].apply(lambda p: os.path.join(img_root, p))
        exists = clean['full_path'].apply(os.path.isfile)
        clean = clean[exists]
        print(f'After file-existence check: {len(clean):,}')
    else:
        clean['full_path'] = clean['img_path'].apply(lambda p: os.path.join(img_root, p))

    if max_n is not None and len(clean) > max_n:
        clean = clean.sample(n=max_n, random_state=seed).reset_index(drop=True)
        print(f'Sampled to {len(clean):,}')
    else:
        clean = clean.reset_index(drop=True)

    # Assign train/val/test split deterministically over source images
    if train_frac + val_frac > 1.0:
        raise SystemExit(f'train_frac + val_frac must be <= 1.0 (got {train_frac}+{val_frac})')
    rng = np.random.default_rng(split_seed)
    n = len(clean)
    order = rng.permutation(n)
    n_train = int(round(n * train_frac))
    n_val = int(round(n * val_frac))
    split = np.empty(n, dtype=object)
    split[order[:n_train]] = 'train'
    split[order[n_train:n_train + n_val]] = 'val'
    split[order[n_train + n_val:]] = 'test'
    clean['split'] = split
    counts = pd.Series(split).value_counts().to_dict()
    print(f'Split (seed={split_seed}, train={train_frac}, val={val_frac}): {counts}')

    keep = ['img_path', 'full_path', 'Mode', 'No_Finding', 'Support_Devices', 'ViewDetailed', 'split']
    keep = [c for c in keep if c in clean.columns]
    out = clean[keep].copy()
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f'Wrote {len(out):,} rows -> {out_csv}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=DEFAULT_SRC_CSV)
    ap.add_argument('--img-root', default=DEFAULT_IMG_ROOT)
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--max-n', type=int, default=None,
                    help='Optional cap on number of rows (random sample). Default: keep all.')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no-verify', action='store_true',
                    help='Skip on-disk file existence check.')
    ap.add_argument('--train-frac', type=float, default=0.90,
                    help='Fraction of source images for train split.')
    ap.add_argument('--val-frac', type=float, default=0.05,
                    help='Fraction for val. Test gets the remainder.')
    ap.add_argument('--split-seed', type=int, default=42,
                    help='Deterministic seed for train/val/test split assignment.')
    ap.add_argument('--view-keep', default='0,1,5',
                    help='Comma-separated ViewDetailed codes to keep '
                         '(0=PA, 1=LAT, 3=AP, 5=LL). Default: 0,1,5. '
                         'Pass empty string to disable the view gate.')
    ap.add_argument('--ap-allowlist',
                    help='Optional text file with img_path per line. Listed AP '
                         '(ViewDetailed==3) images are re-admitted after the view gate.')
    ap.add_argument('--support-devices-keep', default='0,-1,-2',
                    help='Comma-separated Support_Devices values to keep '
                         '(CheXpert: 0=neg, -1=uncertain, -2=unlabeled, 1=pos). '
                         'Default: 0,-1,-2 (rely on NLP gate to scrub -2 unlabeled).')
    ap.add_argument('--include-pathology', action='store_true',
                    help='Skip the No_Finding==1 gate so pathological-but-FB-free images '
                         'are admitted too (larger pool, distribution closer to real '
                         'deployment where devices appear on diseased lungs).')
    args = ap.parse_args()

    view_keep: set[int] | None
    if args.view_keep.strip() == '':
        view_keep = None
    else:
        view_keep = {int(x) for x in args.view_keep.split(',') if x.strip()}

    ap_allowlist: set[str] | None = None
    if args.ap_allowlist:
        with open(args.ap_allowlist) as f:
            ap_allowlist = {ln.strip() for ln in f if ln.strip() and not ln.startswith('#')}

    sd_keep = {int(x) for x in args.support_devices_keep.split(',') if x.strip()}

    build(args.src, args.img_root, args.out, args.max_n, args.seed,
          verify=not args.no_verify,
          train_frac=args.train_frac, val_frac=args.val_frac, split_seed=args.split_seed,
          view_keep=view_keep, ap_allowlist=ap_allowlist,
          support_devices_keep=sd_keep,
          include_pathology=args.include_pathology)


if __name__ == '__main__':
    main()
