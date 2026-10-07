#!/usr/bin/env python3
"""
convert_adni.py — filtered, resumable, parallel DICOM -> NIfTI for ADNI.

Replaces scripts/00_dicom_to_nifti.py, which keys its output directory on
(subject, description) and so collapses every repeat timepoint of the same
protocol into one file. Here every series is written under its own
Image Data ID, which is unique, so nothing is lost.

Layout in:   <root>/<Subject>/<Description>/<Session>/<ImageDataID>/*.dcm
Layout out:  <out>/<Subject>/<ImageDataID>.nii.gz
Manifest:    <out>/manifest.csv   (one row per series, incl. skipped ones)

Filtering, in this order:
  1. Series whose Description names a non-anatomical acquisition
     (B1 calibration, calibration scan, AAHead_Scout, SmartBrain, localizer,
     field mapping) - matched with spaces OR underscores, because LONI writes
     directory names with underscores.
  2. Series with fewer than --min-slices DICOM files. This is the check that
     catches the 112 series named MPRAGE / MPRAGE_SENSE2 that hold 1-29 slices.
     Name-based filtering alone does not catch them.
  3. dcm2niix -i y, which additionally refuses derived, localiser and 2D images.

Usage:
    python3 convert_adni.py --root /data/raw/ADNI --out /data/nifti --jobs 4
    python3 convert_adni.py --dry-run                 # report only, convert nothing
    python3 convert_adni.py --shard 0 --num-shards 4  # run 4 copies concurrently

Resumable: a series whose output already exists is skipped, so re-running after
an interruption costs nothing.

Requires: dcm2niix  (sudo apt-get install -y dcm2niix)
"""

import argparse
import csv
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

NON_T1 = re.compile(
    r'b1[-_ ]?calibration'
    r'|calibration[-_ ]?scan'
    r'|aahead'
    r'|smartbrain'
    r'|localizer|localiser'
    r'|scout'
    r'|field[-_ ]?map',        # matches "Field Mapping" AND "Field_Mapping"
    re.I,
)

DICOM_EXT = ('.dcm', '.dicom', '.ima')


def scan(root):
    """Yield (subject, description, session, image_id, series_dir, n_files)."""
    for subject in sorted(os.listdir(root)):
        sdir = os.path.join(root, subject)
        if not os.path.isdir(sdir):
            continue
        for desc in sorted(os.listdir(sdir)):
            ddir = os.path.join(sdir, desc)
            if not os.path.isdir(ddir):
                continue
            for sess in sorted(os.listdir(ddir)):
                sedir = os.path.join(ddir, sess)
                if not os.path.isdir(sedir):
                    continue
                for img in sorted(os.listdir(sedir)):
                    idir = os.path.join(sedir, img)
                    if not os.path.isdir(idir):
                        continue
                    n = sum(1 for f in os.listdir(idir)
                            if f.lower().endswith(DICOM_EXT) or '.' not in f)
                    yield subject, desc, sess, img, idir, n


def convert_one(task):
    """Run dcm2niix for one series. Returns a manifest row dict."""
    subject, desc, sess, img, series_dir, n_files, out_root = task
    out_dir = os.path.join(out_root, subject)
    os.makedirs(out_dir, exist_ok=True)
    target = os.path.join(out_dir, img + '.nii.gz')

    row = {'subject': subject, 'description': desc, 'session': sess,
           'image_id': img, 'n_dicom': n_files, 'nifti': '', 'status': '',
           'note': ''}

    if os.path.exists(target) and os.path.getsize(target) > 0:
        row['nifti'], row['status'] = target, 'exists'
        return row

    cmd = ['dcm2niix',
           '-z', 'y',          # gzip
           '-b', 'n',          # no JSON sidecar (we keep our own manifest)
           '-i', 'y',          # ignore derived / localiser / 2D
           '-f', img,          # output name = Image Data ID
           '-o', out_dir,
           series_dir]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True,
                           check=False, timeout=600)
    except subprocess.TimeoutExpired:
        row['status'], row['note'] = 'error', 'dcm2niix timeout (600s)'
        return row

    produced = sorted(f for f in os.listdir(out_dir)
                      if f.startswith(img) and f.endswith('.nii.gz'))
    if not produced:
        row['status'] = 'error'
        row['note'] = (p.stderr or p.stdout or '').strip().replace('\n', ' ')[:200]
        return row

    if len(produced) > 1:
        # dcm2niix split the series (echoes, phase/mag, ...). Keep the largest,
        # flag it so it can be reviewed rather than silently trusted.
        produced.sort(key=lambda f: os.path.getsize(os.path.join(out_dir, f)),
                      reverse=True)
        row['note'] = f'dcm2niix produced {len(produced)}: {",".join(produced)}'
        row['status'] = 'ok_multi'
    else:
        row['status'] = 'ok'
    row['nifti'] = os.path.join(out_dir, produced[0])
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/data/raw/ADNI')
    ap.add_argument('--out', default='/data/nifti')
    ap.add_argument('--manifest', default=None,
                    help='default: <out>/manifest.csv (or manifest_shardN.csv)')
    ap.add_argument('--min-slices', type=int, default=30)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--dry-run', action='store_true',
                    help='report what would be converted, write nothing')
    a = ap.parse_args()

    if not a.dry_run and shutil.which('dcm2niix') is None:
        sys.exit('ERROR: dcm2niix not on PATH.  sudo apt-get install -y dcm2niix')
    if not os.path.isdir(a.root):
        sys.exit(f'ERROR: {a.root} does not exist')

    print(f'Scanning {a.root} ...', flush=True)
    all_series = list(scan(a.root))
    print(f'  {len(all_series):,} series found\n', flush=True)

    keep, drop_name, drop_short = [], [], []
    for subject, desc, sess, img, sdir, n in all_series:
        if NON_T1.search(desc):
            drop_name.append((subject, desc, img, n))
        elif n < a.min_slices:
            drop_short.append((subject, desc, img, n))
        else:
            keep.append((subject, desc, sess, img, sdir, n, a.out))

    print(f'  excluded by name      : {len(drop_name):,}')
    print(f'  excluded by <{a.min_slices} slices : {len(drop_short):,}')
    print(f'  to convert            : {len(keep):,}'
          f'  from {len({k[0] for k in keep}):,} subjects\n', flush=True)

    if drop_short:
        from collections import Counter
        print('  short series by description (these are the dangerous ones):')
        for d, n in Counter(x[1] for x in drop_short).most_common(10):
            print(f'    {n:>5}  {d}')
        print(flush=True)

    if a.num_shards > 1:
        keep = [t for i, t in enumerate(keep) if i % a.num_shards == a.shard]
        print(f'  shard {a.shard}/{a.num_shards}: {len(keep):,} series\n', flush=True)

    if a.dry_run:
        print('[dry run] nothing converted')
        return

    os.makedirs(a.out, exist_ok=True)
    manifest = a.manifest or os.path.join(
        a.out, 'manifest.csv' if a.num_shards == 1 else f'manifest_shard{a.shard}.csv')

    t0, rows, done = time.time(), [], 0
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        futs = {ex.submit(convert_one, t): t for t in keep}
        for f in as_completed(futs):
            rows.append(f.result())
            done += 1
            if done % 200 == 0 or done == len(keep):
                el = time.time() - t0
                rate = done / el if el else 0
                eta = (len(keep) - done) / rate / 60 if rate else 0
                print(f'  [{done:,}/{len(keep):,}] {rate:.1f}/s  ETA {eta:.0f} min',
                      flush=True)

    with open(manifest, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['subject', 'description', 'session',
                                           'image_id', 'n_dicom', 'nifti',
                                           'status', 'note'])
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: (r['subject'], r['image_id'])))

    from collections import Counter
    st = Counter(r['status'] for r in rows)
    print(f'\nDone in {(time.time()-t0)/60:.1f} min')
    for k, v in sorted(st.items()):
        print(f'  {k:<10}: {v:,}')
    errs = [r for r in rows if r['status'] == 'error']
    if errs:
        print('\n  first 5 errors:')
        for r in errs[:5]:
            print(f'    {r["image_id"]}  {r["note"][:120]}')
    print(f'\n[written] {manifest}')


if __name__ == '__main__':
    main()
