#!/usr/bin/env python3
"""
TAFNet preprocessing, version 2 -- longitudinal, whole-brain. Used for the paper.

Replaces the five-step v1 pipeline in scripts/01_preprocess.py (brain extraction ->
SyNRA to MNI per scan -> min-max -> Gaussian -> centre crop to 128^3), whose final
crop removed part of the cortex.  The stream is:

  per scan      1. N4 bias-field correction
  per subject   2. unbiased within-subject template (iterative rigid averaging of
                   all of that subject's scans; a subject with one scan is its own
                   template)
                3. brain extraction ONCE, on the subject template (one mask shared
                   by every timepoint of the subject)
                4. subject template -> MNI152NLin2009cAsym (affine by default;
                   --mni-transform SyN for a nonlinear warp)
  per scan      5. compose [scan -> template] + [template -> MNI] and resample ONCE
                   from the native N4 image onto the output grid
                6. percentile intensity normalisation inside the brain mask -> [0, 1]
                7. QC: whole brain inside the box (no brain voxels on any face),
                   within-subject alignment, template-to-MNI alignment

There is no Gaussian smoothing and no cropping.  The output grid is 128^3 at
1.6 mm isotropic (204.8 mm field of view) centred on the MNI brain, so the whole
brain fits and the network input shape is unchanged.  (At 1.5 mm / 192 mm the
181 mm long MNI brain left under 4 voxels of margin and real brains touched the box.)

Input : a manifest CSV with columns image_id, subject, nifti (scripts/convert_adni.py
        writes one; rows whose status is not ok are skipped)
Output: <out>/<subject>_<image_id>.nii.gz   (float32, 128^3, [0,1])
        <work>/<subject>/                    (template, mask, transforms)
        <out>/preprocess_v2_log.csv          (one row per scan, QC numbers)

Example (test on 3 subjects):
  python scripts/preprocess_v2.py --manifest nifti/manifest.csv \
      --out preprocessed_v2 --work work_v2 --limit-subjects 3 --jobs 1
"""
import argparse, csv, os, sys, time, traceback, collections
import multiprocessing as mp

GRID = 128          # voxels per side of the network input
SPACING = 1.6       # mm, isotropic -> 204.8 mm field of view (set with --spacing)
_G = {}             # per-process globals (template, reference grid, options)


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _brain_mask(img, extractor):
    """Binary brain mask of a T1 image."""
    import ants
    if extractor == 'antspynet':
        import antspynet
        prob = antspynet.brain_extraction(img, modality='t1', verbose=False)
        return ants.threshold_image(prob, 0.5, 1.0, 1, 0)
    # 'threshold' is a crude fallback used ONLY for software testing
    return ants.get_mask(img)


def _n4(img):
    """Step 1: N4 bias-field correction on the native scan."""
    import ants, numpy as np
    arr = img.numpy()
    arr = np.clip(arr, 0, np.percentile(arr, 99.9))        # tame extreme voxels
    img = img.new_image_like(arr.astype('float32'))
    return ants.n4_bias_field_correction(img, mask=ants.get_mask(img), shrink_factor=4)


def _mean_image(images):
    import numpy as np
    return images[0].new_image_like(np.mean([i.numpy() for i in images], axis=0).astype('float32'))


def _corr(a, b, m):
    import numpy as np
    x, y = a[m], b[m]
    if x.size < 10 or x.std() < 1e-8 or y.std() < 1e-8:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _dice(a, b):
    s = a.sum() + b.sum()
    return float(2.0 * (a & b).sum() / s) if s else float('nan')


def register_to_mni(tbrain, tmask, wdir, o):
    """Step 4. Align the subject-template brain to the MNI brain.

    A plain affine started from the centre of mass can settle on a tilted solution when
    the head is pitched in the scanner.  We therefore try an affine seeded by a coarse
    multi-start rotation search and the plain affine, add a staged (TRSAA) attempt if both
    are weak, score each on the output grid, and keep the best one.  The score is the
    Dice overlap between the subject's brain mask and the MNI brain mask on the output
    grid.  Intensity correlation with the MNI average is NOT a usable score: an individual
    atrophied brain correlates only about 0.5 with the smooth group average even when it
    is well aligned (it is still logged, for information).
    """
    import ants
    ref, mni = _G['ref'], _G['mni_brain']

    def score(tx):
        m = ants.apply_transforms(fixed=ref, moving=tmask, transformlist=tx,
                                  interpolator='nearestNeighbor').numpy() > 0.5
        return _dice(m, _G['mni_mask_on_ref'])

    best_tx, best_s, best_how = None, -1.0, ''
    for how in ('search', 'plain', 'TRSAA'):
        if how == 'TRSAA' and best_s >= o.good_mni_dice:
            break                                   # third attempt only if the first two are weak
        try:
            kw = {}
            if how == 'search':
                kw['initial_transform'] = ants.affine_initializer(
                    ants.resample_image(mni, (3, 3, 3)), ants.resample_image(tbrain, (3, 3, 3)),
                    search_factor=15, radian_fraction=0.15, use_principal_axis=False,
                    local_search_iterations=10, txfn=os.path.join(wdir, 'mni_init.mat'))
            reg = ants.registration(fixed=mni, moving=tbrain,
                                    type_of_transform='TRSAA' if how == 'TRSAA' else 'Affine',
                                    outprefix=os.path.join(wdir, f'template_to_mni_{how}_'), **kw)
            s = score(reg['fwdtransforms'])
        except Exception:
            continue
        if s == s and s > best_s:
            best_tx, best_s, best_how = reg['fwdtransforms'], s, how
    if best_tx is None:
        raise RuntimeError('template-to-MNI registration failed')
    if o.mni_transform == 'SyN':
        reg = ants.registration(fixed=mni, moving=tbrain, type_of_transform='SyN',
                                initial_transform=best_tx[0],
                                outprefix=os.path.join(wdir, 'template_to_mni_syn_'))
        best_tx, best_s = reg['fwdtransforms'], score(reg['fwdtransforms'])
    return best_tx, best_s, best_how


def build_reference(mni, mni_mask):
    """128^3 grid at 1.5 mm, centred on the MNI brain, same axes as the template."""
    import ants, numpy as np
    idx = np.argwhere(mni_mask.numpy() > 0)
    lo = np.array(ants.transform_index_to_physical_point(mni_mask, idx.min(0).tolist()))
    hi = np.array(ants.transform_index_to_physical_point(mni_mask, idx.max(0).tolist()))
    centre = (lo + hi) / 2.0
    extent = np.abs(hi - lo)
    D = np.array(mni.direction)
    origin = centre - D @ (np.ones(3) * SPACING * (GRID - 1) / 2.0)
    ref = ants.make_image((GRID,) * 3, voxval=0.0, spacing=(SPACING,) * 3,
                          origin=tuple(float(v) for v in origin), direction=D)
    return ref, extent


# ----------------------------------------------------------------------------
# one subject
# ----------------------------------------------------------------------------
def process_subject(job):
    subject, scans = job                       # scans: list of (image_id, nifti_path)
    o = _G['opt']
    import ants, numpy as np
    rows, t0 = [], time.time()
    wdir = os.path.join(o.work, subject); os.makedirs(wdir, exist_ok=True)
    outs = {iid: os.path.join(o.out, f'{subject}_{iid}.nii.gz') for iid, _ in scans}
    if all(os.path.exists(p) for p in outs.values()) and not o.overwrite:
        return [dict(subject=subject, image_id=iid, status='exists') for iid, _ in scans]
    try:
        # 1. N4 on every scan
        n4 = {iid: _n4(ants.image_read(p)) for iid, p in scans}
        ids = [iid for iid, _ in scans]

        # 2. unbiased within-subject template (iterative rigid averaging)
        template = n4[ids[0]]
        rigid = {}
        n_iter = o.template_iters if len(ids) > 1 else 1
        for it in range(n_iter):
            warped = []
            for iid in ids:
                reg = ants.registration(fixed=template, moving=n4[iid], type_of_transform='Rigid',
                                        outprefix=os.path.join(wdir, f'{iid}_to_template_'))
                rigid[iid] = reg['fwdtransforms']
                warped.append(reg['warpedmovout'])
            template = _mean_image(warped)

        # 3. brain extraction once, on the template; refine rigid alignment on brain only
        tmask = _brain_mask(template, o.brain_extractor)
        if len(ids) > 1:
            warped = []
            for iid in ids:
                reg = ants.registration(fixed=template, moving=n4[iid], type_of_transform='Rigid',
                                        mask=tmask, initial_transform=rigid[iid][0],
                                        outprefix=os.path.join(wdir, f'{iid}_to_template_refined_'))
                rigid[iid] = reg['fwdtransforms']
                warped.append(reg['warpedmovout'])
            template = _mean_image(warped)
            tmask = _brain_mask(template, o.brain_extractor)
        ants.image_write(template, os.path.join(wdir, 'template.nii.gz'))
        ants.image_write(tmask, os.path.join(wdir, 'template_mask.nii.gz'))

        # 4. subject template (brain) -> MNI brain (multi-start, best candidate kept)
        to_mni, mni_dice, mni_how = register_to_mni(template * tmask, tmask, wdir, o)

        # one mask per subject, on the output grid
        ref = _G['ref']
        mask = ants.apply_transforms(fixed=ref, moving=tmask, transformlist=to_mni,
                                     interpolator='nearestNeighbor').numpy() > 0.5
        tmpl_out = ants.apply_transforms(fixed=ref, moving=template, transformlist=to_mni,
                                         interpolator='linear').numpy()
        mni_corr = _corr(tmpl_out, _G['mni_on_ref'], mask & _G['mni_mask_on_ref'])
        faces = [int(mask[0].sum()), int(mask[-1].sum()), int(mask[:, 0].sum()),
                 int(mask[:, -1].sum()), int(mask[:, :, 0].sum()), int(mask[:, :, -1].sum())]

        # 5-6. single resampling from native N4 image, then intensity normalisation
        for iid in ids:
            vol = ants.apply_transforms(fixed=ref, moving=n4[iid],
                                        transformlist=to_mni + rigid[iid],
                                        interpolator='linear').numpy()
            raw_corr = _corr(vol, tmpl_out, mask)
            v = vol[mask]
            lo, hi = np.percentile(v, [o.pct_low, o.pct_high])
            vol = np.clip((vol - lo) / max(hi - lo, 1e-8), 0, 1)
            vol[~mask] = 0
            img = ref.new_image_like(vol.astype('float32'))
            status, note = 'ok', ''
            if max(faces) > o.max_face_voxels:
                status, note = 'flag', 'brain touches the edge of the box'
            elif not (raw_corr > o.min_template_corr):
                status, note = 'flag', 'poor within-subject alignment'
            elif not (mni_dice > o.min_mni_dice):
                status, note = 'flag', 'poor template-to-MNI alignment'
            ants.image_write(img, outs[iid])
            rows.append(dict(subject=subject, image_id=iid, status=status, note=note,
                             n_scans_subject=len(ids), brain_voxels=int(mask.sum()),
                             max_face_voxels=max(faces), corr_to_subject_template=round(raw_corr, 4),
                             dice_template_to_mni=round(mni_dice, 4), corr_template_to_mni=round(mni_corr, 4),
                             mni_method=mni_how,
                             seconds=round((time.time() - t0) / len(ids), 1)))
        if o.montage:
            _montage(subject, ids, outs, os.path.join(o.out, 'montage'), _G['mni_on_ref'], mni_dice)
    except Exception as e:
        rows = [dict(subject=subject, image_id=iid, status='error',
                     note=(str(e) or traceback.format_exc())[:200]) for iid, _ in scans]
    return rows


def _montage(subject, ids, outs, mdir, mni_on_ref, mni_dice):
    """Picture per subject: MNI template on the output grid (top row, for comparison),
    then three slices per timepoint."""
    try:
        import matplotlib; matplotlib.use('Agg')
        import matplotlib.pyplot as plt, nibabel as nib
    except Exception:
        return
    os.makedirs(mdir, exist_ok=True)
    fig, ax = plt.subplots(len(ids) + 1, 3, figsize=(9, 3 * (len(ids) + 1)), squeeze=False)
    vols = [('MNI template (target)', mni_on_ref / max(float(mni_on_ref.max()), 1e-8))]
    vols += [(iid, nib.load(outs[iid]).get_fdata()) for iid in ids]
    for r, (iid, a) in enumerate(vols):
        c = [s // 2 for s in a.shape]
        for k, sl in enumerate([a[c[0]], a[:, c[1]], a[:, :, c[2]]]):
            ax[r][k].imshow(sl.T, cmap='gray', origin='lower', vmin=0, vmax=1); ax[r][k].axis('off')
        ax[r][0].set_title(iid, fontsize=9, loc='left')
    fig.suptitle(f'{subject}   brain overlap with MNI (Dice) {mni_dice:.2f}'); fig.tight_layout(); fig.savefig(os.path.join(mdir, f'{subject}.png'), dpi=70)
    plt.close(fig)


# ----------------------------------------------------------------------------
# set-up shared by every worker
# ----------------------------------------------------------------------------
def _init(opt):
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(opt.itk_threads)
    os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
    import ants
    _G['opt'] = opt
    _G['mni_brain'] = ants.image_read(os.path.join(opt.work, '_mni_brain.nii.gz'))
    _G['ref'] = ants.image_read(os.path.join(opt.work, '_reference_grid.nii.gz'))
    _G['mni_on_ref'] = ants.image_read(os.path.join(opt.work, '_mni_on_grid.nii.gz')).numpy()
    _G['mni_mask_on_ref'] = ants.image_read(os.path.join(opt.work, '_mni_mask_on_grid.nii.gz')).numpy() > 0.5


def prepare_template(opt):
    """Brain-extract MNI152 once, build the output grid, and save both for the workers."""
    import ants, numpy as np
    if opt.mni_template and opt.mni_mask:
        mni = ants.image_read(opt.mni_template)
        mask = ants.threshold_image(ants.image_read(opt.mni_mask), 0.5, 1e9, 1, 0)
        src = 'files given on the command line'
    elif opt.mni_template:                       # software testing only
        mni = ants.image_read(opt.mni_template); mask = _brain_mask(mni, opt.brain_extractor)
        src = 'template given, mask estimated (testing only)'
    else:
        # The published MNI152NLin2009cAsym template and ITS OWN brain mask (TemplateFlow).
        # Do not estimate the mask with ANTsPyNet: on the smooth group average it comes out
        # ragged and truncated, which corrupts both the target and the grid centre.
        try:
            import templateflow.api as tflow
        except ImportError:
            sys.exit('ERROR: the templateflow package is required.  Install it first:\n'
                     '       pip install templateflow')
        one = lambda x: str(x[0] if isinstance(x, (list, tuple)) else x)
        mni = ants.image_read(one(tflow.get('MNI152NLin2009cAsym', resolution=1, desc=None,
                                            suffix='T1w', extension='nii.gz')))
        mask = ants.threshold_image(ants.image_read(one(tflow.get(
            'MNI152NLin2009cAsym', resolution=1, desc='brain', suffix='mask', extension='nii.gz'))), 0.5, 1e9, 1, 0)
        src = 'TemplateFlow MNI152NLin2009cAsym, 1 mm, with its published brain mask'
    print('MNI source:', src, flush=True)
    ref, extent = build_reference(mni, mask)
    fov = GRID * SPACING
    print(f'MNI template {mni.shape} spacing {tuple(round(s, 2) for s in mni.spacing)}; '
          f'brain extent {np.round(extent, 0)} mm; output field of view {fov:.0f} mm', flush=True)
    if (extent > fov - 16).any():
        sys.exit('ERROR: the MNI brain does not fit in the output grid with an 8 mm margin; raise --spacing.')
    ants.image_write(mni * mask, os.path.join(opt.work, '_mni_brain.nii.gz'))
    ants.image_write(ref, os.path.join(opt.work, '_reference_grid.nii.gz'))
    ants.image_write(ants.resample_image_to_target(mni * mask, ref, interp_type='linear'),
                     os.path.join(opt.work, '_mni_on_grid.nii.gz'))
    ants.image_write(ants.resample_image_to_target(mask, ref, interp_type='nearestNeighbor'),
                     os.path.join(opt.work, '_mni_mask_on_grid.nii.gz'))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--jobs', type=int, default=1, help='subjects processed in parallel')
    ap.add_argument('--itk-threads', type=int, default=2, help='CPU threads per subject')
    ap.add_argument('--limit-subjects', type=int, default=0, help='process only the first N subjects (testing)')
    ap.add_argument('--subjects', default='', help='comma-separated subject IDs to process')
    ap.add_argument('--mni-transform', default='Affine', choices=['Affine', 'SyN'],
                    help='subject template -> MNI. Affine keeps within-brain shape (atrophy); SyN warps it away.')
    ap.add_argument('--mni-template', default='', help='path to an MNI152 T1 (default: TemplateFlow MNI152NLin2009cAsym)')
    ap.add_argument('--mni-mask', default='', help='path to the brain mask of --mni-template')
    ap.add_argument('--brain-extractor', default='antspynet', choices=['antspynet', 'threshold'])
    ap.add_argument('--spacing', type=float, default=1.6,
                    help='output voxel size in mm; 128 voxels x 1.6 mm = 204.8 mm field of view')
    ap.add_argument('--template-iters', type=int, default=3)
    ap.add_argument('--pct-low', type=float, default=1.0)
    ap.add_argument('--pct-high', type=float, default=99.0)
    ap.add_argument('--max-face-voxels', type=int, default=0, help='brain voxels allowed on a box face before flagging')
    ap.add_argument('--min-template-corr', type=float, default=0.90)
    ap.add_argument('--min-mni-dice', type=float, default=0.85, help='flag below this brain overlap with MNI')
    ap.add_argument('--good-mni-dice', type=float, default=0.92, help='skip the third alignment attempt above this')
    ap.add_argument('--montage', action='store_true', help='write one QC picture per subject')
    ap.add_argument('--overwrite', action='store_true')
    opt = ap.parse_args()
    for k in ('manifest', 'out', 'work', 'mni_template', 'mni_mask'):
        if getattr(opt, k):
            setattr(opt, k, os.path.abspath(os.path.expanduser(getattr(opt, k))))
    os.makedirs(opt.out, exist_ok=True); os.makedirs(opt.work, exist_ok=True)

    by_subject = collections.OrderedDict()
    for r in csv.DictReader(open(opt.manifest)):
        if r.get('status', 'ok') not in ('ok', ''):
            continue
        by_subject.setdefault(r['subject'], []).append((r['image_id'], os.path.expanduser(r['nifti'])))
    jobs = sorted(by_subject.items())
    if opt.subjects:
        keep = set(opt.subjects.split(',')); jobs = [j for j in jobs if j[0] in keep]
    if opt.limit_subjects:
        jobs = jobs[:opt.limit_subjects]
    print(f'subjects {len(jobs)} | scans {sum(len(s) for _, s in jobs)} | MNI transform {opt.mni_transform}', flush=True)

    global SPACING
    SPACING = opt.spacing
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = str(opt.itk_threads)
    prepare_template(opt)

    fields = ['subject', 'image_id', 'status', 'note', 'n_scans_subject', 'brain_voxels', 'max_face_voxels',
              'corr_to_subject_template', 'dice_template_to_mni', 'corr_template_to_mni', 'mni_method', 'seconds']
    log_path = os.path.join(opt.out, 'preprocess_v2_log.csv')
    new = not os.path.exists(log_path)
    counts, done = collections.Counter(), 0
    with open(log_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        if opt.jobs > 1:
            pool = mp.get_context('spawn').Pool(opt.jobs, initializer=_init, initargs=(opt,))
            results = pool.imap_unordered(process_subject, jobs)
        else:
            _init(opt); results = map(process_subject, jobs)
        for rows in results:
            for r in rows:
                counts[r['status']] += 1
                if r['status'] != 'exists':
                    w.writerow({k: r.get(k, '') for k in fields})
            f.flush(); done += 1
            if done % 10 == 0 or done == len(jobs):
                print(f'subjects done {done}/{len(jobs)}  {dict(counts)}', flush=True)
    print('finished:', dict(counts), flush=True)


if __name__ == '__main__':
    main()
