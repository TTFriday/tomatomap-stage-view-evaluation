"""Reproducible checks for sampling scope and support-aware view evaluation.

No model is trained here. Input tables and prediction arrays are read only.
Bootstrap resamples whole plants, using the same draw across matched cameras.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMS = ['cam1_bottom45', 'cam2_horizontal90', 'cam3_tilttop135', 'cam4_vertical180']
BUCKETS = ['bbch_10_19', 'bbch_20_29', 'bbch_50_59', 'bbch_60_69', 'bbch_70_79', 'bbch_80_89']
CLASSES = ['leaf', 'batch of flowers', 'batch of fruits', 'whole plant', 'rispe', 'axillary shoot', 'shoot']


def rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def save(path, values):
    if not values:
        raise ValueError('No rows to write')
    with Path(path).open('w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(values[0]))
        w.writeheader()
        w.writerows(values)
    print(Path(path).name, len(values), flush=True)


def fnum(v):
    return float(v) if v not in (None, '', 'None') else float('nan')




def sensitivity(table_dir, output):
    pc = rows(table_dir / 'T5_perclass.csv')
    stage = rows(table_dir / 'T6_stage_curve.csv')
    selected = [r for r in pc if r['weight'] == 'last' and r['group_kind'] == 'bbch_bucket'
                and r['protocol'] in ['random_img', 'per_set', 'per_plant']
                and '__yolo11s__img800__e60__' in r['run']]
    grouped = defaultdict(dict)
    for r in selected:
        grouped[r['protocol']][r['cam'], r['group'], r['cls']] = r
    sens, view = [], []
    for proto, ix in sorted(grouped.items()):
        for threshold in [1, 5, 10, 20, 50]:
            common = [c for c in CLASSES if all(
                fnum(ix.get((cam, b, c), {}).get('support')) >= threshold
                and np.isfinite(fnum(ix.get((cam, b, c), {}).get('AP50')))
                for cam in CAMS for b in BUCKETS)]
            for cam in CAMS:
                vals = [float(np.mean([fnum(ix[cam, b, c]['AP50']) for c in common]))
                        if common else float('nan') for b in BUCKETS]
                sens.append(dict(protocol=proto, camera=cam, threshold=threshold,
                                 common_classes=';'.join(common), n_common=len(common),
                                 vegetative_change=vals[1] - vals[0],
                                 reproductive_change=vals[-1] - vals[2],
                                 nonadjacent_window_change=vals[-1] - vals[0]))
        for cam in CAMS:
            for b in BUCKETS:
                leaf, plant = ix[cam, b, 'leaf'], ix[cam, b, 'whole plant']
                ref = [ix[CAMS[1], b, c] for c in ['leaf', 'whole plant']]
                a50 = np.mean([fnum(leaf['AP50']), fnum(plant['AP50'])])
                a95 = np.mean([fnum(leaf['AP50_95']), fnum(plant['AP50_95'])])
                ref50 = np.mean([fnum(r['AP50']) for r in ref])
                ref95 = np.mean([fnum(r['AP50_95']) for r in ref])
                view.append(dict(protocol=proto, camera=cam, bucket=b,
                                 AP50_common=a50, AP50_95_common=a95,
                                 leaf_AP50=fnum(leaf['AP50']), whole_plant_AP50=fnum(plant['AP50']),
                                 leaf_support=fnum(leaf['support']), whole_plant_support=fnum(plant['support']),
                                 AP50_retention_vs_cam2=a50 / ref50,
                                 AP50_95_retention_vs_cam2=a95 / ref95))
    save(output / 'T16_support_sensitivity.csv', sens)
    save(output / 'T17_stage_view.csv', view)
    for proto in grouped:
        a = [r['common_classes'] for r in sens if r['protocol'] == proto and r['threshold'] in [5, 10, 20]]
        print(proto, 'common class sets at 5/10/20:', sorted(set(a)), flush=True)


def subset_ap(st, ids, wanted=(0, 3)):
    from ultralytics.utils.metrics import ap_per_class
    ids = np.asarray(ids, int)
    if ids.size == 0:
        return np.array([np.nan, np.nan])
    pp = np.concatenate([np.arange(st['off'][i], st['off'][i + 1]) for i in ids])
    gg = np.concatenate([np.arange(st['goff'][i], st['goff'][i + 1]) for i in ids])
    tcl = st['tcls'][gg]
    if not all(np.any(tcl == c) for c in wanted):
        return np.array([np.nan, np.nan])
    if len(pp) == 0:
        return np.zeros(2)
    ans = ap_per_class(st['tp'][pp], st['conf'][pp], st['pred_cls'][pp], tcl,
                       plot=False, names={i: c for i, c in enumerate(CLASSES)})
    ap, ci = ans[5], ans[6].astype(int)
    ap = ap[[list(ci).index(c) for c in wanted]]
    return np.array([ap[:, 0].mean(), ap.mean()])


def bootstrap(work, output, iters, seed):
    index = {r['name']: r for r in rows(work / 'index.csv')}
    stores = []
    for cam in CAMS:
        stem = 'per_set__e60__last__per_set__%s__test' % cam
        path = work / 'boot' / (stem + '.npz')
        st = dict(np.load(path, allow_pickle=False))
        meta = json.loads((work / 'boot' / (stem + '.json')).read_text(encoding='utf-8'))
        assert meta['gate_abs'] < 0.002
        stores.append(st)
    wanted_sets = [index[str(n)]['image_set_id'] for n in stores[1]['im_name']]
    assert len(set(wanted_sets)) == len(wanted_sets)
    for k, st in enumerate(stores):
        positions = {index[str(n)]['image_set_id']: i for i, n in enumerate(st['im_name'])}
        assert set(positions) == set(wanted_sets), CAMS[k]
        st['_aligned'] = np.array([positions[s] for s in wanted_sets])
    plant = np.array([index[str(n)]['plant_id'] for n in stores[1]['im_name']])
    bbch = np.array([int(index[str(n)]['bbch_stage']) for n in stores[1]['im_name']])
    levels = sorted(set(plant))
    clusters = [np.flatnonzero(plant == p) for p in levels]
    masks = {'all': np.ones(len(bbch), bool), 'rep_start': (bbch >= 51) & (bbch <= 59),
             'rep_end': (bbch >= 80) & (bbch <= 89)}
    specs = [(i, 'all') for i in range(4)] + [(1, 'rep_start'), (1, 'rep_end')]
    def evaluate(idx):
        return np.array([subset_ap(stores[i], stores[i]['_aligned'][idx[masks[m][idx]]])
                         for i, m in specs])
    full = evaluate(np.arange(len(bbch)))
    rng = np.random.default_rng(seed)
    samples = []
    for j in range(iters):
        chosen = rng.integers(0, len(clusters), len(clusters))
        idx = np.sort(np.concatenate([clusters[i] for i in chosen]))
        samples.append(evaluate(idx))
        if (j + 1) % 50 == 0:
            print('plant bootstrap', j + 1, '/', iters, flush=True)
    samples = np.asarray(samples)
    out = []
    def record(name, point, dist, unit, n_plant):
        for k, metric in enumerate(['AP50', 'AP50_95']):
            good = dist[:, k][np.isfinite(dist[:, k])]
            out.append(dict(comparison=name, metric=metric, point=float(point[k]),
                            ci_lo=float(np.quantile(good, .025)), ci_hi=float(np.quantile(good, .975)),
                            bootstrap_se=float(good.std(ddof=1)), n_cluster=len(levels),
                            n_contributing_clusters=n_plant,
                            n_valid=len(good), n_resamples=iters, seed=seed,
                            unit=unit, method='paired_plant_percentile_bootstrap'))
    for i in [0, 2, 3]:
        record(CAMS[i] + '_minus_cam2', full[i] - full[1], samples[:, i] - samples[:, 1],
               'plant_matched_by_image_set', len(levels))
        record(CAMS[i] + '_retention_vs_cam2', full[i] / full[1], samples[:, i] / samples[:, 1],
               'plant_matched_by_image_set', len(levels))
    used_plants = set(plant[masks['rep_start']]) | set(plant[masks['rep_end']])
    record('cam2_rep_end_minus_rep_start', full[5] - full[4], samples[:, 5] - samples[:, 4],
           'whole_plant_across_two_reproductive_windows', len(used_plants))
    save(output / 'T18_matched_view_intervals.csv', out)
    metadata = dict(iters=iters, seed=seed, n_images_per_camera=len(bbch),
                    n_plant=len(levels), common_class_ids=[0, 3],
                    window_support={m: dict(n_img=int(mask.sum()), n_plant=len(set(plant[mask])))
                                    for m, mask in masks.items()},
                    sources_sha256={str(p.relative_to(work)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in (work / 'boot').glob('per_set__e60__last__per_set__*.npz')},
                    note='No absent-class imputation. Undefined resample metrics are omitted and n_valid reported. Camera populations are matched by image_set_id, not organ instance identity.')
    (output / 'T18_matched_view_meta.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['sensitivity', 'bootstrap'])
    parser.add_argument('--work', type=Path, default=ROOT / 'work' / 'tm_yolo')
    parser.add_argument('--tables', type=Path, default=ROOT / 'tables')
    parser.add_argument('--output', type=Path, default=ROOT / 'work' / 'tm_yolo' / 'tables')
    parser.add_argument('--iters', type=int, default=500)
    parser.add_argument('--seed', type=int, default=20261117)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'sensitivity':
        sensitivity(args.tables, args.output)
    else:
        bootstrap(args.work, args.output, args.iters, args.seed)


if __name__ == '__main__':
    main()
