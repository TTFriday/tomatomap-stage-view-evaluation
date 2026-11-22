"""Compare protocols on union(test pools) minus union(training lists), with last.pt and full planned epochs. Cam2 test pools contain 1615/1612/2316 images at 60 epochs; the shared subset has 592 images. Unpaired differences 0.6702-0.6140=0.0562 and 0.6702-0.6629=0.0073 are protocol differences, not channel effects. The two-arm 200-epoch subset has 1014 images. Require compatible labels and operating points; early stopping uses validation. Pairing does not establish causal channel decomposition."""
import argparse
import csv
import itertools
import json
import os
import sys
import time
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from tm_config import load_config, read_image_list, relative_path, OUTPUT_TABLES
CFG = load_config()
WORK = CFG['work_root']
LISTS = os.path.join(WORK, 'lists')
RUNS = os.path.join(WORK, 'runs')
OUT_DIR = os.path.join(WORK, 'paired')
CAM = 'cam2_horizontal90'
CLASSES = [c for c in open(os.path.join(WORK, 'classes.txt'), encoding='utf-8').read().split('\n') if c.strip()]
assert len(CLASSES) == 7
DEFAULT_PROTOS = ['random_img', 'per_set', 'per_plant']

def read_list(p):
    fp = p if os.path.isabs(p) else os.path.join(LISTS, p)
    assert os.path.exists(fp), 'Image list not found: %s' % fp
    return set(read_image_list(fp))

def proto_of(run):
    man = json.load(open(os.path.join(run, 'manifest.json'), encoding='utf-8'))
    return man['protocol']

def find_run(proto, model='yolo11s', imgsz=800, epochs=60):
    d = os.path.join(RUNS, proto)
    if not os.path.isdir(d):
        raise SystemExit('Run directory not found: %s' % d)
    hits = [os.path.join(d, n) for n in os.listdir(d) if n.startswith(proto + '__') and 'e%d' % epochs in n and os.path.isdir(os.path.join(d, n, 'weights'))]
    hits = [h for h in hits if 'calib' not in h and 'mini' not in h]
    assert len(hits) == 1, '%s matches %d formal runs: %s' % (proto, len(hits), hits)
    return os.path.abspath(hits[0])

def assert_full_epochs(run, man):
    rs = os.path.join(run, 'results.csv')
    assert os.path.exists(rs), 'Missing results.csv; cannot verify completed epochs'
    n = sum((1 for _ in open(rs, encoding='utf-8', errors='replace'))) - 1
    ep = int(man.get('epochs', n))
    assert n >= ep, '%s results.csv has %d rows (expected at least %d). Early stopping used validation; reject paired evaluation.' % (os.path.basename(run), n, ep)
    return n

def evaluate(run, img_list_path, weights_name, batch, workers):
    man = json.load(open(os.path.join(run, 'manifest.json'), encoding='utf-8'))
    proto = man['protocol']
    imgsz = int(man['imgsz'])
    w = os.path.join(run, 'weights', weights_name)
    assert os.path.exists(w), w
    n_ep = assert_full_epochs(run, man)
    import yaml as _yaml
    from ultralytics import YOLO
    tag = os.path.splitext(os.path.basename(weights_name))[0]
    edir = os.path.join(run, 'eval_paired__%s' % tag)
    os.makedirs(edir, exist_ok=True)
    src = read_image_list(img_list_path)
    gl = os.path.join(edir, os.path.basename(img_list_path))
    open(gl, 'w', encoding='utf-8').write('\n'.join(src) + '\n')
    base = os.path.join(WORK, 'tm_broken') if proto.startswith('broken__') else WORK
    yf = os.path.join(edir, 'paired.yaml')
    _yaml.safe_dump({'path': base.replace('\\', '/'), 'train': gl.replace('\\', '/'), 'val': gl.replace('\\', '/'), 'nc': 7, 'names': {i: c for (i, c) in enumerate(CLASSES)}}, open(yf, 'w', encoding='utf-8'), sort_keys=False, default_flow_style=False, allow_unicode=True)
    print('  [%-12s] epochs=%d imgsz=%d weights=%s n_img=%d' % (proto, n_ep, imgsz, tag, len(src)), flush=True)
    model = YOLO(w)
    t0 = time.time()
    r = model.val(data=yf, imgsz=imgsz, batch=batch, workers=workers, device='0', split='val', plots=False, save_json=False, verbose=False, project=edir, name='metrics', exist_ok=True)
    bb = r.box
    present = [int(c) for c in np.asarray(getattr(bb, 'ap_class_index', []), dtype=int).ravel()]
    a50 = np.asarray(getattr(bb, 'ap50', np.zeros(0)), dtype=float).ravel()
    a95 = np.asarray(getattr(bb, 'ap', np.zeros(0)), dtype=float).ravel()
    (pc, pc95) = ({}, {})
    for (j, i) in enumerate(present):
        nm = CLASSES[i]
        if j < a50.size:
            pc[nm] = round(float(a50[j]), 4)
        if j < a95.size:
            pc95[nm] = round(float(a95[j]), 4)
    print('      mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f GT_classes=%d %.0f s' % (bb.map50, bb.map, bb.mp, bb.mr, len(present), time.time() - t0), flush=True)
    return {'protocol': proto, 'run': os.path.basename(run), 'weights': tag, 'imgsz': imgsz, 'n_img': len(src), 'mAP50': round(float(bb.map50), 4), 'mAP50_95': round(float(bb.map), 4), 'P': round(float(bb.mp), 4), 'R': round(float(bb.mr), 4), 'n_class_gt': len(present), 'seconds': round(time.time() - t0, 1), 'perclass': pc, 'perclass_50_95': pc95}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', action='append', default=None, help='Protocols to compare; repeatable (default %s)' % '/'.join(DEFAULT_PROTOS))
    ap.add_argument('--pool', action='append', default=None, help='Candidate lists, unioned; default each protocol cam2 test list')
    ap.add_argument('--exclude', action='append', default=None, help='Excluded lists, unioned; default each protocol cam2 training list')
    ap.add_argument('--weights', default='last.pt')
    ap.add_argument('--epochs', type=int, default=60, help='Planned epochs used to select run (60 or 200)')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-images', type=int, default=0, help='Truncate for a quick pipeline check only')
    ap.add_argument('--allow-imgsz-diff', action='store_true')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    protos = a.proto or DEFAULT_PROTOS
    pools = a.pool or ['%s__%s__test.txt' % (p, CAM) for p in protos]
    excl = a.exclude or ['%s__%s__train.txt' % (p, CAM) for p in protos]
    pool = set().union(*[read_list(p) for p in pools])
    drop = set().union(*[read_list(e) for e in excl])
    sub = sorted(pool - drop)
    if a.max_images:
        sub = sub[:a.max_images]
    print('[Paired subset] |pool|=%d (%s) - |train|=%d (%s) = %d images' % (len(pool), [os.path.basename(x) for x in pools], len(drop), [os.path.basename(x) for x in excl], len(sub)))
    assert len(sub) < len(pool), 'No images excluded; check --exclude lists'
    assert len(sub) >= 200, 'Paired subset has only %d images; require >=200' % len(sub)
    runs = [find_run(p, epochs=a.epochs) for p in protos]
    mans = [json.load(open(os.path.join(r, 'manifest.json'), encoding='utf-8')) for r in runs]
    broken = {str(m['protocol']).startswith('broken__') for m in mans}
    assert len(broken) == 1, 'Mixed raw/repaired conventions: %s' % broken
    szs = {int(m['imgsz']) for m in mans}
    if len(szs) > 1 and (not a.allow_imgsz_diff):
        raise SystemExit('Different imgsz values %s; use --allow-imgsz-diff only for explicit operating-point comparisons' % szs)
    assert len({str(m['model']) for m in mans}) == 1 or a.allow_imgsz_diff, 'Different model capacities; use --allow-imgsz-diff only for explicit operating-point comparisons'
    os.makedirs(OUT_DIR, exist_ok=True)
    il = os.path.join(OUT_DIR, 'paired_subset__cam2__%d.txt' % len(sub))
    open(il, 'w', encoding='utf-8').write('\n'.join(sub) + '\n')
    res = [evaluate(r, il, a.weights, a.batch, a.workers) for r in runs]
    tag = os.path.splitext(os.path.basename(a.weights))[0]
    out = a.out or os.path.join(OUT_DIR, 'paired__%s__%s.csv' % ('_vs_'.join((p for p in protos)), tag))
    stamp = time.strftime('%Y-%m-%d %H:%M:%S')
    base = res[-1]
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['generated', 'n_img_paired', 'list', 'protocol', 'run', 'weights', 'imgsz', 'mAP50', 'mAP50_95', 'P', 'R', 'seconds', 'delta_vs_' + base['protocol']])
        for r in res:
            w.writerow([stamp, len(sub), relative_path(il), r['protocol'], r['run'], r['weights'], r['imgsz'], r['mAP50'], r['mAP50_95'], r['P'], r['R'], r['seconds'], round(r['mAP50'] - base['mAP50'], 4)])
        cls = sorted(set.intersection(*[set(r['perclass']) for r in res]))
        w.writerow([])
        w.writerow(['# per-class mAP50 on the paired subset (%d images)' % len(sub)] + [''] * 3)
        w.writerow(['protocol'] + cls)
        for r in res:
            w.writerow([r['protocol']] + [r['perclass'].get(c, '') for c in cls])
    print('Wrote %s' % out)
    json.dump({'generated': stamp, 'n_img_paired': len(sub), 'list': relative_path(il), 'weights': tag, 'results': [{k: v for (k, v) in r.items() if k != 'perclass'} for r in res], 'perclass': {r['protocol']: r['perclass'] for r in res}}, open(out.replace('.csv', '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('Paired estimates (%d images unseen in gradient training for all %d arms):' % (len(sub), len(res)))
    for (r, l) in zip(res, itertools.repeat('')):
        print('  %-14s mAP50=%.4f  mAP50-95=%.4f' % (r['protocol'], r['mAP50'], r['mAP50_95']))
    for i in range(len(res) - 1):
        for j in range(i + 1, len(res)):
            print('  Δ %-12s − %-12s = %+.4f mAP50（%+.4f @50-95）' % (res[i]['protocol'], res[j]['protocol'], res[i]['mAP50'] - res[j]['mAP50'], res[i]['mAP50_95'] - res[j]['mAP50_95']))
    print('These are paired point estimates, not causal channel effects. Cluster inference uses archived AP records in tm_boot.py.')
    return 0
if __name__ == '__main__':
    sys.exit(main())
