"""Group evaluation with Ultralytics val, without changing AP definitions. AP50=bb.ap50; AP50_95=bb.ap. bb.maps fills missing-GT classes with group mAP50-95. Group mAP averages only GT-present classes; across-group comparisons require common class sets and support checks. Keep best/last checkpoint choices separate."""
import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
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
CLASSES = [c for c in open(os.path.join(WORK, 'classes.txt'), encoding='utf-8').read().split('\n') if c.strip()]
assert len(CLASSES) == 7, 'classes.txt has %d classes: %r' % (len(CLASSES), CLASSES)

def load_index():
    with open(os.path.join(WORK, 'index.csv'), encoding='utf-8') as f:
        return {row['name']: row for row in csv.DictReader(f)}

def group_key(row, group, tr_med):
    st = int(row['bbch_stage'])
    if group == 'bbch_tertile':
        return 'early' if st < 55 else 'mid' if st < 75 else 'late'
    if group == 'bbch_stage':
        return 'bbch_%02d' % st
    if group == 'bbch_bucket':
        for lo in range(10, 90, 10):
            if lo <= st < lo + 10:
                return 'bbch_%02d_%02d' % (lo, lo + 9)
        return 'bbch_other'
    if group == 'stage_offset':
        return 'off_%02d' % int(round(abs(st - tr_med) / 10.0) * 10)
    raise ValueError(group)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--group', default='camera', choices=['camera', 'bbch_tertile', 'bbch_stage', 'bbch_bucket', 'stage_offset'])
    ap.add_argument('--cams', default='cam2_horizontal90,cam1_bottom45,cam3_tilttop135,cam4_vertical180')
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--weights', default='best.pt', help='Checkpoint filename under run/weights (best.pt or last.pt)')
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    man = json.load(open(os.path.join(run, 'manifest.json'), encoding='utf-8'))
    proto = man['protocol']
    imgsz = man['imgsz']
    weights = a.weights if os.path.isabs(a.weights) else os.path.join(run, 'weights', a.weights)
    assert os.path.exists(weights), weights
    broken = proto.startswith('broken__')
    real = proto[len('broken__'):] if broken else proto
    base = os.path.join(WORK, 'tm_broken') if broken else WORK
    lists_dir = os.path.join(base, 'lists')
    from ultralytics import YOLO
    import yaml as _yaml
    model = YOLO(weights)
    idx = load_index()
    with open(os.path.join(WORK, 'splits', real + '.csv'), encoding='utf-8') as f:
        tr_st = [int(r['bbch_stage']) for r in csv.DictReader(f) if r['split'] == 'train']
    tr_med = float(np.median(tr_st)) if tr_st else float('nan')
    wtag = os.path.splitext(os.path.basename(weights))[0]
    print('protocol=%s train_camera=%s imgsz=%d median_train_BBCH=%.1f group=%s weights=%s' % (proto, CFG['train_cam'], imgsz, tr_med, a.group, wtag), flush=True)
    edir = os.path.join(run, 'eval_%s__%s' % (a.group, wtag))
    os.makedirs(edir, exist_ok=True)
    (rows, perclass, perclass2) = ([], {}, {})
    for cam in [c.strip() for c in a.cams.split(',') if c.strip()]:
        lp = os.path.join(lists_dir, '%s__%s__test.txt' % (real, cam))
        if not os.path.exists(lp):
            print('  Skip %s: missing list %s' % (cam, lp), flush=True)
            continue
        paths = read_image_list(lp)
        if not paths:
            continue
        if a.group == 'camera':
            groups = {'all': paths}
        else:
            g = defaultdict(list)
            for p in paths:
                row = idx.get(os.path.basename(p))
                if row is not None:
                    g[group_key(row, a.group, tr_med)].append(p)
            groups = dict(g)
        for (gname, gp) in sorted(groups.items()):
            if len(gp) < 20:
                print('  Skip %s/%s: only %d images' % (cam, gname, len(gp)), flush=True)
                continue
            gl = os.path.join(edir, '%s__%s.txt' % (cam, gname))
            open(gl, 'w', encoding='utf-8').write('\n'.join(gp) + '\n')
            yf = os.path.join(edir, '%s__%s.yaml' % (cam, gname))
            _yaml.safe_dump({'path': base.replace('\\', '/'), 'train': gl.replace('\\', '/'), 'val': gl.replace('\\', '/'), 'nc': 7, 'names': {i: c for (i, c) in enumerate(CLASSES)}}, open(yf, 'w', encoding='utf-8'), sort_keys=False, default_flow_style=False, allow_unicode=True)
            t0 = time.time()
            r = model.val(data=yf, imgsz=imgsz, batch=a.batch, workers=a.workers, device='0', split='val', plots=False, save_json=False, verbose=False, project=edir, name='%s__%s' % (cam, gname), exist_ok=True)
            bb = r.box
            names = getattr(bb, 'names', None) or model.names
            pc = {}
            for (k, v) in zip(sorted(names), np.asarray(bb.maps, dtype=float)):
                if v == v:
                    pc[str(names[k])] = round(float(v), 4)
            present = [int(c) for c in np.asarray(getattr(bb, 'ap_class_index', []), dtype=int).ravel()]
            a50 = np.asarray(getattr(bb, 'ap50', np.zeros(0)), dtype=float).ravel()
            a95 = np.asarray(getattr(bb, 'ap', np.zeros(0)), dtype=float).ravel()
            nt = None
            for src in (bb, r):
                v = getattr(src, 'nt_per_class', None)
                if v is not None:
                    nt = np.asarray(v, dtype=float).ravel()
                    break
            (pc50, pc95, sup) = ({}, {}, {})
            for (j, i) in enumerate(present):
                nm = CLASSES[i]
                if j < a50.size:
                    pc50[nm] = round(float(a50[j]), 4)
                if j < a95.size:
                    pc95[nm] = round(float(a95[j]), 4)
                if nt is not None and nt.size == len(CLASSES):
                    sup[nm] = int(nt[i])
                elif nt is not None and nt.size == len(present):
                    sup[nm] = int(nt[j])
            perclass2['%s|%s' % (cam, gname)] = {'AP50': pc50, 'AP50_95': pc95, 'support': sup, 'gt_classes': [CLASSES[i] for i in present], 'n_class_with_gt': len(present), 'mAP50': round(float(bb.map50), 4), 'mAP50_95': round(float(bb.map), 4)}
            rec = dict(protocol=proto, cam=cam, group=gname, n_img=len(gp), mAP50=round(float(bb.map50), 4), mAP50_95=round(float(bb.map), 4), P=round(float(bb.mp), 4), R=round(float(bb.mr), 4), n_class=len(pc), n_class_gt=len(present), seconds=round(time.time() - t0, 1))
            perclass['%s|%s' % (cam, gname)] = pc
            extra = ''
            rows.append(rec)
            print('=> %-16s %-12s n=%5d  mAP50=%.4f  mAP50-95=%.4f  (%.0fs)%s' % (cam, gname, len(gp), bb.map50, bb.map, rec['seconds'], extra), flush=True)
    keys = ['protocol', 'cam', 'group', 'n_img', 'mAP50', 'mAP50_95', 'P', 'R', 'n_class', 'n_class_gt', 'seconds']
    out = os.path.join(run, 'eval_%s__%s.csv' % (a.group, wtag))
    with open(out, 'w', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
    pcj = os.path.join(run, 'eval_%s__%s_perclass.json' % (a.group, wtag))
    json.dump(perclass, open(pcj, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    pc2 = os.path.join(run, 'eval_%s__%s_perclass2.json' % (a.group, wtag))
    json.dump({'schema': 2, 'weights': wtag, 'group': a.group, 'note': 'AP50=bb.ap50 (per-class AP at IoU 0.5); AP50_95=bb.ap (mean at IoUs 0.50:0.95); support=GT counts when available. Group mAP50 averages classes with GT; fix the class set before comparing groups.', 'groups': perclass2}, open(pc2, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('Wrote %s (%d rows) + %s + %s' % (out, len(rows), os.path.basename(pcj), os.path.basename(pc2)))
if __name__ == '__main__':
    main()
