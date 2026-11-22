"""Export det-tagged TomatoMAP annotations and split protocols. Select records by tags, not filename case. Interpret source boxes as top-left [x,y,w,h], write YOLO [x+w/2,y+h/2,w,h], clip centers to [0,1], and retain width/height. Report exceedances under both conventions. Source data are read-only. Seed 888 and ratios 0.70/0.20/0.10 define these protocols, not the upstream split. Training/validation use cam2 only. Full-view membership overlap is descriptive, not experimentally effective camera-sibling leakage or causal channel decomposition."""
import argparse
import csv
import json
import os
import random
import shutil
import sys
import time
import zlib
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
_HERE = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(_HERE, 'config.json')
if not os.path.exists(CFG_PATH):
    CFG_PATH = os.path.join(os.path.dirname(_HERE), 'config.json')
from tm_config import load_config
CFG = load_config(CFG_PATH)
SRC = CFG['src_root']
WORK = CFG['work_root']
CLASSES = ['leaf', 'batch of flowers', 'batch of fruits', 'whole plant', 'rispe', 'axillary shoot', 'shoot']
CLS_IDX = {c: i for (i, c) in enumerate(CLASSES)}
CAMS = {'1': 'cam1_bottom45', '2': 'cam2_horizontal90', '3': 'cam3_tilttop135', '4': 'cam4_vertical180'}
SEED = 888
RATIO = (0.7, 0.2, 0.1)
IMG_DIR = os.path.join(WORK, 'images')
LBL_DIR = os.path.join(WORK, 'labels')
SPLIT_DIR = os.path.join(WORK, 'splits')
LIST_DIR = os.path.join(WORK, 'lists')

def dets_of(s):
    g = s.get('ground_truth')
    return g.get('detections') or [] if isinstance(g, dict) else []

def label_of(det):
    a = det.get('label')
    return a.get('value') if isinstance(a, dict) else a

def cam_of(s):
    v = s.get('camera_angle_deg')
    if isinstance(v, dict):
        v = v.get('$number', v.get('value'))
    return str(int(v)) if v is not None else '?'

def date_of(s):
    return (s.get('capture_datetime') or {}).get('$date', '')[:10]

def bb_of(det):
    return det.get('bounding_box') or []

def load():
    p = os.path.join(SRC, 'samples.json')
    print('[1] Read samples.json (%.0f MB) ...' % (os.path.getsize(p) / 1024 ** 2))
    t0 = time.time()
    D = json.loads(open(p, encoding='utf-8').read())['samples']
    det = [s for s in D if 'det' in (s.get('tags') or [])]
    print('    samples=%d det-tagged=%d elapsed=%.1f s' % (len(D), len(det), time.time() - t0))
    return det

def build_records(det):
    recs = []
    n_bad_cls = 0
    for s in det:
        fp = s['filepath'].replace('\\', '/')
        stem = os.path.basename(fp)[:-4]
        cam = cam_of(s)
        bbch = s.get('bbch_stage')
        if isinstance(bbch, dict):
            bbch = bbch.get('$number', bbch.get('value'))
        boxes = []
        for d in dets_of(s):
            lb = label_of(d)
            if lb not in CLS_IDX:
                n_bad_cls += 1
                continue
            b = bb_of(d)
            if len(b) < 4:
                continue
            boxes.append((CLS_IDX[lb], float(b[0]), float(b[1]), float(b[2]), float(b[3])))
        recs.append({'src': fp, 'name': stem + '.jpg', 'cam': cam, 'camdir': CAMS.get(cam, 'cam?'), 'plant': s.get('plant_id'), 'set': s.get('image_set_id'), 'pose': s.get('pose_id'), 'date': date_of(s), 'bbch': int(bbch) if bbch is not None else -1, 'boxes': boxes})
    print('[2] records=%d discarded unknown-class boxes=%d' % (len(recs), n_bad_cls))
    return recs

def check_semantics(recs):

    def stat(mode):
        over_l = over_t = over_r = over_b = tot = 0
        for r in recs:
            for (ci, a, b_, w, h) in r['boxes']:
                tot += 1
                if mode == 'as_center':
                    (x1, y1) = (a - w / 2.0, b_ - h / 2.0)
                else:
                    (x1, y1) = (a, b_)
                (x2, y2) = (x1 + w, y1 + h)
                over_l += x1 < -1e-09
                over_t += y1 < -1e-09
                over_r += x2 > 1 + 1e-09
                over_b += y2 > 1 + 1e-09
        return (tot, over_l, over_t, over_r, over_b)
    for mode in ('as_center', 'as_topleft'):
        (tot, l, t, rr, b) = stat(mode)
        print('    %-10s boxes=%d outside left=%.2f%% top=%.2f%% right=%.2f%% bottom=%.2f%%' % (mode, tot, 100 * l / tot, 100 * t / tot, 100 * rr / tot, 100 * b / tot))
    return True

def write_labels(recs):
    print('[3] Write repaired YOLO labels ...')
    t0 = time.time()
    n_box = 0
    clip_hi = clip_lo = 0
    for camdir in sorted({r['camdir'] for r in recs}):
        os.makedirs(os.path.join(LBL_DIR, camdir), exist_ok=True)
        os.makedirs(os.path.join(IMG_DIR, camdir), exist_ok=True)
    for r in recs:
        lines = []
        for (ci, x, y, w, h) in r['boxes']:
            xc = x + w / 2.0
            yc = y + h / 2.0
            if xc - w / 2.0 < -1e-06 or yc - h / 2.0 < -1e-06:
                clip_lo += 1
            if xc + w / 2.0 > 1 + 1e-06 or yc + h / 2.0 > 1 + 1e-06:
                clip_hi += 1
            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            lines.append('%d %.6f %.6f %.6f %.6f' % (ci, xc, yc, w, h))
        with open(os.path.join(LBL_DIR, r['camdir'], r['name'][:-4] + '.txt'), 'w') as f:
            f.write('\n'.join(lines))
            if lines:
                f.write('\n')
        n_box += len(lines)
    print('    label files=%d boxes=%d elapsed=%.1f s' % (len(recs), n_box, time.time() - t0))
    print('    Remaining left/top exceedances=%d right/bottom=%d; retain these counts.' % (clip_lo, clip_hi))
    with open(os.path.join(WORK, 'classes.txt'), 'w') as f:
        f.write('\n'.join(CLASSES) + '\n')
    return n_box

def write_index(recs):
    p = os.path.join(WORK, 'index.csv')
    with open(p, 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(['name', 'cam', 'camdir', 'plant_id', 'image_set_id', 'pose_id', 'date', 'bbch_stage', 'n_boxes'])
        for r in recs:
            wr.writerow([r['name'], r['cam'], r['camdir'], r['plant'], r['set'], r['pose'], r['date'], r['bbch'], len(r['boxes'])])
    print('[4] index.csv: %d rows' % len(recs))

def assign(units, ratio):
    keys = sorted(units)
    random.Random(SEED).shuffle(keys)
    n = len(keys)
    a = int(n * ratio[0])
    b = int(n * ratio[0] + n * ratio[1])
    sp = {}
    for (i, k) in enumerate(keys):
        sp[k] = 'train' if i < a else 'val' if i < b else 'test'
    return sp

def build_splits(recs):
    print('[5] Generate split protocols ...')
    (by_plant, by_set, by_date) = ({}, {}, {})
    for r in recs:
        by_plant.setdefault(r['plant'], []).append(r)
        by_set.setdefault(r['set'], []).append(r)
        by_date.setdefault(r['date'], []).append(r)
    prot = {}
    names = [r['name'] for r in recs]
    random.Random(SEED).shuffle(names)
    n = len(names)
    (a, b) = (int(n * RATIO[0]), int(n * RATIO[0] + n * RATIO[1]))
    sp = {x: 'train' for x in names[:a]}
    sp.update({x: 'val' for x in names[a:b]})
    sp.update({x: 'test' for x in names[b:]})
    prot['random_img'] = sp
    sp = assign(list(by_set), RATIO)
    prot['per_set'] = {r['name']: sp[r['set']] for x in sp for r in by_set[x]}
    sp = assign(list(by_plant), RATIO)
    prot['per_plant'] = {r['name']: sp[r['plant']] for x in sp for r in by_plant[x]}
    dates = sorted(by_date)
    (a, b) = (int(len(dates) * RATIO[0]), int(len(dates) * (RATIO[0] + RATIO[1])))
    sp = {d: 'train' if i < a else 'val' if i < b else 'test' for (i, d) in enumerate(dates)}
    prot['temporal_fwd'] = {r['name']: sp[r['date']] for d in dates for r in by_date[d]}
    sp = {d: 'train' if i < a else 'val' if i < b else 'test' for (i, d) in enumerate(sorted(dates, reverse=True))}
    prot['temporal_rev'] = {r['name']: sp[r['date']] for d in dates for r in by_date[d]}
    med = {p: int(np.median([r['bbch'] for r in rs])) for (p, rs) in by_plant.items()}
    pk = sorted(by_plant, key=lambda p: (med[p], p))
    random.Random(SEED).shuffle(pk)
    blocks = [pk[i::3] for i in range(3)]
    sp = {}
    for blk in blocks:
        sub = assign(blk, RATIO)
        for (p, s_) in sub.items():
            sp[p] = s_
    prot['stage_strat'] = {r['name']: sp[r['plant']] for p in by_plant for r in by_plant[p]}
    units = {}
    for r in recs:
        units.setdefault((r['plant'], r['date']), []).append(r)

    def unit_sort_key(k):
        stag = units[k][0]['bbch']
        salt = zlib.crc32(('%s|%s|%s' % (SEED, k[0], k[1])).encode()) % 1000003
        return (stag, salt, str(k[0]), str(k[1]))
    uk = sorted(units, key=unit_sort_key)
    (a, b) = (int(len(uk) * RATIO[0]), int(len(uk) * (RATIO[0] + RATIO[1])))
    sp = {}
    for (i, k) in enumerate(uk):
        s_ = 'train' if i < a else 'val' if i < b else 'test'
        sp.update({r['name']: s_ for r in units[k]})
    prot['stage_holdout'] = sp
    for (pname, sp) in prot.items():
        with open(os.path.join(SPLIT_DIR, pname + '.csv'), 'w', newline='', encoding='utf-8') as f:
            wr = csv.writer(f)
            wr.writerow(['name', 'cam', 'plant_id', 'image_set_id', 'date', 'bbch_stage', 'split'])
            for r in recs:
                wr.writerow([r['name'], r['camdir'], r['plant'], r['set'], r['date'], r['bbch'], sp[r['name']]])
        cnt = {}
        for r in recs:
            cnt[sp[r['name']]] = cnt.get(sp[r['name']], 0) + 1
        print('    %-13s %s' % (pname, cnt))
    return prot

def leak_report(recs, prot):
    print('[6] Full-view train/test membership overlap (descriptive, not effective camera leakage)')
    by_set_cache = {}
    for r in recs:
        by_set_cache.setdefault(r['set'], []).append(r)
    rows = []
    for (pname, sp) in prot.items():
        tr = [r for r in recs if sp[r['name']] == 'train']
        te = [r for r in recs if sp[r['name']] == 'test']
        va = [r for r in recs if sp[r['name']] == 'val']
        tr_names = {r['name'] for r in tr}
        pl_tr = {r['plant'] for r in tr}
        se_tr = {r['set'] for r in tr}
        da_tr = {r['date'] for r in tr}
        st_tr = {r['bbch'] for r in tr}
        pct_plant = 100.0 * sum((r['plant'] in pl_tr for r in te)) / max(1, len(te))
        pct_set = 100.0 * sum((r['set'] in se_tr for r in te)) / max(1, len(te))
        near = sum((any((q['name'] in tr_names for q in by_set_cache[r['set']] if q['name'] != r['name'])) for r in te))
        shared_days = len(da_tr & {r['date'] for r in te})
        shared_bbch = len(st_tr & {r['bbch'] for r in te})
        pct_novel_stage = 100.0 * sum((r['bbch'] not in st_tr for r in te)) / max(1, len(te))

        def prior(rs):
            c = np.zeros(len(CLASSES))
            for r in rs:
                for (ci, *_rest) in r['boxes']:
                    c[ci] += 1
            return c / max(1.0, c.sum())
        tv = 0.5 * float(np.abs(prior(tr) - prior(te)).sum())
        st_a = np.array([r['bbch'] for r in tr if r['bbch'] >= 0], dtype=float)
        st_b = np.array([r['bbch'] for r in te if r['bbch'] >= 0], dtype=float)
        q = np.linspace(0.05, 0.95, 19)
        w1 = float(np.mean(np.abs(np.quantile(st_a, q) - np.quantile(st_b, q)))) if len(st_a) and len(st_b) else float('nan')
        rows.append([pname, len(tr), len(va), len(te), round(pct_plant, 1), round(pct_set, 1), round(100.0 * near / max(1, len(te)), 1), shared_days, shared_bbch, round(pct_novel_stage, 1), round(tv, 4), round(w1, 2), '%.1f/%.1f' % (float(np.median(st_a)), float(np.median(st_b)))])
    hdr = ['protocol', 'n_train', 'n_val', 'n_test', '%test_plant_in_train', '%test_set_in_train', '%test_nearcopy_in_train', 'shared_days', 'shared_bbch', '%test_stage_unseen_in_train', 'class_prior_TV', 'bbch_W1', 'median_bbch_train/test']
    with open(os.path.join(WORK, 'leakage.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(hdr)
        wr.writerows(rows)
    print('    ' + ' | '.join(hdr))
    for row in rows:
        print('    ' + ' | '.join((str(x) for x in row)))

def copy_images(recs):
    print('[7] Copy images to local work; source data remain read-only')
    t0 = time.time()
    done = skipped = 0
    for (i, r) in enumerate(recs):
        dst = os.path.join(IMG_DIR, r['camdir'], r['name'])
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(os.path.join(SRC, r['src'])):
            skipped += 1
        else:
            shutil.copy2(os.path.join(SRC, r['src']), dst)
            done += 1
        if (i + 1) % 5000 == 0:
            el = time.time() - t0
            print('    %5d/%d copied=%d skipped=%d %.0f img/s remaining=%.1f min' % (i + 1, len(recs), done, skipped, (i + 1) / el, (len(recs) - i - 1) / ((i + 1) / el) / 60), flush=True)
    print('    copied=%d skipped=%d elapsed=%.1f min' % (done, skipped, (time.time() - t0) / 60))

def write_yaml(prot):
    print('[8] Write runtime dataset YAML and image lists ...')
    all_cams = ('cam1_bottom45', 'cam2_horizontal90', 'cam3_tilttop135', 'cam4_vertical180')
    for (pname, sp) in prot.items():
        for camdir in all_cams:
            for split in ('train', 'val', 'test'):
                lines = [os.path.join(IMG_DIR, camdir, r['name']).replace('\\', '/') for r in REC_CACHE if r['camdir'] == camdir and sp[r['name']] == split]
                p = os.path.join(LIST_DIR, '%s__%s__%s.txt' % (pname, camdir, split))
                open(p, 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        lines = ['path: ' + WORK.replace('\\', '/'), 'train: ' + (LIST_DIR + '/%s__cam2_horizontal90__train.txt' % pname).replace('\\', '/'), 'val: ' + (LIST_DIR + '/%s__cam2_horizontal90__val.txt' % pname).replace('\\', '/'), 'nc: %d' % len(CLASSES), 'test:']
        for cd in ('cam1_bottom45', 'cam3_tilttop135', 'cam4_vertical180'):
            lines.append('  - ' + (LIST_DIR + '/%s__%s__test.txt' % (pname, cd)).replace('\\', '/'))
        lines.append('names:')
        for (i, c) in enumerate(CLASSES):
            lines.append('  %d: %s' % (i, c))
        open(os.path.join(WORK, 'tm_%s.yaml' % pname), 'w').write('\n'.join(lines) + '\n')
    print('    dataset YAML files=%d image lists=%d' % (len(prot), len(prot) * 4 * 3))
REC_CACHE = []

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skip-copy', action='store_true', help='Export labels and splits without copying images')
    skip_copy = parser.parse_args().skip_copy
    for d in (IMG_DIR, LBL_DIR, SPLIT_DIR, LIST_DIR):
        os.makedirs(d, exist_ok=True)
    det = load()
    recs = build_records(det)
    global REC_CACHE
    REC_CACHE = recs
    check_semantics(recs)
    write_labels(recs)
    write_index(recs)
    prot = build_splits(recs)
    leak_report(recs, prot)
    write_yaml(prot)
    if not skip_copy:
        copy_images(recs)
    print('Completed -> %s' % WORK)
if __name__ == '__main__':
    main()
