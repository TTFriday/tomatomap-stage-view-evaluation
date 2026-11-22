"""Descriptive protocol BBCH distances from archived split CSVs. Training-stage sets use cam2 only; test rows include all views. Distance is minimum absolute distance to a training-stage code. This is support and protocol analysis, not causal channel decomposition."""
import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
import numpy as np
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tm_config import load_config, read_image_list, relative_path, OUTPUT_TABLES
CFG = load_config()
W = CFG['work_root']
CAM = CFG['train_cam']
BUCKETS = [0, 5, 10, 15, 20, 30, 50]

def load_split(proto):
    rows = []
    with open(os.path.join(W, 'splits', proto + '.csv'), encoding='utf-8') as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    protos = sorted((x[:-4] for x in os.listdir(os.path.join(W, 'splits')) if x.endswith('.csv') and x != 'mini.csv'))
    print('=' * 78)
    print('BBCH distances (train=%s)' % CAM)
    print('=' * 78)
    hdr = ('protocol', 'train_n', 'test_n', 'train_stage_span', 'test_unseen_stage_%', 'median_dist', 'p90_dist', 'dist=0_%', 'dist>=20_%')
    out = []
    for p in protos:
        rows = load_split(p)
        tr = [r for r in rows if r['split'] == 'train' and r['cam'] == CAM]
        te = [r for r in rows if r['split'] == 'test']
        tr_st = sorted({int(r['bbch_stage']) for r in tr if r['bbch_stage'].lstrip('-').isdigit()})
        tr_set = set(tr_st)
        te_st = np.array([int(r['bbch_stage']) for r in te if r['bbch_stage'].lstrip('-').isdigit()], dtype=float)
        if len(te_st) == 0 or not tr_st:
            print('  %-13s skipped (missing stage data)' % p)
            continue
        dist = np.array([min((abs(s - x) for x in tr_st)) for s in te_st])
        unseen = 100.0 * np.mean([s not in tr_set for s in te_st.astype(int)])
        span = '%d-%d (%d codes)' % (min(tr_st), max(tr_st), len(tr_st))
        row = (p, len(tr), len(te), span, round(unseen, 1), int(np.median(dist)), int(np.percentile(dist, 90)), round(100.0 * np.mean(dist == 0), 1), round(100.0 * np.mean(dist >= 20), 1))
        out.append(row)
        print('  %-13s train=%6d test=%5d train_stage_span=%-12s unseen_stage=%.1f%%' % row[:5])
        print('                 distance median=%d p90=%d distance=0 %.1f%% distance>=20 %.1f%%' % (row[5], row[6], row[7], row[8]))
    with open(os.path.join(W, 'drift_schedule.csv'), 'w', newline='', encoding='utf-8') as f:
        wr = csv.writer(f)
        wr.writerow(hdr)
        wr.writerows(out)
    print('Wrote %s' % os.path.join(W, 'drift_schedule.csv'))
    print('\n' + '=' * 78)
    print('Test images by minimum absolute stage distance (about 150 images per bin for separate estimates)')
    print('=' * 78)
    for p in protos:
        rows = load_split(p)
        tr = [r for r in rows if r['split'] == 'train' and r['cam'] == CAM]
        te = [r for r in rows if r['split'] == 'test']
        tr_st = sorted({int(r['bbch_stage']) for r in tr if r['bbch_stage'].lstrip('-').isdigit()})
        if not tr_st:
            continue
        c = Counter()
        for r in te:
            s = int(r['bbch_stage'])
            d = min((abs(s - x) for x in tr_st))
            b = next((BUCKETS[i] for i in range(len(BUCKETS) - 1) if BUCKETS[i] <= d < BUCKETS[i + 1]), BUCKETS[-1])
            c[b] += 1
        print('  %-13s %s' % (p, '  '.join(('%d:%d' % (b, c.get(b, 0)) for b in BUCKETS))))
if __name__ == '__main__':
    main()
