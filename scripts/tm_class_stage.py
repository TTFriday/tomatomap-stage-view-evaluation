"""Class-by-BBCH support matrices from archived T5_perclass.csv. A cell is usable when support >=20 by default. Keep all decade bins and separate runs/cameras. Vegetative support <=29; reproductive bins have upper bound >=51 because code 50 and principal stages 3/4 are undefined in modified tomato BBCH. A 50-59 bin contains observed 51-59 codes. Do not connect unsupported class domains into continuous developmental curves."""
import argparse
import csv
import os
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from tm_config import OUTPUT_TABLES
OUT = str(OUTPUT_TABLES)
BUCKET_GRID = ['bbch_%02d_%02d' % (lo, lo + 9) for lo in range(10, 100, 10)]
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

def seg_of(bucket):
    try:
        (lo, hi) = (int(x) for x in str(bucket).split('_')[1:3])
    except (ValueError, IndexError):
        return '?'
    if hi <= 29:
        return 'veg'
    if hi >= 51:
        return 'rep'
    return 'seam' if lo >= 30 else '?'

def crosses(segset):
    return 'Y' if {'veg', 'rep'} <= set(segset) else 'n'

def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None

def main():
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='T5_perclass.csv')
    ap.add_argument('--group-kind', default='bbch_bucket')
    ap.add_argument('--weight', default='last')
    ap.add_argument('--cam', default=None, help='Restrict to one camera; default all, kept separate')
    ap.add_argument('--supp-min', type=int, default=20)
    ap.add_argument('--out', default='T10_class_stage_matrix.csv')
    a = ap.parse_args()
    p = os.path.join(ROOT, 'tables', a.src)
    assert os.path.exists(p), 'Missing archived table %s' % p
    rows = [r for r in csv.DictReader(open(p, encoding='utf-8-sig')) if r.get('group_kind') == a.group_kind and r.get('weight') == a.weight and (not a.cam or r.get('cam') == a.cam)]
    if not rows:
        raise SystemExit('%s contains no group_kind=%s weight=%s cam=%s rows' % (a.src, a.group_kind, a.weight, a.cam or 'ALL'))
    buckets = [b for b in BUCKET_GRID if any((r['group'] == b for r in rows))]
    odd = sorted({r['group'] for r in rows} - set(buckets))
    if odd:
        print('  [Warning] Non-grid bins appended: %s' % '、'.join(odd))
        buckets += odd
    print('  Data-derived bins: %s' % ' '.join(buckets))
    (out, domains) = ([], [])
    for ((run, cam), grp) in sorted(_group(rows).items()):
        by = {(r['group'], r['cls']): r for r in grp}
        cls = sorted({r['cls'] for r in grp})
        for c in cls:
            cells = {}
            for b in buckets:
                r = by.get((b, c))
                (sp, v50, v95) = (fnum(r['support']) if r else None, fnum(r['AP50']) if r else None, fnum(r['AP50_95']) if r else None)
                cells[b] = dict(support=sp, AP50=v50, AP50_95=v95, usable=sp is not None and sp >= a.supp_min)
            ok = [b for b in buckets if cells[b]['usable']]
            have = [b for b in buckets if cells[b]['support'] is not None]
            for b in buckets:
                out.append(dict(run=run, protocol=grp[0]['protocol'], cam=cam, cls=c, bucket=b, support=cells[b]['support'], AP50=cells[b]['AP50'], AP50_95=cells[b]['AP50_95'], usable='Y' if cells[b]['usable'] else 'n' if cells[b]['support'] is not None else 'no_GT', n_usable_bins=len(ok), usable_domain='%s..%s' % (ok[0], ok[-1]) if ok else '', gt_domain='%s..%s' % (have[0], have[-1]) if have else '', dev_segment=seg_of(b)))
            domains.append(dict(run=run, protocol=grp[0]['protocol'], cam=cam, cls=c, n_bin_with_gt=len(have), n_bin_usable=len(ok), usable_domain='%s..%s' % (ok[0], ok[-1]) if ok else 'none (all bins <20)', max_support=max((cells[b]['support'] or 0 for b in buckets)), n_bin_usable_veg=sum((1 for b in ok if seg_of(b) == 'veg')), n_bin_usable_rep=sum((1 for b in ok if seg_of(b) == 'rep')), n_bin_usable_seam=sum((1 for b in ok if seg_of(b) == 'seam')), usable_crosses_seam=crosses([seg_of(b) for b in ok]), gt_crosses_seam=crosses([seg_of(b) for b in have])))
    _w(os.path.join(OUT, a.out), out, ['run', 'protocol', 'cam', 'cls', 'bucket', 'support', 'AP50', 'AP50_95', 'usable', 'n_usable_bins', 'usable_domain', 'gt_domain', 'dev_segment'])
    _w(os.path.join(OUT, a.out.replace('.csv', '_domains.csv')), domains, ['run', 'protocol', 'cam', 'cls', 'n_bin_with_gt', 'n_bin_usable', 'usable_domain', 'max_support', 'n_bin_usable_veg', 'n_bin_usable_rep', 'n_bin_usable_seam', 'usable_crosses_seam', 'gt_crosses_seam'])
    print('  Wrote tables/%s (%d rows) and tables/%s (%d rows)' % (a.out, len(out), a.out.replace('.csv', '_domains.csv'), len(domains)))
    for (run, cam) in sorted({(r['run'], r['cam']) for r in out}):
        print('\n===== %s / %s =====' % (run, cam))
        rs = [r for r in out if r['run'] == run and r['cam'] == cam]
        cls = sorted({r['cls'] for r in rs})
        print('  %-16s %s' % ('class', ' '.join(('%16s' % b[5:] for b in buckets))))
        ix = {(r['cls'], r['bucket']): r for r in rs}
        for c in cls:
            cells = []
            for b in buckets:
                r = ix.get((c, b))
                if not r or r['support'] is None:
                    cells.append('%16s' % '—')
                else:
                    tag = '' if r['usable'] == 'Y' else '*'
                    v = r['AP50'] if r['AP50'] not in (None, '') else '—'
                    cells.append('%16s' % ('%s/%d%s' % (v, int(float(r['support'])), tag)))
            dom = ix.get((c, buckets[0])) or {}
            print('  %-16s %s  [%s]' % (c, ' '.join(cells), dom.get('usable_domain') or ''))
        print('  Cell=AP50/GT support; *=support<%d (excluded from comparable macro AP); dash=no GT' % a.supp_min)

def _group(rows):
    g = {}
    for r in rows:
        g.setdefault((r['run'], r['cam']), []).append(r)
    return g

def _w(path, rows, hdr):
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=hdr, extrasaction='ignore')
        wr.writeheader()
        wr.writerows(rows)
if __name__ == '__main__':
    sys.exit(main())
