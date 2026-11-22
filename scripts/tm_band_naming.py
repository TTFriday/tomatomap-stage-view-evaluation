"""Translate decade-bin names into defined, released and observed tomato BBCH codes from archived index.csv. Reuse S1/S2 code tables and upper-bound segment rule. Code 50 and principal stages 3/4 are undefined; report observed gaps rather than assuming a continuous BBCH axis."""
import argparse
import collections
import csv
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import tm_bbch_gap as G
from tm_class_stage import seg_of
from tm_config import CFG
WORK = CFG['work_root']
from tm_config import OUTPUT_TABLES
OUT = str(OUTPUT_TABLES)
BANDS = ['bbch_%02d_%02d' % (lo, lo + 9) for lo in range(0, 100, 10)]

def _runs(codes):
    (out, s, prev) = ([], None, None)
    for c in codes:
        if s is None:
            s = prev = c
        elif c == prev + 1:
            prev = c
        else:
            out.append('%d' % s if s == prev else '%d–%d' % (s, prev))
            s = prev = c
    if s is not None:
        out.append('%d' % s if s == prev else '%d–%d' % (s, prev))
    return out

def main():
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument('--index', default=os.path.join(WORK, 'index.csv'))
    ap.add_argument('--out', default='T12_band_naming.csv')
    a = ap.parse_args()
    cnt = collections.Counter()
    with open(a.index, encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            try:
                cnt[int(r['bbch_stage'])] += 1
            except (KeyError, TypeError, ValueError):
                pass
    assert sum(cnt.values()) > 0, 'No usable bbch_stage in %s' % a.index
    rows = []
    for b in BANDS:
        (lo, hi) = (int(x) for x in b.split('_')[1:3])
        band = set(range(lo, hi + 1))
        defined = sorted(band & set(G.S1_CODES))
        released = sorted(band & set(G.S2_CODES))
        with_img = sorted((c for c in released if cnt.get(c, 0) > 0))
        n_img = sum((cnt.get(c, 0) for c in band))
        seg = seg_of(b)
        name = G.S1_STAGE_NAMES.get(lo // 10, '')
        if with_img:
            holes = [c for c in range(with_img[0], with_img[-1] + 1) if c not in with_img]
            lab = '%d–%d' % (with_img[0], with_img[-1])
            if holes:
                lab += ' (missing %s)' % '、'.join(_runs(holes))
        elif defined:
            lab = '%s (no images in band)' % '、'.join(_runs(defined))
        else:
            lab = 'undefined (no tomato principal stage %d)' % (lo // 10)
        honest = released == list(range(lo, hi + 1)) and with_img == released
        rows.append(dict(grid_band=b, band_interval='%d–%d' % (lo, hi), dev_segment=seg, band_intersects_seam='Y' if not (hi < G.SEAM_LO or lo > G.SEAM_HI) else 'n', principal_stage=name or '—', n_defined_codes=len(defined), defined_codes=' '.join(map(str, defined)) or '—', n_released_codes=len(released), released_codes=' '.join(map(str, released)) or '—', n_codes_with_img=len(with_img), codes_with_img=' '.join(map(str, with_img)) or '—', n_images=n_img, recommended_label=lab, label_is_honest='Y' if honest else 'n'))
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, a.out)
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print('===== BBCH band naming (%d decade bands) =====' % len(rows))
    print('  %-12s %-6s %-5s %-26s %6s %s' % ('grid_band', 'segment', 'seam', 'recommended_label', 'images', 'literal_label_valid'))
    for r in rows:
        print('  %-12s %-6s %-5s %-26s %6d %s' % (r['grid_band'], r['dev_segment'], r['band_intersects_seam'], r['recommended_label'], r['n_images'], r['label_is_honest']))
    print('dev_segment uses bin upper bound, so bbch_50_59 is reproductive; band_intersects_seam tests intersection of the named integer interval with 30-50.')
    print('Observed support is reproductive although grid name includes undefined code 50.')
    bad = [r['grid_band'] for r in rows if r['label_is_honest'] == 'n']
    print('Grid names differing from observed window: %d / %d -> %s' % (len(bad), len(rows), '、'.join(bad)))
    print('Total defined codes=%d released codes=%d observed codes=%d' % (sum((r['n_defined_codes'] for r in rows)), sum((r['n_released_codes'] for r in rows)), sum((r['n_codes_with_img'] for r in rows))))
    print('Wrote tables/%s (%d rows)' % (a.out, len(rows)))
if __name__ == '__main__':
    sys.exit(main())
