"""Audit det-tagged BBCH coverage against upstream supplementary Tables S1/S2. S1 defines 67 codes (germination only 00/01/03/05/07/09); S2 lists 50 released codes. Codes 30-50 form a structural seam; minimum adjacent crossing is 51-29=22. Aggregate plant observations by date, retaining mixed-stage day diagnostics. Audit upstream samples.json independently of the exported index."""
import argparse
import os
import sys
import csv
from collections import Counter, defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tm_export import CAMS, SRC, cam_of, date_of, load
from tm_config import OUTPUT_TABLES
OUT = str(OUTPUT_TABLES)
S1_STAGE_NAMES = {0: 'Germination', 1: 'Leaf development', 2: 'Side shoot development', 5: 'Inflorescence emergence', 6: 'Flowering', 7: 'Fruit development', 8: 'Maturity of fruit', 9: 'Senescence'}
S1_CODES = sorted([0, 1, 3, 5, 7, 9] + list(range(10, 20)) + list(range(20, 30)) + list(range(51, 60)) + list(range(60, 70)) + list(range(70, 80)) + list(range(80, 90)) + [97, 99])
S2_CODES = sorted([13, 14, 15, 16, 17, 19] + [20, 21, 22, 23, 27, 28, 29] + [51, 52, 53, 54, 55, 56, 59] + list(range(60, 70)) + list(range(70, 80)) + list(range(80, 90)))
CAM_ORDER = ['cam2_horizontal90', 'cam1_bottom45', 'cam3_tilttop135', 'cam4_vertical180']
(SEAM_LO, SEAM_HI) = (30, 50)
GAP_BIG = SEAM_HI - SEAM_LO + 2

def num(v):
    if isinstance(v, dict):
        v = v.get('$number', v.get('value'))
    if v in (None, ''):
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None

def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(os.path.join(SRC, 'samples.json')):
        raise SystemExit('Upstream read-only samples.json unavailable under %s' % SRC)
    det = load()
    (with_stage, no_stage) = ([], [])
    for s in det:
        (with_stage if num(s.get('bbch_stage')) is not None else no_stage).append(s)
    cnt = Counter((num(s['bbch_stage']) for s in with_stage))
    per_cam_stage = defaultdict(Counter)
    stage_cam = defaultdict(Counter)
    for s in with_stage:
        c = CAMS.get(cam_of(s), 'cam?')
        st = num(s['bbch_stage'])
        per_cam_stage[c][st] += 1
        stage_cam[st][c] += 1
    measured = sorted(cnt)
    print('===== A. Record definition =====')
    print('  det-tagged records: %d' % len(det))
    print('  nonempty bbch_stage: %d' % len(with_stage))
    print('  empty bbch_stage: %d' % len(no_stage))
    print('===== B. Stage coverage and seam =====')
    print('  distinct observed BBCH values: %d, range %d-%d' % (len(measured), measured[0], measured[-1]))
    print('  Table S1 defined codes: %d' % len(S1_CODES))
    print('  Table S2 released codes: %d' % len(S2_CODES))
    print('  observed set equals Table S2: %s' % (set(measured) == set(S2_CODES)))
    print('  S2 codes missing from observations: %s' % sorted(set(S2_CODES) - set(measured)))
    print('  observed codes absent from S2: %s' % sorted(set(measured) - set(S2_CODES)))
    holes = [c for c in range(measured[0], measured[-1] + 1) if c not in cnt]
    print('  Missing stages within %d-%d: %d -> %s' % (measured[0], measured[-1], len(holes), holes))
    print('  Missing codes in seam %d-%d: %d; seam images %d / %d' % (SEAM_LO, SEAM_HI, len([c for c in holes if SEAM_LO <= c <= SEAM_HI]), sum((cnt.get(c, 0) for c in range(SEAM_LO, SEAM_HI + 1))), len(with_stage)))
    print('  Seam images per camera: %s' % {c: sum((v for (st, v) in per_cam_stage[c].items() if SEAM_LO <= st <= SEAM_HI)) for c in CAM_ORDER})
    print('  Images/codes per camera: %s' % {c: (sum(per_cam_stage[c].values()), len(per_cam_stage[c])) for c in CAM_ORDER})
    print('  All camera stage sets equal: %s' % all((set(per_cam_stage[c]) == set(measured) for c in CAM_ORDER)))
    defined_missing = sorted(set(S1_CODES) - set(measured))
    print('  S1-defined codes without released images: %d -> %s' % (len(defined_missing), defined_missing))
    print('===== C. Plant time-series crossings =====')
    by_plant = defaultdict(list)
    for s in with_stage:
        by_plant[num(s.get('plant_id'))].append((date_of(s), num(s['bbch_stage'])))
    (plant_rows, gaps_all, crossing) = ([], [], 0)
    for (pid, recs) in sorted(by_plant.items()):
        day = defaultdict(set)
        for (d, st) in recs:
            day[d].add(st)
        mixed = sum((1 for v in day.values() if len(v) > 1))
        seq = sorted(((d, sorted(v)[0]) for (d, v) in day.items()))
        gaps = [seq[i + 1][1] - seq[i][1] for i in range(len(seq) - 1)]
        mx = max(gaps) if gaps else 0
        big = sum((1 for g in gaps if g >= GAP_BIG))
        if mx >= GAP_BIG:
            crossing += 1
        gaps_all.append(mx)
        plant_rows.append(dict(plant_id=pid, n_img=len(recs), n_days=len(seq), first_date=seq[0][0], last_date=seq[-1][0], first_stage=seq[0][1], last_stage=seq[-1][1], max_adjacent_gap=mx, n_gaps_ge_22=big, crosses_seam='Y' if mx >= GAP_BIG else 'n', n_days_mixed_stage=mixed))
    print('  Unique plants: %d' % len(by_plant))
    print('  Plants crossing seam (max adjacent gap >=%d): %d / %d = %.1f%%' % (GAP_BIG, crossing, len(by_plant), 100.0 * crossing / len(by_plant)))
    print('  Max adjacent gap per plant: mean %.1f median %.1f min %d max %d' % (sum(gaps_all) / len(gaps_all), sorted(gaps_all)[len(gaps_all) // 2], min(gaps_all), max(gaps_all)))
    print('  Mixed-stage plant-days: %d' % sum((r['n_days_mixed_stage'] for r in plant_rows)))
    print('  Plants starting at 13: %d; ending at 89: %d' % (sum((1 for r in plant_rows if r['first_stage'] == 13)), sum((1 for r in plant_rows if r['last_stage'] == 89))))
    os.makedirs(OUT, exist_ok=True)
    p1 = os.path.join(OUT, 'T11_bbch_stage_coverage.csv')
    grid = sorted(set(S1_CODES) | set(S2_CODES) | set(measured) | set(range(SEAM_LO, SEAM_HI + 1)))

    def pstage(code):
        if code not in S1_CODES:
            return 'undefined'
        return S1_STAGE_NAMES.get(code // 10, '')
    with open(p1, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.writer(f)
        wr.writerow(['bbch_stage', 'principal_stage', 'in_table_s1', 'in_table_s2', 'det_images', 'in_seam'] + ['det_%s' % c for c in CAM_ORDER])
        for code in grid:
            wr.writerow([code, pstage(code), 'Y' if code in S1_CODES else 'n', 'Y' if code in S2_CODES else 'n', cnt.get(code, 0), 'Y' if SEAM_LO <= code <= SEAM_HI else 'n'] + [stage_cam[code].get(c, 0) for c in CAM_ORDER])
    p2 = os.path.join(OUT, 'T11_plant_seam_crossings.csv')
    with open(p2, 'w', newline='', encoding='utf-8-sig') as f:
        wr = csv.DictWriter(f, fieldnames=list(plant_rows[0].keys()))
        wr.writeheader()
        wr.writerows(plant_rows)
    print('Wrote T11_bbch_stage_coverage.csv (%d rows including empty codes) and T11_plant_seam_crossings.csv (%d plants)' % (len(grid), len(plant_rows)))
if __name__ == '__main__':
    sys.exit(main())
