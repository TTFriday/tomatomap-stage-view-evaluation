"""Archive per-image tp/conf/pred_cls/target_cls from Ultralytics metrics. Matching, NMS and IoU 0.50:0.95 remain library operations. Recompute with the same ap_per_class and reject disagreement >=0.002 for AP50 or AP50-95. Writes prediction archives separately from evaluation CSVs."""
import argparse
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
sys.path.insert(0, HERE)
import tm_paired as TP
WORK = TP.WORK
BOOT = os.path.join(WORK, 'boot')
CLASSES = TP.CLASSES
IOUV = np.linspace(0.5, 0.95, 10)
GATE = 0.002

def ap_table(tp, conf, pred_cls, target_cls):
    from ultralytics.utils.metrics import ap_per_class
    if len(tp) == 0 or len(target_cls) == 0:
        z = np.zeros((0, 0))
        return (z, z, z)
    res = ap_per_class(tp, conf, pred_cls, target_cls, plot=False, names={i: c for (i, c) in enumerate(CLASSES)}, prefix='')[2:]
    return (res[3], res[4], res[6])

def slices(store, idx):
    return cat_slices([(store, idx)])

def cat_slices(parts):
    (tp, conf, pcl, tcl) = ([], [], [], [])
    for (store, idx) in parts:
        for i in idx:
            i = int(i)
            tp.append(store['tp'][store['off'][i]:store['off'][i + 1]])
            conf.append(store['conf'][store['off'][i]:store['off'][i + 1]])
            pcl.append(store['pred_cls'][store['off'][i]:store['off'][i + 1]])
            tcl.append(store['tcls'][store['goff'][i]:store['goff'][i + 1]])
    if not tp:
        return (np.zeros((0, IOUV.size), bool), np.zeros(0), np.zeros(0), np.zeros(0))
    tp = np.concatenate(tp, 0)
    if tp.ndim == 1:
        tp = tp.reshape(-1, IOUV.size)
    return (tp, np.concatenate(conf, 0), np.concatenate(pcl, 0), np.concatenate(tcl, 0))

def map_from(store, idx):
    (tp, conf, pcl, tcl) = slices(store, idx)
    (all_ap, ci, _) = ap_table(tp, conf, pcl, tcl)
    if all_ap.size == 0:
        return (float('nan'), float('nan'), all_ap, ci)
    return (float(all_ap[:, 0].mean()), float(all_ap.mean()), all_ap, ci)

def collect(run, img_list_path, weights_name, batch, workers, device, extra_tag=''):
    import yaml as _yaml
    from ultralytics import YOLO
    from ultralytics.utils.metrics import DetMetrics
    man = json.load(open(os.path.join(run, 'manifest.json'), encoding='utf-8'))
    proto = man['protocol']
    imgsz = int(man['imgsz'])
    n_ep = TP.assert_full_epochs(run, man)
    w = os.path.join(run, 'weights', weights_name)
    assert os.path.exists(w), w
    src = TP.read_image_list(img_list_path)
    assert src, 'Empty list %s' % img_list_path
    os.makedirs(BOOT, exist_ok=True)
    tag = os.path.splitext(os.path.basename(weights_name))[0]
    edir = os.path.join(run, 'eval_dump__%s' % tag)
    os.makedirs(edir, exist_ok=True)
    gl = os.path.join(edir, os.path.basename(img_list_path))
    open(gl, 'w', encoding='utf-8').write('\n'.join(src) + '\n')
    base = os.path.join(WORK, 'tm_broken') if proto.startswith('broken__') else WORK
    yf = os.path.join(edir, 'dump.yaml')
    _yaml.safe_dump({'path': base.replace('\\', '/'), 'train': gl.replace('\\', '/'), 'val': gl.replace('\\', '/'), 'nc': 7, 'names': {i: c for (i, c) in enumerate(CLASSES)}}, open(yf, 'w', encoding='utf-8'), sort_keys=False, default_flow_style=False, allow_unicode=True)
    recs = []
    orig = DetMetrics.update_stats

    def patched(self, stat):
        recs.append({k: np.asarray(stat[k]) for k in ('tp', 'conf', 'pred_cls', 'target_cls', 'target_img')})
        recs[-1]['im_name'] = str(stat['im_name'])
        return orig(self, stat)
    DetMetrics.update_stats = patched
    t0 = time.time()
    try:
        print('  [%-12s] epochs=%d imgsz=%d weights=%s n_img=%d' % (proto, n_ep, imgsz, tag, len(src)), flush=True)
        r = YOLO(w).val(data=yf, imgsz=imgsz, batch=batch, workers=workers, device=device, split='val', plots=False, save_json=False, verbose=False, project=edir, name='metrics', exist_ok=True)
    finally:
        DetMetrics.update_stats = orig
    secs = round(time.time() - t0, 1)
    assert len(recs) == len(src), 'Collected %d image records for a %d-image list; reject skipped images' % (len(recs), len(src))
    got = {rec['im_name'] for rec in recs}
    want = {os.path.basename(p) for p in src}
    assert got == want, 'Collected image names differ from list: %s' % (got ^ want)
    by_name = {}
    for (i, p) in enumerate(src):
        b = os.path.basename(p)
        assert b not in by_name, 'Duplicate basename %s; paired alignment ambiguous' % b
        by_name[b] = i
    pos = [by_name[rec['im_name']] for rec in recs]
    assert sorted(pos) == list(range(len(src))), 'Image records do not map one-to-one to the list'
    recs = [rec for (_, rec) in sorted(zip(pos, recs), key=lambda t: t[0])]
    store = {'im_name': np.array([rec['im_name'] for rec in recs]), 'tp': np.concatenate([rec['tp'].reshape(-1, IOUV.size) for rec in recs], 0), 'conf': np.concatenate([rec['conf'].ravel() for rec in recs], 0), 'pred_cls': np.concatenate([rec['pred_cls'].ravel() for rec in recs], 0), 'tcls': np.concatenate([rec['target_cls'].ravel() for rec in recs], 0), 'timg': np.concatenate([rec['target_img'].ravel() for rec in recs], 0)}
    (np_, ng_) = ([rec['tp'].shape[0] for rec in recs], [rec['target_cls'].size for rec in recs])
    store['off'] = np.concatenate([[0], np.cumsum(np_)]).astype(np.int64)
    store['goff'] = np.concatenate([[0], np.cumsum(ng_)]).astype(np.int64)
    assert store['tp'].shape[0] == store['conf'].size == store['pred_cls'].size
    assert store['tp'].shape[1] == IOUV.size, 'tp columns %d != IoU grid %d' % (store['tp'].shape[1], IOUV.size)
    assert store['tcls'].size == store['goff'][-1]
    (m50, m95, _, _) = map_from(store, range(len(recs)))
    (lib50, lib95) = (float(r.box.map50), float(r.box.map))
    import ultralytics
    meta = {'generated': time.strftime('%Y-%m-%d %H:%M:%S'), 'proto': proto, 'run': os.path.basename(run), 'weights': tag, 'model': str(man.get('model')), 'imgsz': imgsz, 'epochs_run': n_ep, 'epochs_manifest': int(man.get('epochs', n_ep)), 'list': TP.relative_path(img_list_path), 'list_name': os.path.basename(img_list_path), 'n_img': len(recs), 'n_pred': int(store['conf'].size), 'n_gt': int(store['tcls'].size), 'lib_mAP50': round(lib50, 6), 'lib_mAP50_95': round(lib95, 6), 'dump_mAP50': round(m50, 6), 'dump_mAP50_95': round(m95, 6), 'gate_abs': round(max(abs(m50 - lib50), abs(m95 - lib95)), 6), 'gate': GATE, 'batch': batch, 'device': str(device), 'seconds': secs, 'ultralytics': ultralytics.__version__, 'conf_min_observed': round(float(store['conf'].min()), 6), 'extra_tag': extra_tag}
    stem = '%s__e%d__%s__%s%s' % (proto, int(man.get('epochs', n_ep)), tag, os.path.splitext(os.path.basename(img_list_path))[0], '__' + extra_tag if extra_tag else '')
    npz = os.path.join(BOOT, stem + '.npz')
    print('      mAP50 recomputed=%.6f library=%.6f delta=%.6f | mAP50-95 recomputed=%.6f library=%.6f delta=%.6f (%.1f s)' % (m50, lib50, m50 - lib50, m95, lib95, m95 - lib95, secs), flush=True)
    if meta['gate_abs'] >= GATE:
        raise SystemExit('AP gate failed (delta=%.4f >= %.3f): %s; records and library metrics disagree, reject archive.' % (meta['gate_abs'], GATE, stem))
    np.savez_compressed(npz, **store)
    json.dump(meta, open(os.path.join(BOOT, stem + '.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('      Wrote %s (pred=%d gt=%d)' % (os.path.relpath(npz, ROOT), meta['n_pred'], meta['n_gt']))
    return meta

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--proto', action='append', help='Repeatable protocol (default random_img/per_set/per_plant)')
    ap.add_argument('--epochs', type=int, default=60, help='Planned epochs (60 or 200)')
    ap.add_argument('--model', default='yolo11s')
    ap.add_argument('--imgsz', type=int, default=800)
    ap.add_argument('--weights', default='last.pt')
    ap.add_argument('--list', default=None, help='Image list filename or user path; default protocol cam2 test list')
    ap.add_argument('--img', action='append', help='Debug images relative to work/tm_yolo/images/cam2_horizontal90')
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--device', default='0')
    ap.add_argument('--tag', default='', help='Tag a second archive for the same arm and list')
    a = ap.parse_args()
    protos = a.proto or TP.DEFAULT_PROTOS
    runs = [TP.find_run(p, model=a.model, imgsz=a.imgsz, epochs=a.epochs) for p in protos]
    print('[Prediction archive] arms=%d %s epochs=%d' % (len(runs), protos, a.epochs))
    if a.img:
        os.makedirs(BOOT, exist_ok=True)
        lst = os.path.join(BOOT, '_adhoc__img%d.txt' % len(a.img))
        open(lst, 'w', encoding='utf-8').write('\n'.join((p if os.path.isabs(p) else os.path.join(WORK, 'images', 'cam2_horizontal90', p).replace('\\', '/') for p in a.img)) + '\n')
        list_path = lst
    elif a.list:
        list_path = a.list if os.path.isabs(a.list) else os.path.join(WORK, 'paired', a.list)
        if not os.path.exists(list_path):
            list_path = os.path.join(TP.LISTS, a.list)
        assert os.path.exists(list_path), 'Image list not found: %s' % a.list
    else:
        assert len(protos) == 1, 'Without --list specify exactly one --proto'
        list_path = os.path.join(TP.LISTS, '%s__%s__test.txt' % (protos[0], TP.CAM))
    metas = [collect(r, list_path, a.weights, a.batch, a.workers, a.device, a.tag) for r in runs]
    same = {m['list_name'] for m in metas}
    if len(same) > 1:
        print('Different list names %s: records are not paired' % same)
    print('Wrote %d archives under %s; offline inference: scripts/tm_boot.py (benchmark package)' % (len(metas), BOOT))
if __name__ == '__main__':
    main()
