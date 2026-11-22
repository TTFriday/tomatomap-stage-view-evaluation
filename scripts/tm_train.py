"""Train YOLO11 with fixed augmentation, seed=888 and deterministic=True. Train/validation use cam2_horizontal90 only; cam1/3/4 are evaluation domains. Compare split protocols, not causal leakage channels. Save actual settings and versions in a run manifest."""
import argparse
import json
import os
import platform
import shutil
import sys
import time
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
from tm_config import load_config, read_image_list, relative_path, OUTPUT_TABLES
CFG = load_config()
WORK = CFG['work_root']
RUNS = os.path.join(WORK, 'runs')
AUG = dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=8.0, translate=0.1, scale=0.5, shear=2.0, perspective=0.0008, flipud=0.0, fliplr=0.5, mosaic=1.0, mixup=0.1, copy_paste=0.0, erasing=0.4, close_mosaic=10)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protocol', required=True)
    ap.add_argument('--model', default='yolo11s')
    ap.add_argument('--imgsz', type=int, default=CFG['imgsz'])
    ap.add_argument('--epochs', type=int, default=CFG['epochs'])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--device', default='0')
    ap.add_argument('--cache', default='none', choices=['none', 'ram', 'disk'])
    ap.add_argument('--tag', default='')
    ap.add_argument('--fraction', type=float, default=1.0)
    ap.add_argument('--close_mosaic', type=int, default=None)
    a = ap.parse_args()
    from ultralytics import YOLO
    import ultralytics, torch
    src_w = os.path.join(ROOT, 'work', 'weights', a.model + '.pt')
    name = '%s__%s__img%d__e%d__b%d%s' % (a.protocol, a.model, a.imgsz, a.epochs, a.batch, '__' + a.tag if a.tag else '')
    project = os.path.join(RUNS, a.protocol)
    os.makedirs(project, exist_ok=True)
    outdir = os.path.join(project, name)
    data = os.path.join(WORK, 'tm_%s.yaml' % a.protocol)
    assert os.path.exists(data), 'Missing %s; run tm_prepare.py or tm_export.py first' % data
    assert os.path.exists(src_w), 'Missing user-supplied pretrained weights: %s' % src_w
    print('=' * 70)
    print('Train %s' % name)
    print('  data=%s' % data)
    print('  weights=%s (%d MB)' % (src_w, os.path.getsize(src_w) // 1048576))
    print('  out=%s' % outdir)
    print('=' * 70)
    manifest = dict(name=name, protocol=a.protocol, model=a.model, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, fraction=a.fraction, cache=a.cache, seed=CFG['seed'], train_cam=CFG['train_cam'], aug=AUG, data_yaml=relative_path(data), ultralytics=ultralytics.__version__, torch=torch.__version__, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU', python=platform.python_version(), started=time.strftime('%Y-%m-%d %H:%M:%S'))
    aug = dict(AUG)
    if a.close_mosaic is not None:
        aug['close_mosaic'] = a.close_mosaic
    manifest.update(aug=aug)
    t0 = time.time()
    m = YOLO(src_w)
    res = m.train(data=data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, device=a.device, seed=CFG['seed'], deterministic=True, fraction=a.fraction, cache=False if a.cache == 'none' else a.cache, project=project, name=name, exist_ok=True, patience=max(10, a.epochs // 4), cos_lr=True, pretrained=True, verbose=True, time=None, **aug)
    wall = time.time() - t0
    manifest.update(wall_seconds=round(wall, 1), wall_hours=round(wall / 3600, 3), finished=time.strftime('%Y-%m-%d %H:%M:%S'))
    if a.epochs > 0:
        manifest['sec_per_epoch'] = round(wall / a.epochs, 1)
    try:
        shutil.copy2(src_w, os.path.join(outdir, 'init_weights.pt'))
    except Exception as e:
        print('  (Could not copy initial weights: %s)' % e)
    with open(os.path.join(outdir, 'manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.makedirs(os.path.join(WORK, 'logs'), exist_ok=True)
    with open(os.path.join(WORK, 'logs', 'runs.jsonl'), 'a', encoding='utf-8') as f:
        f.write(json.dumps(manifest, ensure_ascii=False) + '\n')
    print('Finished: %.2f h (%.1f s/epoch) -> %s' % (wall / 3600, wall / max(1, a.epochs), outdir))
    box = getattr(res, 'box', None)
    if box is not None:
        print('cam2 val: mAP50=%.4f  mAP50-95=%.4f  P=%.4f R=%.4f' % (box.map50, box.map, box.mp, box.mr))
    print('Next: python scripts/tm_eval.py --run "%s"' % outdir)
if __name__ == '__main__':
    main()
