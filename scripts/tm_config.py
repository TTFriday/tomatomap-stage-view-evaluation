"""Package-relative configuration and image-list paths."""
import json
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent

def package_path(value):
    """Resolve a relative path against this package, not the working directory."""
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()

def load_config(path=None):
    config_path = package_path(path) if path else ROOT / 'config.json'
    with config_path.open(encoding='utf-8') as stream:
        config = json.load(stream)
    source = os.environ.get('TOMATOMAP_DATA_ROOT', config.get('src_root', 'data/TomatoMAP'))
    config['src_root'] = str(package_path(source))
    work = package_path(config.get('work_root', 'work/tm_yolo'))
    if work != ROOT / 'work' / 'tm_yolo':
        raise ValueError('work_root must be work/tm_yolo within this package')
    config['work_root'] = str(work)
    if config['train_cam'] != 'cam2_horizontal90':
        raise ValueError('The archived protocols train and validate on cam2 only')
    return config
CFG = load_config()
WORK = Path(CFG['work_root'])
OUTPUT_TABLES = WORK / 'tables'
os.environ.setdefault('YOLO_CONFIG_DIR', str(WORK / 'settings'))

def read_image_list(path):
    """Read package-relative archived lists or absolute runtime image lists."""
    with Path(path).open(encoding='utf-8') as stream:
        return [str(package_path(line.strip().replace('\\', '/'))).replace('\\', '/') for line in stream if line.strip()]

def relative_path(path):
    """Represent a local output in machine-neutral metadata."""
    return Path(path).resolve().relative_to(ROOT).as_posix()
