"""Generate runtime image lists and YOLO dataset YAML from archived split CSVs."""
import argparse
import csv
from pathlib import Path
from tm_config import ROOT, WORK
CAMS = ('cam1_bottom45', 'cam2_horizontal90', 'cam3_tilttop135', 'cam4_vertical180')

def prepare(protocols=None):
    import yaml
    classes = (WORK / 'classes.txt').read_text(encoding='utf-8').splitlines()
    assert len(classes) == 7
    files = sorted((WORK / 'splits').glob('*.csv'))
    if protocols:
        files = [WORK / 'splits' / (name + '.csv') for name in protocols]
    lists = WORK / 'lists'
    lists.mkdir(parents=True, exist_ok=True)
    for split_file in files:
        with split_file.open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        protocol = split_file.stem
        for cam in CAMS:
            for split in ('train', 'val', 'test'):
                images = [(WORK / 'images' / cam / row['name']).as_posix() for row in rows if row['cam'] == cam and row['split'] == split]
                target = lists / ('%s__%s__%s.txt' % (protocol, cam, split))
                target.write_text('\n'.join(images) + ('\n' if images else ''), encoding='utf-8')
        dataset = {'path': WORK.as_posix(), 'train': (lists / (protocol + '__cam2_horizontal90__train.txt')).as_posix(), 'val': (lists / (protocol + '__cam2_horizontal90__val.txt')).as_posix(), 'test': [(lists / (protocol + '__' + cam + '__test.txt')).as_posix() for cam in CAMS if cam != 'cam2_horizontal90'], 'nc': 7, 'names': dict(enumerate(classes))}
        with (WORK / ('tm_' + protocol + '.yaml')).open('w', encoding='utf-8') as stream:
            yaml.safe_dump(dataset, stream, sort_keys=False, allow_unicode=True)
        print('Prepared %s (%d indexed images); image files are not downloaded.' % (protocol, len(rows)))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', action='append', help='Restrict to named protocols; repeatable')
    args = parser.parse_args()
    prepare(args.protocol)
if __name__ == '__main__':
    main()
