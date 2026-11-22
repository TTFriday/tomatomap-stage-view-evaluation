# TomatoMAP stage-view evaluation

Code, partition manifests and archived numerical results for a support-aware
study of viewpoint transfer and localisation across plant developmental
stages, using the public TomatoMAP four-camera RGB tomato imaging benchmark.

The evaluation protocol is fixed: ground-truth support masks, one unchanged
class set inside each declared comparison domain, categorical developmental
metadata from the released BBCH codes, and camera comparisons matched by
acquisition event. Uncertainty is estimated by resampling whole plants.

## What this repository contains

* `scripts/` — class-support and stage matrices, developmental band naming,
  BBCH coverage audit, drift statistics, export, training, evaluation, paired
  comparison, prediction collection and the matched stage/view analysis.
* `tables/` — archived result tables (T3, T4, T5, T6, T10, T11, T12, T16, T17,
  T18) as used in the study.
* `work/tm_yolo/` — class list, image index, split membership lists, the
  paired image lists and the archived prediction statistics (`.json` and
  `.npz`) for the same 1,612 image sets at each of the four cameras.
* `USAGE.txt` — command-by-command notes, expected outputs and interpretation
  limits. Read it before rerunning anything.
* `release_metadata.json` — authors, funding, upstream DOIs, recorded
  dependency versions, initial-weight fingerprints and the archived inputs
  each script expects.

## What is not included

Source photographs, exported label trees, detector checkpoints and pretrained
weights are not redistributed here. Nothing in this repository grants new
photographic rights; consult the upstream record for image licensing.

## Upstream data

* Data descriptor: <https://doi.org/10.1038/s41597-026-06926-9>
* Dataset: <https://doi.org/10.5447/ipk/2025/14> (CC BY 4.0)

Extract the upstream release so that `samples.json` and its image paths stay
intact, then point the package at it either by placing it in `data/TomatoMAP`
or by setting `TOMATOMAP_DATA_ROOT` in your shell. The independent developmental
coverage audit needs that file; every other analysis in the first section below
runs from the archived material in this repository. The upstream input is only
read; generated material goes to `work/tm_yolo`.

## Environment

Python 3.10.9 was the recorded interpreter.

```
python -m venv .venv
python -m pip install -r requirements.txt
```

The recorded GPU build used `torch 2.6.0+cu124` and `torchvision 0.21.0+cu124`;
install those two from <https://download.pytorch.org/whl/cu124> before
`requirements.txt` if you need the same CUDA build. The pins record what was
observed, they are not recommendations for newer releases.

## Analyses that need no images

```
python scripts/tm_class_stage.py
python scripts/tm_band_naming.py
python scripts/tm_drift.py
python scripts/stage_view_analysis.py sensitivity
python scripts/stage_view_analysis.py bootstrap --iters 500 --seed 20261117
```

The first three read archived tables and the image index and write new outputs
under `work/tm_yolo/tables` (the drift schedule goes to `work/tm_yolo/drift_schedule.csv`);
the archived `tables/` directory is never overwritten. Usable class support is
at least 20 boxes by default, and matrices are kept per run and per camera.
`stage_view_analysis.py bootstrap` resamples the same 100 primary test plants
with common draws over the archived library prediction arrays, reports valid
replicate counts, and performs no imputation for absent classes. Set
`TOMATOMAP_DATA_ROOT` or place the extracted release in `data/TomatoMAP` before
running the coverage audit:

```
python scripts/tm_bbch_gap.py
```

This reads `samples.json` directly instead of trusting the exported index, and
no image inference is involved.

## Rebuilding labels, splits and runs

```
python scripts/tm_export.py
python scripts/tm_prepare.py
python scripts/tm_train.py --protocol per_plant --model yolo11s --imgsz 800 --epochs 60 --batch 16 --workers 12 --tag b1
python scripts/tm_eval.py --run work/tm_yolo/runs/per_plant/per_plant__yolo11s__img800__e60__b16__b1 --group camera --weights last.pt
python scripts/tm_eval.py --run work/tm_yolo/runs/per_plant/per_plant__yolo11s__img800__e60__b16__b1 --group bbch_bucket --weights last.pt
```

`tm_export.py` rebuilds labels, index and splits and copies images locally;
`tm_prepare.py` only regenerates runtime configuration from the archived split
CSVs and downloads nothing. Put `yolo11s.pt` and `yolo11n.pt` in `work/weights`
before training and compare the SHA256 values recorded in
`release_metadata.json` if you need identical initialisation. Paired evaluation
needs trained runs for every selected arm at full planned epochs. The
construction script for the archived reverse-stage split lives in the
companion repository rather than being duplicated here.

## Reading the comparisons

Camera `cam2_horizontal90` is the only camera whose images enter training and
validation; the other three cameras are evaluation domains. Overlap between
views of the same acquisition set is a descriptive property of the partition,
not evidence that camera siblings reached training. Reported differences are
comparisons between evaluation protocols, not causal decompositions. Fix the
class set and check support before comparing a group average across domains,
and report observed developmental code gaps rather than reading decade-bin
names as a continuous axis: the modified tomato BBCH schedule defines 67 codes,
50 appear in the released table, and codes 30 to 50 are structurally undefined.
Developmental bands are labels for observed stage regions, not equal intervals.

## Related repository

`tomatomap-benchmark-audit` holds the companion study of annotation convention
integrity and sampling-aware evaluation on the same source, including the
repair comparison and the split construction scripts referenced above.

## Rights and citation

Code copyright is retained by Lifen Tu and Qi Peng. No software licence has
been selected for this archive; dataset, dependency and model licences are
separate. Cite the upstream descriptor and dataset when using these results.

## Authors and funding

Lifen Tu (<https://orcid.org/0000-0002-5774-8138>) and Qi Peng
(<https://orcid.org/0000-0002-5693-0480>), School of Physics and Electronic
Information Engineering, Hubei Engineering University, Xiaogan, China.

Supported by the Natural Science Foundation of Hubei Province (2026AFB666) and
the Hubei Provincial Natural Science Foundation Xiaogan Innovation and
Development Joint Fund Key Project (2026AFC0666).
