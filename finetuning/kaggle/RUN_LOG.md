# Kaggle Run Log

Running record of every `01_dataset_prep_and_eda.ipynb` (and later, `02_train.ipynb` /
`03_evaluate_finetune_checks.ipynb`) execution attempt on Kaggle — what changed,
what happened, and the exact failure signal, so past attempts don't have to be
re-diagnosed from memory. Newest entry on top. Updated after every run, not just
the ones that worked.

---

## Notebook 1 — `01_dataset_prep_and_eda.ipynb`

### Attempt 4 — kernel version 4 — 2026-11-05 (in progress)

- **Trigger:** `kaggle kernels push` after fixing attempt 3's redirect (see
  below) to actually take effect, not just update the settings dict.
- **Status:** pending push + run.

### Attempt 3 — kernel version 3 — 2026-11-05

- **Trigger:** `kaggle kernels push` after redirecting `datasets_dir` to
  `/kaggle/tmp/datasets` via `settings.update(datasets_dir=...)`.
- **Duration:** ~110 seconds of actual notebook execution (status polling
  showed ~8 min wall-clock including queue/startup overhead).
- **Status:** `KernelWorkerStatus.ERROR`.
- **Key error:** identical to attempt 2 — `RuntimeError: Dataset 'coco.yaml'
  error ❌ Insufficient free disk space 18.673 GB < 27.013 GB required`.
- **Root cause:** `settings.update(datasets_dir=...)` only rewrites the
  persisted `settings.json` file and the live settings dict. By that point
  `ultralytics` had already been imported once (cell 1's `ultralytics.checks()`),
  which already executed `DATASETS_DIR = Path(SETTINGS["datasets_dir"])` as a
  frozen module-level constant in `ultralytics.utils` — and
  `ultralytics/data/utils.py` (where `check_det_dataset()` actually resolves
  download paths) has its **own** separately-bound copy via `from
  ultralytics.utils import DATASETS_DIR`. Neither already-bound name gets
  reassigned by updating the settings dict afterward. Confirmed directly in
  the pulled log: `settings["datasets_dir"]` printed the correct
  `/kaggle/tmp/datasets`, `df -h` showed 967GB free on `/` (plenty), but the
  actual download still targeted `/kaggle/working/datasets/coco2017labels-segments.zip`
  — the stale path.
- **Fix applied:** patch `settings.json` on disk directly, then delete every
  `ultralytics*` entry from `sys.modules` and force a clean re-import so
  `DATASETS_DIR` gets recomputed everywhere it was bound — not just in the
  settings dict. Added an assertion that fails immediately if this doesn't
  work, instead of discovering it ~2 minutes into another download attempt
  (→ attempt 4).
- **Output pulled:** `kaggle kernels output` — log showed `Setup complete ✅
  (4 CPUs, 31.3 GB RAM, 7190.0/8156.9 GB disk)` and `df -h`: `overlay 8.0T
  7.1T 967G 89% /` vs `/dev/loop2 20G 136K 20G 1% /kaggle/working` — confirming
  the container has ~967GB free outside `/kaggle/working`, so the fix target
  (redirecting there) is sound; attempt 3's bug was the stale-constant issue
  above, not a wrong target path.

### Attempt 2 — kernel version 2 — 2026-11-05

- **Trigger:** `kaggle kernels push` (first real CLI-triggered run, after fixing
  the `visionassist-01-dataset-prep-eda` slug mismatch from version 1).
- **Duration:** ~4 minutes (10:21:01 start → 10:25:05 error).
- **Status:** `KernelWorkerStatus.ERROR`.
- **Key error:**
  ```
  RuntimeError: Dataset 'coco.yaml' error ❌ Insufficient free disk space
  18.673 GB < 27.013 GB required, Please free 8.340 GB additional disk
  space and try again.
  ```
- **Where:** the COCO-download cell (first download cell after the attempt-1
  LVIS/COCO reorder fix — so the error moved from LVIS to COCO, confirming the
  LVIS/COCO dedup fix worked, but exposed a deeper problem).
- **Root cause:** `datasets_dir` defaulted to `/kaggle/working/datasets` —
  Kaggle's ~20GB persisted-Output quota, not real available disk. A single COCO
  download (19G+1G zips, ~27GB with extraction headroom) doesn't fit there
  regardless of the dedup fix.
- **Fix applied:** redirect `datasets_dir` to `/kaggle/tmp/datasets` before any
  download (→ attempt 3).
- **Output pulled:** `kaggle kernels output` — 1.98GB partial download +
  `visionassist-01-dataset-prep-eda.log` (156KB), inspected directly for the
  traceback above.

### Attempt 1 — local/manual browser run (pre-CLI-setup) — 2026-11-05

- **Trigger:** user manually ran the notebook in Kaggle's browser UI, before the
  `kaggle kernels push` CLI setup existed.
- **Status:** failed.
- **Key error:**
  ```
  RuntimeError: Dataset 'lvis.yaml' error ❌ Insufficient free disk space
  17.415 GB < 27.013 GB required, Please free 9.599 GB additional disk
  space and try again.
  ```
- **Where:** the LVIS-download cell (LVIS triggered before COCO at this point).
- **Root cause:** `lvis.yaml`'s own `download:` block independently re-fetches
  COCO's `train2017.zip`/`val2017.zip` (~20GB) into `datasets/lvis/images/` — a
  second, separate copy of the exact same images `coco.yaml` downloads into
  `datasets/coco/images/`, since LVIS reuses COCO's JPEGs verbatim but
  Ultralytics doesn't dedupe across the two configs.
- **Fix applied:** download COCO first, symlink `datasets/lvis/images` to
  `datasets/coco/images`, then trigger LVIS (→ attempt 2's reorder; this fix
  itself worked — attempt 2's error moved to a different, smaller root cause).

---

## Notebook 2 — `02_train.ipynb`

_No runs yet — blocked on notebook 1 producing real committed Output
(`kernel_sources` rejected both upstream notebooks as "not valid kernel sources"
on the first push, since neither had completed a run yet)._

## Notebook 3 — `03_evaluate_finetune_checks.ipynb`

_No runs yet — same blocker as notebook 2, plus depends on notebook 2's output._
