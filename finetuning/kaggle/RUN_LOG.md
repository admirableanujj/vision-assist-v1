# Kaggle Run Log

Running record of every `01_dataset_prep_and_eda.ipynb` (and later, `02_train.ipynb` /
`03_evaluate_finetune_checks.ipynb`) execution attempt on Kaggle — what changed,
what happened, and the exact failure signal, so past attempts don't have to be
re-diagnosed from memory. Newest entry on top. Updated after every run, not just
the ones that worked.

---

## Notebook 1 — `01_dataset_prep_and_eda.ipynb`

### Attempt 3 — kernel version 3 — 2026-11-05 (in progress)

- **Trigger:** `kaggle kernels push` after redirecting `datasets_dir` to
  `/kaggle/tmp/datasets` (Ultralytics' default, `/kaggle/working/datasets`, sits
  inside Kaggle's ~20GB persisted-Output quota — too small for even one COCO
  download; `/kaggle/tmp` is the large ephemeral scratch path).
- **Status:** RUNNING (monitor started at 10:32, polling every 60s).
- **Outcome:** pending.

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
