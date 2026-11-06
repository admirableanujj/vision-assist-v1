# Kaggle Run Log

Running record of every `01_dataset_prep_and_eda.ipynb` (and later, `02_train.ipynb` /
`03_evaluate_finetune_checks.ipynb`) execution attempt on Kaggle — what changed,
what happened, and the exact failure signal, so past attempts don't have to be
re-diagnosed from memory. Newest entry on top. Updated after every run, not just
the ones that worked.

---

## Notebook 1 — `01_dataset_prep_and_eda.ipynb`

### Upfront review pass — 2 parallel agents, before attempt 9 — 2026-11-05

User explicitly asked for proactive agent review instead of continuing to
iterate reactively push-by-push. Two agents ran in parallel: one re-auditing
notebook 1 adversarially for the same bug classes already hit, one auditing
notebooks 2/3 (never run even once) before they could cost their own cycles.
Real findings, both verified against actual pinned-library source before
fixing (not taken on faith):

- **[Would have broken 02/03's first run, deterministically]**
  `prepare_dataset.py`/`prepare_coco_forgetting_check.py` baked an absolute
  `path: /kaggle/working/...` into the generated yaml. Verified directly
  against the actual `ultralytics==8.4.106` wheel: `check_det_dataset()`
  uses an absolute `path:` verbatim regardless of whether it exists, and
  only falls back to the yaml's own directory when `path:` is absent
  entirely. Notebook 1 writes these yamls under `/kaggle/working/`;
  notebooks 2/3 read them back from a different mount
  (`/kaggle/input/...`) — the baked-in path would 404 inside
  `model.train()`/`model.val()`. Fixed by dropping `path:` entirely from
  both yaml-writers.
- **[Real, silent performance bug]** `find_candidates()` re-globbed the
  *entire* images directory (118,287 files for COCO `train2017`) on every
  single candidate lookup — O(candidates × dir_size), same bug class as the
  earlier `model.train()` incident: looks cheap, isn't. Fixed with a
  one-time per-directory index.
- **[Design gap, not yet triggered]** `02_train.ipynb`'s session-overrun
  recovery runbook assumes a human watching to interrupt/Save Version —
  this project's actual workflow (confirmed against this very log) is
  100% unattended `kaggle kernels push`. If a real 100-epoch training run
  ever approaches the session cap, nobody's there to save progress, and an
  errored/timed-out batch run doesn't commit an Output version — so the
  otherwise-correct RESUME mechanism can never actually be exercised.
  Documented as a known limitation; not fixed (low risk at the
  estimated 1-3hr runtime, well under the cap).
- **[Cost footgun, not yet triggered]** No `torch.cuda.is_available()`
  assertion before `02_train.ipynb`'s real training call — if GPU
  provisioning silently degrades, `epochs=100` could silently run on CPU
  for the full schedule.
- Minor: neither prep script's output dir was idempotent (no `rmtree`
  before writing) and neither `!python`/`!pip` call checked exit codes
  anywhere in notebook 1 — fixed both (exit-code `assert`s after all three
  `!python` cells; `rmtree(ignore_errors=True)` at the top of both scripts'
  `main()`).
- Minor: `coco.yaml`/`lvis.yaml`'s `download:` default to `delete=False`
  (verified against GitHub source), so both COCO zips are left on disk
  after extraction — real footprint closer to 40GB, not the ~20GB this
  notebook's own markdown assumed. Added explicit cleanup.

Both agents' full reports are preserved in this session's transcript, not
duplicated here — this is the condensed, action-oriented version.

### Attempt 9 — kernel version 9 — 2026-11-05 (in progress)

- **Trigger:** `kaggle kernels push` with the full fix set above, plus a
  real bug found live (not by either agent, since neither had Kaggle
  access): see attempt 7's entry below for the mount-path discovery.
- **Status:** running — script-copy step (now reordered before the
  download, confirmed working in ~30s via `kaggle kernels logs -f`
  immediately after push, rather than waiting ~20min to find out) and
  COCO download both confirmed progressing normally.

### Attempts 7-8 — kernel versions 7-8 — 2026-11-05

- **Trigger:** v7 added a fast `/kaggle/input` diagnostic cell
  (`ls -la /kaggle/input/`) as the very first executable cell, pushed and
  checked within ~20 seconds via `kaggle kernels logs -f` — **deliberately
  not waiting through another ~20min COCO download just to check one
  fact.**
- **Finding:** `/kaggle/input/` contained only `datasets/dceshubh/...` —
  **API-attached `dataset_sources` mount at
  `/kaggle/input/datasets/<owner>/<slug>/`, not the flatter
  `/kaggle/input/<slug>/`** that manual "Add Data" attachment uses (and
  that both attempts 5 and 6 had assumed). This is a genuinely different,
  undocumented-in-this-repo mount convention — not something either
  review agent could have caught without live Kaggle access, since it's a
  platform behavior, not a code bug.
- **Fix applied (v8):** corrected `SCRIPTS_INPUT_DIR` to
  `/kaggle/input/datasets/dceshubh/vision-assist-finetuning-scripts`.
  v8 was pushed, then immediately superseded by v9 once the upfront-review
  findings above landed in the same session — never separately confirmed
  end-to-end on its own, but its one fix carried forward into v9.
- **Open question for notebooks 2/3:** `kernel_sources` (another kernel's
  *output*, as opposed to a dataset) may follow a different mount
  convention than `dataset_sources` — not yet verified, since 02/03 have
  never been pushed. Don't assume `/kaggle/input/visionassist-01-dataset-prep-eda`
  is correct without the same kind of fast diagnostic check first.

### Attempt 5 — kernel version 5 — 2026-11-05

- **Trigger:** `kaggle kernels push` after replacing `model.train(data=...,
  epochs=1)` with direct `check_det_dataset(...)` calls (attempt 4).
- **The big disk/training bugs are confirmed fixed.** Pushing v5 superseded
  attempt 4's stuck session (confirmed: fresh `0.00s`-timestamped logs
  immediately after push). Live-streamed progress: COCO's `val2017.zip`
  (5,001 images) downloaded and unzipped; `train2017.zip` (18GB) downloaded
  in full; LVIS correctly reused the symlinked images — log showed
  `WARNING ⚠️ Skipping ... unzip as destination directory
  /kaggle/tmp/datasets/coco/images/train2017 is not empty` (the dedup fix
  working exactly as designed, zero duplicate image download) — and only
  fetched its own small `lvis-labels-segments.zip` (496.7MB). **`Dataset
  download success ✅ (35.7s)`.** No training epoch ran at all.
- **Status:** `KernelWorkerStatus.ERROR` — but on something trivial.
- **Key error:**
  ```
  AssertionError: Missing expected script(s) — only found [].
  Check SCRIPTS_INPUT_DIR above.
  ```
- **Root cause:** `SCRIPTS_INPUT_DIR` was still the literal placeholder
  string `/kaggle/input/<scripts-slug>` inside the notebook cell itself —
  never filled in with the real slug (`vision-assist-finetuning-scripts`),
  even though `kernel-metadata.json`'s `dataset_sources` already wired the
  dataset's *attachment*. Attaching a dataset and pointing code at its
  actual mounted path are two different things; only the first was done.
  My oversight when filling in the other notebooks' slug placeholders
  earlier — this one in notebook 01 itself was missed.
- **Fix applied:** fill in `SCRIPTS_INPUT_DIR` with the real slug
  (→ attempt 6).
- **Note:** `kaggle kernels logs -f` (live-streaming) was essential here —
  caught the real disk/training fixes working *while the run was still in
  progress*, well before `kernels status` flipped to `ERROR` on this
  unrelated, trivial issue.

### Attempt 4 — kernel version 4 — 2026-11-05

- **Trigger:** `kaggle kernels push` after fixing attempt 3's redirect to
  actually take effect (force-reimport `ultralytics.*` after patching
  `settings.json` on disk — see attempt 3's entry).
- **Status:** disk-space bug fixed (confirmed — got past the point that
  killed attempts 2/3, loaded the model, printed the YOLO26n architecture
  summary), but **discovered a second, much bigger bug**: never reached
  `ERROR` or `COMPLETE` — had to be investigated while still `RUNNING`.
- **What was actually happening (via `kaggle kernels logs -f`, live-streamed
  while still running):** the COCO-trigger cell's `model.train(data="coco.yaml",
  epochs=1, imgsz=640)` doesn't just check/download the dataset — it fully
  trains that one epoch over **all 118,287 COCO `train2017` images**, on
  CPU, since nothing in an unattended `kaggle kernels push` run interrupts
  it the way the notebook's own comment assumed a human would ("interrupt
  after epoch 1 starts"). Live log showed epoch `1/1` at 44% (batch
  3258/7393), ~8.3s/iteration, **~9.5 more hours projected** on top of the
  ~8 already elapsed — ~17+ hours total for a step meant only to trigger a
  download. This would almost certainly have hit Kaggle's session cap
  before finishing, burning the entire session for nothing.
- **Root cause:** the download/convert trigger used `model.train()`, which
  only makes sense as a trigger if something (a human) stops it right
  after the download completes and before real training begins. There's
  nothing to interrupt an API-triggered run.
- **Fix applied:** call `ultralytics.data.utils.check_det_dataset()`
  directly instead — the exact function `model.train()` already calls
  internally to download/verify a dataset, *before* building any model or
  training loop. Same download, zero training compute, no human required
  (→ attempt 5).
- **Note on stopping a stuck session:** no `kaggle kernels` CLI subcommand
  cancels a running kernel (`kaggle kernels --help` lists
  list/files/get/init/push/pull/output/status/logs/update/delete/topics —
  no stop/cancel). Must be stopped manually from the kernel's page on
  kaggle.com, or left to hit Kaggle's own session cap.
- **Also discovered:** `kaggle kernels output` hangs/returns nothing while
  a kernel is still `RUNNING` (no partial/live output via that command) —
  but `kaggle kernels logs -f` *does* stream live logs from a running
  session, which is how this was actually diagnosed without waiting for
  it to finish or fail.

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
