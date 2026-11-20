---
name: kaggle-yolo-finetune
description: Interactive, guided workflow for fine-tuning VisionAssist's YOLO26n lost-and-found model on Kaggle's free GPU tier. Checks/completes Kaggle API setup, walks through EDA/dataset and hyperparameter choices with their defaults and rationale explained one at a time, then runs (or sweeps) the 3-notebook Kaggle pipeline, confirming before every GPU-consuming kernel push. Use when asked to fine-tune YOLO on Kaggle for this project, run or compare a hyperparameter sweep, or set up Kaggle credentials for the finetuning pipeline.
---

# Kaggle YOLO fine-tuning — guided run / sweep

You are walking a user (who may know nothing about Kaggle or fine-tuning) through
either a single fine-tuning run or a multi-variant hyperparameter sweep of
VisionAssist's YOLO26n lost-and-found model, on Kaggle's free GPU tier.

**Source of truth, read fresh every time — don't rely on your own memory of
their contents:**
- `docs/KAGGLE_FINETUNING_SOP.md` — the full human-readable SOP. Its Section 4
  table is the authoritative hyperparameter reference (meaning / Ultralytics
  default / this project's value / why) and Section 8 is the pitfalls list.
  Everything you tell the user about a hyperparameter or a known failure mode
  should come from there, not from general ML knowledge — this project has
  measured, sourced reasons for its specific values.
- `finetuning/kaggle/RUN_LOG.md` — the actual sweep history and its results
  table format. Any new sweep you run should produce a write-up shaped like
  this file's "Training results" / "Catastrophic-forgetting check" tables.
- `docs/FINE_TUNING_YOLO_GUIDE.md` — deeper design rationale, for anything the
  SOP points back to it for (e.g. Section 2's class list, Section 5's
  LoRA-rejection reasoning).
- `finetuning/prepare_dataset.py`, `finetuning/eda.py`, `finetuning/02_train.ipynb`,
  `finetuning/01_dataset_prep_and_eda.ipynb` — the real code. Read the actual
  current argument defaults/cell contents before telling the user what a
  default is; these are what the SOP's table describes, and they're the
  literal thing you'll be editing in Step 5 below.

Read the SOP and RUN_LOG fully before starting Step 1 below.

## Hard rules (apply throughout, not just where mentioned)

- **Never read, print, log, cat, or echo the contents of `~/.kaggle/kaggle.json`.**
  Only ever check that it *exists* and is `chmod 600`. The user's Kaggle API
  key is a real credential; treat it like a password.
- **Never guess** a Kaggle username, a kernel slug, an LVIS class id, or a
  hyperparameter's current value. Ask the user, or read it from this repo's
  own files. If something needed isn't in either place, say so and stop.
- **Confirm before every `kaggle kernels push`** (or `kaggle kernels push` via
  any script you write) — each one can consume real, quota-limited GPU hours
  (~30 hr/week free tier), even though no money changes hands. Show exactly
  what will run (kernel id, the hyperparameters in play, the rough time
  estimate from the SOP) before asking.
- **Confirm before overwriting** any existing file under `finetuning/kaggle/`
  or `finetuning/*.ipynb` — cloning to a new name is almost always right;
  silently overwriting the project's current working notebook is not.
- **One variable per sweep variant.** If the user asks to change two
  hyperparameters in the same variant, point out that this breaks attribution
  (per SOP Section 6) and ask them to confirm that's really what they want.

## Procedure

### Step 1 — Kaggle setup check

1. Check whether the `kaggle` CLI is installed (`kaggle --version` or
   `pip show kaggle`). If missing, offer to run `pip install kaggle`.
2. Check whether `~/.kaggle/kaggle.json` exists and its permissions
   (should be `600`). Do **not** open or read it.
   - If missing: explain (in your own words, drawing on SOP Section 1.3) that
     they need to create a Kaggle account, verify their phone number (needed
     for GPU access), and generate an API token from kaggle.com → Settings →
     API → Create New Token, then save it to `~/.kaggle/kaggle.json` and
     `chmod 600` it. Wait for them to confirm they've done this before
     continuing — you cannot do this step for them, it requires their own
     Kaggle account and browser session.
   - If it exists but isn't `600`: fix the permission yourself
     (`chmod 600 ~/.kaggle/kaggle.json`) — this is a safe, local,
     non-destructive fix, not one that needs confirmation.
3. Smoke-test it: `kaggle kernels list --mine`. If this errors, surface the
   real error message to the user rather than guessing what's wrong.
4. Check every `finetuning/kaggle/**/kernel-metadata.json` and
   `finetuning/kaggle/scripts-dataset/dataset-metadata.json` for the
   placeholder username `dceshubh`. If any are found, ask the user for their
   own Kaggle username (never infer it from `kaggle.json`'s contents — ask
   them directly in chat) and replace every occurrence with it.

### Step 2 — single run or sweep?

Ask the user directly: are they running one fine-tune with some chosen
settings, or comparing several variants (a sweep)? If a sweep, ask which
hyperparameter(s) they want to vary and how many variants — remind them of
the one-variable-per-variant rule above.

### Step 3 — dataset / EDA configuration

Walk through these, one at a time, in conversation — for each: state what it
controls, its current value in this repo (read the actual file, don't quote
from memory), and ask whether to keep it or change it:

- `--coco-sample-size` (`finetuning/prepare_dataset.py`, read inside
  `finetuning/01_dataset_prep_and_eda.ipynb`'s prep cell) — currently `5000`.
  Explain this is the catastrophic-forgetting mitigation sample size, and
  that `5000` itself came from a measured sweep (SOP Section 4/6), not a
  guess.
- `--val-fraction` (default `0.2`) and `--seed` (default `42`) in the same
  script — usually fine to leave as-is; ask anyway.
- `--imbalance-floor` for `eda.py` (default `50`) — the instance-count
  threshold under which a class gets flagged as under-represented.
- Whether to drop a kept COCO class (this project dropped `person`): set
  `LOSTFOUND_DROP_KEPT_CLASSES` (SOP Section 4), do not edit the class list.
- The class list itself (SOP Section 5 / Guide Section 2). Flag this as a
  bigger change than a hyperparameter — only touch it if the user explicitly
  asks to add/remove classes, and if so point them at `prepare_dataset.py`'s
  `NOVEL_CLASSES`/kept-class list and the requirement to confirm any new
  class's real LVIS id rather than guessing it (Guide Section 2 explains how
  the existing ids were verified).

### Step 4 — hyperparameter walkthrough

For **each** of `epochs`, `imgsz`, `freeze`, `optimizer`, `lr0`, `patience`,
`cls_pw` (all set in `finetuning/02_train.ipynb`'s `model.train(...)` cell —
read the actual current values from that cell, don't assume they still match
the SOP's table verbatim):

1. State what it controls, in plain language.
2. State the Ultralytics default **and** this project's current value, and
   explain *why* they differ (or why the project kept the default) — pull
   this reasoning from SOP Section 4, which is sourced from the real sweep
   results and the real Ultralytics source, not invented.
3. Ask: keep the current value, or change it? If changing, get the new value.
4. If a change interacts with another hyperparameter (the clearest case:
   changing `optimizer` away from `AdamW` should prompt reconsidering `lr0`,
   since Ultralytics' own `default.yaml` ties them together — "SGD=1e-2,
   Adam/AdamW=1e-3"), flag that interaction and ask about the linked value
   too rather than silently leaving it inconsistent.

### Step 5 — materialize the configuration

If nothing changed from the current defaults, use the existing notebooks/
kernel folders as-is — no need to clone anything for a plain reproduction run.

If anything changed (single run or any sweep variant):

1. Copy the relevant notebook (`finetuning/02_train.ipynb` for a training
   hyperparameter, `finetuning/01_dataset_prep_and_eda.ipynb` for a
   dataset-prep flag) to a new, descriptively-named file under `finetuning/`
   (e.g. `02_train_lr0_0005.ipynb`) — never edit the original in place for an
   experiment.
2. Edit the literal value(s) in that copy's relevant cell. Load the `.ipynb`
   as JSON, locate the cell by its known content (the `model.train(` call,
   or the `!python -u prepare_dataset.py ...` line), and change only the
   specific argument(s) the user asked to change — leave everything else,
   including every comment explaining prior findings, untouched.
3. Create a new folder under `finetuning/kaggle/` (e.g.
   `finetuning/kaggle/02-train-lr0-0005/`), symlink the new notebook file
   into it (`ln -s ../../<new-notebook>.ipynb <same-name>.ipynb`, matching
   the existing symlink pattern), and copy a sibling folder's
   `kernel-metadata.json` into it, editing: `id` (new, unique slug under the
   user's own username), `title`, `code_file` (the new notebook's filename),
   and `kernel_sources`/`dataset_sources` (point at whichever upstream
   kernel this variant should actually attach to — confirm this with the
   user if it's not obvious which notebook-1 variant a notebook-2 variant
   should read from).
4. Show the user the diff/new files before pushing anything.

### Step 6 — run it (confirm-each-push)

For each kernel that needs to run, in dependency order (scripts dataset →
notebook 1 → notebook 2 → notebook 3):

1. **Confirm with the user first**, showing: the kernel id, what's different
   about this run (if anything), and the SOP's rough time/GPU-hour estimate
   for that notebook.
2. Push it: `kaggle kernels push -p finetuning/kaggle/<folder>` (or
   `kaggle datasets version -p finetuning/kaggle/scripts-dataset -m "..."`
   for the scripts dataset specifically — reminder: a notebook push does
   **not** also push script changes; that's a separate command against the
   dataset, per SOP Section 8 pitfall #1).
3. Check status (`kaggle kernels status -k <owner>/<slug>`) and, if the user
   wants to watch it live, stream logs (`kaggle kernels logs -f -k ...`).
   For a long training run, don't busy-poll — check back periodically
   (e.g. every several minutes) rather than looping tightly.
4. **After notebook 1 specifically finishes**: pull and summarize
   `eda_report/class_distribution.csv` (zero-instance classes, classes under
   the imbalance floor) for the user **before** proceeding to notebook 2 —
   this is the SOP's explicit review gate (Section 3, Step 1); don't skip it
   even if the user seems eager to move on, since a GPU-hour spent training
   on a broken dataset is not easily undone.
5. **After notebook 2 finishes**: pull the weights locally —
   `kaggle kernels output -k <owner>/<slug>-02-train... -p ./downloaded_run`.
6. **After notebook 3 finishes**: report whether the run passed the
   fine-tuning gate (novel classes learned something, kept classes didn't
   regress) — per SOP Section 3, Step 3, this has to pass before anything
   downstream (a golden-set benchmark, or deploying the checkpoint) is worth
   doing.

### Step 7 — for a sweep: build the comparison

Once every variant has completed notebook 3 (or a local matched comparison
run the same way this project's original sweep was scored — see
`RUN_LOG.md`'s "Catastrophic-forgetting check" section for the exact matched
methodology, same held-out images, never comparing variants on different
held-out sets):

1. Build two tables, shaped like `RUN_LOG.md`'s existing ones: one per-variant
   own-validation result (best epoch, training time, mAP50, mAP50-95), one
   matched forgetting-check table (mAP50-95 per kept class, per variant,
   relative change vs. base YOLO).
2. Write a plain-language recommendation — which variant to keep and why —
   not just a table of numbers with no verdict, matching `RUN_LOG.md`'s own
   style.
3. Ask the user whether this write-up should become a new file (e.g.
   `finetuning/kaggle/RUN_LOG_<short-topic>.md`) or a new dated section
   appended to the existing `RUN_LOG.md` — don't assume which they want.

### Step 8 — wrap up

Tell the user, in summary: what ran, where the resulting weights ended up
locally, whether it passed the fine-tuning gate, and the real next steps this
skill deliberately doesn't do for them (per SOP Section 9): the real
three-way golden-set benchmark against self-captured photos
(`finetuning/evaluate.py`), and wiring a new checkpoint into the running app
(Guide Section 9). Don't imply the job is fully done if those are still
outstanding.
