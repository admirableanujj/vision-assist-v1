# SOP: Fine-Tuning VisionAssist's YOLO26n on Kaggle

**Audience: zero prior Kaggle or fine-tuning experience assumed.** This is the
"how do I actually do this, starting from nothing" document. It assumes you've
never opened Kaggle, never fine-tuned a model, and don't know what an epoch or
a learning rate is. If you already know all that and just want the fast
reference, use `finetuning/README.md` ("How to run it") instead — this SOP is
the expanded, explained version of the same steps, plus the hyperparameter
primer and troubleshooting guide those docs assume you already have.

For the **design rationale** (why 48 classes, why this dataset, why these
hyperparameters) see `docs/FINE_TUNING_YOLO_GUIDE.md` — this SOP tells you how
to *execute* that plan, not why it's shaped the way it is, though every
hyperparameter section below links back to the relevant Guide section.

**There's also an interactive agent skill that walks you through all of this
conversationally** — `.agents/skills/kaggle-yolo-finetune/SKILL.md`. It is
written in the open Agent Skills format and is not tied to one tool: Codex,
Claude Code, Gemini CLI, Antigravity, Cursor and other harnesses that support
the format pick it up automatically, so you can just ask your agent to
fine-tune YOLO on Kaggle or run a hyperparameter sweep. If your agent has no
skill support, tell it to read that file and follow it (the repo's root
`AGENTS.md` says the same thing to the agent). This SOP is still worth
reading either way — the skill follows the exact same plan, and this doc is
what you'd follow by hand with no agent at all.

---

## 0. What you're actually doing (30 seconds)

**Fine-tuning** means taking a model someone else already trained (YOLO26n,
pretrained on 80 everyday object classes) and training it a bit further on
your own, narrower dataset, so it learns your specific classes without
starting from zero. Here, that means: take YOLO26n, keep 20 of its 80 classes
(the ones relevant to a lost-and-found camera — `backpack`, `cell phone`,
`keys`... wait, `keys` isn't COCO — see Section 5), add 28 brand-new classes
(`keys`, `wallet`, `screwdriver`, etc.), and retrain just part of the network
on a combined dataset so it learns the new classes without forgetting the old
ones.

**Kaggle** is a free platform (owned by Google) best known for data-science
competitions, but it also gives anyone a free Jupyter-notebook environment
with a GPU attached — no credit card, no local GPU needed. That's the only
reason it's in this pipeline: training a neural net on CPU is painfully slow;
Kaggle hands you a free NVIDIA T4/P100 GPU for up to ~30 hours a week.

**The three-notebook pipeline** (all three already exist in this repo, you're
not writing them from scratch):

| # | Notebook | Does | Needs GPU? |
|---|---|---|---|
| 1 | `01_dataset_prep_and_eda.ipynb` | Downloads the raw data, builds the 48-class dataset, runs an imbalance report | No |
| 2 | `02_train.ipynb` | Actually trains the model | **Yes** |
| 3 | `03_evaluate_finetune_checks.ipynb` | Sanity-checks the result (did it learn? did it forget?) | No |

You run them in order, each one feeding its output into the next.

---

## 1. One-time setup: Kaggle account + API access

Do this once. After this section you'll never need the Kaggle website again
for routine runs — everything else happens from the command line.

### 1.1 Create a Kaggle account

Go to kaggle.com and sign up (Google/email, whichever is easiest). Free.

### 1.2 Verify your phone number (required for GPU access)

Kaggle gates GPU access behind phone verification — without it your notebooks
are CPU-only, and CPU training of even this small model would take many hours
instead of 1–3. In your Kaggle account settings, find the phone verification
step and complete it. This is a one-time Kaggle account requirement, not
something this repo's code checks for — if a kernel push later fails or
silently falls back to CPU, this is the first thing to check.

### 1.3 Install the Kaggle CLI and create an API token

```bash
pip install kaggle
```

Then, on kaggle.com: click your profile picture → **Settings** → **API** →
**Create New Token**. This downloads a `kaggle.json` file containing your
username and a secret API key. Move it into place and lock down its
permissions (the CLI refuses to run otherwise):

```bash
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

**Treat `kaggle.json` like a password.** It's a real API credential tied to
your account and GPU quota — never commit it to this repo, never paste its
contents into a chat with an AI assistant (including when using the
companion skill — it only ever needs to confirm the file *exists*, never read
its contents), and regenerate it from the same Settings page if you ever
suspect it leaked.

Verify it works:

```bash
kaggle kernels list --mine   # should print (possibly empty) output, not an auth error
```

### 1.4 Point this repo's kernel configs at your username

Every `kernel-metadata.json` and `dataset-metadata.json` under
`finetuning/kaggle/` currently has `dceshubh` as the Kaggle username (the
original author's). You need your own copies pointing at *your* account —
Kaggle kernels are scoped per-owner, so pushing with someone else's username
in the `id` field fails. Replace `dceshubh` with your own Kaggle username in
every one of these files:

```bash
grep -rl "dceshubh" finetuning/kaggle/ | xargs sed -i '' 's/dceshubh/<your-kaggle-username>/g'
```

(Drop the `''` after `-i` on Linux; macOS's `sed -i` needs that empty-string
argument.) This is exactly what the skill's setup step automates.

---

## 2. Code & data map — everything in one place

Everything below already exists in this repo on `main`. Nothing in this
section needs to be written — just located and understood.

| What | Where | Notes |
|---|---|---|
| Design rationale (why this plan) | `docs/FINE_TUNING_YOLO_GUIDE.md` | Read this if you want the "why," not just the "how" |
| Implementation + run instructions | `finetuning/README.md` | The condensed version of this SOP; also the canonical file-by-file index |
| Full hyperparameter sweep history | `finetuning/kaggle/RUN_LOG.md` | The 3-run comparison that led to this project's current defaults — **the template for your own sweep's write-up** |
| Dataset-prep script | `finetuning/prepare_dataset.py` | Merges LVIS + COCO into the 48-class dataset |
| Catastrophic-forgetting set builder | `finetuning/prepare_coco_forgetting_check.py` | Fully automatic, no manual annotation |
| EDA / imbalance script | `finetuning/eda.py` | Per-class instance counts + imbalance flag |
| Notebook 1 (dataset prep + EDA) | `finetuning/01_dataset_prep_and_eda.ipynb` (symlinked into `finetuning/kaggle/01-dataset-prep-eda/`) | No GPU |
| Notebook 2 (training) | `finetuning/02_train.ipynb` (symlinked into `finetuning/kaggle/02-train/`) | GPU required |
| Notebook 3 (eval gate) | `finetuning/03_evaluate_finetune_checks.ipynb` (symlinked into `finetuning/kaggle/03-evaluate-finetune-checks/`) | No GPU |
| Three-way golden-set benchmark (separate, later) | `finetuning/evaluate.py` | Base YOLO vs. YOLOE vs. fine-tuned, once you have self-captured photos — not part of the Kaggle loop |
| This project's own already-run kernels (reference, not yours to run) | `https://www.kaggle.com/code/dceshubh/visionassist-01-dataset-prep-eda`, `.../visionassist-02-train`, `.../visionassist-03-evaluate-fine-tune-checks` | Private kernels under the original author's account — useful to see what a finished run looked like, not runnable by you directly. Push your own copies instead (Section 1.4, Section 3). |

**Where the training data comes from** — no manual download, ever:

- **LVIS** (Large Vocabulary Instance Segmentation, ~160K images, built on
  COCO's own photos) supplies all 28 new classes. Ultralytics' own
  `lvis.yaml` config downloads it automatically the first time
  `check_det_dataset("lvis.yaml")` runs — inside notebook 1, on Kaggle. You
  never fetch a zip file yourself.
- **COCO `train2017`** supplies the 20 kept classes, specifically a random
  sample of it (`--coco-sample-size`, default 5000 images) — also
  auto-downloaded by notebook 1, via `check_det_dataset("coco.yaml")`.
- Full rationale for exactly these two sources (and the Roboflow fallback for
  classes that turn out too sparse) is Guide Section 3.

---

## 3. Running the pipeline, step by step

This mirrors `finetuning/README.md`'s "How to run it," expanded for a first
run. All three notebook pushes happen from your own machine's terminal, in
this repo's root.

### Step 0 — first-time kernel creation (once per notebook/dataset)

```bash
# The scripts dataset (prepare_dataset.py, prepare_coco_forgetting_check.py, eda.py)
kaggle datasets create -p finetuning/kaggle/scripts-dataset

# The three notebooks, in order
kaggle kernels push -p finetuning/kaggle/01-dataset-prep-eda
kaggle kernels push -p finetuning/kaggle/02-train
kaggle kernels push -p finetuning/kaggle/03-evaluate-finetune-checks
```

This creates four things on Kaggle under your account: one dataset, three
kernels (notebooks). `kernel-metadata.json`'s `dataset_sources`/
`kernel_sources` fields already wire each kernel's inputs (notebook 2 reads
notebook 1's output; notebook 3 reads both) — no manual "Add Data" clicking
needed, even on this very first push.

### Step 1 — run notebook 1 (dataset prep + EDA, no GPU)

A push alone doesn't execute the notebook with a fixed GPU setting — by
default a pushed kernel runs with whatever accelerator its metadata specifies
(`enable_gpu: false` here, so this one runs on CPU automatically). Watch it:

```bash
kaggle kernels status -k <your-username>/visionassist-01-dataset-prep-eda
kaggle kernels logs -f -k <your-username>/visionassist-01-dataset-prep-eda
```

Expect this to take a while the *first* time (LVIS + COCO downloads run into
the tens of GB) — later runs that reuse the same Kaggle account don't
re-download from scratch unless you change the sample size enough to need a
fresh pull. **Before moving to Step 2**, check the EDA output
(`eda_report/class_distribution.csv`/`.png`, printed inline in the notebook
too) for zero-instance classes — if any class has zero training instances,
training will never learn it, and that's much cheaper to fix now (raise
`--coco-sample-size`, or a Roboflow fallback — Guide Section 3.2) than after
spending GPU time.

### Step 2 — run notebook 2 (training, GPU required)

On kaggle.com, open your pushed `02-train` kernel once and set
**Settings → Accelerator → GPU T4 x2** (or P100) — this is a per-kernel
setting that currently has to be set once in the browser, not via
`kernel-metadata.json`'s `enable_gpu` flag alone (that flag just says "this
kernel is allowed to request a GPU," not which one). After that one-time
click, subsequent pushes keep the setting.

```bash
kaggle kernels push -p finetuning/kaggle/02-train
kaggle kernels status -k <your-username>/visionassist-02-train
kaggle kernels logs -f -k <your-username>/visionassist-02-train
```

Expect roughly **1–3 hours** for this project's dataset size on a single T4
(Guide Section 8). If your session looks like it's about to hit Kaggle's
session-length cap, see Section 7's troubleshooting entry on resuming — don't
just let it get killed.

### Step 3 — run notebook 3 (the eval gate, no GPU)

```bash
kaggle kernels push -p finetuning/kaggle/03-evaluate-finetune-checks
kaggle kernels status -k <your-username>/visionassist-03-evaluate-fine-tune-checks
```

This checks two things: did the 28 new classes learn *something*, and did
the 20 kept classes regress (catastrophic forgetting)? **This has to pass
before you do anything else with the checkpoint** — there's no point
capturing real photos against a model that didn't train properly. If it
doesn't pass, go back to Step 2 (more epochs, a different `freeze` value) or
Step 1 (raise `--coco-sample-size`, or a per-class Roboflow fallback).

### Step 4 — pull the trained weights down locally

```bash
kaggle kernels output -k <your-username>/visionassist-02-train -p ./downloaded_run
```

`downloaded_run/runs/lostfound_yolo26n/weights/best.pt` is your fine-tuned
checkpoint. What happens with it next (the real three-way golden-set
benchmark, then wiring it into the app) is Guide Section 7/9 and outside this
SOP's scope — this SOP covers getting a trained, sanity-checked checkpoint
out of Kaggle, not deploying it.

---

## 4. Hyperparameters — what each one means, this project's value, and why

Every one of these is a real, documented Ultralytics argument to
`model.train(...)` (see `finetuning/02_train.ipynb`, the actual training
cell) — nothing here is invented. "Default" means Ultralytics' own
out-of-the-box value if you never set the argument at all.

| Hyperparameter | What it actually controls | Ultralytics default | Used here | Why changed (or why kept) |
|---|---|---|---|---|
| `epochs` | How many full passes over the training data to attempt, at most. One epoch = the model has seen every training image once. | `100` | `100` (cap — early stopping usually ends it sooner) | Kept at a generous cap since `patience` (below) does the real stopping decision; no reason to lower it and risk cutting off a run that's still improving. |
| `imgsz` | The pixel size images get resized to before going into the model (e.g. `640` means 640×640). Bigger = more detail retained, slower + more memory. | `640` | `640` | Standard YOLO default, no reason to deviate — this model's accuracy bottleneck is class imbalance and dataset size, not input resolution. |
| `freeze` | How many of the model's early layers (blocks) get **locked** — their weights don't update during training at all, only later layers do. Freezing more = faster training, less risk of destroying what the pretrained model already knew, but less ability to adapt early feature extraction to your new classes. | `None` (nothing frozen — full fine-tune) | `10` (blocks 0–9 frozen: the convolutional backbone/stem through SPPF; blocks 10 onward — the attention block, the neck, and the detection head — stay trainable) | This is "small new dataset, visually similar domain to COCO" — exactly the case Ultralytics recommends freezing the backbone for. Measured directly against the real checkpoint: 43.38% of parameters frozen, 56.62% trainable. **Measured, not assumed:** the sweep's `freeze=None` variant (everything trainable) was tried and made catastrophic forgetting *worse* on average (−32.0% vs. −20.1% for the `freeze=10` winner) — see Section 6. |
| `optimizer` | The algorithm that actually updates the model's weights each step, given the computed loss. Different optimizers converge differently depending on how much training is happening and from what starting point. | `"auto"` — Ultralytics picks based on your dataset size: for this project's iteration count (~28,800, over its internal 10,000 threshold), `auto` would have picked **MuSGD** at `lr0=0.01`, not AdamW | `"AdamW"` | `auto`'s heuristic is tuned for large-scale training (including from-scratch). This is fine-tuning a mostly-pretrained, majority-frozen model — AdamW's adaptive per-parameter updates are the better fit for that regime, and the sweep's one unfrozen-backbone variant (closer to the large-scale regime `auto` assumes) needed ~1.75x the epochs for a worse result, consistent with that mismatch. |
| `lr0` | The initial learning rate — how big a step each weight update takes. Too high and training destabilizes/overshoots; too low and it barely moves. | `0.01` (Ultralytics' own `default.yaml` comment: *"SGD=1e-2, Adam/AdamW=1e-3"*) | `0.001` | Directly downstream of the optimizer change above — Ultralytics' own default config documents that `0.01` is calibrated for SGD-family optimizers and should drop to `0.001` under Adam/AdamW. Keeping `0.01` with AdamW on an already-mostly-good pretrained model risks large, careless steps — exactly what accelerates catastrophic forgetting. |
| `patience` | Early stopping: if validation mAP hasn't improved for this many consecutive epochs, stop training rather than running to the full `epochs` cap. | `100` | `15` | All three sweep runs early-stopped well under the 100-epoch cap (best epochs were 40, 48, and 71 out of 100) — `15` is responsive enough to stop soon after a plateau without being so short it bails on a model that's still making slow progress. |
| `cls_pw` | Inverse-class-frequency loss reweighting: classes with fewer training examples get their classification loss scaled up, so the model is penalized more for getting rare classes wrong. `1.0` = full inverse-frequency weighting; `0.0` = off. | `0.0` (off) | `0.5` (dampened, not full `1.0`) | The EDA found a 6534x max/min per-class instance-count ratio with 5 classes under a 50-instance floor. Full (`1.0`) weighting risks destabilizing the well-represented classes to chase a handful of classes that simply don't have enough real examples to learn well regardless of loss weight. This costs **zero** extra GPU time (it only reweights an existing loss term), so there's no reason to ever train without it once you know your dataset is imbalanced. |
| `--coco-sample-size` (notebook 1's `prepare_dataset.py` flag, not a `model.train()` arg) | How many COCO `train2017` images (containing at least one of the 20 kept classes) get folded into the training set, on top of whatever LVIS already contributes. This is the catastrophic-forgetting mitigation — training on COCO's own images for the kept classes, not just the new LVIS-sourced images. | — (script-specific; the script's own hardcoded default is `5000`, raised from an original untested guess of `750`) | `5000` | Measured, not guessed: the sweep (Section 6) found `750` starved the rarest kept classes (`scissors`, `toothbrush`) of training images — raising it to `5000` improved the catastrophic-forgetting check on **every one** of the 20 kept classes, for only a marginal (0.007 mAP50-95) novel-class cost. |

**How to tune any of these yourself:** change the literal value in
`model.train(...)` inside `finetuning/02_train.ipynb` (or `--coco-sample-size`
in notebook 1's `!python eda.py ...`/`prepare_dataset.py` cell), save the
notebook under a new name so you don't overwrite the working version, create
a matching `finetuning/kaggle/<new-folder>/` with its own `kernel-metadata.json`
(copy an existing one, give it a new `id`), and push. Section 6 below walks
through doing this for a real sweep.

---

## 5. The dataset and class list (what the model actually learns)

Full rationale lives in Guide Section 2 — summarized here for reference while
tuning:

- **48 classes total**, not COCO's original 80 plus a few extras appended.
- **20 kept from COCO**: `person`, `bicycle`, `backpack`, `umbrella`,
  `handbag`, `tie`, `suitcase`, `bottle`, `cup`, `book`, `scissors`,
  `teddy bear`, `toothbrush`, `laptop`, `mouse`, `remote`, `keyboard`,
  `cell phone`, `cat`, `dog` — small, portable, personally-owned items
  plausible to search for at home or in a garage.
- **28 new, from LVIS**: `keys`, `wallet`, `sunglasses`, `glasses`, `watch`,
  `earphone`, `flashlight`, `glove`, `belt`, `pen`, `notebook`, `calculator`,
  `stapler`, `battery`, `shoe`, `jacket`, `briefcase`, `passport`, `scarf`,
  `hat`, `screwdriver`, `wrench`, `pliers`, `hammer`, `toolbox`,
  `tape measure`, `ladder`, `game controller`.
- **Deliberately excluded**: fine-grained jewelry (too small/reflective for
  reliable detection at webcam resolution), footwear/headwear subtypes
  (consolidated into single `shoe`/`hat` classes), and `charger`/cable/
  `keychain` (not real LVIS classes — would need a separate community
  dataset, flagged as a known gap, not silently dropped).

If your own fine-tune needs a *different* class list (e.g. your product
cares about different objects), that's a bigger change than a hyperparameter
— it means editing `prepare_dataset.py`'s `NOVEL_CLASSES`/kept-class list and
re-confirming each new class's real LVIS id (Guide Section 2 explains how
that verification was done — don't guess ids).

---

## 6. Running a hyperparameter sweep (for comparing multiple runs)

This is exactly how this project picked its current defaults — not a
hypothetical process. Three variants were actually run and compared; the
record is `finetuning/kaggle/RUN_LOG.md`, which doubles as the template for
writing up your own sweep.

**The pattern, concretely:**

1. Pick what you're varying (e.g. `freeze`, `lr0`, `--coco-sample-size`,
   `cls_pw` — anything from Section 4's table). Change exactly one thing per
   variant, so you can attribute any difference to that one change — this
   project's own sweep varied `--coco-sample-size` in one variant and
   `freeze` in another, never both in the same run.
2. Copy the relevant notebook (`02_train.ipynb` for a training
   hyperparameter, `01_dataset_prep_and_eda.ipynb` for a dataset-prep flag)
   to a new filename describing the change, e.g. `02_train_lr0_0005.ipynb`,
   and edit the one literal value in its training/prep cell.
3. Create a new folder under `finetuning/kaggle/` (e.g.
   `02-train-lr0-0005/`), symlink the new notebook into it, and copy an
   existing `kernel-metadata.json` from a sibling folder, changing its `id`
   to a new, unique slug (e.g. `<your-username>/visionassist-02-train-lr0-0005`)
   and its `kernel_sources` to point at whichever notebook-1 variant it
   should attach to.
4. `kaggle kernels push -p finetuning/kaggle/<new-folder>`, then track its
   progress the same way as Section 3's Step 2.
5. Once every variant has a `best.pt`, run the **same** matched
   catastrophic-forgetting comparison this project used (Section 7 below) —
   never compare variants on different held-out sets, or the comparison is
   meaningless.
6. Write up the result the way `RUN_LOG.md` does: one table per variant's own
   training result (best epoch, training time, mAP50/mAP50-95), one matched
   forgetting-check table with every variant as a column, and a plain-language
   recommendation of which one to keep and why — not just raw numbers with no
   verdict.

**This project's own sweep, as a worked example** (full detail in
`RUN_LOG.md`):

| Run | What varied | Own-validation mAP50-95 | Forgetting-check mAP50-95 (avg, 20 kept classes) | Verdict |
|---|---|---|---|---|
| 1 — baseline | (nothing — this is the starting point) | 0.171 | 0.272 (−25.3% vs. base YOLO) | Superseded by run 2 |
| 2 — bigger COCO sample | `--coco-sample-size` 750 → 5000 | 0.164 | **0.291 (−20.1%)** | **Adopted** — improved every one of the 20 kept classes, marginal novel-class cost |
| 3 — unfrozen backbone | `freeze=10` → `freeze=None` | 0.164 | 0.248 (−32.0%) | Discarded — forgetting got *worse*, needed 1.75x the epochs |

The lesson that generalizes beyond this specific sweep: **a hyperparameter
change can look neutral or even slightly negative on the metric you're
directly training against** (run 2's own-validation mAP50-95 is actually
*lower* than the baseline's) **while being a clear win on the metric that
actually matters for the product** (the forgetting check). Always run both
comparisons, not just the easy one.

---

## 7. Interpreting your results

### 7.1 The metrics, in plain language

- **IoU (Intersection over Union)** — for one predicted box and one real
  (ground-truth) box, the area where they overlap divided by the total area
  either one covers. `1.0` = perfect overlap, `0.0` = no overlap at all. This
  is the building block every other metric below is computed from.
- **Precision** — of everything the model *said* was class X, what fraction
  actually was? Low precision = lots of false alarms.
- **Recall** — of everything that *actually was* class X in the images, what
  fraction did the model actually find? Low recall = lots of misses.
- **AP (Average Precision)** — one number per class summarizing the
  precision/recall tradeoff across every confidence threshold, for one fixed
  IoU threshold. **mAP** is just the average of AP across all classes.
- **mAP50** — mAP computed at a forgiving IoU threshold (0.5 — the predicted
  box just has to roughly overlap the real one). Always the higher of the
  two numbers you'll see reported.
- **mAP50-95** — mAP averaged across ten IoU thresholds from 0.5 to 0.95,
  stricter overall. This is the number to trust more when comparing models —
  mAP50 can look deceptively good on boxes that are only loosely correct.
- **Trend to expect:** higher is always better, for every metric above, in
  every context. What's *not* always the same is which comparison is a fair
  "win": your fine-tuned model is **expected** to score lower than base YOLO
  on the kept classes (some forgetting is the known cost of fine-tuning at
  all) — the real pass/fail bar is beating your *other sweep variants* on
  that same comparison, not beating base YOLO outright.

### 7.2 The catastrophic-forgetting check, specifically

This project's forgetting check scores every variant on the exact same 5,000
COCO `val2017` images (never used in any run's training), base YOLO scored
with its normal 80-class ids, every fine-tuned variant scored with the same
images' labels remapped to the 20 kept classes. That matched, apples-to-apples
setup is what makes the comparison meaningful — scoring different variants on
different held-out images would just measure "which images happened to be
easier," not which hyperparameter choice was actually better.

---

## 8. Known pitfalls (hit once already — don't re-hit them)

Every one of these actually happened on a real run of this exact pipeline,
confirmed against the real Ultralytics source where relevant — not
theoretical risks. Full detail in `finetuning/kaggle/RUN_LOG.md`'s "Critical
findings" section.

1. **The scripts dataset deploys separately from the notebooks.**
   `kaggle kernels push` only updates a notebook file. If you change
   `prepare_dataset.py`/`eda.py`/`prepare_coco_forgetting_check.py`, you also
   need `kaggle datasets version -p finetuning/kaggle/scripts-dataset -m "..."`
   — otherwise the notebook silently keeps running the old script.
2. **Don't trigger the dataset download with `model.train(epochs=1)`.**
   Confirmed live: in an unattended run (which `kaggle kernels push` always
   is — nobody's watching to interrupt it), this actually trains a full
   epoch over all 118,287 COCO images on CPU (~17 projected hours) instead of
   just downloading. Notebook 1 uses `check_det_dataset()` directly instead,
   which is the exact function `model.train()` calls internally to trigger a
   download, with zero training compute.
3. **Redirect `datasets_dir` away from `/kaggle/working`.** Its default
   location sits inside a ~20GB quota meant for your notebook's *committed
   output*, not scratch space — a single COCO download alone can exceed it.
   `/kaggle/tmp` is the right place for large, ephemeral downloads. A plain
   `settings.update(datasets_dir=...)` call is **not enough** — Ultralytics
   freezes the resolved path in two separately-imported module-level
   constants the moment it's first imported; notebook 1 patches
   `settings.json` directly and force-reimports every `ultralytics.*` module,
   with an assertion that fails immediately if it didn't take effect.
4. **LVIS re-downloads COCO's own images unless you dedupe.** `lvis.yaml`'s
   download block independently refetches the same COCO JPEGs LVIS reuses —
   symlinking `lvis/images → coco/images` (download COCO first) avoids
   doubling the download. Side effect: that symlink makes Ultralytics'
   unzip logic think the LVIS folder is "already extracted" and skip
   unzipping the actual labels — force it with `exist_ok=True` right after.
5. **`kernel_sources` mount paths aren't stable.** The exact same
   `kernel_sources` config has mounted at two different path shapes across
   otherwise-identical pushes, with no way to predict which one applies in
   advance. Search for the expected filename under `/kaggle/input` at
   runtime rather than hardcoding either path shape.
6. **`kaggle kernels logs -f` doesn't reliably replay full history on a long
   job.** For a short run it looks like a complete replay; for a long one
   (70+ epochs) it can return a bounded recent window instead, and naively
   numbering lines by position (rather than reading the actual epoch marker
   in the log) can mislabel which epoch a line is really from.
7. **Generated dataset yamls must not bake in an absolute `path:`.** Notebook
   1 writes its yaml under `/kaggle/working/`; notebooks 2/3 read it back
   under `/kaggle/input/...` — a different container, different absolute
   path.
8. **`!python script.py` cells need `-u`.** Without unbuffered output, print
   statements (including progress lines) can sit invisible for tens of
   minutes, making a slow-but-working step indistinguishable from a hang.
9. **`--coco-sample-size` is a uniform random draw, not a guarantee of
   balance.** It draws that many images uniformly at random from a much
   larger candidate pool — rare kept classes stay proportionally
   under-represented purely because they're rare in that pool, regardless of
   training quality. Raising the sample size helps proportionally but isn't
   a precise fix; true class-stratified sampling would be, and isn't
   implemented here.
10. **If a push times out approaching Kaggle's session cap, nothing resumes
    automatically.** `model.train()` checkpoints every epoch to
    `weights/last.pt`, but an **unattended, API-triggered** run that gets
    killed by the session cap never gets a chance to commit that progress as
    a new Output version — so `RESUME=True` has nothing to resume from. If a
    run is approaching the cap, watch it live (`kaggle kernels logs -f`) and
    stop it manually, from the kernel's page on kaggle.com, well before the
    cap — don't rely on an automatic resume that the unattended workflow
    can't actually exercise.

---

## 9. What this SOP deliberately doesn't cover

- **The real three-way golden-set benchmark** (base YOLO vs. YOLOE vs.
  fine-tuned, on self-captured photos) — that's `finetuning/evaluate.py`,
  needs real photography/annotation first, and is Guide Section 7/
  `finetuning/README.md` Step 4's territory, not a Kaggle step.
- **Wiring a new checkpoint into the running app** — Guide Section 9 covers
  exactly how the current `best.pt` was deployed; this SOP stops at "you have
  a trained, sanity-checked checkpoint downloaded locally."
- **Changing the class list itself** (adding/removing which 48 classes exist)
  — a product decision, not a hyperparameter; see Section 5's note.
