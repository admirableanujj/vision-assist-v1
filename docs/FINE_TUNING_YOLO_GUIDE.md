# Fine-Tuning YOLO26n: Plan

**Status: implementation ready, not yet run.** The dataset-prep script,
the three Kaggle notebooks, and the eval harness described below exist in
`finetuning/` (see `finetuning/README.md` for exact run steps) — but no
training job has actually been executed: no dataset downloaded, no
checkpoint produced, nothing merged into the app's deployed weights.

## 1. The problem, precisely

Standard MS-COCO — the dataset every public YOLO checkpoint (v5 through the
current YOLO26) is pretrained on — has 80 fixed classes, and most of them
don't belong in this product's actual use case: a camera fitted in a living
room, home, or garage, helping someone find a misplaced belonging. Wild
animals (`zebra`, `giraffe`, `elephant`, `bear`, `cow`, `horse`, `sheep`),
outdoor infrastructure (`traffic light`, `fire hydrant`, `stop sign`,
`parking meter`), vehicles that aren't ever "lost" (`airplane`, `bus`,
`train`, `boat`), food (`banana`, `pizza`, `hot dog`), and fixed furniture/
appliances (`couch`, `dining table`, `refrigerator`, `toilet`) make up most
of COCO's 80 classes — none of them are things a lost-and-found camera
needs to recognize, and training capacity spent on them is capacity not
spent on what actually matters. On top of that, none of COCO's 80 cover the
classic small lost items either — `keys`, `wallet`, `sunglasses`, `glasses`,
`watch`, and a long tail of other things people genuinely misplace at home
or in a garage (tools, hats, chargers' cousins, etc.). No version of YOLO
fixes either problem by being "better" — they're all trained on the same
80-class label set. `YOLOEVisionEngine` (see `YOLO_VS_YOLOE_GUIDE.md`)
already closes the missing-classes gap with zero training, at ~30% mAP
accuracy, using a ~300-item curated home vocabulary (`HOME_ITEM_CLASSES` in
`vision_engine.py`); this document is the plan for a real fine-tuned,
higher-accuracy YOLO26n that closes both gaps at once — dropping the
irrelevant COCO classes and adding the relevant missing ones — as a third
point of comparison.

## 2. Classes

**48 total, not the original 80 + N appended.** Earlier drafts of this plan
kept all 80 COCO classes and only appended 3 new ones (`keys`/`wallet`/
`sunglasses`) — that was a mistake: it meant training capacity on `zebra`
and `parking meter` while the product's actual job (home/garage lost items)
was covered by only 3 classes. The redesigned scheme instead **keeps 20 of
COCO's 80** (small, portable, personally-owned items plausible to search
for at home or in a garage) and **adds 28 new classes** for things people
actually search for that COCO never covered — chosen to overlap with
`vision_engine.py`'s existing `HOME_ITEM_CLASSES` (YOLOE's curated
vocabulary) where practical, so the two engines' coverage stays reasonably
aligned, though they're trained/prompted independently.

**20 kept COCO classes** (new indices 0–19 — training reassigns fresh
indices regardless, since the detection head's output layer is fully
retrained whenever `nc` changes, so there's no reason to preserve COCO's
original numbering):

`person`, `bicycle`, `backpack`, `umbrella`, `handbag`, `tie`, `suitcase`,
`bottle`, `cup`, `book`, `scissors`, `teddy bear`, `toothbrush`, `laptop`,
`mouse`, `remote`, `keyboard`, `cell phone`, `cat`, `dog`

Dropped: everything else — wild animals, street/outdoor infrastructure,
vehicles (`car`/`motorcycle`/`bus`/`train`/`boat`/`airplane`), all food
classes, sports equipment (niche for "lost at home," not impossible but
deliberately out of scope for v1), and fixed furniture/appliances/fixtures
(`chair`, `couch`, `bed`, `tv`, `toilet`, `oven`, `refrigerator`, etc. — these
are never really "lost," they're stationary).

**28 new classes** (indices 20–47), each confirmed against a real,
downloaded LVIS class id (Ultralytics `lvis.yaml`, v8.4.106 — matching this
repo's pin) — not guessed:

| Index | Class | LVIS id (source) | Index | Class | LVIS id (source) |
|---|---|---|---|---|---|
| 20 | `keys` | 604 | 34 | `shoe` | 947 |
| 21 | `wallet` | 1155 | 35 | `jacket` | 588 |
| 22 | `sunglasses` | 1034 | 36 | `briefcase` | 152 |
| 23 | `glasses` | 994 | 37 | `passport` | 769 |
| 24 | `watch` | 1160 | 38 | `scarf` | 920 |
| 25 | `earphone` | 408 | 39 | `hat` | 543 |
| 26 | `flashlight` | 456 | 40 | `screwdriver` | 925 |
| 27 | `glove` | 499 | 41 | `wrench` | 1195 |
| 28 | `belt` | 87 | 42 | `pliers` | 820 |
| 29 | `pen` | 780 | 43 | `hammer` | 529 |
| 30 | `notebook` | 723 | 44 | `toolbox` | 1100 |
| 31 | `calculator` | 183 | 45 | `tape measure` | 1061 |
| 32 | `stapler` | 1005 | 46 | `ladder` | 620 |
| 33 | `battery` | 69 | 47 | `game controller` | 598 |

The last 6 (`screwdriver` through `ladder`) are the garage-specific
additions. `game controller` (LVIS: `joystick`) is a deliberately
living-room-specific addition — one of the most commonly "lost under the
couch cushions" items there is.

**Deliberately excluded**, despite existing in LVIS, with reasons (not
silently dropped):
- Fine-grained jewelry (`ring` 884, `earring` 410, `necklace` 714,
  `bracelet` 145) — too small/reflective for reliable webcam-resolution
  detection; annotation effort would be disproportionate to the likely
  accuracy. A stretch goal, not ruled out forever.
- Footwear/headwear subtypes (`boot`, `sandal`, `slipper`, `flip-flop`,
  `cowboy hat`, `bowler hat`, `baseball cap`, etc.) — consolidated into
  single `shoe` and `hat` classes rather than kept as separate near-
  identical LVIS subtypes a nano model has little hope of telling apart.
- `charger`/cable, `airpods`-style earbuds case, and `keychain` — not LVIS
  classes at all (checked directly, not assumed); would need a Roboflow
  community dataset or none exists reliably. Flagged as a gap, not papered
  over — `earphone` (408) covers generic headphones/earbuds reasonably,
  but a bare charging cable is thin/deformable and hard to detect reliably
  regardless of data availability.

## 3. Datasets

### 3.1 Primary: LVIS (recommended)

**LVIS** (Large Vocabulary Instance Segmentation, Facebook AI Research) — 1,203
classes over ~160K images, built on COCO's own image set. All 28 novel
classes' ids above were confirmed directly against Ultralytics' current
`lvis.yaml` — not estimated.

```python
from ultralytics import YOLO
model = YOLO("yolo26n.pt")
model.train(data="lvis.yaml", epochs=100, imgsz=640)   # auto-downloads ~20.7GB
model.train(data="coco.yaml", epochs=100, imgsz=640)   # also needed now — see below
```

Don't train on all 1,203 LVIS classes, and don't train on all 80 COCO
classes either — `finetuning/prepare_dataset.py` does both filters: every
LVIS image touching any of the 28 novel classes is kept (remapped to
20–47), and a capped random sample of COCO `train2017` images touching any
of the 20 kept classes is folded in (remapped to 0–19, Section 6) — an
image matching both sources gets its labels **merged**, not one source
overwriting the other, since LVIS reuses COCO's own JPEGs verbatim and a
"person holding keys" photo is valuable signal for both classes at once.

Expect the novel classes' per-class image counts in LVIS to vary
significantly — common items (`shoe`, `watch`, `hat`, `jacket`) likely have
hundreds to low-thousands of instances; rarer ones (`tape measure`,
`passport`, `game controller`) may be in the dozens to low-hundreds. Run
`prepare_dataset.py` first and read its printed per-source counts before
assuming any class needs a fallback.

### 3.2 Fallback: Roboflow Universe (per-class, if LVIS counts are too sparse for a given class)

Open Roboflow Universe and search for the specific class by name — don't
trust any specific dataset name/count cited in an older version of this
doc without reconfirming live; Roboflow's anti-bot protection blocks
automated fetches, so names and counts here would go stale silently.
Confirm image count, license, and annotation quality (spot-check ~50-100
images) before downloading. Standard checklist before merging any
community dataset: deduplication, class-ID remapping to this project's
0-47 scheme, and re-splitting train/val after merging rather than trusting
each source's own split (`prepare_dataset.py` already re-splits its own
LVIS+COCO merge this way; a Roboflow addition needs the same treatment
manually).

### 3.3 EDA — per-class instance counts, run before training

`prepare_dataset.py` only ever prints **image**-level counts — how many
images matched each source, how many overlapped. It never reports how
many box **instances** of each of the 48 classes actually survived the
merge, and those aren't the same thing: a class can appear in plenty of
images while still having very few boxes (one small object in a crowded
scene). `finetuning/eda.py` closes that gap — pure counting over the
label files `prepare_dataset.py` already wrote, no GPU needed:

```bash
python eda.py --dataset-dir ./lostfound --output-dir ./eda_report --imbalance-floor 50
```

Produces `class_distribution.csv` (every class, train/val/total instance
counts), `class_distribution.png` (sorted bar chart, kept-COCO vs. novel
color-coded), and `eda_summary.json` (totals, min/max instance counts, an
imbalance ratio, and two explicit flags: classes with **zero** instances —
training will never learn these at all — and classes under the
instance-count floor). `finetuning/01_dataset_prep_and_eda.ipynb` runs this
as a review gate immediately after building the merged dataset, before any
training happens: review the flagged classes and decide — raise
`--coco-sample-size`, add a Section 3.2 Roboflow fallback for a specific
sparse novel class, or accept and note the limitation — rather than
discovering a class never learned anything from a confusing eval result
after spending GPU time.

## 4. Fine-tuning mechanics

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")          # pretrained COCO weights, not from scratch
model.train(
    data="lostfound.yaml",           # 48-class combined dataset
    epochs=100,
    imgsz=640,
    freeze=10,                       # backbone frozen, neck+head train — see below
    optimizer="AdamW",
    lr0=0.001,
    patience=15,                     # early stopping on val mAP plateau
)
```

`freeze=10` is the right starting point for "small dataset, similar domain to
COCO" — Ultralytics' own guidance. The merged dataset will likely land in the
low thousands of images. Fall back to `freeze=None` if new-class mAP stays
weak after ~50 epochs — safer here since the dataset includes mixed-in COCO
examples (Section 6).

**Exact parameter breakdown** (measured directly against the real
`yolo26n.pt` checkpoint's loaded weights via `model.model.named_parameters()`,
replicating Ultralytics' own freeze logic in `engine/trainer.py`'s
`_setup_train` — not estimated):

| | Tensors | Parameters | % of total |
|---|---|---|---|
| **Total** | 366 | 2,572,280 | 100% |
| Frozen (`freeze=10`) | 99 | 1,115,744 | 43.38% |
| Trainable (`freeze=10`) | 267 | 1,456,536 | **56.62%** |

`freeze=10` does **not** freeze "the whole backbone" cleanly — Ultralytics
freezes every parameter whose name contains `model.0.` through `model.9.`
(plus `.dfl`, which matches no actual parameter in this model). In YOLO26n's
layer indexing that's blocks 0–9 — the Conv/C3k2 stem through the SPPF
layer — but **not** block 10 (`C2PSA`, the attention block that caps the
backbone), which stays trainable along with the full neck (blocks 13, 16,
17, 19, 20, 22) and the `Detect` head (block 23). More precise framing than
"freeze the backbone": *freeze the convolutional backbone (blocks 0–9);
train the backbone's attention block + full neck + detection head (blocks
10–23)*.

For comparison, same model, same method:

| `freeze=` | Frozen params | Trainable params | % trainable | What trains |
|---|---|---|---|---|
| `None` | 0 | 2,572,280 | 100.00% | Everything |
| `10` (planned) | 1,115,744 | 1,456,536 | 56.62% | C2PSA + neck + head |
| `23` | 2,262,624 | 309,656 | 12.04% | `Detect` head only |

## 5. LoRA / QLoRA — researched, not used

**Don't use LoRA/QLoRA for this fine-tune. Use `freeze=10` (above) instead.**

LoRA's entire value proposition is cutting trainable-parameter count and
optimizer memory for models where full fine-tuning doesn't fit in GPU memory —
the problem it solves at 7B+ parameter LLMs. YOLO26n is 2.57M parameters
(measured directly off the loaded checkpoint — see Section 4); full
fine-tuning already fits comfortably on a free Kaggle T4/P100. There's no
memory problem to solve. Ultralytics also has **no built-in LoRA/PEFT support**
— two GitHub issues asking for it
([#16983](https://github.com/ultralytics/ultralytics/issues/16983),
[#8728](https://github.com/ultralytics/ultralytics/issues/8728)) are
stale/closed "not planned." `freeze=N` is the only parameter-efficient
mechanism the package ships, and it's already the right tool here.

For context, not adopted here: applying LoRA to convolutional weights is an
active but young research area (LoRA-C, CoLoRA, and especially **YOLO-PEFT**,
which directly benchmarks LoRA vs. full fine-tuning on YOLO11/12, finding LoRA
*can* beat full fine-tuning on accuracy with ~44% less peak memory at the cost
of ~1.7x longer training time — one 2026 paper, not yet a settled result).
Separately, LoRA on a CLIP-style **text encoder** (YOLOE's/YOLO-World's prompt
pathway) is much more mainstream and precedented (YOLO-UniOW does exactly
this) — a legitimate stretch goal for the YOLOE side specifically, not for this
YOLO26n fine-tune.

## 6. Preventing catastrophic forgetting

Scoped to the 20 **kept** COCO classes only — not all 80. We deliberately
dropped 60 of COCO's classes (Section 2); "forgetting" how to detect a
`zebra` isn't a regression, it's the intended outcome of narrowing scope.
What actually matters is not regressing `person`/`backpack`/`cell phone`/
etc. — the classes this product still cares about.

`prepare_dataset.py` folds in a capped random sample (default 750,
`--coco-sample-size`) of COCO `train2017` images containing at least one
kept class, remapped to indices 0-19, into the merged training set. Keep
`val2017` untouched by this step — reserved for the forgetting-check set in
Section 7, built automatically by the separate
`prepare_coco_forgetting_check.py` script (no manual annotation needed,
since it's just filtering/remapping COCO's own existing ground truth, the
same operation applied to `val2017` instead of `train2017`).

## 7. Evaluation plan — two passes, not one

Two genuinely different checks, run in sequence — not interchangeable, and
not combined into one step:

- **Pass 1 — fine-tuning-specific gate.** Fast, Kaggle, no new photography,
  runs immediately after training in `finetuning/03_evaluate_finetune_checks.ipynb`:
  did this run actually learn the 28 novel classes, and did it regress the
  20 kept classes (catastrophic forgetting, Section 6)? This has to pass
  *before* investing in golden-set photography/annotation — there's no
  point capturing 300-400 images against a checkpoint that didn't train
  properly.
- **Pass 2 — the real three-way benchmark.** Needs the self-captured golden
  set (Section 7.2 below), usually run locally via `finetuning/evaluate.py`:
  **base YOLO26n** (structurally can't detect any of the 28 novel classes)
  vs. **YOLOE zero-shot** (already shipped, prompted with the same 28) vs.
  **fine-tuned YOLO26n** (this plan) — all three scored on the *exact same*
  images, which is what makes it a real benchmark rather than three
  separate, incomparable numbers.

### 7.1 Metrics

| Metric | Base YOLO26n | YOLOE zero-shot | Fine-tuned YOLO26n |
|---|---|---|---|
| mAP@0.5 / mAP@0.5:0.95 — the 28 novel classes | N/A — structural, not a measured 0 | ✅ | ✅ |
| mAP@0.5 / mAP@0.5:0.95 — the 20 kept COCO classes (forgetting check) | ✅ (reference) | — (YOLOE not prompted with these in this eval) | ✅ |
| Precision / Recall per class | ✅ | ✅ | ✅ |
| Inference latency (mean / p50 / p95, ms) | ✅ | ✅ | ✅ |
| Qualitative: same test photos, side by side | ✅ | ✅ | ✅ |

Benchmark latency in whatever environment actually serves production inference
for this app (currently: inside the `backend` container, on CPU) — not on the
Kaggle GPU training box.

### 7.2 Two held-out sets — not one, and not interchangeable

The two checks this needs have genuinely different validity requirements,
so they stay as two sets with different sourcing:

- **Golden set (self-captured)** — this is the one that answers "which
  model actually works for this product": realistic multi-object home/
  garage scenes (desk, bag, couch, countertop, workbench), not isolated
  clean shots, covering both the 28 novel classes and the 20 kept COCO
  classes together. **Scope note:** going from 3 novel classes to 28 is
  roughly a 9x increase in what the golden set needs to cover — a flat
  "150 images, ≥40 instances per class" target from the original 3-class
  plan doesn't scale to 28 classes without an unrealistic photo count
  (28 × 40 = 1,120+ instances minimum). Targets are tiered instead:

  | Tier | Classes | Target instances each |
  |---|---|---|
  | Core (most central to the product) | `keys`, `wallet`, `sunglasses`, `glasses`, `watch`, `cell phone`, `backpack` | 25–30 |
  | Common secondary | `handbag`, `person`, `earphone`, `shoe`, `jacket`, `hat`, `remote`, `laptop`, `book` | 15–20 |
  | Garage-specific | `screwdriver`, `wrench`, `pliers`, `hammer`, `toolbox`, `tape measure`, `ladder` | 10–15 |
  | Everything else kept/novel | the remaining ~25 classes | 8–12 |
  | True negatives / distractors | 15–20 images (e.g. a phone case that resembles a wallet, an empty sunglasses case, a toy screwdriver) — probes false positives, since YOLOE's open-vocabulary prompting is prone to near-matches | — |

  Realistically this needs **~300–400 images** (not ~150) given 48 classes
  and multiple objects per photo — vary lighting, background, angle,
  distance, and partial occlusion on purpose, don't shoot it all in one
  room in one sitting, and cover the garage scenes separately from the
  living-room/home ones (different lighting, different object mix). Capture
  with the same camera/resolution pipeline the app actually uses in
  production. Tag each photo's condition (lighting/location) in the
  filename at capture time — cheap now, expensive to reconstruct later, and
  it's what lets Section 7.4's narrative slice results by condition.

- **Forgetting-check set: a COCO `val2017` sample (~50–100 images), fully
  automated** — `finetuning/prepare_coco_forgetting_check.py` builds this
  directly from COCO's own existing ground truth (filtered to the 20 kept
  classes, remapped to 0-19), no manual photography or annotation needed.
  Kept separate from both the golden set and the `train2017` sample folded
  into training (Section 6) — this check's validity depends on matching
  COCO's own photo distribution (varied photographers/scenes/scale), which
  a self-captured golden set deliberately isn't.

The golden set's images get exported as **three** class-restricted
`data.yaml` variants (Section 7.3) — one per model's own label space, since
each model outputs a different class space:

| Export | nc | Class order | For |
|---|---|---|---|
| `golden_48class.yaml` | 48 | This project's own scheme (`LOSTFOUND_CLASS_NAMES`) | Fine-tuned model |
| `golden_80class_original_coco.yaml` | 80 | **Original, unmodified COCO order** (not the 0-19 remap) | Base YOLO26n |
| `golden_28class_novel_only.yaml` | 28 | The 28 novel classes only, in `NOVEL_CLASSES`' exact order | YOLOE |

Getting the base-YOLO export's class order wrong (using the 0-19 remap
instead of COCO's own) is the same class of bug as reusing one yaml across
all three — it compiles and runs, but silently misaligns indices and
produces meaningless numbers. Never reuse the fine-tune's own training
`val/` split as any of these three, or as the forgetting-check set.

**Realistic effort, so this doesn't become a surprise**: capturing 300-400
photos across genuinely varied conditions (home AND garage) takes roughly
4-7 hours, not one sitting; annotating that many multi-object images
(bounding boxes for every instance, exported three times with different
class filters) runs another 6-10 hours with review. Budget roughly 10-18
hours across several sessions for the golden set alone — meaningfully more
than the original 3-class plan's 6-10 hour estimate, since the class count
grew ~9x. Annotate in Roboflow's own UI, whose model-assisted auto-labeling
can pre-box the already-known COCO classes so manual effort concentrates on
the 28 novel ones.

### 7.3 Running each variant

```python
# Fine-tuned model — full 48-class evaluation
from ultralytics import YOLO
model = YOLO("runs/detect/train/weights/best.pt")
metrics = model.val(data="golden_set/golden_48class.yaml", split="val", imgsz=640)
print(metrics.box.map50, metrics.box.map)
print(metrics.box.maps[20:48])   # per-class array; indices 20-47 = the 28 novel classes

# Base YOLO26n — its own unmodified 80-class output, original COCO order
base = YOLO("yolo26n.pt")
coco_metrics = base.val(data="golden_set/golden_80class_original_coco.yaml", split="val", imgsz=640)
# Report the 28 novel classes as "N/A — not in COCO label space".

# YOLOE zero-shot — its own 28-class-only data.yaml, matching set_classes()'s 0-27 order
from ultralytics import YOLOE
from finetuning.prepare_dataset import NOVEL_CLASSES
yoloe = YOLOE("yoloe-26n-seg.pt")
yoloe.set_classes([name for name, _lvis_id in NOVEL_CLASSES])
yoloe_metrics = yoloe.val(data="golden_set/golden_28class_novel_only.yaml", split="val", imgsz=640)

# Forgetting check — fine-tuned model only, auto-generated set, same 48-class scheme
forgetting_metrics = model.val(data="coco_forgetting_check/lostfound.yaml", split="val", imgsz=640)
```

`finetuning/evaluate.py` implements exactly this (`--golden-set-yaml`,
`--golden-set-base-yaml`, `--golden-set-yoloe-yaml`, `--coco-sample-yaml`) —
see `finetuning/README.md` for how to export the three golden-set variants
from one Roboflow annotation project, and how to generate the forgetting-
check set automatically.

```python
# Identical latency harness for all three
import time

def benchmark(model, frame, conf=0.5, warmup=5, runs=30):
    for _ in range(warmup):
        model.predict(frame, conf=conf, verbose=False)
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model.predict(frame, conf=conf, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"mean_ms": sum(times) / len(times), "p50_ms": times[len(times)//2], "p95_ms": times[int(len(times)*0.95)]}
```

### 7.4 Presenting results

One comparison table (rows = {Base, YOLOE, Fine-tuned}, columns = Section 7.1's
metrics), one qualitative contact sheet (4–6 test photos × three annotated
outputs side by side), and the narrative: base = 0% coverage of the 28 novel
classes by construction (honest baseline, not a weakness of the test); YOLOE
= real zero-shot accuracy at zero training cost; fine-tuned = the accuracy
ceiling, at the cost of dataset curation + GPU time, validated to *not*
regress the 20 kept classes' mAP.

## 8. Free GPU: Kaggle

Kaggle Notebooks free tier: T4/P100, ~30 GPU-hours/week quota, no credit card
required — but GPU access does require phone-number verification (CPU-only
without it). Session length cap is reported inconsistently across sources
(9 vs. 12 hours) and couldn't be settled definitively here — check the
current limit in Kaggle's own notebook settings at creation time rather than
assuming either number. For a nano model on a low-thousands-of-images merged
dataset, expect roughly 1–3 hours for 50–100 epochs on a single T4, comfortably
inside either cap — but the plan below doesn't depend on fitting inside one
session, because it doesn't have to.

### 8.1 Three notebooks, not one — and why

`finetuning/train_kaggle.ipynb` (an earlier, single monolithic notebook —
download, merge, EDA, train, eval, all in one run) is gone. A single long
cell is fragile against Kaggle's session cap for no benefit: losing it
mid-training meant re-downloading LVIS (~20.7GB) from scratch, and there
was no checkpoint/resume path at all. It's replaced by three notebooks that
each commit their own Output (Kaggle's versioned-dataset mechanism) for the
next one to attach as an input, rather than re-doing prior work:

| Notebook | Does | GPU needed? |
|---|---|---|
| `finetuning/01_dataset_prep_and_eda.ipynb` | Downloads LVIS+COCO, runs `prepare_dataset.py` + `prepare_coco_forgetting_check.py`, runs `eda.py` (Section 3.3) as a review gate | No |
| `finetuning/02_train.ipynb` | Trains, checkpoint/resume-aware (Section 8.2) | Yes |
| `finetuning/03_evaluate_finetune_checks.ipynb` | Pass 1 of Section 7 — the fine-tuning-specific gate | No (CPU inference is fine for eval) |

This also means a training interruption only costs re-running `02_train.ipynb`
against `01`'s already-committed output — not re-downloading LVIS.

### 8.2 Checkpoint / resume across session cutoffs

Ultralytics writes `weights/last.pt` after every completed epoch during
`model.train()` — that part is automatic and needs no extra code. What
Kaggle doesn't do automatically is keep `/kaggle/working` around after a
session ends; it only survives if you commit it. The runbook, also inlined
in `02_train.ipynb` itself:

1. If the session clock is running low, **interrupt the training cell**
   (Stop button) rather than letting Kaggle kill it — whatever epoch last
   completed is already on disk.
2. **File → Save Version** — commits `/kaggle/working` (including
   `runs/lostfound_yolo26n/`) as this notebook's Output, which becomes a
   Kaggle Dataset.
3. Next session: open a fresh copy of `02_train.ipynb`, **Add Data** → that
   previous Output version, set the notebook's `RESUME = True` flag and
   point `PREVIOUS_RUN_DIR` at it, re-run from the top. `model.train(resume=True)`
   continues from `last.pt`'s saved epoch + optimizer state — training does
   **not** restart from `yolo26n.pt`.

## 9. Code changes needed in this repo once this is executed

`app/engines/vision/vision_engine.py` already resolves `YOLOVisionEngine`'s
weights via `FINE_TUNED_YOLO_MODEL_PATH` (checked before the legacy shared
`YOLO_MODEL_PATH`, which `YOLOEVisionEngine` also reads — use the
fine-tuned-specific var here, **not** the shared one, since both engines can
be resident in the same process via the live UI toggle; pointing the shared
var at a fine-tuned checkpoint would make `YOLOEVisionEngine` try to load it
too and silently fall back to mock detections). `ultralytics.YOLO()` accepts
a local file path directly, and its boxes/confidence flow through the exact
same `_UltralyticsScanMixin.scan_frame()` code already built for the
pretrained model — no further changes to `vision_engine.py` itself are
needed. What's actually needed:

1. Host the fine-tuned `.pt` file somewhere the container can reach (Hugging
   Face Hub recommended — free, versioned, simplest). Don't commit the raw
   `.pt` via normal git.
2. Add a build-time fetch step to `backend/Dockerfile`, mirroring the existing
   `yolo26n.pt`/`yoloe-26n-seg.pt` pre-download `RUN` steps.
3. Point `FINE_TUNED_YOLO_MODEL_PATH` at it in `docker-compose.yml`'s
   `backend` service — zero change to application code beyond what's
   already there.

**Not needed, but worth noting**: this plan's 28 novel classes were chosen
to overlap with `vision_engine.py`'s existing `HOME_ITEM_CLASSES` (YOLOE's
curated vocabulary) where practical, so the two engines' coverage stays
reasonably aligned conceptually. They remain fully independent in practice —
YOLOE is prompted with `VISION_CUSTOM_CLASSES`/`HOME_ITEM_CLASSES` at
runtime and the fine-tuned model is trained on this doc's fixed 48-class
scheme — and nothing here proposes changing `HOME_ITEM_CLASSES` itself.

## Open questions

1. LVIS vs. Roboflow as the primary dataset per novel class — fine to start
   with LVIS and fall back per-class if a given class's filtered count
   proves too sparse (run `prepare_dataset.py` first and check its printed
   counts before assuming any class needs this).
2. Checkpoint hosting (Section 9) — Hugging Face Hub assumed, confirm no
   existing team/infra preference overrides that.
3. The golden-set class scope (Section 2/7.2) is a judgment call on which
   20 COCO classes to keep and which 28 new ones to add — reasoned through
   explicitly here (small/portable/personally-owned items plausible at
   home or in a garage), but open to revision if real-world usage surfaces
   a commonly-searched-for item this list missed.
