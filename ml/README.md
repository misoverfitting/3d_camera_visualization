# Food auto-labeling (`ml/autolabel`)

Builds the food training set from images without manual labeling. Claude Haiku 5.5 labels every image twice, SAM turns the labels into pixel masks, Haiku re-checks each item, and a stronger model audits a random sample so label quality is measured rather than assumed.

The output is a COCO dataset (boxes, masks, food classes), per-image mask PNGs and classifier crops. These are the inputs for the item segmenter and the item classifier in `tare_impl.md` §6.9–6.11.

```
images ──ingest──▶ workspace ──label ×2──▶ consensus ──segment (SAM)──▶ verify ──▶ decide ──▶ export
 (app captures,       (upright,        (Haiku; pass 2      (match passes)   (box + points    (Haiku     (agreement
  rim tests,          ≤1568 px)        sees a mirrored                       → mask)          per item)   rules)
  backend uploads,                     copy)                                                              audit (Opus,
  any photos)                                                                                             random sample)
```

## Quick start

```sh
cd ml
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt            # torch + transformers only needed for masks
export ANTHROPIC_API_KEY=sk-ant-...

# 1. Collect images. Paths can be folders, app recordings (folder or exported .zip),
#    RimTests/<stamp> folders, the backend's food_uploads folder, or single images.
python -m autolabel ingest ~/Desktop/plates ~/Downloads/scan-*.zip

# 2. Try a few images first, directly (results in seconds):
python -m autolabel label --limit 20
python -m autolabel segment            # --segmenter none for boxes only
python -m autolabel verify
python -m autolabel audit --fraction 1 --max 10
python -m autolabel export ../data/food-dataset-trial
open ../data/food-dataset-trial/report.md

# 3. The full set, as Message Batches (half price, usually done within an hour):
python -m autolabel run ../data/food-dataset --batch
```

`python -m autolabel status` shows progress, pending batches and the cost so far. Every stage is resumable: it only does work that is missing. Re-running after a crash, an expired batch or newly ingested images picks up where things stopped.

## Where images come from

| Source | How to get it | Notes |
| --- | --- | --- |
| App scan recordings | "Export last recording" in Scan mode (zip), or Files → Tare Capture → TareScans | 8 frames per capture by default (`--frames`): the most overhead one, then the most different angles. Rotated upright. |
| Rim tests | Files → Tare Capture → RimTests/<stamp> | Both photos. |
| Plate demo photos | Run the backend with `FOOD_SAVE_UPLOADS=1`; every identification saves its plate crop to `data/food_uploads/<date>/` | Each upload is its own split group. Off by default: real user data needs consent (`tare_impl.md` Q5). |
| Any photo folders | Point `ingest` at the folder | Grouped by folder name, or set `--group` (use the dish ID when you know it). |

**Groups matter.** Splits are by group, never by image (`tare_impl.md` §9.1), so all photos of one dish or capture land in the same split. Pass `--golden golden.txt` (one group ID per line) to `export` or `run` to hold those groups out as the golden set; they never appear in train/val.

## How labels earn trust without a human

1. **Two independent passes.** Pass 2 sees a mirrored copy, so it isn't a repeat of pass 1. Items are matched by box overlap (IoU ≥ 0.4).
2. **Masks.** SAM gets each item's box plus Claude's interior points. Masks that spill far outside the box (usually the plate) or cover little of it are rejected. Overlaps go to the smaller item.
3. **Verification.** Haiku sees each item's crop with everything else dimmed (the same crop the on-device classifier will get). It confirms the class and grades the mask.
4. **Decision** (`verify.decide`). An item is accepted when:
   - both passes and the verifier agree on the class; or
   - the passes disagree and the verifier picks one of their two classes; or
   - only one pass saw it, and the verifier confirms both its class and its label.

   Everything else is rejected with a reason, counted in the report.
5. **Complete images only for detection.** If any real food item on an image was rejected, that image is left out of the detection/segmentation annotations, since the model would learn that the unlabeled food is background. Its accepted items still become classifier crops.
6. **Audit.** Claude Opus 5.5 grades a random sample of accepted labels drawn over the image: is it food, is the class right, is the box OK, and how many items were missed. The report turns this into label precision with a 95% confidence interval, plus a recall estimate. Rerun the audit after any prompt or taxonomy change.

## Output

```
<out>/images/<id>.jpg
<out>/masks/<id>.png                  pixel value n = annotation n of that image (0 = background)
<out>/annotations/{train,val,test,golden}.json   COCO instances; segmentation is uncompressed RLE
<out>/classify/<split>/<class>/<id>_<k>.jpg      crops for the classifier (padded 12.5%, background dimmed)
<out>/dataset.json                    versions, models, counts, the split of every group, audit, cost
<out>/report.md                       the same, readable
```

Annotations also carry Claude's free-text `label` and the pass `agreement`, so items can be re-mapped to a new taxonomy without re-labeling.

## Taxonomy

`autolabel/taxonomy.toml` holds a starter list of about 150 classes in 11 categories, plus `other_food`. **This is not a product decision.** `tare_impl.md` Q1 (fixed list vs open vocabulary) and Q3 (sauce policy) are still open; the file defaults to sauces as separate items, per §6.10. When the list changes, bump `version` in the file: label results from the old version are redone on the next run.

## Cost (rough)

Estimated from typical token counts; `status` and `report.md` show the actual cost from logged usage.

| Step | Per image |
| --- | --- |
| 2 Haiku label passes | ~$0.002 |
| Haiku verification (~3 items) | ~$0.0015 |
| Total direct | ~$0.004 |
| Total with `--batch` | ~$0.002 |

10,000 images cost roughly $20–40. The Opus audit costs about $0.05–0.10 per audited image and is capped at 50 images per run (`--max`).

## Configuration

`autolabel/config.py` holds models, effort levels, prompt versions and thresholds. You can override these with environment variables:

| Variable | Default |
| --- | --- |
| `AUTOLABEL_MODEL` | `claude-haiku-5-5` |
| `AUTOLABEL_VERIFY_MODEL` | same as `AUTOLABEL_MODEL` |
| `AUTOLABEL_AUDIT_MODEL` | `claude-opus-5-5` |
| `AUTOLABEL_EFFORT` | `medium` |
| `AUTOLABEL_VERIFY_EFFORT` | `low` |
| `AUTOLABEL_AUDIT_EFFORT` | `high` |

Changing a prompt means bumping its `*_PROMPT_VERSION`, so old results are redone.

## Tests

```sh
pip install -r requirements-dev.txt
python -m pytest tests -q
```

The tests replace Claude and SAM with fakes that read synthetic meal images, then run every stage for real:
- both label passes, including the mirrored one;
- consensus, masks and verification;
- batch submit and collect;
- export with the golden split, RLE decoding and crops.

They also cover the decision rules, recording ingest with rotation and frame selection, and split stability.

Real SAM (`facebook/sam-vit-base`, on Apple MPS) was also checked on a synthetic meal: mask IoU 0.997–0.999 per item.

## Next: training

`classify/` is ready for an ImageFolder classifier: MobileNetV3 / EfficientNet-Lite fine-tuned from Food-101, per §6.11. `annotations/*.json` with `masks/` is ready for fine-tuning a promptable segmenter or an instance-segmentation model, per §6.9. Both then convert to Core ML with `coremltools`.

# On-device classifier (`ml/foodclass`)

Names each food item on the phone, with no network. The app finds the items itself (`PlateItems.swift`: pixels inside the rim that differ from the bare plate in colour or texture, split into items by colour and connectivity), crops each one the same way the auto-labeller does, and runs this classifier on the crop.

```sh
cd ml && . .venv/bin/activate            # Python 3.12: coremltools does not support 3.13 yet
pip install -r requirements.txt coremltools torchvision pyarrow huggingface_hub
python -c "from huggingface_hub import snapshot_download as d; d('ethz/food101', repo_type='dataset', local_dir='data/food101'); d('EduardoPacheco/FoodSeg103', repo_type='dataset', local_dir='data/FoodSeg103')"

python -m foodclass prepare               # ~100k training images in data/foodclass
python -m foodclass train                 # MobileNetV3-Large from ImageNet weights; resumable
python -m foodclass export                # Core ML → mobile/modules/tare-capture/ios/Assets/FoodClassifier.mlmodelc
```

**Adding Nutrition5k.** These are overhead cafeteria plates under CC BY 4.0, with each dish's ingredients and grams but no outlines. No manual or LLM labelling is needed:

```sh
python -m foodclass.nutrition5k_download  # ~3 GB: overhead RGB + depth, metadata, official splits
python -m foodclass train                 # 1. teacher on Food-101 + FoodSeg103
python -m foodclass nutrition5k           # 2. app segmentation proposes items; teacher picks among each dish's listed ingredients
python -m foodclass train                 # 3. student on all three (starts fresh: new labels)
python -m foodclass export
```

Step 2 runs `plateitems-cli` (the app's `PlateItems` segmentation plus Vision's plate mask, macOS 14+). Each region gets the dish's *listed* visible ingredient that the teacher scores highest; ingredients outside our classes, such as arugula or quinoa, are scored through proxy classes (`nutrition5k.PROXIES`). Ambiguous or tiny regions are dropped, with the counts per reason in `data/foodclass/nutrition5k.json`. Official test dishes become `val`, so `metrics.json` reports a `nutrition5k` score in our setting. That score is measured against pseudo-labels, so spot-check crops before trusting it.

Then rebuild the app. The model goes into the app bundle through the pod's `TareCaptureAssets` resource bundle.

| Source | Used as | Licence |
| --- | --- | --- |
| Food-101 (`ethz/food101`) | Whole-dish photos, 101 classes | Research use. Fine for internal prototypes, not for shipping |
| FoodSeg103 (`EduardoPacheco/FoodSeg103`) | One crop per ingredient on the plate, background dimmed like the app's crops | Research dataset; the Hugging Face copy is marked Apache-2.0, check the original terms before shipping |
| Nutrition5k (`gs://nutrition5k_dataset`) | Pseudo-labelled item crops from overhead plates (see above) | CC BY 4.0: commercial use with attribution |
| Auto-labelled dataset | `prepare --extra <export>/classify`: our own crops; their test and golden splits stay out | Ours |

Labels from the two datasets are merged by name (`foodclass/labels.py`). Each label also gets the app's coarse category from keyword rules.

`data/foodclass/metrics.json` records validation top-1 and top-5 per source after every epoch. Validation on FoodSeg103 crops is the number that matters for the app, because it uses the same kind of item crop. Food-101 validation uses whole-dish photos.
