# GPU training run (handoff)

Status as of 2026-11-24. Branch `feat/gpu-training`.

## Goal

Improve on-device food ID accuracy on real plates. First run budget: **under $50**, on RunPod.

Ilya's review (`docs/ON_DEVICE_PERFORMANCE_DIAGNOSIS.md`) agrees on the priorities:
1. **Segmentation limits accuracy first.** The hand-built PlateItems misses rice on white plates, shadows and similar cases.
2. **Dish and ingredient labels compete in one softmax.**

## Done (smoke-tested on a Mac with small subsets)

### `python -m foodseg train|eval|export` (segmenter)
- **Data and model:** FoodSeg103 semantic segmentation, 104 classes. COCO-pretrained LR-ASPP (`--arch lraspp`) or DeepLabV3 (`--arch deeplabv3`) on MobileNetV3.
- **Eval metrics:** mIoU, food-mask IoU, and item recall at IoU 0.5 (class-agnostic and with class). For reference, the old PlateItems found 47% of items at IoU 0.5 on 120 FoodSeg103 val images, using Vision's plate mask as the interior.
- **Export:** Core ML outputs `food` and `classes`, written to `Assets/FoodSegmenter.mlmodelc`. Export needs macOS (`xcrun coremlcompiler`, `mlmodel.predict`).

### `python -m foodclass train|export` (classifier)
- **Two heads:** `dish` (Food-101) and `item` (everything else).
- **Backbones:** `--arch mobilenet_v3_large | efficientnet_v2_s | convnext_tiny`.
- **Training setup:** full fine-tune with bf16 on CUDA.
- **Augmentations:**
  - FoodSeg103 crops are cut fresh every epoch from `regions.csv` with roughened masks (`foodclass/crops.py`).
  - Food-101 photos are sometimes dimmed outside an ellipse, like the app's whole-plate input.
- **Export:** Core ML outputs `dish` and `item`. The labels file has `format: two-head`. Export needs macOS.

## Not done yet

1. **RunPod access:** the account exists, but the API key saved in `~/.runpod/config.toml` on Morgan's Mac returned 401. A new key with read/write access is needed. The old key was printed in a session transcript, so delete it.
2. **Pod script:**
   - Image: a PyTorch CUDA image, on one A100 80GB (about $1.2–2/h).
   - Steps: `pip install torch torchvision pyarrow scipy pillow huggingface_hub`, clone the repo, then download from Hugging Face: `ethz/food101` → `ml/data/food101` and `EduardoPacheco/FoodSeg103` → `ml/data/FoodSeg103`.
   - Then run `python -m foodclass prepare`, which writes `manifest.csv` and `regions.csv`.
3. **Nutrition5k crops:** they exist only on Morgan's Mac (`ml/data/foodclass/images/nutrition5k`, 60 MB, plus their manifest rows). Upload them to the pod and append the rows, or train the first run without them (losing about 75 Nutrition5k-only labels).
4. **Runs, in order:**
   - Segmenter: `python -m foodseg train --arch lraspp --epochs 80 --batch 32 --workers 16`. Optionally `--arch deeplabv3 --out dl.pt` as well.
   - Classifier: `python -m foodclass train --arch mobilenet_v3_large --epochs 30 --batch 256 --workers 16`. If budget allows, also `--arch efficientnet_v2_s --out effv2.pt`.
   - Stop the pod as soon as the runs finish. Download the checkpoints and `*-metrics.json`.
5. **On a Mac:**
   - Export both models.
   - Teach the app the new formats:
     - `LocalFoodID.swift` reads the `dish` and `item` outputs via `VNCoreMLFeatureValueObservation`.
     - A new segmenter path replaces `PlateItems.segment` when `FoodSegmenter.mlmodelc` is present: food mask = `food > 0.5` inside the rim interior, items = connected components per `classes` value, with points from the existing distance transform.
   - Then compare old vs new on Ilya's FoodScans.
