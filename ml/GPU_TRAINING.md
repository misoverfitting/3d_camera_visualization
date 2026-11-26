# GPU training run 1 (RunPod, 2026-11-23)

Branch `feat/gpu-training`.
- **Hardware:** one A100-SXM4 80GB pod with 16 vCPUs, in US-MO-1.
- **Driver:** `ml/cloud/pod_run.sh`, run unattended. It works on the pod's local disk, saves results and logs to the network volume `tare-train` (`wpn6abgntl`, 50 GB), and the pod deletes itself at the end.
- **Results downloaded to:** `ml/data/gpu-run-1/` (gitignored).

## Cost

| Item | Cost |
| --- | --- |
| A100 pod, 10.0 h at $2.19/h | ≈ $22 |
| A failed first pod, a self-delete test pod and a CPU pod to fetch results | < $0.50 |
| Network volume, 50 GB at about $0.07/GB-month | ≈ $3.50/month until deleted |

The run hit its 10 h cap partway through the last stage, the EfficientNetV2-S classifier (20 of 30 epochs).

## Results (FoodSeg103 val unless noted)

### Segmenter (`python -m foodseg`), 80 epochs

| | mIoU | food-mask IoU | items found (IoU ≥ 0.5) | ... with the right class |
| --- | --- | --- | --- | --- |
| LR-ASPP MobileNetV3 (shipped, 6 MB) | 29.2% | 90.4% | 67.3% | 50.0% |
| DeepLabV3 MobileNetV3 | 29.7% | 90.6% | 68.8% | 50.7% |

For reference, the old colour/texture PlateItems found 47% of items at IoU 0.5 on 120 FoodSeg103 val photos. That earlier test used Vision's plate mask as the interior, so it isn't the same setup, but the gap is large.

### Classifier (`python -m foodclass`), dataset crops, top-1

| | Food-101 (dish) | FoodSeg103 (item) | Nutrition5k (item, pseudo-labels) |
| --- | --- | --- | --- |
| v1: MobileNetV3, one softmax, 6 epochs at 192 px, frozen half | 85.2 | 69.2 | 71.3 |
| Two-head MobileNetV3, 30 epochs, full fine-tune | 85.2 | 68.7 | 72.3 |
| Two-head EfficientNetV2-S, 20 of 30 epochs (shipped, 39 MB) | 87.3 | 71.7 | 72.7 |

### End to end (`python -m foodclass.endtoend`, all 7,462 FoodSeg103 val items)

Pipeline: segmenter, then app-style crops of the proposed regions, then classifier.

| Classifier | named correctly, end to end | named when found | noisy-mask crops | clean crops |
| --- | --- | --- | --- | --- |
| v1 | 48.8% | 72.5% | 61.8% | 69.4% |
| Two-head MobileNetV3 | 50.1% | 74.4% | 66.9% | 69.0% |
| Two-head EfficientNetV2-S | 51.6% | 76.6% | 70.2% | 72.1% |

What the numbers say:
- **The segmenter is the big gain.**
- **The new training makes the classifier robust to messy masks:** +5 to +8 points on noisy crops. On clean crops it barely moves.
- **The bigger backbone adds a few points everywhere.**
- **These are dataset numbers.** Real phone plates still need Ilya's FoodScans.

## Lessons for run 2

- **Classifier training was CPU-bound,** at about 180 img/s on 16 vCPUs, because fresh crops with PIL mask filters are slow. The GPU sat mostly idle. Fixes, any one of which would help:
  - pre-generate several roughened crops per item;
  - do the mask morphology on the GPU;
  - pick a pod with more vCPUs.
- **The segmenter trained fast:** 25 s per epoch.
- **The image's system Python needs `--break-system-packages`.** Install torchvision with `--no-deps`, matched to the image's torch.
- **Results on the network volume can be read without a GPU:** start a CPU pod (`--compute-type cpu`, about $0.06/h) with the volume attached.

## Next

- **Real-plate evaluation:** collect FoodScans from the phone and compare old vs new on real plates.
- **Finish the EfficientNetV2-S run** once the data loading is fixed, or train a teacher and distill it.
- **Generated training plates and a learned "not food / unsure" option.**
