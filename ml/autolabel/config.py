"""Models, prompt versions and thresholds for the auto-labeling pipeline."""
from __future__ import annotations

import os

# Labeling and per-item verification run on Haiku; the audit uses a stronger model to
# estimate label quality on a sample, so nobody has to label by hand.
LABEL_MODEL = os.environ.get("AUTOLABEL_MODEL", "claude-haiku-5-5")
VERIFY_MODEL = os.environ.get("AUTOLABEL_VERIFY_MODEL", LABEL_MODEL)
AUDIT_MODEL = os.environ.get("AUTOLABEL_AUDIT_MODEL", "claude-opus-5-5")
LABEL_EFFORT = os.environ.get("AUTOLABEL_EFFORT", "medium")
VERIFY_EFFORT = os.environ.get("AUTOLABEL_VERIFY_EFFORT", "low")
AUDIT_EFFORT = os.environ.get("AUTOLABEL_AUDIT_EFFORT", "high")

# Stored with every result; results from an older prompt are redone.
LABEL_PROMPT_VERSION = "label-v1"
VERIFY_PROMPT_VERSION = "verify-v1"
AUDIT_PROMPT_VERSION = "audit-v1"

# Long image side sent to Claude (larger images are downscaled by the API anyway).
MAX_SIDE_PX = 1568

# Consensus between the two label passes: same item when boxes overlap this much.
MATCH_IOU = 0.4
# Mask sanity (segment stage): fraction of the image, and mask/box area ratio limits.
MIN_MASK_FRACTION = 0.002
MAX_MASK_TO_BOX = 1.6
MIN_MASK_TO_BOX = 0.15
# Classification crops (tare_impl.md §6.11): padding and how much to dim the background.
CROP_PADDING = 0.125
CROP_BACKGROUND_DIM = 0.35

# Message Batches limit is 256 MB per batch; stay well under it.
MAX_BATCH_BYTES = 180 * 1024 * 1024

# Split by group (dish or capture), never by image (tare_impl.md §9.1).
SPLIT_FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}

# Per-million-token prices for the cost report (USD; batches are half price).
PRICES = {
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-opus-5-5": (4.00, 20.00),
}
