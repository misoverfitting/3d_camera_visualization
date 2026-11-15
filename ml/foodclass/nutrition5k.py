"""Item crops from Nutrition5k (CC BY 4.0) without any manual or LLM labeling.

Nutrition5k gives each dish's ingredient list with grams, but not where each one is.
1. `plateitems-cli` (the app's own segmentation) proposes item regions on every
   overhead photo, so crops match what the phone produces.
2. A classifier trained on Food-101 + FoodSeg103 scores each region; the region gets the
   dish's *listed* visible ingredient it scores highest on (choosing among 2-4 known
   ingredients is far easier than open recognition). Ingredients our classes lack
   (arugula, quinoa...) are scored through proxy classes and become new classes.
3. Regions that are ambiguous between listed ingredients, or tiny, are dropped.

Official splits are kept: rgb_test dishes go to "val", so metrics.json reports accuracy
on our setting (against these pseudo-labels; spot-check overlays before trusting it).
"""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from autolabel.verify import crop

from .train import SIZE, MEAN, STD, build_model, device

# Visible ingredients: at least this many grams and this share of the dish.
MIN_GRAMS, MIN_SHARE = 10.0, 0.08
MIN_REGION_FRACTION = 0.008
MIN_SCORE, MIN_MARGIN = 0.04, 2.0

# Seasonings, oils, dressings and placeholders: on the plate but not a separate item.
NOT_ITEMS = {"olive oil", "salt", "pepper", "vinegar", "lemon juice", "garlic", "mustard", "thyme", "parsley", "shallots",
             "water", "deprecated", "", "plate only", "vegetable oil", "honey", "syrup", "caesar dressing", "vinaigrette",
             "mayonnaise", "ketchup", "apple cider", "grapefruit juice", "buttermilk", "almond milk", "cream", "jams",
             "chia seeds", "milk"}

# Nutrition5k ingredient -> our label (an existing class or a new one).
CANONICAL = {
    "scrambled eggs": "scrambled eggs", "egg whites": "scrambled eggs", "eggs": "egg", "omelets": "omelette",
    "poached eggs": "egg", "cheese pizza": "pizza", "pepperoni pizza": "pizza", "berries": "berries",
    "cantaloupe": "melon", "honeydew melons": "melon", "beef": "beef", "brussels sprouts": "brussels sprouts",
    "cherry tomatoes": "tomato", "tomatoes": "tomato", "spinach (raw)": "spinach", "spinach (cooked)": "spinach",
    "mixed greens": "salad", "garden salad": "salad", "arugula": "arugula", "roasted potatoes": "potato",
    "potatoes": "potato", "mashed potatoes": "mashed potatoes", "hash browns": "hash browns", "cucumbers": "cucumber",
    "squash": "squash", "chayote squash": "squash", "zucchini": "zucchini", "white rice": "rice", "country rice": "rice",
    "brown rice": "brown rice", "wild rice": "brown rice", "bell peppers": "pepper", "salmon": "grilled salmon",
    "corn on the cob": "corn", "almonds": "almond", "nuts": "nuts", "walnuts": "walnut", "cashews": "cashew",
    "grapes": "grape", "watermelon": "watermelon", "mushroom": "white button mushroom", "orange": "orange",
    "mandarin oranges": "orange", "yam": "sweet potato", "sweet potato": "sweet potato", "chicken thighs": "chicken",
    "chicken breast": "chicken", "grilled chicken": "chicken", "roast chicken": "chicken", "chicken": "chicken",
    "chicken apple sausage": "sausage", "sausage": "sausage", "turkey bacon": "bacon", "bacon": "bacon",
    "pizza": "pizza", "tortilla": "tortilla", "steak": "steak", "ham": "ham", "strawberries": "strawberry",
    "raspberries": "raspberry", "blackberries": "berries", "blueberries": "blueberry", "wheat berry": "grain salad",
    "quinoa": "quinoa", "couscous": "couscous", "bulgur": "grain salad", "millet": "grain salad", "barley": "grain salad",
    "oatmeal": "oatmeal", "onions": "onion", "eggplant": "eggplant", "lettuce": "lettuce", "kale": "kale",
    "chard": "kale", "mustard greens": "kale", "collards": "kale", "tatsoi": "bok choy", "bok choy": "bok choy",
    "lemon": "lemon", "corn": "corn", "cheese": "cheese", "mozzarella cheese": "cheese", "parmesan cheese": "cheese",
    "feta cheese": "cheese", "cottage cheese": "cottage cheese", "cream cheese": "cheese", "kiwi": "kiwi",
    "banana": "banana", "potato chips": "potato chips", "salsa": "salsa", "shrimp": "shrimp", "pears": "pear",
    "bread": "bread", "bagels": "bagel", "flatbread": "bread", "multigrain bread": "bread", "sourdough bread": "bread",
    "toast": "bread", "noodles": "noodles", "rice noodles": "noodles", "pasta": "pasta", "fried rice": "fried rice",
    "bean(seed)": "beans", "white beans": "beans", "black beans": "beans", "chickpeas": "chickpeas", "lentils": "lentils",
    "greek yogurt": "yogurt", "yogurt": "yogurt", "frozen yogurt": "frozen yogurt", "ground turkey": "ground meat",
    "tuna": "fish", "fish": "fish", "cod": "fish", "radishes": "white radish", "turnips": "white radish",
    "brownies": "brownie", "waffles": "waffles", "celery": "celery", "cabbage": "cabbage", "kimchi": "kimchi",
    "artichokes": "artichoke", "hummus": "hummus", "dumplings": "dumplings", "parsnips": "carrot", "baby carrots": "carrot",
    "carrot": "carrot", "broccoli": "broccoli", "cauliflower": "cauliflower", "olives": "olives", "pineapple": "pineapple",
    "pork": "pork", "apple": "apple", "tofu": "tofu", "green beans": "green beans", "asparagus": "asparagus",
    "avocado": "avocado", "caesar salad": "caesar salad", "greek salad": "greek salad", "snow peas": "snow peas",
    "cookies": "biscuit", "biscuits": "biscuit", "granola": "granola", "figs": "fig", "edamame": "edamame",
    "lasagna": "lasagna", "spaghetti": "pasta", "crepes": "pancakes", "tacos": "tacos", "lamb": "lamb",
    "brisket": "beef", "turkey": "chicken", "mangos": "mango", "beets": "beet salad", "chilaquiles": "nachos",
}

# New labels scored through existing classes (the max over the listed proxies).
PROXIES = {
    "scrambled eggs": ["omelette", "egg"], "berries": ["strawberry", "blueberry", "raspberry"],
    "beef": ["steak", "fried meat", "prime rib"], "brussels sprouts": ["cabbage", "broccoli"],
    "spinach": ["lettuce", "bok choy", "herbs"], "arugula": ["lettuce", "herbs", "salad"],
    "mashed potatoes": ["potato"], "hash browns": ["potato", "french fries"], "squash": ["pumpkin", "cucumber"],
    "zucchini": ["cucumber", "pumpkin"], "brown rice": ["rice", "fried rice"], "nuts": ["almond", "walnut", "cashew", "peanut"],
    "sweet potato": ["pumpkin", "potato"], "bacon": ["fried meat", "pork"], "tortilla": ["bread", "tacos"],
    "ham": ["pork", "fried meat"], "grain salad": ["rice", "fried rice", "risotto"], "quinoa": ["rice", "fried rice"],
    "couscous": ["rice", "fried rice"], "oatmeal": ["pudding", "rice"], "kale": ["lettuce", "bok choy", "herbs"],
    "cottage cheese": ["cheese", "ice cream"], "potato chips": ["french fries", "potato"], "salsa": ["sauce", "tomato"],
    "bagel": ["bread", "donuts"], "beans": ["red beans", "soy", "edamame"], "chickpeas": ["red beans", "soy", "peanut"],
    "lentils": ["red beans", "soy"], "yogurt": ["frozen yogurt", "ice cream", "pudding"], "ground meat": ["fried meat", "pork"],
    "brownie": ["chocolate cake", "chocolate"], "kimchi": ["cabbage", "sauce"], "artichoke": ["herbs", "bok choy"],
    "granola": ["almond", "biscuit", "popcorn"],
}


def visible_ingredients(root: Path) -> dict[str, list[str]]:
    dishes: dict[str, list[str]] = {}
    for name in ("dish_metadata_cafe1.csv", "dish_metadata_cafe2.csv"):
        for row in csv.reader((root / "metadata" / name).open()):
            total, cells, labels = float(row[2]), row[6:], []
            for i in range(0, len(cells) - 6, 7):
                ingredient, grams = cells[i + 1].strip().lower(), float(cells[i + 2])
                if ingredient in NOT_ITEMS or grams < MIN_GRAMS or grams < MIN_SHARE * total:
                    continue
                label = CANONICAL.get(ingredient)
                labels.append(label if label else f"?{ingredient}")
            dishes[row[0]] = list(dict.fromkeys(labels))
    return dishes


def propose(root: Path, cli: Path) -> Path:
    """Runs the app's segmentation over every overhead photo (resumable)."""
    out = root / "proposals"
    binary = cli / ".build" / "release" / "plateitems"
    if not binary.exists():
        subprocess.run(["swift", "build", "-c", "release"], cwd=cli, check=True)
    images = sorted(str(p) for p in (root / "overhead").glob("*/rgb.png"))
    for start in range(0, len(images), 500):  # Keep argument lists short.
        subprocess.run([str(binary), *images[start:start + 500], "--out", str(out)], check=True)
    return out


@torch.no_grad()
def pseudo_label(root: Path, work: Path, cli: Path) -> None:
    proposals = propose(root, cli)
    # The teacher is the Food-101 + FoodSeg103 model. Its checkpoint is set aside so the
    # next `train` starts fresh on the larger label set instead of resuming it.
    teacher = work / "checkpoint-teacher.pt"
    if not teacher.exists():
        (work / "checkpoint.pt").rename(teacher)
    state = torch.load(teacher, map_location="cpu")
    labels: list[str] = state["labels"]
    index = {name: i for i, name in enumerate(labels)}
    dev = device()
    model = build_model(len(labels), pretrained=False)
    model.load_state_dict(state["model"])
    model = model.to(dev).eval()
    mean, std = torch.tensor(MEAN).view(3, 1, 1), torch.tensor(STD).view(3, 1, 1)

    def proxies(label: str) -> list[int]:
        names = [label] if label in index else PROXIES.get(label, [])
        return [index[n] for n in names if n in index]

    dishes = visible_ingredients(root)
    test = set((root / "dish_ids" / "splits" / "rgb_test_ids.txt").read_text().split())
    rows, reasons, kept = [], Counter(), Counter()
    for meta in sorted(proposals.glob("*.json")):
        dish = meta.stem
        listed = [l for l in dishes.get(dish, []) if not l.startswith("?") and proxies(l)]
        unknown = [l for l in dishes.get(dish, []) if l.startswith("?")]
        info = json.loads(meta.read_text())
        if not info["regions"]:
            continue
        if not listed:
            reasons["no_known_visible_ingredient"] += len(info["regions"]); continue
        photo = Image.open(root / "overhead" / dish / "rgb.png").convert("RGB")
        regions = np.asarray(Image.open(meta.with_suffix(".png")).resize(photo.size, Image.Resampling.NEAREST))
        crops, boxes = [], []
        for k, region in enumerate(info["regions"]):
            mask = regions == k + 1
            if mask.mean() < MIN_REGION_FRACTION:
                reasons["too_small"] += 1; continue
            item = crop(photo, region["box"], mask)
            crops.append(item); boxes.append(k)
        if not crops:
            continue
        batch = torch.stack([(torch.from_numpy(np.asarray(c.resize((SIZE, SIZE), Image.Resampling.BILINEAR), dtype=np.float32) / 255)
                              .permute(2, 0, 1) - mean) / std for c in crops]).to(dev)
        probs = torch.softmax(model(batch), dim=1).cpu().numpy()
        split = "val" if dish in test else "train"
        for item, k, p in zip(crops, boxes, probs):
            scores = sorted(((max(p[i] for i in proxies(l)), l) for l in listed), reverse=True)
            best, label = scores[0]
            second = scores[1][0] if len(scores) > 1 else 0.0
            # An unlisted-class ingredient on the plate could be this region: demand a clear win.
            margin = MIN_MARGIN * (1.5 if unknown else 1.0)
            if best < MIN_SCORE:
                reasons["low_score"] += 1; continue
            if second and best < margin * second:
                reasons["ambiguous"] += 1; continue
            path = work / "images" / "nutrition5k" / split / f"{dish}_{k}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            item.save(path, quality=90)
            rows.append({"path": str(path.relative_to(work)), "label": label, "source": "nutrition5k", "split": split})
            kept[label] += 1

    manifest = work / "manifest.csv"
    existing = [r for r in csv.DictReader(manifest.open()) if r["source"] != "nutrition5k"]
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "label", "source", "split"])
        writer.writeheader(); writer.writerows(existing + rows)
    report = {"crops": len(rows), "train": sum(r["split"] == "train" for r in rows), "val": sum(r["split"] == "val" for r in rows),
              "dropped": dict(reasons), "labels": dict(kept.most_common())}
    (work / "nutrition5k.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "labels"}), f"{len(kept)} labels")
