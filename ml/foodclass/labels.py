"""Merged label space for the on-device item classifier.

Food-101 names whole dishes ("fried_rice", "steak"); FoodSeg103 names the
ingredients you see as separate items on a plate ("rice", "broccoli"). Names are
normalized (lowercase, spaces) and the few duplicates merged, so "french_fries"
and "french fries" are one class.

Each label also gets the coarse category the app shows (same set as the backend's
food_vision.Category), from keyword rules below.
"""
from __future__ import annotations

ALIASES = {
    "hamburg": "hamburger",
    "chicken duck": "chicken",
    "cheese butter": "cheese",
    "french beans": "green beans",
    "rape": "bok choy",
    "hanamaki baozi": "steamed bun",
    "wonton dumplings": "dumplings",
    "celery stick": "celery",
    "cilantro mint": "herbs",
}
# FoodSeg103 classes that are not a food item to label.
SKIP = {"background", "other ingredients"}


def normalize(name: str) -> str:
    name = name.replace("_", " ").strip().lower()
    return ALIASES.get(name, name)


CATEGORY_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("drink", ("wine", "milkshake", "coffee", "juice", "milk", "tea", "soda")),
    ("sauce", ("sauce", "guacamole", "hummus", "dressing", "gravy", "salsa")),
    ("dessert", ("cake", "pie", "pudding", "ice cream", "chocolate", "candy", "biscuit", "cookie", "tart", "donut",
                 "macaron", "churros", "cannoli", "tiramisu", "baklava", "creme brulee", "cheesecake", "mousse",
                 "panna cotta", "beignets", "waffles", "pancakes", "french toast", "cup cakes", "frozen yogurt",
                 "strawberry shortcake", "bread pudding", "popcorn", "crepes")),
    ("protein", ("steak", "pork", "chicken", "sausage", "meat", "lamb", "crab", "fish", "shellfish", "shrimp", "egg",
                 "tofu", "beef", "ribs", "salmon", "tuna", "scallops", "mussels", "oysters", "lobster", "duck",
                 "sashimi", "ceviche", "tartare", "calamari", "clam", "escargots", "foie gras", "wings", "filet",
                 "prime rib", "pulled pork", "hot dog", "hamburger", "bulgogi", "kebab", "omelette", "benedict",
                 "huevos", "deviled", "red beans", "soy", "edamame", "peanut", "almond", "cashew", "walnut",
                 "falafel", "takoyaki", "carpaccio", "chop", "fried calamari")),
    ("dairy", ("cheese", "yogurt", "butter", "cream")),
    ("fruit", ("apple", "date", "apricot", "avocado", "banana", "strawberry", "cherry", "blueberry", "raspberry",
               "mango", "olives", "peach", "lemon", "pear", "fig", "pineapple", "grape", "kiwi", "melon", "orange",
               "watermelon", "cranberries", "fruit")),
    ("starch", ("bread", "corn", "pizza", "bun", "dumpling", "pasta", "noodle", "rice", "potato", "fries", "spaghetti",
                "ravioli", "gnocchi", "lasagna", "macaroni", "risotto", "paella", "ramen", "pho", "udon", "pad thai",
                "nachos", "tacos", "burrito", "quesadilla", "sandwich", "bruschetta", "garlic bread", "toast",
                "bagel", "samosa", "spring rolls", "gyoza", "sushi", "bibimbap", "poutine", "hash", "onion rings",
                "croque", "panini", "pumpkin", "breakfast burrito")),
    ("vegetable", ("eggplant", "garlic", "cauliflower", "tomato", "kelp", "seaweed", "onion", "bok choy", "ginger",
                   "okra", "lettuce", "cucumber", "radish", "carrot", "asparagus", "bamboo", "broccoli", "celery",
                   "herbs", "peas", "cabbage", "sprouts", "pepper", "beans", "mushroom", "shiitake", "enoki", "salad",
                   "beet", "caprese", "edamame", "seaweed salad", "greek salad", "caesar")),
]


def category(label: str) -> str:
    # Most specific rules first for a few dishes the keyword order would misplace.
    special = {"french fries": "starch", "sweet potato": "starch", "egg tart": "dessert", "ice cream": "dessert",
               "chocolate": "dessert", "soup": "other", "hot and sour soup": "other", "miso soup": "other",
               "french onion soup": "other", "clam chowder": "other", "lobster bisque": "other", "salad": "vegetable",
               "caesar salad": "vegetable", "greek salad": "vegetable", "beet salad": "vegetable",
               "seaweed salad": "vegetable", "caprese salad": "vegetable", "fried rice": "starch",
               "chicken curry": "protein", "chicken quesadilla": "starch", "lobster roll sandwich": "starch",
               "club sandwich": "starch", "pulled pork sandwich": "starch", "grilled cheese sandwich": "starch",
               "cheese plate": "dairy", "macaroni and cheese": "starch", "shrimp and grits": "protein",
               "eggs benedict": "protein", "egg": "protein", "peking duck": "protein", "fish and chips": "protein",
               "red beans": "protein", "pumpkin": "vegetable", "corn": "vegetable", "potato": "starch",
               "dumplings": "starch", "steamed bun": "starch", "spring onion": "vegetable", "pepper": "vegetable",
               "garlic bread": "starch", "onion rings": "starch", "green beans": "vegetable", "candy": "dessert",
               "date": "fruit", "olives": "fruit", "butter": "dairy", "cheese": "dairy", "milk": "drink",
               "pancakes": "dessert", "waffles": "dessert", "breakfast burrito": "starch", "tofu": "protein",
               "chocolate mousse": "dessert", "crab cakes": "protein", "chocolate cake": "dessert", "carrot cake": "dessert",
               "cheesecake": "dessert", "apple pie": "dessert", "strawberry shortcake": "dessert"}
    if label in special:
        return special[label]
    padded = f" {label} "
    for name, words in CATEGORY_KEYWORDS:
        # Whole words only: "tea" must not match "steak".
        if any(f" {w} " in padded or f" {w}s " in padded for w in words):
            return name
    return "other"
