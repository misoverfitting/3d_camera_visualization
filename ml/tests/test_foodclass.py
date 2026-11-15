from foodclass.labels import category, normalize


def test_labels_merge_across_datasets():
    assert normalize("french_fries") == normalize("french fries") == "french fries"
    assert normalize("hamburg") == "hamburger"
    assert normalize("chicken duck") == "chicken"


def test_categories():
    assert category("steak") == "protein"
    assert category("fried rice") == "starch"
    assert category("broccoli") == "vegetable"
    assert category("chocolate cake") == "dessert"
    assert category("ice cream") == "dessert"
    assert category("banana") == "fruit"
    assert category("coffee") == "drink"
    assert category("sauce") == "sauce"
    assert category("cheese") == "dairy"
