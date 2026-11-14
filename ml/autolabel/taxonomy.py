"""The closed set of food classes labelers choose from (taxonomy.toml)."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).with_name("taxonomy.toml")
OTHER = "other_food"


@dataclass(frozen=True)
class Taxonomy:
    version: str
    sauce_policy: str
    category_of: dict[str, str]  # class -> category, in file order

    @property
    def classes(self) -> list[str]:
        return list(self.category_of)

    @property
    def categories(self) -> list[str]:
        return list(dict.fromkeys(self.category_of.values()))

    def prompt_block(self) -> str:
        lines = [f"- {category}: {', '.join(c for c, cat in self.category_of.items() if cat == category)}"
                 for category in self.categories]
        return "\n".join(lines)


def load(path: Path = DEFAULT_PATH) -> Taxonomy:
    data = tomllib.loads(path.read_text())
    category_of: dict[str, str] = {}
    for category, section in data.items():
        if not isinstance(section, dict):
            continue
        for name in section["classes"]:
            if name in category_of:
                raise ValueError(f"{path}: class {name!r} listed twice")
            category_of[name] = category
    if OTHER not in category_of:
        raise ValueError(f"{path}: must define {OTHER!r}")
    return Taxonomy(version=str(data["version"]), sauce_policy=data.get("sauce_policy", "separate"),
                    category_of=category_of)
