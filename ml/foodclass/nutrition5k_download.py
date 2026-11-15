"""Downloads Nutrition5k's overhead RGB + depth images, metadata and splits (CC BY 4.0)
from the public Google Cloud bucket over plain HTTPS. Resumable: existing files are kept.

    python -m foodclass.nutrition5k_download [out_dir]
"""
from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

BUCKET = "https://storage.googleapis.com/nutrition5k_dataset/"
LIST = "https://storage.googleapis.com/storage/v1/b/nutrition5k_dataset/o"
FILES = ("rgb.png", "depth_raw.png")
META = ["metadata/dish_metadata_cafe1.csv", "metadata/dish_metadata_cafe2.csv", "metadata/ingredients_metadata.csv",
        "dish_ids/splits/rgb_train_ids.txt", "dish_ids/splits/rgb_test_ids.txt",
        "dish_ids/splits/depth_train_ids.txt", "dish_ids/splits/depth_test_ids.txt"]


def overhead_dishes() -> list[str]:
    dishes, token = [], None
    while True:
        query = {"prefix": "nutrition5k_dataset/imagery/realsense_overhead/", "delimiter": "/", "maxResults": "1000"}
        if token:
            query["pageToken"] = token
        page = json.load(urllib.request.urlopen(f"{LIST}?{urllib.parse.urlencode(query)}", timeout=60))
        dishes += [p.rstrip("/").split("/")[-1] for p in page.get("prefixes", [])]
        token = page.get("nextPageToken")
        if not token:
            return dishes


def fetch(path: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        return True
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = urllib.request.urlopen(BUCKET + "nutrition5k_dataset/" + path, timeout=120).read()
    except Exception as exc:  # Some dishes lack a file; keep going.
        print(f"skip {path}: {exc}", flush=True)
        return False
    tmp = out.with_suffix(out.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(out)
    return True


def main(out: Path) -> None:
    for path in META:
        fetch(path, out / path)
    dishes = overhead_dishes()
    print(f"{len(dishes)} overhead dishes", flush=True)
    jobs = [(f"imagery/realsense_overhead/{d}/{f}", out / "overhead" / d / f) for d in dishes for f in FILES]
    done = 0
    with ThreadPoolExecutor(16) as pool:
        for ok in pool.map(lambda job: fetch(*job), jobs):
            done += 1
            if done % 500 == 0:
                print(f"{done}/{len(jobs)} files", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data" / "nutrition5k")
