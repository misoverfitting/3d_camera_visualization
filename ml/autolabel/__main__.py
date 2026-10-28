"""Auto-labeling CLI. See ml/README.md.

    python -m autolabel [--workspace DIR] <command> ...
"""
from __future__ import annotations

import argparse
from pathlib import Path

from . import taxonomy as taxonomy_module
from .export import cost, export
from .ingest import ingest
from .pipeline import collect, decisions, run_audit, run_label, run_segment, run_verify
from .store import Workspace, read_json

DEFAULT_WORKSPACE = Path(__file__).resolve().parents[1] / "data" / "work"


def make_segmenter(name: str, checkpoint: str):
    if name == "none":
        return None
    from .segment import SamSegmenter
    return SamSegmenter(checkpoint)


def status(ws: Workspace, passes: int) -> None:
    records = ws.records()
    labeled = sum(all((read_json(ws.label_path(r.id, p)) or {}).get("labels") is not None for p in range(passes)) for r in records)
    segmented = sum(ws.candidates_path(r.id).exists() for r in records)
    accepted = candidates = 0
    for r in records:
        decided = decisions(ws, r) or []
        candidates += len(decided)
        accepted += sum(1 for _, final, _ in decided if final)
    print(f"images: {len(records)} in {len({r.group for r in records})} groups")
    print(f"labeled ({passes} passes): {labeled} · segmented: {segmented} · candidates: {candidates} · accepted: {accepted}")
    batches = ws.batches()
    if batches:
        print(f"pending batches: {', '.join(f'{k} ({v['stage']})' for k, v in batches.items())}")
    print(f"cost so far: ${cost(ws)['total']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="autolabel", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--taxonomy", type=Path, default=taxonomy_module.DEFAULT_PATH)
    parser.add_argument("--passes", type=int, choices=[1, 2], default=2, help="independent label passes per image")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="add images, folders, recordings (.zip or folder) and rim tests")
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--group", help="split group (dish ID) for everything in this call")
    p.add_argument("--frames", type=int, default=8, help="frames kept per recording")

    for name, text in (("label", "label passes with Claude"), ("verify", "verify each candidate item with Claude")):
        p = sub.add_parser(name, help=text)
        p.add_argument("--batch", action="store_true", help="submit as Message Batches (half price, async)")
        p.add_argument("--workers", type=int, default=8)
        p.add_argument("--limit", type=int, help="at most this many requests (for trial runs)")

    p = sub.add_parser("collect", help="save results of finished batches")
    p.add_argument("--wait", action="store_true", help="poll until every batch has ended")

    p = sub.add_parser("segment", help="consensus + SAM masks")
    p.add_argument("--segmenter", choices=["sam", "none"], default="sam")
    p.add_argument("--checkpoint", default="facebook/sam-vit-base")

    p = sub.add_parser("audit", help="grade a random sample with a stronger model")
    p.add_argument("--fraction", type=float, default=0.05)
    p.add_argument("--max", type=int, default=50)

    p = sub.add_parser("export", help="write COCO annotations, masks, classifier crops and a report")
    p.add_argument("out", type=Path)
    p.add_argument("--golden", type=Path, help="file listing golden-set group IDs, one per line")

    p = sub.add_parser("run", help="every stage end to end (direct API calls unless --batch)")
    p.add_argument("out", type=Path)
    p.add_argument("--batch", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--segmenter", choices=["sam", "none"], default="sam")
    p.add_argument("--checkpoint", default="facebook/sam-vit-base")
    p.add_argument("--audit-fraction", type=float, default=0.05)
    p.add_argument("--golden", type=Path)

    sub.add_parser("status", help="progress, pending batches and cost so far")

    args = parser.parse_args()
    ws = Workspace(args.workspace)
    taxonomy = taxonomy_module.load(args.taxonomy)
    golden = set(args.golden.read_text().split()) if getattr(args, "golden", None) else set()

    match args.command:
        case "ingest":
            ingest(ws, args.paths, args.group, args.frames)
        case "label":
            print(f"{run_label(ws, taxonomy, args.passes, args.batch, args.workers, args.limit)} label request(s)")
        case "verify":
            print(f"{run_verify(ws, taxonomy, args.batch, args.workers, args.limit)} verify request(s)")
        case "collect":
            pending = collect(ws, taxonomy, args.wait)
            print(f"{pending} batch(es) still processing" if pending else "all batches collected")
        case "segment":
            print(f"segmented {run_segment(ws, taxonomy, args.passes, make_segmenter(args.segmenter, args.checkpoint))} image(s)")
        case "audit":
            print(f"audited {run_audit(ws, args.fraction, args.max)} image(s)")
        case "export":
            summary = export(ws, taxonomy, args.out, golden)
            print((args.out / "report.md").read_text())
            print(f"wrote {args.out} ({summary['stats'].get('items_accepted', 0)} accepted items)")
        case "run":
            run_label(ws, taxonomy, args.passes, args.batch, args.workers, None)
            if args.batch:
                collect(ws, taxonomy, wait=True)
            run_segment(ws, taxonomy, args.passes, make_segmenter(args.segmenter, args.checkpoint))
            run_verify(ws, taxonomy, args.batch, args.workers, None)
            if args.batch:
                collect(ws, taxonomy, wait=True)
            if args.audit_fraction > 0:
                run_audit(ws, args.audit_fraction, 50)
            export(ws, taxonomy, args.out, golden)
            print((args.out / "report.md").read_text())
        case "status":
            status(ws, args.passes)


if __name__ == "__main__":
    main()
