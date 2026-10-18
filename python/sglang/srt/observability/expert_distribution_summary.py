"""Merge complete rank JSON files: python -m ...expert_distribution_summary RUN_DIR."""

import argparse
import csv
import json
import math
from pathlib import Path


def summarize(directory):
    directory = Path(directory)
    records = [json.loads(p.read_text()) for p in sorted(directory.glob("rank-*.json"))]
    if not records:
        raise ValueError("No rank records")
    first = records[0]
    expected = first["topology"]["ep_size"]
    if len(records) != expected or {r["topology"]["ep_rank"] for r in records} != set(
        range(expected)
    ):
        raise ValueError("Missing or duplicate EP ranks")
    layers = len(first["logical_count"])
    experts = len(first["logical_count"][0])
    slots = len(first["physical_count"][0])
    logical = [[0] * experts for _ in range(layers)]
    rank_rows, physical_rows = [], []
    for r in records:
        for key in ("schema_version", "run_id", "start_at", "end_at", "config"):
            if r[key] != first[key]:
                raise ValueError(f"Inconsistent {key}")
        if r["schema_version"] != 1:
            raise ValueError("Unsupported schema")
        if r["config"].get("dump_gemm") and not r.get("gemm_vs_dispatch_match", False):
            raise ValueError("GEMM/dispatch count verification failed")
        topology = {k: v for k, v in r["topology"].items() if k != "ep_rank"}
        reference = {k: v for k, v in first["topology"].items() if k != "ep_rank"}
        if topology != reference:
            raise ValueError("Inconsistent topology/workload")
        if len(r["logical_count"]) != layers or len(r["physical_count"]) != layers:
            raise ValueError("Inconsistent layer count")
        ep_rank = r["topology"]["ep_rank"]
        for layer, (lc, pc) in enumerate(zip(r["logical_count"], r["physical_count"])):
            if len(lc) != experts or len(pc) != slots:
                raise ValueError("Inconsistent expert count")
            if any(type(n) is not int or n < 0 for n in lc + pc):
                raise ValueError("Invalid count")
            if sum(lc) != sum(pc):
                raise ValueError("Logical/physical conservation failed")
            logical[layer] = [a + b for a, b in zip(logical[layer], lc)]
            rank_rows.append((layer, ep_rank, sum(pc)))
            physical_rows.extend((layer, ep_rank, slot, n) for slot, n in enumerate(pc))
    rows = []
    logical_rows = []
    for layer, lc in enumerate(logical):
        total = sum(lc)
        loads = [n for lid, _, n in rank_rows if lid == layer]
        mean = sum(loads) / expected
        rows.append(
            dict(
                layer=layer,
                assignments=total,
                max_over_mean=max(loads) / mean if mean else None,
                cv=math.sqrt(sum((n - mean) ** 2 for n in loads) / expected) / mean
                if mean
                else None,
            )
        )
        logical_rows.extend(
            (layer, expert, n, n / total if total else 0) for expert, n in enumerate(lc)
        )
    for name, header, data in (
        ("logical_experts.csv", ("layer", "expert", "count", "share"), logical_rows),
        ("physical_slots.csv", ("layer", "ep_rank", "slot", "count"), physical_rows),
        ("rank_load.csv", ("layer", "ep_rank", "count"), rank_rows),
    ):
        with (directory / name).open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(data)
    summary = dict(
        run_id=first["run_id"],
        complete=True,
        ranks=expected,
        assignments=sum(map(sum, logical)),
        steps_by_rank={str(r["rank"]): r["steps"] for r in records},
        graph_steps_by_rank={str(r["rank"]): r["graph_steps"] for r in records},
        layers=rows,
    )
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    print(json.dumps(summarize(parser.parse_args().run_dir), indent=2))
