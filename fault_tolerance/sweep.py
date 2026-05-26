"""Sweep file loading, item expansion, server-config fingerprint."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


# CLI-arg translation rules:
#   key foo_bar -> --foo-bar
#   bool True   -> "--foo-bar" (flag, no value); False/None -> omitted
#   list        -> "--foo-bar a,b,c"
#   other       -> "--foo-bar str(value)"
def _kv_to_cli(key: str, value: Any) -> list[str]:
    flag = "--" + key.replace("_", "-")
    if value is None or value is False:
        return []
    if value is True:
        return [flag]
    if isinstance(value, list):
        return [flag, ",".join(str(v) for v in value)]
    return [flag, str(value)]


def dict_to_cli(d: dict[str, Any]) -> list[str]:
    args: list[str] = []
    for k, v in d.items():
        args.extend(_kv_to_cli(k, v))
    return args


@dataclass(frozen=True)
class Item:
    id: str
    server: dict[str, Any]
    bench: dict[str, Any]

    def server_fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.server, sort_keys=True).encode()
        ).hexdigest()[:12]

    def server_cli(
        self, model: str, port: int, override_cmd: list[str] | None = None
    ) -> list[str]:
        base = list(override_cmd) if override_cmd is not None else ["python3", "-m", "miniengine"]
        return base + ["--model", model, "--port", str(port)] + dict_to_cli(self.server)

    def bench_cli(
        self, model: str, port: int, override_cmd: list[str] | None = None
    ) -> list[str]:
        bench = dict(self.bench)
        script = bench.pop("script")
        if override_cmd is not None:
            base = list(override_cmd)
        else:
            base = ["python3", "-m", f"benchmark.{script}"]
        return base + [
            "--model", model,
            "--base-url", f"http://localhost:{port}",
        ] + dict_to_cli(bench)


@dataclass
class Sweep:
    sweep_id: str
    model: str
    port: int = 8000
    server_warmup_timeout_s: int = 600
    bench_timeout_s: int = 7200
    max_attempts_per_item: int = 2
    items: list[Item] = field(default_factory=list)


def _merge(defaults: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    out.update(override)
    return out


def _expand_concurrencies(item: Item) -> list[Item]:
    """Split a bench_serving item with multiple concurrency levels into one item per level.

    Splitting after defaults are merged so each sibling is fully self-contained.
    Returns the original item unchanged if not applicable.
    """
    if item.bench.get("script") != "bench_serving":
        return [item]
    levels = item.bench.get("concurrencies")
    if not isinstance(levels, list) or len(levels) <= 1:
        return [item]

    out: list[Item] = []
    for level in levels:
        bench = dict(item.bench)
        bench["concurrencies"] = [level]
        out.append(
            Item(
                id=f"{item.id}.c{level}",
                server=item.server,
                bench=bench,
            )
        )
    return out


def load_sweep(path: str | Path) -> Sweep:
    raw = yaml.safe_load(Path(path).read_text())

    sweep_id = raw["sweep_id"]
    model = raw["model"]
    port = int(raw.get("port", 8000))
    defaults = raw.get("defaults", {}) or {}
    default_server = defaults.get("server", {}) or {}
    default_bench = defaults.get("bench", {}) or {}

    raw_items = raw.get("items", [])
    items: list[Item] = []
    seen_ids: set[str] = set()
    for raw_item in raw_items:
        item_id = raw_item["id"]
        item = Item(
            id=item_id,
            server=_merge(default_server, raw_item.get("server", {}) or {}),
            bench=_merge(default_bench, raw_item.get("bench", {}) or {}),
        )
        for expanded in _expand_concurrencies(item):
            if expanded.id in seen_ids:
                raise ValueError(f"duplicate item id: {expanded.id}")
            seen_ids.add(expanded.id)
            items.append(expanded)

    return Sweep(
        sweep_id=sweep_id,
        model=model,
        port=port,
        server_warmup_timeout_s=int(raw.get("server_warmup_timeout_s", 600)),
        bench_timeout_s=int(raw.get("bench_timeout_s", 7200)),
        max_attempts_per_item=int(raw.get("max_attempts_per_item", 2)),
        items=items,
    )
