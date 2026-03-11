#!/usr/bin/env python3
"""Utilities for Medusa Qwen smoke data workflow."""

import argparse
import json
import os
import random
from typing import Any


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def cmd_sample(args: argparse.Namespace) -> None:
    src = _read_json(args.input)
    if not isinstance(src, list):
        raise ValueError(f"Input must be a list, got: {type(src).__name__}")

    if len(src) < args.num_samples:
        raise ValueError(
            f"Not enough samples in input: requested {args.num_samples}, found {len(src)}"
        )

    rng = random.Random(args.seed)
    subset = rng.sample(src, args.num_samples)
    _write_json(args.output, subset)
    print(f"subset_size {len(subset)}")
    print(f"output {args.output}")


def cmd_validate(args: argparse.Namespace) -> None:
    data = _read_json(args.input)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Generated data must be a non-empty list")
    first = data[0]
    if not isinstance(first, list) or len(first) == 0:
        raise ValueError("Each sample must be a non-empty list of messages")
    msg = first[0]
    if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
        raise ValueError("Each message must be a dict with keys: role, content")
    print(f"generated_samples {len(data)}")
    print("validation_ok")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Medusa Qwen smoke data utilities")
    subparsers = parser.add_subparsers(dest="command", required=True)

    sample_parser = subparsers.add_parser(
        "sample",
        help="Sample records from a source JSON list",
    )
    sample_parser.add_argument("--input", required=True, help="Path to source JSON list")
    sample_parser.add_argument("--output", required=True, help="Path to output subset JSON")
    sample_parser.add_argument("--num-samples", type=int, default=1000, help="Sample count")
    sample_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sample_parser.set_defaults(func=cmd_sample)

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate generated Medusa training JSON shape",
    )
    validate_parser.add_argument("--input", required=True, help="Generated JSON path")
    validate_parser.set_defaults(func=cmd_validate)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
