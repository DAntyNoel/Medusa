#!/usr/bin/env python3
"""Utilities for Medusa Qwen smoke data workflow."""

import argparse
import json
import os
import random
from typing import Any

ALLOWED_ROLES = {"user", "assistant", "system"}
MAX_ERRORS = 20


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)


def _validate_generated_data(data: Any) -> dict[str, int]:
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Generated data must be a non-empty list")

    stats = {
        "generated_samples": len(data),
        "assistant_samples": 0,
        "no_assistant_samples": 0,
        "invalid_samples": 0,
    }
    errors: list[str] = []

    for sample_index, sample in enumerate(data):
        sample_errors: list[str] = []
        has_nonempty_assistant = False

        if not isinstance(sample, list) or len(sample) == 0:
            sample_errors.append(
                f"sample {sample_index}: each sample must be a non-empty list of messages"
            )
        else:
            for message_index, message in enumerate(sample):
                prefix = f"sample {sample_index} message {message_index}"
                if not isinstance(message, dict):
                    sample_errors.append(f"{prefix}: message must be a dict")
                    continue

                if set(message.keys()) != {"role", "content"}:
                    sample_errors.append(
                        f"{prefix}: message must contain only role/content keys"
                    )

                role = message.get("role")
                content = message.get("content")
                if role not in ALLOWED_ROLES:
                    sample_errors.append(f"{prefix}: unsupported role {role!r}")
                if not isinstance(content, str):
                    sample_errors.append(f"{prefix}: content must be a string")
                    continue

                if role == "assistant" and content.strip():
                    has_nonempty_assistant = True

        if has_nonempty_assistant:
            stats["assistant_samples"] += 1
        else:
            stats["no_assistant_samples"] += 1
            sample_errors.append(
                f"sample {sample_index}: sample must contain at least one non-empty assistant turn"
            )

        if sample_errors:
            stats["invalid_samples"] += 1
            remaining = MAX_ERRORS - len(errors)
            if remaining > 0:
                errors.extend(sample_errors[:remaining])

    if errors:
        details = "\n".join(errors)
        raise ValueError(
            "Generated data validation failed.\n"
            f"generated_samples={stats['generated_samples']}\n"
            f"assistant_samples={stats['assistant_samples']}\n"
            f"no_assistant_samples={stats['no_assistant_samples']}\n"
            f"invalid_samples={stats['invalid_samples']}\n"
            f"first_errors:\n{details}"
        )

    return stats


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
    stats = _validate_generated_data(data)
    print(f"generated_samples {stats['generated_samples']}")
    print(f"assistant_samples {stats['assistant_samples']}")
    print(f"no_assistant_samples {stats['no_assistant_samples']}")
    print(f"invalid_samples {stats['invalid_samples']}")
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
