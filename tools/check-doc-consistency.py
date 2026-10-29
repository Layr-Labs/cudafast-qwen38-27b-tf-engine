#!/usr/bin/env python3
"""Check participant docs against the CUDA track manifests."""

from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
FINDINGS: list[str] = []


def need(condition: bool, message: str) -> None:
    if not condition:
        FINDINGS.append(message)


def main() -> int:
    manifest = json.loads((ROOT / "benchmark.json").read_text())
    fixture = json.loads((ROOT / manifest["contractPath"]).read_text())
    declaration = json.loads((ROOT / "mtp-head.manifest.json").read_text())
    docs = {
        name: (ROOT / name).read_text()
        for name in ("TASK.md", "README.md", "AGENTS.md", "docs/participant-contract.md")
    }

    track = manifest["trackId"]
    editable = manifest["editablePaths"]
    scoring = manifest["scoring"]
    depths = fixture["permitted_speculative_depths"]

    for name, text in docs.items():
        need(track in text, f"{name}: missing track id {track}")

    need(editable == ["Vendor/TensorFold"], f"benchmark.json: editablePaths is {editable!r}")
    for name in ("TASK.md", "AGENTS.md", "docs/participant-contract.md"):
        need("Vendor/TensorFold" in docs[name], f"{name}: missing editable path")

    need(depths == [1], f"fixture: permitted_speculative_depths is {depths!r}")
    need(
        sorted(fixture["serve_configuration"]["depths"]) == ["dflash1", "serial"],
        "fixture: serve depths are not serial and dflash1",
    )
    for prompt in fixture["live_goldens"]:
        need(
            sorted(fixture["speculative_oracles"].get(prompt, {})) == ["dflash1"],
            f"fixture: speculative_oracles.{prompt} is not dflash1 only",
        )
    for name in ("TASK.md", "README.md"):
        need("DFlash depth 1" in docs[name], f"{name}: missing DFlash depth 1")
    need(
        "DFlash mode" in docs["docs/participant-contract.md"]
        and "depth 1" in docs["docs/participant-contract.md"],
        "docs/participant-contract.md: missing DFlash depth 1",
    )

    need(fixture["official_pairs"] == 2, "fixture: official_pairs must be 2")
    need(fixture["decode_speedup_floor"] == 0.95, "fixture: decode floor must be 0.95")
    need(fixture["prefill_speedup_floor"] == 0.93, "fixture: prefill floor must be 0.93")
    need(fixture["prefill_gain_exponent"] == 0.25, "fixture: prefill exponent must be 0.25")
    need(fixture["decode_gain_exponent"] == 0.75, "fixture: decode exponent must be 0.75")
    need(fixture["kv_backend"] == scoring["kvBackend"], "fixture: kv backend differs from benchmark.json")
    need(
        scoring["scoredExponents"] == {
            "prefillGainExponent": 0.25,
            "decodeGainExponent": 0.75,
        },
        "benchmark.json: scoring exponents differ from the fixture",
    )

    required_root = {
        "vocab_size": 248320,
        "num_hidden_layers": 64,
        "seed_tokens": 496,
        "local_submit_benchmark_decode_steps": 1023,
        "score_prefill_weight": 0.25,
        "score_decode_weight": 0.75,
    }
    for key, value in required_root.items():
        need(fixture.get(key) == value, f"fixture: {key} is {fixture.get(key)!r}, expected {value!r}")
    for key in (
        "prefill_band_up_tolerance",
        "prefill_band_down_tolerance",
        "decode_band_up_tolerance",
        "decode_band_down_tolerance",
    ):
        need(
            isinstance(fixture.get(key), (int, float)) and fixture[key] > 0,
            f"fixture: {key} must be positive",
        )
    for key in ("prefill_band_down_enabled", "decode_band_down_enabled"):
        need(isinstance(fixture.get(key), bool), f"fixture: {key} must be boolean")

    need(
        fixture["target"]["upstream_model_id"] == "TensorFold/Qwen3.8-27B-MLX-4bit",
        "fixture: wrong target model",
    )
    need(
        fixture["target"]["drafter"]["upstream_model_id"] == "z-lab/Qwen3.8-27B-DFlash2",
        "fixture: wrong drafter model",
    )
    need(
        declaration.get("spec") == {"enabled": True, "tokens": 1},
        "mtp-head.manifest.json: declaration is not DFlash depth 1",
    )

    if FINDINGS:
        print(f"check-doc-consistency: {len(FINDINGS)} finding(s)")
        for finding in FINDINGS:
            print(f"  - {finding}")
        return 1
    print("check-doc-consistency: the docs agree with benchmark.json and the fixture")
    return 0


if __name__ == "__main__":
    sys.exit(main())
