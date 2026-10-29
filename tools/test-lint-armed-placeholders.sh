#!/usr/bin/env bash
# Armed scoring must reject every golden and reference placeholder.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
python3 - "${ROOT}" <<'PY'
import copy
import importlib.util
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("lint_manifest", root / "tools/lint-benchmark-manifest.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
contract = json.loads((root / "fixtures/qwen3_8_27b_tf_cuda_v1_track.json").read_text())
manifest = json.loads((root / "benchmark.json").read_text())

assert contract["seed_window_max_ratio"] == 1.7
assert contract["decode_speedup_ceiling"] == 5.0
assert contract["prefill_speedup_ceiling"] == 2.0
assert contract["official_prefill_timed_runs"] == 1
assert contract["decode_golden_offset"] == 1
assert contract["decode_window_includes_seed"] is True
assert manifest["scoring"]["decodeSpeedupCeiling"] == contract["decode_speedup_ceiling"]

valid = copy.deepcopy(contract)
valid["official_scoring_enabled"] = True
valid_pins = (
    valid["timed_prompt_pool"]
    + [valid["hidden_correctness_golden"], valid["warmup_golden"]]
    + [o for per_prompt in valid["speculative_oracles"].values() for o in per_prompt.values()]
)
for index, pin in enumerate(valid_pins, 1):
    pin["sha256"] = f"{index:064x}"
    pin["bytes"] = index
valid_linter = module.Linter(str(root), str(root / "benchmark.json"))
valid_linter.check_golden_pins(valid)
valid_linter.check_live_goldens(valid)
assert not valid_linter.failures, valid_linter.failures

armed_contract = copy.deepcopy(valid)
armed_contract["baseline_reference_commit"] = "UNRECORDED"
armed_pins = (
    armed_contract["timed_prompt_pool"]
    + [armed_contract["hidden_correctness_golden"], armed_contract["warmup_golden"]]
    + [o for per_prompt in armed_contract["speculative_oracles"].values() for o in per_prompt.values()]
)
for pin in armed_pins:
    pin["sha256"] = "RERECORD_TBD"
    pin["bytes"] = 0
armed = module.Linter(str(root), str(root / "benchmark.json"))
armed.check_golden_pins(armed_contract)
armed.check_live_goldens(armed_contract)
armed.check_paired_baseline(armed_contract)
assert any("declared serial golden" in message for message in armed.failures), armed.failures
assert any("hidden_correctness_golden" in message for message in armed.failures), armed.failures
assert any("warmup_golden" in message for message in armed.failures), armed.failures
assert any("baseline_reference_commit" in message for message in armed.failures), armed.failures
PY
printf 'test-lint-armed-placeholders: PASS\n'
