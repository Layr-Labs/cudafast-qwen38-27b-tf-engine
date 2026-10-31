#!/usr/bin/env python3
"""Measure DFlash acceptance on independent prompts through Engine Protocol v1.

Runs inside the pinned CUDA image after the worker is built. Each case owns a
fresh resident. An optional authored warm-up precedes the measured prompt with
a phase close between them, to inspect phase-reset behavior.
The optional trace is diagnostic: use the official leg for scored comparisons.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import select
import statistics
import subprocess
import tempfile
import time

from qwen38_landing_replay import parse as parse_landing


PRECISIONS = ("a4", "a4_mse", "non_mlp8", "a4_mse_non_mlp8", "a8", "fc8", "selector8",
        "fc_selector8", "attention8", "mlp8", "convolution8", "selector_bf16",
        "convolution_bf16", "small_bf16", "a8_small_bf16")

CASES = {
    "literary": (
        "Write the opening of a literary short story from the situation below.\n",
        "A museum conservator returns to an island where she spent one summer as a child. The ferry arrives "
        "during a power outage. She has been invited to examine a cabinet of miniature stage sets left by a "
        "retired radio announcer, but the family disagrees about whether they are artworks or records of "
        "events. The announcer's sister remembers him as generous and unreliable; his former producer calls "
        "him meticulous and cruel. Both descriptions should become plausible through small actions rather "
        "than exposition. The conservator carries a letter that she has never opened. At the harbor, a "
        "teenager offers to carry her equipment in exchange for a ride back to the mainland next week. She "
        "recognizes the teenager's surname but cannot place it. A loudspeaker repeats an obsolete weather "
        "forecast even though every other electrical device seems dead. This is a mechanical fault, not a "
        "supernatural event. The miniature rooms are built from materials that react differently to humidity: "
        "salt-stained paper, wax, copper wire, wool, and fragments of photographic negatives. One room has "
        "a staircase that is too narrow for its door. A local carpenter insists this is accurate. Nobody "
        "agrees about which house it represents. The conservator initially mistakes the disagreement for "
        "a practical obstacle, but gradually understands that each person needs a different account to "
        "remain believable. The harbor smells of diesel and cut fennel. A plastic chair has been tied to "
        "a lamppost with a complicated sailor's knot. A gull steals a receipt from a pocket. These details "
        "should matter to the characters' actions without becoming symbols that the narrator explains. "
        "The protagonist is observant about surfaces and evasive about obligations. Give the sister a "
        "precise, dry way of speaking and the teenager a tendency to answer a question literally before "
        "revealing what was really meant. Avoid a dramatic revelation in the opening scene.\n",
        "\nBegin with an unexpected concrete action, use close third person, and let a disagreement emerge through dialogue.\n",
    ),
    "technical": (
        "You are reviewing a research proposal. Read the notes and identify the most serious weaknesses.\n",
        "The proposed instrument estimates turbulent heat flux from asynchronous distributed acoustic sensing "
        "and infrared radiometry. Its observation operator combines a sparse line-integral approximation with "
        "a nonlinear emissivity correction. A hierarchical Bayesian inversion uses a Matérn covariance with "
        "spatially varying length scales, but the hyperparameters were fitted on the same time intervals later "
        "used for validation. Wind direction changes during each acquisition, whereas the reconstruction assumes "
        "a frozen advection field. The authors report a narrower posterior credible interval after removing "
        "measurements whose standardized residual exceeds three. They estimate this threshold from the fitted "
        "residual distribution rather than a separately specified noise model. Calibration panels have known "
        "laboratory emissivity but accumulate condensation outdoors, and the infrared sensor views them at "
        "different incidence angles. The acoustic fibers have imperfect mechanical coupling to the substrate; "
        "their response may depend on both temperature and strain. The paper assumes the two effects can be "
        "separated using a single transfer function measured before deployment. There is no uncertainty term for "
        "the transfer function, sensor clock drift, or wind shear across the measurement volume. Missing samples "
        "are filled by cubic interpolation before spectral estimation, creating smooth intervals where real "
        "fluctuations are unknown. The reported validation compares hourly averages, while the claimed benefit "
        "concerns sub-second dynamics. A competing method uses a different filtering bandwidth and a longer "
        "training interval, so their error bars are not directly comparable. The proposal calls cross-validation "
        "independent even though neighboring folds share overlapping windows. Spatial extrapolation is assessed "
        "only within the convex hull of the calibration sites. Investigators would like to infer a causal "
        "effect of canopy thinning, but treatment locations were selected according to accessibility and "
        "preexisting vegetation density. They do not record soil moisture or cloud cover. To save storage, "
        "raw waveforms are discarded once the derived features have been computed. A final deployment is "
        "planned for a coastal site with salt spray and a different seasonal temperature range.\n",
        "\nExplain three concrete failure modes, their statistical consequences, and experiments that distinguish them.\n",
    ),
    "mixed_language": (
        "Translate and reconcile these multilingual engineering notes into a coherent English incident report.\n",
        "Español: El martes a las 06:17 el colector de telemetría registró una secuencia duplicada después de "
        "restablecerse la conexión. La réplica secundaria confirmó el lote, pero el nodo principal perdió la "
        "respuesta antes de persistir el marcador. Français: Le mécanisme de reprise a réémis les événements "
        "avec de nouveaux identifiants de transport, tandis que leurs identifiants métier étaient inchangés. "
        "L'équipe avait configuré la déduplication sur le mauvais champ et n'avait pas conservé les journaux "
        "du répartiteur. Deutsch: Währenddessen änderte sich die Zeitzone des Exportdienstes. Die angezeigten "
        "Zeitstempel lagen deshalb eine Stunde auseinander, obwohl die monotonen Zähler dieselbe Reihenfolge "
        "belegen. 日本語: 再送されたメッセージは異なる接続から到着したため、到着順序だけでは処理の因果関係を判断できません。"
        "担当者は状態遷移の記録と永続化の境界を比較する必要があります。 Русский: Повторный запуск обработчика "
        "привёл к повторному начислению скидки, но не к повторной отправке товара. Откат выполнялся вручную "
        "и затронул только записи за последние десять минут. Português: A métrica de sucesso contou as "
        "confirmações de transporte, e não as transações concluídas. O alerta de atraso permaneceu verde "
        "porque a fila foi esvaziada antes da gravação no banco de dados.\n",
        "\nSeparate observations from hypotheses, reconcile the apparent contradictions, and propose a recovery protocol.\n",
    ),
    "identifiers": (
        "Review this distributed-systems API and sketch an implementation with the exact identifiers given.\n",
        "The service HydrometeorologicalReconciliationCoordinator owns PartitionLeaseEpochTracker, "
        "ObservationDeduplicationLedger, and MonotonicAcknowledgementWatermark. Each "
        "CrossRegionObservationEnvelope contains upstreamObservationFingerprint, originatingInstrumentSequence, "
        "ingestionLeaseEpoch, provisionalCalibrationRevision, and reconciliationDisposition. The method "
        "reconcileUnacknowledgedObservationBatch must call validateInstrumentSequenceContinuity before "
        "persisting provisionalReconciliationCheckpoint. A lease transition from "
        "AwaitingSecondaryDurabilityConfirmation to RecoveringUncommittedPrimaryAcknowledgements must never "
        "discard entries whose downstreamMaterializationReceipt is absent. If "
        "compareAndAdvanceMonotonicAcknowledgementWatermark rejects staleLeaseEpoch, the caller must reread "
        "partitionOwnershipGeneration and revalidate every batch item. The counter "
        "observationsAwaitingCrossRegionDurability must not decrease until recordSecondaryDurabilityReceipt "
        "has committed atomically with the relevant deduplication key. A retry changes "
        "transportAttemptCorrelationIdentifier but preserves upstreamObservationFingerprint. The method "
        "enumerateRecoveryEligibleObservationFingerprints returns a stable snapshot and a "
        "snapshotReconciliationEpoch. An iterator must reject changes to that epoch instead of silently "
        "skipping fingerprints. Every error carries failedReconciliationStage, "
        "lastDurableObservationSequence, and retryEligibilityDisposition. The integration uses Rust with "
        "async traits, explicit transaction scopes, and structured cancellation. Avoid holding a partition "
        "mutex across a network await. Cancellation after secondary commit but before primary confirmation "
        "must be recoverable without duplicate materialization. The reference state machine has three "
        "independent epochs: ownership, calibration, and reconciliation; never substitute one for another.\n",
        "\nWrite Rust types and pseudocode for the recovery path, preserving the identifiers and explaining one race.\n",
    ),
    "python": (
        "Implement a Python library for processing time series. Here are the requirements and examples.\n",
        "A sample has a timestamp, station name, and floating point reading. Missing readings must be ignored. "
        "Group samples by station, preserve timestamp order, and compute a rolling mean with a configurable window. "
        "Reject negative window sizes and duplicate timestamps. Use dataclasses and type annotations. "
        "Example: station cedar has readings 12.0, 14.0, 16.0; a two-reading window gives 12.0, 13.0, 15.0.\n",
        "\nWrite the implementation and a short usage example.\n```python\n",
    ),
    "sql": (
        "Design SQL queries for an inventory database. Schema and constraints follow.\n",
        "products(product_id, name, category, unit_price); warehouses(warehouse_id, city); "
        "movements(movement_id, product_id, warehouse_id, occurred_at, quantity). Positive quantity is an arrival, "
        "negative quantity is a departure. Stock is the sum of all movements at a warehouse. "
        "Include products with no movements in stock reports. Return ties in product_id order. "
        "A product may be stored in several warehouses. Use PostgreSQL syntax and common table expressions.\n",
        "\nWrite queries for total stock, low stock by warehouse, and the five highest-value categories.\n",
    ),
    "prose": (
        "Continue a story using the following setting and characters.\n",
        "Mara keeps the lighthouse on a small northern island. Her brother Tomas repairs the fishing boats. "
        "The old radio receives a melody every evening, although nobody on the mainland admits transmitting it. "
        "A storm has uncovered a staircase beneath the harbor wall. Mara finds a brass key with a blue ribbon. "
        "The village librarian recognizes the ribbon but asks Mara to return after sunset. "
        "Keep the story grounded in sensory detail and reveal the mystery gradually.\n",
        "\nContinue with Mara visiting the library after sunset.\n",
    ),
    "science": (
        "Prepare an accessible explanation based on these classroom notes.\n",
        "A garden pond receives sunlight throughout the day. Algae use light to convert carbon dioxide and water "
        "into organic material, releasing oxygen. Animals consume oxygen, and decomposers also consume it while "
        "breaking down organic material. Water temperature changes how much dissolved oxygen the pond can hold. "
        "Measurements at dawn and afternoon can differ even when the weather is stable. "
        "Students should distinguish an observation, a hypothesis, and a controlled experiment.\n",
        "\nExplain the daily oxygen cycle and propose a simple classroom investigation.\n",
    ),
    "chat": (
        "Read this planning conversation and draft the assistant's next reply.\n",
        "User: We have six volunteers for a community repair workshop on Saturday.\n"
        "Assistant: What stations and equipment do you have?\n"
        "User: Two sewing machines, a bicycle stand, and a table for small wooden items. "
        "We open at ten and close at four. Everyone needs a lunch break.\n"
        "Assistant: We can stagger breaks and keep one person welcoming visitors.\n"
        "User: New volunteers need a short orientation and we want a clear sign-up system.\n",
        "\nUser: Please give me a practical schedule and a short checklist for setup.\nAssistant:",
    ),
    "json": (
        "Produce a JSON example that follows this catalog specification.\n",
        "The root object has version, store, and products. A product has sku, title, price_cents, tags, "
        "dimensions_cm, and inventory. Dimensions has width, height, and depth. Inventory is an array of "
        "objects containing location and quantity. Prices and quantities are nonnegative integers. "
        "Use distinct SKUs, descriptive titles, and two locations named north and south. "
        "Products are a notebook, a desk lamp, a ceramic mug, a pencil case, and a shelf.\n",
        "\nReturn only valid JSON for the complete five-product catalog.\n",
    ),
}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def prompts(target, tree):
    from tokenizers import Tokenizer
    path = target / "tokenizer.json"
    fixture = json.loads((tree / "fixtures/qwen3_8_27b_tf_cuda_v1_track.json").read_text())
    pin = next(x for x in fixture["target"]["files"] if x["path"] == "tokenizer.json")
    if digest(path.read_bytes()) != pin["sha256"]:
        raise ValueError("target tokenizer does not match the pinned digest")
    tok = Tokenizer.from_file(str(path))
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids
    count = fixture["seed_tokens"]
    public = tree / "correctness_prompts/public_longcopy_gate_english_4096.txt"
    result = {"public_text_prefix": encode(public.read_text())[:count]}
    for name, (prefix, body, suffix) in CASES.items():
        tail = encode(suffix)
        main = encode(prefix + body * 12)
        result[name] = main[:count - len(tail)] + tail
    # Explicit message delimiters form independent authored prompts. These are
    # not asserted to reproduce a checkpoint's configurable chat template.
    if all(tok.token_to_id(marker) is not None for marker in ("<|im_start|>", "<|im_end|>")):
        for name in ("literary", "technical", "mixed_language", "chat"):
            prefix, body, suffix = CASES[name]
            tail = encode(suffix + "<|im_end|>\n<|im_start|>assistant\n")
            main = encode("<|im_start|>user\n" + prefix + body * 12)
            result["message_" + name] = main[:count - len(tail)] + tail
    # Independent Gutenberg mixtures supplied for distributional screening.
    for source in sorted((Path(__file__).parent / "proxy_prompts").glob("gut-*.txt")):
        result[source.stem] = encode(source.read_text())[:count]
    for name, ids in result.items():
        if len(ids) != count:
            raise ValueError(f"{name}: expected {count} tokens, got {len(ids)}")
    return result, tok


def stop(process):
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def response(process, timeout=300):
    if not select.select([process.stdout], [], [], timeout)[0]:
        raise TimeoutError("worker response timed out")
    line = process.stdout.readline()
    if not line:
        raise RuntimeError(f"worker closed stdout (exit {process.poll()})")
    reply = json.loads(line)
    if not reply.get("ok"):
        raise RuntimeError(f"worker error: {reply}")
    return reply


def request(process, request_id, kind, **fields):
    wire = dict(id=request_id, kind=kind, **fields)
    process.stdin.write(json.dumps(wire).encode() + b"\n")
    process.stdin.flush()
    result = response(process)
    if result["id"] != request_id:
        raise RuntimeError("response ID does not match request")
    return result


def legacy_round_receipts(text):
    rounds, current = [], {"drafter_ms": 0.0, "tree_ms": 0.0, "verify_ms": 0.0, "rows": 0, "bucket": 0}
    for line in text.splitlines():
        if m := re.search(r"proposal_plan kind=(\w+) nodes=(\d+) context_copy=(\d+)", line):
            current.update(plan_kind=m[1], proposed_nodes=int(m[2]), context_copy=bool(int(m[3])))
        if m := re.search(r"\[timing\] (absorb|hold) ([0-9.]+) ms", line):
            current["drafter_ms"] += float(m[2])
        if m := re.search(r"\[timing\] tree_plan ([0-9.]+) ms", line):
            current["tree_ms"] += float(m[1])
        if m := re.search(r"verify_round offered=(\d+) kept=(\d+) bucket=(\d+) ms=([0-9.]+)", line):
            current["rows"] += int(m[1])
            current["bucket"] = max(current["bucket"], int(m[3]))
            current["verify_ms"] += float(m[4])
        if m := re.search(r"bench_round committed=(\d+) drafted=(\d+) accepted=(\d+) ms=([0-9.]+)", line):
            current.update(committed=int(m[1]), drafted=int(m[2]), accepted=int(m[3]), round_ms=float(m[4]))
            rounds.append(current)
            current = {"drafter_ms": 0.0, "tree_ms": 0.0, "verify_ms": 0.0, "rows": 0, "bucket": 0}
    return rounds


def round_receipts(text, lengths=None):
    if "bench_round committed=" in text:
        rounds = legacy_round_receipts(text)
    else:
        rounds = []
        current = {"drafter_ms": 0.0, "tree_ms": 0.0, "verify_ms": 0.0, "rows": 0, "bucket": 0}
        for line in text.splitlines():
            if m := re.search(r"\[timing\] (absorb|hold) ([0-9.]+) ms", line):
                current["drafter_ms"] += float(m[2])
            if m := re.search(r"\[timing\] tree_plan ([0-9.]+) ms", line):
                current["tree_ms"] += float(m[1])
            if m := re.search(r"verify_round offered=(\d+) kept=(\d+) bucket=(\d+) ms=([0-9.]+)", line):
                current.update(rows=int(m[1]), bucket=int(m[3]), verify_ms=float(m[4]),
                               committed=int(m[2]), raw_committed=int(m[2]), drafted=int(m[1]) - 1,
                               accepted=max(0, int(m[2]) - 1))
                current["round_ms"] = current["drafter_ms"] + current["tree_ms"] + current["verify_ms"]
                rounds.append(current)
                current = {"drafter_ms": 0.0, "tree_ms": 0.0, "verify_ms": 0.0, "rows": 0, "bucket": 0}
    if lengths is not None:
        if len(rounds) != len(lengths):
            raise RuntimeError("callback count does not match protocol acceptance audit")
        remaining = sum(lengths)
        for i, (row, expected) in enumerate(zip(rounds, lengths)):
            raw = row["committed"]
            if expected <= 0 or raw < expected or min(raw, remaining) != expected:
                raise RuntimeError("callback acceptance differs from returned tokens")
            if i != len(lengths) - 1 and raw != expected:
                raise RuntimeError("only the final callback may be clipped")
            row["raw_committed"] = raw
            row["committed"] = expected
            remaining -= expected
    return rounds


def run_case(args, name, ids, paths):
    worker_path, target, drafter, out, tokenizer = paths
    label = f"{name}-{args.precision}-nodes{args.nodes}-block{args.block}-probe{args.min_probe_rows}-b{args.branches}-e{args.edge_weight}-s{args.tree_scale}-rep{args.rep}"
    if args.candidate_topk or args.calibrated or args.landing_prior != 2:
        label += f"-k{args.candidate_topk}-cal{args.calibrated}-prior{args.landing_prior}"
    if args.wide_backend:
        label += f"-wide{args.wide_backend}"
    if args.observe_all:
        label += "-observe-all"
    if args.fixed_tree:
        label += "-fixed-tree"
    case_dir = out / (label + ("-replay" if args.replay else ""))
    case_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / "prompt.json").write_text(json.dumps(ids))
    resident = worker = None
    with tempfile.TemporaryDirectory(prefix="tf-accept-") as tmp, \
            (case_dir / "resident.log").open("wb", buffering=0) as log, \
            (case_dir / "worker.log").open("wb") as worker_log:
        socket = Path(tmp) / "r.sock"
        env = dict(os.environ, TF_TIMING="1", BENCH_WORKER_RESIDENT_SOCKET=str(socket))
        env["TF_WIDE_ECONOMICS"] = str(args.wide_backend)
        env["TF_DFLASH_PRECISION"] = args.precision
        env["TF_DRAFT_NODES"] = str(args.nodes)
        env["TF_DFLASH_BLOCK"] = str(args.block)
        env["TF_DFLASH_COST_PROBE_MIN_ROWS"] = str(args.min_probe_rows)
        if args.landing_trace:
            env["TF_DFLASH_LANDING_TRACE"] = "1"
        if args.capture_every_token:
            env["TF_DFLASH_CAPTURE_EVERY_TOKEN"] = "1"
        if args.replay:
            env["TF_DFLASH_REPLAY"] = "1"
        if args.candidate_topk:
            env["TF_DFLASH_TOPK"] = str(args.candidate_topk)
        env["TF_DFLASH_FIXED_TREE"] = str(args.fixed_tree)
        env["TF_DFLASH_OBSERVE_ALL"] = str(args.observe_all)
        env["TF_DFLASH_CALIBRATED"] = str(args.calibrated)
        env["TF_DFLASH_LANDING_PRIOR"] = str(args.landing_prior)
        env["TF_DFLASH_BRANCHES"] = str(args.branches)
        env["TF_DFLASH_EDGE"] = str(args.edge_weight)
        env["TF_DFLASH_SCALE"] = str(args.tree_scale)
        try:
            resident = subprocess.Popen([str(worker_path), "resident", "--weights", str(target),
                "--drafter", str(drafter), "--socket", str(socket)], env=env,
                stdout=log, stderr=log)
            deadline = time.monotonic() + 300
            while not socket.exists():
                if resident.poll() is not None:
                    raise RuntimeError(f"resident exited {resident.returncode}; see {case_dir / 'resident.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("resident startup timed out")
                time.sleep(0.1)
            worker = subprocess.Popen([str(worker_path), "runtime-worker", "--weights", str(target)],
                env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log)
            hello = response(worker)
            warmup = None
            if args.warmup_case is not None:
                warm_offset = (case_dir / "resident.log").stat().st_size
                warm_seed = request(worker, 101, "free_decode_begin", seed_tokens=args.warmup_ids,
                    spec={"mode": "dflash", "dflash": {"depth": 1}})
                warm_decode = request(worker, 102, "free_decode_run", count=args.tokens)
                warm_trace = (case_dir / "resident.log").read_bytes()[warm_offset:].decode(errors="replace")
                (case_dir / "warmup.log").write_text(warm_trace)
                warm_close = request(worker, 103, "phase_diagnostics")
                warmup = {"name": args.warmup_case, "prompt_tokens": args.warmup_ids,
                          "seed": warm_seed, "decoded": warm_decode, "closed": warm_close,
                          "rounds": round_receipts(warm_trace, warm_decode["acceptance_lengths"])}
                (case_dir / "warmup.json").write_text(json.dumps(warmup, indent=2) + "\n")
            # Untimed prompt pass exercises newly allocated buffers; the next request resets the session.
            request(worker, 1, "prefill", prompt_tokens=ids)
            request(worker, 2, "phase_diagnostics")
            proposal_offset = (case_dir / "resident.log").stat().st_size
            started = time.monotonic()
            seed = request(worker, 3, "free_decode_begin", seed_tokens=ids,
                spec={"mode": "dflash", "dflash": {"depth": 1}})
            seed_ms = (time.monotonic() - started) * 1000
            offset = (case_dir / "resident.log").stat().st_size
            started = time.monotonic()
            decoded = request(worker, 4, "free_decode_run", count=args.tokens)
            decode_ms = (time.monotonic() - started) * 1000
            (case_dir / "protocol.json").write_text(json.dumps(dict(hello=hello, seed=seed, decoded=decoded,
                seed_ms=seed_ms, decode_ms=decode_ms), indent=2) + "\n")
            trace = (case_dir / "resident.log").read_bytes()[offset:].decode(errors="replace")
            (case_dir / "rounds.log").write_text(trace)
            proposal = (case_dir / "resident.log").read_bytes()[proposal_offset:].decode(errors="replace")
            (case_dir / "proposal.log").write_text(proposal)
            request(worker, 5, "phase_diagnostics")
            lengths = decoded["acceptance_lengths"]
            rounds = round_receipts(trace, lengths)
            if sum(lengths) != args.tokens or [r["committed"] for r in rounds] != lengths:
                raise RuntimeError("round trace does not match the protocol acceptance audit")
            if args.fixed_tree and any(r["rows"] != args.nodes + 1 for r in rounds[1:-1]):
                raise RuntimeError("fixed-width economics round did not verify the requested full tree")
            if args.capture_every_token and (len(lengths) != args.tokens or any(n != 1 for n in lengths)):
                raise RuntimeError("all-position capture must commit exactly one target token per round")
            if args.capture_every_token:
                headers = [dict(part.split("=") for part in line.split()[1:])
                           for line in proposal.splitlines() if line.startswith("[tree_replay_begin] ")]
                for p in range(args.tokens):
                    match = [h for h in headers if int(h["position"]) == len(ids) + 1 + p]
                    pending = seed["seed_token"] if p == 0 else decoded["tokens"][p - 1]
                    if len(match) != 1 or int(match[0]["pending"]) != pending or int(match[0]["topk"]) < 32:
                        raise RuntimeError(f"incomplete every-position capture at {p}")
                cost_rows = {}
                for line in (case_dir / "resident.log").read_text().splitlines():
                    if line.startswith("[tree_replay_cost] "):
                        fields = dict(part.split("=") for part in line.split()[1:])
                        cost_rows[int(fields["rows"])] = float(fields["ms"])
                required_rows = 128 if args.nodes >= 64 else 64
                if not all(row in cost_rows and cost_rows[row] > 0 for row in range(1, required_rows + 1)):
                    raise RuntimeError("capture missing measured verify cost rows")
                (case_dir / "verify-costs.json").write_text(json.dumps(cost_rows, indent=2) + "\n")
            if args.landing_trace:
                landing = parse_landing(proposal)
                if len(landing) < len(rounds) - 1:
                    raise RuntimeError("missing ordinary Landing observations")
                (case_dir / "landing.json").write_text(json.dumps(landing, indent=2) + "\n")
            result = dict(name=name, hello=hello, prompt_tokens=ids, seed=seed, decoded=decoded,
                warmup_case=args.warmup_case, capture_every_token=args.capture_every_token,
                precision=args.precision, nodes=args.nodes, block=args.block, replay=args.replay, rep=args.rep, cost_probe_min_rows=args.min_probe_rows,
                branches=args.branches, edge_weight=args.edge_weight, tree_scale=args.tree_scale,
                wide_backend=args.wide_backend, candidate_topk=args.candidate_topk, calibrated=args.calibrated, landing_prior=args.landing_prior, observe_all=args.observe_all, fixed_tree=args.fixed_tree,
                seed_ms=seed_ms, decode_ms=decode_ms, rounds=rounds,
                tokens_per_round=args.tokens / len(lengths), round_ms=decode_ms / len(lengths),
                drafter_ms=statistics.mean(r["drafter_ms"] for r in rounds),
                tree_ms=statistics.mean(r["tree_ms"] for r in rounds),
                rows_per_round=statistics.mean(r["rows"] for r in rounds),
                plan_kinds={kind: sum(r.get("plan_kind", "unknown") == kind for r in rounds)
                            for kind in sorted({r.get("plan_kind", "unknown") for r in rounds})},
                token_sha256=digest(json.dumps([seed["seed_token"], *decoded["tokens"]]).encode()))
            (case_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            (case_dir / "generated.txt").write_text(tokenizer.decode([seed["seed_token"], *decoded["tokens"]]))
            print(json.dumps({k: result[k] for k in ("name", "precision", "nodes", "block", "replay", "rep", "cost_probe_min_rows", "tokens_per_round", "round_ms", "drafter_ms", "tree_ms", "rows_per_round", "seed_ms", "decode_ms", "token_sha256")}), flush=True)
            return result
        finally:
            stop(worker)
            stop(resident)


def parse_variant(value):
    try:
        fields = value.split(":")
        mode, nodes, block, minimum, branches, edge, scale = fields[:7]
        topk, calibrated, prior = (fields[7:10] if len(fields) >= 10 else ("0", "0", "2"))
        observe_all = int(fields[10]) if len(fields) >= 11 else 0
        fixed_tree = int(fields[11]) if len(fields) >= 12 else 0
        wide_backend = int(fields[12]) if len(fields) == 13 else 0
        if len(fields) not in (7, 10, 11, 12, 13):
            raise ValueError("expected seven, ten, eleven, twelve or thirteen fields")
        topk, calibrated, prior = int(topk), int(calibrated), float(prior)
        nodes, block, minimum, branches = map(int, (nodes, block, minimum, branches))
        edge, scale = float(edge), float(scale)
        if (mode not in PRECISIONS or nodes not in (11, 13, 15, 23, 31, 47, 63, 127) or block not in (8, 16)
                or minimum not in (1, 16, 32, 65, 129) or not 1 <= branches <= 32 or topk not in (0, 16, 20, 32) or calibrated not in (0, 1) or observe_all not in (0, 1) or fixed_tree not in (0, 1) or wide_backend not in (0, 1, 2, 3)
                or not math.isfinite(prior) or prior <= 0
                or not math.isfinite(edge) or not 0 <= edge <= 4 or not math.isfinite(scale) or scale <= 0):
            raise ValueError("unsupported value")
        return mode, nodes, block, minimum, branches, edge, scale, topk, calibrated, prior, observe_all, fixed_tree, wide_backend
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected precision:nodes:block:probe_min_rows:branches:edge:scale") from exc


def configure_variant(args, variant):
    if len(variant) == 7:
        variant = (*variant, 0, 0, 2)
    if len(variant) == 10:
        variant = (*variant, 0)
    if len(variant) == 11:
        variant = (*variant, 0)
    if len(variant) == 12:
        variant = (*variant, 0)
    (args.precision, args.nodes, args.block, args.min_probe_rows,
     args.branches, args.edge_weight, args.tree_scale, args.candidate_topk, args.calibrated, args.landing_prior, args.observe_all, args.fixed_tree, args.wide_backend) = variant


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variants", nargs="+", type=parse_variant, help="explicit precision:nodes:block:probe_min_rows:branches:edge:scale[:topk:calibrated:prior[:observe_all[:fixed_tree[:wide_backend]]]] configurations, overrides the grid")
    parser.add_argument("--landing-trace", action="store_true", help="record exact Landing inputs and pre-update bins")
    parser.add_argument("--landing-baselines", action="store_true", help="also run ordinary rounds before each all-position capture")
    parser.add_argument("--capture-every-token", action="store_true", help="diagnostic 32-best receipt at every target token; disables copy proposals")
    parser.add_argument("--replay", action="store_true", help="capture scoring receipts; timings are perturbed")
    parser.add_argument("--capture-cases", nargs="*", default=[], help="additional every-position K32 captures after timed variants")
    parser.add_argument("--replay-cases", nargs="*", default=[], help="additional independent first-variant scoring captures after the timed screen")
    parser.add_argument("--reps", type=int, default=1, help="independent fresh-resident repetitions")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--precisions", nargs="+", default=["a4"], choices=PRECISIONS)
    parser.add_argument("--node-budgets", nargs="+", type=int, choices=(11, 13, 15, 23, 31, 47, 63, 127), default=[15])
    parser.add_argument("--probe-min-rows", type=int, nargs="+", choices=(1, 16, 32, 65, 129), default=[1], help="minimum forced cost-trial width; 129 disables trials at this geometry")
    parser.add_argument("--blocks", nargs="+", type=int, choices=(8, 16), default=[16])
    parser.add_argument("--branches", type=int, default=4, choices=range(1, 33))
    parser.add_argument("--edge-weight", type=float, default=0.6)
    parser.add_argument("--tree-scale", type=float, default=1.5)
    parser.add_argument("--warmup-case", help="authored case to decode and phase-close before each measured case")
    parser.add_argument("--cases", nargs="*", help="case names, default all cases")
    args = parser.parse_args()
    if args.landing_baselines:
        args.landing_trace = True
        if not args.capture_every_token:
            parser.error("landing-baselines requires capture-every-token")
    if args.capture_every_token:
        args.replay = True
    if args.reps < 1 or args.tokens < 1:
        parser.error("reps and tokens must be positive")
    tree = Path(os.environ["Q38_TREE"])
    target = Path(os.environ["Q38_TARGET_DIR"])
    out = Path(os.environ["Q38_OUT"]) / "acceptance"
    out.mkdir(parents=True, exist_ok=False)
    cases, tokenizer = prompts(target, tree)
    if args.warmup_case is not None:
        if args.warmup_case not in cases or args.warmup_case == "public_text_prefix":
            parser.error("warmup-case must name an independent case")
        args.warmup_ids = cases[args.warmup_case]
    paths = Path(os.environ["Q38_WORKER"]), target, Path(os.environ["Q38_DRAFTER_DIR"]), out, tokenizer
    results = []
    variants = args.variants or [parse_variant(f"{mode}:{nodes}:{block}:{minimum}:{args.branches}:{args.edge_weight}:{args.tree_scale}")
                for mode in args.precisions for nodes in args.node_budgets
                for block in args.blocks for minimum in args.probe_min_rows]
    expected = {}
    for rep in range(args.reps):
        args.rep = rep
        for i, name in enumerate(args.cases or cases):
            # Alternate order across both prompts and independent repetitions.
            for variant in (variants if (i + rep) % 2 == 0 else variants[::-1]):
                configure_variant(args, variant)
                if args.landing_baselines:
                    ordinary = copy.copy(args)
                    ordinary.capture_every_token = False
                    ordinary.replay = False
                    baseline = run_case(ordinary, name, cases[name], paths)
                    expected.setdefault(name, [baseline["seed"]["seed_token"], *baseline["decoded"]["tokens"]])
                results.append(run_case(args, name, cases[name], paths))
                tokens = [results[-1]["seed"]["seed_token"], *results[-1]["decoded"]["tokens"]]
                if tokens != expected.setdefault(name, tokens):
                    raise RuntimeError(f"{name}: output differs between variants/repetitions; receipts retained")
    (out / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    if args.capture_cases:
        args.replay_cases.extend(args.capture_cases)
        args.capture_every_token = True
    if args.replay_cases:
        args.rep = 0
        args.replay = True
        configure_variant(args, variants[0])
        replay_results = []
        for name in args.replay_cases:
            captured = run_case(args, name, cases[name], paths)
            prior = next((r for r in results if r["name"] == name), None)
            if prior is not None and captured["decoded"]["tokens"] != prior["decoded"]["tokens"]:
                raise RuntimeError(f"{name}: scoring capture changed output")
            replay_results.append(captured)
        (out / "replay-summary.json").write_text(json.dumps(replay_results, indent=2) + "\n")


if __name__ == "__main__":
    main()
