"""CLI for ĀRU's opt-in decision evidence workflow.

Examples:
  python experiment_cli.py collect --prompts prompts.txt --decisions decisions.jsonl
  python experiment_cli.py label --decisions decisions.jsonl --labels labels.jsonl \
    --decision-id ID --reward 0.8 --grader evaluator-1 --rubric-id rubric-v1 \
    --source-kind external
  python experiment_cli.py evaluate --decisions decisions.jsonl --labels labels.jsonl \
    --report-json report.json --report-html report.html
  python experiment_cli.py demo --out-dir demo-report --n 900 --seed 7
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from evidence_dashboard import save_dashboard
from evidence_audit import build_manifest, save_manifest, verify_manifest
from evidence_lab import add_label, evaluate_experiment
from experiment_protocol import (
    POLICIES, load_protocol, make_protocol, protocol_digest, save_protocol,
)
from route_experiments import (
    ExperimentRuntime, action_id, append_jsonl, load_decisions, validate_decision,
)


def lock_protocol(args) -> None:
    protocol = make_protocol(
        args.study_id, args.baseline_policy, args.candidate_policy,
        args.rubric_id, replicates=args.replicates, seed=args.seed,
        max_weight=args.max_weight, min_labeled=args.min_labeled,
        min_ess=args.min_ess, min_uplift=args.min_uplift,
    )
    save_protocol(protocol, args.protocol)
    print(json.dumps({"protocol": args.protocol,
                      "sha256": protocol_digest(protocol),
                      "warning": "Publish or witness this digest before collecting outcomes to establish timing."}, indent=2))


def collect(args) -> None:
    prompts = [line.strip() for line in Path(args.prompts).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not prompts:
        raise ValueError("prompts file contains no tasks")
    runtime = ExperimentRuntime(args.decisions, epsilon=args.epsilon, seed=args.seed,
                                protocol_path=args.protocol)
    for index, prompt in enumerate(prompts, 1):
        state = runtime.run(prompt, session_id=f"experiment_{index}")
        event = state.agent_outputs["planner_node"]["experiment"]
        print(json.dumps({
            "task_number": index,
            "decision_id": event["decision_id"],
            "action": event["action"],
            "propensity": event["propensity"],
            "internal_gate_passed_not_external_reward": state.evaluation.passed,
        }))
    print(f"Logged {len(prompts)} route decisions. Obtain independent labels before evaluation.")


def label(args) -> None:
    if args.protocol:
        protocol = load_protocol(args.protocol)
        if args.rubric_id != protocol["rubric_id"]:
            raise ValueError("label rubric differs from the decision-bound protocol")
        digest = protocol_digest(protocol)
        if any(row.get("protocol_sha256") != digest for row in load_decisions(args.decisions)):
            raise ValueError("decisions do not match the supplied protocol")
    print(json.dumps(add_label(args.decisions, args.labels, args.decision_id,
                               args.reward, args.grader, args.rubric_id,
                               args.source_kind), indent=2))


def evaluate(args) -> dict:
    protocol_path = getattr(args, "protocol", None)
    protocol = load_protocol(protocol_path) if protocol_path else None
    analysis = protocol["analysis"] if protocol else {}
    def fixed(name, default):
        supplied = getattr(args, name, None)
        planned = (protocol.get(name) if name in ("baseline_policy", "candidate_policy")
                   else analysis.get(name)) if protocol else default
        if protocol and supplied is not None and supplied != planned:
            raise ValueError(f"{name} differs from the decision-bound protocol")
        return planned if supplied is None else supplied

    report = evaluate_experiment(
        args.decisions, args.labels,
        candidate_policy=fixed("candidate_policy", "shortest"),
        baseline_policy=fixed("baseline_policy", "greedy"),
        replicates=fixed("replicates", 1000), seed=fixed("seed", 7),
        max_weight=fixed("max_weight", 20.0),
        min_labeled=fixed("min_labeled", 200), min_ess=fixed("min_ess", 30.0),
        min_uplift=fixed("min_uplift", 0.02), protocol_path=protocol_path,
    )
    if args.report_json:
        target = Path(args.report_json)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.report_html:
        save_dashboard(report, args.report_html)
    print(json.dumps({
        "n_logged": report["n_logged"],
        "n_labeled": report["n_labeled"],
        "baseline_snips": report["baseline"]["snips"],
        "candidate_snips": report["candidate"]["snips"],
        "paired_difference": report["paired_difference"],
        "review_gate": report["review_gate"],
        "report_html": args.report_html,
    }, indent=2))
    return report


def demo(args) -> None:
    """Generate a clearly marked SYNTHETIC fixture, never real-world evidence."""
    if args.n < 10:
        raise ValueError("--n must be at least 10")
    target = Path(args.out_dir)
    target.mkdir(parents=True, exist_ok=True)
    decisions_path = target / "decisions.synthetic.jsonl"
    labels_path = target / "labels.synthetic.jsonl"
    if decisions_path.exists() or labels_path.exists():
        raise FileExistsError("demo logs already exist; choose a new --out-dir")
    rng = random.Random(args.seed)
    routes = [
        ["systems_design_node", "simulation_modeling_node", "coherence_node", "compression_node"],
        ["research_node", "truth_evidence_node", "meaning_synthesis_node", "coherence_node"],
        ["systems_design_node", "coherence_node"],
    ]
    epsilon = 0.40
    probabilities = {
        "design": (0.65, 0.55, 0.80),
        "research": (0.60, 0.88, 0.70),
        "mixed": (0.65, 0.72, 0.78),
    }
    for index in range(args.n):
        context = ("design", "research", "mixed")[index % 3]
        k = len(routes)
        chosen = rng.randrange(k) if rng.random() < epsilon else 0
        propensity = epsilon / k + (1 - epsilon if chosen == 0 else 0)
        decision_id = f"synthetic-{args.seed}-{index:07d}"
        decision = {
            "schema_version": 1,
            "decision_id": decision_id,
            "timestamp_unix": 0,
            "task_signature": context + "|synthetic",
            "logging_policy": "epsilon_greedy",
            "epsilon": epsilon,
            "propensity": propensity,
            "action": action_id(routes[chosen]),
            "candidates": [
                {"action": action_id(route), "route": route,
                 "predicted_score": 0.9 - 0.1 * j}
                for j, route in enumerate(routes)
            ],
            "runtime_kind": "synthetic_fixture",
            "loop_count": 0,
        }
        validate_decision(decision)
        append_jsonl(str(decisions_path), decision)
        label = {
            "schema_version": 1,
            "decision_id": decision_id,
            "reward": float(rng.random() < probabilities[context][chosen]),
            "grader": "synthetic-rng",
            "rubric_id": "synthetic-bernoulli-v1",
            "source_kind": "synthetic",
            "graded_at_unix": 0,
        }
        append_jsonl(str(labels_path), label)
    evaluation_args = argparse.Namespace(
        decisions=str(decisions_path), labels=str(labels_path),
        candidate_policy="shortest", baseline_policy="greedy",
        replicates=400, seed=args.seed,
        report_json=str(target / "report.json"),
        report_html=str(target / "index.html"),
    )
    evaluate(evaluation_args)
    print("SYNTHETIC DEMO ONLY — generated outcomes do not validate the actual agent.")


def _hmac_secret(env_name):
    if not env_name:
        return None
    value = os.environ.get(env_name)
    if not value:
        raise ValueError("HMAC key environment variable is missing or empty")
    return value.encode("utf-8")


def attest(args) -> None:
    manifest = build_manifest(
        args.decisions, args.labels, report_path=args.report_json,
        secret=_hmac_secret(args.hmac_key_env),
        protocol_path=args.protocol,
    )
    save_manifest(manifest, args.manifest)
    print(json.dumps({"manifest": args.manifest,
                      "authentication": manifest["authentication"],
                      "counts": manifest["counts"]}, indent=2))


def verify(args) -> None:
    result = verify_manifest(
        args.manifest, args.decisions, args.labels,
        report_path=args.report_json, secret=_hmac_secret(args.hmac_key_env),
        protocol_path=args.protocol,
    )
    print(json.dumps(result, indent=2))
    if not result["verified"]:
        raise SystemExit(1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    protocol_parser = sub.add_parser("lock-protocol", help="Write a non-overwriting analysis plan before data collection")
    protocol_parser.add_argument("--protocol", required=True)
    protocol_parser.add_argument("--study-id", required=True)
    protocol_parser.add_argument("--rubric-id", required=True)
    protocol_parser.add_argument("--baseline-policy", choices=POLICIES, default="greedy")
    protocol_parser.add_argument("--candidate-policy", choices=POLICIES, default="shortest")
    protocol_parser.add_argument("--replicates", type=int, default=1000)
    protocol_parser.add_argument("--seed", type=int, default=7)
    protocol_parser.add_argument("--max-weight", type=float, default=20.0)
    protocol_parser.add_argument("--min-labeled", type=int, default=200)
    protocol_parser.add_argument("--min-ess", type=float, default=30.0)
    protocol_parser.add_argument("--min-uplift", type=float, default=0.02)
    protocol_parser.set_defaults(handler=lock_protocol)

    collect_parser = sub.add_parser("collect", help="Log randomized routes (no reward inferred)")
    collect_parser.add_argument("--prompts", required=True, help="UTF-8 text, one task per line")
    collect_parser.add_argument("--decisions", required=True, help="Append-only JSONL decision log")
    collect_parser.add_argument("--epsilon", type=float, default=0.25)
    collect_parser.add_argument("--seed", type=int, default=7)
    collect_parser.add_argument("--protocol", help="Bind every logged decision to this analysis plan")
    collect_parser.set_defaults(handler=collect)

    label_parser = sub.add_parser("label", help="Append one independently graded outcome")
    label_parser.add_argument("--decisions", required=True)
    label_parser.add_argument("--labels", required=True)
    label_parser.add_argument("--decision-id", required=True)
    label_parser.add_argument("--reward", type=float, required=True)
    label_parser.add_argument("--grader", required=True)
    label_parser.add_argument("--rubric-id", required=True)
    label_parser.add_argument("--source-kind", choices=("external", "synthetic"), required=True)
    label_parser.add_argument("--protocol", help="Check rubric and decision binding before writing")
    label_parser.set_defaults(handler=label)

    eval_parser = sub.add_parser("evaluate", help="Compare policies and produce evidence report")
    eval_parser.add_argument("--decisions", required=True)
    eval_parser.add_argument("--labels", required=True)
    eval_parser.add_argument("--protocol", help="Require every decision/label to match the locked analysis plan")
    eval_parser.add_argument("--baseline-policy", choices=POLICIES)
    eval_parser.add_argument("--candidate-policy", choices=POLICIES)
    eval_parser.add_argument("--replicates", type=int)
    eval_parser.add_argument("--seed", type=int)
    eval_parser.add_argument("--report-json")
    eval_parser.add_argument("--report-html")
    eval_parser.set_defaults(handler=evaluate)

    demo_parser = sub.add_parser("demo", help="Generate and evaluate a clearly labeled synthetic fixture")
    demo_parser.add_argument("--out-dir", required=True)
    demo_parser.add_argument("--n", type=int, default=900)
    demo_parser.add_argument("--seed", type=int, default=7)
    demo_parser.set_defaults(handler=demo)
    attest_parser = sub.add_parser("attest", help="Create non-overwriting evidence checksum/HMAC manifest")
    attest_parser.add_argument("--decisions", required=True)
    attest_parser.add_argument("--labels", required=True)
    attest_parser.add_argument("--report-json")
    attest_parser.add_argument("--protocol", help="Include the locked protocol in the evidence bundle")
    attest_parser.add_argument("--manifest", required=True)
    attest_parser.add_argument("--hmac-key-env", help="Environment variable holding an optional secret")
    attest_parser.set_defaults(handler=attest)

    verify_parser = sub.add_parser("verify", help="Verify evidence bundle against its manifest")
    verify_parser.add_argument("--decisions", required=True)
    verify_parser.add_argument("--labels", required=True)
    verify_parser.add_argument("--report-json")
    verify_parser.add_argument("--protocol")
    verify_parser.add_argument("--manifest", required=True)
    verify_parser.add_argument("--hmac-key-env")
    verify_parser.set_defaults(handler=verify)

    return parser


def main():
    arguments = build_parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
