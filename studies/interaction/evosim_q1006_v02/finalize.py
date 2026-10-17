"""전체 감사·후보 잠금 뒤 원래 환경에서 시험하며 중단 작업도 계수한다."""

from datetime import datetime, timezone
from pathlib import Path
import uuid

from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.evaluation import evaluate_policy, observation_probe, selection_key
from .network import Genome
from evosim_interaction_v02.policies import FixedSequencePolicy, CoordinatePolicy, UniformRandomPolicy, blind_patterns
from evosim_interaction_v02.settings import digest_json
from evosim_interaction_v02.storage import read_json, write_json, write_once, file_digest
from .model import NETWORK
from .probe import probe
from .source import ROOT
from .protocol import load, evaluate_split, RESULTS
from .search import directory, candidate_manifest, lock_candidates, verify_candidate_lock, attempt_accounting
from .audit import audit_job, check_metrics, verify_trajectories, audit_test_job
from .analysis import summarize


def cached_evaluation(root, artifact, identity, maximum_calls, function, *, single_step_actions=0):
    """결과는 불변 캐시, 실제 작업은 UUID 원장. 미확정 중단은 상한으로 남긴다."""
    root, artifact = Path(root), Path(artifact)
    if artifact.exists():
        saved = read_json(artifact)
        if saved["identity"] != identity or saved["payload_digest"] != digest_json(saved["payload"]):
            raise ValueError("Cached evaluation identity/checksum differs")
        return saved["payload"]
    attempt_path = root / "attempts" / (uuid.uuid4().hex + ".json")
    attempt = {"status": "started", "started_utc": datetime.now(timezone.utc).isoformat(),
               "maximum_episode_calls": maximum_calls, "identity": identity,
               "maximum_single_step_actions": single_step_actions}
    write_json(attempt_path, attempt)
    payload, actual_calls = function()
    if not 0 <= actual_calls <= maximum_calls:
        raise ValueError("Evaluation exceeded registered call bound")
    attempt.update(status="evaluated", completed_episode_calls=actual_calls,
                   completed_single_step_actions=single_step_actions)
    write_json(attempt_path, attempt)
    write_once(artifact, {"identity": identity, "payload": payload, "payload_digest": digest_json(payload)})
    attempt.update(status="committed")
    write_json(attempt_path, attempt)
    return payload


def validate_blind(root, selected):
    root = Path(root)
    config = load(root)
    candidates = read_json(root / "blind_validation.json.gz")["candidates"]
    patterns = blind_patterns()
    if len(candidates) != len(patterns):
        raise ValueError("Blind candidate pool incomplete")
    best = None
    lookup = {ep.episode_id: ep for ep in make_split().validation}
    for index, (pattern, candidate) in enumerate(zip(patterns, candidates)):
        if candidate["index"] != index or candidate["pattern"] != list(pattern):
            raise ValueError("Blind pattern differs")
        if [(r["episode_id"], r["repeat"]) for r in candidate["records"]] != [(ep.episode_id, 0) for ep in make_split().validation]:
            raise ValueError("Blind selection used wrong split")
        check_metrics(candidate["metrics"], candidate["records"])
        if any(r["actions"] != "".join(str(pattern[i % len(pattern)]) for i in range(len(r["actions"]))) for r in candidate["records"]):
            raise ValueError("Blind actions differ from fixed pattern")
        if best is None or selection_key(candidate["metrics"]) > selection_key(best["metrics"]):
            best = candidate
    expected = {"index": best["index"], "pattern": best["pattern"], "validation_metrics": best["metrics"],
                "validation_artifact_sha256": file_digest(root / "blind_validation.json.gz")}
    if expected != selected:
        raise ValueError("Blind winner is not validation-only first tie")
    verify_trajectories([r for candidate in candidates for r in candidate["records"]], lookup)
    return expected


def lock_all(root):
    root = Path(root)
    config = load(root)
    if config["phase"] not in ("main", "confirmation"):
        raise ValueError("Only official production can lock held-out candidates")
    candidate_manifest(root, config)
    audits = []
    for job in config["jobs"]:
        audits.append(audit_job(root, **job))
        print(f"AUDIT {job['condition']}/{job['method']}/{job['seed']} passed", flush=True)
    write_once(root / "search_audit.json", {"passed": True, "jobs": audits})
    path = root / "blind_validation.json.gz"
    if not path.exists():
        if config["phase"] == "confirmation":
            if (file_digest(RESULTS / "main/summary.json") != config["main_summary_sha256"]
                    or file_digest(RESULTS / "main/blind_validation.json.gz") != config["shared_blind_sha256"]):
                raise ValueError("Shared main evidence changed")
            write_once(path, read_json(RESULTS / "main/blind_validation.json.gz"))
        else:
            candidates = []
            for index, pattern in enumerate(blind_patterns()):
                identity = {"protocol_digest": config["protocol_digest"], "pattern": list(pattern), "purpose": "blind_validation"}
                def calculate(pattern=pattern, index=index):
                    metrics, records = evaluate_policy(FixedSequencePolicy(pattern), make_split().validation, 71002, "blind_validation")
                    return {"index": index, "pattern": list(pattern), "metrics": metrics, "records": records}, len(records)
                candidate = cached_evaluation(root / "blind_work", root / "blind_work" / f"pattern_{index:03d}.json.gz",
                                              identity, 252, calculate)
                candidates.append(candidate)
            write_once(path, {"protocol_digest": config["protocol_digest"], "candidates": candidates})
    best = max(read_json(path)["candidates"], key=lambda c: selection_key(c["metrics"]))
    selected = {"index": best["index"], "pattern": best["pattern"], "validation_metrics": best["metrics"],
                "validation_artifact_sha256": file_digest(path)}
    return lock_candidates(root, selected)


def evaluate_locked(root):
    root = Path(root)
    config = load(root)
    locked = verify_candidate_lock(root)
    split = make_split()
    lookup = {ep.episode_id: ep for ep in split.test}
    for job in config["jobs"]:
        path = directory(root, **job)
        if (path / "test.json.gz").exists():
            audit_test_job(root, job)
            continue
        selected = read_json(path / "result.json")
        tested, seen = {}, set()
        for role in ("best", "initial", "final"):
            genome = Genome.from_dict(selected[role]["genome"], NETWORK)
            artifact = path / "role_cache" / (genome.digest + ".json.gz")
            identity = {"protocol_digest": config["protocol_digest"], "lock_digest": locked["lock_digest"],
                        **job, "genome_digest": genome.digest, "purpose": "test", "repeats": 3}
            def calculate(genome=genome):
                metrics, records = evaluate_split(root, **job, genome=genome, episodes=split.test, purpose="test", repeats=3)
                return {"metrics": metrics, "records": records}, len(records)
            payload = cached_evaluation(path / "test_work", artifact, identity, 1512, calculate)
            tested[role] = {**payload, "digest": genome.digest, "cache_reused": genome.digest in seen,
                            "cache_sha256": file_digest(artifact)}
            seen.add(genome.digest)
        # 프로브는 환경 에피소드와 단위가 다르며 별도 단일 행동 호출로 기록한다.
        probe_identity = {"protocol_digest": config["protocol_digest"], "lock_digest": locked["lock_digest"],
                          **job, "purpose": "initial_state_probe", "genome_digest": selected["best"]["digest"]}
        probe_payload = cached_evaluation(path / "probe_work", path / "probe_cache.json.gz", probe_identity, 0,
            lambda: (dict(zip(("metrics","records"),probe(Genome.from_dict(selected["best"]["genome"], NETWORK)))), 0),
            single_step_actions=441)
        write_once(path / "probe.json.gz", probe_payload)
        diagnostics = verify_trajectories(tested["best"]["records"], lookup)
        write_once(path / "diagnostics.json", diagnostics)
        write_once(path / "test.json.gz", {"lock_digest": locked["lock_digest"], "roles": tested,
                   "logical_episode_slots": 4536, "unique_role_episode_calls": len(seen) * 1512,
                   "actual_episode_accounting": attempt_accounting(path / "test_work"),
                   "probe_states": len(probe_payload["records"]), "probe_single_step_actions": len(probe_payload["records"])})
        print(f"TEST {job['condition']}/{job['method']}/{job['seed']} done", flush=True)
    if not (root / "baseline_results.json").exists():
        if config["phase"] == "confirmation":
            if file_digest(RESULTS / "main/baseline_results.json") != config["shared_baseline_sha256"]:
                raise ValueError("Shared baseline changed")
            write_once(root / "baseline_results.json", read_json(RESULTS / "main/baseline_results.json"))
        else:
            policies = {"blind": (FixedSequencePolicy(tuple(locked["blind"]["pattern"])), 71002, "blind_test", 1),
                        "uniform": (UniformRandomPolicy(), 71003, "uniform_test", 20),
                        "coordinate": (CoordinatePolicy(), 71003, "coordinate_test", 1)}
            baseline, raw = {}, {}
            for name, (policy, seed, purpose, repeats) in policies.items():
                def calculate(policy=policy, seed=seed, purpose=purpose, repeats=repeats):
                    metrics, records = evaluate_policy(policy, split.test, seed, purpose, repeats=repeats)
                    return {"metrics": metrics, "records": records}, len(records)
                payload = cached_evaluation(root / "baseline_work", root / "baseline_work" / (name + ".json.gz"),
                    {"protocol_digest": config["protocol_digest"], "purpose": purpose, "repeats": repeats}, 504 * repeats, calculate)
                baseline[name], raw[name] = payload["metrics"], payload["records"]
            write_once(root / "baseline_records.json.gz", raw)
            write_once(root / "baseline_results.json", baseline)
    if config["phase"] == "main":
        raw = read_json(root / "baseline_records.json.gz")
        for name, records in raw.items():
            check_metrics(read_json(root / "baseline_results.json")[name], records)
            verify_trajectories(records, lookup)
    audits = [audit_test_job(root, job) for job in config["jobs"]]
    write_once(root / "test_audit.json", {"passed": True, "jobs": audits})
    summary = summarize(root)
    write_once(root / "summary.json", summary)
    return summary
