"""모든 후보와 blind 선발을 잠근 후에만 동결 테스트를 연다."""

from pathlib import Path

from .audit import audit_job, check_metrics
from .datasets import make_split
from .evaluation import evaluate_policy, selection_key, observation_probe
from .policies import FixedSequencePolicy, CoordinatePolicy, UniformRandomPolicy, blind_patterns
from .protocol import ROOT, load, evaluate_split
from .search import candidate_manifest, lock_candidates, verify_candidate_lock, directory
from .network import Genome
from .settings import Settings
from .storage import file_digest, read_json, write_once


def validate_blind_selection(root, selection):
    root = Path(root)
    config = load(root)
    saved = read_json(root / "blind_validation.json.gz")
    if saved["protocol_digest"] != config.get("shared_baseline_protocol", config["protocol_digest"]):
        raise ValueError("Blind validation protocol mismatch")
    patterns, episodes = blind_patterns(), make_split().validation
    candidates = saved["candidates"]
    if len(candidates) != len(patterns):
        raise ValueError("Blind candidate pool incomplete")
    best = None
    for index, (pattern, candidate) in enumerate(zip(patterns, candidates)):
        if candidate["index"] != index or candidate["pattern"] != list(pattern):
            raise ValueError("Blind pattern/index mismatch")
        if [(r["episode_id"], r["repeat"]) for r in candidate["records"]] != [(e.episode_id, 0) for e in episodes]:
            raise ValueError("Blind selection used incorrect split or repeat")
        check_metrics(candidate["metrics"], candidate["records"])
        if best is None or selection_key(candidate["metrics"]) > selection_key(best["metrics"]):
            best = candidate
    expected = {"index": best["index"], "pattern": best["pattern"], "validation_metrics": best["metrics"],
                "validation_artifact_sha256": file_digest(root / "blind_validation.json.gz")}
    if selection != expected:
        raise ValueError("Blind selection does not match validation-only winner")
    return expected


def lock_all_candidates(root):
    root = Path(root)
    config = load(root)
    if config["phase"] not in ("main", "confirmation"):
        raise ValueError("Only main may select candidates for held-out evaluation")
    candidate_manifest(root, config)  # 미완료 실행이 있으면 검증셋 선발도 시작하지 않음
    audit = [audit_job(root, c, m, s) for c in config["conditions"] for m in config["methods"] for s in config["seeds"]]
    write_once(root / "search_audit.json", {"protocol_digest": config["protocol_digest"], "jobs": audit})
    path = root / "blind_validation.json.gz"
    if not path.exists():
        if config["phase"] == "confirmation":
            shared = ROOT / config["shared_baseline_root"]
            if file_digest(shared / "summary.json") != config["main_summary_sha256"] or file_digest(shared / path.name) != config["shared_blind_sha256"]:
                raise ValueError("Shared main results changed")
            write_once(path, read_json(shared / path.name))
        else:
            candidates = []
            for index, pattern in enumerate(blind_patterns()):
                metrics, records = evaluate_policy(FixedSequencePolicy(pattern), make_split().validation, 71002, "blind_validation")
                candidates.append({"index": index, "pattern": list(pattern), "metrics": metrics, "records": records})
            write_once(path, {"protocol_digest": config["protocol_digest"], "candidates": candidates})
    saved = read_json(path)
    best = max(saved["candidates"], key=lambda c: selection_key(c["metrics"]))  # first tie preserved
    selection = {"index": best["index"], "pattern": best["pattern"], "validation_metrics": best["metrics"],
                 "validation_artifact_sha256": file_digest(path)}
    validate_blind_selection(root, selection)
    return lock_candidates(root, selection)


def evaluate_locked(root):
    """모든 실행을 감사·잠금한 뒤 역할 캐시를 사용해 최종 평가한다."""
    from .analysis import diagnostic_summary, summarize_results
    root = Path(root)
    config = load(root)
    locked = verify_candidate_lock(root)
    split = make_split()
    for condition in config["conditions"]:
        settings = Settings(condition)
        for method in config["methods"]:
            for seed in config["seeds"]:
                path = directory(root, condition, method, seed)
                if (path / "test.json.gz").exists():
                    saved = read_json(path / "test.json.gz")
                    if saved["lock_digest"] != locked["lock_digest"]:
                        raise ValueError("Existing test belongs to another candidate lock")
                    continue
                selected = read_json(path / "result.json")
                tested, cache, actual_calls = {}, {}, 0
                for role in ("best", "initial", "final"):
                    genome = Genome.from_dict(selected[role]["genome"], settings)
                    if genome.digest in cache:
                        tested[role] = {**cache[genome.digest], "cache_reused": True}
                        continue
                    metrics, records = evaluate_split(root, genome, split.test, seed, "test", repeats=3,
                                                       observe=role == "best" and config["phase"] == "main")
                    actual_calls += len(records)
                    if role == "best":
                        write_once(path / "diagnostics.json.gz", {"lock_digest": locked["lock_digest"],
                                   "summary": diagnostic_summary(records), "records": records})
                    plain = [{key: value for key, value in row.items() if key != "observations"} for row in records]
                    entry = {"digest": genome.digest, "metrics": metrics, "records": plain, "cache_reused": False}
                    cache[genome.digest] = entry
                    tested[role] = entry
                probe = observation_probe(Genome.from_dict(selected["best"]["genome"], settings), seed, settings)
                write_once(path / "probe.json.gz", probe)
                write_once(path / "test.json.gz", {"lock_digest": locked["lock_digest"], "roles": tested,
                           "logical_episode_slots": 3 * len(split.test) * 3, "actual_episode_calls": actual_calls})
                print(f"TEST {condition}/{method}/{seed} done", flush=True)
    baseline_path = root / "baseline_results.json"
    if not baseline_path.exists():
        if config["phase"] == "confirmation":
            write_once(baseline_path, read_json(ROOT / config["shared_baseline_root"] / baseline_path.name))
        else:
            patterns = {"blind": (FixedSequencePolicy(tuple(locked["blind"]["pattern"])), 71002, "blind_test", 1),
                        "uniform": (UniformRandomPolicy(), 71003, "uniform_test", 20),
                        "coordinate": (CoordinatePolicy(), 71003, "coordinate_test", 1)}
            baseline, raw = {}, {}
            for name, (policy, seed, purpose, repeats) in patterns.items():
                metrics, records = evaluate_policy(policy, split.test, seed, purpose, repeats=repeats)
                baseline[name], raw[name] = metrics, records
            write_once(root / "baseline_records.json.gz", raw)
            write_once(baseline_path, baseline)
    from .audit import audit_test_job
    audit = [audit_test_job(root, c, m, s) for c in config["conditions"] for m in config["methods"] for s in config["seeds"]]
    write_once(root / "test_audit.json", {"passed": True, "jobs": audit})
    summary = summarize_results(root)
    write_once(root / "summary.json", summary)
    return summary
