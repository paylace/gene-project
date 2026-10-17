"""새 실험의 소스·수치 계획·실행 환경과 단계 접근을 고정한다."""
from pathlib import Path
import platform
import sys
import numpy as np
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.storage import read_json, write_once, file_digest, manifest_digest
from evosim_interaction_v02.settings import digest_json
from evosim_q1005_v01.source import source_manifest as dependency_manifest
from .model import CONDITIONS

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "studies/interaction/results_q1009_v01"
PLAN = ROOT / "docs/protocols/q1_009_v01_plan.json"
DEPENDENCY = "ccbf6e00f455aeabd76fb06e02574a45191a70c6e3637db34fd85463354a824c"


def sources():
    dependency = dependency_manifest()
    if manifest_digest(dependency) != DEPENDENCY:
        raise ValueError("Frozen dependency changed")
    package = Path(__file__).parent
    return {**{"frozen/"+name: value for name,value in dependency.items()},
            **{"q1009/"+p.relative_to(package).as_posix(): file_digest(p) for p in sorted(package.rglob("*.py"))}}


def plan():
    p = read_json(PLAN)
    if (p["experiment_id"] != "Q1-009" or p["conditions"] != list(CONDITIONS)
            or p["seeds"] != list(range(6000,6500)) or p["pilot_seeds"] != [5900,5901]
            or p["dataset_digest"] != make_split().digest or p["episode_repeats"] != 1):
        raise ValueError("Unsupported plan")
    return p


def runtime():
    return {"python": platform.python_version(), "numpy": np.__version__, "platform": sys.platform}


def prepare(root, phase, gate=None, *, fixture=False, fixture_seeds=(41,), fixture_episodes=4):
    root = Path(root).resolve()
    p, manifest = plan(), sources()
    if not fixture and root != (RESULTS / phase).resolve():
        raise ValueError("Official output must use registered location")
    if phase not in ("pilot", "main"):
        raise ValueError("Unknown phase")
    if fixture:
        seeds = list(fixture_seeds)
        episodes = list(make_split().train[:fixture_episodes])
    else:
        evidence = read_json(gate) if gate else {}
        if not evidence.get("passed") or evidence.get("source_digest") != manifest_digest(manifest) or evidence.get("plan_digest") != digest_json(p):
            raise ValueError("Matching execution gate required")
        seeds = p["pilot_seeds"] if phase == "pilot" else p["seeds"]
        episodes = list(make_split().train[:24] if phase == "pilot" else (*make_split().train, *make_split().validation, *make_split().test))
        if phase == "main":
            pilot = read_json(RESULTS / "pilot/cost_report.json")
            if not pilot["passed"] or pilot["source_digest"] != manifest_digest(manifest) or pilot["plan_digest"] != digest_json(p):
                raise ValueError("Passing matching pilot required")
    config = {"experiment_id":"Q1-009", "phase":phase, "fixture":fixture, "plan":p,
              "plan_digest":digest_json(p), "sources":manifest, "source_digest":manifest_digest(manifest),
              "runtime":runtime(), "conditions":list(CONDITIONS), "seeds":seeds,
              "episode_ids":[ep.episode_id for ep in episodes], "dataset_digest":make_split().digest}
    if not fixture:
        config.update(gate_path=Path(gate).resolve().relative_to(ROOT).as_posix(), gate_sha256=file_digest(gate))
        if phase == "main":
            config["pilot_report_sha256"] = file_digest(RESULTS / "pilot/cost_report.json")
    config["protocol_digest"] = digest_json(config)
    write_once(root / "protocol.json", config)
    return config


def load(root):
    root = Path(root).resolve()
    config = read_json(root / "protocol.json")
    if config["protocol_digest"] != digest_json({k:v for k,v in config.items() if k != "protocol_digest"}):
        raise ValueError("Protocol checksum differs")
    if config["sources"] != sources() or config["source_digest"] != manifest_digest(sources()):
        raise ValueError("Source changed")
    if config["plan_digest"] != digest_json(plan()) or config["plan"] != plan() or config["runtime"] != runtime():
        raise ValueError("Plan/runtime changed")
    if config["dataset_digest"] != make_split().digest:
        raise ValueError("Dataset changed")
    if not config["fixture"]:
        if root != (RESULTS / config["phase"]).resolve() or file_digest(ROOT / config["gate_path"]) != config["gate_sha256"]:
            raise ValueError("Official path/gate changed")
        if config["phase"] == "main" and file_digest(RESULTS / "pilot/cost_report.json") != config["pilot_report_sha256"]:
            raise ValueError("Pilot changed")
    all_episodes = {ep.episode_id:ep for split in (make_split().train,make_split().validation,make_split().test) for ep in split}
    if len(set(config["episode_ids"])) != len(config["episode_ids"]):
        raise ValueError("Repeated episode IDs")
    return config, tuple(all_episodes[identity] for identity in config["episode_ids"])
