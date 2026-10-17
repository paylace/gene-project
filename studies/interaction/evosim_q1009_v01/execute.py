"""유전체별 불변 원자료·감사 캐시와 실제 작업 원장."""
from datetime import datetime, timezone
from pathlib import Path
import uuid
from evosim_interaction_v02.storage import read_json, write_json, write_once, file_digest
from evosim_interaction_v02.settings import digest_json
from .protocol import load
from .model import CONDITIONS, initial, control, transform
from .evaluation import evaluate
from .probe import probe
from .audit import audit_job

CONTROLS = ("zero", "left", "right", "collect", "coordinate")


def accounting(path):
    result = {"known_episode_calls":0, "unknown_episode_calls_upper":0,
              "known_probe_states":0, "unknown_probe_states_upper":0, "interrupted_attempts":0}
    for artifact in (Path(path)/"attempts").glob("*.json"):
        item = read_json(artifact)
        if item["status"] in ("evaluated", "committed"):
            result["known_episode_calls"] += item["completed_episode_calls"]
            result["known_probe_states"] += item["completed_probe_states"]
        else:
            result["unknown_episode_calls_upper"] += item["maximum_episode_calls"]
            result["unknown_probe_states_upper"] += item["maximum_probe_states"]
            result["interrupted_attempts"] += 1
    return result


def begin(path, episodes, states):
    target = Path(path)/"attempts"/(uuid.uuid4().hex+".json")
    record = {"status":"started", "started_utc":datetime.now(timezone.utc).isoformat(),
              "maximum_episode_calls":episodes, "maximum_probe_states":states}
    write_json(target, record)
    return target, record


def complete(target, record, episodes, states):
    record.update(status="evaluated", completed_episode_calls=episodes, completed_probe_states=states)
    write_json(target, record)


def execute(root, seed, control_name=None):
    root = Path(root)
    config, episodes = load(root)
    if (control_name is None and seed not in config["seeds"]) or (control_name is not None and
            (control_name not in CONTROLS or seed != 73009 or config["phase"] != "main")):
        raise ValueError("Job outside protocol")
    name = f"control_{control_name}" if control_name else f"seed_{seed}"
    work = root/"work"/name
    artifact, audited = root/"raw"/(name+".json.gz"), root/"audits"/(name+".json")
    identity = {"protocol_digest":config["protocol_digest"], "seed":seed, "control":control_name}
    if artifact.exists():
        payload = read_json(artifact)
        if payload["identity"] != identity or payload["payload_digest"] != digest_json({k:v for k,v in payload.items() if k != "payload_digest"}):
            raise ValueError("Saved result identity/checksum differs")
    else:
        attempt_path, attempt = begin(work, 4*len(episodes), 1764)
        base = control(control_name) if control_name else initial(seed)
        rows = []
        for condition in CONDITIONS:
            genome = transform(base, condition)
            metrics, records = evaluate(genome, episodes, seed, "q1009:"+config["phase"], condition)
            probed, probe_records = probe(genome, condition)
            rows.append({"condition":condition,"genome":genome.to_dict(),"genome_digest":genome.digest,
                         "metrics":metrics,"records":records,"probe":probed,"probe_records":probe_records})
        payload = {"identity":identity, "seed":seed, "control":control_name, "base_genome":base.to_dict(), "conditions":rows}
        payload["payload_digest"] = digest_json(payload)
        complete(attempt_path, attempt, 4*len(episodes), 1764)
        write_once(artifact, payload)
        attempt.update(status="committed")
        write_json(attempt_path, attempt)
    if audited.exists():
        audit = read_json(audited)
        if (audit["audit_digest"] != digest_json({k:v for k,v in audit.items() if k != "audit_digest"})
                or audit["identity"] != identity or audit["result_sha256"] != file_digest(artifact) or not audit["passed"]):
            raise ValueError("Saved audit changed or raw result differs")
    else:
        attempt_path, attempt = begin(root/"audit_work"/name, 0, 1764)
        audit = audit_job(payload, episodes)
        audit.update(identity=identity, result_sha256=file_digest(artifact))
        audit["audit_digest"] = digest_json(audit)
        complete(attempt_path, attempt, 0, 1764)
        write_once(audited, audit)
        attempt.update(status="committed")
        write_json(attempt_path, attempt)
    return {"name":name,"result_sha256":file_digest(artifact),"audit_sha256":file_digest(audited),"passed":True}
