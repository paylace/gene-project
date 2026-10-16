"""행동의 환경 재생과 변환·초기 반응 원자료 감사."""
import math
import numpy as np
from evosim_q1005_v01.audit import verify_trajectories
from .evaluation import summarize_records
from .probe import choices, summarize_probe, probe
from .model import transform, initial, control


def audit_records(records, episodes, repeats=1):
    if [(r["episode_id"], r["repeat"]) for r in records] != [(ep.episode_id, rep) for rep in range(repeats) for ep in episodes]:
        raise ValueError("Episode coverage/order differs")
    diagnostics = verify_trajectories(records, {ep.episode_id: ep for ep in episodes})
    lengths = np.array([len(r["actions"]) for r in records])
    specifications = list(episodes)*repeats
    position = np.array([ep.start for ep in specifications])
    targets = np.array([ep.targets for ep in specifications])
    collected = np.zeros(len(records), dtype=int)
    walls = np.zeros(len(records), dtype=int)
    failed = np.zeros(len(records), dtype=int)
    action_matrix = np.array([[int(a) for a in r["actions"]] + [-1]*(24-len(r["actions"])) for r in records])
    for t in range(24):
        active = t < lengths
        a = action_matrix[:, t]
        goal = targets[np.arange(len(records)), np.minimum(collected, 2)]
        new = np.clip(position + np.where(a == 0, -1, np.where(a == 1, 1, 0)), 0, 6)
        tail = active & (t >= lengths-6)
        walls += tail & (a != 2) & (new == position)
        failed += tail & (a == 2) & (position != goal)
        collected += active & (a == 2) & (position == goal)
        position = np.where(active, new, position)
    for i, record in enumerate(records):
        same = lengths[i] >= 6 and len(set(record["actions"][-6:])) == 1
        kind = "wall" if same and walls[i] == 6 else "failed_collect" if same and failed[i] == 6 else "other"
        if record["tail_identical"] != same or record["tail_kind"] != kind:
            raise ValueError("Unproductive tail classification differs")
        if not math.isfinite(record["peak_abs_logit"]) or record["peak_abs_logit"] < 0:
            raise ValueError("Invalid recorded peak")
    return diagnostics


def audit_job(payload, episodes):
    base = control(payload["control"]) if payload.get("control") else initial(payload["seed"])
    if payload["base_genome"] != base.to_dict():
        raise ValueError("Initial genotype differs")
    result = {}
    for row in payload["conditions"]:
        condition = row["condition"]
        genome = transform(base, condition)
        if row["genome"] != genome.to_dict() or row["genome_digest"] != genome.digest:
            raise ValueError("Condition transformation differs")
        diagnostic = audit_records(row["records"], episodes, payload.get("evaluation_repeats",1))
        metrics = summarize_records(row["records"])
        if any(row["metrics"].get(k) != value for k, value in metrics.items()):
            raise ValueError("Aggregate metrics differ")
        rebuilt, records = probe(genome, condition)
        if rebuilt != row["probe"] or records != row["probe_records"]:
            raise ValueError("Cold-state probe differs")
        if choices([r["logits"] for r in records]) != [r["choices"] for r in records]:
            raise ValueError("Choice sets differ")
        result[condition] = diagnostic
    rows = {row["condition"]: row for row in payload["conditions"]}
    if set(rows) != {"C0N0", "C1N0", "C0N1", "C1N1"}:
        raise ValueError("All conditions required")
    for c in ("0", "1"):
        if rows[f"C{c}N0"]["probe_records"] != rows[f"C{c}N1"]["probe_records"]:
            raise ValueError("Output recurrence affected a zero-history first step")
    return {"passed": True, "episode_records_replayed": 4*len(episodes)*payload.get("evaluation_repeats",1),
            "probe_states_recomputed": 1764, "diagnostics": result,
            "network_episode_actions_recomputed": False, "recorded_logit_peaks_recomputed": False}
