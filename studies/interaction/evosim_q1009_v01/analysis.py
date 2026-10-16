"""초기 유전체 짝 차이와 기술 통계. 후보 선택·확인 판정은 없다."""
from collections import Counter
from pathlib import Path
import numpy as np
from evosim_interaction_v02.storage import read_json, file_digest
from .protocol import load
from .execute import CONTROLS, accounting, execute
from .model import CONDITIONS


def distribution(values):
    actual = np.array([v for v in values if v is not None], dtype=float)
    return {"mean":float(actual.mean()) if len(actual) else None,
            "median":float(np.median(actual)) if len(actual) else None,
            "sd":float(actual.std(ddof=1)) if len(actual)>1 else None,
            "minimum":float(actual.min()) if len(actual) else None,
            "maximum":float(actual.max()) if len(actual) else None,
            "observed":len(actual),"missing":len(values)-len(actual),"values":values}


def paired(values, draws):
    a = np.array(values,dtype=float)
    return {**distribution(values), "ci95":np.percentile(a[draws].mean(axis=1),[2.5,97.5],method="linear").tolist(),
            "positive":int((a>0).sum()),"negative":int((a<0).sum()),"zero":int((a==0).sum())}


def summarize(root):
    root = Path(root)
    config, episodes = load(root)
    if config["phase"] != "main" or config["fixture"]:
        raise ValueError("Official main cohort required")
    rows, hashes, controls, counts = [], {}, {}, Counter()
    def get(name):
        raw, audit = root/"raw"/(name+".json.gz"), root/"audits"/(name+".json")
        hashes[name] = {"raw":file_digest(raw),"audit":file_digest(audit)}
        for prefix,path in (("evaluation",root/"work"/name),("audit",root/"audit_work"/name)):
            for k,v in accounting(path).items():
                counts[prefix+"_"+k] += v
        return read_json(raw),read_json(audit)
    for seed in config["seeds"]:
        execute(root,seed)  # 캐시 식별자·해시만 재확인. 완료 감사는 재평가하지 않는다.
        payload,audit = get(f"seed_{seed}")
        row = {"seed":seed,"conditions":{}}
        for item in payload["conditions"]:
            name = item["condition"]
            row["conditions"][name] = {"metrics":item["metrics"],"probe":item["probe"],"trajectory":audit["diagnostics"][name]}
        rows.append(row)
    control_counts_before = dict(counts)
    for name in CONTROLS:
        execute(root,73009,name)
        payload,audit = get("control_"+name)
        controls[name] = {r["condition"]:{"metrics":r["metrics"],"probe":r["probe"]} for r in payload["conditions"]}
    groups = {}
    for c in CONDITIONS:
        groups[c] = {scope:{key:distribution([r["conditions"][c][scope][key] for r in rows])
                            for key in rows[0]["conditions"][c][scope]}
                     for scope in ("metrics","probe")}
    draws = np.random.default_rng(72009).integers(0,len(rows),size=(10000,len(rows)))
    contrasts = {}
    for c in CONDITIONS[1:]:
        contrasts[c] = {}
        for scope,key in (("metrics","unproductive_tail_rate"),("metrics","success_rate"),
                          ("probe","strict_direction_pair_rate"),("probe","strict_collection_rate")):
            contrasts[c][key] = paired([r["conditions"][c][scope][key]-r["conditions"]["C0N0"][scope][key] for r in rows],draws)
    interaction = paired([r["conditions"]["C1N1"]["metrics"]["unproductive_tail_rate"]
        -r["conditions"]["C1N0"]["metrics"]["unproductive_tail_rate"]
        -r["conditions"]["C0N1"]["metrics"]["unproductive_tail_rate"]
        +r["conditions"]["C0N0"]["metrics"]["unproductive_tail_rate"] for r in rows],draws)
    return {"experiment_id":"Q1-009","source_digest":config["source_digest"],"protocol_digest":config["protocol_digest"],
            "sample_size":len(rows),"groups":groups,"contrasts":contrasts,"interaction_unproductive_tail":interaction,
            "rows":rows,"controls":controls,"counts_all_including_controls":dict(counts),
            "counts_genomes_only":control_counts_before,"artifacts":hashes,
            "primary_metric":"unproductive_tail_rate_all_episodes","statistical_scope":"descriptive initial-genome paired intervals; no hypothesis testing, selection or evolution claim",
            "episode_records_replayed":len(episodes)*4*(len(rows)+len(CONTROLS)),"independent_review":False}
