"""훈련 목록만 사용한 실제 본 크기 비용 표본. 성능을 설계에 쓰지 않는다."""
from pathlib import Path
import time
from evosim_interaction_v02.datasets import make_split
from evosim_interaction_v02.storage import write_once, read_json, file_digest
from .model import initial,transform,CONDITIONS
from .evaluation import evaluate
from .probe import probe
from .audit import audit_job
from .execute import begin,complete,accounting


def measure(root):
    root = Path(root)
    metadata = root/"cost_samples/report.json"
    if metadata.exists():
        result = read_json(metadata)
        for name,digest in result["files"].items():
            if file_digest(root/"cost_samples"/name) != digest:
                raise ValueError("Cost sample changed")
        return result
    started = time.perf_counter()
    files = {}
    for seed in (5900,5901):
        attempt_path,attempt = begin(root/"cost_work"/f"seed_{seed}",6048,1764)
        base = initial(seed)
        rows = []
        for condition in CONDITIONS:
            genome = transform(base,condition)
            metrics,records = evaluate(genome,make_split().train,seed,"q1009:cost",condition,repeats=2)
            probed,probe_records = probe(genome,condition)
            rows.append({"condition":condition,"genome":genome.to_dict(),"genome_digest":genome.digest,
                         "metrics":metrics,"records":records,"probe":probed,"probe_records":probe_records})
        payload = {"seed":seed,"control":None,"base_genome":base.to_dict(),"conditions":rows,"evaluation_repeats":2}
        artifact = root/"cost_samples"/f"seed_{seed}.json.gz"
        if artifact.exists():
            raise ValueError("Unfinished cost sample retained; review rather than recompute")
        complete(attempt_path,attempt,6048,1764)
        write_once(artifact,payload)
        attempt.update(status="committed")
        from evosim_interaction_v02.storage import write_json
        write_json(attempt_path,attempt)
        audit_attempt,audit_record = begin(root/"cost_audit_work"/f"seed_{seed}",0,1764)
        audit = audit_job(read_json(artifact),make_split().train)
        audit_path = root/"cost_samples"/f"audit_{seed}.json"
        write_once(audit_path,audit)
        complete(audit_attempt,audit_record,0,1764)
        audit_record.update(status="committed")
        write_json(audit_attempt,audit_record)
        files[artifact.name],files[audit_path.name] = file_digest(artifact),file_digest(audit_path)
    elapsed = time.perf_counter()-started
    size = sum((root/"cost_samples"/name).stat().st_size for name in files)
    result = {"elapsed_seconds":elapsed,"bytes":size,"files":files,"episode_calls":12096,
              "probe_states":3528,"audit_probe_states":3528,"record_replays":12096,
              "episode_source":"training756 repeated twice; no validation/test episodes",
              "actual_accounting":{str(s):accounting(root/"cost_work"/f"seed_{s}") for s in (5900,5901)},
              "audit_accounting":{str(s):accounting(root/"cost_audit_work"/f"seed_{s}") for s in (5900,5901)},
              "performance_used_for_design":False}
    write_once(metadata,result)
    return result
