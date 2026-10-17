"""검사 게이트→훈련 목록 비용 예비→고정 초기 유전체 진단."""
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import time
from evosim_interaction_v02.storage import write_once,write_json,read_json
from .protocol import RESULTS,prepare,load
from .execute import execute,accounting,CONTROLS
from .analysis import summarize


def run(root,config,seeds,workers):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(execute,root,seed):seed for seed in seeds}
        for future in as_completed(jobs):
            try:
                future.result()
                print("AUDITED",jobs[future],flush=True)
            except Exception as error:
                write_json(Path(root)/"failure.json",{"seed":jobs[future],"error":repr(error)})
                raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase",choices=("pilot","main"))
    parser.add_argument("--gate",type=Path,required=True)
    parser.add_argument("--workers",type=int,default=2)
    args = parser.parse_args()
    if args.workers not in (1,2,3,4):
        raise ValueError("Use one to four workers")
    root = RESULTS/args.phase
    if (root/"failure.json").exists():
        raise ValueError("Preserved failure requires review")
    config = prepare(root,args.phase,args.gate)
    finished = root/("cost_report.json" if args.phase == "pilot" else "summary.json")
    if finished.exists():
        record = read_json(finished)
        if record["source_digest"] != config["source_digest"]:
            raise ValueError("Completed artifact belongs to another source")
        print("ALREADY COMPLETE",args.phase,flush=True)
        return
    started = time.perf_counter()
    if args.phase == "pilot":
        # 실제 단계 경계에서 정지한 뒤 같은 파일을 다시 읽어 호출 0 재개를 검사한다.
        run(root,config,config["seeds"][:1],args.workers)
        before = accounting(root/"work"/f"seed_{config['seeds'][0]}")
        run(root,config,config["seeds"],args.workers)
        after = accounting(root/"work"/f"seed_{config['seeds'][0]}")
        elapsed = time.perf_counter()-started
        size = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
        from .cost import measure
        cost = measure(root)
        # 본과 같은1512열을 훈련 목록으로만 측정. 병렬 이득 없이1.5배 여유를 둔다.
        estimated_seconds = cost["elapsed_seconds"] * (500/2)*1.5
        estimated_bytes = cost["bytes"] * (500/2)*1.5
        unknown = sum(accounting(root/scope/f"seed_{s}")["interrupted_attempts"] for s in config["seeds"] for scope in ("work","audit_work","cost_work","cost_audit_work"))
        passed = before == after and not unknown and estimated_seconds <= config["plan"]["cost_limits"]["estimated_main_seconds"] and estimated_bytes <= config["plan"]["cost_limits"]["estimated_main_bytes"]
        report = {"passed":passed,"all_jobs_complete":True,"source_digest":config["source_digest"],"plan_digest":config["plan_digest"],
                  "elapsed_seconds":elapsed,"bytes":size,"estimated_main_seconds":estimated_seconds,
                  "estimated_main_bytes":estimated_bytes,"model":"serial full-shape training-only cost sample scaled by500/2 and1.5 margin; not ETA or guaranteed bound",
                  "resume_added_calls":after["known_episode_calls"]-before["known_episode_calls"],"unknown_interrupted_attempts":unknown,
                  "known_episode_calls":192,"probe_states":3528,"audit_probe_states":3528,
                  "additional_cost_sample":cost,"performance_used_for_design":False}
        write_once(root/"cost_report.json",report)
        print(report,flush=True)
        if not passed:
            raise SystemExit("Cost/resume gate failed; main not started")
    else:
        run(root,config,config["seeds"],args.workers)
        for name in CONTROLS:
            execute(root,73009,name)
            print("CONTROL AUDITED",name,flush=True)
        result = summarize(root)
        result["completion_seconds"] = time.perf_counter()-started
        write_once(root/"summary.json",result)
        print("MAIN COMPLETE",len(config["seeds"]),flush=True)


if __name__ == "__main__":
    main()
