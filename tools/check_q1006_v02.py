"""Q1-006 새 검사와 기존173검사를 실행한 뒤 소스·계획을 잠근다."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/"studies/interaction"


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve()
    if out.exists() or not out.is_relative_to(ROOT):raise ValueError("Use a new workspace gate file")
    sys.path.insert(0,str(STUDY))
    from evosim_q1006_v02.source import source_manifest as sources,plan
    import platform
    import numpy as np
    def runtime():return {"python":platform.python_version(),"numpy":np.__version__,"platform":sys.platform}
    from evosim_interaction_v02.storage import manifest_digest
    from evosim_interaction_v02.settings import digest_json
    before=sources()
    commands=[(STUDY,[sys.executable,"-B","-X","utf8","-m","unittest","discover","-s","evosim_q1006_v02/tests","-v"]),
        (ROOT,[sys.executable,"-B","-X","utf8","tools/check_q1005_portable.py"]),
        (ROOT,[sys.executable,"-B","-X","utf8","tools/check_q1_core.py","--output","dist/q1006_v02_gate/dependency.json"]),
        (ROOT,[sys.executable,"-B","-X","utf8","tools/check_research_layout.py","--tests"]),
        (ROOT,[sys.executable,"-B","-X","utf8","-m","unittest","discover","-s","tools/tests","-v"])]
    logs,records=[],[]
    for cwd,cmd in commands:
        start=time.perf_counter();r=subprocess.run(cmd,cwd=cwd,capture_output=True,text=True,encoding="utf-8")
        log=r.stdout+r.stderr;logs.append(log)
        count=re.findall(r"Ran (\d+) tests?",log)
        records.append({"cwd":cwd.relative_to(ROOT).as_posix(),"command":cmd,"returncode":r.returncode,
                        "tests":sum(map(int,count)),"seconds":time.perf_counter()-start})
    commands_extra=(STUDY,[sys.executable,"-B","-X","utf8","-m","unittest","discover","-s","evosim_q1009_v01/tests","-v"])
    r=subprocess.run(commands_extra[1],cwd=STUDY,capture_output=True,text=True,encoding="utf-8")
    logs.append(r.stdout+r.stderr)
    records.append({"cwd":"studies/interaction","command":commands_extra[1],"returncode":r.returncode,"tests":sum(map(int,re.findall(r"Ran (\d+) tests?",r.stdout+r.stderr)))})
    dependency=json.loads((ROOT/"dist/q1006_v02_gate/dependency.json").read_text(encoding="utf-8"))
    legacy=json.loads((ROOT/"docs/management/verification.json").read_text(encoding="utf-8"))
    records[2]["tests"]=dependency["tests_run"]
    records[3]["tests"]=sum(study["tests"] for study in legacy["studies"])
    passed=all(r["returncode"]==0 for r in records) and before==sources()
    result={"checked_utc":datetime.now(timezone.utc).isoformat(),"passed":passed,"source_digest":manifest_digest(before),
        "full_gate":True,"sources":before,"plan_digest":digest_json(plan()),"runtime":runtime(),"commands":records,
        "tests":sum(r["tests"] for r in records),"new_tests":records[0]["tests"],"previous_tests":sum(r["tests"] for r in records[1:]),
        "gate_tool_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"independent_review":False}
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding="utf-8",newline="\n")
    out.with_suffix('.log').write_text('\n'.join(logs),encoding="utf-8",newline="\n")
    print(json.dumps({k:result[k] for k in ("passed","source_digest","tests","new_tests","previous_tests")},ensure_ascii=False),flush=True)
    if not passed:
        print('\n'.join(logs));raise SystemExit(1)


if __name__=="__main__":main()
