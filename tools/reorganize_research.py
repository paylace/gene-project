"""Prepare a content-preserving relocation manifest; generate readable research docs."""

import argparse
import hashlib
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATE = "2026-10-15"
MANAGEMENT = ROOT / "docs" / "management"
MOVES = [
    ("evosim_research", "studies/working_memory"),
    ("evosim_interaction", "studies/interaction/evosim_interaction"),
    ("audit_interaction.py", "studies/interaction/audit_interaction.py"),
    ("diagnose_interaction.py", "studies/interaction/diagnose_interaction.py"),
    ("probe_action_bottleneck.py", "studies/interaction/probe_action_bottleneck.py"),
    ("summarize_interaction.py", "studies/interaction/summarize_interaction.py"),
    ("evosim", "archive/original_simulator/evosim"),
    ("tests", "archive/original_simulator/tests"),
    ("evolutionary_simulator.py", "archive/original_simulator/evolutionary_simulator.py"),
    ("phase1_review_20261015", "archive/reviews/phase1_review_20261015"),
    ("pdf_read.txt", "archive/reference_materials/pdf_read.txt"),
    ("__pycache__", "archive/generated_cache/root_pycache"),
    ("연구_목표와_진행계획.md", "archive/documents/2026-10-15_연구_목표와_진행계획_통합원본.md"),
]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def contained(relative):
    resolved = (ROOT / relative).resolve()
    if not resolved.is_relative_to(ROOT):
        raise ValueError(f"Outside workspace: {relative}")
    return resolved


def write(relative, text):
    path = contained(relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = text.strip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") == content:
            return
        raise FileExistsError(f"Do not overwrite an edited research document: {relative}")
    path.write_text(content, encoding="utf-8")


def prepare():
    MANAGEMENT.mkdir(parents=True, exist_ok=True)
    manifest_path = MANAGEMENT / "relocation_manifest.json"
    if manifest_path.exists():
        raise FileExistsError("An existing relocation manifest must not be overwritten")
    files, moves = [], []
    for old, new in MOVES:
        source, target = contained(old), contained(new)
        if not source.exists():
            raise FileNotFoundError(source)
        if target.exists():
            raise FileExistsError(target)
        if source.is_symlink():
            raise ValueError("Do not relocate symlinked research folders")
        paths = sorted(source.rglob("*")) if source.is_dir() else [source]
        for path in paths:
            if path.is_symlink():
                raise ValueError(f"Symlink in research tree: {path}")
            if path.is_file():
                suffix = path.relative_to(source) if source.is_dir() else Path()
                destination = target / suffix if source.is_dir() else target
                files.append({"old": path.relative_to(ROOT).as_posix(),
                              "new": destination.relative_to(ROOT).as_posix(),
                              "size": path.stat().st_size, "sha256": sha(path)})
        moves.append({"old": old, "new": new, "kind": "directory" if source.is_dir() else "file"})
    manifest = {"date": DATE, "created_utc": datetime.now(timezone.utc).isoformat(),
                "workspace": str(ROOT), "moves": moves, "files": files,
                "policy": "No deletion; code, protocols and raw results preserved byte for byte"}
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"moves": len(moves), "files": len(files), "bytes": sum(f["size"] for f in files)}))


def footer(experiment_date=DATE):
    return f"""

## 수행·작성·검토 기록

- 연구 방향·실험 요청자: 사용자(실명은 기록하지 않음).
- 실험 수행자: Codex(OpenAI AI 코딩 에이전트). 이 대화에서 로컬 Python 프로그램으로 구현·실행·집계를 수행함.
- 실험 수행일: {experiment_date}.
- 보고서 작성자: Codex. 이번 폴더 정리에서 기존 원자료와 실행 기록을 근거로 분리·작성함.
- 문서 정리일: {DATE}.
- 모델·버전: 실험 당시의 정확한 호출 모델 ID는 실행 기록에 없어 미기록. 추정 모델명을 기입하지 않음.
- 독립 검토 상태: 인간 또는 별도 주체의 독립 검토 기록 없음. 자동 테스트·원자료 점검은 독립 검토와 구분함.
"""


def report(identifier, title, kind, question, plan, evidence, body):
    return f"""# {identifier} — {title}

| 항목 | 내용 |
|---|---|
| 실험 ID | {identifier} |
| 문서 버전 | 1.0 — 기존 실행 기록의 문서 분리본 |
| 실험 구분 | {kind} |
| 문서 상태 | 실행 완료·작성 완료·독립 검토 미완료 |
| 수행일 / 정리일 | {DATE} / {DATE} |
| 연구 질문 | {question} |
| 계획·관련 문서 | {plan} |
| 원자료·근거 | {evidence} |

{body.strip()}
{footer()}
"""


def section(text, start, end=None):
    if start not in text:
        raise ValueError(f"Missing source heading: {start}")
    result = text.split(start, 1)[1]
    if end is not None:
        result = result.split(end, 1)[0]
    return result.strip()


def generate_documents():
    original = ROOT / "archive/documents/2026-10-15_연구_목표와_진행계획_통합원본.md"
    text = original.read_text(encoding="utf-8")
    wm_results = (ROOT / "studies/working_memory/research_results/RESULTS.md").read_text(encoding="utf-8")
    report_dir = "docs/reports"
    qdata = "../../studies/interaction/evosim_interaction/results_v01"
    wdata = "../../studies/working_memory"
    purpose = section(text, "## 1. 가장 큰 목표", "## 3. 현재까지 확인한 것과 확인하지 못한 것")
    write("docs/01_연구목적.md", f"""# 연구 목적과 질문 체계

문서 역할: 연구 전체의 목적과 용어를 관리한다. 실행 조건은 연구별 계획, 결과는 실험별 보고서에서 관리한다.

{purpose}

## 이번까지의 질문과 연결

- WM 계열: 정해진 지연 분류 과제의 기억·생애 학습·구조 의존성을 확인했다.
- Q1 계열: 행동이 다음 상태를 바꾸는 과제에서 현재 관측에 맞춰 이동·수집하는지를 확인했다.
- 두 연구는 환경·입력·지표가 다르다. 정확도 82%와 완료율 0.18%를 같은 척도의 성능 변화로 비교하지 않는다.

다음 문서: [연구 계획](02_연구계획.md), [현재 현황](03_연구현황.md).
""")
    write("docs/02_연구계획.md", """# 연구 계획과 점검 지점

현재 작업은 폴더와 보고 체계 정리다. 새 실험은 실행하지 않는다.

## 큰 흐름

| 단계 | 해야 할 일 | 통과 근거 |
|---|---|---|
| 질문 설정 | 큰 목표에서 이번에 답할 질문 하나를 고른다 | 관찰할 행동·반증 조건·해석 한계 |
| 환경 설계 | 관측·행동·전이·보상·종료를 정의한다 | 손으로 따라갈 수 있는 예시와 기준 정책 |
| 평가 계획 | 비교 대상·분할·예산·선발·판정을 고정한다 | 실행 전 프로토콜과 버전 |
| 구현 검증 | 환경·정책·난수·동결·재개를 검사한다 | 자동 검사와 실패·수정 기록 |
| 예비 실행 | 별도 시드로 오류와 실행 비용을 확인한다 | 본 테스트를 사용하지 않은 예비 기록 |
| 본 실험 | 고정 조건의 모든 시드를 실행한다 | 정책·원자료·실패 실행·코드 해시 |
| 해석 | 사전 질문에 답하고 한계와 다음 질문을 정한다 | 본 결과와 사후 진단을 구분한 보고서 |

## 현재까지 완료

- WM-001~004: 작업 기억 본 비교와 세 종류의 후속 검사.
- Q1-000: 구현·측정 검증.
- Q1-001: 이동·수집 본 실험.
- Q1-002~003: 표현·기억 제거 및 이동·수집 보완의 사후 진단.

## 다음 연구의 우선 질문 — 아직 미실행

“필요한 이동 방향 선택이 현재 진화 과정에서 왜 충분히 형성되지 않는가?”

환경을 더 복잡하게 하기 전에 이동 선택을 분리한 과제, 초기 구조 편향, 탐색 방식의 영향을 하나씩 비교한다. 아직 새 조건·성공 기준·예산을 확정하지 않았다. 기존 결과가 낮다는 이유로 같은 실험의 예산을 사후 연장하지 않는다.

## 점검 때 묻는 사항

실험 전에는 질문·바꾸는 요소·유지하는 요소·비교 기준·철회 조건을 확인한다. 실험 후에는 실제 행동·단순 규칙의 설명력·진화와 생애 학습의 구분·실패 범위·다음 질문을 점검한다.

작성 규칙은 [보고서 작성 규칙](rules/보고서_작성규칙.md)을 따른다. 진행 순서는 [연구일지](logs/연구일지.md)에 남긴다.
""")
    write("docs/03_연구현황.md", f"""# 현재 연구 현황

갱신일: {DATE}. 현재 실행 중인 실험은 없다. 이번 작업은 폴더·문서 체계 정리다.

| 계열 | 상태 | 현재 결론 | 공식 보고서 |
|---|---|---|---|
| 작업 기억 | 본 비교·후속 검사 완료 | 짧은 지연의 분류 수행은 높지만 생애 학습의 이득과 긴 지연 유지에는 한계 | [WM-001](reports/WM-001_작업기억_본실험.md)~[WM-004](reports/WM-004_시간상태_정밀검사.md) |
| 환경 상호작용 | 본 실험·감사·사후 진단 완료 | 현재 구조·보상·100회차 예산에서 안정적 이동·수집 정책 형성 근거를 얻지 못함 | [Q1-001](reports/Q1-001_이동수집_본실험.md) |
| 실패 원인 분리 | 탐색적 검사 완료 | 표현은 가능하며 이동 선택의 병목이 시사됨. 단일 원인은 미확정 | [Q1-002](reports/Q1-002_표현과기억_진단.md), [Q1-003](reports/Q1-003_행동병목_진단.md) |
| 후속 연구 | 미설계·미실행 | 기본 이동 반응의 형성을 우선 분리할 계획 | [연구 계획](02_연구계획.md) |

## Q1의 주 결과

| 조건 | 세 목표 완료율 |
|---|---:|
| 진화 | 0.1786% |
| 같은 예산의 무작위 개체 탐색 | 0.1521% |
| 무작위 행동 | 1.0516% |
| 선발한 고정 행동 순서 | 4.3651% |
| 좌표 비교 규칙 | 100% |

이 표는 Q1 조건의 요약이다. 진화의 우월성이나 일반적인 지능의 유무를 확정하지 않는다. 수치의 세부 근거와 불확실성은 실험 보고서에서 확인한다.

## 읽는 순서

처음 읽는 사람은 [목적](01_연구목적.md) → [계획](02_연구계획.md) → 이 현황 → [보고서 목록](reports/README.md) 순으로 읽는다. 실험을 추가하는 사람은 먼저 [작성 규칙](rules/보고서_작성규칙.md)과 [양식](templates/실험보고서_양식.md)을 확인한다.
""")
    environment = section(text, "### 2단계 설계안 — 일곱 칸 이동·수집 과제 v0.1", "### 3단계 — 성공 기준과 비교 대상 고정")
    evaluation = section(text, "### 3단계 평가 계획안 — Q1 프로토콜 v0.1", "### 4단계 — 구현과 소규모 예비 실행")
    implementation = section(text, "### 4단계 구현 순서와 완료 조건", "### 5단계 — 고정된 조건에서 본 실험")
    for filename, title, body in (
        ("Q1_환경설계.md", "Q1 환경 설계 v0.1", environment),
        ("Q1_평가계획.md", "Q1 평가 계획 v0.1", evaluation),
        ("Q1_구현과검증.md", "Q1 구현 순서와 검증 기록", implementation),
    ):
        write("docs/protocols/" + filename, f"""# {title}

문서 유형: 기존 통합 문서의 해당 절을 분리한 설계·구현 기록. 본문 속 예정·미완료 표현은 작성 당시의 상태다. 최종 실행 상태는 [연구 현황](../03_연구현황.md), 실행값은 [고정 프로토콜]({qdata}/protocol.json)을 따른다.

본문의 `evosim_interaction/` 경로는 이제 `studies/interaction/`을 작업 디렉터리로 삼은 상대 경로다. 후속 진단과 실행 결과는 별도 보고서로 관리한다.

{body}

기록 분리·정리: Codex, {DATE}. 통합 원문은 [보존본](../../archive/documents/2026-10-15_연구_목표와_진행계획_통합원본.md)에 있다.
""")
    qresults = section(text, "## 8. Q1 v0.1 결과와 결론")
    qmain = qresults.split("### 사후 진단: 본 판정과 분리", 1)[0]
    qconclusion = section(qresults, "### 이번에 내릴 수 있는 결론", "### 기록 위치와 실행상의 문제")
    write(report_dir + "/Q1-001_이동수집_본실험.md", report(
        "Q1-001", "현재 관측에 따른 이동·수집의 진화", "사전 조건을 고정한 본 실험",
        "개체가 현재 관측에 맞춰 행동하는가? 같은 평가 예산에서 진화가 독립 무작위 탐색보다 유리한가?",
        "[환경 설계](../protocols/Q1_환경설계.md) · [평가 계획](../protocols/Q1_평가계획.md)",
        f"[명세]({qdata}/protocol.json) · [주 집계]({qdata}/summary.json) · [감사]({qdata}/audit.json)",
        f"""## 연구 목적과 방법

일곱 칸의 현재 위치·목표 위치·남은 목표 수·남은 행동 수를 관측한다. 행동은 왼쪽·오른쪽·수집이며 24회 안에 목표 세 개를 수집한다. 생애 학습은 끄고 초기화·선택·교차·변이의 조합을 평가했다. 입력과 환경이 다른 WM 실험과 성공률을 직접 비교하지 않는다.

{qmain}

## 결론과 한계

{qconclusion}

두 탐색 방법의 차이는 +0.0265%포인트이고 95% 구간은 −0.2381~+0.2976%포인트다. 차이를 확인하지 못한 것이 동등성을 증명하지는 않는다. 사후 진단의 지원 성적을 이 본 실험의 성능으로 바꾸지 않는다.

## 오류·변경과 재현

환경 규칙·실행 예산·판정 경계는 본 결과를 본 뒤 바꾸지 않았다. 구현 전 검증·실행상의 문제는 [Q1-000](Q1-000_구현검증.md)에 기록했다. 실행 디렉터리는 `studies/interaction/`이며 명령은 [재현 안내](../../studies/interaction/README.md)를 따른다.

그림: [Q1 결과]({qdata}/q1_results.svg). 사후 진단은 [Q1-002](Q1-002_표현과기억_진단.md)와 [Q1-003](Q1-003_행동병목_진단.md)에서 다룬다.
"""))
    write(report_dir + "/Q1-002_표현과기억_진단.md", report(
        "Q1-002", "표현 가능성과 기억 상태 제거 진단", "본 결과 확인 후 설계한 탐색적 사후 진단",
        "과제 표현 자체가 불가능한가? 같은 후보의 시간 상태를 없애면 수행이 회복되는가?",
        "[본 실험](Q1-001_이동수집_본실험.md)",
        f"[진단 명세]({qdata}/exploratory_diagnostics/protocol.json) · [집계]({qdata}/exploratory_diagnostics/summary.json)",
        """## 목적·방법

Q1-001의 낮은 성능을 확인한 뒤 설계했다. 후보를 다시 고르거나 재훈련하지 않고 진화 주 후보 10개에 세 가지 제거를 적용했다. 이미 사용한 504개 테스트 목록·3반복을 재사용했으므로 독립적인 확인 실험이 아니다.

모듈 이전 상태 제거는 EMA의 계수와 latch 조건식을 유지한 채 이전 상태만 0으로 입력한다. 연산 함수를 tanh로 교체하는 기존 스위치와 구별한다.

## 결과

| 조작 | 평균 완료 성공률 |
|---|---:|
| 원래 후보 | 0.1786% |
| 그래프 순환 제거 | 0.2976% |
| 모듈 이전 상태 제거 | 0.0198% |
| 두 시간 상태 모두 제거 | 0.6944% |

모든 시간 상태 제거의 개선은 시드 208에서만 나타났고 다른 시드는 0%였다. 제거만으로 전반적인 성능 회복이 나타나지 않았다. 제거된 조건에서 처음부터 진화시킨 결과는 아직 없다.

별도로 허용 연산·가중치 범위 안에서 구성한 네 모듈 신경망은 전체 1,512조합을 최단 행동 수로 해결했다. 이는 표현 가능한 정책의 존재를 보인 것이다. 진화로 발견한 개체의 성적이 아니다.

## 검증·한계·결론

원래 주 후보 10개의 최종 평가 행동을 개별 실행기로 재계산해 묶음 평가와 일치함을 확인했다. 표현 불가능성만으로 실패를 설명하기 어렵고, 기억 상태 제거만으로 문제가 해결된다는 근거도 얻지 못했다. 초기화·보상·탐색의 어느 요소가 주원인인지는 미확정이다.

재현: `studies/interaction/`에서 `python diagnose_interaction.py evosim_interaction/results_v01`. 조건별 행동·보상·전후 상태는 같은 진단 폴더에 보존했다.
"""))
    write(report_dir + "/Q1-003_행동병목_진단.md", report(
        "Q1-003", "이동과 수집 행동의 병목 분리", "본 결과 확인 후 설계한 탐색적 사후 진단",
        "같은 후보의 이동과 수집 중 한 부분을 정답으로 보완하면 성능이 어떻게 달라지는가?",
        "[본 실험](Q1-001_이동수집_본실험.md) · [표현·기억 진단](Q1-002_표현과기억_진단.md)",
        f"[명세]({qdata}/action_bottleneck/protocol.json) · [집계]({qdata}/action_bottleneck/summary.json)",
        """## 방법과 비교

같은 진화 주 후보 10개, 이미 사용한 테스트 목록 504개·3반복을 사용했다. 원래 네트워크는 매 스텝 계산한다. 수집 보완은 목표 칸에서만 정답 수집을 강제하고, 이동 보완은 목표 밖에서만 정답 이동 방향을 강제한다. 다른 상황의 행동은 원래 정책이 선택한다. 재선발·재훈련은 없다.

## 결과

| 조건 | 평균 완료 성공률 |
|---|---:|
| 원래 정책 | 0.1786% |
| 수집 행동만 보완 | 8.1151% |
| 이동 방향만 보완 | 66.25% |

## 해석과 한계

이동 방향 선택이 현재 후보들의 큰 병목임을 시사한다. 지원 성적은 개체 자체의 능력이 아니다. 보완하면 이후 방문 상태와 내부 상태 경로도 달라지므로 원인을 완전히 분해한 인과 효과로 해석하지 않는다. 본 결과를 본 뒤 설계한 탐색이므로 미노출 확인 자료가 아니다.

## 결론·다음 질문·재현

더 복잡한 환경으로 넘어가기 전에 기본 이동 반응이 형성되는 과정을 분리하는 연구가 우선이다. 이는 제안이며 후속 본 실험은 아직 실행하지 않았다.

재현: `studies/interaction/`에서 `python probe_action_bottleneck.py evosim_interaction/results_v01`. 시드·조건별 행동·보상·전후 상태와 실행 전 명세를 같은 원자료 폴더에 보존했다.
"""))
    write(report_dir + "/Q1-000_구현검증.md", report(
        "Q1-000", "환경·네트워크·평가·재개 검증", "구현 및 측정 정합성 검사 — 능력 본 실험과 구분",
        "설계한 환경과 평가기가 의도대로 작동하고 같은 실행을 재현할 수 있는가?",
        "[구현·검증 기록](../protocols/Q1_구현과검증.md)",
        "[검사 코드](../../studies/interaction/evosim_interaction/tests/) · [예비 실행](../../studies/interaction/evosim_interaction/pilot_v01/)",
        """## 방법과 결과

- 환경·분할·기준 정책의 11개 검사와 네트워크·평가·재개 검사를 포함한 총 21개 테스트가 통과했다.
- 규칙 정책은 모든 1,512조합에서 성공했고 거리 합+수집 3회와 실제 행동 수가 일치했다. 최악의 최단 행동 수는 21회였다.
- 시작 2·목표 4→1→5 예시는 12회·보상 2.88이었다.
- 벽 충돌·실패 수집·마지막 행동의 세 번째 수집·종료 후 호출·잘못된 입력을 검사했다.
- 네 번째 관측 전달, 원래 매개변수 불변, 순환 상태 초기화, 모듈 ID 중복 방지, 교차 유효성을 확인했다.
- 모든 모듈 연산의 개별·묶음 출력과 에피소드 행동·보상, 동률 처리, 에피소드 순서 독립성을 비교했다.
- 중단 후 재개가 연속 실행과 같은 후보와 회차 기록을 만들었다.
- 예비 시드 901·902에서 두 방법을 각 20개체·10회차 실행했다. 훈련 집합만 사용했다.

## 발생 문제와 처리

첫 재개 테스트는 Windows 임시 폴더 권한 때문에 중단됐고, 허용받아 같은 검사를 재실행해 통과했다. 집계용 일회성 셸 명령의 인용 오류는 별도 스크립트로 처리했다. 그림 생성 시 Matplotlib 부재로 실패한 뒤 설치된 ReportLab으로 SVG를 생성했다. 오류를 성능이 나쁜 시드에서 제외하기 위한 이유로 사용하지 않았다.

## 결론과 한계

이 결과는 구현과 측정의 검사다. 모든 가능한 버그가 없다는 증명이나 진화 개체의 능력 증거가 아니다. 이후 본 실험에서는 원자료 집계와 최종 행동 재생도 별도로 검사했다.

재현 작업 디렉터리: `studies/interaction/`. 명령: `python -m unittest discover -s evosim_interaction/tests -v`. 재배치 후 검사 시 임시 파일은 허용된 작업 폴더 안에 생성하도록 검사 실행기에서 지정한다.
"""))
    wm_main = wm_results.split("## 탐색적 후속 실험: TD 부트스트랩과 장기 적응", 1)[0]
    wm_main = wm_main.replace("# 실행 결과", "## 방법과 결과", 1)
    wm_main = wm_main.replace("## 같은 기본 개체", "### 같은 기본 개체").replace("## 해석 범위", "### 해석 범위")
    write(report_dir + "/WM-001_작업기억_본실험.md", report(
        "WM-001", "작업 기억·생애 학습·초기 구조의 비교", "사전 실행 조건을 저장한 다조건 비교",
        "짧은 지연 기억 수행에 어떤 구조와 학습 경로가 기여하는가?",
        f"[실행 명세]({wdata}/research_results/protocol.json)",
        f"[원 집계]({wdata}/research_results/RESULTS.md) · [요약]({wdata}/research_results/summary.json)",
        f"""## 목적과 지표

과거 단서를 지연 구간에서 분류하는 과제다. 기억 정확도는 마스킹된 시점들의 분류 정확도이며 환경을 바꾸는 행동의 성공률이 아니다. 미래의 Q1 이동·수집 성적과 같은 척도로 비교하지 않는다.

{wm_main}

## 결론과 재현

기본 조건의 짧은 기억 정확도는 높고 구조 차단 시 낮아졌다. 두 생애 학습 경로를 끈 조건도 높은 성능을 보였다. 학습의 필요성·우월성·동등성을 일반화하지 않는다. 은닉 구조의 기억 기여를 순수하게 분리한 후속 검사는 WM-004에 있다.

작업 디렉터리 `studies/working_memory/`: `python run_research.py --workers 3`, `python analyze_research.py`. 기존 프로토콜과 소스를 보존했으며 이번 정리에서는 실험을 다시 실행하지 않았다.
"""))
    wm_followups = [
        ("WM-002", "WM-002_생애적응_후속실험.md", "추가 생애 적응과 TD 할인율", "## 탐색적 후속 실험: TD 부트스트랩과 장기 적응", "## 학습 지연을 넘어서 유지되는가?", "adaptation_followup", "run_adaptation_followup.py",
         "같은 선택 개체의 추가 학습량과 TD 할인율을 바꾸면 가중치 적응의 이득이 나타나는가?",
         "여기서 18→180은 학습 경험량 증가다. 기억 유지 시간 증가와 다르다. 18회 이득은 불확실했고 일부 180회 조건은 하락했다. 부트스트랩 제거만으로 문제가 해결된다는 근거는 없었다."),
        ("WM-003", "WM-003_기억지연_검사.md", "기억 지연 길이의 확장", "## 학습 지연을 넘어서 유지되는가?", "## 모듈의 이전 상태만 제거한 정밀 검사", "delay_probe", "probe_memory_delay.py",
         "짧은 지연으로 선발한 정책이 더 긴 지연에서도 정답을 유지하는가?",
         "지표는 마지막 마스킹 시점의 정확도다. 기본 조건은 3스텝 80.67%에서 32스텝 51.08%로 감소했다. 계산 스텝은 실제 초 단위 기억 시간과 같지 않다."),
        ("WM-004", "WM-004_시간상태_정밀검사.md", "시간 상태와 연산 함수 효과의 분리", "## 모듈의 이전 상태만 제거한 정밀 검사", None, "temporal_state_probe", "probe_temporal_state.py",
         "기존 기억 OFF 검사에서 상태 제거와 연산 함수 변경의 기여를 구분할 수 있는가?",
         "이전 상태만 제거한 하락은 기존 tanh 치환보다 작았다. 기존 OFF 하락 전체를 모듈 기억의 기여로 해석하면 안 된다. 같은 후보의 의존성을 측정했으며 기능 없이 재진화하는 실험과 다르다."),
    ]
    for identifier, filename, title, start, end, data_dir, script, question, conclusion in wm_followups:
        body = section(wm_results, start, end)
        write(report_dir + "/" + filename, report(
            identifier, title, "본 실험 이후의 탐색적 후속 검사", question,
            f"[본 실험](WM-001_작업기억_본실험.md) · [후속 명세]({wdata}/{data_dir}/protocol.json)",
            f"[후속 원자료]({wdata}/{data_dir}/) · [원 집계]({wdata}/research_results/RESULTS.md)",
            f"""## 방법·결과

{body}

## 결론·한계

{conclusion}

본 결과를 확인한 뒤 설계했으므로 탐색적 검사다. 선택된 정책의 조건부 결과이며 일반 개체 전체의 능력이나 일반적인 지능의 증거로 확대하지 않는다.

## 재현과 보존

`studies/working_memory/`에서 `python {script}`를 실행한다. 현재 원자료·명세·실행 스크립트를 보존했으며 폴더 정리를 위해 실험을 다시 실행하지 않았다.
"""))
    journal = section(text, "## 7. 진행 기록", "### 현재 다음 행동")
    write("docs/logs/연구일지.md", f"""# 연구일지

아래는 기존 통합 문서의 당시 진행 기록을 보존해 분리한 것이다. 현재 현황은 [현황 문서](../03_연구현황.md)가 기준이다.

{journal}

| {DATE} | 폴더·문서 정리 | 연구별 작업 폴더와 목적·계획·현황·프로토콜·공식 보고서·양식·규칙을 분리 | 이전 파일의 경로와 해시를 보존하고 재배치 후 재현·링크·수행자 표기를 점검 |
""")
    print("Generated purpose, plan, status, protocols, journal, and 8 signed reports")


def verify_preservation():
    manifest = json.loads((MANAGEMENT / "relocation_manifest.json").read_text(encoding="utf-8"))
    changed = []
    for item in manifest["files"]:
        target = contained(item["new"])
        if not target.exists() or sha(target) != item["sha256"]:
            changed.append(item["new"])
    result = {"original_file_count": len(manifest["files"]), "unchanged_file_count": len(manifest["files"]) - len(changed),
              "changed_or_missing": changed, "passed": not changed}
    (MANAGEMENT / "preservation_check.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if changed:
        raise AssertionError(result)
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("prepare", "documents", "verify"))
    args = parser.parse_args()
    {"prepare": prepare, "documents": generate_documents, "verify": verify_preservation}[args.phase]()
