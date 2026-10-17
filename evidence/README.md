# 원자료 다운로드와 복원

코드·문서·프로토콜·집계 요약은 Git에 포함한다. 실행별 기록·체크포인트·과거 검토본은 같은 저장소의 [기존 Release](https://github.com/paylace/gene-project/releases/tag/evidence-2026-10-15) 첨부 파일로 분리한다. 기존 원본 경로와 SHA-256은 catalog.json에 기록한다.

저장소 접근 권한이 있는 계정으로 Release에서 ZIP을 내려받은 뒤 저장소 루트에서 실행한다.

```powershell
python tools/restore_evidence.py 다운로드경로/working_memory-evidence-2026-10-15.zip
python tools/restore_evidence.py 다운로드경로/interaction-evidence-2026-10-15.zip
python tools/restore_evidence.py 다운로드경로/archive-evidence-2026-10-15.zip
python -B -X utf8 tools/check_research_layout.py --tests --require-data
```

복원 도구는 압축 파일과 구성 파일의 해시를 검사하고, 기존 내용이 다른 파일은 덮어쓰지 않는다. 기본 문서 검사는 catalog에 등록된 원자료 미설치를 따로 표시한다. 원자료가 필요한 지표 재계산은 복원 후 실행한다. 단위 테스트만으로 실험 결론이 재현되었다고 주장하지 않는다.

로컬 원본은 이동·삭제하지 않았다. 재배포에서 제외한 항목은 캐시와 연구와 무관한 reference_materials다. 기존 보존 목록에는 이 파일들도 남아 있다.

## Q1-004 v0.2 원자료

새 보관본은 [Q1-004 Release](https://github.com/paylace/gene-project/releases/tag/q1-004-v02-2026-10-16)의 `q1_004_v02_raw_2026-10-16.zip`이며 약580MB·20,685파일이다. [직접 다운로드](https://github.com/paylace/gene-project/releases/download/q1-004-v02-2026-10-16/q1_004_v02_raw_2026-10-16.zip)와 [별도 목록](q1_004_v02.json)을 제공한다. ZIP·내부 MANIFEST의 SHA-256을 기록했고 GitHub의 첨부 파일 해시도 로컬과 일치했다. 기존 세 ZIP과 목록은 유지한다. 새 ZIP에는 동결 소스 보관본, 예비·본·확인 전체 회차·attempt·체크포인트·후보·시험·관찰·프로브·감사와 기술 재검사 기록이 들어 있다. [게시 기록](../docs/management/q1_004_release.json)에서 대상 커밋과 첨부 파일 식별자를 확인한다.

다운로드 후 저장소 루트에서 실행한다.

```powershell
python -B -X utf8 tools/restore_q1_004.py 다운로드경로/q1_004_v02_raw_2026-10-16.zip --verify-only
python -B -X utf8 tools/restore_q1_004.py 다운로드경로/q1_004_v02_raw_2026-10-16.zip
```

첫 명령은 SHA-256·구성 파일을 검증하고, 두 번째 명령은 없는 파일만 복원한다. 기존 파일 내용이 다르면 중단한다. 본 비교와 확인 집계는 Git에 함께 두어 큰 원자료 다운로드 없이도 [공식 보고서](../docs/reports/Q1-004_구조제약_재진화비교.md)를 읽을 수 있다. 자동 검증은 과학적 독립 검토와 구분한다.

## Q1-005 v0.1 원자료

[Q1-005 Release](https://github.com/paylace/gene-project/releases/tag/q1-005-v01-2026-10-16)에 `q1005_v01_raw_2026-10-16.zip` 약432MB·21,678파일을 보관한다. [보관 목록](q1_005_v01.json)에 ZIP·MANIFEST 해시가 있다. 공식 예비·본100실행 전체 기록, 동결 소스 ZIP, 비용 표본, 역할 캐시·실제 작업 원장·프로브·행동 진단·감사·집계와 CI 실패 발췌를 포함한다. 본 진행 기준 미충족이므로 확인 원자료는 존재하지 않는다. 기존 Release는 유지한다.

```powershell
python -B -X utf8 tools/restore_q1005.py 다운로드경로/q1005_v01_raw_2026-10-16.zip --verify-only
python -B -X utf8 tools/restore_q1005.py 다운로드경로/q1005_v01_raw_2026-10-16.zip
```

검증 후 없는 파일만 복원하며 기존 내용이 다르면 중단한다. 큰 ZIP 없이도 [공식 보고서](../docs/reports/Q1-005_보상과번식_분리비교.md)와 Git의 본 요약을 읽을 수 있다. 실제 동결 런타임은 Python3.12.14·NumPy2.3.5·Windows다. Linux CI의 원본 혼합-purpose fixture 문제와 별도49검사는 보고서에 구분했다. 게시 후 대상 커밋·GitHub 자산 해시는 별도 게시 기록에 남긴다.

## Q1-009 v0.1 원자료

[Q1-009 Release](https://github.com/paylace/gene-project/releases/tag/q1-009-v01-2026-10-16)에 `q1009_v01_raw_2026-10-16.zip` 72,003,906바이트·2,055파일을 보관한다. [보관 목록](q1_009_v01.json)과 [게시 기록](../docs/management/q1_009_release.json)에 ZIP·MANIFEST 및 GitHub 첨부 SHA-256을 기록했다. 전체/파일별 해시·CRC 검사를 통과했고 GitHub 첨부 해시도 로컬과 일치한다. 실제 평가는2026-10-16, 게시 기록 정리는2026-10-17이다.

초기500원형과 네 변환·전체 행동/보상·초기 반응 출력·감사·동결 소스·훈련 예비·비용 표본을 포함한다. 학습·진화·후보 선발은 없다. 큰 ZIP 없이도 [공식 보고서](../docs/reports/Q1-009_초기반응과고착_요인분리.md)와 Git의 집계·원형별 CSV를 읽을 수 있다. 별도 Q1-006 수작업 템플릿 검사는 이 ZIP에 합치지 않았다.

```powershell
python -B -X utf8 tools/restore_q1009.py 다운로드경로/q1009_v01_raw_2026-10-16.zip --verify-only
python -B -X utf8 tools/restore_q1009.py 다운로드경로/q1009_v01_raw_2026-10-16.zip
```

검증 후 없는 파일만 복원하며 기존 내용이 다르면 중단한다. 자동 검사는 독립 검토와 구분한다.

## Q1-006 v0.2 원자료

[Q1-006 Release](https://github.com/paylace/gene-project/releases/tag/q1-006-v02-2026-10-17)에 `q1006_v02_raw_2026-10-17.zip` 493,054,016바이트·26,203파일을 보관한다. [목록](q1_006_v02.json), [파일별 검증](../docs/management/q1_006_archive_verification.json), [게시 기록](../docs/management/q1_006_release.json)에 전체 ZIP·MANIFEST·GitHub 첨부 해시를 기록했다. 로컬 전 파일과 GitHub 첨부 SHA-256이 일치했다. 기존 원자료 Release는 유지한다.

본40짝 시드·120작업의 전체 회차/attempt/체크포인트·후보·테스트·초기 상태 반응·감사·집계, v0.1 실패 예비와 v0.2 예비·별도 비용 표본·두 동결 소스 보관본·사후 로그 진단을 포함한다. 주 진행 기준 미충족으로 확인 실험 원자료는 없다. 큰 ZIP 없이도 [공식 결과](../docs/reports/Q1-006_고정구조와구조탐색_비교.md)와 Git의 시드 CSV·본 요약을 읽을 수 있다.

```powershell
python -B -X utf8 tools/restore_q1006.py 다운로드경로/q1006_v02_raw_2026-10-17.zip --verify-only
python -B -X utf8 tools/restore_q1006.py 다운로드경로/q1006_v02_raw_2026-10-17.zip
```

검증 후 없는 파일만 복원하며 기존 내용이 다르면 중단한다. 게시 후 만든 배포 기록·최종 CI 기록은 ZIP 바깥의 Git 기록으로 관리한다. 자동 검사를 독립 검토로 표현하지 않는다.
