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

새 보관본은 `q1_004_v02_raw_2026-10-16.zip`이며 약580MB·20,685파일이다. [별도 목록](q1_004_v02.json)에 ZIP·내부 MANIFEST의 SHA-256을 기록했다. 기존 세 ZIP과 목록은 유지한다. 새 ZIP에는 동결 소스 보관본, 예비·본·확인 전체 회차·attempt·체크포인트·후보·시험·관찰·프로브·감사와 기술 재검사 기록이 들어 있다. Release 게시 상태는 별도 배포 기록에서 확인한다.

다운로드 후 저장소 루트에서 실행한다.

```powershell
python -B -X utf8 tools/restore_q1_004.py 다운로드경로/q1_004_v02_raw_2026-10-16.zip --verify-only
python -B -X utf8 tools/restore_q1_004.py 다운로드경로/q1_004_v02_raw_2026-10-16.zip
```

첫 명령은 SHA-256·구성 파일을 검증하고, 두 번째 명령은 없는 파일만 복원한다. 기존 파일 내용이 다르면 중단한다. 본 비교와 확인 집계는 Git에 함께 두어 큰 원자료 다운로드 없이도 [공식 보고서](../docs/reports/Q1-004_구조제약_재진화비교.md)를 읽을 수 있다. 자동 검증은 과학적 독립 검토와 구분한다.
