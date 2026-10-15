# 원자료 다운로드와 복원

코드·문서·프로토콜·집계 요약은 Git에 포함한다. 실행별 기록·체크포인트·과거 검토본은 같은 비공개 저장소의 [Release](https://github.com/paylace/gene-project/releases/tag/evidence-2026-10-15) 첨부 파일로 분리한다. 원본 경로와 SHA-256은 catalog.json에 기록한다. 릴리스 게시 여부는 해당 페이지에서 확인한다.

저장소 접근 권한이 있는 계정으로 Release에서 ZIP을 내려받은 뒤 저장소 루트에서 실행한다.

```powershell
python tools/restore_evidence.py 다운로드경로/working_memory-evidence-2026-10-15.zip
python tools/restore_evidence.py 다운로드경로/interaction-evidence-2026-10-15.zip
python tools/restore_evidence.py 다운로드경로/archive-evidence-2026-10-15.zip
python -B -X utf8 tools/check_research_layout.py --tests --require-data
```

복원 도구는 압축 파일과 구성 파일의 해시를 검사하고, 기존 내용이 다른 파일은 덮어쓰지 않는다. 기본 문서 검사는 catalog에 등록된 원자료 미설치를 따로 표시한다. 원자료가 필요한 지표 재계산은 복원 후 실행한다. 단위 테스트만으로 실험 결론이 재현되었다고 주장하지 않는다.

로컬 원본은 이동·삭제하지 않았다. 재배포에서 제외한 항목은 캐시와 연구와 무관한 reference_materials다. 기존 보존 목록에는 이 파일들도 남아 있다.
