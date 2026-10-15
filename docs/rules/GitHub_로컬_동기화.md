# GitHub와 로컬 작업 동기화

작성일: 2026-10-16, 한국 시간. 적용 저장소: [paylace/gene-project](https://github.com/paylace/gene-project).

현재 연구 폴더 자체를 Git 저장소로 연결한다. 기존 원자료와 완료 실험 소스는 유지하고, 코드·문서·요약만 작업 브랜치와 PR을 통해 주고받는다. 파일 저장 즉시 양방향으로 덮어쓰는 방식은 사용하지 않는다. 별도 상주 동기화 작업이나 예약 작업은 설치하지 않는다.

## 작업 시작

PowerShell에서 저장소 루트를 열고 다음 명령을 실행한다. Git이 PATH에 없으면 이 PC의 Codex 포함 Git을 찾아 사용한다.

```powershell
.\tools\git_sync.ps1 status
.\tools\git_sync.ps1 fetch
```

fetch는 원격 이력만 갱신한다. 현재 브랜치의 원격 변경을 비교하고, 로컬 변경이 없을 때 다음 명령으로 fast-forward 반영한다.

```powershell
.\tools\git_sync.ps1 pull
```

커밋하지 않은 변경이 있거나 양쪽 이력이 갈라졌으면 멈춰 내용을 검토한다. 자동 stash, reset --hard, 강제 체크아웃, 강제 push로 해결하지 않는다. 다른 에이전트의 PR은 현재 브랜치에 자동 병합하지 않는다. PR 검토와 main 반영은 별도로 결정한다.

## 작업 완료

변경 목적에 맞는 검증을 실행하고 변경 파일을 확인한다. `git add .` 대신 필요한 경로를 지정해 stage한다. 큰 원자료·압축 파일·실행별 로그·인증 정보는 추가하지 않는다. 수행자 기록과 문서 규칙도 기존대로 적용한다.

Git이 일반 터미널의 PATH에 없는 이 PC에서는 다음처럼 실행 파일을 지정할 수 있다.

```powershell
$projectGit = Join-Path $env:USERPROFILE '.cache/codex-runtimes/codex-primary-runtime/dependencies/native/git/cmd/git.exe'
& $projectGit diff --stat
& $projectGit add -- '변경한/파일경로'
& $projectGit diff --cached
& $projectGit commit -m '변경 이유와 결과'
.\tools\git_sync.ps1 push
```

같은 브랜치에 push하면 기존 PR도 갱신된다. 원격에 다른 변경이 먼저 올라왔으면 push가 거부될 수 있다. 다시 fetch하고 차이를 검토한다. helper는 main/master 직접 업로드를 거부한다. 이는 helper의 동작 제한이며 GitHub 서버의 브랜치 보호 설정을 대신하지 않는다.

## 인증·큰 파일·검사 기록

GitHub 연결 앱의 로그인과 PC Git의 인증은 별개다. Git 명령의 인증은 Git Credential Manager를 통해 처리하고 토큰은 Windows 자격 증명 관리자에 둔다. 토큰을 연구 폴더·스크립트·문서·원격 URL에 넣지 않는다.

이 PC는 2026-10-16에 사용자 인증과 비대화형 push 권한 시험을 완료했다. 실제 업로드·재수신 검증의 commit은 로컬 연결 기록과 PR #2의 연결 완료 댓글에 남긴다.

기존 `.gitignore`의 원자료 제외 목록을 유지한다. 새 원자료 경로는 업로드 전에 제외 여부를 점검하고 필요하면 기존 Release 방식으로 배포한다. `docs/management/layout_check.json`과 `verification.json`은 실행 때 갱신될 수 있는 기존 추적 파일이므로, 변경을 숨기지 않고 커밋 포함 여부를 검토한다.

연결 전 파일 해시 목록과 검증 기록은 로컬 `dist/git_connection/`에 보존한다. 이 폴더는 업로드하지 않는다. 연결 상태를 확인할 때는 helper status와 원격 브랜치의 commit 일치를 확인한다. 최종 인증·push 확인 결과는 연결 기록을 기준으로 한다.

## 수행·작성·검토 기록

- 연구 방향·실험 요청자: 사용자. 현재 작업 폴더와 GitHub의 Git 연결을 요청함.
- 실험 수행자: 신규 실험 없음. Codex 에이전트가 Git 연결·보존 검사·동기화 helper를 구성함.
- 실험 수행일: 신규 실험 미실행. 연결 작업일 2026-10-16 한국 시간.
- 보고서 작성자: Codex 에이전트.
- 문서 작성일 또는 정리일: 2026-10-16 한국 시간.
- 모델·버전: 정확한 호출 모델 ID 미기록.
- 독립 검토 상태: 독립 검토 기록 없음. 연결·자동 검사는 과학적 독립 검토가 아님.
