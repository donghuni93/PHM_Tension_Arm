# 품질공학 팀 프로젝트 — 9조 텐션암 이상감지

## 세션 시작
- `vault/00_NOW.md`와 거기 링크된 최신 `vault/50_세션/` 노트를 기준으로 이어서 한다.
- 상위 `C:\@Ajou_university\CLAUDE.md`의 코치 규칙을 따르되, 이 프로젝트에서는
  `docs/학습_계획.md` → `vault/00_NOW.md`, `docs/프로젝트_지도.md` → `vault/01_코드지도.md`로 읽는다.

## 과제
- 텐션암 진동신호로 정상(0)/고장(1) 판별. 평가 F2. 성능 70% + 발표 30%.
- 데이터: `data/TeamProject_train_변경/normal/` 120개, `data/TeamProject_train_변경/abnormal/` 30개.
  파일당 1열 50,000행 = 2초 × 25 kHz, 단위 g. val 40(11/19 공개), test 40(11/27 공개).
- 마감: 11/26 23:59 val 제출, 11/30 23:59 test·PPTX 제출, 발표 12/7·12/10.
- 제출 파일명: `submission_val_9조.csv`, `submission_test_9조.csv`, `team_project_9조.pptx`.

## 자동화 단계 (현재 단계는 `vault/00_NOW.md`에 기록)
| 단계 | 사용자 | Claude |
|---|---|---|
| L1 가이드 | 계획·코드·git 직접 | `/stuck` 요청 시에만 가이드. 노트·인수인계는 Claude |
| L2 제안 | 선택·붙여넣기 | 다음 단계 제안, 코드 초안, 읽기 전용 git |
| L3 위임 | 승인 | 승인 후 `src/` 구현·실행, 커밋은 승인 후 |
| L4 자동화 | 검토 | 실험 로그·인수인계 자동, 승인된 범위 실험 반복 |
- 단계는 사용자가 선언할 때만 올린다. Claude가 먼저 올리자고 하지 않는다.

## 어느 단계에서나 승인받을 것 (설명 후 AskUserQuestion)
- vault에서 `검증됨`이 아닌 개념·방법을 처음 쓸 때
- 새 도구·라이브러리 (`environment.yml` 변경)
- `src/` 함수 이름·인자·반환값 변경
- git commit/push, 노션 쓰기, 파일 삭제·이동

## 폴더
- `data/` 원본, 읽기 전용, git 제외 · `notebooks/` 탐색 · `src/` 공용 함수
- `vault/` Obsidian 볼트 = 모든 기록의 원본 (구조와 frontmatter는 `vault/_templates/` 참고)
- 노션 = vault에서 골라 다듬은 공유본. `/publish`로만 갱신

## 실행
- 환경: conda `qe-phm` (Python 3.12). conda init을 하지 않았으므로 전체 경로 사용:
  `& "C:\Users\dongh\anaconda3\Scripts\conda.exe" run -n qe-phm python ...`
- 노트북 커널: VS Code에서 `qe-phm` 선택

## 기록 규칙
- 틀린 노트는 지우지 않고 `status: 폐기` + 대체 노트 링크
- 실험 결과는 `vault/30_실험/EXP-nnn.md`, 커밋 해시 포함
- `vault/10_개념/`에는 개념만 (정의·식·물리 의미). 데이터 결과는 실험 노트·`decisions`·커밋에, 실험에서 나온 도메인 질문은 `vault/60_회의·질문/도메인_질문.md`
- git 커밋 메시지: `feat:` 코드 · `exp:` 실험 · `docs:` 문서 · `notes:` vault

## 도구
- 쓴다: 프로젝트 스킬(`/stuck` `/learn` `/paper` `/decide` `/handoff` `/publish`),
  `korean-report-style`, `pptx`, `dataviz`, 노션 MCP(`/publish` 전용)
- 쓰지 않는다: `wiki` 스킬(vault와 중복), OMC 자율 모드(autopilot·ralph·team·ultragoal),
  Slack·Gmail 등 무관한 커넥터
