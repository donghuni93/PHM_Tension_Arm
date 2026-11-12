# 품질공학 팀 프로젝트 — 9조 텐션암 이상감지

## 세션 시작
- `notes/00_NOW.md`(훅이 자동 주입)를 기준으로 이어서 한다.

## 과제
- 텐션암 진동신호로 정상(0)/고장(1) 판별. 평가 F2. 성능 70% + 발표 30%.
- 데이터: `data/TeamProject_train_변경/normal/` 120개, `data/TeamProject_train_변경/abnormal/` 30개.
  파일당 1열 50,000행 = 2초 × 25 kHz, 단위 g. val 40(11/19 공개), test 40(11/27 공개).
- 마감: 11/26 23:59 val 제출, 11/30 23:59 test·PPTX 제출, 발표 12/7·12/10.
- 제출 파일명: `submission_val_9조.csv`, `submission_test_9조.csv`, `team_project_9조.pptx`.
- 참고: `팀원 공유 정보/25 품공 소스/`의 25년 3조(최종 0.93)는 **다른 데이터**다. 우리 데이터에서는 0.538 (EXP-003).

## 작업 원칙
1. **이해하며 한 단계씩, 정식 순서로.** 문제·평가 → EDA → 전처리 → 피처 → 모델 비교 → 오류 분석 → 최종 고정·val. 단계마다 목적·방법·결과 해석을 먼저 설명하고 사용자가 확인한 뒤 다음으로 간다. 여러 단계를 한꺼번에 실행하거나, 점수만 보고 피처·모델을 쌓지 않는다. 미확정(모델·피처 수·전처리)은 같은 중첩 CV로 비교해 정한다.
2. **물리 도메인 기반 개발.** 피처·모델을 추가할 때는 `notes/물리가설.md`의 가설(H1…)에 연결하고, 실험 후 검증 열을 채운다.
3. **정직한 평가.** 점수는 중첩 CV(`src/evaluate.py`)로만 비교한다. 라벨을 본 뒤 대역·주파수를 손으로 고르지 않는다.
4. **놓친 파일을 본다.** 실험마다 자주 놓친 고장·오경보를 기록하고, 정상 5–95% 범위와 겹친 그림으로 원인을 찾는다.

## 승인 규칙
- 승인 없이 진행: `src/`·`experiments/`·`eda/`·`notes/` 수정(함수 시그니처 변경 포함, 호출부도 함께 고침), 실험 실행, 피처·모델 추가, 실험 로그 기록
- 승인 받을 것 (설명 후 AskUserQuestion): git commit/push, 새 라이브러리(`environment.yml` 변경), `data/` 변경, 제출 파일 확정, 노션·외부 공유

## 폴더
- `data/` 원본, 읽기 전용, git 제외
- `src/` 공용 함수: `data`·`features`·`evaluate`·`viz`(그림)·`exp`(캐시·모델·`Experiment` 기록)
- `experiments/EXP-nnn_이름/` 실험 1개 = `run.py`(`# %%` 셀) + `figs/` + `result.json` + `README.md`(자동 표·그림 + 손 메모)
  - 새 실험 = `_template/` 복사. 기록 후 `run.py`는 고치지 않는다(바꾸려면 새 EXP)
  - 매 실험 놓친 파일 그림(정상 5–95% 겹침)을 `figs/`에 남긴다. 커밋은 실험 단위로 코드·그림·결과 함께
- `eda/` 데이터 이해 그림 (`run.py 파일명…`으로 지정 파일 바꿔 재생성) · `notebooks/` 예전 탐색(01–05, 새 작업 안 함)
- `outputs/cache/` 데이터·피처 캐시 (git 제외). 피처 함수를 고치면 해당 `feat_*.csv` 삭제
- `notes/` 기록 (Obsidian 아님, 일반 md)
  - `00_NOW.md` 현재 상태·다음 할 일 (60줄 이내)
  - `실험로그.md` 모든 실험 한 표 + 짧은 메모
  - `물리가설.md` 가설 → 신호 징후 → 피처 → 검증
  - `결정.md` 유효한 결정과 이유
  - `코드지도.md` · `개념/` (모델 이해 단계용) · `_보관/` 예전 vault

## 실행
- 환경: conda `qe-phm` (Python 3.12). PowerShell에서 `conda run`으로 실행:
  `& "C:\Users\User\WorkSpace\Donghun\miniconda3\Scripts\conda.exe" run -n qe-phm python 스크립트.py`
- `python.exe`를 직접 부르면 DLL 로드 실패(0xC06D007F), Git Bash에서는 exit 127 → 반드시 `conda run`
- `conda run` 출력은 한글이 깨진다 → 결과는 UTF-8 파일로 저장해서 읽는다
- 노트북 커널: VS Code에서 `qe-phm` 선택

## 기록 규칙
- 실험 = `notes/실험로그.md` 표에 한 줄 (가설 id, F2 ± sd, 재현율, 정밀도, 자주 놓친 고장, 결론). 긴 메모만 아래 절에
- git 커밋 메시지: `feat:` 코드 · `exp:` 실험 · `docs:` 문서 · `notes:` 기록

## 도구
- 쓴다: `/handoff`, `pptx`, `dataviz`
- 쓰지 않는다: OMC 자율 모드(autopilot·ralph·team·ultragoal), 무관한 커넥터
