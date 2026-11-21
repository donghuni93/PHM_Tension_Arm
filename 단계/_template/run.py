# %% [markdown]
# EXP-nnn 제목
# - 가설: H? (단계/1_기준·기록/물리가설.md) — 이 실험이 확인하려는 물리 변화와 예상 신호 징후
# - 바꾼 것: 이전 최고 대비 무엇을 바꾸는가 (하나만)
# 사용: 단계/5_모델비교/EXP-nnn_이름/ 으로 폴더째 복사 → VS Code에서 셀 단위 실행(Shift+Enter), 또는
#   conda run -n qe-phm python 단계/5_모델비교/EXP-nnn_이름/run.py
# 결과가 나오면 폴더를 5_모델비교/base/ 또는 기각/ 으로 내린다 (루트는 위로 올라가며 src/를 찾으므로 깊이와 무관)
# 기록 후에는 이 파일을 고치지 않는다. 바꾸려면 새 EXP 폴더.
# 규칙 (11-22 사용자): 코드 · 출력은 이 파일 하나, json은 result.json 하나. psd_missed · hard_files 없음. 텍스트(README · 해석)는 최소로.

# %% 설정
import os, sys
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *

# member·model·changes = 노션 카드 속성. key_changes = README '주요 변경점'(기준 실험 대비 바뀐 것, 짧게). F2 점수는 main=True 구성에서 자동
ex = Experiment(__file__, "EXP-nnn", "제목", ["H?"],
                member="이동훈", model="ExtraTrees", changes="이전 대비 바꾼 것 한 줄",
                key_changes=["기준 실험 대비 바뀐 것 1", "바뀐 것 2"])
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X)
g = np.arange(len(y))

# %% 중첩 CV. 대표 구성만 main=True, 비교 · 절제 · 이전 값은 mid=True → README 맨 아래 '중간 단계 결과'로
et = make_et()
ex.record("기준 ext150", nested_cv(E.values, y, g, et), names, main=True)
# ex.record("비교 구성", nested_cv(...), names, mid=True)

# %% 핵심 그림 (key=True, 필수): 결합 모델 실험 = logistic_plane (ET × AE 평면, OOF seed 0, 데이터 범위로 크롭)
#   ① ET 단독 경계(세로선: 학습 OOF ET logit에서 F2 최대 문턱) ② 결합 경계 ③ 최종 판정으로 색(고장을 잡은 부품 구분)
#   예시 코드: 단계/5_모델비교/EXP-044_1122_AE추가/run.py
# ex.save_fig(fig, "logistic_plane", "한 줄 설명", key=True)

# %% 오류 분석 표 (md, 가로): 우리 판단(정답 라벨) · 모델 판단(seed 5번 중 고장/정상 횟수) · 결과 · 근거
# cnt = 파일별 고장 판정 횟수 (고장 = 5 − miss, 정상 = fp)
# ex.save_md("오류분석_표", error_md_table(names, y, cnt, files, extra={"ET 점수": {...}}))

# %% val 판정 (제출 파일 아님): val_판정.md(표) + figs/val_판정.png(ET 단독 경계 포함). csv 만들지 않음
# ex.save_fig(fig, "val_판정", "..."); ex.save_md("val_판정", md_table(V))

# %% t-SNE (라벨 색, README 그림 1장, 필요할 때만)
# ex.save_fig(fig, "tsne_라벨", "라벨로 색을 입힌 t-SNE")
