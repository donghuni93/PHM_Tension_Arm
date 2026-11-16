# %% [markdown]
# EXP-nnn 제목
# - 가설: H? (notes/물리가설.md) — 이 실험이 확인하려는 물리 변화와 예상 신호 징후
# - 바꾼 것: 이전 최고 대비 무엇을 바꾸는가 (하나만)
# 사용: 단계/5_모델비교/스크리닝/EXP-nnn_이름/ 으로 폴더째 복사 → VS Code에서 셀 단위 실행(Shift+Enter), 또는
#   conda run -n qe-phm python 단계/5_모델비교/스크리닝/EXP-nnn_이름/run.py
# 결과가 나오면 폴더를 5_모델비교/유의미/ 또는 기각/ 으로 옮긴다 (루트는 위로 올라가며 src/를 찾으므로 깊이와 무관)
# 기록 후에는 이 파일을 고치지 않는다. 바꾸려면 새 EXP 폴더.

# %% 설정
import os, sys
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *

# member·model·changes = 노션 카드 속성 (팀원·모델·변경 사항). F2 점수는 main=True 구성에서 자동
ex = Experiment(__file__, "EXP-nnn", "제목", ["H?"],
                member="이동훈", model="ExtraTrees", changes="이전 대비 바꾼 것 한 줄")
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X)
g = np.arange(len(y))

# %% 그림 1: 놓친 파일 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_focus",
            "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")

# %% 그림 2: t-SNE (매 실험 기본). 이 실험이 쓰는 입력 공간마다 패널 하나. score= 를 주면 이상 점수 색 줄 추가
fig, _ = viz.tsne({"ext150 피처": E.values}, y, names, mark=ex.focus)
ex.save_fig(fig, "tsne", "입력 공간에서 고장이 정상과 따로 모이는가, 놓친 파일은 어디에 있는가")

# %% 그림 3: 이 실험의 가설을 보여주는 그림 = 핵심 그림 (key=True)
# README에서 해석 첫 문단(결론 요약) 바로 아래에 자동으로 놓인다. 그 아래에 '그림 읽는 법' → 상세 설명을 손으로 쓴다
# 결합 모델 실험은 boundary_lr(x = 로지스틱 결정값, y = 새 축, 테두리 = 규칙/새 축이 바꾼 파일)이 기본
# fig = ...; ex.save_fig(fig, "이름", "무엇을 보는 그림인지", key=True)

# %% 중첩 CV
et = make_et()
ex.record("기준 ext150", nested_cv(E.values, y, g, et), names, main=True)
