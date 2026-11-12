# %% [markdown]
# EXP-nnn 제목
# - 가설: H? (notes/물리가설.md) — 이 실험이 확인하려는 물리 변화와 예상 신호 징후
# - 바꾼 것: 이전 최고 대비 무엇을 바꾸는가 (하나만)
# 사용: 폴더째 복사 → experiments/EXP-nnn_이름/ → VS Code에서 셀 단위 실행(Shift+Enter), 또는
#   conda run -n qe-phm python experiments/EXP-nnn_이름/run.py
# 기록 후에는 이 파일을 고치지 않는다. 바꾸려면 새 EXP 폴더.

# %% 설정
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..")))
from src.exp import *

ex = Experiment(__file__, "EXP-nnn", "제목", ["H?"])
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X)
g = np.arange(len(y))

# %% 그림 1: 놓친 파일 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_focus",
            "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")

# %% 그림 2: 이 실험의 가설을 보여주는 그림
# fig = ...; ex.save_fig(fig, "이름", "무엇을 보는 그림인지")

# %% 중첩 CV
et = make_et()
ex.record("기준 ext150", nested_cv(E.values, y, g, et), names)
