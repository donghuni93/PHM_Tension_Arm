# %% [markdown]
# EXP-007 확장 150 피처에서 다중공선성 제거
# - 가설: 없음 (평가·피처 위생). 확장 150은 같은 계산을 대역마다 반복해 공선성이 크다
#   (rank 139, |r|>0.9 쌍 93개, VIF 중앙값 약 5,000)
# - 바꾼 것: 기준 ext150 + ExtraTrees 에 공선성 필터(CollinearityFilter)만 앞에 붙임. 모델·설정 같음
#   필터는 Pipeline 안 → 중첩 CV의 각 학습 폴드 피처만 보고 fit (라벨 안 씀)
# - 주 구성(사전 지정): |r|>0.9 제거 + VIF ≤ 10. 나머지는 민감도 확인용
# - 예상: 트리 모델은 공선성에 예측이 크게 흔들리지 않으므로 F2 변화는 작을 수 있다.
#   대신 같은 점수를 훨씬 적은 피처로 내면 해석·이후 피처 작업이 쉬워진다

# %% 설정
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..")))
from src.exp import *
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt

ex = Experiment(__file__, "EXP-007", "확장 150 피처 다중공선성 제거", [],
                member="이동훈", model="ExtraTrees", changes="ext150 앞에 폴드 안 공선성 필터(|r|>0.9 + VIF≤10)")
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X)
g = np.arange(len(y))

# %% 그림 1: 놓친 파일 vs 정상 5–95% (매 실험 기본)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_focus",
            "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")

# %% 그림 2: 150개 전체에 필터를 적용했을 때 (설명용, 라벨 안 씀. 평가는 아래 폴드 안 fit)
cfgs = {"|r|>0.95": dict(r_max=0.95), "|r|>0.9": dict(r_max=0.9),
        "|r|>0.9 + VIF≤10": dict(r_max=0.9, vif_max=10)}
Ev = E.values.astype(float)
nonconst = Ev.std(0) > 0
lines = [f"원본 150 (상수 열 {int((~nonconst).sum())}개): rank {np.linalg.matrix_rank(Ev)}, "
         f"VIF 중앙값 {np.median(vif(Ev[:, nonconst])):.0f}, |r|>0.9 쌍 "
         f"{int((np.triu(np.abs(np.corrcoef(Ev[:, nonconst], rowvar=False)), 1) > 0.9).sum())}"]
kept = {}
for k, c in cfgs.items():
    cols = CollinearityFilter(**c).fit(Ev).cols_
    kept[k] = cols
    v = vif(Ev[:, cols])
    lines.append(f"{k}: {len(cols)}개 남음, VIF 중앙값 {np.median(v):.1f}, 최대 {v.max():.1f}")
lines.append("VIF≤10 후 남은 피처: " + " ".join(E.columns[kept["|r|>0.9 + VIF≤10"]]))
ex.note("전체 데이터 기준 필터 결과", "\n".join(lines))

fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
for a, (t, cols) in zip(ax, [("원본 (상수 열 제외)", np.flatnonzero(nonconst)),
                             ("|r|>0.9 + VIF≤10 후", kept["|r|>0.9 + VIF≤10"])]):
    im = a.imshow(np.abs(np.corrcoef(Ev[:, cols], rowvar=False)), vmin=0, vmax=1, cmap="viridis")
    a.set_title(f"{t}: {len(cols)}개, |상관계수|")
fig.colorbar(im, ax=ax, shrink=0.8)
ex.save_fig(fig, "corr_before_after", "피처 간 |상관계수|. 밝은 덩어리 = 같은 정보를 반복하는 피처 묶음")

# %% 중첩 CV (필터는 Pipeline 안 = 학습 폴드에서만 fit)
et = make_et()
ex.record("기준 ext150", nested_cv(Ev, y, g, et), names)
for k, c in cfgs.items():
    mk = (lambda c=c: lambda: make_pipeline(CollinearityFilter(**c), et()))()
    ex.record(k, nested_cv(Ev, y, g, mk), names, main=(k == "|r|>0.9 + VIF≤10"))
