# %% [markdown]
# 4-5. D+(81열) 무라벨 t-SNE — 라벨 없이 train 150 + val 40이 어떻게 분포하는가
# 거리 = D+ 81열 유클리드(라벨 안 씀), 임베딩 seed는 KL 최소(라벨 안 씀). 라벨은 그림 색칠에만 쓴다. val은 임베딩에 함께 넣되 라벨이 없다.
# 추가 확인(무라벨): 81차원 공간에서 val이 train에서 얼마나 떨어져 있나 (train 안 leave-one-out 거리와 비교)
# 실행: python "3_단계/4_피처/4_피처/4-5_1120_D플러스_val_tSNE/run.py"
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.dplus import DPlus
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)

# %% 데이터 · D+ (변환·군집은 train 150개로만 fit, val은 변환만)
X, y, names = load_train(); E = cached_table("ext", feature_table_ext, X)
dp = DPlus().fit(X, E); D = dp.D_; assert D.shape == (150, 81)
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val")); Dv, _ = dp.transform(Xv)
n = len(y); A = np.vstack([D, Dv])
MARK = [("nor", 116), ("nor", 59), ("nor", 55), ("nor", 106), ("nor", 25), ("abn", 22), ("abn", 14), ("abn", 17), ("abn", 2)]
mark = [i for i, nm in enumerate(names) if any(nm.startswith(p) and int(nm.split("_")[-1]) == k for p, k in MARK)]

# %% 무라벨 t-SNE (perplexity 15 · 30, 각 seed 5개 중 KL 최소)
def tsne(A, perp):
    best = None
    for s in range(5):
        t = TSNE(2, perplexity=perp, init="pca", random_state=s, learning_rate="auto"); Em = t.fit_transform(A)
        if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, Em)
    return best[1]
EMB = {15: tsne(A, 15), 30: tsne(A, 30)}

# %% 81차원 거리: val이 train에서 떨어져 있나 (라벨 안 씀)
nb = NearestNeighbors(n_neighbors=6).fit(D)
dtr = nb.kneighbors(D)[0][:, 1:].mean(1)                     # train 파일 → 다른 train 5개 평균 거리 (자기 자신 제외)
dva = nb.kneighbors(Dv)[0][:, :5].mean(1)                    # val 파일 → train 5개 평균 거리
q = np.percentile(dtr, [50, 90, 95])
print(f"train 안 5-NN 거리: 중앙 {q[0]:.2f} · 90% {q[1]:.2f} · 95% {q[2]:.2f}")
print(f"val → train 5-NN 거리: 중앙 {np.median(dva):.2f} · 최대 {dva.max():.2f} | train 95% 초과 {int((dva > q[2]).sum())}/40 · 90% 초과 {int((dva > q[1]).sum())}/40")
print("val 중 거리 상위 5:", [(nv[i][-3:], round(float(dva[i]), 2)) for i in np.argsort(dva)[::-1][:5]])
pd.DataFrame({"file": nv, "val_to_train_5NN": dva.round(3)}).to_csv(os.path.join(HERE, "val_거리.csv"), index=False, encoding="utf-8-sig")

# %% 그림: 위 = 무라벨(회색 + 보라), 아래 = 같은 좌표에 라벨 색
GRAY, NOR, ABN, PUR = "#9aa0a6", "#2a78d6", "#eb6834", "#7B1FA2"
fig, ax = plt.subplots(2, 2, figsize=(16, 14))
for c, perp in enumerate((15, 30)):
    Em = EMB[perp]; Et, Ev = Em[:n], Em[n:]
    a = ax[0, c]; a.scatter(*Et.T, s=26, color=GRAY, alpha=.6, lw=0, label="train 150 (라벨 숨김)"); a.scatter(*Ev.T, s=60, color=PUR, edgecolor="white", lw=.6, label="val 40")
    a.set_title(f"무라벨 · D+ 81열 · perplexity {perp}\n(회색 = train, 보라 = val)", fontsize=11); a.legend(fontsize=9)
    a = ax[1, c]; a.scatter(*Et[y == 0].T, s=24, color=NOR, alpha=.55, lw=0, label="train 정상"); a.scatter(*Et[y == 1].T, s=42, color=ABN, alpha=.9, lw=0, label="train 고장")
    a.scatter(*Ev.T, s=60, color=PUR, edgecolor="white", lw=.6, label="val 40")
    for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Et[i], fontsize=8, xytext=(3, 3), textcoords="offset points")
    a.set_title(f"같은 좌표에 라벨 색 입힘 (perplexity {perp})", fontsize=11); a.legend(fontsize=9)
for a in ax.ravel(): a.set_xticks([]); a.set_yticks([])
fig.savefig(os.path.join(FIG, "tsne_Dplus_무라벨.png"), dpi=115, bbox_inches="tight"); plt.close(fig)
# val 거리 히스토그램
fig, a = plt.subplots(figsize=(7.5, 4.2)); a.hist(dtr, bins=25, color=GRAY, alpha=.7, label="train 파일 (다른 train 5개까지)"); a.hist(dva, bins=25, color=PUR, alpha=.6, label="val 파일 (train 5개까지)")
a.axvline(q[2], color="k", ls=":"); a.set_xlabel("81차원 5-NN 평균 거리"); a.set_title("val은 train에서 더 멀리 떨어져 있나 (라벨 안 씀)"); a.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "val_거리.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
print("완료")
