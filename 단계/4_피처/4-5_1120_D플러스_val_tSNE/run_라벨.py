# %% [markdown]
# 4-5 후반. D+(81열) 라벨 사용 t-SNE + val — 거리 = ExtraTrees 근접도 (train 150 라벨로 학습, val은 라벨 없이 변환만)
# train 점은 ET가 라벨을 본 점이라 분리가 과장된다(in-sample). val 점은 ET가 본 적 없어 위치와 ET 고장 확률이 정직하다.
# 실행: python "단계/4_피처/4_피처/4-5_1120_D플러스_val_tSNE/run_라벨.py"
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
from sklearn.ensemble import ExtraTreesClassifier
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.dplus import DPlus
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
X, y, names = load_train(); E = cached_table("ext", feature_table_ext, X)
dp = DPlus().fit(X, E); D = dp.D_
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val")); Dv, _ = dp.transform(Xv)
n = len(y); A = np.vstack([D, Dv])
MARK = [("nor", 116), ("nor", 59), ("nor", 55), ("nor", 106), ("nor", 25), ("abn", 22), ("abn", 14), ("abn", 17), ("abn", 2)]
mark = [i for i, nm in enumerate(names) if any(nm.startswith(p) and int(nm.split("_")[-1]) == k for p, k in MARK)]
et = ExtraTreesClassifier(500, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=-1).fit(D, y)
L = et.apply(A); Dm = 1 - np.mean([(L[:, t][:, None] == L[:, t][None, :]) for t in range(L.shape[1])], axis=0); np.fill_diagonal(Dm, 0)
best = None
for s in range(5):
    t = TSNE(2, perplexity=15, init="random", random_state=s, learning_rate="auto", metric="precomputed"); Em = t.fit_transform(Dm)
    if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, Em)      # KL 최소 seed (라벨 안 씀)
Em = best[1]; Et, Ev = Em[:n], Em[n:]
pv = et.predict_proba(Dv)[:, 1]                                                      # val의 ET 고장 확률 (val은 학습에 안 쓰임)
pd.DataFrame({"file": nv, "ET_고장확률": pv.round(3)}).to_csv(os.path.join(HERE, "val_ET확률.csv"), index=False, encoding="utf-8-sig")
far = pd.read_csv(os.path.join(HERE, "val_거리.csv")); q95 = 13.17
farset = set(far[far.val_to_train_5NN > q95].file)
print("ET 고장 확률 ≥ 0.5 val:", [(f[-3:], round(float(p), 2)) for f, p in zip(nv, pv) if p >= .5])
print("확률 구간별 개수: <0.2:", int((pv < .2).sum()), "| 0.2–0.5:", int(((pv >= .2) & (pv < .5)).sum()), "| ≥0.5:", int((pv >= .5).sum()))
print("거리 밖 6개(train 95% 초과)의 ET 확률:", [(f[-3:], round(float(p), 2)) for f, p in zip(nv, pv) if f in farset])
NOR, ABN, PUR = "#2a78d6", "#eb6834", "#7B1FA2"
fig, ax = plt.subplots(1, 2, figsize=(18, 8))
a = ax[0]; a.scatter(*Et[y == 0].T, s=24, color=NOR, alpha=.55, lw=0, label="train 정상"); a.scatter(*Et[y == 1].T, s=42, color=ABN, alpha=.9, lw=0, label="train 고장")
a.scatter(*Ev.T, s=70, color=PUR, edgecolor="white", lw=.7, zorder=5, label="val 40 (라벨 없음)")
for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Et[i], fontsize=8, xytext=(3, 3), textcoords="offset points")
a.set_title("D+ 81열 · ET 근접도 거리 (train 라벨로 학습)\ntrain 점은 라벨을 본 점이라 분리가 과장됨 · val은 정직", fontsize=11); a.legend(fontsize=9)
a = ax[1]; a.scatter(*Et.T, s=18, color="#c8cdd2", lw=0, label="train"); sc = a.scatter(*Ev.T, s=90, c=pv, cmap="plasma", vmin=0, vmax=1, edgecolor="white", lw=.7, zorder=5)
for i, f in enumerate(nv):
    if pv[i] >= .3: a.annotate(f[-3:], Ev[i], fontsize=8, weight="bold", xytext=(4, 4), textcoords="offset points")
for i, f in enumerate(nv):
    if f in farset: a.scatter(*Ev[i], s=260, facecolor="none", edgecolor="k", lw=1.3, zorder=6)
plt.colorbar(sc, ax=a, label="val의 ET 고장 확률 (0 = 정상, 1 = 고장)"); a.set_title("val을 ET 고장 확률로 색칠 (번호 = 확률 ≥ 0.3, 검은 원 = 81차원에서 train 95% 밖)", fontsize=11)
for a in ax: a.set_xticks([]); a.set_yticks([])
fig.savefig(os.path.join(FIG, "tsne_Dplus_라벨.png"), dpi=115, bbox_inches="tight"); plt.close(fig); print("완료")
