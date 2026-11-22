# %% [markdown]
# 16-2. D 피처(76열)로 train + val 함께 t-SNE. 세 방식: (a) 균등 (b) AUC 소프트 가중(train 라벨) (c) ExtraTrees 근접도(train 라벨로 학습, val은 변환만)
# 색: 정상 파랑 · 고장 주황 · val 보라(라벨 없음). val은 임베딩 위치와 train 이웃의 고장 비율만 본다. 실행: python "단계/4_피처/4-4_1119_휴대폰_피처·모델탐색/16_D피처모델/tsne_val.py"
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
src16 = open(os.path.join(HERE, "run.py"), encoding="utf-8").read().split("# %% 모델")[0].replace('os.path.dirname(__file__) if "__file__" in dir() else "."', repr(HERE))
g16 = {"__file__": os.path.join(HERE, "run.py")}; exec(compile(src16, "run16", "exec"), g16)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
D, Dv, y, names, nv, mark = g16["D"], g16["Dv"], g16["y"], g16["names"], g16["nv"], g16["mark"]
FIG = os.path.join(HERE, "figs")
def tsne(A, pre=False, perp=15):
    best = None
    for s in range(5):
        t = TSNE(2, perplexity=perp, init="random" if pre else "pca", random_state=s, learning_rate="auto", metric="precomputed" if pre else "euclidean"); E = t.fit_transform(A)
        if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, E)               # 라벨 없이 KL 최소 seed
    return best[1]
A = np.vstack([D, Dv]); n = len(y)
a = np.array([abs(roc_auc_score(y, D[:, j]) - 0.5) for j in range(D.shape[1])]); w = np.clip(a - 0.10, 0, None); w /= w.max() + 1e-12
et = ExtraTreesClassifier(500, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=-1).fit(D, y)
L = et.apply(A); P = np.mean([(L[:, t][:, None] == L[:, t][None, :]) for t in range(L.shape[1])], axis=0); Dm = 1 - P; np.fill_diagonal(Dm, 0)
EMB = {"(a) 균등 (76열)": tsne(A), f"(b) AUC 가중 ({int((w > 0).sum())}열)": tsne((A * w)[:, w > 0]), "(c) ET 근접도 (train 라벨로 학습)": tsne(Dm, pre=True)}
sv = pd.read_csv(os.path.join(HERE, "val_scores.csv")); flag16 = set(sv[sv.pred == 1].file)
scr = "/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad/val_근거표_025.csv"
flag25 = set(pd.read_csv(scr).query("pred == 1").file) if os.path.exists(scr) else set()
NOR, ABN, PUR, DARK = "#2a78d6", "#eb6834", "#7B1FA2", "#4A148C"
fig, ax = plt.subplots(1, 3, figsize=(24, 7.6)); res = {}
for a_, (tt, E) in zip(ax, EMB.items()):
    Et, Ev = E[:n], E[n:]; nb = NearestNeighbors(n_neighbors=5).fit(Et).kneighbors(Ev, return_distance=False); res[tt] = y[nb].mean(1)
    a_.scatter(*Et[y == 0].T, s=22, color=NOR, alpha=.55, lw=0, label="train 정상"); a_.scatter(*Et[y == 1].T, s=36, color=ABN, alpha=.9, lw=0, label="train 고장")
    both = [f in flag16 and f in flag25 for f in nv]; one = [(f in flag16) != (f in flag25) for f in nv]
    a_.scatter(*Ev.T, s=70, color=PUR, edgecolor="white", lw=.7, zorder=5, label="val 40 (라벨 없음)")
    for i, f in enumerate(nv):
        if f in flag16 or f in flag25: a_.annotate(f[-3:], Ev[i], fontsize=8, color=DARK, weight="bold", xytext=(4, 4), textcoords="offset points")
    for i in mark: a_.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Et[i], fontsize=7, color="k")
    a_.set_title(f"{tt}\n(보라 번호 = 025 모델 또는 D 스택이 고장 판정한 val)", fontsize=10); a_.set_xticks([]); a_.set_yticks([]); a_.legend(fontsize=8, loc="best")
fig.savefig(os.path.join(FIG, "tsne_val_D.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
T = pd.DataFrame({"file": nv, **{k: v for k, v in res.items()}}); T["025 고장판정"] = [int(f in flag25) for f in nv]; T["D스택 고장판정"] = [int(f in flag16) for f in nv]
T.round(2).to_csv(os.path.join(HERE, "val_tsne_neighbors.csv"), index=False, encoding="utf-8-sig")
for k, v in res.items(): print(k, "| val의 train 이웃 5개 중 고장 비율 분포(0/5…5/5):", np.bincount((v * 5).round().astype(int), minlength=6).tolist(), "| ≥3/5:", int((v >= 0.6).sum()))
c = list(res)[2]; print("ET 근접도 기준 이웃 고장 ≥ 3/5인 val:", [f[-3:] for f, v in zip(nv, res[c]) if v >= 0.6])
print("025 고장 판정:", sorted(f[-3:] for f in flag25), "| D스택:", sorted(f[-3:] for f in flag16))
print("ET 근접도 ≥3/5 ∩ 025:", sorted(f[-3:] for f, v in zip(nv, res[c]) if v >= 0.6 and f in flag25), "| ∩ D스택:", sorted(f[-3:] for f, v in zip(nv, res[c]) if v >= 0.6 and f in flag16))
