# EXP-044 질문답용: (1) 최종 판정(J)을 그대로 보여 주는 크롭 평면 — 누가 각 고장을 잡았나  (2) 2차원 투영 3장 + 큐브 문턱선. 큐브_경계.py의 결과 json을 읽는다 (CV 다시 안 돌림)
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from src.exp2util import *
from src.models2 import TwoAE
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figs")
ns = NS(); y, names, g = ns.y, ns.names, ns.g; nor = y == 0
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); idx = np.arange(ns.n, dtype=float); F = np.column_stack([ns.M, idx, r83])
mkJ = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
O = {s: dict(a=np.empty(ns.n), z=np.empty(ns.n), ae=np.empty(ns.n), main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool), R83=np.zeros(ns.n, bool)) for s in range(5)}
for s in range(5):                                                       # 중첩 CV와 같은 seed 5개 (바깥 5겹 OOF)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g):
        f = mkJ().fit(F[tr], y[tr]).fires(F[te])
        for k in O[s]: O[s][k][te] = f[k]
fin = np.array([O[s]["main"] | O[s]["E2"] | O[s]["R83"] for s in range(5)])       # (seed, 파일) 최종 판정
n_seed_caught = fin[:, ~nor].sum(1); n_seed_fp = fin[:, nor].sum(1)
print("seed별 고장 잡은 수 (30개 중):", n_seed_caught.tolist(), "· seed별 오경보 수 (120개 중):", n_seed_fp.tolist())
o = O[0]; a, aez, zz = o["a"], o["ae"], o["z"]; caught = fin[0]
cat = np.where(o["R83"], "8.3k 규칙", np.where(o["E2"], "E2", np.where(o["main"], "결합(로지스틱)", "못 잡음")))
box = json.load(open(os.path.join(OUT, "큐브_결과.json"), encoding="utf-8"))["box"]; te_, ta_, tz_ = box
print("고장 잡은 부품 (seed 0):", {k: int(((cat == k) & ~nor).sum()) for k in ["결합(로지스틱)", "E2", "8.3k 규칙", "못 잡음"]})
# ---- (1) 크롭 평면: 최종 판정으로 색을 입힘
x0, x1, y0, y1 = a.min() - .3, a.max() + .3, aez.min() - .3, aez.max() + .3
mJ = mkJ().fit(F, y).m_
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = mJ._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(13, 8.2))
ax.contour(xx, yy, D, levels=[mJ.tc_], colors="k", linewidths=1.3, linestyles="--"); ax.text(x0 + .1, y1 - .35, "점선 = 결합(로지스틱) 경계 (박자 z = 0 단면). 최종 판정은 이것에 E2 · 8.3k · 박자 z가 OR로 더해진 것", fontsize=9)
ax.scatter(a[nor & ~caught], aez[nor & ~caught], color=NOR, s=40, marker="s", lw=0, alpha=.75, label="정상 → 정상 판정")
ax.scatter(a[nor & caught], aez[nor & caught], color=NOR, s=110, marker="s", edgecolor="#d62728", lw=2.2, label="정상 → 오경보 (최종)")
CC = {"결합(로지스틱)": "#e8590c", "E2": "#7a3db8", "8.3k 규칙": "#000000", "못 잡음": "#d62728"}
for k, c in CC.items():
    m = (cat == k) & ~nor
    if m.any(): ax.scatter(a[m], aez[m], s=100 if k == "결합(로지스틱)" else 160, color=ABN, edgecolor=c, lw=2.4 if k != "결합(로지스틱)" else .5, label=f"고장 → {k}이 잡음 ({int(m.sum())})")
big = ~nor & (zz > 0.5)
ax.scatter(a[big], aez[big], s=260, marker="o", facecolor="none", edgecolor="#1f9d55", lw=1.8, label=f"고장 · 박자 z > 0.5 (세로 방향으로 올라간 파일 {int(big.sum())})")
for f_ in ("abn_002", "abn_017", "abn_022", "abn_014", "nor_116", "nor_059", "nor_065", "nor_025", "nor_037"):
    i = names.index(f_); ax.annotate(f"{short(f_)} (z {zz[i]:.1f})", (a[i], aez[i]), xytext=(9, 9), textcoords="offset points", fontsize=9, color=ABN if y[i] else NOR, weight="bold")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xlabel("ET 점수 (logit, OOF seed 0)"); ax.set_ylabel("AE z (정상 기준, OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(0, .93))
ax.set_title(f"최종 판정(J) 기준 ET × AE 평면 — seed 0에서 고장 {int(caught[~nor].sum())}/30 잡음 (5 seed 평균 {n_seed_caught.mean():.1f}/30), 정상 오경보 {int(caught[nor].sum())}/120", fontsize=11); fig.tight_layout()
fig.savefig(os.path.join(OUT, "ET_AE_평면_최종판정.png"), dpi=120); plt.close(fig)
# ---- (2) 2차원 투영 3장 + 큐브 문턱
zc = np.clip(zz, -1, 10)
fig, axs = plt.subplots(1, 3, figsize=(22, 7.2))
for ax, (xs, ys, xl, yl) in zip(axs, ((a, aez, "ET 점수", "AE z"), (a, zc, "ET 점수", "박자 z"), (aez, zc, "AE z", "박자 z"))):
    ax.scatter(xs[nor], ys[nor], color=NOR, s=30, marker="s", lw=0, alpha=.65); ax.scatter(xs[~nor], ys[~nor], color=ABN, s=40, lw=0)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3)
axs[0].add_patch(plt.Rectangle((te_, ta_), 10, 10, fill=False, edgecolor="#2ca02c", lw=2.2, ls="--")); axs[0].set_xlim(x0, x1); axs[0].set_ylim(y0, y1)
axs[1].axvline(te_, color="#2ca02c", ls="--", lw=2); axs[2].axvline(ta_, color="#2ca02c", ls="--", lw=2)
if tz_ is not None: axs[1].axhline(tz_, color="#7a3db8", ls="--", lw=2); axs[2].axhline(tz_, color="#7a3db8", ls="--", lw=2)
for ax in axs[1:]:
    for f_ in ("abn_022", "abn_014", "abn_002", "abn_017"):
        i = names.index(f_); xs_ = a if ax is axs[1] else aez; ax.annotate(short(f_), (xs_[i], zc[i]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=ABN, weight="bold")
for f_ in ("abn_002", "abn_017", "abn_022"):
    i = names.index(f_); axs[0].annotate(short(f_), (a[i], aez[i]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=ABN, weight="bold")
fig.suptitle(f"세 축을 두 축씩 본 투영 (OOF seed 0). 초록 = ET·AE 문턱(ET ≥ {te_:.2f} 그리고 AE z ≥ {ta_:.2f}), 보라 = 박자 z ≥ {tz_ if tz_ is not None else float('inf'):.2f}. 큐브 규칙 = (초록 안) 또는 (보라 위)", x=.01, ha="left", fontsize=11); fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(os.path.join(OUT, "큐브_투영.png"), dpi=110); plt.close(fig)
print("박스 문턱", box)
