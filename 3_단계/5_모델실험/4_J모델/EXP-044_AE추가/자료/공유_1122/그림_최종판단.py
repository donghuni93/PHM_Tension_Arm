# EXP-044 공유용 그림 (11-22): 배경 = J의 최종 판단 영역(정상 파랑 · 고장 주황, 박자 z = 0 단면), 점 색 = 최종 판단
# EXP-044 run.py와 같은 계산 (QE_FULL=1 로 실행해야 나무 300 · 같은 결과)
# 실행: QE_FULL=1 python "3_단계/5_모델실험/4_J모델/EXP-044_AE추가/자료/공유_1122/그림_최종판단.py"
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.model_selection import StratifiedGroupKFold
from src.evaluate import FULL
from src.exp2util import NS, Add83, rule83_flags, short
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
assert FULL, "QE_FULL=1 로 실행 (EXP-044와 같은 나무 300)"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
BG_N, BG_A = "#d6e6fa", "#fbe0d0"          # 배경: 정상 판단 영역 · 고장 판단 영역
C_N, C_A = "#2a78d6", "#e0622a"            # 점: 정상 판단 · 고장 판단
ns = NS(); y, g, names = ns.y, ns.g, ns.names; nor = y == 0
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); sv, rv = rule83_flags(ns.Xv); r83v = (sv | rv).astype(float)
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), r83])
mk = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
o = {k: np.empty(ns.n) for k in ("a", "z", "ae")}; o.update({k: np.zeros(ns.n, bool) for k in ("main", "E2", "R83")})
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    f = mk().fit(M[tr], y[tr]).fires(M[te])
    for k in o: o[k][te] = f[k]
fin = o["main"] | o["E2"] | o["R83"]
mJ = mk().fit(M, y); mm = mJ.m_
mm.ae_override_ = _ae_score(_fit_ae(LP_TR[y == 0]), segment_logpsd(ns.Xv)[1])
fv = mJ.fires(np.column_stack([ns.Mv, np.zeros(len(ns.Mv)), r83v])); finv = fv["main"] | fv["E2"] | fv["R83"]

def region(ax, x0, x1, y0, y1):
    xx, yy = np.meshgrid(np.linspace(x0, x1, 400), np.linspace(y0, y1, 400))
    D = mm._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape) >= mm.tc_
    ax.contourf(xx, yy, D.astype(int), levels=[-.5, .5, 1.5], cmap=ListedColormap([BG_N, BG_A]), zorder=0)
    ax.contour(xx, yy, D.astype(int), levels=[.5], colors="#333333", linewidths=1.2, linestyles="--", zorder=1)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
why = lambda f, i: "8.3k" if f["R83"][i] else ("E2" if f["E2"][i] else ("결합" if f["main"][i] else ""))

# 그림 1: 학습 150 (OOF seed 0)
a, ae = o["a"], o["ae"]
fig, ax = plt.subplots(figsize=(11, 8)); region(ax, a.min() - .3, a.max() + .3, ae.min() - .3, ae.max() + .3)
for m, mkr, s in ((nor, "s", 40), (~nor, "o", 70)):
    ax.scatter(a[m], ae[m], marker=mkr, s=s, c=np.where(fin[m], C_A, C_N), edgecolor="white", lw=.6, zorder=3)
fp = nor & fin; ax.scatter(a[fp], ae[fp], marker="s", s=150, facecolor="none", edgecolor="#c00000", lw=2.2, zorder=4)
inN = ~(mm._comb_score(a, o["z"], ae) >= mm.tc_)          # 각 점의 실제 박자 z로 본 전체 학습 경계 기준 정상 쪽
bz = ~nor & fin & (o["z"] > .5) & ~o["R83"] & ~o["E2"]
ax.scatter(a[bz], ae[bz], marker="o", s=170, facecolor="none", edgecolor="#1f9d55", lw=2, zorder=4)
for k, col in (("R83", "#000000"), ("E2", "#7a3db8")):
    m = ~nor & o[k]; ax.scatter(a[m], ae[m], marker="o", s=170, facecolor="none", edgecolor=col, lw=2, zorder=4)
for f_ in ("abn_002", "abn_017", "abn_022", "nor_116", "nor_065", "nor_059"):
    i = names.index(f_); ax.annotate(short(f_), (a[i], ae[i]), xytext=(8, 7), textcoords="offset points", fontsize=10, weight="bold", color="#222222", zorder=5)
ax.legend(handles=[Patch(color=BG_N, label="정상 판단 영역"), Patch(color=BG_A, label="고장 판단 영역"), Line2D([], [], color="#333333", ls="--", label="결합 경계 (박자 z = 0 단면)"),
                   Line2D([], [], marker="s", ls="", color=C_N, label="정상 파일 → 정상 판단"), Line2D([], [], marker="o", ls="", color=C_A, label="고장 파일 → 고장 판단"),
                   Line2D([], [], marker="s", ls="", mfc=C_A, mec="#c00000", mew=2, ms=9, label=f"정상 파일 → 고장 판단 (오경보 {int(fp.sum())})"),
                   Line2D([], [], marker="o", ls="", mfc=C_A, mec="#000000", mew=2, ms=10, label=f"규칙으로 고장 판단: 8.3k ({int((~nor & o['R83']).sum())})"),
                   Line2D([], [], marker="o", ls="", mfc=C_A, mec="#7a3db8", mew=2, ms=10, label=f"규칙으로 고장 판단: E2 ({int((~nor & o['E2'] & ~o['R83']).sum())})"),
                   Line2D([], [], marker="o", ls="", mfc=C_A, mec="#1f9d55", mew=2, ms=10, label=f"박자 z > 0.5로 결합이 고장 판단 ({int(bz.sum())})")],
          fontsize=9, loc="upper left", framealpha=.95)
ax.set_xlabel("ET 점수 (logit, OOF)"); ax.set_ylabel("AE z (OOF)"); ax.grid(alpha=.25, zorder=1)
ax.set_title(f"학습 150개 최종 판단 — 고장 {int(fin[~nor].sum())}/30 · 오경보 {int(fp.sum())}/120 (OOF seed 0)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림1_학습_최종판단.png"), dpi=150); plt.close(fig)

# 그림 2: val 40 (학습 150 전체 학습)
av, aev = fv["a"], fv["ae"]
x0, x1 = min(a.min(), av.min()) - .3, max(a.max(), av.max()) + .3; y0, y1 = min(ae.min(), aev.min()) - .3, max(ae.max(), aev.max()) + .3
fig, ax = plt.subplots(figsize=(11, 8)); region(ax, x0, x1, y0, y1)
ax.scatter(a, ae, marker=".", s=40, color="#8a8f99", alpha=.45, lw=0, zorder=2)
ax.scatter(av, aev, marker="^", s=95, c=np.where(finv, C_A, C_N), edgecolor="black", lw=.8, zorder=4)
BORDER = ["027", "025", "018", "021", "012", "036", "009", "001"]
OFF = {"009": (-30, -30), "012": (22, -22), "031": (-12, 10), "035": (8, 6), "024": (-4, -16), "018": (6, 6), "021": (6, 6)}
for i, n_ in enumerate(ns.nv):
    t = n_[-3:]
    if finv[i] or t in BORDER:
        lab = t + (" (8.3k 규칙)" if fv["R83"][i] else (" (E2 규칙)" if fv["E2"][i] else ""))
        ax.annotate(lab, (av[i], aev[i]), xytext=OFF.get(t, (7, 6)), textcoords="offset points", fontsize=9.5, weight="bold" if t in BORDER else "normal", color="#222222", zorder=5,
                    arrowprops=dict(arrowstyle="-", color="#444444", lw=.8) if t in ("009", "012") else None)
ax.legend(handles=[Patch(color=BG_N, label="정상 판단 영역"), Patch(color=BG_A, label="고장 판단 영역"), Line2D([], [], color="#333333", ls="--", label="결합 경계 (박자 z = 0 단면)"),
                   Line2D([], [], marker="^", ls="", mfc=C_N, mec="k", ms=10, label=f"val → 정상 판단 ({int((~finv).sum())})"),
                   Line2D([], [], marker="^", ls="", mfc=C_A, mec="k", ms=10, label=f"val → 고장 판단 ({int(finv.sum())})"),
                   Line2D([], [], marker=".", ls="", color="#8a8f99", ms=9, label="학습 150 (OOF, 참고)")], fontsize=9, loc="upper left", framealpha=.95)
ax.set_xlabel("ET 점수 (logit)"); ax.set_ylabel("AE z (학습 정상 기준)"); ax.grid(alpha=.25, zorder=1)
ax.set_title(f"val 40개 최종 판단 — 고장 {int(finv.sum())} · 정상 {int((~finv).sum())} (학습 150 전체로 학습)", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림2_val_최종판단.png"), dpi=150); plt.close(fig)
off = [ns.nv[i][-3:] for i in np.flatnonzero(finv & ~(mm._comb_score(av, np.zeros(40), aev) >= mm.tc_))]
print("val 고장:", " ".join(ns.nv[i][-3:] for i in np.flatnonzero(finv)), "| 평면 정상 영역에 있으나 고장(규칙·박자):", off)
print("train 오경보:", [names[i] for i in np.flatnonzero(fp)], "| 고장인데 평면 정상 영역:", [names[i] for i in np.flatnonzero(~nor & fin & ~(mm._comb_score(a, np.zeros(ns.n), ae) >= mm.tc_))])
