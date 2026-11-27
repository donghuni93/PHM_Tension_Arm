# PHO-008 보조 분석: 점수가 아니라 분포 — 1단 전문가별 z 분포(정상 vs 고장)와 구성별 결합 여유 분포
# OOF 5-fold seed 0 (quick 모드 = ET 100, run.py와 같은 설정). 여유 = (결합 점수 − 그 fold 모델의 경계) / 그 fold 학습 정상 점수 IQR
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.exp2util import NS, Add83, rule83_flags
from src.models2 import TwoStack
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED, GRID = "#1d2430", "#5b6575", "#c9ced6"; C_N, C_A = "#2a78d6", "#e0622a"
ns = NS(); y, g, names = ns.y, ns.g, ns.names; nor = y == 0
spec, res = rule83_flags(ns.X); M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)])
mk = lambda extra: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), extra=extra))
EXN = ("ext", "shape", "mspc", "knn")
CFG = [("J", ()), ("J + ext", ("ext",)), ("J + shape", ("shape",)), ("J + mspc", ("mspc",)), ("J + knn", ("knn",)), ("J + 넷 다", EXN)]
FOC = {"nor_116": "N116", "nor_065": "N065", "nor_059": "N059", "abn_022": "A022", "abn_014": "A014", "abn_009": "A009"}
def oof(extra):
    n = ns.n; o = dict(mar=np.empty(n), rule=np.zeros(n, bool), a=np.empty(n), z=np.empty(n), ae=np.empty(n), ex=np.empty((n, len(extra))))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        m = mk(extra).fit(M[tr], y[tr]); f = m.fires(M[te]); mm = m.m_
        iqr = np.subtract(*np.percentile(mm.c_.decision_function(mm.Z_)[y[tr] == 0], [75, 25]))
        o["mar"][te] = (f["score"] - mm.tc_) / iqr; o["rule"][te] = f["E2"] | f["R83"]
        o["a"][te], o["z"][te], o["ae"][te] = f["a"], f["z"], f["ae"]
        if extra: o["ex"][te] = f["ex"]
    return o
R = {nm: oof(ex) for nm, ex in CFG}
rng = np.random.default_rng(0)
def strip(ax, v, title, ylab, zero=True, clip=None):
    vv = np.clip(v, *clip) if clip else v
    for j, (msk, c) in enumerate(((nor, C_N), (~nor, C_A))):
        ax.scatter(j + rng.uniform(-.22, .22, msk.sum()), vv[msk], s=18, color=c, alpha=.6, lw=0)
        q = np.percentile(vv[msk], [25, 50, 75]); ax.plot([j - .3, j + .3], [q[1]] * 2, color=INK, lw=2); ax.plot([j - .2, j + .2], [q[0]] * 2, color=INK, lw=.8); ax.plot([j - .2, j + .2], [q[2]] * 2, color=INK, lw=.8)
    for f_, lab in FOC.items():
        i = names.index(f_); j = 0 if nor[i] else 1
        ax.annotate(lab, (j, vv[i]), xytext=(18 if j == 0 else -40, 0), textcoords="offset points", fontsize=8, color=INK, va="center", arrowprops=dict(arrowstyle="-", color="#999", lw=.6))
    if zero: ax.axhline(0, color=MUTED, lw=.8, ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상", "고장"], color=INK); ax.set_xlim(-.6, 1.6); ax.set_title(title, fontsize=11.5, color=INK, loc="left"); ax.set_ylabel(ylab, color=MUTED, fontsize=9)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="y", alpha=.3, lw=.6)
    if clip: ax.set_ylim(clip[0] - .5, clip[1] + .5)

# 그림 A: 1단 전문가 7개의 z 분포 (J + 넷 다의 OOF)
o = R["J + 넷 다"]
P = [("ET (logit)", o["a"], None), ("박자 z", o["z"], (-2, 15)), ("AE z", o["ae"], None)] + [(f"{nm} z", o["ex"][:, j], (-5, 25)) for j, nm in enumerate(EXN)]
fig, axs = plt.subplots(1, 7, figsize=(22, 5.6))
for ax, (t, v, cl) in zip(axs, P): strip(ax, v, t, "", zero=True, clip=cl)
axs[0].set_ylabel("1단 점수 (OOF, 학습 정상 강건 z / ET는 logit)", color=MUTED, fontsize=9)
fig.suptitle("1단 전문가별 점수 분포 — 정상 120 vs 고장 30 (OOF seed 0). 굵은 선 = 중앙값, 가는 선 = 25 · 75%. 박자 · ext~knn은 화면 범위로 자름", fontsize=11, color=MUTED, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(os.path.join(HERE, "분포1_전문가별.png"), dpi=130); plt.close(fig)

# 그림 B: 구성별 결합 여유 분포 (규칙으로 잡힌 고장은 빈 원)
fig, axs = plt.subplots(1, 6, figsize=(22, 6), sharey=True)
lo, hi = -3, 3
for ax, (nm, _) in zip(axs, CFG):
    r = R[nm]; v = np.clip(r["mar"], lo, hi)
    ax.axhspan(0, hi + .5, color="#fbe0d0", alpha=.55, zorder=0); ax.axhspan(lo - .5, 0, color="#d6e6fa", alpha=.55, zorder=0)
    ax.scatter(rng.uniform(-.22, .22, nor.sum()), v[nor], s=18, color=C_N, alpha=.7, lw=0)
    fa = ~nor & ~r["rule"]; fr = ~nor & r["rule"]
    ax.scatter(1 + rng.uniform(-.22, .22, fa.sum()), v[fa], s=22, color=C_A, alpha=.8, lw=0)
    ax.scatter(1 + rng.uniform(-.22, .22, fr.sum()), v[fr], s=30, facecolor="none", edgecolor=C_A, lw=1.3)
    for f_, lab in FOC.items():
        i = names.index(f_); j = 0 if nor[i] else 1
        ax.annotate(lab, (j, v[i]), xytext=(18 if j == 0 else -40, 0), textcoords="offset points", fontsize=8, color=INK, va="center", arrowprops=dict(arrowstyle="-", color="#999", lw=.6))
    fpn = int((nor & ((r["mar"] >= 0) | r["rule"])).sum()); mis = int((~nor & (r["mar"] < 0) & ~r["rule"]).sum())
    ax.set_title(f"{nm}\n오경보 {fpn} · 놓침 {mis}", fontsize=11.5, color=INK, loc="left")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상", "고장"], color=INK); ax.set_xlim(-.6, 1.6); ax.set_ylim(lo - .3, hi + .3)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="y", alpha=.3, lw=.6)
axs[0].set_ylabel("여유 (결합 점수 - 경계, 학습 정상 IQR 단위, ±3에서 자름)", color=MUTED, fontsize=9)
fig.legend(handles=[Line2D([], [], marker="o", ls="", color=C_N, label="정상"), Line2D([], [], marker="o", ls="", color=C_A, label="고장 (결합으로 판단)"),
                    Line2D([], [], marker="o", ls="", mfc="none", mec=C_A, label="고장 (규칙 E2 · 8.3k가 잡음, 여유와 무관)")], ncol=3, frameon=False, loc="upper right", fontsize=9.5)
fig.suptitle("구성별 결합 여유 분포 (OOF seed 0, quick ET 100). 주황 배경 = 고장 판단, 파랑 배경 = 정상 판단", fontsize=11, color=MUTED, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .92)); fig.savefig(os.path.join(HERE, "분포2_구성별여유.png"), dpi=130); plt.close(fig)
for nm, _ in CFG:
    r = R[nm]; print(nm, "정상 여유 중앙 %.2f · 고장(결합) 중앙 %.2f · knn·ext 범위 확인" % (np.median(r["mar"][nor]), np.median(r["mar"][~nor & ~r["rule"]])))
print("knn z 범위:", np.percentile(R["J + knn"]["ex"][:, 0], [0, 50, 100]).round(1), "| ext z:", np.percentile(R["J + ext"]["ex"][:, 0], [0, 50, 100]).round(1))
