# EXP-051 분석 (11-22): 배깅 판정이 어떻게 나오는가 — 전체 학습(150) 배깅 모델의 bag 10개가 val 40개에 낸 여유와 평균
# run.py와 같은 Shrink · Bag 정의(복사, run.py는 import하면 전부 다시 돌아서 쓰지 않음). 모드 = full(ET 300)과 같은 설정으로 실행
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
os.environ["QE_FULL"] = "1"
import numpy as np, pandas as pd
from src.exp import md_table
from src.exp2util import NS, Add83, rule83_flags, NOR, ABN
from src.models2 import TwoStack
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore"); plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED = "#1d2430", "#5b6575"
ALPHA, NB, FRAC = 3.0, 10, 0.8
RING = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]
ns = NS(); y = ns.y; nor = y == 0; nv = [n[-3:] for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83, r83v = spec | res, sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
F0 = list(ns.FIXED)
M = np.column_stack([ns.T[F0].values, ns.T[["L1_S"]].values, ns.L, np.arange(ns.n, dtype=float), r83.astype(float)])
Mv = np.column_stack([ns.Tv[F0].values, ns.Tv[["L1_S"]].values, ns.Lv])

class Shrink:
    def __init__(self, mk, ring_idx, i1, alpha=ALPHA): self.mk, self.ring_idx, self.i1, self.alpha = mk, list(ring_idx), i1, alpha
    def _tf(self, F):
        F = np.array(F, float); n = np.rint(np.expm1(F[:, self.i1])); has = n > 0
        R = F[np.ix_(has, self.ring_idx)]; k = R * n[has, None]
        F[np.ix_(has, self.ring_idx)] = (k + self.alpha * self.p0_) / (n[has, None] + self.alpha); return F
    def fit(self, F, y):
        F = np.asarray(F, float); y = np.asarray(y); n = np.rint(np.expm1(F[:, self.i1])); m = (y == 0) & (n > 0)
        self.p0_ = (F[np.ix_(m, self.ring_idx)] * n[m, None]).sum(0) / n[m].sum()
        self.m_ = self.mk().fit(self._tf(F), y); return self

def shrink_mk(seed=0):
    base = lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(F0), nf=len(F0), df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True, seed=seed))
    return lambda: Shrink(base, [F0.index(c) for c in RING], F0.index("H1_nimp"))

def val_of(top):
    ts = top.m_.m_; fv = ts.fires_raw(top._tf(Mv), RAW)
    return fv["score"] - ts.tc_, fv["E2"], ts

# 단일 J-축소 (seed 0, 150 전체) · 배깅 10개 (같은 부분표본 · seed)
mJ, e2J, tsJ = val_of(shrink_mk(0)().fit(M, y))
sp = StratifiedShuffleSplit(NB, train_size=FRAC, random_state=0)
B = [val_of(shrink_mk(b)().fit(M[s], y[s])) for b, (s, _) in enumerate(sp.split(M, y))]
MG = np.array([b[0] for b in B]); E2 = np.array([b[1] for b in B]); mean = MG.mean(0)
finJ = (mJ >= 0) | e2J | r83v; finB = (mean >= 0) | (E2.mean(0) >= .5) | r83v
vote = (MG >= 0).sum(0)

# 표
D = pd.DataFrame({"val": nv, "J-축소 여유": mJ.round(2), "배깅 평균 여유": mean.round(2), "bag 여유 최소": MG.min(0).round(2), "bag 여유 최대": MG.max(0).round(2),
                  "결합 고장 bag 수": [f"{v}/10" for v in vote], "E2 bag 수": [f"{v}/10" for v in E2.sum(0)], "8.3k": np.where(r83v, "켜짐", ""),
                  "J-축소": np.where(finJ, "고장", "정상"), "배깅": np.where(finB, "고장", "정상")})
o = np.argsort(-mean); D = D.iloc[o]
coef = pd.DataFrame({"bag": range(NB), "ET": [round(b[2].c_.coef_[0][0], 2) for b in B], "박자": [round(b[2].c_.coef_[0][1], 2) for b in B],
                     "AE": [round(b[2].c_.coef_[0][2], 2) for b in B], "문턱": [round(b[2].tc_, 2) for b in B], "val 고장 수(이 bag 혼자)": ((MG >= 0) | E2 | r83v).sum(1)})
coef.loc[len(coef)] = ["J-축소(단일)", round(tsJ.c_.coef_[0][0], 2), round(tsJ.c_.coef_[0][1], 2), round(tsJ.c_.coef_[0][2], 2), round(tsJ.tc_, 2), int(finJ.sum())]
open(os.path.join(HERE, "bag별_판정.md"), "w", encoding="utf-8").write(
    "# EXP-051 배깅 판정 풀어 보기 (train 150 전체 학습 모델 → val 40)\n\n"
    f"val 고장: J-축소 {int(finJ.sum())} · 배깅 {int(finB.sum())}. 여유 = 결합 점수 − 그 모델의 문턱(0 이상 = 결합 고장). 배깅 판정 = 평균 여유 ≥ 0 OR E2 과반 OR 8.3k\n\n"
    "## bag별 결합기\n\n" + md_table(coef) + "\n\n## val 40 (배깅 평균 여유 큰 순)\n\n" + md_table(D) + "\n")

# 그림: val 파일마다 bag 10개 여유(회색 점) · 평균(굵은 표시) · J-축소(네모)
k = np.argsort(mean); xs = np.arange(40)
fig, ax = plt.subplots(figsize=(16, 5.6))
for j, i in enumerate(k):
    ax.plot([j, j], [MG[:, i].min(), MG[:, i].max()], color="#c8ccd3", lw=3, zorder=1, solid_capstyle="round")
    ax.scatter(np.full(NB, j), MG[:, i], s=10, color="#7d8590", zorder=2)
ax.scatter(xs, mean[k], marker="D", s=46, c=np.where(finB[k], ABN, NOR), edgecolor="k", lw=.6, zorder=4)
ax.scatter(xs + .28, mJ[k], marker="s", s=28, facecolor="none", edgecolor=INK, lw=1, zorder=3)
for j, i in enumerate(k):
    if r83v[i] or (E2.mean(0)[i] >= .5): ax.text(j, MG[:, i].max() + .25, "규칙", ha="center", fontsize=7, color=ABN)
    if finB[i] != finJ[i]: ax.annotate("바뀜", (j, mean[i]), xytext=(0, -18), textcoords="offset points", ha="center", fontsize=8, weight="bold", color=INK)
ax.axhline(0, color=INK, lw=1); ax.set_xticks(xs); ax.set_xticklabels([nv[i] for i in k], rotation=90, fontsize=8)
ax.set_ylabel("여유 (결합 점수 − 문턱)", color=MUTED); ax.grid(axis="y", alpha=.25)
ax.legend(handles=[Line2D([], [], marker="o", ls="", color="#7d8590", ms=4, label="bag 하나의 여유 (10개, 회색 막대 = 범위)"),
                   Line2D([], [], marker="D", ls="", mfc=ABN, mec="k", label="배깅 평균 → 최종 고장"), Line2D([], [], marker="D", ls="", mfc=NOR, mec="k", label="배깅 평균 → 최종 정상"),
                   Line2D([], [], marker="s", ls="", mfc="none", mec=INK, label="J-축소 단일 모델 여유")], fontsize=8.5, loc="upper left")
ax.set_title(f"val 40 — bag 10개가 낸 여유와 평균. 평균이 0 위면 결합 고장. '규칙' = E2 과반 또는 8.3k(여유와 무관하게 고장). val 고장 J-축소 {int(finJ.sum())} → 배깅 {int(finB.sum())}", loc="left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "bag별_판정.png"), dpi=150)
print("done", int(finJ.sum()), int(finB.sum()))
