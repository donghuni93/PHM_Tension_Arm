# EXP-044 J의 val 40개 판정 — 파일별 여유(결합 점수 − 경계, 학습 정상 IQR 단위)를 크기순 막대로. QE_FULL=1 (ET 300)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from src.evaluate import FULL
from src.exp2util import NS, Add83, rule83_flags
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
assert FULL, "QE_FULL=1 로 실행"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED, GRID = "#1d2430", "#5b6575", "#c9ced6"; C_N, C_A = "#2a78d6", "#e0622a"; CUT = .35
ns = NS(); y = ns.y; nor = y == 0; nv = [n[-3:] for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv)
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)]); Mv = np.column_stack([ns.Mv, np.zeros(40), (sv | rv).astype(float)])
mJ = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK))).fit(M, y); mm = mJ.m_
mm.ae_override_ = _ae_score(_fit_ae(LP_TR[nor]), segment_logpsd(ns.Xv)[1])
fv = mJ.fires(Mv); rule = fv["E2"] | fv["R83"]; fin = fv["main"] | rule
iqr = np.subtract(*np.percentile(mm.c_.decision_function(mm.Z_)[nor], [75, 25])); mar = (fv["score"] - mm.tc_) / iqr
o = np.argsort(-mar); xs = np.arange(40)
fig, ax = plt.subplots(figsize=(16, 6.4))
ax.axhspan(-CUT, CUT, color="#eef0f3", zorder=0); ax.axhline(0, color=INK, lw=1.2, zorder=1)
ax.bar(xs, mar[o], width=.72, color=[C_A if fin[i] else C_N for i in o], zorder=2)
for x, i in zip(xs, o):
    if rule[i]: ax.text(x, max(mar[i], 0) + .05, "8.3k" if fv["R83"][i] else "E2", ha="center", fontsize=8.5, color=INK, weight="bold")
ax.set_xticks(xs); ax.set_xticklabels([nv[i] for i in o], rotation=90, fontsize=9.5, color=INK)
ax.set_xlim(-.7, 39.7); ax.set_ylabel("여유 (결합 점수 - 경계, 학습 정상 IQR 단위)", color=MUTED)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="y", alpha=.3, lw=.6)
ax.legend(handles=[Patch(color=C_A, label=f"고장 판단 ({int(fin.sum())})"), Patch(color=C_N, label=f"정상 판단 ({int((~fin).sum())})"),
                   Patch(color="#eef0f3", label=f"경계 띠 |여유| < {CUT} ({int((np.abs(mar) < CUT).sum())}개)")], frameon=False, fontsize=10, loc="upper right")
ax.set_title("val 40개 — J(EXP-044) 판단과 경계까지의 여유 (크기순)", fontsize=14, color=INK, loc="left", pad=26)
ax.text(0, 1.02, "+ = 고장 쪽 · - = 정상 쪽. 막대 위 8.3k · E2 = 규칙이 고장으로 판단(결합 점수와 무관). 학습 150 전체로 학습, 제출 파일 아님", transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림3_val_여유.png"), dpi=150); plt.close(fig)
print("saved")
