# 2-23 추가 (11-22): EXP-044 J의 val 판정으로 val을 '확실한 정상 / 확실한 고장 / 경계'로 나눠 평균 스펙트럼 비교
# 경계 = |여유| < 0.35 (결합 점수 − 경계, 학습 정상 IQR 단위) 또는 결합은 정상인데 규칙만 고장 (001). 실행: QE_FULL=1 (EXP-044와 같은 ET 300)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.evaluate import FULL
from src.exp2util import NS, Add83, rule83_flags
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
assert FULL, "QE_FULL=1 로 실행"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000; INK, MUTED = "#1d2430", "#5b6575"; CUT = 0.35
ns = NS(); y = ns.y; nor = y == 0; nv = [n[-3:] for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv)
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)]); Mv = np.column_stack([ns.Mv, np.zeros(40), (sv | rv).astype(float)])
mJ = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK))).fit(M, y); mm = mJ.m_
mm.ae_override_ = _ae_score(_fit_ae(LP_TR[nor]), segment_logpsd(ns.Xv)[1])
fv = mJ.fires(Mv); rule = fv["E2"] | fv["R83"]; fin = fv["main"] | rule
iqr = np.subtract(*np.percentile(mm.c_.decision_function(mm.Z_)[nor], [75, 25])); mar = (fv["score"] - mm.tc_) / iqr
border = (np.abs(mar) < CUT) | (rule & ~fv["main"])
vn, va = ~fin & ~border, fin & ~border
lines = [f"확실한 고장 {va.sum()}: {' '.join(np.array(nv)[va])}", f"확실한 정상 {vn.sum()}: {' '.join(np.array(nv)[vn])}",
         f"경계 (제외) {border.sum()}: " + " ".join(f"{nv[i]}({'고장' if fin[i] else '정상'} {mar[i]:+.2f}{', 규칙' if rule[i] and not fv['main'][i] else ''})" for i in np.flatnonzero(border))]
open(os.path.join(HERE, "val분할.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))

def psd(A):
    A = np.asarray(A, float); f, P = welch(A - A.mean(1, keepdims=True), fs=FS, nperseg=4096, axis=-1); return f, P
f, Pn = psd(ns.X[nor]); _, Pa = psd(ns.X[~nor]); _, Pv = psd(ns.Xv); k = f / 1000
win = max(1, int(round(100 / (f[1] - f[0])))); sm = lambda P: 10 * np.log10(uniform_filter1d(P.mean(0), win, mode="nearest"))
S = [(f"train 정상 ({nor.sum()})", sm(Pn), "#2a78d6", "-"), (f"train 고장 ({(~nor).sum()})", sm(Pa), "#eb6834", "-"),
     (f"val 확실한 정상 ({vn.sum()})", sm(Pv[vn]), "#2a78d6", "--"), (f"val 확실한 고장 ({va.sum()})", sm(Pv[va]), "#eb6834", "--")]
fig, ax = plt.subplots(figsize=(13, 6)); m = k <= 12.4
for nm, s, c, ls in S: ax.plot(k[m], s[m], color=c, lw=2.4 if ls == "-" else 2.0, ls=ls, dashes=(5, 2.5) if ls == "--" else (None, None))
ax.legend(handles=[Line2D([], [], color=c, lw=2.2, ls=ls, label=nm) for nm, s, c, ls in S], ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0),
          frameon=False, fontsize=10.5, handlelength=2.6, columnspacing=1.6, borderaxespad=0)
lo_, hi_ = min(s[m].min() for _, s, *_ in S), max(s[m][k[m] > .05].max() for _, s, *_ in S)
ax.set_xlim(0, 12.4); ax.set_ylim(np.floor(lo_ / 5) * 5 - 2, np.ceil(hi_ / 5) * 5 + 2); ax.set_xticks(range(0, 13))
ax.grid(axis="y", alpha=.3, lw=.6); ax.grid(axis="x", alpha=.12, lw=.6)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.spines["left"].set_color("#c9ced6"); ax.spines["bottom"].set_color("#c9ced6"); ax.tick_params(colors=MUTED, length=0)
ax.set_xlabel("주파수 (kHz)", color=MUTED); ax.set_ylabel("평균 PSD (dB re 1 g²/Hz)", color=MUTED)
ax.set_title("평균 파워 스펙트럼 — train vs val (EXP-044 판정, 경계 제외)", fontsize=14, color=INK, loc="left", pad=48)
ax.text(0, 1.085, f"val 경계 {border.sum()}개 제외(|여유| < {CUT} 또는 규칙만 고장) · 파일별 Welch PSD(4096점) 선형 평균, 100 Hz 이동평균, 필터 없음",
        transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.11); fig.savefig(os.path.join(HERE, "스펙트럼_val분할.png"), dpi=150); plt.close(fig)
print("saved")
