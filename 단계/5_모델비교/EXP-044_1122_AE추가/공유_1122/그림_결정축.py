# EXP-044 J — 2차원 단면 대신 '결정축' 관점. 로지스틱 입력 3개(ET logit · 박자 z · AE z) 공간에서
#  x = 결정 방향 w·Z + b − 경계(정확, 모든 점에서 경계 = x 0 세로선), y = w에 수직인 나머지 2차원의 첫 주성분(PCA, 학습 OOF로 맞춤)
#  아래 = 파일별 기여 막대(계수 × 입력): 경계 근처 val 파일이 무엇 때문에 그쪽에 있는지. QE_FULL=1 (ET 300)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.evaluate import FULL
from src.exp2util import NS, Add83, rule83_flags
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
assert FULL, "QE_FULL=1 로 실행"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED, GRID = "#1d2430", "#5b6575", "#c9ced6"; C_N, C_A = "#2a78d6", "#e0622a"
ns = NS(); y = ns.y; nor = y == 0; nv = [n[-3:] for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv)
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)]); Mv = np.column_stack([ns.Mv, np.zeros(40), (sv | rv).astype(float)])
mJ = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK))).fit(M, y); mm = mJ.m_
ft = mJ.fires(M); rule_t = ft["E2"] | ft["R83"]             # 학습 쪽 규칙 켜짐 (전체 학습 모델, 참고용)
mm.ae_override_ = _ae_score(_fit_ae(LP_TR[nor]), segment_logpsd(ns.Xv)[1])
fv = mJ.fires(Mv); rule_v = fv["E2"] | fv["R83"]; fin_v = fv["main"] | rule_v
w, b, t = mm.c_.coef_[0], mm.c_.intercept_[0], mm.tc_
Zt = mm.Z_; Zv = np.column_stack([fv["a"], fv["z"], fv["ae"]])
xt, xv = Zt @ w + b - t, Zv @ w + b - t
u = w / np.linalg.norm(w); P = np.eye(3) - np.outer(u, u)                  # w에 수직인 평면으로 투영
Rt = Zt @ P; mu = Rt.mean(0); _, _, Vt = np.linalg.svd(Rt - mu, full_matrices=False); v1 = Vt[0]
yt, yv = (Rt - mu) @ v1, (Zv @ P - mu) @ v1
names_in = ["ET", "박자", "AE"]; load = " + ".join(f"{c:+.2f}·{n_}" for c, n_ in zip(v1 @ np.eye(3), names_in))
print("계수 w", np.round(w, 3), "절편", round(b, 3), "경계 t", round(t, 3), "| 세로축 =", load)
for f_ in ("027", "025", "018", "031", "024", "036", "001"):
    i = nv.index(f_); c = w * Zv[i]
    print(f"{f_}: 결정값 {xv[i]:+.2f} = ET {c[0]:+.2f} + 박자 {c[1]:+.2f} + AE {c[2]:+.2f} + (절편-경계) {b - t:+.2f} | 규칙 {'켜짐' if rule_v[i] else '-'} → {'고장' if fin_v[i] else '정상'}")

fig = plt.figure(figsize=(17, 12)); gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 1])
ax = fig.add_subplot(gs[0]); x0, x1 = min(xt.min(), xv.min()) - .5, max(xt.max(), xv.max()) + .5
ax.axvspan(x0, 0, color="#d6e6fa", zorder=0); ax.axvspan(0, x1, color="#fbe0d0", zorder=0); ax.axvline(0, color=INK, lw=1.3, ls="--")
ax.scatter(xt[nor], yt[nor], s=22, marker="s", color="#9db7d8", lw=0, label="train 정상 (OOF)")
ax.scatter(xt[~nor & ~rule_t], yt[~nor & ~rule_t], s=26, color="#f1b393", lw=0, label="train 고장 (OOF)")
ax.scatter(xt[~nor & rule_t], yt[~nor & rule_t], s=34, facecolor="none", edgecolor="#e08a5f", lw=1.2, label="train 고장 (규칙이 잡음)")
ax.scatter(xv, yv, marker="^", s=95, c=np.where(fin_v, C_A, C_N), edgecolor="k", lw=.8, zorder=4)
ax.scatter(xv[rule_v], yv[rule_v], marker="o", s=280, facecolor="none", edgecolor="k", lw=1.6, zorder=4)
for i, t_ in enumerate(nv):
    if abs(xv[i]) < 1.2 or rule_v[i]: ax.annotate(t_, (xv[i], yv[i]), xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold", zorder=5)
ax.set_xlim(x0, x1); ax.grid(alpha=.25)
ax.set_xlabel("결정값 = w·(ET, 박자, AE) + 절편 - 경계   (0 = 경계, 모든 파일에 정확)", color=MUTED)
ax.set_ylabel(f"경계와 나란한 방향 (PCA 1축: {load})", color=MUTED)
ax.legend(handles=[Line2D([], [], marker="s", ls="", color="#9db7d8", label="train 정상 (OOF)"), Line2D([], [], marker="o", ls="", color="#f1b393", label="train 고장 (OOF)"),
                   Line2D([], [], marker="o", ls="", mfc="none", mec="#e08a5f", label="train 고장 (규칙)"), Line2D([], [], marker="^", ls="", mfc=C_N, mec="k", ms=9, label="val → 정상"),
                   Line2D([], [], marker="^", ls="", mfc=C_A, mec="k", ms=9, label="val → 고장"), Line2D([], [], marker="o", ls="", mfc="none", mec="k", ms=13, label="val 규칙 켜짐(결정값 무관 고장)")],
          fontsize=9, loc="upper left", ncol=2, framealpha=.95)
ax.set_title("J 결정축 관점 — 가로축 하나로 판단이 정해진다 (경계 = 세로선)", fontsize=13, color=INK, loc="left")
# 기여 막대: 결정값 |x| < 1.2 또는 규칙 켜진 val
ax2 = fig.add_subplot(gs[1]); sel = [i for i in np.argsort(-xv) if abs(xv[i]) < 1.2 or rule_v[i]]
C = np.array([w * Zv[i] for i in sel]); xs = np.arange(len(sel)); off = b - t
cols = ["#7a3db8", "#1f9d55", "#c9a227"]
pos = np.zeros(len(sel)); neg = np.zeros(len(sel))
for j in range(3):
    v = C[:, j]; ax2.bar(xs, np.where(v > 0, v, 0), bottom=pos, color=cols[j], width=.62, label=f"{names_in[j]} 기여 (계수 {w[j]:.2f} × 입력)")
    ax2.bar(xs, np.where(v < 0, v, 0), bottom=neg, color=cols[j], width=.62); pos += np.where(v > 0, v, 0); neg += np.where(v < 0, v, 0)
ax2.bar(xs, [off] * len(sel), bottom=np.where(off < 0, neg, pos), color="#b8bdc6", width=.62, label=f"절편 - 경계 ({off:+.2f}, 모든 파일 같음)")
ax2.scatter(xs, xv[sel], marker="D", s=60, color=INK, zorder=5, label="결정값 (합계)")
ax2.axhline(0, color=INK, lw=1.2, ls="--"); ax2.set_xticks(xs)
ax2.set_xticklabels([f"{nv[i]}\n{'고장' if fin_v[i] else '정상'}{' (규칙)' if rule_v[i] else ''}" for i in sel], fontsize=9.5, color=INK)
for s in ("top", "right"): ax2.spines[s].set_visible(False)
ax2.spines["left"].set_color(GRID); ax2.spines["bottom"].set_color(GRID); ax2.tick_params(colors=MUTED, length=0); ax2.grid(axis="y", alpha=.3, lw=.6)
ax2.set_ylabel("결정값에 대한 기여", color=MUTED); ax2.legend(fontsize=9, ncol=5, loc="lower left", frameon=False)
ax2.set_title("경계 근처 val 파일의 결정값 분해 — 막대 합 = ◆, ◆ > 0 이면 고장", fontsize=13, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림4_결정축.png"), dpi=130); plt.close(fig)
print("saved")
