# 임시 (11-22 밤): 경계 묶음 ③ 결합기 흔들림. J의 로지스틱(ET logit · 박자 z · AE z → 문턱 = 학습 F2 최대)이 학습 데이터에 따라 얼마나 도는가
# 그림5 = ET × AE 평면(박자 z = 0 단면)에 경계선 6개(바깥 5겹 seed 0 각 폴드 모델 + 전체 학습 모델) · train OOF 점 · val 점
#         + 전체 학습 모델의 결합 입력(안쪽 OOF)을 부트스트랩 300번 → w_ET × w_AE 산점, 문턱 분포
# 그림6 = 같은 파일 좌표에 경계 6개를 적용한 결합 여유의 범위(경계만 바꾸고 파일 좌표는 고정). 범위가 0을 가로지르면 "경계에 따라 판정이 바뀌는 파일"
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from src.exp import *
from src.exp2util import *
from src.models2 import TwoAE
from src.evaluate import best_f2_threshold
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
HERE = os.path.dirname(os.path.abspath(__file__))
ns = NS(); y, g = ns.y, ns.g; names = np.array(ns.names); nor = y == 0
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv)
r83, r83v = (spec | res), (specv | resv)
Mr = np.column_stack([ns.M, np.arange(ns.n, dtype=float), r83.astype(float)])
mk = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
out = []; P = lambda s="": (out.append(str(s)), print(s))

# %% 바깥 5겹 (seed 0): 폴드 모델의 경계 + OOF 좌표
A = np.zeros((ns.n, 3)); rule = np.zeros(ns.n, bool); BND = []
for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g)):
    m = mk().fit(Mr[tr], y[tr]); f = m.fires(Mr[te]); A[te] = np.column_stack([f["a"], f["z"], f["ae"]]); rule[te] = f["E2"] | f["R83"]
    c = m.m_.c_; BND.append(dict(name=f"폴드{k}", w=c.coef_[0].copy(), b=c.intercept_[0] - m.m_.tc_, tc=m.m_.tc_, b0=c.intercept_[0]))
mJ = mk().fit(Mr, y); mm = mJ.m_
BND.append(dict(name="전체(val용)", w=mm.c_.coef_[0].copy(), b=mm.c_.intercept_[0] - mm.tc_, tc=mm.tc_, b0=mm.c_.intercept_[0]))
mm.ae_override_ = _ae_score(_fit_ae(LP_TR[y == 0]), segment_logpsd(ns.Xv)[1])
fv = mJ.fires(np.column_stack([ns.Mv, np.zeros(len(ns.Mv)), r83v.astype(float)])); Av = np.column_stack([fv["a"], fv["z"], fv["ae"]]); rulev = fv["E2"] | r83v
for B in BND: P(f"{B['name']}: w_ET {B['w'][0]:.2f} · w_박자 {B['w'][1]:.2f} · w_AE {B['w'][2]:.2f} · 절편 {B['b0']:.2f} · 문턱 {B['tc']:.2f} → 경계 기울기 AE/ET {-B['w'][0] / B['w'][2]:.2f}")

# %% 부트스트랩: 전체 학습 모델의 결합 입력(안쪽 OOF Z_)으로 로지스틱 + 문턱을 300번 다시
Z, yz = mm.Z_, mm.y_; rs = np.random.RandomState(0); BS = []
for _ in range(300):
    ii = rs.choice(len(yz), len(yz), replace=True)
    if yz[ii].sum() < 5: continue
    c = LogisticRegression(C=1.0, class_weight="balanced").fit(Z[ii], yz[ii]); tc = float(best_f2_threshold(yz[ii], c.decision_function(Z[ii])))
    BS.append(list(c.coef_[0]) + [c.intercept_[0], tc])
BS = np.array(BS)
P(f"부트스트랩 {len(BS)}회: w_ET {np.percentile(BS[:, 0], 5):.2f}–{np.percentile(BS[:, 0], 95):.2f} · w_AE {np.percentile(BS[:, 2], 5):.2f}–{np.percentile(BS[:, 2], 95):.2f} · 문턱 {np.percentile(BS[:, 4], 5):.2f}–{np.percentile(BS[:, 4], 95):.2f} (5–95 %)")
P(f"  w_ET · w_AE 상관 {np.corrcoef(BS[:, 0], BS[:, 2])[0, 1]:+.2f} · 결합 입력 상관(ET logit, AE z): 정상 {np.corrcoef(Z[yz == 0, 0], Z[yz == 0, 2])[0, 1]:+.2f} · 고장 {np.corrcoef(Z[yz == 1, 0], Z[yz == 1, 2])[0, 1]:+.2f}")
# 문턱만의 몫: 계수를 전체 모델에 고정하고 문턱만 부트스트랩
TC = []
for _ in range(300):
    ii = rs.choice(len(yz), len(yz), replace=True); TC.append(float(best_f2_threshold(yz[ii], mm.c_.decision_function(Z[ii]))))
TC = np.array(TC); P(f"계수 고정, 문턱만 부트스트랩: 문턱 {np.percentile(TC, 5):.2f}–{np.percentile(TC, 95):.2f} (5–95 %)")

# %% 그림5
fig, axs = plt.subplots(1, 3, figsize=(20, 6.8), gridspec_kw=dict(width_ratios=[1.6, 1, 1]))
ax = axs[0]
ax.scatter(A[nor, 0], A[nor, 2], s=22, marker="s", color=NOR, alpha=.55, lw=0, label="train 정상 (OOF)")
ax.scatter(A[~nor & ~rule, 0], A[~nor & ~rule, 2], s=34, color=ABN, alpha=.8, lw=0, label="train 고장 (OOF)")
ax.scatter(A[~nor & rule, 0], A[~nor & rule, 2], s=34, color=ABN, alpha=.35, lw=0, label="train 고장 · 규칙이 잡음 (경계 무관)")
ax.scatter(Av[:, 0], Av[:, 2], s=60, marker="^", color=PUR, alpha=.75, edgecolor="white", lw=.5, label="val (전체 학습 모델)")
xs = np.linspace(min(A[:, 0].min(), Av[:, 0].min()) - .3, max(A[:, 0].max(), Av[:, 0].max()) + .3, 50)
cm = plt.get_cmap("Greys")
for k, B in enumerate(BND):
    yy = -(B["w"][0] * xs + B["b"]) / B["w"][2]; full = B["name"].startswith("전체")
    ax.plot(xs, yy, color="#7a3db8" if full else cm(.45 + .1 * k), lw=2.6 if full else 1.3, ls="-" if full else "--", label=B["name"] + f" (w_AE {B['w'][2]:.2f})")
for n in ["abn_009", "abn_028", "nor_025", "nor_037", "nor_116", "nor_059", "nor_065"]:
    i = int(np.flatnonzero(names == n)[0]); ax.annotate(short(n), A[i, [0, 2]], xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR, weight="bold")
for j, n in enumerate(ns.nv):
    if abs(np.dot(BND[-1]["w"], Av[j]) + BND[-1]["b"]) < 1.2: ax.annotate("V" + n[-3:], Av[j, [0, 2]], xytext=(4, -9), textcoords="offset points", fontsize=7.5, color=PUR)
ax.set_ylim(min(A[:, 2].min(), Av[:, 2].min()) - .5, max(A[:, 2].max(), Av[:, 2].max()) + .5); ax.set_xlim(xs[0], xs[-1])
ax.set_xlabel("ET logit"); ax.set_ylabel("AE z"); ax.grid(alpha=.3); ax.legend(fontsize=7.5, loc="lower left")
ax.set_title("경계 6개 (박자 z = 0 단면): 폴드마다 학습 데이터가 4/5씩 다를 뿐인데 경계가 돈다", fontsize=9.5)
ax = axs[1]
ax.scatter(BS[:, 0], BS[:, 2], s=10, color="#999", alpha=.6, label="부트스트랩 300회")
for k, B in enumerate(BND): ax.scatter(B["w"][0], B["w"][2], s=70, color="#7a3db8" if B["name"].startswith("전체") else "#333", marker="*" if B["name"].startswith("전체") else "o", zorder=3)
ax.set_xlabel("w_ET"); ax.set_ylabel("w_AE"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title(f"계수 흔들림: w_ET × w_AE (상관 {np.corrcoef(BS[:, 0], BS[:, 2])[0, 1]:+.2f}). 점 = 폴드 · ★ = 전체", fontsize=9.5)
ax = axs[2]
ax.hist(BS[:, 4], bins=30, color="#999", alpha=.7, label="계수 + 문턱 부트스트랩"); ax.hist(TC, bins=30, color="#e8590c", alpha=.5, label="계수 고정 · 문턱만")
for B in BND: ax.axvline(B["tc"], color="#7a3db8" if B["name"].startswith("전체") else "#333", lw=1.2)
ax.set_xlabel("결합 문턱 (로지스틱 결정값, F2 최대)"); ax.legend(fontsize=8); ax.grid(alpha=.3); ax.set_title("문턱 흔들림 (세로선 = 폴드 · 전체 모델)", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림5_결합경계_흔들림.png"), dpi=110); plt.close(fig)

# %% 그림6: 경계 6개를 같은 좌표에 → 결합 여유 범위
mg = lambda Ax: np.array([[np.dot(B["w"], a) + B["b"] for B in BND] for a in Ax])
Mt, Mv_ = mg(A), mg(Av)
flip_t = (Mt.min(1) < 0) & (Mt.max(1) >= 0) & ~rule; flip_v = (Mv_.min(1) < 0) & (Mv_.max(1) >= 0) & ~rulev
P(f"경계 6개에 따라 판정이 바뀌는 파일 (규칙 켜진 파일 제외): train {flip_t.sum()}개 ({' '.join(short(n) for n in names[flip_t])}) · val {flip_v.sum()}개 ({' '.join(n[-3:] for n in np.array(ns.nv)[flip_v])})")
P(f"val 결합 고장 판정 수 (규칙 제외), 경계별: " + " · ".join(f"{B['name']} {int(((Mv_[:, k] >= 0) & ~rulev).sum())}" for k, B in enumerate(BND)))
sel_t = np.flatnonzero(((np.abs(Mt).min(1) < 1.5) | flip_t) & ~rule); sel_v = np.flatnonzero(((np.abs(Mv_).min(1) < 1.5) | flip_v) & ~rulev)
lab = [(short(names[i]), Mt[i], ABN if y[i] else NOR, flip_t[i]) for i in sel_t] + [("V" + ns.nv[j][-3:], Mv_[j], PUR, flip_v[j]) for j in sel_v]
lab.sort(key=lambda r: r[1][-1])
fig, ax = plt.subplots(figsize=(11, .3 * len(lab) + 1.5))
for r, (n, v, c, fl) in enumerate(lab):
    ax.plot([v.min(), v.max()], [r, r], color=c, lw=5 if fl else 2.5, alpha=.85 if fl else .45, solid_capstyle="round")
    ax.scatter(v[:5], [r] * 5, s=12, color="#333", zorder=3); ax.scatter(v[-1], r, s=60, marker="*", color="#7a3db8", zorder=4)
ax.axvline(0, color="#000", lw=1.2); ax.set_yticks(range(len(lab))); ax.set_yticklabels([r[0] for r in lab], fontsize=8)
for t_, r_ in zip(ax.get_yticklabels(), lab): t_.set_color(r_[2]); t_.set_fontweight("bold" if r_[3] else "normal")
ax.set_xlabel("결합 여유 (+ = 결합이 고장 판정). 점 = 폴드 경계 5개 · ★ = 전체(val용) 경계. 파일 좌표는 고정, 경계만 바꿈")
ax.set_title(f"경계에 따라 판정이 바뀌는 파일 (굵은 막대 = 0을 가로지름): train {flip_t.sum()} · val {flip_v.sum()}. 규칙이 켜진 파일은 뺌", fontsize=9.5, loc="left")
ax.grid(axis="x", alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림6_파일별_여유범위.png"), dpi=110); plt.close(fig)
open(os.path.join(HERE, "결합기흔들림_log.txt"), "w", encoding="utf-8").write("\n".join(out))
