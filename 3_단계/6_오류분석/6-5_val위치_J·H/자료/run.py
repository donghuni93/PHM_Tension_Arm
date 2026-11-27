# %% [markdown]
# 6-5 (11-22) val에서 J(EXP-044)와 H(신효섭 RF, EXP-048)가 갈리는 10개 파일을 두 모델의 판단 평면에 나란히 놓는다
# - 왼쪽 = J의 평면(ET + 박자 z × AE z, 배경 = 결합 판단, 보라선 = ET 단독 경계) · 오른쪽 = H의 평면(RF 고장 확률 × 8.3k 돌출, 점선 = 임계값)
# - train 점 = OOF seed 0(회색), val 40개 = 삼각형. 갈리는 10개만 색 · 번호를 붙인다(J만 고장 3 = 빨강, H만 고장 7 = 초록)
# - 판단은 하지 않는다: 위치만 보여 준다. 모델 · 규칙은 EXP-044 · EXP-048 코드 그대로(고치지 않음)
# 실행: conda run -n qe-phm python "3_단계/6_오류분석/6-5_val위치_J·H/자료/run.py"
# %% 설정
import os, sys, pickle, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp import CACHE
from src.exp2util import *
from src.models2 import TwoAE
from src.evaluate import best_f2_threshold
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
from src import rf_hyoseop as HY
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
ns = NS(); names, y, g = ns.names, ns.y, ns.g; nor = y == 0; n = ns.n; k3 = [s[-3:] for s in ns.nv]

# %% J (EXP-044 코드 그대로): OOF seed 0, 전체 학습, val
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res).astype(float); r83v = (specv | resv).astype(float)
idx = np.arange(n, dtype=float); M_rule = np.column_stack([ns.M, idx, r83])
mkAE = lambda experts=(): (lambda: TwoAE("lin", experts, fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
o = dict(a=np.empty(n), z=np.empty(n), ae=np.empty(n))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    f = Add83(mkAE(("E2",))).fit(M_rule[tr], y[tr]).fires(M_rule[te])
    for k in o: o[k][te] = f[k]
mJ = Add83(mkAE(("E2",))).fit(M_rule, y); mm = mJ.m_
t_et = float(best_f2_threshold(y, mm.Z_[:, 0])); c_ = mm.c_.coef_[0]; SH = c_[1] / c_[0]
ap = o["a"] + SH * o["z"]
ae_tr = _fit_ae(LP_TR[y == 0]); _, LPv = segment_logpsd(ns.Xv); mm.ae_override_ = _ae_score(ae_tr, LPv)
Mv = np.column_stack([ns.Mv, np.zeros(len(ns.Mv)), r83v]); fv = mJ.fires(Mv); pvJ = mJ.decision_function(Mv) >= .5
fvp = fv["a"] + SH * fv["z"]

# %% H (EXP-048 코드 그대로): OOF seed 0 확률, 전체 학습 val 확률 · 임계값
def hy_tables(tag, X, nm):
    p = os.path.join(CACHE, f"feat_hyoseop_{tag}.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    out = HY.tables(X, nm); pickle.dump(out, open(p, "wb")); return out
XH, AC, r83s, P83, W83 = hy_tables("train", ns.X, names); XHv, ACv, r83sv, P83v, W83v = hy_tables("val", ns.Xv, ns.nv)
jobs = [(0, k, tr, te) for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g))]
cand = HY.candidates(XH)


def htask(s, k, tr, te):
    ytr = y[tr]; seed = 1000 * s + 10 * k; Xf = HY.prep(XH, AC, y, tr)
    return te, HY.bag(Xf.iloc[tr], ytr, Xf.iloc[te], seed + 9, cand)


with parallel_config(backend="loky", inner_max_num_threads=1):
    HR = Parallel(n_jobs=-1)(delayed(htask)(*j) for j in jobs)
pH = np.zeros(n)
for te, p in HR: pH[te] = p
X2 = pd.concat([XH, XHv], ignore_index=True); AC2 = np.vstack([AC, ACv]); r2 = np.r_[r83s, r83sv]
yall = np.r_[y, np.zeros(len(ns.nv), int)]; tr_, te_ = np.arange(n), np.arange(n, n + len(ns.nv))
pHv, thrH, predH = HY.fit_predict(X2, AC2, yall, r2, tr_, list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr_, y)), te_, 777)
predH = predH.astype(bool)

# %% 갈리는 파일
only_J = [k for k, a, b in zip(k3, pvJ, predH) if a and not b]; only_H = [k for k, a, b in zip(k3, pvJ, predH) if b and not a]
both = [k for k, a, b in zip(k3, pvJ, predH) if a and b]
print("J만", only_J, "H만", only_H, "둘 다", both)
col = {k: "#d62728" for k in only_J} | {k: "#1a9850" for k in only_H}
diff = [k for k in k3 if k in col]


def mark_val(ax, X, Yv, bold):
    for i, k in enumerate(k3):
        if k in col: continue
        ax.scatter(X[i], Yv[i], s=60, marker="^", color="#8a8a8a", edgecolor="white", lw=.6, zorder=3)
    for i, k in enumerate(k3):
        if k not in col: continue
        ax.scatter(X[i], Yv[i], s=170, marker="^", color=col[k], edgecolor="black", lw=1.3 if bold[i] else .4, zorder=5)
        ax.annotate(k, (X[i], Yv[i]), xytext=(7, 6), textcoords="offset points", fontsize=11, weight="bold", color=col[k], zorder=6)


fig, axs = plt.subplots(1, 2, figsize=(21, 8.8))
# 왼쪽: J 평면
ax = axs[0]
x0, x1, y0, y1 = min(ap.min(), fvp.min()) - .4, max(ap.max(), fvp.max()) + .4, min(fv["ae"].min(), o["ae"].min()) - .3, max(fv["ae"].max(), o["ae"].max()) + .3
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = mm._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
ax.contourf(xx, yy, D, levels=[-1e9, mm.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.85, zorder=0)
ax.axvline(t_et, color="#7a3db8", lw=2.4, label=f"ET 단독 경계 ({t_et:.2f})")
ax.scatter(ap[nor], o["ae"][nor], color="#c0c0c0", s=22, marker="s", lw=0, label="train 정상 (OOF)", zorder=2)
ax.scatter(ap[~nor], o["ae"][~nor], color="#f0b49a", s=26, lw=0, label="train 고장 (OOF)", zorder=2)
mark_val(ax, fvp, fv["ae"], pvJ)
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("ET 점수 (logit) + 박자 z 환산"); ax.set_ylabel("AE z (val은 val 신호로 계산, 학습 정상 기준)")
ax.set_title(f"내 모델 J (EXP-044) — val 고장 판정 {int(pvJ.sum())}개 · 굵은 테두리 = J가 고장으로 판정 · 배경 주황 = 결합 판단 고장", fontsize=11)
# 오른쪽: H 평면
ax = axs[1]
ymax = max(P83.max(), P83v.max()) * 1.08
ax.axhspan(HY.P83_MIN, ymax, color="#fbe3d6", alpha=.6, zorder=0, label=f"8.3k 돌출 ≥ {HY.P83_MIN} (폭 ≥ {HY.W83_MIN:g} Hz도 맞아야 규칙 켜짐)")
ax.axvspan(thrH, 1.0, color="#fbe3d6", alpha=.6, zorder=0)
ax.axvline(thrH, color="#1a9850", ls="--", lw=2, label=f"H 임계값 {thrH:.3f}")
ax.scatter(pH[nor], P83[nor], color="#c0c0c0", s=22, marker="s", lw=0, label="train 정상 (OOF)", zorder=2)
ax.scatter(pH[~nor], P83[~nor], color="#f0b49a", s=26, lw=0, label="train 고장 (OOF)", zorder=2)
mark_val(ax, pHv, P83v, predH)
ax.set_xlim(-.02, 1.02); ax.set_ylim(min(P83.min(), P83v.min()) - .05, ymax); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper left")
ax.set_xlabel("RF 배깅 고장 확률"); ax.set_ylabel("8.3k 돌출 (dec)")
ax.set_title(f"팀원 모델 H (신효섭 RF, EXP-048) — val 고장 판정 {int(predH.sum())}개 · 굵은 테두리 = H가 고장으로 판정 · 주황 = 확률 ≥ 임계값", fontsize=11)
fig.suptitle(f"val 40개 중 두 모델이 갈리는 10개: 빨강 = J만 고장({' '.join(only_J)}), 초록 = H만 고장({' '.join(only_H)}), 회색 ▲ = 나머지 val 30개", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(HERE, "figs", "val_위치_J_H.png"), dpi=140); plt.close(fig)

# %% 표 (md, 가로): 갈리는 10개의 두 평면 좌표
ix = {k: i for i, k in enumerate(k3)}
rows = [dict(파일=k, 갈림="J만 고장" if k in only_J else "H만 고장", J_ET=round(float(fv["a"][ix[k]]), 2), J_AE_z=round(float(fv["ae"][ix[k]]), 2), J_박자z=round(float(fv["z"][ix[k]]), 1), J판정="고장" if pvJ[ix[k]] else "정상",
             H_확률=round(float(pHv[ix[k]]), 3), H임계값대비=round(float(pHv[ix[k]] - thrH), 3), H_8_3k돌출=round(float(P83v[ix[k]]), 2), H판정="고장" if predH[ix[k]] else "정상") for k in diff]
tb = pd.DataFrame(rows)
open(os.path.join(HERE, "표_갈리는10개.md"), "w", encoding="utf-8").write(f"# 6-5 val에서 J와 H가 갈리는 10개 (H 임계값 {thrH:.3f})\n\n![val](figs/val_위치_J_H.png)\n\n{md_table(tb)}\n")
print(tb.to_string())
