# val 40개의 경계 파일: J(EXP-044) · S(EXP-047)를 train 150 전체로 학습 → 결합 여유(점수 − 문턱), 규칙 켜짐, ET 단독 판정
# 경계 = 규칙으로 켜지지 않았고 |결합 여유| < 1.5 (두 모델 중 하나라도). 1.5 ≈ ET logit 1 만큼 (ET 계수 J 1.53 · S 1.3대)
# 실행: conda run -n qe-phm python "3_단계/5_모델실험/검토문서/아카이브/비교_044vs047_1122/val_경계.py"
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from src.exp import *
from src.exp2util import *
from src.models2 import TwoAE
from src.evaluate import best_f2_threshold
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
from src.features2 import cached_hi_split, HI_SPLIT
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
ns = NS(); y = ns.y; nv = [s[-3:] for s in ns.nv]; idx = np.arange(ns.n, dtype=float)
DROPC = ["H10_ring_8-12.5k", "H15_share_8-12.5k"]
SP, SPv = cached_hi_split("train", ns.X), cached_hi_split("val", ns.Xv)
Hs, Hsv = pd.concat([ns.T, SP], axis=1), pd.concat([ns.Tv, SPv], axis=1)
FX47 = [c for c in ns.FIXED if c not in DROPC] + HI_SPLIT
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res).astype(float); r83v = (specv | resv).astype(float)
raw_v = _ae_score(_fit_ae(LP_TR[y == 0]), segment_logpsd(ns.Xv)[1])
CFG = {"J": (ns.FIXED, ns.M, ns.Mv), "S": (FX47, np.hstack([Hs[FX47].values, ns.T[["L1_S"]].values, ns.L]), np.hstack([Hsv[FX47].values, ns.Tv[["L1_S"]].values, ns.Lv]))}
V = pd.DataFrame(dict(파일=nv))
for nm, (fx, M, Mv) in CFG.items():
    m = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(fx), nf=len(fx), df=ns.df, peak=tuple(ns.PEAK))).fit(np.column_stack([M, idx, r83]), y)
    m.m_.ae_override_ = raw_v
    f = m.fires(np.column_stack([Mv, np.zeros(len(Mv)), r83v])); t_et = float(best_f2_threshold(y, m.m_.Z_[:, 0]))
    V[f"{nm} 판정"] = np.where(f["main"] | f["E2"] | f["R83"], "고장", "정상")
    V[f"{nm} 여유"] = (f["score"] - m.m_.tc_).round(2)
    V[f"{nm} ET"] = f["a"].round(2); V[f"{nm} ET단독"] = np.where(f["a"] > t_et, "고장", "정상")
    if nm == "J": V["AE z"], V["박자 z"], V["규칙"] = f["ae"].round(2), f["z"].round(1), np.where(f["R83"], "8.3k", np.where(f["E2"], "E2", ""))
rule = V.규칙 != ""
V["경계"] = ~rule & ((V["J 여유"].abs() < 1.5) | (V["S 여유"].abs() < 1.5))
V["갈림"] = (V["J 판정"] != V["S 판정"]) | (V["J ET단독"] != V["J 판정"]) | (V["S ET단독"] != V["S 판정"])
B = V[V.경계 | (V["J 판정"] != V["S 판정"])].copy(); B["min여유"] = B[["J 여유", "S 여유"]].abs().min(1); B = B.sort_values("min여유")
cols = ["파일", "J 판정", "J 여유", "S 판정", "S 여유", "J ET", "J ET단독", "S ET단독", "AE z", "박자 z"]
txt = [f"# val 경계 파일 (J = EXP-044, S = EXP-047, train 150 전체 학습)\n",
       "경계 = 규칙(8.3k · E2)으로 켜지지 않았고 두 모델 중 하나라도 |결합 여유| < 1.5. 여유 = 결합 점수 − 문턱 (0 위 = 고장). 1.5는 ET logit 약 1만큼\n",
       f"J 고장 {int((V['J 판정'] == '고장').sum())}개 · S 고장 {int((V['S 판정'] == '고장').sum())}개 · 규칙으로 켜진 val: {' '.join(V.파일[rule]) or '없음'}\n",
       md_table(B[cols]), "\n전체 40개\n", md_table(V[cols + ["규칙"]])]
open(os.path.join(HERE, "val_경계.md"), "w", encoding="utf-8").write("\n".join(txt))
# 그림: J 여유 vs S 여유 (val), 경계 띠
fig, ax = plt.subplots(figsize=(8.5, 8))
ax.axhspan(-1.5, 1.5, color="#eeeeee", zorder=0); ax.axvspan(-1.5, 1.5, color="#eeeeee", zorder=0)
ax.axhline(0, color="#555", lw=1); ax.axvline(0, color="#555", lw=1)
lo, hi = min(V["J 여유"].min(), V["S 여유"].min()) - .5, max(V["J 여유"].max(), V["S 여유"].max()) + .5; ax.plot([lo, hi], [lo, hi], color="#aaa", ls="--", lw=1)
fj = V["J 판정"] == "고장"
ax.scatter(V["J 여유"][~fj & ~rule], V["S 여유"][~fj & ~rule], marker="^", s=60, color=NOR, label="J 정상 판정")
ax.scatter(V["J 여유"][fj & ~rule], V["S 여유"][fj & ~rule], marker="^", s=70, color=ABN, label="J 고장 판정")
ax.scatter(V["J 여유"][rule], V["S 여유"][rule], marker="^", s=90, facecolor="none", edgecolor="#000", lw=1.4, label="규칙으로 켜짐 (여유와 무관하게 고장)")
for i in np.flatnonzero(V.경계.values | (V["J 판정"] != V["S 판정"]).values | rule.values): ax.annotate(V.파일[i], (V["J 여유"][i], V["S 여유"][i]), xytext=(4, 4), textcoords="offset points", fontsize=8.5)
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_xlabel("J (EXP-044) 결합 여유 (0 위 = 고장)"); ax.set_ylabel("S (EXP-047) 결합 여유")
ax.set_title(f"val 40개: 두 모델의 결합 여유. 회색 띠 = |여유| < 1.5 (경계). J 고장 {int(fj.sum())} · S 고장 {int((V['S 판정'] == '고장').sum())}", fontsize=10)
ax.legend(fontsize=8.5, loc="upper left"); ax.grid(alpha=.3); fig.tight_layout(); fig.savefig(os.path.join(FIG, "val_경계.png"), dpi=110)
print("done")
