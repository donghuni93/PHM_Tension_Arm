# %% EDA 뷰어 "경계" 화면용: EXP-047 평면(OOF seed 0 점 · 경계 · E2 켜짐 횟수) + val 40개 좌표·판정을 기존 json에 합친다 (11-22)
#   plane_models.json · plane_val.json에 "EXP-047" 키를 추가한다 (EXP-039 · 044는 건드리지 않음). 정의는 EXP-047 run.py의 대표 구성 S(이상치 포함 학습)와 같다
#   실행: conda run -n qe-phm python plane_047.py   (E2 켜짐을 seed 5개로 세느라 몇 분)
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else "."); HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.exp2util import NS, rule83_flags
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
from src.features2 import cached_hi_split, HI_SPLIT
warnings.filterwarnings("ignore")
ns = NS(); y, g, n = ns.y, ns.g, ns.n
SP, SPv = cached_hi_split("train", ns.X), cached_hi_split("val", ns.Xv)
FX = [c for c in ns.FIXED if c not in ["H10_ring_8-12.5k", "H15_share_8-12.5k"]] + HI_SPLIT
Hs = pd.concat([ns.T, SP], axis=1); Hsv = pd.concat([ns.Tv, SPv], axis=1)
M = np.hstack([Hs[FX].values, ns.T[["L1_S"]].values, ns.L]); Mv = np.hstack([Hsv[FX].values, ns.Tv[["L1_S"]].values, ns.Lv])
Fn = np.column_stack([M, np.arange(n, dtype=float)])
mk = lambda: TwoAE("lin", ("E2",), fixed=tuple(FX), nf=len(FX), df=ns.df, peak=tuple(ns.PEAK))
e2n = np.zeros(n, int); a0 = aez0 = z0 = None
for seed in range(5):
    a, aez, zz, e2 = np.empty(n), np.empty(n), np.empty(n), np.zeros(n, bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(g, y, g):
        f = mk().fit(Fn[tr], y[tr]).fires(Fn[te]); a[te], aez[te], zz[te], e2[te] = f["a"], f["ae"], f["z"], f["E2"]
    e2n += e2
    if seed == 0: a0, aez0, z0 = a.copy(), aez.copy(), zz.copy()
    print("047 seed", seed, flush=True)
mm = mk().fit(Fn, y); c = mm.c_.coef_[0]
P = json.load(open(os.path.join(HERE, "figs", "plane_models.json"), encoding="utf-8"))
P["EXP-047"] = dict(pts=np.round(np.c_[a0, aez0], 3).tolist(), w=[float(c[0]), float(c[2])], wz=float(c[1]), z=np.round(z0, 3).tolist(), b=float(mm.c_.intercept_[0] - mm.tc_),
                    xl="ET 점수 (logit, OOF)", yl="AE z (OOF)",
                    lim=[float(np.floor(a0.min() - .3)), float(np.ceil(a0.max() + .3)), float(np.floor(aez0.min() - .3)), float(np.ceil(aez0.max() + .3))],
                    thr=float(mm.tc_), e2n=e2n.tolist(),
                    label="ET logit(8–10 · 10–12.5k 4열) × AE z, 박자 z = 0 단면, OOF seed 0 · 결정값 0 = 결합 문턱 %.2f (E2·8.3k 규칙은 이 경계와 별개로 고장 판정)" % mm.tc_)
json.dump(P, open(os.path.join(HERE, "figs", "plane_models.json"), "w", encoding="utf-8"))
# val: train 150 전체 학습, val AE는 val 신호로 계산 (run.py와 같음)
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83v = (specv | resv)
_, LPv = segment_logpsd(ns.Xv); mm.ae_override_ = _ae_score(_fit_ae(LP_TR[y == 0]), LPv)
fv = mm.fires(np.column_stack([Mv, np.zeros(len(Mv))])); pv = fv["main"] | fv["E2"] | r83v
cat = np.where(r83v, "8.3k", np.where(fv["E2"], "E2", np.where(fv["main"], "결합", "—")))
V = json.load(open(os.path.join(HERE, "figs", "plane_val.json"), encoding="utf-8"))
V["EXP-047"] = dict(names=[str(x) for x in ns.nv], pts=np.round(np.c_[fv["a"], fv["ae"]], 3).tolist(), z=np.round(fv["z"], 3).tolist(), flag=pv.astype(int).tolist(), cat=cat.tolist())
json.dump(V, open(os.path.join(HERE, "figs", "plane_val.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("047 val 고장 판정", int(pv.sum()), flush=True)
