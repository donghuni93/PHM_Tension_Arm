# %% EDA 뷰어 "경계" 화면용: EXP-050 J-축소 평면(OOF seed 0 점 · 경계 · E2 켜짐 횟수) + val 40개 좌표·판정을 기존 json에 합친다 (11-22)
#   plane_models.json · plane_val.json에 "EXP-050" 키를 추가한다 (다른 키는 건드리지 않음). 정의는 EXP-050 run.py의 J-축소(울림 비율 H7–H10 충격 수 축소, α = 3)와 같다
#   실행: conda run -n qe-phm python plane_050.py   (E2 켜짐을 seed 5개로 세느라 몇 분)
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else "."); HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from src.exp2util import NS, rule83_flags
from src.models2 import TwoStack
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
warnings.filterwarnings("ignore")
ALPHA = 3.0
RING = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]
ns = NS(); y, g, n = ns.y, ns.g, ns.n; F0 = list(ns.FIXED)
M = np.hstack([ns.T[F0].values, ns.T[["L1_S"]].values, ns.L]); Mv = np.hstack([ns.Tv[F0].values, ns.Tv[["L1_S"]].values, ns.Lv])
Fn = np.column_stack([M, np.arange(n, dtype=float)])


class Shrink:
    """EXP-050 run.py와 같은 축소. 안쪽 모델 = TwoStack(J). p0 = fit 행의 학습 정상 충격 전체에서 대역별 비율"""
    def __init__(self): self.ri, self.i1 = [F0.index(c) for c in RING], F0.index("H1_nimp")
    def _tf(self, F):
        F = np.array(F, float); k = np.rint(np.expm1(F[:, self.i1])); has = k > 0
        F[np.ix_(has, self.ri)] = (F[np.ix_(has, self.ri)] * k[has, None] + ALPHA * self.p0_) / (k[has, None] + ALPHA); return F
    def fit(self, F, y):
        F = np.asarray(F, float); k = np.rint(np.expm1(F[:, self.i1])); m = (y == 0) & (k > 0)
        self.p0_ = (F[np.ix_(m, self.ri)] * k[m, None]).sum(0) / k[m].sum()
        self.m_ = TwoStack("lin", ("E2",), fixed=tuple(F0), nf=len(F0), df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True).fit(self._tf(F), y); return self
    def fires(self, F): return self.m_.fires(self._tf(F))


e2n = np.zeros(n, int); a0 = aez0 = z0 = None
for seed in range(5):
    a, aez, zz, e2 = np.empty(n), np.empty(n), np.empty(n), np.zeros(n, bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(g, y, g):
        f = Shrink().fit(Fn[tr], y[tr]).fires(Fn[te]); a[te], aez[te], zz[te], e2[te] = f["a"], f["ae"], f["z"], f["E2"]
    e2n += e2
    if seed == 0: a0, aez0, z0 = a.copy(), aez.copy(), zz.copy()
    print("050 seed", seed, flush=True)
mm = Shrink().fit(Fn, y); ts = mm.m_; c = ts.c_.coef_[0]
P = json.load(open(os.path.join(HERE, "figs", "plane_models.json"), encoding="utf-8"))
P["EXP-050"] = dict(pts=np.round(np.c_[a0, aez0], 3).tolist(), w=[float(c[0]), float(c[2])], wz=float(c[1]), z=np.round(z0, 3).tolist(), b=float(ts.c_.intercept_[0] - ts.tc_),
                    xl="ET 점수 (logit, OOF, 울림 비율 축소)", yl="AE z (OOF)",
                    lim=[float(np.floor(a0.min() - .3)), float(np.ceil(a0.max() + .3)), float(np.floor(aez0.min() - .3)), float(np.ceil(aez0.max() + .3))],
                    thr=float(ts.tc_), e2n=e2n.tolist(), p0=np.round(mm.p0_, 3).tolist(),
                    label="J-축소: ET logit(울림 비율 H7–H10을 충격 수로 축소, α = 3) × AE z, 박자 z = 0 단면, OOF seed 0 · 결정값 0 = 결합 문턱 %.2f (E2·8.3k 규칙은 이 경계와 별개로 고장 판정)" % ts.tc_)
json.dump(P, open(os.path.join(HERE, "figs", "plane_models.json"), "w", encoding="utf-8"))
# val: train 150 전체 학습, val AE는 val 신호로 계산 (run.py와 같음)
specv, resv = rule83_flags(ns.Xv); r83v = specv | resv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[y == 0]), LPv)}
fv = ts.fires_raw(mm._tf(Mv), RAW); pv = fv["main"] | fv["E2"] | r83v
cat = np.where(r83v, "8.3k", np.where(fv["E2"], "E2", np.where(fv["main"], "결합", "—")))
V = json.load(open(os.path.join(HERE, "figs", "plane_val.json"), encoding="utf-8"))
V["EXP-050"] = dict(names=[str(x) for x in ns.nv], pts=np.round(np.c_[fv["a"], fv["ae"]], 3).tolist(), z=np.round(fv["z"], 3).tolist(), flag=pv.astype(int).tolist(), cat=cat.tolist())
json.dump(V, open(os.path.join(HERE, "figs", "plane_val.json"), "w", encoding="utf-8"), ensure_ascii=False)
print("050 val 고장 판정", int(pv.sum()), "문턱", round(ts.tc_, 3), flush=True)
