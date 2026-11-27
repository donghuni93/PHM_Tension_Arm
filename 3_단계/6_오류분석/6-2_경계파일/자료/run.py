# %% [markdown]
# 6-3 (11-22) 경계 파일 진단: J(EXP-044)의 결합 여유를 항별로 나눈다 — 어느 항 때문에 경계에 걸리는가
# - 결합 여유 = w_ET·ET logit + w_S·박자 z + w_AE·AE z + 절편 − 결합 문턱  (로지스틱 결정함수, 규칙 E2 · 8.3k는 별도)
# - train = OOF seed 0 (폴드마다 그 폴드 모델의 계수), val = train 150 전체 학습 모델(EXP-044 val 판정과 같음)
# - 경계 = |여유| ≤ 1.5, 또는 규칙만으로 잡힌 고장(여유 < 0), 또는 오경보. 각 파일의 H 열 중 정상 기준 |z| 큰 3개(map_scale)도 적는다
# 실행: conda run -n qe-phm python "3_단계/6_오류분석/6-2_경계파일/자료/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.exp import *
from src.exp2util import *
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
HERE = os.path.dirname(os.path.abspath(__file__))
ns = NS(); y, g = ns.y, ns.g; names = np.array(ns.names); nor = y == 0
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv)
Mr = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)])
mk = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
BAND = 1.5


def parts(m, f):
    """결합 여유를 항으로: (ET, 박자, AE, 절편−문턱). 합 = score − tc"""
    w = m.m_.c_.coef_[0]; b = m.m_.c_.intercept_[0] - m.m_.tc_
    return np.column_stack([w[0] * f["a"], w[1] * f["z"], w[2] * f["ae"], np.full(len(f["a"]), b)])


# %% train OOF seed 0
C = np.zeros((ns.n, 4)); part = np.empty(ns.n, object); W = []
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    m = mk().fit(Mr[tr], y[tr]); f = m.fires(Mr[te]); C[te] = parts(m, f)
    assert np.allclose(C[te].sum(1), f["score"] - m.m_.tc_)
    part[te] = np.where(f["R83"], "8.3k", np.where(f["E2"], "E2", np.where(f["main"], "결합", "—")))
    W.append(list(m.m_.c_.coef_[0]) + [m.m_.c_.intercept_[0] - m.m_.tc_])
# %% val (train 150 전체)
mJ = mk().fit(Mr, y); mm = mJ.m_
mm.ae_override_ = _ae_score(_fit_ae(LP_TR[y == 0]), segment_logpsd(ns.Xv)[1])
fv = mJ.fires(np.column_stack([ns.Mv, np.zeros(len(ns.Mv)), (specv | resv).astype(float)])); Cv = parts(mJ, fv)
partv = np.where(fv["R83"], "8.3k", np.where(fv["E2"], "E2", np.where(fv["main"], "결합", "—")))
W.append(list(mm.c_.coef_[0]) + [mm.c_.intercept_[0] - mm.tc_])

# %% 각 파일이 정상 기준으로 가장 벗어난 H 열 (map_scale, 기준 = train 정상)
from src.features2 import PeakRef
Tn, Tvn = ns.T.copy(), ns.Tv.copy(); pr = PeakRef(ns.df).fit(ns.L[nor])
for D_, L_ in ((Tn, ns.L), (Tvn, ns.Lv)):
    pk = pr.transform(L_); D_["H16_peak_h"], D_["H17_peak_w"] = pk[:, 0], pk[:, 1]
cols = list(ns.cols); Zt, Zv = map_scale(Tn[cols].values[nor], Tn[cols].values, Tvn[cols].values)
top = lambda z: " · ".join(f"{cols[j]} {z[j]:+.1f}" for j in np.argsort(-np.abs(z))[:3])


def table(Cm, lab, partx, nmx, Zx, sel):
    rows = []
    for i in sel:
        c = Cm[i]; s = c.sum()
        rows.append({"파일": nmx[i], "정답": lab[i], "결합 여유": f"{s:+.2f}", "ET": f"{c[0]:+.2f}", "박자": f"{c[1]:+.2f}", "AE": f"{c[2]:+.2f}", "절편−문턱": f"{c[3]:+.2f}",
                     "판정 부품": partx[i], "정상 기준 큰 |z| H열 3개": top(Zx[i])})
    return pd.DataFrame(rows)


ms, msv = C.sum(1), Cv.sum(1)
sel_t = [i for i in np.argsort(ms) if abs(ms[i]) <= BAND or (y[i] and ms[i] < 0) or (not y[i] and part[i] != "—")]
sel_v = [i for i in np.argsort(msv) if abs(msv[i]) <= BAND or partv[i] == "8.3k"]
Tt = table(C, np.where(y == 1, "고장", "정상"), part, [short(n) for n in names], Zt, sel_t)
Tv = table(Cv, np.array(["?"] * len(ns.nv)), partv, [n[-3:] for n in ns.nv], Zv, sel_v)
Wd = pd.DataFrame(W, columns=["w_ET", "w_박자", "w_AE", "절편−문턱"], index=[f"fold{k}" for k in range(5)] + ["전체(val용)"]).round(3)

# 기여 분포 요약: 정상 · 고장 · val 중앙
summ = pd.DataFrame({"정상 중앙": np.median(C[nor], 0), "고장 중앙": np.median(C[~nor], 0), "val 중앙": np.median(Cv, 0)}, index=["ET", "박자", "AE", "절편−문턱"]).round(2).T
summ["여유 합"] = summ.sum(1).round(2)

md = [f"# 6-3 경계 파일 진단 (J = EXP-044)\n\n결합 여유 = ET + 박자 + AE + (절편 − 문턱). 경계 = |여유| ≤ {BAND} · 규칙으로만 잡힌 고장 · 오경보 · (val) 8.3k로 고장.\n",
      "\n## 로지스틱 계수 (폴드별, val용 전체 학습)\n\n", md_table(Wd.reset_index().rename(columns={"index": "모델"})),
      "\n\n## 항별 기여 중앙값\n\n", md_table(summ.reset_index().rename(columns={"index": "묶음"})),
      f"\n\n## train 경계 파일 (OOF seed 0, {len(sel_t)}개, 여유 순)\n\n", md_table(Tt),
      f"\n\n## val 경계 파일 ({len(sel_v)}개, 여유 순)\n\n", md_table(Tv), "\n"]
open(os.path.join(HERE, "경계파일_표.md"), "w", encoding="utf-8").write("".join(md))
print("done")
