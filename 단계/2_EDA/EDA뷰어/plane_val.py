# %% EDA 뷰어 "경계" 화면용: val 40개를 EXP-039 · EXP-044 평면 위에 놓을 좌표·판정을 figs/plane_val.json에 저장 (11-22)
#   train 150 전체로 학습한 모델이 val을 판정한 값 (제출 파일 아님, val 라벨 없음). 평면 좌표 정의는 plane_models.py · 각 EXP run.py와 같다
#   실행: conda run -n qe-phm python plane_val.py
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else "."); HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from sklearn.model_selection import StratifiedKFold
warnings.filterwarnings("ignore")
OUT = {}

# %% EXP-039 (AE z × ET(D+) z, 규칙 없음): EXP-039 run.py의 val 계산과 같음
from src.exp import load_train, cached_table, feature_table_ext
from src.evaluate import best_f2_threshold
from src.dplus import DPlus, AEET, rule_flags
from src.data import load_folder
from src.final import _fit_ae, _ae_score
X, y, names = load_train(); g = np.arange(len(y))
E = cached_table("ext", feature_table_ext, X); dp = DPlus().fit(X, E); F_ = np.c_[g, dp.D_]
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val")); Dv, Tv = dp.transform(Xv)
S = np.zeros((len(y), 2)); sc = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F_, y):
    m = AEET(rules=False).fit(F_[tr], y[tr]); S[te], _ = m.expert_scores(F_[te]); sc[te] = m.st_.decision_function(m.sc_.transform(S[te]))
thr = float(best_f2_threshold(y, sc)); fin = AEET(rules=False).fit(F_, y)
mu, sd = S[y == 0].mean(0), S[y == 0].std(0)
ae_v = _ae_score(_fit_ae(dp.T_["LP"][y == 0]), Tv["LP"])
Sv = np.c_[ae_v, fin._et(fin.X_, y, Dv)]; sv = fin.st_.decision_function(fin.sc_.transform(Sv))
OUT["EXP-039"] = dict(names=[str(n) for n in nv], pts=np.round((Sv - mu) / sd, 3).tolist(), flag=(sv >= thr).astype(int).tolist(), cat=["결합" if f else "—" for f in sv >= thr])
print("039 val 고장 판정", int((sv >= thr).sum()), flush=True)

# %% EXP-044 (ET logit × AE z, 박자 z = 0 단면): EXP-044 run.py의 val 계산과 같음
from src.exp2util import NS, Add83, rule83_flags
from src.models2 import TwoAE
from src.experts import LP as LP_TR, segment_logpsd
ns = NS(); assert (ns.y == y).all()
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res).astype(float); r83v = (specv | resv).astype(float)
idx = np.arange(ns.n, dtype=float); M_rule = np.column_stack([ns.M, idx, r83])
mJ = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK))).fit(M_rule, ns.y); mm = mJ.m_
_, LPv = segment_logpsd(ns.Xv); mm.ae_override_ = _ae_score(_fit_ae(LP_TR[ns.y == 0]), LPv)
Mv = np.column_stack([ns.Mv, np.zeros(len(ns.Mv)), r83v]); fv = mJ.fires(Mv); pv = mJ.decision_function(Mv) >= .5
cat = np.where(fv["R83"], "8.3k", np.where(fv["E2"], "E2", np.where(fv["main"], "결합", "—")))
assert [str(n) for n in ns.nv] == [str(n) for n in nv], "val 파일 순서가 다름"
OUT["EXP-044"] = dict(names=[str(n) for n in ns.nv], pts=np.round(np.c_[fv["a"], fv["ae"]], 3).tolist(), z=np.round(fv["z"], 3).tolist(), flag=pv.astype(int).tolist(), cat=cat.tolist())
print("044 val 고장 판정", int(pv.sum()), flush=True)
json.dump(OUT, open(os.path.join(HERE, "figs", "plane_val.json"), "w", encoding="utf-8"), ensure_ascii=False)
