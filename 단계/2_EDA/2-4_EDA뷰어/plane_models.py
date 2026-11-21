# %% EDA 뷰어 "경계" 화면용: EXP-039 · EXP-044 평면(점 좌표 · 경계 w,b · 축 범위 · E2 켜짐 횟수)을 계산해 figs/plane_models.json로 저장 (11-22)
#   build_viewer.py가 이 파일을 읽는다. 두 실험의 run.py와 같은 정의(OOF seed 0). 결정값 0 = 그 실험의 최종 임계값, 점선 ±1 = 로지스틱 결정값 ±1
#   실행: conda run -n qe-phm python plane_models.py   (오래 걸림: EXP-044 E2 켜짐을 seed 5개로 센다)
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
warnings.filterwarnings("ignore")
OUT = {}

# %% EXP-039: AE z × ET(D+) z, 규칙 없음 (대표 구성)
from src.exp import load_train, cached_table, feature_table_ext
from src.evaluate import best_f2_threshold
from src.dplus import DPlus, AEET
X, y, names = load_train(); g = np.arange(len(y))
E = cached_table("ext", feature_table_ext, X); dp = DPlus().fit(X, E); F_ = np.c_[g, dp.D_]
S = np.zeros((len(y), 2)); sc = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F_, y):
    m = AEET(rules=False).fit(F_[tr], y[tr]); S[te], _ = m.expert_scores(F_[te]); sc[te] = m.st_.decision_function(m.sc_.transform(S[te]))
thr = float(best_f2_threshold(y, sc)); fin = AEET(rules=False).fit(F_, y)
mu, sd = S[y == 0].mean(0), S[y == 0].std(0); Z = (S - mu) / sd
w = fin.st_.coef_[0] / fin.sc_.scale_; b = float(fin.st_.intercept_[0] - (fin.st_.coef_[0] * fin.sc_.mean_ / fin.sc_.scale_).sum())
OUT["EXP-039"] = dict(pts=np.round(Z, 3).tolist(), w=[float(w[0] * sd[0]), float(w[1] * sd[1])], b=float(w @ mu + b - thr),
                      xl="AE z (스펙트럼 모양 이상)", yl="ET z (충격 피처 이상, D+ 81열)", lim=[-3, 5.5, -3, 5.5], thr=thr,
                      label="AE z × ET(D+) z, OOF seed 0 · 결정값 0 = 임계값 %.2f" % thr)
print("039 done thr", thr, flush=True)

# %% EXP-044: ET logit × AE z (박자 z = 0 단면), J = 결합 OR E2 OR 8.3k
from src.exp2util import NS, Add83, rule83_flags
from src.models2 import TwoAE
ns = NS(); n = ns.n; assert (ns.y == y).all()
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float)
M_rule = np.column_stack([ns.M, np.arange(n, dtype=float), r83])
mk = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
e2n = np.zeros(n, int); a0 = aez0 = None
for seed in range(5):
    a, aez, e2 = np.empty(n), np.empty(n), np.zeros(n, bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(ns.g, ns.y, ns.g):
        f = mk().fit(M_rule[tr], ns.y[tr]).fires(M_rule[te])
        a[te], aez[te], e2[te] = f["a"], f["ae"], f["E2"]
    e2n += e2
    if seed == 0: a0, aez0 = a.copy(), aez.copy()
    print("044 seed", seed, flush=True)
mm = mk().fit(M_rule, ns.y).m_
c = mm.c_.coef_[0]
OUT["EXP-044"] = dict(pts=np.round(np.c_[a0, aez0], 3).tolist(), w=[float(c[0]), float(c[2])], b=float(mm.c_.intercept_[0] - mm.tc_),
                      xl="ET 점수 (logit, OOF)", yl="AE z (OOF)",
                      lim=[float(np.floor(a0.min() - .3)), float(np.ceil(a0.max() + .3)), float(np.floor(aez0.min() - .3)), float(np.ceil(aez0.max() + .3))],
                      thr=float(mm.tc_), e2n=e2n.tolist(),
                      label="ET logit × AE z, 박자 z = 0 단면, OOF seed 0 · 결정값 0 = 결합 문턱 %.2f (E2·8.3k 규칙은 이 경계와 별개로 고장 판정)" % mm.tc_)
json.dump(OUT, open(os.path.join(HERE, "figs", "plane_models.json"), "w", encoding="utf-8"))
print("done", flush=True)
