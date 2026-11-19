# %% [markdown]
# 17. AE + ET(D 피처) 결합 — 16번에서 빠졌던 AE(스펙트럼 모양 전문가)를 D 피처 ET와 결합해 같은 프로토콜로 비교
# 구성: 전문가 ① AE(src.experts.ae_mean, 정상만 학습) ② ET(입력 = ext150 원본 또는 D 76열) → 안쪽 4겹 OOF 점수를 로지스틱 결합 (+ 선택: 8.3k · 유격 규칙 = EXP-025)
# 비교(모두 nested_cv seed 5 × 바깥 5 × 안쪽 4): P0 AE+ET(ext150) = EXP-014 재현 · P1 AE+ET(D) · P2 P0 + 규칙 = EXP-025 재현 · P3 P1 + 규칙
# 실행: python "휴대폰 분석/17_AE_ET_D/run.py" > log (AE 학습이 많아 수 분 걸린다)

# %% 설정
import os, sys, warnings, copy, time
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from sklearn.base import BaseEstimator
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from src.experts import X, y, names, E, ae_mean, Rule83, KnockRule
from src.evaluate import nested_cv
from src.exp import load_train, cached_table, feature_table_ext
HERE = os.path.dirname(os.path.abspath(__file__))
# D 피처: 16번과 같은 방식 (대표 62열 + 울림 4 + 물리 10). 16번 run.py의 함수를 가져와 쓴다
src16 = open(os.path.join(HERE, "..", "16_D피처모델", "run.py"), encoding="utf-8").read().split("# %% 모델")[0]
src16 = src16.replace('os.path.dirname(__file__) if "__file__" in dir() else "."', repr(os.path.join(HERE, "..", "16_D피처모델")))
g16 = {"__file__": os.path.join(HERE, "..", "16_D피처모델", "run.py")}; exec(compile(src16, "run16", "exec"), g16)
D = g16["D"]; print("D", D.shape, flush=True)
def logit(p): p = np.clip(p, 0.01, 0.99); return np.log(p / (1 - p))
class AEET(BaseEstimator):
    """F = [파일 번호, 피처...]. 전문가 AE(파일 번호로 동작) + ET(피처) → 안쪽 OOF → 로지스틱. rules=True면 규칙이 켜진 파일 +100"""
    def __init__(self, rules=False): self.rules = rules
    @staticmethod
    def _et(Xa, ya, Xb): return logit(ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=1).fit(Xa, ya).predict_proba(Xb)[:, 1])
    def fit(self, F, yy):
        F = np.asarray(F); idx, Xf = F[:, 0].astype(int), F[:, 1:]; oof = np.zeros((len(yy), 2))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1).split(Xf, yy):
            oof[te, 0] = ae_mean(idx[tr], idx[te])[:, 0]; oof[te, 1] = self._et(Xf[tr], yy[tr], Xf[te])
        self.sc_ = StandardScaler().fit(oof); self.st_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(self.sc_.transform(oof), yy)
        self.idx_, self.X_, self.y_ = idx, Xf, yy
        self.rules_ = [copy.deepcopy(KnockRule()).fit(idx), Rule83()] if self.rules else []
        return self
    def decision_function(self, F):
        F = np.asarray(F); te = F[:, 0].astype(int)
        s = np.c_[ae_mean(self.idx_, te)[:, 0], self._et(self.X_, self.y_, F[:, 1:])]
        out = self.st_.decision_function(self.sc_.transform(s))
        for r in self.rules_: out = out + 100.0 * r(te)
        return out
g = np.arange(len(y))
CFG = {"P0 AE+ET(ext150)  [EXP-014 재현]": (np.c_[g, E], False), "P1 AE+ET(D 76열)": (np.c_[g, D], False),
       "P2 P0 + 규칙 [EXP-025 재현]": (np.c_[g, E], True), "P3 AE+ET(D) + 규칙": (np.c_[g, D], True)}
rows = []
for nm, (F_, rl) in CFG.items():
    t0 = time.time(); r = nested_cv(F_, y, g, lambda rl=rl: AEET(rules=rl), seeds=range(3), n_jobs=-1)
    print(f"{nm:34s} F₂ {r['f2'].mean():.3f} ± {r['f2'].std():.3f} | 재현율 {r['recall'].mean():.3f} | 오경보율 {r['fpr'].mean():.3f} | 자주 놓침 {[names[i][-3:] for i in np.where(r['miss_counts'] >= 3)[0]]} | 오경보 3회+ {[names[i][-3:] for i in np.where(r['fp_counts'] >= 3)[0]]} | {time.time() - t0:.0f}s", flush=True)
    rows.append((nm, r["f2"].mean(), r["f2"].std(), r["recall"].mean(), r["precision"].mean(), r["fpr"].mean()))
pd.DataFrame(rows, columns=["구성", "F2", "sd", "재현율", "정밀도", "오경보율"]).round(3).to_csv(os.path.join(HERE, "nested_cv.csv"), index=False, encoding="utf-8-sig")
