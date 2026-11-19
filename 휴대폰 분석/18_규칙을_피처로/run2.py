# %% [markdown]
# 18-2. 유격 규칙의 '정상 중심 거리 r'(K1 · K2를 학습 정상의 중앙값 · MAD로 맞춘 z의 RMS)을 폴드 안에서 계산해 ET 피처로 추가 (Q2)
# 이유: 트리는 열 하나씩 자르므로 두 열의 거리 r을 직접 못 만든다. r은 폴드 학습 정상만으로 정해져 누수 없음(규칙의 문턱 2 · 0.7은 쓰지 않음).
# 비교(3 seed): Q0 AE+ET(D+) · Q2 = Q0 + r + K3 · P3 AE+ET(D)+규칙. 실행: python "휴대폰 분석/18_규칙을_피처로/run2.py"
import os, sys, warnings, time, copy
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "run.py"), encoding="utf-8").read().split("g = np.arange(len(y)); CFG")[0]
G = {"__file__": os.path.join(HERE, "run.py")}; exec(compile(src, "run18", "exec"), G)
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.experts import KnockRule
from src.evaluate import nested_cv, best_f2_threshold
AEET, ae_mean, y, names, Dp, Dpv, Tt, Tv, nv = G["AEET"], G["ae_mean"], G["y"], G["names"], G["Dp"], G["Dpv"], G["Tt"], G["Tv"], G["nv"]
class AEET2(AEET):
    """AEET + 폴드 안 유격 거리 r 피처 (r은 학습 정상의 중앙값 · MAD로 맞춘 K1 · K2 z의 RMS)"""
    def fit(self, F, yy):
        F = np.asarray(F); idx, Xf = F[:, 0].astype(int), F[:, 1:]; oof = np.zeros((len(yy), 2))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1).split(Xf, yy):
            kr = KnockRule().fit(idx[tr]); oof[te, 0] = ae_mean(idx[tr], idx[te])[:, 0]
            oof[te, 1] = self._et(np.c_[Xf[tr], kr.r(idx[tr])], yy[tr], np.c_[Xf[te], kr.r(idx[te])])
        self.sc_ = StandardScaler().fit(oof); self.st_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(self.sc_.transform(oof), yy)
        self.kr_ = KnockRule().fit(idx); self.idx_, self.y_ = idx, yy; self.X_ = np.c_[Xf, self.kr_.r(idx)]
        self.rules_ = [copy.deepcopy(self.kr_)] if self.rules else []; return self
    def decision_function(self, F):
        F = np.asarray(F); te = F[:, 0].astype(int)
        s = np.c_[ae_mean(self.idx_, te)[:, 0], self._et(self.X_, self.y_, np.c_[F[:, 1:], self.kr_.r(te)])]
        return self.st_.decision_function(self.sc_.transform(s))
g = np.arange(len(y)); Fq = np.c_[g, Dp]
CFG = {"Q0 AE+ET(D+) (r 없음)": (AEET, False), "Q2 AE+ET(D+ + 유격 r)": (AEET2, False)}
for nm, (cls, rl) in CFG.items():
    t0 = time.time(); r = nested_cv(Fq, y, g, lambda cls=cls, rl=rl: cls(rules=rl), seeds=range(3), n_jobs=-1)
    print(f"{nm:24s} F₂ {r['f2'].mean():.3f} ± {r['f2'].std():.3f} | 재현율 {r['recall'].mean():.3f} | 오경보율 {r['fpr'].mean():.3f} | 놓침 횟수 { {names[i][-3:]: int(r['miss_counts'][i]) for i in np.where(r['miss_counts'] >= 1)[0]} } | 오경보 3회+ {[names[i][-3:] for i in np.where(r['fp_counts'] >= 3)[0]]} | {time.time() - t0:.0f}s", flush=True)
