# %% [markdown]
# 20. 정상만 학습하는 이상 탐지 전문가(LOF)를 세 번째로 추가: 017 · 022 같은 단독 유형을 가를 수 있는가
# 동기: EXP-023 기록 — 017은 정상만 학습한 MiniRocket LOF 0.99 · iforest 0.99 · kNN 0.98로 잘 갈리나 AE는 0.44. 정상 기준 거리(novelty)는 AE(스펙트럼 모양)와 다른 정보를 줄 수 있다.
# 1부: 폴드 안(학습 정상만)으로 LOF를 맞추고 시험 파일의 이상 점수 → 파일별 정상 대비 백분위(고장이 정상의 몇 %보다 이상한가) · AUC
# 2부: AE + ET(D+) + LOF 스택을 EXP-039과 같은 중첩 CV(seed 5 × 바깥 5 × 안쪽 4)로 비교. AE 캐시가 있어 빠르다.
# 실행: python "3_단계/4_피처/4-4_1119_휴대폰_피처·모델탐색/20_새규칙없는_이상탐지전문가/run.py"
import os, sys, warnings, time
warnings.filterwarnings("ignore")
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.exp import load_train, cached_table, feature_table_ext
from src.evaluate import nested_cv
from src.dplus import DPlus, AEET
HERE = os.path.dirname(os.path.abspath(__file__))
X, y, names = load_train(); E = cached_table("ext", feature_table_ext, X); g = np.arange(len(y)); dp = DPlus().fit(X, E); D = dp.D_
FOCUS = ["abn_002", "abn_014", "abn_017", "abn_022", "abn_029", "nor_116", "nor_055", "nor_025", "nor_106", "nor_059"]
fi = [names.index(n) for n in FOCUS]
# %% 1부: LOF 이상 점수 OOF
def lof_score(Xa, ya, Xb, k=15): return -LocalOutlierFactor(n_neighbors=k, novelty=True).fit(Xa[ya == 0]).decision_function(Xb)
def knn_score(Xa, ya, Xb, k=5):
    from sklearn.neighbors import NearestNeighbors
    return NearestNeighbors(n_neighbors=k).fit(Xa[ya == 0]).kneighbors(Xb)[0].mean(1)
res = {}
for nm, fn in (("LOF k=15", lof_score), ("kNN k=5", knn_score)):
    sc = np.zeros((3, len(y)))
    for s in range(3):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(D, y): sc[s, te] = fn(D[tr], y[tr], D[te])
    m = sc.mean(0); pct = np.array([(m[y == 0] < m[i]).mean() for i in range(len(y))])             # 정상 중 이 파일보다 덜 이상한 비율
    res[nm] = (m, pct); print(f"{nm}: 전체 AUC {roc_auc_score(y, m):.3f} | 고장 중 정상의 95%보다 이상 {(pct[y==1] >= .95).sum()}/30 | 정상 중 95% 초과 {(pct[y==0] >= .95).sum()}/120")
    print("  파일별 백분위(정상 대비):", {names[i][-7:]: round(float(pct[i]), 2) for i in fi})
# %% 2부: 3 전문가 스택 (AE + ET(D+) + LOF)
class AEETN(AEET):
    def fit(self, F, yy):
        from src.experts import KnockRule, ae_mean
        import copy
        F = np.asarray(F); idx, Xf = F[:, 0].astype(int), F[:, 1:]; oof = np.zeros((len(yy), 3))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1).split(Xf, yy):
            oof[te, 0] = ae_mean(idx[tr], idx[te])[:, 0]; oof[te, 1] = self._et(Xf[tr], yy[tr], Xf[te]); oof[te, 2] = lof_score(Xf[tr], yy[tr], Xf[te])
        self.sc_ = StandardScaler().fit(oof); self.st_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(self.sc_.transform(oof), yy)
        self.idx_, self.X_, self.y_ = idx, Xf, yy
        from src.dplus import _rule83
        self.rules_ = [copy.deepcopy(KnockRule()).fit(idx), _rule83()] if self.rules else []; return self
    def expert_scores(self, F):
        from src.experts import ae_mean
        F = np.asarray(F); te = F[:, 0].astype(int)
        s = np.c_[ae_mean(self.idx_, te)[:, 0], self._et(self.X_, self.y_, F[:, 1:]), lof_score(self.X_, self.y_, F[:, 1:])]
        on = np.any([r(te) for r in self.rules_], axis=0) if self.rules_ else np.zeros(len(te), bool); return s, on
Fd = np.c_[g, D]
for nm, rl, cls in (("AE+ET(D+) 규칙 없음 [EXP-039]", False, AEET), ("AE+ET(D+)+LOF 규칙 없음", False, AEETN), ("AE+ET(D+) + 규칙 [EXP-039]", True, AEET), ("AE+ET(D+)+LOF + 규칙", True, AEETN)):
    t0 = time.time(); r = nested_cv(Fd, y, g, lambda cls=cls, rl=rl: cls(rules=rl), n_jobs=-1)
    print(f"{nm:34s} F₂ {r['f2'].mean():.3f} ± {r['f2'].std():.3f} | 재현율 {r['recall'].mean():.3f} | 오경보율 {r['fpr'].mean():.3f} | 놓침 { {names[i][-3:]: int(r['miss_counts'][i]) for i in np.where(r['miss_counts'] >= 1)[0]} } | 오경보 3회+ {[names[i][-3:] for i in np.where(r['fp_counts'] >= 3)[0]]} | {time.time() - t0:.0f}s", flush=True)
