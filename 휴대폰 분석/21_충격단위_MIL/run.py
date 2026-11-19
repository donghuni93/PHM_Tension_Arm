# %% [markdown]
# 21. 충격 단위 다중 인스턴스(MIL): 충격 하나하나를 샘플로 학습하고 파일 점수 = 상위 충격 집계 — 파일 평균에 묻히는 022 · 017이 가려지는가
# 입력: 14번이 뽑은 충격 1497개(파일 150개)의 (울림 f, 감쇠비 ζ, fit, 높이) + 파일 RMS로 나눈 상대 높이. 충격 라벨 = 파일 라벨(MIL 가정). 폴드는 파일 단위.
# 평가: 5겹 × 3회 OOF. 파일 점수 = 충격 로짓의 상위 k개 평균(k = 1 · 3). 정상 대비 백분위 · AUC. 실행: python "휴대폰 분석/21_충격단위_MIL/run.py"
import os, sys, warnings
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); _d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np, pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.exp import load_train
X, y, names = load_train()
I = pd.read_csv(os.path.join(HERE, "..", "14_감쇠비_정밀구분", "impacts_fz.csv"))
rms = X.std(1); I["amp_rel"] = np.log10(I["amp"] / rms[I["file"].values] + 1e-6); I["log_amp"] = np.log10(I["amp"] + 1e-6); I["n_file"] = I.groupby("file")["f"].transform("size")
cols = ["f", "zeta", "fit", "amp_rel", "log_amp"]; Xi = I[cols].values; fi = I["file"].values
FOCUS = ["abn_002", "abn_014", "abn_017", "abn_022", "abn_029", "nor_116", "nor_055", "nor_025", "nor_106"]; foc = [names.index(n) for n in FOCUS]
has = np.isin(np.arange(len(y)), np.unique(fi)); print("충격 보유 파일:", int(has.sum()), "/ 150 | 충격 없는 파일:", [names[i] for i in np.where(~has)[0]][:8])
out = {1: np.full((3, len(y)), np.nan), 3: np.full((3, len(y)), np.nan)}
for s in range(3):
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(len(y)), y):
        mtr = np.isin(fi, tr); mte = np.isin(fi, te)
        et = ExtraTreesClassifier(400, class_weight="balanced_subsample", min_samples_leaf=3, max_features="sqrt", random_state=0, n_jobs=-1).fit(Xi[mtr], y[fi[mtr]])
        p = np.clip(et.predict_proba(Xi[mte])[:, 1], 0.01, 0.99); lg = np.log(p / (1 - p)); df = pd.DataFrame({"file": fi[mte], "lg": lg})
        for k in (1, 3):
            sc = df.groupby("file")["lg"].apply(lambda v: np.sort(v.values)[::-1][:k].mean()); out[k][s, sc.index.values] = sc.values
for k in (1, 3):
    m = np.nanmean(out[k], axis=0); ok = ~np.isnan(m); mm = np.where(ok, m, np.nanmin(m))
    pct = np.array([(mm[(y == 0)] < mm[i]).mean() for i in range(len(y))])
    print(f"상위 {k}개 집계: 파일 AUC {roc_auc_score(y[ok], m[ok]):.3f} | 고장 중 정상의 90%보다 이상 {(pct[(y == 1) & ok] >= .9).sum()}/{((y == 1) & ok).sum()} | 정상 중 90% 초과 {(pct[(y == 0) & ok] >= .9).sum()}/{((y == 0) & ok).sum()}")
    print("  파일별 백분위(정상 대비):", {names[i][-7:]: round(float(pct[i]), 2) for i in foc})

# %% 2부: 충격 단위 MIL 점수를 세 번째 전문가로 (AE + ET(D+) + MIL), EXP-036과 같은 중첩 CV
import copy, time
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from src.evaluate import nested_cv
from src.dplus import DPlus, AEET, _rule83
from src.exp import cached_table, feature_table_ext
E = cached_table("ext", feature_table_ext, X); D = DPlus().fit(X, E).D_; g = np.arange(len(y))
def mil_score(tr_idx, ytr, te_idx, k=3):
    """학습 파일의 충격으로 ET 학습 → 시험 파일 점수 = 상위 k개 충격 로짓 평균 (충격 없는 파일 = 낮은 값 −3)"""
    lab = dict(zip(tr_idx, ytr)); mtr = np.isin(fi, tr_idx); yi = np.array([lab[f] for f in fi[mtr]])
    et = ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=3, max_features="sqrt", random_state=0, n_jobs=1).fit(Xi[mtr], yi)
    mte = np.isin(fi, te_idx); p = np.clip(et.predict_proba(Xi[mte])[:, 1], 0.01, 0.99); lg = pd.Series(np.log(p / (1 - p))).groupby(fi[mte]).apply(lambda v: np.sort(v.values)[::-1][:k].mean())
    return np.array([lg.get(i, -3.0) for i in te_idx])
class AEETM(AEET):
    def fit(self, F, yy):
        from src.experts import KnockRule, ae_mean
        F = np.asarray(F); idx, Xf = F[:, 0].astype(int), F[:, 1:]; oof = np.zeros((len(yy), 3))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1).split(Xf, yy):
            oof[te, 0] = ae_mean(idx[tr], idx[te])[:, 0]; oof[te, 1] = self._et(Xf[tr], yy[tr], Xf[te]); oof[te, 2] = mil_score(idx[tr], yy[tr], idx[te])
        self.sc_ = StandardScaler().fit(oof); self.st_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(self.sc_.transform(oof), yy)
        self.idx_, self.X_, self.y_ = idx, Xf, yy
        self.rules_ = [copy.deepcopy(KnockRule()).fit(idx), _rule83()] if self.rules else []; return self
    def expert_scores(self, F):
        from src.experts import ae_mean
        F = np.asarray(F); te = F[:, 0].astype(int)
        s = np.c_[ae_mean(self.idx_, te)[:, 0], self._et(self.X_, self.y_, F[:, 1:]), mil_score(self.idx_, self.y_, te)]
        on = np.any([r(te) for r in self.rules_], axis=0) if self.rules_ else np.zeros(len(te), bool); return s, on
Fd = np.c_[g, D]
for nm, rl, cls in (("AE+ET(D+) 규칙 없음 [EXP-036]", False, AEET), ("AE+ET(D+)+MIL 규칙 없음", False, AEETM), ("AE+ET(D+) + 규칙 [EXP-036]", True, AEET), ("AE+ET(D+)+MIL + 규칙", True, AEETM)):
    t0 = time.time(); r = nested_cv(Fd, y, g, lambda cls=cls, rl=rl: cls(rules=rl), n_jobs=-1)
    print(f"{nm:34s} F₂ {r['f2'].mean():.3f} ± {r['f2'].std():.3f} | 재현율 {r['recall'].mean():.3f} | 오경보율 {r['fpr'].mean():.3f} | 놓침 { {names[i][-3:]: int(r['miss_counts'][i]) for i in np.where(r['miss_counts'] >= 1)[0]} } | 오경보 3회+ {[names[i][-3:] for i in np.where(r['fp_counts'] >= 3)[0]]} | {time.time() - t0:.0f}s", flush=True)
