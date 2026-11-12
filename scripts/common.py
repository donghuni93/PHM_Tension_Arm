"""실험 스크립트 공용: 데이터·피처 캐시, 모델, 결과 출력"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.data import load_dataset
from src.features import feature_table_ext, feature_table_phys
from src.evaluate import nested_cv
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier

OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data", "TeamProject_train_변경")
os.makedirs(OUT, exist_ok=True)


def load_train():
    """X, y, names (캐시: outputs/train_X.npz)"""
    p = os.path.join(OUT, "train_X.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True)
        return z["X"], z["y"], list(z["names"])
    X, y, names = load_dataset(DATA)
    np.savez(p, X=X, y=y, names=np.array(names))
    return X, y, names


def cached_table(name, fn, X):
    """피처 표 캐시 (outputs/feat_{name}.csv). inf·NaN은 0"""
    p = os.path.join(OUT, f"feat_{name}.csv")
    if os.path.exists(p):
        T = pd.read_csv(p)
    else:
        T = fn(X)
        T.to_csv(p, index=False)
    return T.replace([np.inf, -np.inf], np.nan).fillna(0)


def make_et(n=300, seed=0):
    return lambda: ExtraTreesClassifier(n, class_weight="balanced_subsample", min_samples_leaf=2,
                                        max_features="sqrt", random_state=seed, n_jobs=-1)


class ColModel(BaseEstimator):
    """열 일부만 쓰는 모델"""
    def __init__(self, cols, make): self.cols, self.make = cols, make
    def fit(self, X, y): self.m_ = self.make().fit(X[:, self.cols], y); return self
    def predict_proba(self, X): return self.m_.predict_proba(X[:, self.cols])


class RankAvg(BaseEstimator):
    """여러 모델의 고장 확률을 학습 데이터 점수 분포 기준 백분위로 바꿔 평균"""
    def __init__(self, makers): self.makers = makers
    def fit(self, X, y):
        self.ms_ = [m().fit(X, y) for m in self.makers]
        self.ref_ = [np.sort(m.predict_proba(X)[:, 1]) for m in self.ms_]
        return self
    def decision_function(self, X):
        s = [np.searchsorted(r, m.predict_proba(X)[:, 1], side="right") / len(r) for m, r in zip(self.ms_, self.ref_)]
        return np.mean(s, axis=0)


def report(name, r, names, log):
    mi = [names[i] for i in np.flatnonzero(r["miss_counts"] >= 3)]
    fp = int((r["fp_counts"] >= 3).sum())
    line = (f"{name:40s} F2 {r['f2'].mean():.3f} ± {r['f2'].std():.3f} | R {r['recall'].mean():.3f} "
            f"P {r['precision'].mean():.3f} | 자주 놓침 {' '.join(m.replace('abn_', '') for m in mi)} | 오경보(3회+) {fp}")
    print(line, flush=True)
    log.write(line + "\n"); log.flush()
