# EXP-042 해석용: D2에 8.3k 규칙(src/final.py와 같은 문턱·정의, 학습 없음)을 OR로 붙이면 train 중첩 CV가 어떻게 되나. run.py는 고치지 않음
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d, median_filter
from sklearn.base import BaseEstimator
from src.exp2util import *
from src.evaluate import nested_cv
from src.final import ring_table, P83_MIN, W83_MIN, RES_FIT_MIN, FS

HERE = os.path.dirname(os.path.abspath(__file__))
ns = NS(); y, names = ns.y, ns.names


def spec_table(X):          # final.signal_tables의 P83·W83 부분만 (같은 식)
    Xc = X - X.mean(1, keepdims=True)
    f, P8 = welch(Xc, fs=FS, nperseg=8192); L = np.log10(P8); df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
    PR = Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest")
    band = (f >= 8200) & (f < 8450); ia = np.flatnonzero(band)
    P83, W83 = PR[:, band].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, band].argmax()]; h = PR[i, k] / 2; l = r = k
        while l > 0 and PR[i, l - 1] >= h: l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h: r += 1
        W83[i] = f[r] - f[l]
    return P83, W83


P83, W83 = spec_table(ns.X); _, RES, _ = ring_table(ns.X)
spec = (P83 >= P83_MIN) & (W83 >= W83_MIN); res = RES >= RES_FIT_MIN
print("규칙이 켜진 train 파일 — 스펙트럼: 고장", [names[i] for i in np.flatnonzero(spec & (y == 1))], "정상", [names[i] for i in np.flatnonzero(spec & (y == 0))])
print("                       공진:     고장", [names[i] for i in np.flatnonzero(res & (y == 1))], "정상", [names[i] for i in np.flatnonzero(res & (y == 0))])


class WithRule(BaseEstimator):
    """마지막 열 = 규칙 켜짐(0/1). 나머지로 base 학습, 판정 = base OR 규칙 (이진)"""
    def __init__(self, mk=None): self.mk = mk
    def fit(self, F, yy): self.m_ = self.mk().fit(F[:, :-1], yy); return self
    def decision_function(self, F): return np.maximum(self.m_.decision_function(F[:, :-1]), F[:, -1])


W = json.load(open(os.path.join(find_exp("EXP-041"), "result.json"), encoding="utf-8"))["notes"]["선택"]["winner"]
mk = ns.mk(W, ("E1", "E2"))
CFG = {"D2 (규칙 없음)": np.zeros(ns.n), "D2 + 8.3k 스펙트럼": spec.astype(float), "D2 + 8.3k (스펙트럼 OR 공진, 최종과 같은 정의)": (spec | res).astype(float)}
lines = []
for nm, r in CFG.items():
    R = nested_cv(np.column_stack([ns.M, r]), y, ns.g, lambda: WithRule(mk), n_jobs=-1)
    mc = {names[i]: int(R["miss_counts"][i]) for i in np.flatnonzero(R["miss_counts"] >= 3)}; fc = {names[i]: int(R["fp_counts"][i]) for i in np.flatnonzero(R["fp_counts"] >= 3)}
    lines.append(f"{nm:44s} F2 {R['f2'].mean():.3f} ± {R['f2'].std():.3f} · 재현율 {R['recall'].mean():.3f} · 오경보율 {R['fpr'].mean():.3f} · 미탐(3회+) {mc} · 오경보(3회+) {fc}")
    print(lines[-1], flush=True)
open(os.path.join(HERE, "figs", "D2_8.3k규칙_log.txt"), "w", encoding="utf-8").write("\n".join(lines))
