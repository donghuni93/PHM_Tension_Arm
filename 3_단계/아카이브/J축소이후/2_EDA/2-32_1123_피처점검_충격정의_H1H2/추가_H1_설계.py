# %% H1 설계 변형 비교: (A) 문턱 K 격자 (B) 시그마(MAD) 문턱 (C) 잔향 꼬리 제거 / 병합 (D) 크기 반영(로그 높이 합 · 최댓값). 지표 = 분리도 · val 이동도 · 정상 0개 비율 · quick_cv PR-AUC(기준 ET에서 H1 열만 교체)
# 주의: 라벨로 고르는 비교라 최댓값 하나보다 평평한 구간을 본다. quick_cv도 같은 150개라 낙관적일 수 있다
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from src.exp import make_et
from src.evaluate import quick_cv
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "run.py"), encoding="utf-8").read().split("# %% 그림 1")[0])      # ns, X, RAT, peaks, grp, y, ntr 등 재사용
E, BASE = [], []
for x in X:
    e = sosfiltfilt(_sos("lowpass", SMOOTH_HZ, FS), np.abs(hilbert(highpass(x)))); E.append(e); BASE.append(_baseline(e))
edge = int(round(EDGE_MS * FS / 1000)); dist = max(1, int(round(DIST_MS * FS / 1000)))
def pk_of(i, K=None, sig=None):
    r = RAT[i]
    if sig is not None:                                   # 시그마 문턱: 포락선 e가 중앙값 + sig·1.4826·MAD 위
        e = E[i]; thr = np.median(e) + sig * 1.4826 * stats.median_abs_deviation(e); p = find_peaks(e, height=thr, distance=dist)[0]
    else: p = find_peaks(r, height=K, distance=dist)[0]
    return p[(p >= edge) & (p < len(r) - edge)]
def dedupe(p, h, W, a):                                   # 직전 W ms 안에 a배 이상 큰 봉우리가 있으면 꼬리로 보고 제거 (a=1: 그 안의 모든 뒤 봉우리 병합)
    w = int(W * FS / 1000); keep = []
    for j in range(len(p)):
        if any((p[j] - p[k] <= w) and (h[k] >= a * h[j]) for k in range(j)): continue
        keep.append(j)
    return p[keep]
V = {}                                                    # 이름 → 파일별 특징값 (H1 후보)
for K in (2, 3, 4, 4.5, 6, 8, 10, 15, 20): V[f"A K={K}"] = np.array([np.log1p(len(pk_of(i, K))) for i in range(len(X))])
for s in (3, 4, 5, 6, 8, 10, 15): V[f"B 시그마 {s}"] = np.array([np.log1p(len(pk_of(i, sig=s))) for i in range(len(X))])
for W, a in ((20, 2), (30, 2), (50, 2), (30, 1), (100, 1)):
    for K in (4.5, 8):
        V[f"C K={K} 꼬리제거 {W}ms×{a}"] = np.array([np.log1p(len(dedupe(pk_of(i, K), RAT[i][pk_of(i, K)], W, a))) for i in range(len(X))])
for K in (4.5, 8):
    V[f"D K={K} 로그높이 합"] = np.array([np.log1p(np.log(RAT[i][pk_of(i, K)]).sum()) if len(pk_of(i, K)) else 0.0 for i in range(len(X))])
V["D 최대 로그높이"] = np.array([np.log(RAT[i][pk_of(i, 4.5)].max()) if len(pk_of(i, 4.5)) else 0.0 for i in range(len(X))])
# %% 지표
isN, isA, isV = grp == "정상", grp == "고장", grp == "val"
fx = list(ns.FIXED) + ["L1_S"]; B0 = ns.T[fx].values.copy(); i1 = fx.index("H1_nimp")
rows = []
def evaluate(name, v):
    af = roc_auc_score(np.r_[np.zeros(isN.sum()), np.ones(isA.sum())], np.r_[v[isN], v[isA]]); av = roc_auc_score(np.r_[np.zeros(isN.sum()), np.ones(isV.sum())], np.r_[v[isN], v[isV]])
    Xc = B0.copy(); Xc[:, i1] = v[:ntr]; r = quick_cv(Xc, y, ns.g, make_et(300), seeds=range(5))
    rows.append(dict(변형=name, 분리도=round(abs(af - .5) * 2, 2), 이동도=round(abs(av - .5) * 2, 2), 정상0개=round(float((v[isN] == 0).mean()), 2), 정상중앙=round(float(np.median(v[isN])), 2), 고장중앙=round(float(np.median(v[isA])), 2), val중앙=round(float(np.median(v[isV])), 2),
                     PR_AUC=round(float(r["pr_auc"].mean()), 3), sd=round(float(r["pr_auc"].std()), 3)))
    print(rows[-1], flush=True)
evaluate("기준: 현재 H1 (K=4.5)", H1)
for k, v in V.items(): evaluate(k, v)
Xd = np.delete(B0, i1, axis=1); r = quick_cv(Xd, y, ns.g, make_et(300), seeds=range(5)); rows.append(dict(변형="대조: H1 열 제거", PR_AUC=round(float(r["pr_auc"].mean()), 3), sd=round(float(r["pr_auc"].std()), 3)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "표_H1_설계.csv"), index=False, encoding="utf-8-sig")
md = lambda df: "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r_) + " |" for r_ in df.values)
open(os.path.join(HERE, "결과_H1_설계.md"), "w", encoding="utf-8").write(md(R)); print(md(R))
