# %% H2 재설계 후보 (반복 충격을 재는 피처): 독립 충격(30 ms 안 뒤 봉우리 병합) 기준으로 C1 큰 충격 개수 · C2 2위÷1위 · C3 상위 3개 크기 균일도 · C4 포락선 자기상관(반복 주기성)
# 지표: 전체 분리도 · n≥2 파일만의 분리도(현재 H2의 약점) · val 이동도 · quick_cv PR-AUC (기준 ET에서 H2 열만 교체)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from src.exp import make_et
from src.evaluate import quick_cv
HERE = os.path.dirname(os.path.abspath(__file__))
exec(open(os.path.join(HERE, "run.py"), encoding="utf-8").read().split("# %% 그림 1")[0])
W = int(0.030 * FS)
def indep(i):
    p = peaks(RAT[i]); h = RAT[i][p]; keep = [j for j in range(len(p)) if not any(p[j] - p[k] <= W for k in range(j))]; return p[keep], h[keep]
IM = [indep(i) for i in range(len(X))]
C = {k: np.zeros(len(X)) for k in ("C1 큰 충격 개수(≥0.5×최대)", "C2 2위÷1위(독립 충격)", "C3 상위3 크기 균일도(min÷max)", "C4 포락선 자기상관 최대(30~1000 ms)", "C5 큰 충격 개수(≥0.3×최대, 높이 ≥ 10)")}
ks = list(C)
for i, (p, h) in enumerate(IM):
    if len(h):
        s = np.sort(h)[::-1]; C[ks[0]][i] = np.log1p((h >= 0.5 * s[0]).sum()); C[ks[4]][i] = np.log1p(((h >= 0.3 * s[0]) & (h >= 10)).sum())
        if len(s) >= 2: C[ks[1]][i] = s[1] / s[0]
        if len(s) >= 3: C[ks[2]][i] = s[2] / s[0]
    r = RAT[i][::25] - RAT[i][::25].mean(); n = len(r); f = np.fft.rfft(r, 2 * n); ac = np.fft.irfft(f * np.conj(f))[:n]; ac = ac / ac[0]; C[ks[3]][i] = ac[30:1000].max()
isN, isA, isV = grp == "정상", grp == "고장", grp == "val"; n_ind = np.array([len(h) for _, h in IM]); sub = n_ind >= 2
fx = list(ns.FIXED) + ["L1_S"]; B0 = ns.T[fx].values.copy(); i2 = fx.index("H2_ratio21"); rows = []
def ev(name, v):
    a = lambda m1, m2: roc_auc_score(np.r_[np.zeros(m1.sum()), np.ones(m2.sum())], np.r_[v[m1], v[m2]])
    af, afs, av = a(isN, isA), a(isN & sub, isA & sub), a(isN, isV)
    Xc = B0.copy(); Xc[:, i2] = v[:ntr]; r = quick_cv(Xc, y, ns.g, make_et(300), seeds=range(5))
    rows.append(dict(후보=name, 분리도=round(abs(af - .5) * 2, 2), 분리도_n2이상=round(abs(afs - .5) * 2, 2), 이동도=round(abs(av - .5) * 2, 2), 정상중앙=round(float(np.median(v[isN])), 2), 고장중앙=round(float(np.median(v[isA])), 2), val중앙=round(float(np.median(v[isV])), 2), PR_AUC=round(float(r["pr_auc"].mean()), 3), sd=round(float(r["pr_auc"].std()), 3))); print(rows[-1], flush=True)
ev("기준: 현재 H2", H2)
for k in ks: ev(k, C[k])
r = quick_cv(np.delete(B0, i2, axis=1), y, ns.g, make_et(300), seeds=range(5)); rows.append(dict(후보="대조: H2 열 제거", PR_AUC=round(float(r["pr_auc"].mean()), 3), sd=round(float(r["pr_auc"].std()), 3)))
R = pd.DataFrame(rows); R.to_csv(os.path.join(HERE, "표_H2_설계.csv"), index=False, encoding="utf-8-sig")
md = lambda df: "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join("" if (isinstance(x, float) and np.isnan(x)) else str(x) for x in r_) + " |" for r_ in df.values)
open(os.path.join(HERE, "결과_H2_설계.md"), "w", encoding="utf-8").write(md(R)); print(md(R))
