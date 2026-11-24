# 충격으로 센 봉우리 중 '직전 큰 봉우리의 잔향 꼬리'로 보이는 것의 비율 (직전 30 ms 안에 2배 이상 큰 봉우리가 있는 경우) · H2의 2위가 1위 30 ms 안인 비율
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), encoding="utf-8").read().split("# %% 그림 1")[0])
W = int(0.030 * FS); tail_frac, h2near = [], []
for i, r in enumerate(RAT):
    pk = peaks(r); h = r[pk]
    if len(pk) == 0: tail_frac.append(np.nan); h2near.append(np.nan); continue
    tail = [any((pk[j] - pk[k] <= W) and (pk[j] > pk[k]) and (h[k] >= 2 * h[j]) for k in range(j)) for j in range(len(pk))]
    tail_frac.append(np.mean(tail))
    if len(pk) >= 2:
        o = np.argsort(h)[::-1]; h2near.append(float(abs(pk[o[0]] - pk[o[1]]) <= W))
    else: h2near.append(np.nan)
tf, hn = np.array(tail_frac), np.array(h2near)
for g in ("정상", "고장", "val"):
    m = grp == g; print(f"{g}: 잔향 꼬리로 보이는 충격 비율 중앙 {np.nanmedian(tf[m]):.2f} (평균 {np.nanmean(tf[m]):.2f}) · 2위가 1위 30 ms 안 {np.nanmean(hn[m]):.2f}")
print("전체 충격 수 중 꼬리 비율(파일 가중):", {g: round(float(np.nansum(tf[grp == g] * n0[grp == g]) / n0[grp == g].sum()), 2) for g in ("정상", "고장", "val")})
