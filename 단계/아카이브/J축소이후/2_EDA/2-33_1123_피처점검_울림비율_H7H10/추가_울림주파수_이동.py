# 충격 단위 울림 주파수 격자(250 Hz) 분포: 정상 주봉우리 7–7.5 kHz가 val에서 어디로 가는가 · 2 kHz 군집이 옥타브 경계(2000 Hz)에 걸치는가
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py"), encoding="utf-8").read().split("# %% 그림 1")[0])
F = {g: np.concatenate([IM[i]["rings"][:, 0] for i in np.flatnonzero(grp == g) if n_imp[i]]) for g in ("정상", "고장", "val")}
grid = np.arange(1500, 3001, 250)
print("2 kHz 근방 격자별 충격 비율(%) — 옥타브 경계 2000 Hz: 1750 = H7(1–2k), 2000 이상 = H8(2–4k)")
print(pd.DataFrame({g: [round(100 * float((F[g] == f).mean()), 1) for f in grid] for g in F}, index=grid).T.to_string())
grid2 = np.arange(5500, 8251, 250)
print("\n5.5–8 kHz 격자별 충격 비율(%)")
print(pd.DataFrame({g: [round(100 * float((F[g] == f).mean()), 1) for f in grid2] for g in F}, index=grid2).T.to_string())
med = {g: np.array([np.median(IM[i]["rings"][:, 0][(IM[i]["rings"][:, 0] >= 5000) & (IM[i]["rings"][:, 0] <= 8000)]) for i in np.flatnonzero(grp == g) if n_imp[i] and ((IM[i]["rings"][:, 0] >= 5000) & (IM[i]["rings"][:, 0] <= 8000)).any()]) for g in ("정상", "고장", "val")}
print("\n파일별 5–8 kHz 충격의 울림 주파수 중앙값: " + " · ".join(f"{g} n={len(v)} 중앙 {np.median(v):.0f} Hz (25~75%: {np.percentile(v,25):.0f}~{np.percentile(v,75):.0f})" for g, v in med.items()))
# 2 kHz(1750~2250 Hz) 울림 충격 비율이 30% 이상인 파일
fr = np.array([float(np.mean((IM[i]["rings"][:, 0] >= 1750) & (IM[i]["rings"][:, 0] <= 2250))) if n_imp[i] else 0.0 for i in range(len(X))])
for g in ("정상", "고장", "val"):
    m = (grp == g) & (fr >= 0.3) & (n_imp >= 2); print(f"{g}: 1.75~2.25 kHz 울림 ≥ 30% 파일 {int(m.sum())}개 / {int((grp == g).sum())}", [allnames[i] for i in np.flatnonzero(m)][:20])
f7 = np.array([float(np.mean((IM[i]["rings"][:, 0] >= 7000) & (IM[i]["rings"][:, 0] <= 7500))) if n_imp[i] else 0.0 for i in range(len(X))])
for g in ("정상", "고장", "val"):
    m = (grp == g) & (n_imp >= 2); print(f"{g}: 7.0~7.5 kHz 울림 ≥ 50% 파일 {int(((f7 >= .5) & m).sum())}개 / 충격 ≥ 2개 {int(m.sum())}개")
