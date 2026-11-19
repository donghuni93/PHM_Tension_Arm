# %% [markdown]
# 2-9 보조: "주변과 다른 유의미한 극댓값" (사용자 재정의)
# 봉우리가 있느냐가 아니라, 주변 봉우리들보다 유난히 솟은 극댓값이 있느냐. 정상의 봉우리는 주변에도 비슷한 봉우리가 있다
# 정의 (그림 보기 전에 고정, 주파수를 손으로 고르지 않음):
#  돌출 PR = log PSD 55 Hz 평활 − 1.2 kHz 이동 중앙값 (run.py와 같음)
#  2–12 kHz의 국소 최대(150 Hz 간격, PR ≥ 0.1) 각각에 대해 두드러짐 d = PR − (±1 kHz 안, ±150 Hz 밖 다른 봉우리 PR 최대)
#  파일 값 D = max d, D_f = 그 봉우리 위치
# 비교: ET가 쓰는 B04_ratio_08125 (8.125–8.75 kHz 에너지 비율)가 8.3형·8.6형을 가르는가
# 실행: conda run -n qe-phm python "단계/2_EDA/8kHz공진/2-9_1114_공진이동/distinct.py"

# %% 설정
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train, cached_table, feature_table_ext
import src.viz  # noqa: F401
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d
from sklearn.metrics import roc_auc_score

FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; N = len(y); nor = y == 0
Xc = X - X.mean(1, keepdims=True)
f, P = welch(Xc, fs=FS, nperseg=4096, detrend=False); df = f[1] - f[0]
Ls = uniform_filter1d(np.log10(P + 1e-30), 9, axis=1)
PR = Ls - median_filter(Ls, size=(1, 201), mode="nearest")
w = (f >= 2000) & (f <= 12000); fw = f[w]

D, Df, Dp = np.zeros(N), np.zeros(N), np.zeros(N)
for i in range(N):
    pr = PR[i, w]
    pk, _ = find_peaks(pr, height=0.1, distance=int(150 / df))
    best = (-9, 0, 0)
    for k in pk:
        near = pk[(np.abs(fw[pk] - fw[k]) <= 1000) & (np.abs(fw[pk] - fw[k]) > 150)]
        d = pr[k] - (pr[near].max() if len(near) else 0.0)
        if d > best[0]: best = (d, fw[k], pr[k])
    D[i], Df[i], Dp[i] = best

# %% 묶음 (모델 결과)
def counts(exp_dir):
    r = json.load(open(os.path.join(ROOT, "단계", "5_모델비교", exp_dir, "result.json"), encoding="utf-8"))
    c = next(iter(r["configs"].values())); return c["miss_counts"], c["fp_counts"]
aem, aef = counts("base/EXP-012_1113_스펙트럼AE_tSNE"); etm, etf = counts("기각/EXP-007_1113_공선성제거")
def tag(n):
    if n.startswith("abn"):
        t = "+".join(k for k, d in [("ET", etm), ("AE", aem)] if d.get(n, 0) >= 3); return "놓침 " + t if t else "잡힘"
    t = "+".join(k for k, d in [("ET", etf), ("AE", aef)] if d.get(n, 0) >= 3); return "오탐 " + t if t else "정상"
TAG = [tag(n) for n in names]

E = cached_table("ext", feature_table_ext, X); R = E["B04_ratio_08125"].values
pct = lambda v, i: (v[nor] < v[i]).mean() * 100

lines = [f"두드러짐 D (dec): 정상 중앙 {np.median(D[nor]):.2f} · 95% {np.percentile(D[nor],95):.2f} | 고장 중앙 {np.median(D[~nor]):.2f} | AUC {roc_auc_score(y, D):.2f}"]
p95 = np.percentile(D[nor], 95)
lines.append(f"정상 95% 넘는 고장 {int((D[~nor] > p95).sum())}/30: " + ", ".join(f"{names[i]} {D[i]:.2f}@{Df[i]/1000:.2f}k" for i in np.flatnonzero(~nor & (D > p95))))
lines.append(f"정상 95% 넘는 정상 {int((D[nor] > p95).sum())}/120: " + ", ".join(f"{names[i]} {D[i]:.2f}@{Df[i]/1000:.2f}k ({TAG[i]})" for i in np.flatnonzero(nor & (D > p95))))
lines.append("")
lines.append("놓친 고장·오탐 정상: D (정상 중 %) @ 위치 · 그 봉우리 돌출 | B04_ratio_08125 (정상 중 %)")
for i in range(N):
    if TAG[i].startswith(("놓침", "오탐")):
        lines.append(f"  {names[i]} [{TAG[i]}]: D {D[i]:.2f} ({pct(D,i):.0f}%) @ {Df[i]/1000:.2f}k · 돌출 {Dp[i]:.2f} | B04 {R[i]:.4f} ({pct(R,i):.0f}%)")
# B04_ratio_08125가 8.3형과 8.6형을 섞는지: 2-9 run.py의 봉우리 목록 사용
H = pd.read_csv(os.path.join(HERE, "figs", "humps.csv"))
best = H.loc[H.groupby("file").prom.idxmax()].set_index("file")
lines.append("")
lines.append("B04_ratio_08125 (8.125–8.75 kHz 에너지 비율) 상위 15 파일과 그 파일의 가장 높은 7.8–9.2 kHz 봉우리 위치")
for i in np.argsort(R)[::-1][:15]:
    b = best.loc[names[i]] if names[i] in best.index else None
    lines.append(f"  {names[i]} [{TAG[i]}] B04 {R[i]:.4f} · 봉우리 {'' if b is None else f'{b.f_Hz/1000:.2f}k ({b.prom:.2f})'} · D {D[i]:.2f}@{Df[i]/1000:.2f}k")
open(os.path.join(HERE, "figs", "distinct_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))
pd.DataFrame(dict(file=names, y=y, tag=TAG, D=D, D_f=Df, D_prom=Dp, B04_08125=R)).to_csv(
    os.path.join(HERE, "figs", "distinct_per_file.csv"), index=False, encoding="utf-8-sig")

# %% 그림: 왼쪽 = D 위치 vs D (모든 파일), 오른쪽 = 002·029 vs 8.6형 정상 2개의 돌출 곡선
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5))
a = ax[0]
for g, c, lab in [(0, "#2a78d6", "정상"), (1, "#eb6834", "고장")]:
    m = y == g; a.scatter(Df[m] / 1000, D[m], s=26, color=c, alpha=0.75, edgecolors="white", linewidths=1, label=lab)
a.axhline(p95, color="#85847e", lw=0.8, ls=":"); a.text(12, p95, " 정상 95%", va="bottom", ha="right", fontsize=8, color="#85847e")
for i in range(N):
    if TAG[i].startswith(("놓침 ET", "오탐 ET+AE")) or D[i] > p95:
        a.annotate(names[i][:3] + names[i][4:], (Df[i] / 1000, D[i]), xytext=(4, 3), textcoords="offset points", fontsize=7, color="#52514e")
a.set_xlabel("가장 두드러진 봉우리 위치 kHz"); a.set_ylabel("두드러짐 D (dec) = 봉우리 − 주변 ±1 kHz 다른 봉우리 최대")
a.set_title("파일마다 '주변보다 유난히 솟은 봉우리' 하나 (2–12 kHz)", loc="left", fontsize=11); a.legend(frameon=False)
a.grid(color="#e6e5e0", lw=0.6); a.spines[["top", "right"]].set_visible(False)
a = ax[1]; ww = (f >= 6000) & (f <= 11000)
# 비교용 정상: 7.8–9.2 kHz 가장 높은 봉우리가 8.45–8.75 kHz이고 돌출이 큰 정상 2개
cand = best[(best.f_Hz >= 8450) & (best.f_Hz < 8750) & best.index.str.startswith("nor")].sort_values("prom", ascending=False).index[:2]
for n, c, lw in [("abn_002", "#eb6834", 1.8), ("abn_029", "#c4471c", 1.8)] + [(n, "#2a78d6", 1.2) for n in cand]:
    i = names.index(n); a.plot(f[ww] / 1000, PR[i, ww], color=c, lw=lw, label=f"{n} D {D[i]:.2f}")
a.axvspan(8.125, 8.75, color="#f0efec", zorder=0); a.text(8.44, -0.95, "B04_ratio_08125 칸", ha="center", fontsize=8, color="#85847e")
a.set_xlabel("kHz"); a.set_ylabel("돌출 (dec)"); a.set_ylim(-1.1, 1.5)
a.set_title("ET 미탐 002·029 vs 8.6 kHz 봉우리 정상: 같은 칸, 다른 모양", loc="left", fontsize=11); a.legend(frameon=False, fontsize=8)
a.grid(color="#e6e5e0", lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "distinct.png"), dpi=120)
print("done")
