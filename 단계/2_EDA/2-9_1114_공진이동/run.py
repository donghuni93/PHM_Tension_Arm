# %% [markdown]
# 2-9 공진 이동 가설 — 정상의 8.5–8.7 kHz 봉우리와 고장의 8.2–8.4 kHz 봉우리는 같은 공진이 왼쪽으로 옮겨 간 것인가
# 가설(사용자, H2 구체화): 같은 모드인데 고장(강성 K↓)으로 주파수가 내려갔다. 8.6 → 8.3 kHz = −3.5% → K 약 −7%
# 같은 모드의 이동이라면: ① 한 파일에 봉우리가 하나만 ② 봉우리 모양(폭·높이)이 비슷 ③ 위치가 정상 8.6 / 고장 8.3으로 갈림
# 봉우리 정의 (그림 보기 전에 고정):
#  PSD = 평균 제거 Welch 4096 (6.1 Hz 칸, detrend 없음) → log10 → 9칸(≈55 Hz) 이동평균
#  바탕 = 201칸(≈1.2 kHz) 이동 중앙값. 돌출 = 평활 log − 바탕
#  7.8–9.2 kHz에서 돌출 ≥ 0.3 dec 인 국소 최대 (서로 150 Hz 이상 떨어짐) = 봉우리. 반높이 폭 = 돌출이 (최고 − 0.301) 이상인 연속 폭 (파워 반값)
# 실행: conda run -n qe-phm python "단계/2_EDA/2-9_1114_공진이동/run.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train
import src.viz  # noqa: F401
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d

FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; N = len(y); nor = y == 0
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
Xc = X - X.mean(1, keepdims=True)
f, P = welch(Xc, fs=FS, nperseg=4096, detrend=False); df = f[1] - f[0]
Ls = uniform_filter1d(np.log10(P + 1e-30), 9, axis=1)
PR = Ls - median_filter(Ls, size=(1, 201), mode="nearest")
LO, HI, TH = 7800, 9200, 0.3
w = (f >= LO) & (f <= HI); fw = f[w]

rows = []
for i in range(N):
    pr = PR[i, w]
    pk, _ = find_peaks(pr, height=TH, distance=int(150 / df))
    for k in pk:
        a = k
        while a > 0 and pr[a - 1] >= pr[k] - 0.301: a -= 1
        b = k
        while b < len(pr) - 1 and pr[b + 1] >= pr[k] - 0.301: b += 1
        rows.append(dict(file=names[i], y=int(y[i]), f_Hz=float(fw[k]), prom=float(pr[k]), fwhm_Hz=float((b - a + 1) * df),
                         Q=float(fw[k] / ((b - a + 1) * df))))
T = pd.DataFrame(rows)
T.to_csv(os.path.join(HERE, "figs", "humps.csv"), index=False, encoding="utf-8-sig")

# %% 요약
lines = [f"봉우리 정의: 7.8–9.2 kHz, 돌출 ≥ {TH} dec (≈2배), 바탕 = 1.2 kHz 이동 중앙값"]
for lab, g in [("정상", 0), ("고장", 1)]:
    sub = T[T.y == g]; files = sub.file.nunique(); tot = int((y == g).sum())
    nper = sub.groupby("file").size()
    lines.append(f"{lab}: 봉우리 있는 파일 {files}/{tot} ({files/tot*100:.0f}%), 그중 봉우리 2개 이상 {int((nper >= 2).sum())}개")
BIN = [(7800, 8200), (8200, 8450), (8450, 8750), (8750, 9200)]
lines.append("위치 구간별 봉우리 수 (정상 / 고장) · 돌출 중앙 · 반높이 폭 중앙 · Q 중앙")
for lo, hi in BIN:
    s = T[(T.f_Hz >= lo) & (T.f_Hz < hi)]
    for lab, g in [("정상", 0), ("고장", 1)]:
        t = s[s.y == g]
        if len(t):
            lines.append(f"  {lo/1000:.2f}–{hi/1000:.2f}k {lab}: {len(t)}개 ({t.file.nunique()}파일) · 돌출 {t.prom.median():.2f} · 폭 {t.fwhm_Hz.median():.0f} Hz · Q {t.Q.median():.0f}")
        else:
            lines.append(f"  {lo/1000:.2f}–{hi/1000:.2f}k {lab}: 0")
# 공존: 같은 파일에 8.2–8.45 와 8.45–8.75 봉우리가 함께 있나
A = set(T[(T.f_Hz >= 8200) & (T.f_Hz < 8450)].file); B = set(T[(T.f_Hz >= 8450) & (T.f_Hz < 8750)].file)
lines.append(f"8.20–8.45k 봉우리 파일: {len(A)}개 (고장 {sum(n.startswith('abn') for n in A)}) · 8.45–8.75k: {len(B)}개 (고장 {sum(n.startswith('abn') for n in B)}) · 둘 다: {sorted(A & B)}")
lines.append("8.20–8.45k 봉우리 파일 목록: " + ", ".join(sorted(A)))
lines.append("8.45–8.75k 봉우리 파일 목록: " + ", ".join(sorted(B)))
# 파일별 대표 봉우리(가장 높은 것)
best = T.loc[T.groupby("file").prom.idxmax()]
lines.append("")
lines.append("파일별 가장 높은 봉우리 위치 분포 (100 Hz 칸): 칸 | 정상 | 고장")
for lo in range(7800, 9200, 100):
    s = best[(best.f_Hz >= lo) & (best.f_Hz < lo + 100)]
    lines.append(f"  {lo/1000:.1f}–{(lo+100)/1000:.1f}k | {'■' * int((s.y == 0).sum()):<14} {int((s.y==0).sum())} | {'■' * int((s.y == 1).sum())} {int((s.y==1).sum())}")
open(os.path.join(HERE, "figs", "hump_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))

# %% 그림: 왼쪽 = 가장 높은 봉우리 위치 vs 돌출 (점), 오른쪽 = 봉우리 위치별 대표 파일들의 돌출 곡선 겹침
fig, ax = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[1, 1.3]))
a = ax[0]
for g, c, lab in [(0, "#2a78d6", "정상"), (1, "#eb6834", "고장")]:
    s = T[T.y == g]
    a.scatter(s.f_Hz / 1000, s.prom, s=18 + 0 * s.prom, color=c, alpha=0.75, edgecolors="white", linewidths=1, label=f"{lab} 봉우리")
for n in ["abn_002", "abn_029"]:
    s = T[T.file == n]
    for _, r in s.iterrows():
        a.annotate(n[4:], (r.f_Hz / 1000, r.prom), xytext=(4, 4), textcoords="offset points", fontsize=8, color="#52514e")
a.axvspan(8.2, 8.45, color="#f4b391", alpha=0.25, lw=0); a.axvspan(8.45, 8.75, color="#9ec5f4", alpha=0.25, lw=0)
a.set_xlabel("봉우리 위치 kHz"); a.set_ylabel("돌출 (dec, 1.2 kHz 바탕 대비)")
a.set_title("7.8–9.2 kHz 봉우리 전부 (파일당 여러 개 가능)", loc="left", fontsize=11); a.legend(frameon=False, fontsize=9)
a.grid(color="#e6e5e0", lw=0.6); a.spines[["top", "right"]].set_visible(False)
a = ax[1]; ww = (f >= 7500) & (f <= 9500)
for fl, c in [(sorted(A - B), "#eb6834"), (sorted(B - A), "#2a78d6"), (sorted(A & B), "#0b0b0b")]:
    for n in fl:
        i = names.index(n)
        a.plot(f[ww] / 1000, PR[i, ww], color=c if not (c != "#0b0b0b" and n.startswith("abn") != (c == "#eb6834")) else "#85847e",
               lw=1.6 if n.startswith("abn") else 0.9, alpha=0.9 if n.startswith("abn") else 0.6)
a.axvspan(8.2, 8.45, color="#f4b391", alpha=0.25, lw=0); a.axvspan(8.45, 8.75, color="#9ec5f4", alpha=0.25, lw=0)
a.axhline(TH, color="#85847e", lw=0.8, ls=":")
a.set_title("돌출 곡선: 주황 = 8.20–8.45k에만 봉우리 · 파랑 = 8.45–8.75k에만 · 검정 = 둘 다\n(굵은 선 = 고장, 회색 = 구간 색과 라벨이 다른 파일)", loc="left", fontsize=10)
a.set_xlabel("kHz"); a.set_ylabel("돌출 (dec)"); a.grid(color="#e6e5e0", lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "hump_shift.png"), dpi=120)
print("done")
