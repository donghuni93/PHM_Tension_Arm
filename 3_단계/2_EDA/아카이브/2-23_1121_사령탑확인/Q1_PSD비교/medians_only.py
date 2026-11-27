# Q1 보충: 중앙값만 보기 + 나이퀴스트(12.5 kHz) 부근 확대 (겹침/거울 반사 가설 점검용)
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.exp import load_train
from src.data import load_folder
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FS = 25000; NOR, ABN, VAL = "#2a78d6", "#eb6834", "#2a9d5c"
X, y, names = load_train(); y = np.asarray(y)
XV, vn = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
out = []
def lp(S, n):
    S = S - S.mean(1, keepdims=True)
    f, P = welch(S, fs=FS, nperseg=n, noverlap=n // 2, window="hann", axis=-1)
    return f, np.log10(P + 1e-20)
fig, ax = plt.subplots(3, 1, figsize=(11, 11))
for r, (n, lo, hi, step) in enumerate(((2048, 1000, 12500, 500), (8192, 9500, 12500, 250))):
    f, Lt = lp(X, n); _, Lv = lp(XV, n)
    mn, ma, mv = np.median(Lt[y == 0], 0), np.median(Lt[y == 1], 0), np.median(Lv, 0)
    m = (f >= lo) & (f <= hi)
    a = ax[r if r == 0 else 1]
    a.plot(f[m], mn[m], color=NOR, lw=1.6, label="정상 중앙 (120)")
    a.plot(f[m], ma[m], color=ABN, lw=1.6, label="고장 중앙 (30)")
    a.plot(f[m], mv[m], color=VAL, lw=1.6, ls="--", label="val 중앙 (40)")
    a.axvline(12500, color="k", lw=1, ls="--"); a.text(12500, a.get_ylim()[1], " 나이퀴스트 12.5k", va="top", ha="right", fontsize=8)
    a.set_xticks(np.arange(lo, hi + 1, step)); a.tick_params(axis="x", labelrotation=90, labelsize=8)
    a.set_xlim(lo, hi); a.grid(alpha=.3); a.legend(); a.set_ylabel("log10 PSD")
    a.set_title(f"중앙값만 (Welch {n}점, 선형 x) {lo/1000:g}–{hi/1000:g} kHz")
    if r == 1:
        a3 = ax[2]
        a3.plot(f[m], (ma - mn)[m], color=ABN, label="고장 중앙 - 정상 중앙")
        a3.plot(f[m], (mv - mn)[m], color=VAL, ls="--", label="val 중앙 - 정상 중앙")
        a3.axhline(0, color="k", lw=.8); a3.set_xticks(np.arange(lo, hi + 1, step)); a3.tick_params(axis="x", labelrotation=90, labelsize=8)
        a3.set_xlim(lo, hi); a3.grid(alpha=.3); a3.legend(); a3.set_ylabel("차이 (dec)"); a3.set_xlabel("주파수 (Hz)")
        for hz in (10000, 10500, 11000, 11500, 12000, 12400):
            i = np.argmin(np.abs(f - hz))
            out.append(f"{hz} Hz: 정상 {mn[i]:.2f}, 고장 {ma[i]:.2f}, val {mv[i]:.2f}, 고장-정상 {ma[i]-mn[i]:+.2f}, val-정상 {mv[i]-mn[i]:+.2f}")
        d = (ma - mn)[m]; ff = f[m]
        out.append(f"차이 최소: {ff[np.argmin(d)]:.0f} Hz ({d.min():+.2f} dec); 11.5k 이후 최대: {ff[ff>=11500][np.argmax(d[ff>=11500])]:.0f} Hz ({d[ff>=11500].max():+.2f})")
        sl = lambda arr, lo2, hi2: arr[(f >= hi2 - 100) & (f <= hi2)].mean() - arr[(f >= lo2) & (f <= lo2 + 100)].mean()
        out.append(f"11.0k→12.4k 로그 파워 변화: 정상 {sl(mn,11000,12400):+.2f}, 고장 {sl(ma,11000,12400):+.2f}, val {sl(mv,11000,12400):+.2f}")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "q1_medians_only.png"), dpi=90); plt.close(fig)
open(os.path.join(HERE, "result_medians.txt"), "w", encoding="utf-8").write(chr(10).join(out))
print("done")
