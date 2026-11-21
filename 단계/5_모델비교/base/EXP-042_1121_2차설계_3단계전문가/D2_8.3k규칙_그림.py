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

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
W = json.load(open(os.path.join(HERE, "..", "EXP-041_1121_2차설계_2단계결합", "result.json"), encoding="utf-8"))["notes"]["선택"]["winner"]
fo = oof_fires(ns, ns.mk(W, ("E1", "E2")), seed=0)
a, z = fo["a"], np.clip(fo["z"], -3, 20); nor = y == 0
d2 = fo["main"] | fo["E1"] | fo["E2"]; rule = spec | res
fig, axs = plt.subplots(1, 2, figsize=(22, 9))
ax = axs[0]
ax.scatter(a[nor & ~rule], z[nor & ~rule], color=NOR, s=40, marker="s", lw=0, alpha=.7, label="정상")
ax.scatter(a[~nor & ~rule], z[~nor & ~rule], color=ABN, s=60, lw=0, label="고장 (규칙 안 켜짐)")
ax.scatter(a[~nor & rule], z[~nor & rule], s=170, marker="D", facecolor=ABN, edgecolor="k", lw=1.4, label="고장 (8.3k 규칙이 켬)")
ax.scatter(a[nor & fo["E1"]], z[nor & fo["E1"]], s=130, marker="o", facecolor="none", edgecolor="#2ca02c", lw=1.5, label="정상 오경보: E1")
ax.set_yscale("symlog", linthresh=1.0); ax.axvline(-.115, color="k", ls=":", lw=1)
for i in np.flatnonzero(~nor & rule & ~d2):
    ax.annotate(short(names[i]) + " (규칙만 잡음)", (a[i], z[i]), xytext=(10, 18), textcoords="offset points", fontsize=11, weight="bold", color=ABN, arrowprops=dict(arrowstyle="-", color="#888"))
ax.set_xlabel("ET 점수 (logit, OOF seed 0)"); ax.set_ylabel("박자 z (symlog)"); ax.grid(alpha=.3); ax.legend(fontsize=10, loc="upper left")
ax.set_title("D2 평면 + 8.3k 규칙이 켠 고장 (◆)", fontsize=12)
ax = axs[1]
ax.scatter(W83[nor], P83[nor], color=NOR, s=40, marker="s", lw=0, alpha=.7, label="정상 120"); ax.scatter(W83[~nor], P83[~nor], color=ABN, s=60, lw=0, label="고장 30")
ax.axhline(P83_MIN, color="#7a3db8", ls="--"); ax.axvline(W83_MIN, color="#7a3db8", ls="--", label=f"스펙트럼 규칙 (돌출 ≥ {P83_MIN} · 폭 ≥ {W83_MIN:.0f} Hz)")
ax.scatter(W83[res], P83[res], s=200, marker="o", facecolor="none", edgecolor="k", lw=1.4, label="공진 규칙(충격 fit ≥ 0.7)이 켠 파일")
for i in np.flatnonzero(spec | res | ((P83 > .5) & ~nor) | ((P83 > .6) & nor)):
    ax.annotate(short(names[i]), (W83[i], P83[i]), xytext=(6, 6), textcoords="offset points", fontsize=9, color=ABN if y[i] else NOR, weight="bold")
ax.set_xlabel("W83 반높이 폭 (Hz)"); ax.set_ylabel("P83 8.2–8.45 kHz 봉우리 돌출 (dec)"); ax.grid(alpha=.3); ax.legend(fontsize=10, loc="upper left")
ax.set_title("8.3k 규칙의 두 축 — 규칙이 켜진 고장 / 정상 0개", fontsize=12)
fig.suptitle("EXP-042 D2 + 8.3k 규칙 (train 중첩 F2 0.916 → 0.942, 재현율 0.960 → 0.993)", x=.01, ha="left", fontsize=13); fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(os.path.join(HERE, "figs", "D2_8.3k규칙.png"), dpi=120)
print("규칙만 잡은 고장(D2 OOF seed0 기준):", [names[i] for i in np.flatnonzero(~nor & rule & ~d2)])
