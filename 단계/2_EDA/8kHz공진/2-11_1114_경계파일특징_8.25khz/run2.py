# %% [markdown]
# 2-11 README2 — 8–9 kHz 봉우리만 본다 (사용자 아이디어 11-14)
# - 생각: abn_002 등은 이제 막 나기 시작한 조용한 고장이고, 잡으려면 8.25 kHz를 써야 하는데 지금 모델은 제대로 못 쓴다
# - 그림: 8–9 kHz만. 이 대역 안에 봉우리가 있는 파일만 골라 겹쳐 그린다
# - 봉우리 정의 (2-9와 같음, 라벨 안 씀): log10 Welch PSD(평균 제거, nperseg 8192) → 55 Hz 이동평균 평활
#   → 돌출 = 평활 − 1.2 kHz 이동 중앙값 → 돌출 ≥ 0.3 dec(약 2배)인 국소 최대, 서로 150 Hz 이상 떨어짐
# 실행: conda run -n qe-phm python "단계/2_EDA/8kHz공진/2-11_1114_경계파일특징_8.25khz/run2.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); names = [str(n) for n in names]
NOR, ABN = "#2a78d6", "#eb6834"
FOCUS = ["abn_002", "abn_029", "abn_009"]        # 사용자 언급 002·009 + 지금까지 논의한 029
FC = {"abn_002": "#7a3fbf", "abn_029": "#3fa34d", "abn_009": "#c9a227"}
LO, HI = 8000, 9000

f, L = viz.log_psd(X)                            # (150, 4097), 3.05 Hz 간격
df = f[1] - f[0]
S = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)               # 55 Hz 평활
PR = S - median_filter(S, size=(1, int(round(1200 / df))), mode="nearest")  # 돌출 (dec)
band = (f >= LO) & (f <= HI)

peaks = {}
for i in range(len(y)):
    k, _ = find_peaks(PR[i], height=0.3, distance=max(1, int(round(150 / df))))
    k = k[(f[k] >= LO) & (f[k] <= HI)]
    if len(k):
        peaks[i] = [(float(f[j]), float(PR[i, j])) for j in k]
has = np.array([i in peaks for i in range(len(y))])
nor_p, abn_p = np.flatnonzero(has & (y == 0)), np.flatnonzero(has & (y == 1))
lines = [f"8–9 kHz에 봉우리(돌출 ≥ 0.3 dec)가 있는 파일: 정상 {len(nor_p)}/120 · 고장 {len(abn_p)}/30"]
for i in np.flatnonzero(has):
    lines.append(f"  {names[i]}: " + " · ".join(f"{fr/1000:.3f} kHz ({p:.2f} dec)" for fr, p in peaks[i]))
for n in FOCUS:
    i = names.index(n)
    lines.append(f"[강조] {n}: " + ("봉우리 " + " · ".join(f"{fr/1000:.3f} kHz ({p:.2f})" for fr, p in peaks[i]) if has[i]
                                   else f"8–9 kHz 봉우리 없음 (이 대역 최대 돌출 {PR[i, band].max():.2f} dec @ {f[band][PR[i, band].argmax()]/1000:.3f} kHz)"))
print("\n".join(lines))


# %% 그림: 2 × 2
fig, axs = plt.subplots(2, 2, figsize=(17, 11), sharex=True)
fb = f[band]
nb = np.percentile(L[y == 0][:, band], [5, 95], axis=0)
for a, idx, col, lab in ((axs[0, 0], nor_p, NOR, "정상"), (axs[0, 1], abn_p, ABN, "고장")):
    a.fill_between(fb, nb[0], nb[1], color="#9cc3ec", alpha=.35, label="정상 전체 5–95%")
    for i in idx:
        if names[i] in FOCUS: continue
        a.plot(fb, S[i, band], color=col, lw=.8, alpha=.55)
    for n in FOCUS:
        i = names.index(n)
        if i in idx:
            a.plot(fb, S[i, band], color=FC[n], lw=2.2, label=n)
    a.plot([], [], color=col, lw=.8, label=f"{lab} 중 봉우리 있는 파일 ({len(idx)})")
    a.set_title(f"{lab}: 8–9 kHz에 봉우리가 있는 파일만 (log PSD, 55 Hz 평활)", fontsize=10)
    a.set_ylabel("log10 PSD"); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=.3)
for a, idx, col, lab in ((axs[1, 0], nor_p, NOR, "정상"), (axs[1, 1], abn_p, ABN, "고장")):
    for i in idx:
        if names[i] in FOCUS: continue
        a.plot(fb, PR[i, band], color=col, lw=.8, alpha=.55)
    for n in FOCUS:
        i = names.index(n)
        a.plot(fb, PR[i, band], color=FC[n], lw=2.2 if i in idx else 1.0, ls="-" if i in idx else ":",
               label=n + ("" if i in idx else " (봉우리 없음, 참고)"))
    for i in idx:
        for fr, p in peaks[i]:
            a.plot(fr, p, "v", color=FC.get(names[i], col), ms=5)
    a.axhline(.3, color="k", lw=.6, ls="--", label="봉우리 문턱 0.3 dec")
    a.axvspan(8200, 8450, color="#eb6834", alpha=.06); a.axvspan(8450, 8750, color="#2a78d6", alpha=.06)
    a.set_title(f"{lab}: 돌출 (주변 1.2 kHz 대비 얼마나 솟았나). 주황 띠 8.20–8.45 · 파랑 띠 8.45–8.75 kHz", fontsize=10)
    a.set_ylabel("돌출 (dec)"); a.set_xlabel("Hz"); a.legend(fontsize=8, loc="upper right"); a.grid(alpha=.3)
ymax = max(axs[1, 0].get_ylim()[1], axs[1, 1].get_ylim()[1])
for a in axs[1]: a.set_ylim(-.6, ymax)
y0 = (min(axs[0, 0].get_ylim()[0], axs[0, 1].get_ylim()[0]), max(axs[0, 0].get_ylim()[1], axs[0, 1].get_ylim()[1]))
for a in axs[0]: a.set_ylim(*y0)
axs[1, 0].set_xlim(LO, HI)
fig.suptitle(f"8–9 kHz 봉우리가 있는 파일만 겹침 — 정상 {len(nor_p)}/120, 고장 {len(abn_p)}/30. 굵은 선 = 002·029·009", fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "r2_8k_peaks.png"), dpi=80); plt.close(fig)

# 봉우리 위치 vs 돌출 (파일 하나에 봉우리가 여러 개면 모두)
fig, ax = plt.subplots(figsize=(11, 5.5))
for i in np.flatnonzero(has):
    for fr, p in peaks[i]:
        c = FC.get(names[i], ABN if y[i] else NOR)
        ax.scatter(fr, p, color=c, s=60 if names[i] in FOCUS else 26, alpha=.9 if y[i] else .6, lw=0,
                   marker="o" if y[i] else "s")
        if y[i] or p > 0.9:
            ax.annotate(names[i].replace("abn_", "A").replace("nor_", "N"), (fr, p), xytext=(3, 3), textcoords="offset points",
                        fontsize=7, color=c)
ax.scatter([], [], color=NOR, marker="s", label="정상 봉우리"); ax.scatter([], [], color=ABN, label="고장 봉우리")
ax.axvspan(8200, 8450, color="#eb6834", alpha=.07); ax.axvspan(8450, 8750, color="#2a78d6", alpha=.07)
ax.set_xlim(LO, HI); ax.set_xlabel("봉우리 위치 (Hz)"); ax.set_ylabel("돌출 (dec)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("8–9 kHz 봉우리의 위치와 높이 — 고장 봉우리는 어디에 모이나", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "r2_8k_peak_scatter.png"), dpi=80); plt.close(fig)

open(os.path.join(FIG, "r2_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("done")
