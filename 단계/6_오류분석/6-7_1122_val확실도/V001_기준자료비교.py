# V001을 8.3k 규칙의 기준 자료 위에 올린다 (11-22)
#  ① 2-11 r2_8k_peak_scatter와 같은 봉우리 정의(Welch 8192 · 평균 제거 → log PSD 55 Hz 평활 − 주변 1.2 kHz 이동 중앙값 = 돌출, 돌출 ≥ 0.3 dec 국소 최대, 150 Hz 간격)로
#     train 정상 · 고장 + V001의 8–9 kHz 봉우리 위치 × 돌출
#  ② V001 지속음의 시간별 주파수(0.1 s STFT, 7.8–9.0 kHz 최대) — 8200–8450(고장 쪽 구간) · 8450–8750(정상 큰 봉우리 구간)을 언제 지나나. 비교: 8.6 kHz 정상 N045 · N009
#  ③ 2-16 · 신효섭 그림 3의 근거(충격 울림): V001의 충격 수
# 실행: conda run -n qe-phm python V001_기준자료비교.py
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = HERE
while not os.path.isdir(os.path.join(R, "src")) and os.path.dirname(R) != R:
    R = os.path.dirname(R)
sys.path.insert(0, R)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.signal import welch, stft, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d
from src.exp2util import NS
from src.impacts import all_impacts
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
FS = 25000; NOR, ABN, VAL, INK, MUTED = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575"
ns = NS(); y = ns.y; names = list(ns.names); nv = [v[-3:] for v in ns.nv]; kV = nv.index("001")


def prom(X):
    X = np.asarray(X, float); Xc = X - X.mean(1, keepdims=True)
    f, P = welch(Xc, fs=FS, nperseg=8192); L = np.log10(P); df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
    return f, Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest"), df


f, PRt, df = prom(ns.X); _, PRv, _ = prom(ns.Xv)
def peaks(pr):
    pk, _ = find_peaks(pr, height=.3, distance=max(1, int(round(150 / df)))); pk = pk[(f[pk] >= 8000) & (f[pk] <= 9000)]
    return f[pk], pr[pk]


def track(x):
    ff, tt, Z = stft(x - x.mean(), fs=FS, nperseg=2500, noverlap=1250)
    L = 10 * np.log10(np.abs(Z) ** 2 + 1e-20); base = median_filter(L, size=(41, 1), mode="nearest"); b = (ff >= 7800) & (ff <= 9000)
    P = (L - base)[b]; return tt, ff[b][P.argmax(0)], P.max(0)


fig, axs = plt.subplots(1, 2, figsize=(17, 5.8), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]
for lo, hi, c in ((8200, 8450, "#fbe0d0"), (8450, 8750, "#d6e6fa")): ax.axvspan(lo, hi, color=c, alpha=.7, zorder=0)
for i in range(len(y)):
    fp, hp = peaks(PRt[i]); big = hp >= .75
    ax.scatter(fp, hp, marker="s" if y[i] == 0 else "o", s=18 if y[i] == 0 else 24, color=NOR if y[i] == 0 else ABN, alpha=.55, edgecolor="none", zorder=2)
    for a, b in zip(fp[big], hp[big]): ax.annotate(names[i].replace("nor_", "N").replace("abn_", "A"), (a, b), xytext=(3, 3), textcoords="offset points", fontsize=7, color=INK)
fp, hp = peaks(PRv[kV]); ax.scatter(fp, hp, marker="*", s=260, color=VAL, edgecolor="k", lw=.8, zorder=5, label="V001 봉우리")
for a, b in zip(fp, hp): ax.annotate(f"V001 {a:.0f} Hz", (a, b), xytext=(6, 4), textcoords="offset points", fontsize=9, weight="bold", color=VAL)
ax.axhline(.75, color="#2ca02c", ls="--", lw=1); ax.text(8990, .77, "규칙 돌출 문턱 0.75", ha="right", fontsize=8, color="#2ca02c")
ax.set_xlim(8000, 9000); ax.set_ylim(.25, max(1.5, hp.max() + .1) if len(hp) else 1.5)
ax.scatter([], [], marker="s", color=NOR, label="train 정상 봉우리"); ax.scatter([], [], marker="o", color=ABN, label="train 고장 봉우리"); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("봉우리 위치 (Hz)", color=MUTED); ax.set_ylabel("돌출 (dec)", color=MUTED)
ax.set_title("① 2-11 봉우리 산점도(같은 정의) + V001 — 빨강 = 규칙 구간 8200–8450, 파랑 = 정상 큰 봉우리 구간 8450–8750", loc="left", fontsize=9.5)
ax = axs[1]
for lo, hi, c in ((8200, 8450, "#fbe0d0"), (8450, 8750, "#d6e6fa")): ax.axhspan(lo, hi, color=c, alpha=.7, zorder=0)
for nm, c, lab in (("val", VAL, "V001"), ("nor_045", NOR, "N045 (정상)"), ("nor_009", "#7fb0e8", "N009 (정상)")):
    x = ns.Xv[kV] if nm == "val" else ns.X[names.index(nm)]; tt, fr, pp = track(x); on = pp >= 10
    ax.plot(tt[on], fr[on], "-o" if nm == "val" else "-", ms=3, lw=1.6 if nm == "val" else 1.1, color=c, label=f"{lab} (솟은 프레임 {on.mean() * 100:.0f} %)")
tt, fr, pp = track(ns.Xv[kV]); on = pp >= 10
ax.set_ylim(7800, 9000); ax.set_xlim(0, 2); ax.set_xlabel("시간 (s)", color=MUTED); ax.set_ylabel("7.8–9.0 kHz 최대 봉우리 주파수 (Hz)", color=MUTED); ax.legend(fontsize=8, loc="upper right")
ax.set_title(f"② 지속음 주파수의 시간 변화 (0.1 s 창, 10 dB 넘게 솟은 프레임만) — V001 {fr[on].max():.0f} → {fr[on].min():.0f} Hz", loc="left", fontsize=9.5)
ax.grid(alpha=.25)
IMv = all_impacts(ns.Xv[[kV]])[0]
in_rule = float(((fr >= 8200) & (fr < 8450) & on).sum() / max(on.sum(), 1)); in_nor = float(((fr >= 8450) & (fr < 8750) & on).sum() / max(on.sum(), 1))
fig.suptitle(f"V001과 8.3k 규칙의 기준 자료 — ③ 충격 울림 근거(2-16 · 신효섭 그림 3: 8.2–8.45 kHz로 울린 충격 정상 2/936 · 고장 25/569)에서 V001의 충격 = {len(IMv['t'])}개",
             x=.01, ha="left", fontsize=10.5)
fig.tight_layout(rect=(0, 0, 1, .93)); fig.savefig(os.path.join(HERE, "figs", "V001_기준자료비교.png"), dpi=150); plt.close(fig)
open(os.path.join(HERE, "V001_기준자료비교.txt"), "w", encoding="utf-8").write(
    f"V001 8–9 kHz 봉우리(2-11 정의): " + ", ".join(f"{a:.0f} Hz 돌출 {b:.2f}" for a, b in zip(*peaks(PRv[kV]))) + "\n"
    f"V001 지속음 주파수: {fr[on].max():.0f} → {fr[on].min():.0f} Hz (솟은 프레임 {on.mean() * 100:.0f} %), 규칙 구간(8200–8450)에 있던 프레임 {in_rule * 100:.0f} % · 정상 큰 봉우리 구간(8450–8750) {in_nor * 100:.0f} %\n"
    f"V001 충격 수: {len(IMv['t'])}\n"
    "train 규칙 구간(8200–8450) 돌출 ≥ 0.75 봉우리: " + ", ".join(f"{names[i]} {a:.0f} Hz {b:.2f}" for i in range(len(y)) for a, b in zip(*peaks(PRt[i])) if 8200 <= a < 8450 and b >= .75) + "\n"
    "train 8450–8750 돌출 ≥ 0.75 봉우리: " + ", ".join(f"{names[i]} {a:.0f} Hz {b:.2f}" for i in range(len(y)) for a, b in zip(*peaks(PRt[i])) if 8450 <= a < 8750 and b >= .75) + "\n")
print("done")
