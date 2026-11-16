# %% [markdown]
# 2-5 대역 필터 피크 — P02_peak_025(2.5–3.75 kHz 대역 필터 후 최대 |진폭|)가 어떤 순간에서 나오는가
# - 비교: abn_001(고장, 잘 잡힘) vs nor_055(정상, 오경보 5/5)
# - 열: ① 평균 제거 원신호 2초 ② 2.5–3.75 kHz 대역 신호 2초(Peak 표시, ±RMS 띠) ③ 대역 Peak 시점 ±5 ms 확대
# - 필터는 src/features.py의 P02와 같다 (4차 Butterworth bandpass, sosfiltfilt)
# 실행: conda run -n qe-phm python "단계/2_EDA/2-5_1114_대역필터피크/run.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train, cached_table, feature_table_ext
import src.viz  # noqa: F401  (한글 글꼴 설정)
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis

FS = 25000
FILES = ["abn_001", "nor_055"]
LO, HI = 2500, 3750
C = {"abn": "#eb6834", "nor": "#2a78d6"}
INK, MUTED, GRID = "#0b0b0b", "#85847e", "#e6e5e0"
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

X, y, names = load_train(); names = [str(n) for n in names]
E = cached_table("ext", feature_table_ext, X)
sos = butter(4, [LO, HI], btype="bandpass", fs=FS, output="sos")
nor = y == 0
pn = lambda col, v: (E[col].values[nor] < v).mean() * 100

# %% 그림
t = np.arange(X.shape[1]) / FS
fig, ax = plt.subplots(len(FILES), 3, figsize=(15, 3.4 * len(FILES)), gridspec_kw=dict(width_ratios=[2, 2, 1.2]))
B = {n: sosfiltfilt(sos, X[names.index(n)] - X[names.index(n)].mean()) for n in FILES}
ymax_b = max(np.abs(b).max() for b in B.values()) * 1.1
lines = []
for r, n in enumerate(FILES):
    i = names.index(n); x = X[i] - X[i].mean(); b = B[n]; c = C[n[:3]]
    k = np.abs(b).argmax(); pk, rms = np.abs(b[k]), b.std()
    lines.append(f"{n}: 대역 Peak {pk:.3f} g @ {t[k]:.4f} s, 대역 RMS {rms:.4f} g, 파고율 {pk/rms:.1f}, 첨도(Pearson) {kurtosis(b, fisher=False):.1f} | "
                 f"원신호 Peak {np.abs(x).max():.3f} g @ {t[np.abs(x).argmax()]:.4f} s | "
                 f"정상 중 백분위: peak_025 {pn('P02_peak_025', E['P02_peak_025'][i]):.0f}%, crest_025 {pn('P02_crest_025', E['P02_crest_025'][i]):.0f}%, rms_025 {pn('P02_rms_025', E['P02_rms_025'][i]):.0f}%")
    # ① 원신호
    a = ax[r, 0]; a.plot(t, x, color=c, lw=0.6)
    a.axvline(t[k], color=MUTED, lw=1, ls="--")
    a.set_title(f"{n} ({'고장' if n.startswith('abn') else '정상'}) — 평균 제거 원신호", loc="left", fontsize=11)
    a.set_ylabel("g")
    # ② 대역 신호
    a = ax[r, 1]; a.axhspan(-rms, rms, color=GRID, zorder=0)
    a.plot(t, b, color=c, lw=0.6)
    a.plot(t[k], b[k], "o", ms=8, mfc="none", mec=INK, mew=1.5)
    a.annotate(f"Peak {pk:.3f} g\n파고율 {pk/rms:.0f} (= Peak / RMS {rms:.4f} g)", (t[k], b[k]), xytext=(10, 0), textcoords="offset points",
               fontsize=9, color=INK, va="center")
    a.set_ylim(-ymax_b, ymax_b)
    a.set_title(f"{LO/1000:g}–{HI/1000:g} kHz만 통과 (회색 띠 = ±RMS, y축 두 파일 같음)", loc="left", fontsize=11)
    # ③ 확대
    a = ax[r, 2]; w = slice(max(0, k - 125), k + 125); tz = (t[w] - t[k]) * 1000
    a.plot(tz, x[w], color=MUTED, lw=1, label="원신호")
    a.plot(tz, b[w], color=c, lw=1.5, label=f"{LO/1000:g}–{HI/1000:g} kHz")
    a.set_title(f"Peak 시점 ±5 ms ({t[k]:.3f} s)", loc="left", fontsize=11)
    a.set_xlabel("ms"); a.legend(fontsize=8, frameon=False, loc="lower right")
for a in ax.flat:
    a.grid(color=GRID, lw=0.6); a.spines[["top", "right"]].set_visible(False)
    for s in ["left", "bottom"]: a.spines[s].set_color("#b9b8b1")
for a in ax[-1, :2]: a.set_xlabel("s")
fig.tight_layout()
p = os.path.join(HERE, "figs", "band_peak_abn001_nor055.png")
fig.savefig(p, dpi=130)
open(os.path.join(HERE, "figs", "band_peak_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("done")
