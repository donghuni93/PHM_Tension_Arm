# 2-24 추가: 11 kHz 근처 봉우리 (시각화). 돌출 = 10.8–11.2 kHz 최대 − 이웃(10.2–10.6 · 11.4–11.8 kHz) 중앙값, dB. Welch 4096점
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import welch, spectrogram
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp import load_train
from src.data import load_folder
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000; INK, MUTED, GRID = "#1d2430", "#5b6575", "#c9ced6"
X, y, names = load_train(); names = [str(n) for n in names]
vd = os.path.join(_d, "data", "TeamProject_val", "val"); vd = vd if os.path.isdir(vd) else os.path.dirname(vd); Xv, nv = load_folder(vd)
def psd(A): f, P = welch(A - A.mean(1, keepdims=True), fs=FS, nperseg=4096, axis=-1); return f, 10 * np.log10(P + 1e-20)
f, P = psd(X); _, Pv = psd(Xv)
pk = (f >= 10800) & (f < 11200); nb = ((f >= 10200) & (f < 10600)) | ((f >= 11400) & (f < 11800))
prom = lambda D: D[:, pk].max(1) - np.median(D[:, nb], 1)
G = [("정상 (120)", prom(P[y == 0]), "#2a78d6"), ("고장 (30)", prom(P[y == 1]), "#eb6834"), ("val (40)", prom(Pv), "#8a4fc9")]
fig = plt.figure(figsize=(17, 9)); gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.9], height_ratios=[1, 1])
ax = fig.add_subplot(gs[:, 0]); rng = np.random.default_rng(0)
for j, (nm, v, c) in enumerate(G):
    ax.scatter(j + rng.uniform(-.2, .2, len(v)), v, s=26, color=c, alpha=.7, lw=0)
    q = np.percentile(v, [25, 50, 75]); ax.plot([j - .3, j + .3], [q[1]] * 2, color=INK, lw=2.2)
    ax.plot([j - .22, j + .22], [q[0]] * 2, color=INK, lw=1); ax.plot([j - .22, j + .22], [q[2]] * 2, color=INK, lw=1)
    ax.text(j, v.max() + 1.2, f"≥ 6 dB {np.mean(v >= 6):.0%}", ha="center", fontsize=10, color=INK)
ax.axhline(6, color=MUTED, lw=.8, ls=":"); ax.set_xticks(range(3)); ax.set_xticklabels([g[0] for g in G], fontsize=11, color=INK)
ax.set_ylabel("11 kHz 봉우리 돌출 (dB, 이웃 대비)", color=MUTED)
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="y", alpha=.3, lw=.6)
ax.set_title("파일별 11 kHz 봉우리 돌출 (굵은 선 = 중앙값, 가는 선 = 25 · 75%)", fontsize=12, color=INK, loc="left")
pn = prom(P); order = np.argsort(-pn)
for r, fn in enumerate([names[order[0]], names[order[1]]]):
    a = fig.add_subplot(gs[r, 1]); x = X[names.index(fn)]; x = x - x.mean()
    fs_, ts, S = spectrogram(x, fs=FS, nperseg=1024, noverlap=896); mm = fs_ >= 9000; Z = 10 * np.log10(S[mm] + 1e-20)
    a.pcolormesh(ts, fs_[mm] / 1000, Z, shading="auto", cmap="magma", vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 99.8))
    a.axhline(11.0, color="white", lw=.6, ls=":", alpha=.7); a.set_ylabel("kHz", color=MUTED); a.tick_params(colors=MUTED, length=0)
    a.set_title(f"정상 {fn[-3:]} 스펙트로그램 9–12.5 kHz (돌출 {pn[names.index(fn)]:.0f} dB, 정상 중 1·2위, 점선 = 11 kHz)", fontsize=11.5, color=INK, loc="left")
    if r == 1: a.set_xlabel("시간 (s)", color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림4_11k봉우리.png"), dpi=130); plt.close(fig)
print("top:", [names[i] for i in order[:2]])
