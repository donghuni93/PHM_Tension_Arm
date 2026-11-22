# Q1 보충: 크기를 나눈 "모양" PSD 비교. 파일마다 PSD를 200 Hz 이상 전체 파워 합으로 나눔(= 각 주파수가 전체의 몇 몫인가)
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks
from scipy.stats import spearmanr
from src.exp import load_train
from src.data import load_folder
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FS = 25000; NOR, ABN, VAL = "#2a78d6", "#eb6834", "#2a9d5c"; FIG = os.path.join(HERE, "figs")
X, y, names = load_train(); y = np.asarray(y); names = [str(n) for n in names]
XV, vn = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val"))
an = [n for n, k in zip(names, y) if k == 1]
def shape(S):
    S = S - S.mean(1, keepdims=True)
    f, P = welch(S, fs=FS, nperseg=2048, noverlap=1024, window="hann", axis=-1)
    m = f >= 200
    return f, np.log10(P / P[:, m].sum(1, keepdims=True) + 1e-20)
f, Lt = shape(X); _, Lv = shape(XV)
nor, abn = y == 0, y == 1
lo, med, hi = np.percentile(Lt[nor], 5, 0), np.median(Lt[nor], 0), np.percentile(Lt[nor], 95, 0)
am, vm = np.median(Lt[abn], 0), np.median(Lv, 0)
out = ["모양 PSD = 파일 PSD / (200 Hz 이상 파워 합). 값은 log10(몫). 차이 = dec"]

# 그림 1: 곡선 (1–12.5 kHz 선형 x) + 차이 곡선, 상승·하락 봉우리 번호
xlo, xhi = 200, 12500
m = (f >= xlo) & (f <= xhi)
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
a1.fill_between(f[m], lo[m], hi[m], color=NOR, alpha=.18, label="정상 5–95% 띠")
a1.plot(f[m], med[m], color=NOR, lw=1.6, label="정상 중앙"); a1.plot(f[m], am[m], color=ABN, lw=1.6, label="고장 중앙"); a1.plot(f[m], vm[m], color=VAL, lw=1.6, ls="--", label="val 중앙")
a1.set_ylabel("log10 (전체 파워 몫)"); a1.legend(ncol=2, loc="lower left"); a1.grid(alpha=.3)
a1.set_title("모양 PSD (크기 나눔, 200 Hz 이상 합 = 1) — 선형 x")
d = am - med; dv = vm - med
a2.axhline(0, color="k", lw=.8)
a2.fill_between(f[m], (lo - med)[m], (hi - med)[m], color=NOR, alpha=.18, label="정상 5–95% 띠")
a2.plot(f[m], d[m], color=ABN, label="고장 중앙 - 정상 중앙"); a2.plot(f[m], dv[m], color=VAL, ls="--", label="val 중앙 - 정상 중앙")
ds = np.convolve(d, np.ones(3) / 3, mode="same")
mm = np.where(m)[0]
up, _ = find_peaks(ds[m], prominence=0.15, distance=15); dn, _ = find_peaks(-ds[m], prominence=0.15, distance=15)
up = up[np.argsort(-ds[m][up])[:4]]; dn = dn[np.argsort(ds[m][dn])[:3]]
items = []
for k, i in enumerate(sorted(list(up) + list(dn), key=lambda i: f[m][i]), 1):
    hz, v = f[m][i], d[m][i]
    for a in (a1, a2): a.axvline(hz, color=ABN, lw=.7, ls=":", alpha=.8)
    a2.plot(hz, v, "o", ms=11, mfc="white", mec=ABN, zorder=5); a2.text(hz, v, str(k), ha="center", va="center", fontsize=7, color=ABN, fontweight="bold", zorder=6)
    items.append(f"{k}  {hz:.0f} Hz  {v:+.1f} dec"); out.append(f"봉우리/골 {k}: {hz:.0f} Hz, {v:+.2f} dec")
a2.text(0.01, 0.03, "고장-정상 차이 봉우리(+)와 골(-)" + chr(10) + chr(10).join(items), transform=a2.transAxes, va="bottom", ha="left", fontsize=8, color=ABN,
        bbox=dict(boxstyle="round", fc="white", ec=ABN, alpha=.9), zorder=7)
a2.set_ylabel("차이 (dec)"); a2.set_xlabel("주파수 (Hz)"); a2.legend(ncol=3, fontsize=8, loc="upper right"); a2.grid(alpha=.3)
a2.set_xticks(np.arange(0, 12501, 500)); a2.tick_params(axis="x", labelrotation=90, labelsize=8); a2.set_xlim(xlo, xhi)
fig.savefig(os.path.join(FIG, "q1s_curves.png"), dpi=90, bbox_inches="tight"); plt.close(fig)

# 그림 2: 히트맵 (고장 30 + val 40, 정상 95% 선 초과 dec)
E = np.vstack([np.clip(Lt[abn] - hi, 0, None)[:, m], np.clip(Lv - hi, 0, None)[:, m]])
rows = an + vn
fig, ax = plt.subplots(figsize=(12, 14))
im = ax.imshow(E, aspect="auto", cmap="magma_r", vmin=0, vmax=1.0, extent=[f[m][0], f[m][-1], len(rows), 0], interpolation="nearest")
ax.axhline(abn.sum(), color="k", lw=1.5); ax.set_yticks(np.arange(len(rows)) + .5); ax.set_yticklabels(rows, fontsize=6)
ax.set_xticks(np.arange(0, 12501, 500)); ax.tick_params(axis="x", labelrotation=90, labelsize=8)
ax.set_xlabel("주파수 (Hz)"); ax.set_title("모양 PSD: 정상 95% 선을 넘는 주파수 (위 30행 = 고장, 아래 40행 = val)")
fig.colorbar(im, ax=ax, fraction=.025, label="정상 95% 선을 넘은 크기 (dec)")
fig.savefig(os.path.join(FIG, "q1s_heat.png"), dpi=90, bbox_inches="tight"); plt.close(fig)

# 그림 3·숫자: 대역 A(8–10k), B(11.5–12.5k), C(1.2–6k)의 모양 몫 상관
def band(L, a, b):
    mk = (f >= a) & (f <= b); return (L - med)[:, mk].mean(1)
R = {t: (band(Lt, a, b), band(Lv, a, b)) for t, (a, b) in {"A": (8000, 10000), "B": (11500, 12500), "C": (1200, 6000)}.items()}
for g, sel in (("정상", lambda t: R[t][0][nor]), ("고장", lambda t: R[t][0][abn]), ("val", lambda t: R[t][1])):
    r1, p1 = spearmanr(sel("A"), sel("B")); r2, _ = spearmanr(sel("A"), sel("C")); r3, _ = spearmanr(sel("B"), sel("C"))
    out.append(f"{g}: Spearman A-B {r1:+.2f} (p {p1:.3f}) | A-C {r2:+.2f} | B-C {r3:+.2f} | 중앙 A {np.median(sel('A')):+.2f} B {np.median(sel('B')):+.2f} C {np.median(sel('C')):+.2f}")
tA, tB = [np.percentile(R[t][0][nor], 95) for t in "AB"]
for t, th in (("A", tA), ("B", tB)):
    out.append(f"{t}: 정상 95% 선 {th:+.2f} dec 초과 — 고장 {(R[t][0][abn] > th).sum()}/30, val {(R[t][1] > th).sum()}/40, 정상 {(R[t][0][nor] > th).sum()}/120")
aA, aB = R["A"][0][abn], R["B"][0][abn]
out.append("A만 초과: " + ", ".join(n for n, a, b in zip(an, aA, aB) if a > tA and b <= tB))
out.append("B만 초과: " + ", ".join(n for n, a, b in zip(an, aA, aB) if b > tB and a <= tA))
out.append("A·B 둘 다: " + ", ".join(n for n, a, b in zip(an, aA, aB) if a > tA and b > tB))
vA, vB = R["A"][1], R["B"][1]
out.append("val A·B 둘 다: " + ", ".join(n for n, a, b in zip(vn, vA, vB) if a > tA and b > tB) + " | A만: " + ", ".join(n for n, a, b in zip(vn, vA, vB) if a > tA and b <= tB) + " | B만: " + ", ".join(n for n, a, b in zip(vn, vA, vB) if b > tB and a <= tA))
out.append("val 세 파일(004 031 035) 모양 대역값 A/B/C: " + "; ".join(f"{n} {R['A'][1][vn.index(n)]:+.2f}/{R['B'][1][vn.index(n)]:+.2f}/{R['C'][1][vn.index(n)]:+.2f}" for n in ("val_004", "val_031", "val_035")))
fig, axs = plt.subplots(1, 2, figsize=(13, 6))
for ax, (xk, yk) in zip(axs, (("A", "B"), ("A", "C"))):
    ax.scatter(R[xk][0][nor], R[yk][0][nor], c=NOR, s=18, alpha=.5, label="정상"); ax.scatter(R[xk][0][abn], R[yk][0][abn], c=ABN, s=40, label="고장"); ax.scatter(R[xk][1], R[yk][1], c=VAL, marker="^", s=40, label="val")
    for n, a, b in zip(an, R[xk][0][abn], R[yk][0][abn]): ax.annotate(n[-3:], (a, b), fontsize=6, color=ABN, xytext=(2, 2), textcoords="offset points")
    for n, a, b in zip(vn, R[xk][1], R[yk][1]): ax.annotate(n[-3:], (a, b), fontsize=6, color=VAL, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6); ax.grid(alpha=.3); ax.legend()
    ax.set_xlabel({"A": "A: 8–10 kHz 모양 몫 (dec)"}[xk]); ax.set_ylabel({"B": "B: 11.5–12.5 kHz 모양 몫 (dec)", "C": "C: 1.2–6 kHz 모양 몫 (dec)"}[yk])
axs[0].set_title("모양 몫: 8–10 kHz vs 11.5–12.5 kHz"); axs[1].set_title("모양 몫: 8–10 kHz vs 1.2–6 kHz")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q1s_band_corr.png"), dpi=90); plt.close(fig)
open(os.path.join(HERE, "result_shape.txt"), "w", encoding="utf-8").write(chr(10).join(out)); print("done")
