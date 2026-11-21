# %% [markdown]
# Q6 Welch 잡음 실측 (사령탑 확인 요청, 11-21)
# - 정상 120개를 앞 1 s / 뒤 1 s로 나눠 각각 Welch(50% 겹침, Hann). 정상은 앞뒤가 같은 신호여야 하니
#   앞 - 뒤 PSD 차이(dec = log10 비율)가 그 창 길이의 측정 잡음 크기다
# - 2048점(칸 12.2 Hz, 조각 약 23) / 8192점(칸 3 Hz, 조각 약 5, detrend 끔). 파일 평균은 먼저 뺀다
# - 비교용: 같은 창으로 구한 정상 파일 사이 흩어짐(2 s 전체 PSD, 파일 120개의 5-95% 폭)
# - 한계: 앞뒤에 충격이 다르게 들어 있으면 그 차이도 섞인다. 1 s 신호의 잡음이라 2 s 전체(지금 방식)는 더 작을 것
# 실행: conda run -n qe-phm python run.py
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from src.exp import load_train
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FS = 25000; NOR = "#2a78d6"; C2 = "#eb6834"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); y = np.asarray(y); X = X[y == 0]
X = X - X.mean(1, keepdims=True)
front, back = X[:, :25000], X[:, 25000:]

def lp(S, n, det):
    f, P = welch(S, fs=FS, nperseg=n, noverlap=n // 2, window="hann", detrend=det, axis=-1)
    return f, np.log10(P + 1e-20)

out = ["정상 120개, 앞 1 s - 뒤 1 s PSD 차이 (dec). |차이| 중앙 = 파일 x 주파수 칸 전체의 중앙", ""]
R = {}
for n, det, tag in ((2048, "constant", "2048점"), (8192, False, "8192점")):
    f, Lf = lp(front, n, det); _, Lb = lp(back, n, det); _, L2 = lp(X, n, det)
    d = Lf - Lb
    nseg = (25000 - n) // (n // 2) + 1
    R[tag] = (f, d, L2)
    out.append(f"[{tag}] 칸 간격 {f[1]:.1f} Hz, 1 s 안 조각 {nseg}개")
    for lo, hi in ((3, 200), (200, 1000), (1000, 5000), (5000, 12500)):
        mk = (f >= lo) & (f <= hi)
        a = np.abs(d[:, mk]); sd_b = (np.percentile(L2[:, mk], 95, 0) - np.percentile(L2[:, mk], 5, 0)).mean() / 3.29
        out.append(f"  {lo:>5}-{hi:<5} Hz: |차이| 중앙 {np.median(a):.3f} dec · 90% {np.percentile(a, 90):.3f} · 차이 표준편차 {d[:, mk].std():.3f}"
                   f" | 정상 파일 사이 흩어짐(5-95% 폭/3.29) {sd_b:.3f} dec")
    out.append("")
# 2048 vs 8192, 같은 대역에서 잡음 배율
for lo, hi in ((3, 200), (200, 1000), (1000, 5000), (5000, 12500)):
    r = []
    for tag in ("2048점", "8192점"):
        f, d, _ = R[tag]; mk = (f >= lo) & (f <= hi); r.append(np.median(np.abs(d[:, mk])))
    out.append(f"{lo}-{hi} Hz: 8192점 |차이| / 2048점 |차이| = {r[1] / r[0]:.2f}")
# 파일별로 앞뒤가 많이 다른 정상 (충격이 한쪽에만 있을 가능성)
f, d, _ = R["2048점"]; mk = (f >= 200)
big = np.median(np.abs(d[:, mk]), 1); nm = np.array([n for n, k in zip(names, y) if k == 0])
out.append("")
out.append("2048점 앞뒤 차이가 큰 정상 파일 (|차이| 중앙 상위 6): " + ", ".join(f"{nm[i]}({big[i]:.2f})" for i in np.argsort(-big)[:6]))
out.append(f"그 중앙 {np.median(big):.3f}, 90% {np.percentile(big, 90):.3f}")
open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8").write("\n".join(out))

# %% 그림
fig, axs = plt.subplots(3, 1, figsize=(11, 12))
for ax, (tag, xlim, step, xlog) in zip(axs[:2], (("2048점", (20, 12500), None, True), ("8192점", (3, 200), 10, False))):
    f, d, _ = R[tag]; mk = (f >= xlim[0]) & (f <= xlim[1])
    ax.fill_between(f[mk], np.percentile(d, 5, 0)[mk], np.percentile(d, 95, 0)[mk], color=NOR, alpha=.2, label="정상 120개 5-95%")
    ax.plot(f[mk], np.median(d, 0)[mk], color=NOR, lw=1.5, label="중앙")
    ax.axhline(0, color="k", lw=.7)
    if xlog:
        ax.set_xscale("log"); ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["20", "50", "100", "200", "500", "1k", "2k", "5k", "10k"])
    else:
        ax.set_xticks(np.arange(0, 201, step))
    ax.set_xlim(*xlim); ax.set_ylabel("앞 1 s - 뒤 1 s (dec)"); ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8)
    ax.set_title(f"Welch {tag}: 정상 파일의 앞뒤 PSD 차이 (0 = 같음)")
ax = axs[2]
for tag, col in (("2048점", NOR), ("8192점", C2)):
    f, d, _ = R[tag]; mk = (f >= 3) & (f <= 12500)
    med = np.median(np.abs(d), 0)
    k = max(1, int(len(f) / 400)); sm = np.convolve(med, np.ones(k) / k, mode="same")
    ax.plot(f[mk], sm[mk], color=col, lw=1.4, label=f"{tag} |앞-뒤| 중앙")
ax.set_xscale("log"); ax.set_xticks([5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]); ax.set_xticklabels(["5", "10", "20", "50", "100", "200", "500", "1k", "2k", "5k", "10k"])
ax.set_xlabel("주파수 (Hz)"); ax.set_ylabel("|차이| 중앙 (dec)"); ax.set_title("측정 잡음 크기: 창 길이별 |앞-뒤| 중앙 (작을수록 안정)"); ax.grid(alpha=.3, which="both"); ax.legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "q6_welch_noise.png"), dpi=90); plt.close(fig)
print("done")
