# Q1 보충: 8-10 kHz 들뜸과 11.5-12.5 kHz 들뜸이 같은 파일에서 나타나나 (사후 대역: 중앙값 곡선에서 정함)
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
p = HERE
while not os.path.isdir(os.path.join(p, "src")): p = os.path.dirname(p)
ROOT = p; sys.path.insert(0, ROOT)
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch
from scipy.stats import spearmanr
from src.exp import load_train
from src.data import load_folder
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FS = 25000; NOR, ABN, VAL = "#2a78d6", "#eb6834", "#2a9d5c"
X, y, names = load_train(); y = np.asarray(y); names = [str(n) for n in names]
XV, vn = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val"))
def lp(S):
    S = S - S.mean(1, keepdims=True)
    f, P = welch(S, fs=FS, nperseg=2048, noverlap=1024, window="hann", axis=-1)
    return f, np.log10(P + 1e-20)
f, Lt = lp(X); _, Lv = lp(XV)
med = np.median(Lt[y == 0], 0)
def band(L, lo, hi):   # 파일별 평균 (로그 PSD - 정상 중앙), dec
    m = (f >= lo) & (f <= hi)
    return (L - med)[:, m].mean(1)
res = {}
for tag, (lo, hi) in {"A": (8000, 10000), "B": (11500, 12500), "C": (1200, 6000)}.items():
    res[tag] = (band(Lt, lo, hi), band(Lv, lo, hi))
out = ["A = 8–10 kHz, B = 11.5–12.5 kHz, C = 1.2–6 kHz (비교용 '전체 크기' 대역). 값 = 파일별 평균(로그 PSD − 정상 중앙), dec"]
groups = {"정상": (res["A"][0][y == 0], res["B"][0][y == 0], res["C"][0][y == 0]),
          "고장": (res["A"][0][y == 1], res["B"][0][y == 1], res["C"][0][y == 1]),
          "val":  (res["A"][1], res["B"][1], res["C"][1])}
for g, (a, b, c) in groups.items():
    r1, p1 = spearmanr(a, b); r2, _ = spearmanr(a, c); r3, _ = spearmanr(b, c)
    ra, _ = spearmanr(a - c, b - c)
    out.append(f"{g}: Spearman A-B {r1:+.2f} (p {p1:.3f}) | A-C {r2:+.2f} | B-C {r3:+.2f} | 크기 C 제거 후 (A-C)-(B-C) {ra:+.2f}")
    out.append(f"   중앙 A {np.median(a):+.2f}, B {np.median(b):+.2f}, C {np.median(c):+.2f}")
# 정상 90% 선을 넘는 파일 (정상 기준 상위 5% 선)
for tag in "AB":
    thr = np.percentile(res[tag][0][y == 0], 95)
    ab = res[tag][0][y == 1] > thr; vv = res[tag][1] > thr
    out.append(f"{tag}: 정상 95% 선 {thr:+.2f} dec 초과 — 고장 {ab.sum()}/30, val {vv.sum()}/40, 정상 {(res[tag][0][y==0]>thr).sum()}/120")
tA = np.percentile(res["A"][0][y == 0], 95); tB = np.percentile(res["B"][0][y == 0], 95)
an = [n for n, k in zip(names, y) if k == 1]
aA, aB = res["A"][0][y == 1], res["B"][0][y == 1]
out.append("A만 초과: " + ", ".join(n for n, a, b in zip(an, aA, aB) if a > tA and b <= tB))
out.append("B만 초과: " + ", ".join(n for n, a, b in zip(an, aA, aB) if b > tB and a <= tA))
out.append("A·B 둘 다 초과: " + ", ".join(n for n, a, b in zip(an, aA, aB) if a > tA and b > tB))
out.append("둘 다 아님: " + ", ".join(n for n, a, b in zip(an, aA, aB) if a <= tA and b <= tB))
vA, vB = res["A"][1], res["B"][1]
out.append("val A·B 둘 다 초과: " + ", ".join(n for n, a, b in zip(vn, vA, vB) if a > tA and b > tB))
out.append("val A만: " + ", ".join(n for n, a, b in zip(vn, vA, vB) if a > tA and b <= tB))
out.append("val B만: " + ", ".join(n for n, a, b in zip(vn, vA, vB) if b > tB and a <= tA))

fig, axs = plt.subplots(1, 2, figsize=(13, 6))
for ax, (xk, yk, xl, yl) in zip(axs, (("A", "B", "A: 8–10 kHz 들뜸 (dec)", "B: 11.5–12.5 kHz 들뜸 (dec)"),
                                       ("A", "C", "A: 8–10 kHz 들뜸 (dec)", "C: 1.2–6 kHz 들뜸 (dec) = 전체 크기"))):
    ax.scatter(res[xk][0][y == 0], res[yk][0][y == 0], c=NOR, s=18, alpha=.5, label="정상")
    ax.scatter(res[xk][0][y == 1], res[yk][0][y == 1], c=ABN, s=40, label="고장")
    ax.scatter(res[xk][1], res[yk][1], c=VAL, marker="^", s=40, label="val")
    for n, a, b in zip(an, res[xk][0][y == 1], res[yk][0][y == 1]): ax.annotate(n[-3:], (a, b), fontsize=6, color=ABN, xytext=(2, 2), textcoords="offset points")
    for n, a, b in zip(vn, res[xk][1], res[yk][1]): ax.annotate(n[-3:], (a, b), fontsize=6, color=VAL, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="k", lw=.6); ax.axvline(0, color="k", lw=.6)
    ax.set_xlabel(xl); ax.set_ylabel(yl); ax.grid(alpha=.3); ax.legend()
axs[0].set_title("8–10 kHz 들뜸 vs 11.5–12.5 kHz 들뜸 (파일별)")
axs[1].set_title("8–10 kHz 들뜸 vs 전체 크기 (비교)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "q1_band_corr.png"), dpi=90); plt.close(fig)
open(os.path.join(HERE, "result_band_corr.txt"), "w", encoding="utf-8").write(chr(10).join(out))
print("done")
