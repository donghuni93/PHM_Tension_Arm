# %% [markdown]
# # 2-2 피크 주파수 수치 비교 (정상 120 vs 고장 30)
# `figs/2-2_fft_amp.npz`(평균 제거 · Hann · 0.5 Hz 해상도 단측 진폭)를 읽는다.
# 1. 대역별 최대 피크 주파수·진폭: 2–300 Hz(원 FFT) / 300–5000 Hz · 5000–12500 Hz (12.5 Hz 이동평균 후)
# 2. 파일별 상위 10개 피크(2 Hz 이상, 서로 5 Hz 이상 떨어짐)가 어느 주파수에 몰리나: 1 Hz 칸으로 반올림해 파일 비율
# AUC·Mann–Whitney는 설명용. 이 결과로 피처 대역을 고르지 않는다 (작업 원칙 3)

# %%
import sys
from pathlib import Path
ROOT_DIR = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
                if (p / "src" / "exp.py").exists())
import numpy as np, pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score

HERE = ROOT_DIR / "단계" / "2_EDA" / "2-2_주파수개관"; FIGS = HERE / "figs"
d = np.load(FIGS / "2-2_fft_amp.npz")
f, A, y, names = d["f"], d["A"].astype(float), d["y"], list(d["names"])
As = uniform_filter1d(A, 25, axis=1)                       # 12.5 Hz 이동평균
BANDS = [("저 2–300 Hz", 2, 300, A), ("중 300–5000 Hz", 300, 5000, As), ("고 5000–12500 Hz", 5000, 12500, As)]

rows = []
for i, n in enumerate(names):
    r = {"name": n, "y": y[i]}
    for lab, lo, hi, M in BANDS:
        m = np.where((f >= lo) & (f < hi))[0]
        j = m[M[i, m].argmax()]
        r[f"{lab} 피크Hz"] = f[j]; r[f"{lab} 피크g"] = M[i, j]
    rows.append(r)
P = pd.DataFrame(rows)
P.to_csv(FIGS / "2-2_peaks_by_band.csv", index=False, encoding="utf-8-sig")

out = []
for c in [c for c in P.columns if "피크" in c]:
    a, b = P.loc[P.y == 0, c], P.loc[P.y == 1, c]
    out.append({"항목": c,
                "정상 5%": a.quantile(.05), "정상 중앙값": a.median(), "정상 95%": a.quantile(.95),
                "고장 5%": b.quantile(.05), "고장 중앙값": b.median(), "고장 95%": b.quantile(.95),
                "AUC": roc_auc_score(P.y, P[c]), "MW p": mannwhitneyu(a, b).pvalue})
T1 = pd.DataFrame(out)


def top_peaks(a, k=10, fmin=2, sep_hz=5):
    picked = []
    for j in np.argsort(a)[::-1]:
        if f[j] < fmin or any(abs(f[j] - f[p]) < sep_hz for p in picked):
            continue
        picked.append(j)
        if len(picked) == k:
            break
    return [f[j] for j in picked]


PK = [top_peaks(A[i]) for i in range(len(names))]
pd.DataFrame({"name": names, "y": y, **{f"p{k + 1}": [p[k] for p in PK] for k in range(10)}}) \
  .to_csv(FIGS / "2-2_top10_peaks.csv", index=False, encoding="utf-8-sig")
# 1 Hz 칸(반올림)마다 그 칸에 상위 10 피크가 있는 파일 비율
cnt = {}
for i, p in enumerate(PK):
    for fr in set(np.round(p).astype(int)):
        cnt.setdefault(fr, [0, 0])[y[i]] += 1
n0, n1 = (y == 0).sum(), (y == 1).sum()
T2 = pd.DataFrame([{"주파수Hz": k, "정상 파일 수": v[0], "정상 %": v[0] / n0 * 100,
                    "고장 파일 수": v[1], "고장 %": v[1] / n1 * 100} for k, v in cnt.items()])
T2["차이 %p"] = T2["고장 %"] - T2["정상 %"]
T2 = T2.sort_values("주파수Hz")
T2.to_csv(FIGS / "2-2_peak_freq_share.csv", index=False, encoding="utf-8-sig")
top1 = pd.Series([p[0] for p in PK])

with open(FIGS / "2-2_peaks_summary.txt", "w", encoding="utf-8") as fh:
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
    fh.write("== 대역별 최대 피크\n" + T1.round(4).to_string(index=False) + "\n\n")
    fh.write("== 최대 피크(top1) 주파수 빈도\n")
    for g in (0, 1):
        fh.write(f"[{'정상' if g == 0 else '고장'}] " + top1[y == g].round().value_counts().head(12).to_string().replace("\n", " | ") + "\n")
    fh.write("\n== 상위10 피크 등장 비율: 정상 또는 고장 20% 이상인 주파수\n")
    fh.write(T2[(T2["정상 %"] >= 20) | (T2["고장 %"] >= 20)].round(1).to_string(index=False) + "\n\n")
    fh.write("== 차이 큰 순 (고장 더 많음)\n" + T2.sort_values("차이 %p", ascending=False).head(12).round(1).to_string(index=False) + "\n")
    fh.write("== 차이 큰 순 (정상 더 많음)\n" + T2.sort_values("차이 %p").head(12).round(1).to_string(index=False) + "\n")
    fh.write("\n== 파일별 상위 5 피크 (고장)\n")
    for i in np.where(y == 1)[0]:
        fh.write(f"{names[i]}: " + ", ".join(f"{v:g}" for v in PK[i][:5]) + "\n")

# %% 상위 10 피크 중 300 Hz 이상인 개수 (공진·충격 성분이 최상위로 올라오나)
hi_cnt = pd.Series([sum(v >= 300 for v in p) for p in PK])
with open(FIGS / "2-2_peaks_summary.txt", "a", encoding="utf-8") as fh:
    fh.write("\n== 상위10 피크 중 300 Hz 이상 개수\n")
    for g in (0, 1):
        s = hi_cnt[y == g]
        fh.write(f"[{'정상' if g == 0 else '고장'}] 0개 {(s == 0).mean() * 100:.0f}% · 1개 이상 {(s >= 1).mean() * 100:.0f}% · "
                 f"5개 이상 {(s >= 5).mean() * 100:.0f}% · 평균 {s.mean():.2f}\n")
    fh.write(f"AUC {roc_auc_score(y, hi_cnt):.3f}\n")

# %% 그림
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")
BLUE, ORANGE, INK = "#2a78d6", "#eb6834", "#222222"
MISSED = ["abn_002", "abn_022", "abn_029"]
rng = np.random.default_rng(0); jit = rng.uniform(-.18, .18, len(P))

fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1])
specs = [(f"{lab} 피크Hz", lab, lab.startswith("저")) for lab, *_ in BANDS] + [(None, "상위10 중 300 Hz 이상 개수", False)]
for k, (c, lab, logy) in enumerate(specs):
    a = fig.add_subplot(gs[0, k])
    v = P[c] if c else hi_cnt
    for g, col in ((0, BLUE), (1, ORANGE)):
        m = (P.y == g).values
        a.scatter(g + jit[m], v[m], s=14, color=col, alpha=.6, lw=0)
        a.hlines(np.median(v[m]), g - .3, g + .3, color=INK, lw=1.5)
    for n in MISSED:
        i = names.index(n)
        a.scatter(1 + jit[i], v[i], s=55, facecolor="none", edgecolor=INK, lw=1.1)
    if logy:
        a.set_yscale("log"); a.yaxis.set_major_formatter(PLAIN); a.yaxis.set_minor_formatter(NullFormatter())
    a.set_xticks([0, 1]); a.set_xticklabels([f"정상 {n0}", f"고장 {n1}"]); a.set_xlim(-.6, 1.6)
    a.set_title(f"{lab}" + (" 최대 피크 (Hz)" if c else "") + f"\nAUC {roc_auc_score(y, v):.2f}", fontsize=9.5, loc="left")
    a.grid(axis="y", alpha=.25); a.spines[["top", "right"]].set_visible(False)

a = fig.add_subplot(gs[1, :])
S = T2[T2["주파수Hz"] <= 150].set_index("주파수Hz").reindex(range(2, 151), fill_value=0)
a.bar(S.index - .2, S["정상 %"], width=.4, color=BLUE, label=f"정상 {n0}개 중 %")
a.bar(S.index + .2, S["고장 %"], width=.4, color=ORANGE, label=f"고장 {n1}개 중 %")
a.set_xlim(1, 151); a.set_xlabel("Hz (1 Hz 칸으로 반올림)"); a.set_ylabel("상위10 피크에 그 주파수가 있는 파일 %")
a.set_title("2–150 Hz: 상위 10 피크가 몰리는 주파수 (정상 vs 고장)", fontsize=10, loc="left")
a.legend(frameon=False); a.grid(axis="y", alpha=.25); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("피크 주파수 비교 — 검은 가로선 = 중앙값, ○ = 병목 고장 002·022·029", fontsize=11, x=.01, ha="left")
fig.tight_layout()
fig.savefig(FIGS / "2-2_peaks_compare.png", dpi=120)
