# %% 2-34 피처 점검 ③ 첨도 · 봉우리: H3 = 1–2 kHz 대역 log10(Pearson 첨도), H5 = 4–8 kHz, H16 = 가장 솟은 좁은 봉우리 높이(정상 중앙 log PSD 대비 dec), H17 = 그 봉우리의 반높이 폭(Hz)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.signal import sosfiltfilt
from scipy.stats import kurtosis
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp2util import NS, rule83_flags
from src.features2 import _band_sos, octave_kurtosis, welch_log, OCT
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
_av = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((n for n in ["Malgun Gothic", "NanumGothic", "WenQuanYi Zen Hei"] if n in _av), "DejaVu Sans"); plt.rcParams["axes.unicode_minus"] = False
ns = NS(); y = ns.y; names = list(ns.names); nv = [n[-3:] for n in ns.nv]; ntr = len(names)
allnames = names + ["val_" + n for n in nv]; X = np.vstack([ns.X, ns.Xv]); grp = np.array(["정상" if v == 0 else "고장" for v in y] + ["val"] * len(nv))
isN, isA, isV = grp == "정상", grp == "고장", grp == "val"
T = pd.concat([ns.T.reset_index(drop=True), ns.Tv.reset_index(drop=True)], ignore_index=True)
# 첨도 재계산 검증
K = np.array([octave_kurtosis(x) for x in X]); chk = max(np.abs(K[:, 0] - T["H3_kurt_1-2k"]).max(), np.abs(K[:, 2] - T["H5_kurt_4-8k"]).max())
# 봉우리: 학습 정상 중앙 log PSD 대비 (정상 자기 자신은 제외한 중앙값으로 계산해 자기 영향을 뺀다)
L = np.vstack([ns.L, ns.Lv]); df = ns.df; f_all, _ = welch_log(X[0]); fr = f_all[f_all >= 1000]; nor_idx = np.flatnonzero(isN)
ref_full = np.median(L[nor_idx], axis=0)
def peak(d):
    k = int(d.argmax()); h = d[k]; lo = hi = k
    while lo > 0 and d[lo - 1] >= h / 2: lo -= 1
    while hi < len(d) - 1 and d[hi + 1] >= h / 2: hi += 1
    return h, (hi - lo + 1) * df, fr[k]
PK = np.array([peak(L[i] - (np.median(L[[j for j in nor_idx if j != i]], axis=0) if isN[i] else ref_full)) for i in range(len(X))])
H16, H17, FK = PK[:, 0], PK[:, 1], PK[:, 2]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83 = np.r_[spec | res, sv | rv]
V = {"H3_kurt_1-2k": T["H3_kurt_1-2k"].values, "H4_kurt_2-4k(제외됨)": T["H4_kurt_2-4k"].values, "H5_kurt_4-8k": T["H5_kurt_4-8k"].values, "H6_kurt_8-12.5k(제외됨)": T["H6_kurt_8-12.5k"].values, "H16_peak_h": H16, "H17_peak_w": H17}
n_imp = T["H1_nimp"].values

# %% 그림 1: 첨도 정의 — 1–2 kHz · 4–8 kHz 대역 신호
ex = [("nor_014", "정상 (조용)"), ("nor_006", "정상 (큰 충격 1개)"), ("nor_025", "정상 (오경보, 작은 충격 많음)"), ("abn_010", "고장 (큰 타격 반복)"), ("abn_022", "고장 (미탐)"), ("val_004", "val 004 (J: 고장)")]
ex = [(n, t) for n, t in ex if n in allnames]; fig, axs = plt.subplots(len(ex), 2, figsize=(14, 2.0 * len(ex)), sharex=True)
for row, (n, t) in zip(axs, ex):
    i = allnames.index(n); x = X[i] - X[i].mean(); tt = np.arange(len(x)) / 25000
    for ax, b, tag in ((row[0], 0, "1–2 kHz"), (row[1], 2, "4–8 kHz")):
        lo, hi = OCT[b]; s = sosfiltfilt(_band_sos(lo, hi), x); ax.plot(tt, s, lw=.4, color="#3b7dd8"); ax.set_title(f"{n} — {t} · {tag} 대역 · log10(첨도) = {K[i, b]:.2f} (σ = {s.std():.3f} g)", fontsize=8.5, loc="left")
axs[-1, 0].set_xlabel("초"); axs[-1, 1].set_xlabel("초"); fig.suptitle("H3 · H5 정의: 옥타브 대역통과 신호의 Pearson 첨도(정규분포 = 3)의 log10. 간헐적으로 튀는 신호일수록 큼", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(os.path.join(FIG, "1_첨도_정의.png"), dpi=110); plt.close(fig)

# %% 그림 2: 봉우리 정의 — d = log PSD − 정상 중앙, 최대(▼)와 반높이 폭
ex2 = [("nor_116", "정상 (9.5 kHz 지속음)"), ("nor_065", "정상 (10.2 kHz 지속음)"), ("nor_006", "정상"), ("abn_002", "고장 (8.3k 규칙)"), ("abn_029", "고장 (8.3k 규칙)"), ("val_001", "val 001 (8.3k만 고장)"), ("val_021", "val 021 (경계)")]
ex2 = [(n, t) for n, t in ex2 if n in allnames]; fig, axs = plt.subplots(len(ex2), 1, figsize=(13, 1.9 * len(ex2)), sharex=True)
for ax, (n, t) in zip(axs, ex2):
    i = allnames.index(n); d = L[i] - (np.median(L[[j for j in nor_idx if j != i]], axis=0) if isN[i] else ref_full); k = int(d.argmax())
    ax.plot(fr / 1000, d, lw=.7, color="#3b7dd8"); ax.plot(fr[k] / 1000, d[k], "v", color="#e8743b", ms=8); ax.axhline(d[k] / 2, color="k", ls=":", lw=.6)
    ax.set_title(f"{n} — {t} · H16 높이 {H16[i]:.2f} dec · H17 폭 {H17[i]:.0f} Hz · 봉우리 {FK[i]/1000:.2f} kHz", fontsize=8.5, loc="left"); ax.set_ylabel("dec", fontsize=8)
axs[-1].set_xlabel("kHz"); fig.suptitle("H16 · H17 정의: log10 PSD(1 kHz 위) − 정상 중앙 → 최대 높이(▼) · 최대의 절반 높이(점선) 위 연속 구간 폭", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(os.path.join(FIG, "2_봉우리_정의.png"), dpi=110); plt.close(fig)

# %% 그림 3: 분포 + 첨도 대 충격 수 + 봉우리 주파수
C = {"정상": "#3b7dd8", "고장": "#e8743b", "val": "#2ca02c"}
fig, axs = plt.subplots(2, 3, figsize=(17, 8.4))
for ax, (nm, v) in zip(axs.ravel()[:4], [(k, V[k]) for k in ("H3_kurt_1-2k", "H5_kurt_4-8k", "H16_peak_h", "H17_peak_w")]):
    bins = np.linspace(np.percentile(v, 1), np.percentile(v, 99), 25)
    for g in C: ax.hist(np.clip(v[grp == g], bins[0], bins[-1]), bins=bins, alpha=.5, density=True, color=C[g], label=g)
    ax.set_title(nm); ax.grid(alpha=.2)
axs[0, 0].legend(fontsize=8)
for g in C: axs[1, 1].scatter(n_imp[grp == g], K[grp == g, 0], s=18, color=C[g], alpha=.7, label=g, edgecolor="none"); axs[1, 2].hist(FK[grp == g] / 1000, bins=np.arange(1, 12.6, .25), alpha=.5, density=True, color=C[g], label=g)
axs[1, 1].set_xlabel("H1 = log(1+충격 수)"); axs[1, 1].set_ylabel("H3 (1–2 kHz 첨도)"); axs[1, 1].set_title("첨도는 충격 수와 얼마나 묶이나"); axs[1, 2].set_title("가장 솟은 봉우리의 주파수 (kHz)"); axs[1, 2].legend(fontsize=8)
axs[1, 0].axis("off"); fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_분포.png"), dpi=110); plt.close(fig)

# %% 표
auc = lambda v, a, b: roc_auc_score(np.r_[np.zeros(a.sum()), np.ones(b.sum())], np.r_[v[a], v[b]])
rows = []
for nm, v in V.items():
    a, s = auc(v, isN, isA), auc(v, isN, isV)
    rows.append(dict(피처=nm, 분리도=round(abs(a - .5) * 2, 2), 이동도=round(abs(s - .5) * 2, 2), 정상중앙=round(float(np.median(v[isN])), 2), 고장중앙=round(float(np.median(v[isA])), 2), val중앙=round(float(np.median(v[isV])), 2)))
T1 = pd.DataFrame(rows)
# 첨도가 충격 수를 넘어서는 정보를 주는가: 충격 수 층별 분리도
strata = [("n≤3", n_imp <= np.log1p(3)), ("n=4~9", (n_imp > np.log1p(3)) & (n_imp <= np.log1p(9))), ("n≥10", n_imp > np.log1p(9))]
T2 = []
for s, m in strata:
    for nm in ("H3_kurt_1-2k", "H5_kurt_4-8k"):
        v = V[nm]; ok = (isN & m).sum() > 3 and (isA & m).sum() > 3
        T2.append(dict(층=s, 피처=nm, 정상수=int((isN & m).sum()), 고장수=int((isA & m).sum()), val수=int((isV & m).sum()), 분리도=round(abs(auc(v, isN & m, isA & m) - .5) * 2, 2) if ok else None, val이동도=round(abs(auc(v, isN & m, isV & m) - .5) * 2, 2) if (isV & m).sum() > 3 else None))
T2 = pd.DataFrame(T2)
pf = {g: np.mean((FK[grp == g] >= 8200) & (FK[grp == g] <= 8450)) for g in C}; pk95 = np.percentile(H16[isN], 95)
T3 = pd.DataFrame([dict(그룹=g, 봉우리가_8_2to8_45k=round(float(pf[g]), 2), 봉우리가_9k이상=round(float(np.mean(FK[grp == g] >= 9000)), 2), H16_정상95초과=round(float(np.mean(H16[grp == g] > pk95)), 2), 규칙8_3k켜짐=round(float(r83[grp == g].mean()), 2)) for g in C])
B = [("nor_025", "오경보"), ("nor_055", "오경보"), ("nor_059", "오경보"), ("nor_065", "오경보"), ("nor_116", "오경보"), ("abn_022", "미탐"), ("abn_005", "경계"), ("abn_002", "8.3k규칙"), ("abn_029", "8.3k규칙"), ("val_025", "val경계"), ("val_018", "val경계"), ("val_021", "val경계"), ("val_012", "val경계"), ("val_027", "val경계"), ("val_001", "val(8.3k만)")]
pct = lambda v, i: float((v[isN] <= v[i]).mean() * 100)
T4 = pd.DataFrame([dict(파일=n, 구분=t, **{c: f"{V[c][allnames.index(n)]:.2f} ({pct(V[c], allnames.index(n)):.0f})" for c in ("H3_kurt_1-2k", "H5_kurt_4-8k", "H16_peak_h", "H17_peak_w")}, 봉우리kHz=round(FK[allnames.index(n)] / 1000, 2)) for n, t in B if n in allnames for t in [t]])
D = pd.DataFrame({k: v[:ntr] for k, v in V.items()}); D["H1"] = T["H1_nimp"].values[:ntr]; D["H2"] = T["H2_ratio21"].values[:ntr]
for c in ["H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k", "H12_share_1-2k", "H13_share_2-4k", "H14_share_4-8k", "H15_share_8-12.5k", "H18_hfl", "L1_S"]: D[c[:4]] = T[c].values[:ntr]
Cr = D.corr(method="spearman").loc[["H3_kurt_1-2k", "H5_kurt_4-8k", "H16_peak_h", "H17_peak_w"]].drop(columns=["H3_kurt_1-2k", "H5_kurt_4-8k", "H16_peak_h", "H17_peak_w"]).round(2)
md = lambda df: "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join("" if x is None else str(x) for x in r_) + " |" for r_ in df.values)
txt = [f"정의 검증(모델 H3 · H5와 재계산 최대 차이): {chk:.1e}\n", "분리도 · 이동도 (H16 · H17은 정상 자기 자신을 뺀 중앙 대비)\n" + md(T1), "\n충격 수 층별 첨도 분리도 (첨도가 충격 수 이상을 주는가)\n" + md(T2), "\n봉우리 위치 (8.3k 규칙과의 관계)\n" + md(T3), "\n경계 파일: 값 (정상 내 백분위)\n" + md(T4), "\n학습 150 스피어만 상관\n" + md(Cr.reset_index())]
open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt))
