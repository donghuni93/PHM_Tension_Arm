# %% [markdown]
# 4-5 후반. val의 AE 점수가 왜 이동하나 — 주파수 대역별 · 게인 확인 (라벨 안 씀)
# AE 입력 = 구간별 log PSD(1025 bin)를 정상 평균·표준편차로 표준화한 값. 점수 = 복원 오차 평균(log10).
# (1) 표준화 값의 파일 평균을 대역별로 비교: val이 정상 평균에서 어느 대역이 얼마나 벗어나나 (2) AE 복원 오차를 대역별로 비교 (3) 전체 게인(log PSD 평균)
# 대역 = 0–12.5 kHz 8등분(라벨과 무관하게 고정). train 정상의 오차는 정상 5겹 OOF(그 파일을 학습에 안 쓴 AE), 고장 · val은 정상 120개 전체로 학습한 AE.
# 실행: python "3_단계/4_피처/4-5_1120_D플러스_val_tSNE/run_AE이동원인.py"
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
from sklearn.model_selection import KFold
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.dplus import DPlus
from src.final import _fit_ae
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs")
X, y, names = load_train(); E = cached_table("ext", feature_table_ext, X)
dp = DPlus().fit(X, E); Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val")); Dv, Tv = dp.transform(Xv)
LP, LPv = dp.T_["LP"], Tv["LP"]; print("LP shape", LP.shape, LPv.shape)
nb = LP.shape[-1]; f = np.linspace(0, 12500, nb); edges = np.linspace(0, 12500, 9); band = np.digitize(f, edges[1:-1])    # 0..7
blab = [f"{edges[k]/1000:.1f}–{edges[k+1]/1000:.1f}" for k in range(8)]
N = np.where(y == 0)[0]; Fl = np.where(y == 1)[0]
mu, sd = LP[N].reshape(-1, nb).mean(0), LP[N].reshape(-1, nb).std(0) + 1e-8

# %% (1) 표준화 값의 파일 평균 (구간 평균) — 대역별
def zmean(L): return ((L - mu) / sd).mean(1)                                    # (파일, bin)
Zn, Zf, Zv = zmean(LP[N]), zmean(LP[Fl]), zmean(LPv)
def bmean(Zm): return np.stack([Zm[:, band == k].mean(1) for k in range(8)], 1)  # (파일, 8대역)
Bn, Bf, Bv = bmean(Zn), bmean(Zf), bmean(Zv)
print("\n(1) 표준화 평균 Z (정상 기준 0) — 대역별 중앙값")
print(pd.DataFrame({"대역kHz": blab, "train정상": np.median(Bn, 0).round(2), "train고장": np.median(Bf, 0).round(2), "val": np.median(Bv, 0).round(2)}).to_string(index=False))

# %% (2) AE 복원 오차 — 대역별 (정상 OOF / 고장 · val: 정상 전체로 학습한 AE)
def berr(ae, L):
    mu_, sd_, m = ae; Zs = (L - mu_) / sd_; R = m.predict(Zs.reshape(-1, nb)).reshape(Zs.shape); err = ((R - Zs) ** 2).mean(1)    # (파일, bin) 구간 평균
    return np.stack([err[:, band == k].mean(1) for k in range(8)], 1)
En = np.zeros((len(N), 8))
for tr, te in KFold(5, shuffle=True, random_state=0).split(N):
    En[te] = berr(_fit_ae(LP[N[tr]]), LP[N[te]])
ae_full = _fit_ae(LP[N]); Ef, Ev = berr(ae_full, LP[Fl]), berr(ae_full, LPv)
print("\n(2) AE 복원 오차 (log10) — 대역별 중앙값 · 전체 오차 합 중 몫")
tot = lambda M: np.log10(M.mean(1))
share = lambda M: (M / M.sum(1, keepdims=True))
print(pd.DataFrame({"대역kHz": blab, "정상OOF": np.log10(np.median(En, 0)).round(2), "고장": np.log10(np.median(Ef, 0)).round(2), "val": np.log10(np.median(Ev, 0)).round(2),
                    "val−정상(log10)": (np.log10(np.median(Ev, 0)) - np.log10(np.median(En, 0))).round(2), "몫 정상": np.median(share(En), 0).round(2), "몫 val": np.median(share(Ev), 0).round(2)}).to_string(index=False))
print(f"전체 점수(log10 평균 오차) 중앙값: 정상OOF {np.median(tot(En)):.2f} · 고장 {np.median(tot(Ef)):.2f} · val {np.median(tot(Ev)):.2f}")

# %% (3) 게인: 파일 평균 log PSD
g_n, g_f, g_v = LP[N].mean((1, 2)), LP[Fl].mean((1, 2)), LPv.mean((1, 2))
print(f"\n(3) 전체 게인(log PSD 평균) 중앙값: 정상 {np.median(g_n):.3f} · 고장 {np.median(g_f):.3f} · val {np.median(g_v):.3f} | 정상 표준편차 {g_n.std():.3f} → val은 정상 중앙에서 {(np.median(g_v) - np.median(g_n)) / g_n.std():+.2f} sd")
print(f"val 중 게인이 정상 5–95% 밖: {int(((g_v < np.percentile(g_n, 5)) | (g_v > np.percentile(g_n, 95))).sum())}/40")
pd.DataFrame({"file": nv, "gain": g_v.round(3), **{f"Z_{blab[k]}": Bv[:, k].round(2) for k in range(8)}, **{f"err_{blab[k]}": np.log10(Ev[:, k]).round(2) for k in range(8)}}).to_csv(os.path.join(HERE, "val_AE이동_대역표.csv"), index=False, encoding="utf-8-sig")

# %% 그림
NOR, ABN, PUR = "#2a78d6", "#eb6834", "#7B1FA2"
fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
a = ax[0]
for Zm, c, lb in ((Zn, NOR, "train 정상"), (Zf, ABN, "train 고장"), (Zv, PUR, "val 40")): a.plot(f / 1000, np.median(Zm, 0), color=c, lw=2, label=lb); a.fill_between(f / 1000, np.percentile(Zm, 25, 0), np.percentile(Zm, 75, 0), color=c, alpha=.15)
a.axhline(0, color="k", lw=.6); a.set_xlabel("주파수 (kHz)"); a.set_ylabel("표준화 log PSD (정상 평균 = 0)"); a.set_title("(1) 대역별 표준화 값 (중앙값 · 사분위)"); a.legend(fontsize=8)
a = ax[1]; w_ = .27; xs = np.arange(8)
for k, (M, c, lb) in enumerate(((En, NOR, "정상 OOF"), (Ef, ABN, "고장"), (Ev, PUR, "val"))):
    a.bar(xs + (k - 1) * w_, np.log10(np.median(M, 0)), w_, color=c, label=lb)
a.set_xticks(xs); a.set_xticklabels(blab, fontsize=8, rotation=20); a.set_xlabel("대역 (kHz)"); a.set_ylabel("AE 복원 오차 log10 (중앙값)"); a.set_title("(2) AE 복원 오차 — 대역별"); a.legend(fontsize=8)
a = ax[2]; bins = np.linspace(min(g_n.min(), g_v.min(), g_f.min()), max(g_n.max(), g_v.max(), g_f.max()), 30)
a.hist(g_n, bins, color=NOR, alpha=.55, density=True, label="train 정상"); a.hist(g_f, bins, color=ABN, alpha=.55, density=True, label="train 고장"); a.hist(g_v, bins, histtype="step", color=PUR, lw=2, density=True, label="val 40")
a.set_xlabel("파일 평균 log PSD (게인)"); a.set_title("(3) 전체 게인 분포"); a.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "AE이동원인_대역·게인.png"), dpi=115, bbox_inches="tight"); plt.close(fig); print("완료")
