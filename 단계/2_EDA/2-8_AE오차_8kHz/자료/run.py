# %% [markdown]
# 2-10 AE 칸별 복원 오차와 8 kHz 대역 폭 — EXP-014 회색지대(abn_002·029 ↔ nor_059·062)를 가를 수 있나
# ① AE가 002·029를 높게 보는 이유가 8.2–8.4 kHz 봉우리인가, 059·062의 높은 오차는 어느 주파수에서 오나
#    - EXP-014와 같은 AE(입력: 200 Hz 고역통과 → 1초 × 8구간 log Welch 2048, 정상만 학습 1025-128-16-128-1025, 60 epoch)
#    - 같은 분할 StratifiedGroupKFold(5, shuffle, random_state=0) → 파일마다 바깥 폴드(OOF) 칸별 오차
#    - 칸별 초과 = log10 오차 − 같은 칸 정상 OOF 중앙값 (55 Hz ≈ 5칸 평활)
# ② 8.2–8.4 kHz로 좁히면 나아지나 (사용자 제안). ET 피처 B04_ratio_08125 = 8.125–8.75 kHz 파워 / 전체 파워
#    - 같은 정의로 8.2–8.4 kHz, 그리고 분모를 200 Hz 이상 파워로 바꾼 것도 비교 (전체 파워는 저주파 흔들림이 지배)
#    - 8.2–8.4 kHz는 라벨을 본 뒤 고른 대역 → 성적은 낙관적. 200 Hz 고정 격자 62칸 중 이 칸의 AUC 순위도 같이 본다
# 실행: conda run -n qe-phm python "단계/2_EDA/2-8_AE오차_8kHz/자료/run.py"

# %% 설정
import os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = next(_p for _p in (os.path.abspath(os.path.join(HERE, *[".."] * _k)) for _k in range(8)) if os.path.isdir(os.path.join(_p, "src")))
sys.path.insert(0, ROOT)
from src.exp import load_train, cached_table, feature_table_ext, CACHE
import src.viz  # noqa: F401
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
warnings.filterwarnings("ignore", category=ConvergenceWarning)

FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; N = len(y); nor = y == 0
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
Z8 = np.load(os.path.join(CACHE, "seg_logpsd_hp200_8x1s.npz")); LP, fa = Z8["LP"], Z8["f"]   # EXP-014 캐시 (N, 8, 1025)

# %% ① AE 칸별 OOF 오차
p = os.path.join(HERE, "figs", "ae_binerr_oof.npy")
if os.path.exists(p):
    BE = np.load(p)
else:
    BE = np.empty((N, LP.shape[-1])); g = np.arange(N)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        S = LP[tr[y[tr] == 0]].reshape(-1, LP.shape[-1]); mu, sd = S.mean(0), S.std(0) + 1e-8
        m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3,
                         max_iter=60, alpha=0.0, random_state=0).fit((S - mu) / sd, (S - mu) / sd)
        Z = (LP[te] - mu) / sd; R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
        BE[te] = np.log10(((R - Z) ** 2).mean(1) + 1e-12)
    np.save(p, BE)
EXC = uniform_filter1d(BE - np.median(BE[nor], 0), 5, axis=1)       # 칸별 초과 (dec)
tot = np.log10((10 ** BE).mean(1))                                    # 파일 AE 점수 (EXP-014 ae_scores와 같은 꼴)
POS = np.clip(10 ** BE - 10 ** np.median(BE[nor], 0), 0, None)        # 정상 중앙보다 많은 오차 (선형)
band = lambda lo, hi: (fa >= lo) & (fa < hi)
FOCUS = ["abn_002", "abn_029", "nor_059", "nor_062", "nor_039", "nor_106"]
lines = ["[①] AE 칸별 오차: 파일 점수 = log10 평균 오차 (정상 중 %), 초과 오차 중 각 대역이 차지하는 몫, 초과가 가장 큰 곳"]
pc = lambda v, i: (v[nor] < v[i]).mean() * 100
for n in FOCUS + ["abn_010", "abn_006", "nor_019", "nor_092"]:
    i = names.index(n); sh = lambda lo, hi: POS[i, band(lo, hi)].sum() / POS[i].sum() * 100
    top = fa[EXC[i].argmax()]
    lines.append(f"  {n}: 점수 {tot[i]:.2f} ({pc(tot,i):.0f}%) | 초과 몫 8.2–8.4k {sh(8200,8400):.0f}% · 8.4–8.8k {sh(8400,8800):.0f}% · "
                 f"0.2–2k {sh(200,2000):.0f}% · 2–8.2k {sh(2000,8200):.0f}% · 8.8–12.5k {sh(8800,12501):.0f}% | 최대 초과 {top/1000:.2f} kHz ({EXC[i].max():.2f} dec)")
lines.append(f"  (참고: 8.2–8.4 kHz는 1025칸 중 {band(8200,8400).sum()}칸 = {band(8200,8400).mean()*100:.1f}%)")

# %% ② 대역 비율: B04 정의(전체 파워 분모) vs 200 Hz 이상 분모, 8.125–8.75 vs 8.2–8.4
Xc = X - X.mean(1, keepdims=True)
pw = np.abs(np.fft.rfft(Xc, axis=1)) ** 2; fq = np.fft.rfftfreq(X.shape[1], 1 / FS)
tot_all, tot_hp = pw.sum(1), pw[:, fq >= 200].sum(1)
bp = lambda lo, hi: pw[:, (fq >= lo) & (fq < hi)].sum(1)
V = {"8.125–8.75 / 전체 (=ET B04)": bp(8125, 8750) / tot_all, "8.2–8.4 / 전체": bp(8200, 8400) / tot_all,
     "8.125–8.75 / 200 Hz 이상": bp(8125, 8750) / tot_hp, "8.2–8.4 / 200 Hz 이상": bp(8200, 8400) / tot_hp}
Ec = cached_table("ext", feature_table_ext, X)["B04_ratio_08125"].values
lines += ["", f"[②] 확인: 직접 계산한 8.125–8.75/전체와 ET 피처 B04_ratio_08125 상관 {np.corrcoef(Ec, V['8.125–8.75 / 전체 (=ET B04)'])[0,1]:.4f}"]
lines.append("대역 비율 | AUC | 002 · 029 정상 중 % | 002·029 중 작은 값보다 큰 정상 수 (그 정상들) | 059 · 062 · 019 · 092 정상 중 %")
for k, v in V.items():
    i2, i9 = names.index("abn_002"), names.index("abn_029"); lo = min(v[i2], v[i9])
    above = [names[j] for j in np.flatnonzero(nor & (v >= lo))]
    lines.append(f"  {k}: AUC {roc_auc_score(y, v):.2f} | {pc(v,i2):.0f} · {pc(v,i9):.0f} | {len(above)} ({', '.join(above[:8])}{' …' if len(above) > 8 else ''}) | "
                 + " · ".join(f"{pc(v, names.index(n)):.0f}" for n in ["nor_059", "nor_062", "nor_019", "nor_092"]))
# 200 Hz 고정 격자에서 8.2–8.4 칸의 순위 (200 Hz 이상 분모)
edges = np.arange(200, 12600, 200); aucs = []
for lo in edges[:-1]:
    v = bp(lo, lo + 200) / tot_hp; aucs.append((roc_auc_score(y, v), lo))
rk = sorted(aucs, reverse=True); pos = [lo for _, lo in rk].index(8200) + 1
lines.append(f"200 Hz 고정 격자 {len(aucs)}칸 (200 Hz 이상 분모) 단변량 AUC: 8.2–8.4 kHz 칸은 {pos}위 ({dict((l, a) for a, l in aucs)[8200]:.2f}). "
             "상위 8: " + ", ".join(f"{lo/1000:.1f}k {a:.2f}" for a, lo in rk[:8]))
open(os.path.join(HERE, "figs", "numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))

# %% 그림 1: 칸별 AE 초과 오차 (002·029 vs 059·062·039·106)
fig, ax = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
for a, grp, title in [(ax[0], ["abn_002", "abn_029"], "ET가 놓친 고장"), (ax[1], ["nor_059", "nor_062", "nor_039", "nor_106"], "AE 오경보 정상 (회색지대)")]:
    a.fill_between(fa / 1000, np.percentile(EXC[nor], 5, 0), np.percentile(EXC[nor], 95, 0), color="#2a78d6", alpha=0.15, lw=0, label="정상 5–95%")
    cols = ["#eb6834", "#c4471c"] if grp[0].startswith("abn") else ["#184f95", "#3987e5", "#6da7ec", "#9ec5f4"]
    for n, c in zip(grp, cols):
        i = names.index(n); a.plot(fa / 1000, EXC[i], color=c, lw=1.2, label=f"{n} (AE 점수 정상 중 {pc(tot,i):.0f}%)")
    a.axvspan(8.2, 8.4, color="#f4b391", alpha=0.35, lw=0)
    a.axhline(0, color="#85847e", lw=0.6)
    a.set_title(f"{title}: AE 칸별 복원 오차 - 정상 중앙 (dec, 음영 = 8.2–8.4 kHz)", loc="left", fontsize=11)
    a.set_ylabel("dec"); a.legend(frameon=False, fontsize=8, ncol=3, loc="upper left"); a.grid(color="#e6e5e0", lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
ax[-1].set_xlabel("kHz")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "ae_binerr.png"), dpi=120)

# %% 그림 2: 대역 비율 4가지 — 정상/고장 점 + 회색지대 파일 이름
fig, ax = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
rng = np.random.default_rng(0)
NAMED = {"abn_002", "abn_029", "nor_059", "nor_062", "nor_019", "nor_092", "nor_116", "nor_055", "nor_025"}
for a, (k, v) in zip(ax, V.items()):
    lv = np.log10(v + 1e-12)
    for g_, c, xo in [(0, "#2a78d6", 0), (1, "#eb6834", 1)]:
        idx = np.flatnonzero(y == g_); xs = xo + rng.uniform(-0.2, 0.2, len(idx))
        a.scatter(xs, lv[idx], s=16, color=c, alpha=0.6, edgecolors="white", linewidths=0.8)
        for xx, j in zip(xs, idx):
            if names[j] in NAMED:
                a.annotate(names[j][:1] + names[j][4:], (xx, lv[j]), xytext=(5, 0), textcoords="offset points", fontsize=7, va="center", color="#0b0b0b")
    a.axhline(np.percentile(lv[nor], 95), color="#85847e", lw=0.8, ls=":")
    a.set_xticks([0, 1], ["정상", "고장"]); a.set_title(f"{k}\nAUC {roc_auc_score(y, v):.2f}", loc="left", fontsize=10)
    a.set_ylabel("log10 비율"); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.spines[["top", "right"]].set_visible(False)
fig.suptitle("8 kHz대 에너지 비율: 대역 폭(8.125–8.75 vs 8.2–8.4)과 분모(전체 vs 200 Hz 이상). 점선 = 정상 95%, n = 정상, a = 고장",
             x=0.01, ha="left", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(os.path.join(HERE, "figs", "band_ratio.png"), dpi=120)
print("done")
