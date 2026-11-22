# %% [markdown]
# 2-13 유사 파일 — 022처럼 주기에 잠긴 충격이 있는 파일이 또 있나 (사용자 질문 11-14)
# - 입력: run.py가 만든 figs/lock_score.csv (S·r_LP·T·hit)
# - 그림 4: S 상위 9개 파형 (느린 파형 + T만큼 민 것 + 8σ 사건)
# - 그림 5: 022 · 014 · 001 비교 (사건 굵기 = 높이) + 느린 파형 r(τ)
# - 그림 6: 022 · 014 와 느린 파형만 반복되는 정상(r_LP ≥ 0.85: 051 · 042 · 034)
# 실행: conda run -n qe-phm python "단계/2_EDA/2-6_주기잠금_022/run_similar.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt

FIG = os.path.join(HERE, "figs")
X, y, names = load_train(); names = [str(n) for n in names]
FS = 25000; t = np.arange(X.shape[1]) / FS
A = pd.read_csv(os.path.join(FIG, "lock_score.csv"), index_col=0)
NOR, ABN = "#2a78d6", "#eb6834"


def prep(n):
    i = names.index(n); x = X[i] - X[i].mean()
    L = sosfiltfilt(butter(4, 30, "low", fs=FS, output="sos"), x)
    h = sosfiltfilt(butter(4, 200, "high", fs=FS, output="sos"), x)
    sig = 1.4826 * np.median(np.abs(h - np.median(h)))
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if len(idx):
        g = np.split(idx, np.flatnonzero(np.diff(idx) > int(0.01 * FS)) + 1)
        ev = np.array([gg[np.argmax(np.abs(h[gg]))] for gg in g])
    else:
        ev = np.array([], dtype=int)
    return i, x, L, h, sig, ev


def wave_panel(a, n, shifted=True):
    i, x, L, h, sig, ev = prep(n); T = A.loc[n, "T"]
    a.plot(t, x, lw=.3, color="0.6"); a.plot(t, L, lw=1.6, color=ABN if y[i] else NOR)
    if shifted: a.plot(t + T, L, lw=1, ls="--", color="#7a3fbf")
    for e in ev:
        a.axvline(e / FS, color="k", lw=.5 + 1.5 * min(abs(h[e]) / sig / 40, 1), alpha=.6)
    a.set_title(f"{n}  S {A.loc[n,'S']:.2f} · r_LP {A.loc[n,'r_LP']:.2f} · T {T:.2f} s · hit {A.loc[n,'hit']:.2f} · 사건 {len(ev)}",
                fontsize=9, loc="left")
    lo, hi = np.percentile(x, [.5, 99.5]); a.set_ylim(lo - .3 * (hi - lo), hi + .3 * (hi - lo)); a.set_xlim(0, 2)


# %% 그림 4: S 상위 9
top = list(A.sort_values("S", ascending=False).head(9).index)
fig, axs = plt.subplots(len(top), 1, figsize=(16, 2.1 * len(top)), sharex=True)
for a, n in zip(axs, top): wave_panel(a, n)
axs[-1].set_xlabel("s")
fig.suptitle("S 상위 9개 — 실선 = 30 Hz 저역, 점선 = T만큼 민 것, 세로선 = 8σ 사건(굵을수록 큼)", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_top_S.png"), dpi=70); plt.close(fig)

# %% 그림 5: 022 · 014 · 001 + r(τ)
lines = []
FILES = ["abn_022", "abn_014", "abn_001"]
fig, axs = plt.subplots(len(FILES), 2, figsize=(19, 3.2 * len(FILES)), gridspec_kw={"width_ratios": [3.2, 1]})
for r, n in enumerate(FILES):
    i, x, L, h, sig, ev = prep(n)
    wave_panel(axs[r, 0], n)
    Ld = L[::50]; nn = len(Ld); ks = np.arange(5, int(1.5 * 500), 2)
    rr = np.array([np.corrcoef(Ld[:nn - k], Ld[k:])[0, 1] for k in ks])
    m = ks >= 150                                         # 0.3 s 이상에서 최대
    axs[r, 1].plot(ks / 500, rr, color=ABN); axs[r, 1].axhline(0, color="k", lw=.5); axs[r, 1].axvspan(.3, 1.2, color="0.9")
    axs[r, 1].set_ylim(-1, 1); axs[r, 1].grid(alpha=.3); axs[r, 1].set_title(f"{n}: 느린 파형 r(τ), 회색 = 탐색 범위", fontsize=9)
    lines.append(f"== {n}: r(τ) 최대 {rr[m].max():.2f} @ {ks[m][rr[m].argmax()]/500:.2f} s · 사건 (시각 s · 높이/σ · 저주파 g)")
    for e in ev:
        lines.append(f"   {e/FS:.3f}  {abs(h[e])/sig:6.1f}  {L[e]:+.3f}")
axs[-1, 0].set_xlabel("s"); axs[-1, 1].set_xlabel("τ (s)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "5_022_014_001.png"), dpi=70); plt.close(fig)

# %% 그림 6: 느린 파형만 반복되는 정상
SH = ["abn_022", "abn_014"] + list(A[(A.y == 0) & (A.r_LP >= .85)].sort_values("r_LP", ascending=False).index)
fig, axs = plt.subplots(len(SH), 1, figsize=(16, 2.1 * len(SH)), sharex=True)
for a, n in zip(axs, SH): wave_panel(a, n)
axs[-1].set_xlabel("s")
fig.suptitle("022 · 014 vs 느린 파형 반복 상관이 높은 정상(r_LP ≥ 0.85) — 정상은 작은 빠른 출렁임·드리프트이고 충격 짝이 없다", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "6_sway_normals.png"), dpi=70); plt.close(fig)

open(os.path.join(FIG, "similar.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))
