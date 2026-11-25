# %% 2-35 충격 검출기 비교: 같은 파일에서 정의마다 충격 개수 · 위치가 얼마나 다른가 (train 150 + val 40, 라벨은 색칠에만 씀)
#  A = 우리(2차 설계): 1 kHz 고역 포락선 ÷ 100 ms 국소 바탕 ≥ 4.5, 간격 ≥ 6.75 ms, 앞뒤 10 ms 제외
#  B = 효섭 · 소린(1119 최종모델): 10 Hz 고역 → AR(40) 백색화 → |잔차| ≥ 8σ(MAD), 간격 ≥ 10 ms, 앞뒤 20 ms 제외
#  C = 효섭 1차(1116 보고서): 1 kHz 고역 포락선(500 Hz 평활) ≥ 파일 전체 중앙값 × 6, 간격 ≥ 5 ms
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import linalg, signal
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp2util import NS
from src.impacts import detect_impacts, highpass, FS
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
_av = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((n for n in ["Malgun Gothic", "NanumGothic", "WenQuanYi Zen Hei"] if n in _av), "DejaVu Sans"); plt.rcParams["axes.unicode_minus"] = False

ns = NS(); y = ns.y; names = list(ns.names); nvn = ["val_" + n[-3:] for n in ns.nv]
allnames = names + nvn; X = np.vstack([ns.X, ns.Xv]); ntr = len(names); N = len(allnames)
grp = np.array(["정상" if v == 0 else "고장" for v in y] + ["val"] * len(nvn))
OUT = ("nor_059", "nor_065", "nor_116"); isout = np.array([n in OUT for n in allnames])
GCOL = {"정상": "#3b7dd8", "고장": "#e8743b", "val": "#8a8a8a"}

# %% 검출기 세 개 — 모두 충격 시각(초) 배열을 돌려준다
_sos = lambda kind, f: signal.butter(4, f, btype=kind, fs=FS, output="sos")
SOS_HP10, SOS_HP1K, SOS_LP500 = _sos("highpass", 10), _sos("highpass", 1000), _sos("lowpass", 500)
TR, P = 500, 40

def det_A(x): return detect_impacts(x)["t"]

def ar_whiten(x, p=P):
    r = np.correlate(x[:20000], x[:20000], "full")[19999:19999 + p + 1]
    a = linalg.solve_toeplitz(r[:p], r[1:p + 1])
    return signal.lfilter(np.r_[1, -a], [1], x)[p:]

def det_B(x):
    x = x - x.mean(); xh = signal.sosfiltfilt(SOS_HP10, x); w = ar_whiten(xh)[TR:-TR]    # 효섭 코드와 같은 순서
    sig = np.median(np.abs(w)) / 0.6745
    pk = signal.find_peaks(np.abs(w), height=8 * sig, distance=250)[0]
    return (pk + P + TR) / FS                                                             # 원 신호 시각으로 되돌림

def det_C(x):
    x = x - x.mean(); xs = signal.sosfiltfilt(SOS_HP1K, x)
    e = signal.sosfiltfilt(SOS_LP500, np.abs(signal.hilbert(xs)))
    pk = signal.find_peaks(e, height=6 * np.median(e), distance=int(round(0.005 * FS)))[0]
    return pk / FS

DET = {"A 우리 (포락선÷국소바탕 4.5)": det_A, "B 효섭·소린 (AR 백색화 8σ)": det_B, "C 효섭 1차 (포락선 > 중앙값×6)": det_C}
KEYS = list(DET); SHORT = ["A", "B", "C"]
T = {k: [f(x) for x in X] for k, f in DET.items()}
cnt = pd.DataFrame({s: [len(t) for t in T[k]] for s, k in zip(SHORT, KEYS)}, index=allnames); cnt["group"] = grp; cnt["outlier"] = isout
cnt.to_csv(os.path.join(HERE, "충격수_파일별.csv"), encoding="utf-8-sig")

# %% 그림 1: 파일당 충격 개수 — 정의 3개 × 그룹 3개 (점 = 파일, ◆ = 이상치 정상 3개, 가로선 = 중앙값)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.6), sharey=False); rng = np.random.default_rng(0)
for ax, s, k in zip(axs, SHORT, KEYS):
    for j, g in enumerate(["정상", "고장", "val"]):
        m = grp == g; v = cnt.loc[m, s].values; jit = j + rng.uniform(-.28, .28, m.sum()); o = isout[m]
        ax.scatter(jit[~o], v[~o], s=14, color=GCOL[g], alpha=.65, lw=0); ax.scatter(jit[o], v[o], s=55, marker="D", color=GCOL[g], edgecolor="k", lw=.8)
        ax.hlines(np.median(v), j - .38, j + .38, color="k", lw=2); ax.text(j + .4, np.median(v), f"{np.median(v):g}", va="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels([f"정상 ({(grp=='정상').sum()})", f"고장 ({(grp=='고장').sum()})", f"val ({(grp=='val').sum()})"]); ax.set_title(k, fontsize=10); ax.set_ylabel("파일당 충격 개수"); ax.grid(alpha=.25)
fig.suptitle("같은 190개 파일, 정의별 충격 개수 (검은 선 = 중앙값, ◆ = 이상치 정상 059·065·116)", fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_개수분포.png"), dpi=130); plt.close(fig)

# %% 그림 2: 충격이 생긴 시각의 분포 — 0~2 s를 50 ms 칸으로 나눠 "파일당 충격 수" (그룹별, 정의별)
bins = np.arange(0, 2.0001, 0.05); ctr = (bins[:-1] + bins[1:]) / 2
fig, axs = plt.subplots(3, 1, figsize=(14, 8.5), sharex=True)
for ax, s, k in zip(axs, SHORT, KEYS):
    for g in ["정상", "고장", "val"]:
        idx = np.where(grp == g)[0]; h = sum(np.histogram(T[k][i], bins)[0] for i in idx) / len(idx)
        ax.step(ctr, h, where="mid", color=GCOL[g], lw=1.8, label=g)
    ax.set_ylabel("파일당 충격 수 / 50 ms"); ax.set_title(k, fontsize=10, loc="left"); ax.grid(alpha=.25); ax.set_ylim(bottom=0)
axs[0].legend(ncol=3, loc="upper center"); axs[-1].set_xlabel("시간 (s)")
fig.suptitle("충격이 생긴 위치(시각) 분포 — 한쪽 끝에 몰리면 검출기의 끝단 처리 영향", fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_위치분포.png"), dpi=130); plt.close(fig)

# %% 그림 3: 파일별 충격 위치 (줄 = 파일, 점 = 충격 하나). 정의 3개 나란히, 위 정상 → 고장 → val
order = np.concatenate([np.where(grp == g)[0] for g in ["정상", "고장", "val"]]); cut = [(grp == "정상").sum(), (grp == "정상").sum() + (grp == "고장").sum()]
fig, axs = plt.subplots(1, 3, figsize=(16, 13), sharey=True)
for ax, s, k in zip(axs, SHORT, KEYS):
    for r, i in enumerate(order):
        t = T[k][i]; ax.plot(t, np.full(len(t), r), "|", color=GCOL[grp[i]], ms=3.2, mew=.9)
        if isout[i]: ax.plot(-0.06, r, "D", color=GCOL[grp[i]], ms=4, mec="k", mew=.6, clip_on=False)
    for c in cut: ax.axhline(c - .5, color="k", lw=.8)
    ax.set_xlim(0, 2); ax.set_ylim(N - .5, -.5); ax.set_title(k, fontsize=10); ax.set_xlabel("시간 (s)")
axs[0].set_yticks([cut[0] / 2, (cut[0] + cut[1]) / 2, (cut[1] + N) / 2]); axs[0].set_yticklabels(["정상", "고장", "val"])
fig.suptitle("파일별 충격 위치 (◆ = 이상치 정상)", fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_파일별위치.png"), dpi=110); plt.close(fig)

# %% 그림 4: 정의 사이 일치 — A와 B가 같은 충격을 잡았나 (±5 ms 안에서 짝이 있으면 일치)
def match(a, b, tol=0.005):
    return int(sum(np.any(np.abs(b - t) <= tol) for t in a)) if len(a) and len(b) else 0
pairs = [("A", "B"), ("A", "C"), ("B", "C")]; kk = dict(zip(SHORT, KEYS)); rows = []
for a, b in pairs:
    for i in range(N):
        ta, tb = T[kk[a]][i], T[kk[b]][i]; rows.append((a + b, allnames[i], grp[i], len(ta), len(tb), match(ta, tb), match(tb, ta)))
M = pd.DataFrame(rows, columns=["쌍", "파일", "그룹", "n_앞", "n_뒤", "앞중_일치", "뒤중_일치"]); M.to_csv(os.path.join(HERE, "정의간_일치.csv"), index=False, encoding="utf-8-sig")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (a, b) in zip(axs, pairs):
    for g in ["정상", "고장", "val"]:
        m = grp == g; ax.scatter(cnt.loc[m, a] + .3, cnt.loc[m, b] + .3, s=16, color=GCOL[g], alpha=.6, lw=0, label=g)
    ax.set_xscale("log"); ax.set_yscale("log"); lim = [.25, max(cnt[a].max(), cnt[b].max()) * 1.6]; ax.plot(lim, lim, "k--", lw=.8); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel(f"{a} 충격 수 (+0.3)"); ax.set_ylabel(f"{b} 충격 수 (+0.3)"); ax.grid(alpha=.25); ax.set_title(f"{a} vs {b}")
axs[0].legend(); fig.suptitle("같은 파일에서 두 정의가 센 충격 수 (점선 = 같은 수)", fontsize=11); fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_정의간_개수.png"), dpi=130); plt.close(fig)

# %% 요약 수치 → 결과.md
L = ["# 2-35 충격 검출기 비교 — 수치 (run.py 자동 생성)", "", "## 파일당 충격 수 (중앙값 [사분위]) · 충격 0개 비율", "", "| 정의 | 정상 | 고장 | val | 0개: 정상 · 고장 · val |", "|---|---|---|---|---|"]
for s, k in zip(SHORT, KEYS):
    c = []
    for g in ["정상", "고장", "val"]:
        v = cnt.loc[grp == g, s]; c.append(f"{v.median():g} [{v.quantile(.25):g}–{v.quantile(.75):g}]")
    z = " · ".join(f"{(cnt.loc[grp == g, s] == 0).mean():.0%}" for g in ["정상", "고장", "val"]); L.append(f"| {k} | {c[0]} | {c[1]} | {c[2]} | {z} |")
L += ["", "## 충격 수 순위 상관 (스피어만, 190개)", ""]
for a, b in pairs: L.append(f"- {a}–{b}: {cnt[a].corr(cnt[b], method='spearman'):.2f}")
L += ["", "## 위치: 파일 끝 100 ms(1.9–2.0 s) · 시작 100 ms(0–0.1 s)에 있는 충격 비율", "", "| 정의 | 시작 100 ms | 끝 100 ms |", "|---|---|---|"]
for k in KEYS:
    allt = np.concatenate(T[k]); L.append(f"| {k} | {(allt < .1).mean():.1%} | {(allt > 1.9).mean():.1%} |")
L += ["", "## 일치율 (±5 ms 안에 상대 정의의 충격이 있는 비율, 190개 합)", ""]
for a, b in pairs:
    m = M[M["쌍"] == a + b]; L.append(f"- {a}→{b}: {m['앞중_일치'].sum() / max(1, m['n_앞'].sum()):.0%} ({a}의 충격 {m['n_앞'].sum()}개 중 {b}도 잡은 것) · {b}→{a}: {m['뒤중_일치'].sum() / max(1, m['n_뒤'].sum()):.0%} ({b}의 충격 {m['n_뒤'].sum()}개 중)")
open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
