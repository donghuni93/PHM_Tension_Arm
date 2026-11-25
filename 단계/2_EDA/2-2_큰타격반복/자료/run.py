# %% [markdown]
# 2-14 불안정 고장 005·020 — 가정한 고장 유형(1-3)으로 찾는 확실한 특징 (사용자 요청 11-14)
# 과적합 방지: 분리 결과를 보기 전에 1-3 열화 단계 가설에서 피처 정의를 고정하고,
#   (1) 정상 중 몇 개가 위에 오나 (2) 같은 유형의 다른 고장도 올라오나 (3) 설정을 바꿔도 유지되나 를 본다
# 사건 = 200 Hz 고역 |h| > kσ (σ = 1.4826·MAD), 병합 폭 안은 한 사건 (2-12와 같은 정의)
#
# 사전 고정 피처
#   L2 부싱 유격 확대  F1 저주파(<20 Hz) 분산 비율 × log10(고역 첨도)  ← 1-3 탐지 방향 원문
#                      F2 20 ms 창 |저주파 변위|와 고역 RMS의 스피어만 상관 (변위가 클 때 친다)
#   L3 연속 충돌       F3 100 ms 안 8σ 사건(2 ms 병합) 최대 개수 (연속 충돌 열)
#                      F4 사건 뒤 3–30 ms 안에 다음 사건이 있는 비율 (되튐)
#   L1 초기 미세 열화  F5 0.1 s 창 고역 RMS 최대/중앙 (서브윈도우 max-pooling)
#                      F6 log10(9–12.5 kHz 파워 / 200 Hz 이상 파워)
#   H1′ 일반화         K1_k = log10(k번째 큰 사건 / σ), k = 2–6 (2위/1위 R 대신 바탕 대비 k번째 타격 크기)
# 실행: conda run -n qe-phm python "단계/2_EDA/2-2_큰타격반복/자료/run.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, welch
from scipy.stats import kurtosis, spearmanr

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); names = [str(n) for n in names]; NM = np.array(names)
FS = 25000; nor = y == 0; t = np.arange(X.shape[1]) / FS
X = X - X.mean(1, keepdims=True)
HP = sosfiltfilt(butter(4, 200, "high", fs=FS, output="sos"), X, axis=1)
LP = sosfiltfilt(butter(4, 20, "low", fs=FS, output="sos"), X, axis=1)
SIG = 1.4826 * np.median(np.abs(HP - np.median(HP, 1, keepdims=True)), 1)
f, P = welch(X, FS, nperseg=4096, axis=1)
i5, i20 = names.index("abn_005"), names.index("abn_020")
NOR, ABN = "#2a78d6", "#eb6834"


def events(i, k=8, merge=0.010):
    h = HP[i]; s = SIG[i]
    idx = np.flatnonzero(np.abs(h) > k * s)
    if not len(idx): return np.array([]), np.array([])
    g = np.split(idx, np.flatnonzero(np.diff(idx) > int(merge * FS)) + 1)
    p = np.array([gg[np.argmax(np.abs(h[gg]))] for gg in g])
    return p / FS, np.abs(h[p]) / s


def train(i, win=0.1, merge=0.002, k=8):
    ev, _ = events(i, k, merge)
    return max(((ev >= e) & (ev < e + win)).sum() for e in ev) if len(ev) else 0


def kth(i, kk, k=8, merge=0.010):
    h = np.sort(events(i, k, merge)[1])[::-1]
    return np.log10(max(h[kk - 1], 1.0)) if len(h) >= kk else 0.0


def line(name, v, files=(i5, i20)):
    vn = v[nor]; top = np.flatnonzero(nor)[vn.argmax()]
    s = f"{name:26s} 정상 95% {np.percentile(vn, 95):7.3f} 최대 {vn.max():7.3f}({names[top]})"
    for k in files:
        s += f" | {names[k][4:]} {v[k]:7.3f} 백분위 {(vn < v[k]).mean()*100:5.1f} 위 정상 {(vn >= v[k]).sum():3d}"
    s += f" | 정상 95% 넘는 고장 {[names[j][4:] for j in np.flatnonzero((~nor) & (v > np.percentile(vn, 95)))]}"
    return s


# %% 1. 사전 고정 가설 피처
F = pd.DataFrame(index=names)
F["F1_L2_interact"] = [LP[i].var() / X[i].var() * np.log10(kurtosis(HP[i], fisher=False)) for i in range(len(y))]
w = int(0.02 * FS); m = X.shape[1] // w * w
F["F2_L2_gate"] = [spearmanr(np.abs(LP[i, :m] - np.median(LP[i])).reshape(-1, w).mean(1),
                             np.sqrt((HP[i, :m] ** 2).reshape(-1, w).mean(1)))[0] for i in range(len(y))]
F["F3_L3_train"] = [train(i) for i in range(len(y))]
def rehit(i):
    ev, _ = events(i, 8, 0.002)
    if not len(ev): return 0.0
    d = np.diff(ev); return ((d >= .003) & (d <= .03)).sum() / len(ev)
F["F4_L3_rehit"] = [rehit(i) for i in range(len(y))]
w = int(0.1 * FS); m = X.shape[1] // w * w
F["F5_L1_burst"] = [(lambda r: r.max() / np.median(r))(np.sqrt((HP[i, :m] ** 2).reshape(-1, w).mean(1))) for i in range(len(y))]
F["F6_L1_hf_ratio"] = np.log10(P[:, (f >= 9000) & (f <= 12500)].sum(1) / P[:, f >= 200].sum(1))
for kk in (2, 3, 4, 5, 6):
    F[f"K1_{kk}"] = [kth(i, kk) for i in range(len(y))]
F["y"] = y
F.to_csv(os.path.join(FIG, "features.csv"))
lines = ["## 1. 사전 고정 피처 — 005 · 020의 정상 중 위치"]
for c in F.columns[:-1]: lines.append(line(c, F[c].values))

# %% 2. 민감도
lines.append("\n## 2a. 005 — K1_k (k번째 큰 사건/σ) 민감도: 문턱 × 병합")
sens5 = []
for k in (6, 8, 10):
    for merge in (.005, .010, .020):
        row = f"{k:2d}σ 병합 {int(merge*1000):2d} ms"
        for kk in (3, 4, 5):
            v = np.array([kth(i, kk, k, merge) for i in range(len(y))]); vn = v[nor]
            row += f" | k{kk}: 005 {v[i5]:.2f} 위 정상 {(vn >= v[i5]).sum()} 차 {v[i5]-vn.max():+.2f}"
            if kk == 4: sens5.append((f"{k}σ {int(merge*1000)}ms", v[i5], vn.max()))
        lines.append(row)
lines.append("\n## 2b. 020 — L3 = 연속 충돌 열(F3) AND 고주파 비율(F6): 둘 다 020 이상인 정상")
sens20 = []
for win in (.05, .1, .2):
    for merge in (.001, .002):
        for k in (8, 10):
            v3 = np.array([train(i, win, merge, k) for i in range(len(y))], float)
            only = (nor & (v3 >= v3[i20])).sum()
            both = nor & (v3 >= v3[i20]) & (F.F6_L1_hf_ratio.values >= F.F6_L1_hf_ratio.values[i20])
            fa = (~nor) & (v3 >= v3[i20]) & (F.F6_L1_hf_ratio.values >= F.F6_L1_hf_ratio.values[i20])
            sens20.append((f"{int(win*1000)}ms {merge*1000:.0f}ms {k}σ", only, both.sum()))
            lines.append(f"창 {int(win*1000):3d} ms 병합 {merge*1000:.0f} ms {k:2d}σ | F3만 위 정상 {only} | 둘 다 위 정상 {both.sum()} {list(NM[both])} | 둘 다 위 고장 {[n[4:] for n in NM[fa]]}")
open(os.path.join(FIG, "numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

# %% 그림 1: 005 — K1_k
fig, axs = plt.subplots(1, 2, figsize=(16, 5))
a = axs[0]; rng = np.random.default_rng(0)
for j, kk in enumerate((2, 3, 4, 5, 6)):
    v = F[f"K1_{kk}"].values
    a.scatter(j + rng.uniform(-.25, -.02, nor.sum()), v[nor], s=10, color=NOR, alpha=.6)
    a.scatter(j + rng.uniform(.02, .25, (~nor).sum()), v[~nor], s=10, color=ABN, alpha=.6)
    a.scatter(j + .13, v[i5], s=70, color="#7a3fbf", zorder=5); a.hlines(v[nor].max(), j - .3, j + .3, color="k", lw=.8)
a.scatter([], [], color=NOR, label="정상"); a.scatter([], [], color=ABN, label="고장"); a.scatter([], [], color="#7a3fbf", s=70, label="abn_005")
a.plot([], [], color="k", lw=.8, label="정상 최대")
a.set_xticks(range(5), [f"k={kk}" for kk in (2, 3, 4, 5, 6)]); a.set_ylabel("log10(k번째 큰 사건 / σ)"); a.legend(fontsize=8); a.grid(alpha=.3)
a.set_title("H1′ 일반화: k번째로 큰 타격도 바탕보다 큰가 — 005는 k=3–6에서 정상 전체 위", fontsize=10)
a = axs[1]
xs = np.arange(len(sens5))
a.bar(xs - .2, [s[1] for s in sens5], .4, color="#7a3fbf", label="abn_005"); a.bar(xs + .2, [s[2] for s in sens5], .4, color=NOR, label="정상 최대")
a.set_xticks(xs, [s[0] for s in sens5], rotation=25, fontsize=8); a.set_ylim(1, 1.8); a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
a.set_title("K1_4 민감도 (문턱 × 병합) — 9개 모두 005 > 정상 최대", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_005_K1k.png"), dpi=80); plt.close(fig)

# %% 그림 2: 020 — F3 × F6 지도
fig, ax = plt.subplots(figsize=(9, 6.5))
v3, v6 = F.F3_L3_train.values, F.F6_L1_hf_ratio.values
jit = np.random.default_rng(1).uniform(-.3, .3, len(y))
ax.scatter(v3[nor] + jit[nor], v6[nor], s=20, color=NOR, alpha=.7, label="정상 120")
ax.scatter(v3[~nor] + jit[~nor], v6[~nor], s=28, color=ABN, label="고장 30")
ax.axvline(v3[i20], color="k", lw=.6, ls="--"); ax.axhline(v6[i20], color="k", lw=.6, ls="--")
for n in ["abn_020", "abn_007", "abn_013", "abn_024", "abn_026", "nor_116", "nor_015", "nor_106", "nor_055"]:
    k = names.index(n); ax.annotate(n, (v3[k] + jit[k], v6[k]), fontsize=8, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("F3 연속 충돌 열: 100 ms 안 8σ 사건 최대 개수 (2 ms 병합)"); ax.set_ylabel("F6 log10(9–12.5 kHz / 200 Hz 이상 파워)")
ax.set_title("L3 연속 충돌(1-3: 잦은 연발 충격 + 고주파 상승) — 점선 = 020 값", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_020_L3_map.png"), dpi=80); plt.close(fig)

# %% 그림 3: 파형 확대 (005 단발 반복 · 020 연속 충돌 · nor_116)
Z = [("abn_005", 0.2, 1.0), ("abn_020", 0.38, 0.52), ("nor_116", None, None)]
fig, axs = plt.subplots(3, 1, figsize=(16, 7.5))
for a, (n, lo, hi) in zip(axs, Z):
    i = names.index(n)
    if lo is None:
        v = np.array([((events(i, 8, .002)[0] >= e) & (events(i, 8, .002)[0] < e + .1)).sum() for e in events(i, 8, .002)[0]])
        e0 = events(i, 8, .002)[0][v.argmax()]; lo, hi = e0 - .03, e0 + .11
    mm = (t >= lo) & (t <= hi)
    a.plot(t[mm], HP[i, mm] / SIG[i], lw=.4, color=ABN if y[i] else NOR); a.axhline(8, color="k", lw=.5, ls="--"); a.axhline(-8, color="k", lw=.5, ls="--")
    a.set_title(f"{n} {lo:.2f}–{hi:.2f} s, 200 Hz 고역 / σ (점선 ±8σ)", fontsize=9, loc="left"); a.grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_waves.png"), dpi=80); plt.close(fig)
print("done")
