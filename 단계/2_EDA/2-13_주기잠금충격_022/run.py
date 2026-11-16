# %% [markdown]
# 2-13 주기 잠금 충격 — abn_022를 가르는 축 (사용자 요청 11-14 "22를 가를 방법")
# - 관찰: 022는 느린 파형이 0.94 s마다 같은 모양으로 반복되고, 충격이 매번 같은 자리(느린 파형 꼭대기 → 급강하 직전)에서 난다
# - 가설 H4: 하중 반전 시 유격(볼조인트·부싱) → 충격이 하중 주기에 잠긴다
# - 축 (라벨 무관 정의):
#   T     = 30 Hz 저역 신호를 τ만큼 민 자기 자신과의 피어슨 r이 최대인 τ (0.3–1.2 s)
#   r_LP  = 그 최대 r (느린 파형이 얼마나 똑같이 반복되나)
#   hit   = 8σ 사건(200 Hz 고역, σ = 1.4826·MAD, 10 ms 안은 한 사건) 중 T ± 3% 떨어진 곳에 다른 사건이 있는 비율
#   S     = r_LP × hit  (느린 파형 반복 AND 충격이 그 주기에 잠김)
# - 주의: 022를 본 뒤 만든 축 → 사후. 민감도(차단·지연 범위·허용 오차·포락선)로 운인지 확인
# 실행: conda run -n qe-phm python "단계/2_EDA/2-13_주기잠금충격_022/run.py"

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

FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); names = [str(n) for n in names]
FS = 25000; D = 50; fs2 = FS / D
X = X - X.mean(1, keepdims=True)
HP = sosfiltfilt(butter(4, 200, "high", fs=FS, output="sos"), X, axis=1)
nor = y == 0; i22 = names.index("abn_022")
NOR, ABN = "#2a78d6", "#eb6834"


def pear_r(s, k):
    n = len(s)
    return np.corrcoef(s[:n - k], s[k:])[0, 1]


def events(h, k_sigma=8, merge_s=0.010):
    sig = 1.4826 * np.median(np.abs(h - np.median(h)))
    idx = np.flatnonzero(np.abs(h) > k_sigma * sig)
    ev = []
    for j in idx:
        if not ev or j - ev[-1][-1] > int(merge_s * FS): ev.append([j])
        else: ev[-1].append(j)
    return np.array([e[np.argmax(np.abs(h[e]))] for e in ev]) / FS if ev else np.array([])


def lock_score(fc=30, lo=0.3, hi=1.2, tol=0.03):
    L = sosfiltfilt(butter(4, fc, "low", fs=FS, output="sos"), X, axis=1)[:, ::D]
    ks = np.arange(int(lo * fs2), int(hi * fs2) + 1, 2)
    rows = []
    for i in range(len(y)):
        r = np.array([pear_r(L[i], k) for k in ks]); T = ks[r.argmax()] / fs2
        ev = events(HP[i])
        if len(ev) >= 2:
            d = np.abs(ev[None, :] - ev[:, None])
            hit = (np.abs(d - T) <= tol * T).any(1).mean()
        else:
            hit = 0.0
        rows.append((r.max(), T, hit, len(ev)))
    A = pd.DataFrame(rows, columns=["r_LP", "T", "hit", "n_ev"], index=names)
    A["S"] = A.r_LP * A.hit; A["y"] = y
    return A


# %% 기본 설정 + 민감도
A = lock_score()
A.to_csv(os.path.join(FIG, "lock_score.csv"))
lines = ["## 기본 (30 Hz, 0.3–1.2 s, ±3%)"]
for c in ("r_LP", "hit", "S"):
    v = A[c].values; vn = v[nor]; top = np.flatnonzero(nor)[vn.argmax()]
    ab = [names[i] for i in np.flatnonzero((y == 1) & (v > vn.max()))]
    lines.append(f"{c:5s} 022={v[i22]:.3f} | 정상 중앙 {np.median(vn):.3f} · 95% {np.percentile(vn, 95):.3f} · 최대 {vn.max():.3f}({names[top]})"
                 f" | 022 이상 정상 {(vn >= v[i22]).sum()} | 정상 최대 초과 고장 {ab}")
lines.append(f"022: T {A.T.iloc[i22] if False else A['T'].iloc[i22]:.3f} s · 사건 {int(A.n_ev.iloc[i22])}개 · hit {A.hit.iloc[i22]:.2f}")
lines.append("S 상위 10: " + ", ".join(f"{n}({v:.2f})" for n, v in A.S.sort_values(ascending=False).head(10).items()))
lines.append("\n## 민감도 (S) — 차단 Hz · 지연 범위 s · 허용 오차")
SENS = [(10, .3, 1.2, .03), (20, .3, 1.2, .03), (50, .3, 1.2, .03), (100, .3, 1.2, .03),
        (30, .2, 1.5, .03), (30, .5, 1.0, .03), (30, .3, 1.2, .05), (30, .3, 1.2, .02), (100, .2, 1.5, .05)]
sens = []
for fc, lo, hi, tol in SENS:
    B = lock_score(fc, lo, hi, tol); v = B.S.values; vn = v[nor]; top = np.flatnonzero(nor)[vn.argmax()]
    ab = [names[i][4:] for i in np.flatnonzero((y == 1) & (v > vn.max()))]
    sens.append((f"{fc}Hz {lo}–{hi}s ±{int(tol*100)}%", v[i22], vn.max(), names[top], int((vn >= v[i22]).sum()), ab))
    lines.append(f"{fc:3d} Hz {lo}–{hi} s ±{int(tol*100)}% | 022 {v[i22]:.3f} | 정상 최대 {vn.max():.3f}({names[top]}) | 차 {v[i22]-vn.max():+.3f}"
                 f" | 022 이상 정상 {(vn >= v[i22]).sum()} | 정상 최대 초과 고장 {ab}")
open(os.path.join(FIG, "numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("\n".join(lines))

# %% 그림 1: 022 파형 — 느린 파형, 충격, 주기 T
L30 = sosfiltfilt(butter(4, 30, "low", fs=FS, output="sos"), X[i22])
t = np.arange(X.shape[1]) / FS; T22 = A["T"].iloc[i22]; ev = events(HP[i22])
fig, axs = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
axs[0].plot(t, X[i22], lw=.4, color="0.4", label="원신호")
axs[0].plot(t, L30, lw=1.8, color=ABN, label="30 Hz 저역 (느린 파형)")
axs[0].plot(t + T22, L30, lw=1.2, ls="--", color="#7a3fbf", label=f"느린 파형을 T = {T22:.2f} s 민 것")
axs[0].set_ylabel("g"); axs[0].legend(fontsize=8, loc="upper right"); axs[0].grid(alpha=.3)
axs[0].set_title(f"abn_022 — 느린 파형이 {T22:.2f} s마다 같은 모양으로 반복 (r_LP {A.r_LP.iloc[i22]:.2f})", fontsize=10)
axs[1].plot(t, HP[i22], lw=.4, color=NOR)
for e in ev: axs[1].axvline(e, color=ABN, lw=.8, alpha=.7)
axs[1].set_ylabel("200 Hz 고역 (g)"); axs[1].set_xlabel("s"); axs[1].grid(alpha=.3); axs[1].set_xlim(0, 2)
axs[1].set_title(f"8σ 사건 {len(ev)}개 (주황 세로선) — 모두 T ± 3% 뒤/앞에 짝 사건이 있음 (hit {A.hit.iloc[i22]:.2f})", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_022_wave.png"), dpi=80); plt.close(fig)

# %% 그림 2: 두 축 지도 + S 분포
fig, axs = plt.subplots(1, 2, figsize=(15, 5.5))
a = axs[0]
a.scatter(A.r_LP[nor], A.hit[nor], s=20, color=NOR, alpha=.7, label="정상 120")
a.scatter(A.r_LP[~nor], A.hit[~nor], s=26, color=ABN, label="고장 30")
for n in list(A.S.sort_values(ascending=False).head(6).index):
    a.annotate(n, (A.r_LP[n], A.hit[n]), fontsize=8, xytext=(3, 3), textcoords="offset points")
a.set_xlabel("r_LP: 느린 파형 반복 상관"); a.set_ylabel("hit: 주기 T에 짝이 있는 충격 비율")
a.set_title("느린 파형 반복 × 충격 주기 잠금 — 오른쪽 위 = 하중 주기에 잠긴 충격 (H4)", fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8)
a = axs[1]
rng = np.random.default_rng(0)
a.scatter(rng.uniform(-.15, .15, nor.sum()), A.S[nor], s=18, color=NOR, label="정상 120")
a.scatter(1 + rng.uniform(-.15, .15, (~nor).sum()), A.S[~nor], s=18, color=ABN, label="고장 30")
a.axhline(A.S[nor].max(), color="k", lw=.7, ls="--", label="정상 최대")
for n in list(A.S.sort_values(ascending=False).head(5).index):
    a.annotate(n, (int(A.y[n]) + .17, A.S[n]), fontsize=8)
a.set_xticks([0, 1], ["정상", "고장"]); a.set_ylabel("S = r_LP × hit"); a.grid(alpha=.3); a.legend(fontsize=8)
a.set_title("주기 잠금 충격 점수 S", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_map.png"), dpi=80); plt.close(fig)

# %% 그림 3: 민감도
fig, ax = plt.subplots(figsize=(11, 4.5))
xs = np.arange(len(sens))
ax.bar(xs - .2, [s[1] for s in sens], .4, color=ABN, label="abn_022")
ax.bar(xs + .2, [s[2] for s in sens], .4, color=NOR, label="정상 최대")
ax.set_xticks(xs, [s[0] for s in sens], rotation=25, fontsize=8); ax.set_ylabel("S"); ax.grid(alpha=.3, axis="y"); ax.legend(fontsize=8)
ax.set_title("민감도: 설정을 바꿔도 022가 정상 최대보다 높은가", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_sensitivity.png"), dpi=80); plt.close(fig)
print("done")
