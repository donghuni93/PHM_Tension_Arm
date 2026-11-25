# %% [markdown]
# 4-6 사령탑 7b–7f — 충격 검출 정의를 처음부터 다시 정한다 (7b–7e = 정상 120 + val 40, 라벨 없이 / 7f = 라벨 보고만)
# 고정: 평균 제거 → 1 kHz 고역(4차 Butterworth sosfiltfilt) → |hilbert| → 저역 다듬기(500 / 1000 Hz) → 포락선 / 바탕 ≥ k 인 봉우리
# 선택 규칙은 실행 전에 고정(결과를 보고 바꾸지 않음):
#   7b 바탕: 100 ms 이동 중앙값의 (최대/최소) 파일 중앙값 ≥ 2 → 이동 중앙값(100 ms), 아니면 전체 중앙값 (판단은 500 Hz 다듬은 포락선 기준, 1000 Hz·50·200 ms는 보고만)
#   7c 문턱: 대리 신호 가짜 충격 평균 ≤ 0.05/파일 인 가장 작은 k (다듬기마다). 다듬기: 합성 울림 검출률 평균이 높은 쪽(차 < 0.05면 500 Hz)
#   7d 간격: 간격 히스토그램(0.5 ms 칸, 3칸 이동평균)의 1–10 ms 안 첫 국소 최솟값, 없으면 3 ms
# 실행: conda run -n qe-phm python "단계/4_피처/4-1_충격정의/자료/사령탑_7b-7e/run.py"
import os, sys, json, warnings
from functools import lru_cache
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
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d
from src.data import load_dataset, load_folder
import src.final as F

for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000
NOR_C, ABN_C, VAL_C, GRAY = "#2a78d6", "#eb6834", "#7B1FA2", "#9aa0a6"
R = {}   # result.json

# 확정값 (첫 실행의 선택 결과를 아래 detect_impacts 기본값으로 옮겼고, 맨 끝에서 이번 선택과 같은지 assert)
CHOSEN = dict(smooth=1000, base="moving", k=4.5, distance_ms=6.75)   # 7b–7d 규칙이 고른 값 (맨 끝에서 이번 선택과 같은지 assert)


# %% 신호 처리 도구 -----------------------------------------------------------
@lru_cache(None)
def _sos(kind, f, fs):
    return butter(4, f, btype=kind, fs=fs, output="sos")


def hp1k(X, fs=FS, f=1000):
    X = np.atleast_2d(np.asarray(X, float)); X = X - X.mean(1, keepdims=True)
    return sosfiltfilt(_sos("highpass", f, fs), X, axis=-1)


def envelope(S, lp, fs=FS):
    """|hilbert| 후 4차 저역 다듬기 (lp = None 이면 다듬지 않음)"""
    E = np.abs(hilbert(S, axis=-1))
    return E if lp is None else sosfiltfilt(_sos("lowpass", lp, fs), E, axis=-1)


def moving_median_dec(E, win_ms, fs=FS):
    """이동 중앙값을 1 kHz로 솎아 계산 (포락선은 이미 저역이라 충분). 반환 (값 (n, m), 솎음 간격)"""
    dec = fs // 1000
    size = int(round(win_ms)) | 1
    return median_filter(E[:, ::dec], size=(1, size), mode="reflect"), dec


def baseline(E, mode, win_ms=100, fs=FS):
    """바탕 (n, N): 'global' = 파일 전체 중앙값, 'moving' = win_ms 이동 중앙값(선형 보간해 원래 길이로)"""
    n, N = E.shape
    if mode == "global":
        return np.repeat(np.median(E, 1, keepdims=True), N, 1)
    M, dec = moving_median_dec(E, win_ms, fs)
    xs = np.arange(0, N, dec); xi = np.arange(N)
    return np.stack([np.interp(xi, xs, m) for m in M])


def count_peaks(ratio_row, k, dist):
    return len(find_peaks(ratio_row, height=k, distance=dist)[0])


# %% 데이터: train 정상 120 · 고장 30 (보고용), val 40 -----------------------------
X, y, names = load_dataset(os.path.join(_d, "data", "TeamProject_train_변경"))
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val"))
names = [str(n) for n in names]; nv = [str(n) for n in nv]
assert X.shape == (150, 50000) and Xv.shape == (40, 50000)
nor = y == 0
Xall = np.vstack([X, Xv]); nall = names + nv
grp = np.array(["정상"] * 0 + ["정상" if v == 0 else "고장" for v in y] + ["val"] * 40)
HP = hp1k(Xall)                                   # (190, 50000) 1 kHz 고역
use = np.flatnonzero(grp != "고장")               # 7b–7e에 쓰는 파일 = 정상 120 + val 40
HPu = HP[use]; nu = [nall[i] for i in use]
assert len(use) == 160
R["n_files"] = dict(normal=int(nor.sum()), val=40, abnormal=int((~nor).sum()))
print("로드 완료", Xall.shape, flush=True)

# %% 7b 바탕 -------------------------------------------------------------------
print("== 7b", flush=True)
ratio7b = {}
for lp in (500, 1000):
    E = envelope(HPu, lp)
    for w in (50, 100, 200):
        M, dec = moving_median_dec(E, w)
        half = int(round(w)) // 2 + 1
        Mv = M[:, half:M.shape[1] - half]            # 창이 온전히 들어가는 구간만
        ratio7b[(lp, w)] = Mv.max(1) / Mv.min(1)
tab7b = {f"{lp}Hz_{w}ms": dict(median=float(np.median(v)), p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)),
                              min=float(v.min()), max=float(v.max())) for (lp, w), v in ratio7b.items()}
R["7b"] = dict(ratios=tab7b)
med100 = float(np.median(ratio7b[(500, 100)]))
BASE = "moving" if med100 >= 2 else "global"
R["7b"].update(rule="100 ms 이동 중앙값 (최대/최소)의 파일 중앙값(500 Hz 포락선) >= 2 이면 이동", median_ratio_500_100=med100,
               median_ratio_1000_100=float(np.median(ratio7b[(1000, 100)])), chosen=BASE)
print("7b", {k: round(v["median"], 2) for k, v in tab7b.items()}, "->", BASE, flush=True)

# 그림 7b
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
bins = np.logspace(0, np.log10(max(v.max() for v in ratio7b.values()) * 1.05), 40)
for c, lp in enumerate((500, 1000)):
    for w, col in ((50, "#7fb3e6"), (100, "#2a78d6"), (200, "#0b3d7a")):
        ax[c].hist(ratio7b[(lp, w)], bins=bins, histtype="step", lw=2, color=col, label=f"창 {w} ms (중앙값 {np.median(ratio7b[(lp, w)]):.2f})")
    ax[c].axvline(2, color="k", ls=":"); ax[c].set_xscale("log"); ax[c].set_xlabel("이동 중앙값의 최대 / 최소 (파일마다)"); ax[c].set_ylabel("파일 수")
    ax[c].set_title(f"{lp} Hz 다듬은 포락선" + (" (규칙 판단 기준)" if lp == 500 else " (보고만)"), fontsize=10); ax[c].legend(fontsize=8)
fig.suptitle("7b 바탕이 파일 안에서 얼마나 변하나 — 정상 120 + val 40 (점선 = 규칙 문턱 2)")
fig.savefig(os.path.join(FIG, "7b_바탕변동비.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
# 그림 7b-2: 비율 큰/작은/중간 파일 바탕 예시
E5 = envelope(HPu, 500); Bm = baseline(E5, "moving", 100); r100 = ratio7b[(500, 100)]
order = np.argsort(r100); pick = [order[-1], order[len(order) // 2], order[0]]
fig, ax = plt.subplots(3, 1, figsize=(14, 7.5), sharex=True)
t = np.arange(50000) / FS
for a, i in zip(ax, pick):
    a.plot(t, E5[i], lw=.6, color=GRAY, label="500 Hz 다듬은 포락선"); a.plot(t, Bm[i], color=ABN_C, lw=2, label="100 ms 이동 중앙값")
    a.axhline(np.median(E5[i]), color=NOR_C, ls="--", lw=1.2, label="전체 중앙값"); a.set_ylabel("g")
    a.set_title(f"{nu[i]} — 이동 중앙값 최대/최소 {r100[i]:.2f}", fontsize=9); a.legend(fontsize=7, loc="upper right")
ax[-1].set_xlabel("s"); fig.savefig(os.path.join(FIG, "7b_바탕예시.png"), dpi=100, bbox_inches="tight"); plt.close(fig)

# %% 7c 대리 신호 · 문턱 k --------------------------------------------------------
print("== 7c", flush=True)
KS = [3, 3.5, 4, 4.5, 5, 6]; LPS = (500, 1000); NREP = 5
SEED = 2111
rng = np.random.default_rng(SEED)
spec = np.fft.rfft(HPu, axis=-1); amp = np.abs(spec)
rms_loc = np.sqrt(uniform_filter1d(HPu ** 2, 2500, axis=-1, mode="reflect"))      # 100 ms RMS 포락선
rms_glob = np.sqrt((HPu ** 2).mean(1, keepdims=True))
KG = np.linspace(1, 8, 36)                                   # 꼬리 분포용 k 격자
variants = ["stat", "mod"] if BASE == "moving" else ["stat"]
PRIMARY = "mod" if BASE == "moving" else "stat"
cnt = {(v, lp): np.zeros((NREP, len(use), len(KS))) for v in variants for lp in LPS}
tail_sur = {(v, lp): np.zeros(len(KG)) for v in variants for lp in LPS}
tail_org = {lp: np.zeros(len(KG)) for lp in (None,) + LPS}
tail_sur_raw = np.zeros(len(KG))
DIST1 = int(0.001 * FS)
SUR_FIRST = None                                              # 합성 단계용 첫 번째 대리 신호(정상 120개 행)
SURS = {}                                                     # 합성용: variant → (NREP, 160, N) 는 메모리가 커서 정상 120개만, 필요한 variant만
nor_rows = np.flatnonzero(grp[use] == "정상")


def tails(Rm):
    return np.array([(Rm > kg).sum() for kg in KG], float)


for rep in range(NREP):
    ph = rng.uniform(0, 2 * np.pi, amp.shape); ph[:, 0] = 0; ph[:, -1] = 0
    S0 = np.fft.irfft(amp * np.exp(1j * ph), n=50000, axis=-1)
    Sm = S0 * rms_loc / rms_glob
    for v, S in (("stat", S0), ("mod", Sm)):
        if v not in variants: continue
        if v == PRIMARY: SURS[rep] = S[nor_rows].copy()
        for lp in LPS:
            E = envelope(S, lp); B = baseline(E, BASE, 100); Rm = E / B
            tail_sur[(v, lp)] += tails(Rm)
            for i in range(len(use)):
                for j, k in enumerate(KS):
                    cnt[(v, lp)][rep, i, j] = count_peaks(Rm[i], k, DIST1)
    print(f"  대리 {rep + 1}/{NREP}", flush=True)
for lp in (None,) + LPS:                                      # 원 신호 꼬리 (전체 중앙값 · 이동 바탕 둘 다 기록하려면 기준 바탕만)
    E = envelope(HPu, lp); B = baseline(E, BASE, 100); tail_org[lp] = tails(E / B)
Ntot = HPu.size
fa = {}
for (v, lp), c in cnt.items():
    fa[(v, lp)] = c.mean((0, 1))
fa_tab = {f"{v}_{lp}Hz": {str(k): float(m) for k, m in zip(KS, fa[(v, lp)])} for (v, lp) in fa}
R["7c"] = dict(baseline=BASE, primary_surrogate=PRIMARY, surrogate_n=int(NREP * len(use)), seed=SEED, ks=KS, false_per_file=fa_tab)
KSEL = {}
for lp in LPS:
    ok = [k for k, m in zip(KS, fa[(PRIMARY, lp)]) if m <= 0.05]
    KSEL[lp] = ok[0] if ok else None
R["7c"]["rule_k"] = {str(lp): KSEL[lp] for lp in LPS}
print("7c 가짜 충격/파일 (기준 대리)", {lp: dict(zip(KS, np.round(fa[(PRIMARY, lp)], 3))) for lp in LPS}, "→", KSEL, flush=True)
degenerate_k = [lp for lp in LPS if KSEL[lp] is None]
R["7c"]["degenerate_k"] = degenerate_k
for lp in LPS:
    if KSEL[lp] is None: KSEL[lp] = KS[-1]                      # 퇴화 시 가장 큰 후보로 진행(보고)

# 그림 7c: 가짜 충격 수 vs k, 꼬리 분포
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for lp, col in ((500, NOR_C), (1000, ABN_C)):
    for v, ls in (("stat", ":"), ("mod", "-")):
        if (v, lp) in fa:
            ax[0].plot(KS, np.maximum(fa[(v, lp)], 1e-4), ls, color=col, marker="o", ms=4, label=f"{lp} Hz · " + ("대리(정상성)" if v == "stat" else "대리×100 ms RMS (기준)"))
ax[0].axhline(0.05, color="k", ls="--", lw=1); ax[0].set_yscale("log"); ax[0].set_xlabel("k (포락선 / 바탕)"); ax[0].set_ylabel("가짜 충격 / 파일 (대리 신호 평균)")
ax[0].set_title("7c 가짜 충격 수 (검은 점선 = 0.05/파일, 0 은 1e-4로 표시)", fontsize=10); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, which="both")
kk = np.linspace(1, 8, 200); ax[1].plot(kk, 2.0 ** (-kk ** 2), "k--", lw=1.5, label="레일리 이론 2^(-k²)")
for lp, col in ((500, NOR_C), (1000, ABN_C)):
    ax[1].plot(KG, np.maximum(tail_org[lp] / Ntot, 1e-10), "-", color=col, lw=2, label=f"원 신호 {lp} Hz")
    ax[1].plot(KG, np.maximum(tail_sur[(PRIMARY, lp)] / (Ntot * NREP), 1e-10), ":", color=col, lw=2, label=f"대리 {lp} Hz")
ax[1].plot(KG, np.maximum(tail_org[None] / Ntot, 1e-10), "-", color="#2e9e5b", lw=1.5, label="원 신호 (다듬기 없음)")
ax[1].set_yscale("log"); ax[1].set_ylim(1e-9, 1); ax[1].set_xlim(1, 8); ax[1].set_xlabel("k"); ax[1].set_ylabel("P(포락선/바탕 > k)  (표본 점 기준)")
ax[1].set_title("꼬리 분포: 원 신호 vs 대리 신호 vs 레일리", fontsize=10); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
fig.savefig(os.path.join(FIG, "7c_가짜충격_꼬리.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

# %% 7c 다듬기 선택: 합성 감쇠 사인파 검출률 ------------------------------------------
print("== 7c 합성", flush=True)
RINGS = [(2000, 0.08), (7000, 0.02), (8250, 0.01), (3000, 0.16), (1500, 0.2)]
rng20 = np.random.default_rng(20)
tt = np.arange(int(0.05 * FS)) / FS
syn = {lp: np.zeros((len(RINGS), 0)) for lp in LPS}
hits = {lp: [[] for _ in RINGS] for lp in LPS}
for rep in range(NREP):
    S = SURS[rep].copy()                                      # (120, N) 정상 대리 신호
    E0 = np.abs(hilbert(S, axis=-1))
    Bun = baseline(E0, BASE, 100)                             # 다듬지 않은 |hilbert|의 바탕 (처음 높이 = 바탕의 6배)
    t0s = np.zeros((len(S), len(RINGS)), int)
    for i in range(len(S)):
        while True:                                           # 50 ms 이상 간격, 앞뒤 여유
            c = np.sort(rng20.integers(int(0.03 * FS), 50000 - int(0.06 * FS), len(RINGS)))
            if (np.diff(c) >= int(0.05 * FS)).all(): break
        c = rng20.permutation(c); t0s[i] = c
        for r, (f, z) in enumerate(RINGS):
            w = 2 * np.pi * f; wd = w * np.sqrt(1 - z ** 2); ph0 = rng20.uniform(0, 2 * np.pi)
            tl = np.arange(min(int(0.05 * FS), 50000 - c[r])) / FS
            S[i, c[r]:c[r] + len(tl)] += 6 * Bun[i, c[r]] * np.exp(-z * w * tl) * np.sin(wd * tl + ph0)
    for lp in LPS:
        E = envelope(S, lp); B = baseline(E, BASE, 100); Rm = E / B
        for i in range(len(S)):
            pk = find_peaks(Rm[i], height=KSEL[lp], distance=DIST1)[0]
            for r in range(len(RINGS)):
                hits[lp][r].append(bool(((pk >= t0s[i, r] - int(0.003 * FS)) & (pk <= t0s[i, r] + int(0.006 * FS))).any()))
syn_rate = {lp: [float(np.mean(h)) for h in hits[lp]] for lp in LPS}
syn_mean = {lp: float(np.mean(syn_rate[lp])) for lp in LPS}
SMOOTH = 1000 if syn_mean[1000] - syn_mean[500] >= 0.05 else (500 if syn_mean[500] - syn_mean[1000] >= 0.05 else 500)
R["7c"]["synthetic"] = dict(rings=[list(r) for r in RINGS], amplitude="처음 높이 = 6 x 다듬지 않은 |hilbert| 바탕", n_per_ring=len(hits[500][0]),
                            rate={str(lp): syn_rate[lp] for lp in LPS}, mean={str(lp): syn_mean[lp] for lp in LPS},
                            rule="평균 검출률이 0.05 이상 높은 쪽, 아니면 500 Hz")
CHOSEN_NOW = dict(smooth=SMOOTH, base=BASE, k=KSEL[SMOOTH], distance_ms=None)
print("합성 검출률", syn_rate, syn_mean, "→ 다듬기", SMOOTH, "k", KSEL[SMOOTH], flush=True)

fig, ax = plt.subplots(figsize=(9, 4.5)); xs = np.arange(len(RINGS))
for j, (lp, col) in enumerate(((500, NOR_C), (1000, ABN_C))):
    ax.bar(xs + (j - .5) * .38, syn_rate[lp], .38, color=col, label=f"{lp} Hz · k = {KSEL[lp]} (평균 {syn_mean[lp]:.2f})")
ax.set_xticks(xs, [f"{f/1000:g} kHz\nζ {z}" for f, z in RINGS]); ax.set_ylabel("검출률"); ax.set_ylim(0, 1.05); ax.legend(fontsize=8)
ax.set_title("합성 울림(처음 높이 = 바탕 6배)을 대리 바탕에 넣고 검출", fontsize=10)
fig.savefig(os.path.join(FIG, "7c_합성검출률.png"), dpi=110, bbox_inches="tight"); plt.close(fig)


# %% detect_impacts (이 폴더의 새 정의) --------------------------------------------
def detect_impacts(x, fs=FS, smooth=None, base=None, k=None, distance_ms=None, base_win_ms=100, hp=1000, search_ms=30):
    """한 신호(1차원) → 충격 목록 dict. 기본값 = 7b–7d 선택 결과 (CHOSEN)

    평균 제거 → hp Hz 고역 → |hilbert| → smooth Hz 저역 = 포락선, 바탕 = 전체 중앙값 또는 이동 중앙값 → 포락선/바탕 ≥ k 인 봉우리(최소 distance_ms 간격)
    반환: t (s) · height_x (바탕 대비 배수) · height_g (다듬은 포락선 봉우리, g) · rise_ms (바탕 위 높이 10→90 %) · decay_ms (봉우리 → 높이의 1/e) · idx
    rise/decay 는 search_ms 안에서 못 찾으면 NaN
    """
    smooth = CHOSEN["smooth"] if smooth is None else smooth
    base = CHOSEN["base"] if base is None else base
    k = CHOSEN["k"] if k is None else k
    distance_ms = CHOSEN["distance_ms"] if distance_ms is None else distance_ms
    s = hp1k(np.asarray(x, float)[None], fs, hp)
    E = envelope(s, smooth, fs); B = baseline(E, base, base_win_ms, fs)
    e, b = E[0], B[0]
    pk = find_peaks(e / b, height=k, distance=max(1, int(round(distance_ms * fs / 1000))))[0]
    W = int(search_ms * fs / 1000)
    rise, dec = np.full(len(pk), np.nan), np.full(len(pk), np.nan)
    for n, p in enumerate(pk):
        h = e[p] - b[p]
        back = e[max(0, p - W):p + 1][::-1] - b[p]
        i90 = np.flatnonzero(back <= 0.9 * h); i10 = np.flatnonzero(back <= 0.1 * h)
        if len(i90) and len(i10): rise[n] = (i10[0] - i90[0]) / fs * 1000
        fwd = e[p:p + W + 1] - b[p]
        j = np.flatnonzero(fwd <= h / np.e)
        if len(j): dec[n] = j[0] / fs * 1000
    return dict(t=pk / fs, height_x=e[pk] / b[pk], height_g=e[pk], rise_ms=rise, decay_ms=dec, idx=pk, env=e, base=b, hp=s[0])


# %% 7d 분리 간격 -----------------------------------------------------------------
print("== 7d", flush=True)
E = envelope(HPu, SMOOTH); B = baseline(E, BASE, 100); Rm = E / B
gaps = []
for i in range(len(use)):
    pk = find_peaks(Rm[i], height=KSEL[SMOOTH], distance=DIST1)[0]
    gaps.append(np.diff(pk) / FS * 1000)
gall = np.concatenate(gaps)
edges = np.arange(0, 30.5, 0.5); h, _ = np.histogram(gall, edges); cen = (edges[:-1] + edges[1:]) / 2
sm = np.convolve(h, np.ones(3) / 3, mode="same")
dist_sel = None
for i in range(1, len(sm) - 1):
    if 1 <= cen[i] <= 10 and sm[i] < sm[i - 1] and sm[i] <= sm[i + 1]:
        dist_sel = float(cen[i]); break
d_rule = dist_sel is not None
if dist_sel is None: dist_sel = 3.0
R["7d"] = dict(n_peaks=int(len(gall) + len(use)), n_gaps=int(len(gall)), hist_0_30=[int(v) for v in h], smoothed=[round(float(v), 2) for v in sm],
               rule_found_local_min=d_rule, chosen_ms=dist_sel, gaps_below_5ms=int((gall < 5).sum()), gaps_below_10ms=int((gall < 10).sum()),
               ring_decay_ms="1/(zeta*2*pi*f): 0.5-2 ms (2k 0.08=1.0, 7k 0.02=1.1, 8.25k 0.01=1.9, 3k 0.16=0.33, 1.5k 0.2=0.53)")
print("7d 선택 간격", dist_sel, "규칙 성립", d_rule, flush=True)
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.bar(cen, np.maximum(h, 0.5), width=.5, color=GRAY, alpha=.7, label="봉우리 사이 간격 (distance = 1 ms)")
ax.plot(cen, np.maximum(sm, 0.5), color=NOR_C, lw=2, label="3칸 이동평균")
ax.axvline(dist_sel, color=ABN_C, lw=2, label=f"선택 간격 {dist_sel:g} ms" + ("" if d_rule else " (국소 최솟값 없음 → 3 ms)"))
for tau, nm in ((0.33, "3k·0.16"), (1.0, "2k·0.08"), (1.93, "8.25k·0.01")):
    ax.axvline(tau, color="k", ls=":", lw=1); ax.text(tau, ax.get_ylim()[1] * .5, nm, rotation=90, fontsize=7, va="top")
ax.set_yscale("log"); ax.set_xlabel("연속한 두 봉우리 사이 간격 (ms)"); ax.set_ylabel("개수 (정상 120 + val 40)"); ax.legend(fontsize=8)
ax.set_title("7d 충격 사이 간격 히스토그램 (점선 = 합성 울림 꺼짐 시간 τ)", fontsize=10)
fig.savefig(os.path.join(FIG, "7d_간격히스토그램.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
CHOSEN_NOW["distance_ms"] = dist_sel

# %% 7e 모양 확인 -----------------------------------------------------------------
print("== 7e", flush=True)
CH = CHOSEN_NOW
if CHOSEN["k"] is not None:
    for kk_ in CHOSEN: assert CHOSEN[kk_] == CH[kk_], (kk_, CHOSEN[kk_], CH[kk_])
DET = {}                                                       # 모든 190 파일
for i in range(len(Xall)):
    DET[i] = detect_impacts(Xall[i], **CH)
rows = []
for i, nm in enumerate(nall):
    d = DET[i]
    for n in range(len(d["t"])):
        rows.append((nm, grp[i], d["t"][n], d["height_x"][n], d["height_g"][n], d["rise_ms"][n], d["decay_ms"][n]))
IMP = pd.DataFrame(rows, columns=["file", "group", "t_s", "height_x", "height_g", "rise_ms", "decay_ms"])
IMP.to_csv(os.path.join(HERE, "충격표_전체.csv"), index=False, encoding="utf-8-sig")
shape = {}
for g in ("정상", "val", "고장"):
    d = IMP[IMP.group == g]; ok = d.dropna(subset=["rise_ms", "decay_ms"])
    shape[g] = dict(n_impacts=int(len(d)), n_both=int(len(ok)), n_rise_nan=int(d.rise_ms.isna().sum()), n_decay_nan=int(d.decay_ms.isna().sum()),
                    rise_median=float(ok.rise_ms.median()), rise_p10=float(ok.rise_ms.quantile(.1)), rise_p90=float(ok.rise_ms.quantile(.9)),
                    decay_median=float(ok.decay_ms.median()), decay_p10=float(ok.decay_ms.quantile(.1)), decay_p90=float(ok.decay_ms.quantile(.9)),
                    lump_frac=float((ok.rise_ms > ok.decay_ms).mean()), lump_n=int((ok.rise_ms > ok.decay_ms).sum()))
nv_ = IMP[IMP.group.isin(["정상", "val"])].dropna(subset=["rise_ms", "decay_ms"])
shape["정상+val"] = dict(n_both=int(len(nv_)), lump_frac=float((nv_.rise_ms > nv_.decay_ms).mean()), lump_n=int((nv_.rise_ms > nv_.decay_ms).sum()),
                       lump_files=int(nv_[nv_.rise_ms > nv_.decay_ms].file.nunique()))
R["7e"] = shape
print("7e", shape, flush=True)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
bb = np.logspace(-0.8, 1.5, 40)
for g, col in (("정상", NOR_C), ("val", VAL_C)):
    d = IMP[IMP.group == g]
    ax[0].hist(d.rise_ms.dropna(), bins=bb, histtype="step", lw=2, color=col, density=True, label=f"{g} (중앙값 {d.rise_ms.median():.2f} ms)")
    ax[1].hist(d.decay_ms.dropna(), bins=bb, histtype="step", lw=2, color=col, density=True, label=f"{g} (중앙값 {d.decay_ms.median():.2f} ms)")
    ax[2].scatter(d.rise_ms, d.decay_ms, s=8, alpha=.5, color=col, label=g)
for a, ttl in zip(ax[:2], ("오르는 시간 (10→90 %)", "꺼지는 시간 (1/e)")): a.set_xscale("log"); a.set_xlabel(ttl + " (ms)"); a.set_ylabel("밀도"); a.legend(fontsize=8)
ax[2].plot([.1, 40], [.1, 40], "k--", lw=1); ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("오르는 시간 (ms)"); ax[2].set_ylabel("꺼지는 시간 (ms)")
ax[2].set_title(f"점선 위쪽 = 덩어리(오름 > 꺼짐) 정상+val {shape['정상+val']['lump_frac']*100:.0f} %", fontsize=10); ax[2].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "7e_오름꺼짐분포.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
# 예시 6개: 덩어리 3 + 비덩어리 3 (정상+val에서 seed 고정 무작위)
rngx = np.random.default_rng(7)
lump = nv_[nv_.rise_ms > nv_.decay_ms]; nolump = nv_[nv_.rise_ms <= nv_.decay_ms]
ex = pd.concat([lump.sample(min(3, len(lump)), random_state=7), nolump.sample(3, random_state=7)])
fig, axs = plt.subplots(2, 3, figsize=(16, 7))
for a, (_, r) in zip(axs.ravel(), ex.iterrows()):
    i = nall.index(r.file); d = DET[i]; c = int(round(r.t_s * FS)); lo, hi = max(0, c - 250), min(50000, c + 251)
    tms = (np.arange(lo, hi) - c) / FS * 1000
    a.plot(tms, d["hp"][lo:hi], lw=.6, color=GRAY, label="1 kHz 고역 파형")
    a.plot(tms, d["env"][lo:hi], color=ABN_C, lw=2, label="다듬은 포락선"); a.plot(tms, d["base"][lo:hi], color=NOR_C, ls="--", lw=1.2, label="바탕")
    a.set_title(f"{r.file} t={r.t_s*1000:.0f} ms · 오름 {r.rise_ms:.2f} / 꺼짐 {r.decay_ms:.2f} ms · {r.height_x:.1f}배" + ("  [덩어리]" if r.rise_ms > r.decay_ms else ""), fontsize=9)
    a.set_xlabel("ms (충격 기준)"); a.set_ylabel("g")
axs[0, 0].legend(fontsize=7)
fig.savefig(os.path.join(FIG, "7e_예시6개.png"), dpi=100, bbox_inches="tight"); plt.close(fig)
R["7e"]["examples"] = [dict(file=r.file, t_s=float(r.t_s), rise_ms=float(r.rise_ms), decay_ms=float(r.decay_ms), lump=bool(r.rise_ms > r.decay_ms)) for _, r in ex.iterrows()]

# %% 7f 라벨 보고 -------------------------------------------------------------------
print("== 7f", flush=True)


def events_A(x):                                              # 2-13 정의 A: 200 Hz 고역, 8σ(1.4826·MAD), 10 ms 병합
    h = sosfiltfilt(_sos("highpass", 200, FS), x - x.mean())
    sig = 1.4826 * np.median(np.abs(h - np.median(h)))
    idx = np.flatnonzero(np.abs(h) > 8 * sig); ev = []
    for j in idx:
        if not ev or j - ev[-1][-1] > int(0.010 * FS): ev.append([j])
        else: ev[-1].append(j)
    return np.array([e[np.argmax(np.abs(h[e]))] for e in ev]) / FS if ev else np.array([])


def events_B(xhp):                                            # src/final.py impact_rings와 같은 시점 (8 ms 안 들어가는 것만)
    env = sosfiltfilt(_sos("lowpass", 500, FS), np.abs(hilbert(xhp)))
    pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
    return pk[pk + F._SEG <= len(xhp)]


evA = [events_A(Xall[i]) for i in range(len(Xall))]
evB = [events_B(HP[i]) for i in range(len(Xall))]
RB = F.impact_rings(X)                                         # train 150에서 B 개수 검증
assert (np.bincount(RB[:, 0].astype(int), minlength=150) == np.array([len(evB[i]) for i in range(150)])).all()
cntt = pd.DataFrame(dict(file=nall, group=grp, new=[len(DET[i]["t"]) for i in range(len(nall))], A=[len(e) for e in evA], B=[len(e) for e in evB]))
cntt.to_csv(os.path.join(HERE, "충격수_파일별_새_A_B.csv"), index=False, encoding="utf-8-sig")
dist = {}
for g in ("정상", "val", "고장"):
    d = cntt[cntt.group == g]
    dist[g] = {c: dict(n=int(len(d)), mean=float(d[c].mean()), median=float(d[c].median()), p10=float(d[c].quantile(.1)), p90=float(d[c].quantile(.9)),
                      min=int(d[c].min()), max=int(d[c].max()), zero=int((d[c] == 0).sum())) for c in ("new", "A", "B")}
R["7f"] = dict(count_distribution=dist)
R["7f"]["spearman_new_vs_A"] = {g: float(cntt[cntt.group == g][["new", "A"]].corr(method="spearman").iloc[0, 1]) for g in ("정상", "val", "고장")}
R["7f"]["spearman_new_vs_B"] = {g: float(cntt[cntt.group == g][["new", "B"]].corr(method="spearman").iloc[0, 1]) for g in ("정상", "val", "고장")}
fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
for a, c, ttl in zip(ax, ("new", "A", "B"), ("새 정의", "A (200 Hz·8σ)", "B (500 Hz·6배·5 ms)")):
    rg = np.random.default_rng(0)
    for j, (g, col) in enumerate((("정상", NOR_C), ("val", VAL_C), ("고장", ABN_C))):
        v = cntt[cntt.group == g][c].values
        a.scatter(j + rg.uniform(-.2, .2, len(v)), v, s=14, color=col, alpha=.7); a.hlines(np.median(v), j - .3, j + .3, color="k", lw=2)
    a.set_xticks([0, 1, 2], ["정상 120", "val 40", "고장 30(보고)"]); a.set_ylabel("파일당 충격 수"); a.set_title(ttl + " (검은 선 = 중앙값)", fontsize=10); a.grid(alpha=.3)
fig.savefig(os.path.join(FIG, "7f_파일당충격수.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

# 파일 추적
TRACE = ["abn_015", "abn_022", "abn_002", "abn_014", "abn_017", "nor_045", "nor_116"]
atoms, par = F._ring_atoms()
trace_rows = []; trace_sum = []
for nm in TRACE:
    i = nall.index(nm); d = DET[i]; xh = HP[i]
    trace_sum.append(dict(file=nm, new=int(len(d["t"])), A=int(len(evA[i])), B=int(len(evB[i]))))
    for n, p in enumerate(d["idx"]):
        if p + F._SEG <= len(xh):
            seg = xh[p:p + F._SEG]; seg = seg / (np.linalg.norm(seg) + 1e-12)
            ft = np.linalg.norm(np.einsum("s,ast->at", seg, atoms), axis=1); b = int(ft.argmax()); fr, zr, fit = float(par[b, 0]), float(par[b, 1]), float(ft[b])
        else:
            fr = zr = fit = np.nan
        trace_rows.append(dict(file=nm, t_ms=round(d["t"][n] * 1000, 2), height_x=round(float(d["height_x"][n]), 2), height_g=round(float(d["height_g"][n]), 4),
                               rise_ms=round(float(d["rise_ms"][n]), 2), decay_ms=round(float(d["decay_ms"][n]), 2), f_Hz=fr, zeta=zr, fit=round(fit, 3),
                               B_near=bool(any(abs(d["t"][n] - e / FS) < 0.005 for e in evB[i])), A_near=bool(any(abs(d["t"][n] - e) < 0.005 for e in evA[i]))))
TR = pd.DataFrame(trace_rows); TR.to_csv(os.path.join(HERE, "추적파일_충격표.csv"), index=False, encoding="utf-8-sig")
R["7f"]["trace_counts"] = trace_sum
R["7f"]["trace_ring_summary"] = {nm: dict(n=int((TR.file == nm).sum()), f_median=float(TR[TR.file == nm].f_Hz.median()) if (TR.file == nm).any() else None,
                                           fit_median=float(TR[TR.file == nm].fit.median()) if (TR.file == nm).any() else None) for nm in TRACE}

# 2-13 주기 잠금 S
D50 = 50; fs2 = FS / D50
Xm = Xall - Xall.mean(1, keepdims=True)
LP30 = sosfiltfilt(_sos("lowpass", 30, FS), Xm, axis=1)[:, ::D50]
ks = np.arange(int(0.3 * fs2), int(1.2 * fs2) + 1, 2)


def rT(s):
    r = np.array([np.corrcoef(s[:len(s) - k], s[k:])[0, 1] for k in ks]); return r.max(), ks[r.argmax()] / fs2


rT_all = [rT(LP30[i]) for i in range(len(Xall))]


def hit_of(ev, T, tol=0.03):
    if len(ev) < 2: return 0.0
    d = np.abs(ev[None, :] - ev[:, None]); return float((np.abs(d - T) <= tol * T).any(1).mean())


def S_table(evs):
    rows = [(rT_all[i][0], rT_all[i][1], hit_of(evs[i], rT_all[i][1]), len(evs[i])) for i in range(len(Xall))]
    A = pd.DataFrame(rows, columns=["r_LP", "T", "hit", "n_ev"], index=nall); A["S"] = A.r_LP * A.hit; A["group"] = grp; return A


S_new = S_table([DET[i]["t"] for i in range(len(Xall))])
S_old = S_table(evA)                                           # 2-13 원래 정의 (재현 확인)
S_new.to_csv(os.path.join(HERE, "S_새충격.csv"), encoding="utf-8-sig"); S_old.to_csv(os.path.join(HERE, "S_원래A.csv"), encoding="utf-8-sig")


def Ssum(A):
    out = {}
    for g in ("정상", "val", "고장"):
        v = A[A.group == g].S; out[g] = dict(max=float(v.max()), max_file=str(v.idxmax()), median=float(v.median()), p95=float(v.quantile(.95)))
    out["abn_022"] = float(A.loc["abn_022", "S"]); out["abn_014"] = float(A.loc["abn_014", "S"])
    out["abn_022_T"] = float(A.loc["abn_022", "T"]); out["abn_022_r_LP"] = float(A.loc["abn_022", "r_LP"]); out["abn_022_hit"] = float(A.loc["abn_022", "hit"]); out["abn_022_n"] = int(A.loc["abn_022", "n_ev"])
    out["normal_ge_022"] = int((A[A.group == "정상"].S >= A.loc["abn_022", "S"]).sum())
    out["normal_n_over_0.3"] = int((A[A.group == "정상"].S > 0.3).sum())
    out["abnormal_above_normal_max"] = [str(n) for n in A[(A.group == "고장") & (A.S > A[A.group == "정상"].S.max())].index]
    out["top5"] = [(str(n), round(float(v), 3)) for n, v in A.S.sort_values(ascending=False).head(5).items()]
    return out


R["7f"]["S_new"] = Ssum(S_new); R["7f"]["S_old_A_reproduced"] = Ssum(S_old)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
for a, A, ttl in ((ax[0], S_old, "원래 (정의 A 사건)"), (ax[1], S_new, "새 충격 정의")):
    rg = np.random.default_rng(0)
    for j, (g, col) in enumerate((("정상", NOR_C), ("val", VAL_C), ("고장", ABN_C))):
        v = A[A.group == g].S; a.scatter(j + rg.uniform(-.2, .2, len(v)), v, s=16, color=col, alpha=.75)
    for n in ("abn_022", "abn_014"): a.annotate(n, (2.0 + .22, A.loc[n, "S"]), fontsize=8)
    a.axhline(A[A.group == "정상"].S.max(), color="k", ls="--", lw=1); a.set_xticks([0, 1, 2], ["정상 120", "val 40", "고장 30(보고)"]); a.set_ylabel("S = r_LP × hit"); a.set_title(ttl + " — 점선 = 정상 최대", fontsize=10); a.grid(alpha=.3)
fig.savefig(os.path.join(FIG, "7f_주기잠금S.png"), dpi=110, bbox_inches="tight"); plt.close(fig)

# 022 추적 그림
i = nall.index("abn_022"); d = DET[i]; tt_ = np.arange(50000) / FS
fig, ax = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
ax[0].plot(tt_, d["hp"], lw=.4, color=GRAY); [ax[0].axvline(t_, color=ABN_C, lw=.8, alpha=.7) for t_ in d["t"]]; ax[0].set_ylabel("1 kHz 고역 (g)")
ax[0].set_title(f"abn_022 새 충격 {len(d['t'])}개 (주황선) · A {len(evA[i])} · B {len(evB[i])}", fontsize=10)
ax[1].plot(tt_, d["env"] / d["base"], lw=.6, color=NOR_C); ax[1].axhline(CH["k"], color=ABN_C, ls="--"); ax[1].set_ylabel("포락선 / 바탕"); ax[1].set_xlabel("s")
fig.savefig(os.path.join(FIG, "7f_022_충격.png"), dpi=100, bbox_inches="tight"); plt.close(fig)

R["chosen"] = CH
R["chosen_text"] = f"1 kHz 고역 → |hilbert| → {CH['smooth']} Hz 다듬기 → 바탕 = " + ("100 ms 이동 중앙값" if CH["base"] == "moving" else "전체 중앙값") + f" → 포락선/바탕 ≥ {CH['k']} 봉우리, 간격 ≥ {CH['distance_ms']:g} ms"
json.dump(R, open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print(R["chosen_text"]); print("완료")
