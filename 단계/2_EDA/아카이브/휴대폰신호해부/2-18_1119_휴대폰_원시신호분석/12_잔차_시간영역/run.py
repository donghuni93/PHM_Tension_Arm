# %% [markdown]
# 12. 시간 영역 잔차: AR(32) 백색화 잔차 · 잔차 STFT · 로컬 포락선 대비 잔차 비율 · 충격 주기성
# 질문: ① 오탐 경계 정상 잔차는 원신호 포락선에 비례해 커지는 '본체 모드 연동형'인가?
#       ② 미탐 고장(022·014·017·002)은 특정 간격으로 반복해서 튀는 '틱' 충격인가? (017 외에도 보이는가)
# 방법: 200 Hz 고역통과 → 파일별 표준편차로 정규화 → 일반 정상(오탐 경계 4개 제외)으로 AR(32) 최소제곱 → 전체 파일 잔차 e = x − 예측
#       로컬 RMS(2 ms 창): 잔차 / 원신호 = 정규화 잔차 비율. 충격 = 잔차 로컬 RMS > 중앙값 × 6 (간격 ≥ 5 ms).
#       충격 주기성 = 잔차 로컬 RMS의 자기상관 피크(5–200 ms).
# 주의: AR은 보기용으로 일반 정상 전체에 맞춤(폴드 밖). 피처로 쓰면 폴드 안에서 다시 맞춰야 한다.
# 실행(PC): conda run -n qe-phm python "단계/2_EDA/아카이브/휴대폰신호해부/2-18_1119_휴대폰_원시신호분석/12_잔차_시간영역/run.py"

# %% 설정
import os, sys
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):      # 한글 글꼴 (있는 첫 번째)
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
from scipy.signal import butter, sosfiltfilt, lfilter, find_peaks, stft
from scipy.stats import kurtosis
from src.data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000; P = 32; W = 50                       # AR 차수, 로컬 RMS 창 50점 = 2 ms
X, y, names = load_dataset(os.path.join(_d, "data", "TeamProject_train_변경"))

FP_NORMAL = [116, 59, 55, 106]
FN_FAULT = [22, 14, 17, 2]
def find(num, label):
    for i, (n, l) in enumerate(zip(names, y)):
        if l == label and int("".join(c for c in n if c.isdigit()) or -1) == num:
            return i
    raise KeyError(f"{num} (label {label}) 없음. names 예: {names[:3]}")
IDX_FP = [find(n, 0) for n in FP_NORMAL]
IDX_FN = [find(n, 1) for n in FN_FAULT]
GEN_N = np.array([i for i in np.where(y == 0)[0] if i not in IDX_FP])
OTH_F = np.array([i for i in np.where(y == 1)[0] if i not in IDX_FN])
def cohort(i):
    return "오탐 경계 정상" if i in IDX_FP else "미탐 경계 고장" if i in IDX_FN else "일반 정상" if y[i] == 0 else "일반 고장"
COLOR = {"일반 정상": "gray", "오탐 경계 정상": "tab:blue", "미탐 경계 고장": "tab:red", "일반 고장": "darkorange"}

# %% AR(32) 학습 (정규방정식을 파일별로 누적 → 메모리 절약) · 잔차
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=-1)
H = H / H.std(1, keepdims=True)
A = np.zeros((P, P)); b = np.zeros(P)
for i in GEN_N:
    x = H[i]
    Wm = np.lib.stride_tricks.sliding_window_view(x[:-1], P)[:, ::-1]      # 행 t: x[t+P-1] … x[t] → 예측 대상 x[t+P]
    A += Wm.T @ Wm; b += Wm.T @ x[P:]
a = np.linalg.solve(A, b)
E = np.stack([lfilter(np.r_[1, -a], [1], H[i])[P:] for i in range(len(y))])   # (150, N−P) 잔차
Xa = H[:, P:]
def lrms(v):                                                                    # 2 ms 로컬 RMS
    return np.sqrt(np.apply_along_axis(lambda r: np.convolve(r ** 2, np.ones(W) / W, "same"), -1, v))
Le, Lx = lrms(E), lrms(Xa)
R = Le / (Lx + 1e-9)                                                            # 정규화 잔차 비율
t = np.arange(E.shape[1]) / FS

# %% 파일별 요약표 (충격 수 · 간격 규칙성 · 자기상관 피크)
def acf_peak(env):
    v = env - env.mean()
    n = len(v); F = np.fft.rfft(v, 2 * n); ac = np.fft.irfft(F * np.conj(F))[:5001]
    ac = ac / (ac[0] + 1e-12)
    pk, _ = find_peaks(ac[125:], prominence=0.05)      # 중심 로브가 줄어드는 경계가 아니라 진짜 극대만
    if len(pk) == 0: return ac, np.nan, np.nan
    k = pk[np.argmax(ac[125:][pk])] + 125
    return ac, ac[k], k / FS * 1000
rows = []
ACS = {}
PKT = {}
for i in range(len(y)):
    e = E[i]; env = Le[i]
    pk, _ = find_peaks(env, height=6 * np.median(env), distance=125)
    iv = np.diff(pk) / FS * 1000
    PKT[i] = pk / FS + P / FS
    ac, ach, lag = acf_peak(env)
    ACS[i] = ac
    rows.append({"파일": names[i], "코호트": cohort(i), "잔차 첨도": kurtosis(e), "잔차/원신호 RMS": e.std() / Xa[i].std(),
                 "최대 로컬비율": R[i].max(), "충격 수": len(pk), "간격 CV": (iv.std() / iv.mean()) if len(iv) >= 3 else np.nan,
                 "자기상관 피크": ach, "피크 지연(ms)": lag})
T = pd.DataFrame(rows)
T.round(3).to_csv(os.path.join(HERE, "residual_summary.csv"), index=False, encoding="utf-8-sig")
print(T[T["코호트"].isin(["오탐 경계 정상", "미탐 경계 고장"])].round(3).to_string(index=False))

# %% 그림 1: 8개 파일 잔차 로컬 RMS · 정규화 잔차 비율 (시간축)
FOCUS = IDX_FP + IDX_FN
fig, ax = plt.subplots(len(FOCUS), 2, figsize=(15, 1.7 * len(FOCUS)), sharex=True)
for r, i in enumerate(FOCUS):
    c = COLOR[cohort(i)]
    ax[r, 0].plot(t, Le[i], color=c, lw=0.6); ax[r, 0].plot(t, Lx[i], color="k", lw=0.5, alpha=0.5)
    ax[r, 0].set_ylabel(f"{cohort(i)[:4]} {names[i]}", fontsize=7)
    ax[r, 1].plot(t, R[i], color=c, lw=0.6); ax[r, 1].set_ylim(0, max(1.0, np.percentile(R[FOCUS], 99.9)))
ax[0, 0].set_title("로컬 RMS: 색 = 잔차, 검정 = 원신호 (정규화 단위)"); ax[0, 1].set_title("정규화 잔차 비율 = 잔차 / 원신호 (로컬 2 ms)")
ax[-1, 0].set_xlabel("시간 (s)"); ax[-1, 1].set_xlabel("시간 (s)")
fig.savefig(os.path.join(FIG, "residual_time.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 그림 2: 잔차 STFT (같은 색 범위 = 일반 정상 10개 표본 1–99.5%)
def spec(v):
    ff, tt, Z = stft(v, fs=FS, nperseg=256, noverlap=192, boundary=None, padded=False)
    return ff, tt, 10 * np.log10(np.abs(Z) ** 2 + 1e-15)
samp = GEN_N[np.linspace(0, len(GEN_N) - 1, 10).astype(int)]
pool = np.concatenate([spec(E[i])[2].ravel() for i in samp]); vmin, vmax = np.percentile(pool, [1, 99.5])
fig, ax = plt.subplots(2, 4, figsize=(18, 7), sharey=True)
for k, i in enumerate(FOCUS):
    ff, tt, S = spec(E[i]); a = ax[k // 4, k % 4]
    m = ff <= 12500
    im = a.pcolormesh(tt, ff[m] / 1000, S[m], vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
    a.set_title(f"{cohort(i)} {names[i]}", fontsize=9, color=COLOR[cohort(i)])
    if k % 4 == 0: a.set_ylabel("주파수 (kHz)")
    if k >= 4: a.set_xlabel("시간 (s)")
fig.colorbar(im, ax=ax, label="잔차 STFT dB", shrink=0.6)
fig.savefig(os.path.join(FIG, "residual_stft.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 그림 3: 잔차 로컬 RMS 자기상관 (5–200 ms). 일정한 지연에 피크 = 주기적으로 반복되는 충격
lag_ms = np.arange(5001) / FS * 1000
fig, ax = plt.subplots(2, 4, figsize=(18, 6), sharey=True, sharex=True)
for k, i in enumerate(FOCUS):
    a = ax[k // 4, k % 4]
    a.plot(lag_ms, ACS[i], color=COLOR[cohort(i)], lw=0.8)
    a.set_title(f"{cohort(i)} {names[i]}", fontsize=9, color=COLOR[cohort(i)]); a.set_xlim(0, 200)
    if k >= 4: a.set_xlabel("지연 (ms)")
fig.savefig(os.path.join(FIG, "residual_acf.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 그림 4: 코호트별 산점도 (150개 전부, 색 = 코호트, 경계 파일만 이름 표시)
fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
pairs = [("잔차 첨도", "잔차/원신호 RMS", "log"), ("충격 수", "자기상관 피크", "lin")]
for a, (xc, yc, sc) in zip(ax, pairs):
    for g, c in COLOR.items():
        s = T[T["코호트"] == g]
        a.scatter(s[xc], s[yc], s=14 if g in ("일반 정상", "일반 고장") else 50, color=c, alpha=0.5 if g in ("일반 정상", "일반 고장") else 0.95, label=g)
    for _, rw in T[T["코호트"].isin(["오탐 경계 정상", "미탐 경계 고장"])].iterrows():
        a.annotate(rw["파일"], (rw[xc], rw[yc]), fontsize=7)
    a.set_xlabel(xc); a.set_ylabel(yc)
    if sc == "log": a.set_xscale("log"); a.set_yscale("log")
ax[0].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "residual_scatter.png"), dpi=130, bbox_inches="tight"); plt.close(fig)

# %% 그림 5: 충격 시각 히스토그램 (파일 시작 시각이 가진 위상과 연동돼 있는가? 코호트별 충격이 몰리는 시각)
fig, ax = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
for a, (g, idx) in zip(ax, (("일반 정상", GEN_N), ("일반 고장", OTH_F), ("오탐 경계 정상", IDX_FP), ("미탐 경계 고장", IDX_FN))):
    allt = np.concatenate([PKT[i] for i in idx]); a.hist(allt, bins=np.arange(0, 2.05, 0.05), color=COLOR[g])
    a.set_ylabel(f"{g} (파일 {len(idx)}개)\n충격 수", fontsize=8)
ax[-1].set_xlabel("충격 시각 (s), 빈 50 ms")
fig.savefig(os.path.join(FIG, "impact_time_hist.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 합본 (휴대폰용): 그림 5장을 세로로
import matplotlib.image as mpimg
parts = ["residual_time.png", "residual_stft.png", "residual_acf.png", "residual_scatter.png", "impact_time_hist.png"]
imgs = [mpimg.imread(os.path.join(FIG, p))[..., :3] for p in parts]
Wd = max(im.shape[1] for im in imgs)
big = np.concatenate([np.pad(im, ((0, 0), ((Wd - im.shape[1]) // 2, Wd - im.shape[1] - (Wd - im.shape[1]) // 2), (0, 0)), constant_values=1.0) for im in imgs])
plt.imsave(os.path.join(FIG, "residual_all.png"), big)
print("저장:", FIG)
