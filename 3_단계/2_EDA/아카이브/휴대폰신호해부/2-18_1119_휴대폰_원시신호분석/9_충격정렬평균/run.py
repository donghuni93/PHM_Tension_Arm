# %% [markdown]
# 9. 충격 정렬 평균 (시간 동기 평균) + 트리거 편향 대조 + 017용 보조 진단(스펙트럼 첨도 · 스케일 불변 기울기)
# 질문: ① 022처럼 충격이 몇 개 있는 파일에서 정렬 평균을 하면 결함 특유의 울림(ring-down)이 드러나는가?
#       ② 017(RMS 0.002 g)은 정렬해도 파형이 모이지 않는가? (모이면 충격성 결함, 안 모이면 noise-riding)
# 트리거 편향 대조: 같은 파일의 **위상 무작위 대용 신호**(스펙트럼은 같고 충격 구조는 파괴)에 똑같은 검출·정렬을 해서 평균을 낸다.
#   실제 평균의 진폭(peak-to-peak)이 대용 평균보다 충분히 크면(R = 실제/대용 ≫ 1) 노이즈를 주워 모은 것이 아니라 실제 충격 구조다.
# 방법: 200 Hz 고역통과 신호를 평균(극성 정렬). 충격 검출 = 1 kHz 고역통과 → Hilbert → 500 Hz 저역 → 포락선 봉우리, 파일당 상위 K=10개(임계값 대신 고정 K),
#       봉우리 ±1 ms 안의 |x| 최대점으로 미세 정렬. 창 = −2 ms … +20 ms. 가장자리 20 ms는 제외(필터 가장자리 효과 회피, 길이는 미검증).
# 보조 진단: 스펙트럼 첨도 SK = <|Z|^4>/<|Z|^2>^2 − 2 (STFT 512), PSD 로그-로그 기울기(1–2 kHz 미만/이상 두 구간, 진폭 스케일과 무관).
# 보기용이다. 실행(PC 또는 컨테이너): python "3_단계/2_EDA/아카이브/휴대폰신호해부/2-18_1119_휴대폰_원시신호분석/9_충격정렬평균/run.py"

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
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, stft, welch
from scipy.stats import kurtosis
from src.data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000; PRE, POST = 50, 500; K = 10; NSUR = 10; MARGIN = int(0.02 * FS)
X, y, names = load_dataset(os.path.join(_d, "2_data", "TeamProject_train_변경"))
rng = np.random.default_rng(0)

FP_NORMAL = [116, 59, 55, 106]
FN_FAULT = [22, 14, 17, 2]
def find(num, label):
    for i, (n, l) in enumerate(zip(names, y)):
        if l == label and int("".join(c for c in n if c.isdigit()) or -1) == num:
            return i
    raise KeyError(f"{num} (label {label}) 없음. names 예: {names[:3]}")
IDX_FP = [find(n, 0) for n in FP_NORMAL]; IDX_FN = [find(n, 1) for n in FN_FAULT]; FOCUS = IDX_FP + IDX_FN
GEN_N = np.array([i for i in np.where(y == 0)[0] if i not in IDX_FP])
OTH_F = np.array([i for i in np.where(y == 1)[0] if i not in IDX_FN])
def cohort(i):
    return "오탐 경계 정상" if i in IDX_FP else "미탐 경계 고장" if i in IDX_FN else "일반 정상" if y[i] == 0 else "일반 고장"
COLOR = {"일반 정상": "gray", "오탐 경계 정상": "tab:blue", "미탐 경계 고장": "tab:red", "일반 고장": "darkorange"}

# %% 필터 · 검출 · 정렬 평균 함수
SOS200 = butter(4, 200, btype="highpass", fs=FS, output="sos")
SOS1K = butter(4, 1000, btype="highpass", fs=FS, output="sos")
SOSLP = butter(4, 500, btype="lowpass", fs=FS, output="sos")
Xc = X - X.mean(1, keepdims=True)
HP = sosfiltfilt(SOS200, Xc, axis=-1)

def pick(x, k=K):
    """x(200 Hz 고역통과 신호)에서 포락선 상위 k개 충격 시점(샘플 번호)"""
    env = sosfiltfilt(SOSLP, np.abs(hilbert(sosfiltfilt(SOS1K, x))))
    pk, _ = find_peaks(env, distance=125)
    pk = pk[(pk >= MARGIN + PRE) & (pk < len(x) - MARGIN - POST)]
    top = pk[np.argsort(env[pk])[::-1][:k]]
    return np.array(sorted({p - 25 + int(np.argmax(np.abs(x[p - 25:p + 26]))) for p in top}))

def cmean(x, tr):
    """극성을 맞춘 정렬 평균 파형 (PRE+POST,)과 사용한 충격 수"""
    segs = [x[q - PRE:q + POST] * np.sign(x[q]) for q in tr if q - PRE >= 0 and q + POST <= len(x)]
    return (np.mean(segs, 0) if segs else np.full(PRE + POST, np.nan)), len(segs)

def surrogate(x):
    """위상 무작위: 진폭 스펙트럼은 그대로, 시간 구조(충격)는 파괴"""
    F = np.fft.rfft(x); ph = rng.uniform(0, 2 * np.pi, len(F)); ph[0] = 0; ph[-1] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=len(x))

def ring(m):
    """평균 파형 m의 울림 주파수(Hz)와 감쇠비 ζ (피크 이후 15 ms, 포락선 1–8 ms 로그 선형 피팅)"""
    seg = m[PRE:PRE + 375] * np.hanning(2 * 375)[375:]
    F = np.abs(np.fft.rfft(seg, 8192)); fr = np.fft.rfftfreq(8192, 1 / FS)
    mk = (fr > 800) & (fr < 12000); f0 = fr[mk][np.argmax(F[mk])]
    a = np.abs(hilbert(m))[PRE + 25:PRE + 200]; tt = np.arange(len(a)) / FS
    sl = np.polyfit(tt, np.log(a + 1e-12), 1)[0]
    return f0, (-sl / (2 * np.pi * f0) if sl < 0 else np.nan)

# %% 150개 전부: 정렬 평균, 대용 평균, R, 울림 특성
rows, CM, SUR = [], {}, {}
for i in range(len(y)):
    tr = pick(HP[i]); m, n = cmean(HP[i], tr); CM[i] = m
    surs = []
    for _ in range(NSUR):
        xs = surrogate(HP[i]); xs = sosfiltfilt(SOS200, xs)
        ms, _n = cmean(xs, pick(xs)); surs.append(ms)
    surs = np.array(surs); SUR[i] = surs
    p2p = np.nanmax(m) - np.nanmin(m); sp2p = np.mean(np.nanmax(surs, 1) - np.nanmin(surs, 1))
    f0, z = ring(m) if n > 0 else (np.nan, np.nan)
    rows.append({"파일": names[i], "코호트": cohort(i), "사용 충격 수": n, "RMS(g)": HP[i].std(), "R(실제/대용)": p2p / sp2p,
                 "피크 진폭/RMS": abs(m[PRE]) / HP[i].std(), "울림 주파수(Hz)": f0, "감쇠비 ζ": z})
T = pd.DataFrame(rows)
T.round(4).to_csv(os.path.join(HERE, "aligned_mean_summary.csv"), index=False, encoding="utf-8-sig")
print(T[T["코호트"].isin(["오탐 경계 정상", "미탐 경계 고장"])].round(3).to_string(index=False))
print("R 코호트 중앙값:\n", T.groupby("코호트")["R(실제/대용)"].median().round(2).to_string())

# %% 그림 1: 8개 파일 정렬 평균 파형 + 대용(노이즈 편향) 대역
t_ms = (np.arange(PRE + POST) - PRE) / FS * 1000
fig, ax = plt.subplots(4, 2, figsize=(14, 11), sharex=True)
for k, i in enumerate(FOCUS):
    a = ax[k % 4, k // 4]; s = SUR[i]
    a.fill_between(t_ms, np.nanpercentile(s, 2, axis=0), np.nanpercentile(s, 98, axis=0), color="lightgray", label="대용(위상 무작위) 2–98%")
    a.plot(t_ms, CM[i], color=COLOR[cohort(i)], lw=1.1, label=f"{cohort(i)} {names[i]}")
    a.set_title(f"{names[i]} ({cohort(i)}): R={T.loc[i, 'R(실제/대용)']:.1f}, 충격 {T.loc[i, '사용 충격 수']}개, RMS {T.loc[i, 'RMS(g)']:.4f} g", fontsize=9)
    a.axvline(0, color="k", lw=0.4)
for a in ax[-1]: a.set_xlabel("충격 기준 시간 (ms)")
fig.savefig(os.path.join(FIG, "aligned_waveforms.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 그림 2: 정렬 평균의 스펙트럼 (울림 주파수) vs 일반 정상 5–95%
def mspec(m):
    seg = m[PRE:] * np.hanning(2 * len(m[PRE:]))[len(m[PRE:]):]
    return 20 * np.log10(np.abs(np.fft.rfft(seg, 4096)) + 1e-12)
fr = np.fft.rfftfreq(4096, 1 / FS)
SP = np.array([mspec(CM[i]) if not np.isnan(CM[i]).any() else np.full(len(fr), np.nan) for i in range(len(y))])
SP_n = SP - np.nanmax(SP, axis=1, keepdims=True)                      # 진폭 정규화(피크 0 dB): 모양만 비교
lo, hi = np.nanpercentile(SP_n[GEN_N], [5, 95], axis=0)
fig, ax = plt.subplots(1, 1, figsize=(13, 4.5))
ax.fill_between(fr / 1000, lo, hi, color="gray", alpha=0.35, label="일반 정상 5–95%")
for i in IDX_FP: ax.plot(fr / 1000, SP_n[i], color="tab:blue", lw=0.9, label=f"오탐 정상 {names[i]}")
for i in IDX_FN: ax.plot(fr / 1000, SP_n[i], color="tab:red", lw=0.9, label=f"미탐 고장 {names[i]}")
ax.plot(fr / 1000, np.nanmedian(SP_n[OTH_F], 0), color="darkorange", lw=2, label="일반 고장 중앙값")
ax.set_xlim(0, 12.5); ax.set_xlabel("주파수 (kHz)"); ax.set_ylabel("정렬 평균 스펙트럼 (피크 0 dB)"); ax.legend(fontsize=7, ncol=3)
fig.savefig(os.path.join(FIG, "aligned_spectra.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 그림 3: R · 울림 주파수/감쇠비 · 충격 수 (코호트 색)
fig, ax = plt.subplots(1, 3, figsize=(18, 5))
for g, c in COLOR.items():
    s = T[T["코호트"] == g]; big = g in ("오탐 경계 정상", "미탐 경계 고장")
    kw = dict(s=55 if big else 14, color=c, alpha=0.95 if big else 0.45, label=g)
    ax[0].scatter(s["RMS(g)"], s["R(실제/대용)"], **kw); ax[1].scatter(s["울림 주파수(Hz)"] / 1000, s["감쇠비 ζ"], **kw)
    ax[2].scatter(s["사용 충격 수"] + rng.uniform(-0.2, 0.2, len(s)), s["피크 진폭/RMS"], **kw)
    if big:
        for _, r in s.iterrows():
            ax[0].annotate(r["파일"], (r["RMS(g)"], r["R(실제/대용)"]), fontsize=7)
            ax[1].annotate(r["파일"], (r["울림 주파수(Hz)"] / 1000, r["감쇠비 ζ"]), fontsize=7)
ax[0].set_xscale("log"); ax[0].set_xlabel("RMS (g)"); ax[0].set_ylabel("R = 실제 / 대용 (1이면 노이즈 편향)"); ax[0].axhline(1, color="k", lw=0.5); ax[0].legend(fontsize=8)
ax[1].set_xlabel("울림 주파수 (kHz)"); ax[1].set_ylabel("감쇠비 ζ"); ax[1].set_yscale("log")
ax[2].set_xlabel("사용 충격 수"); ax[2].set_ylabel("피크 진폭 / RMS")
fig.savefig(os.path.join(FIG, "aligned_scatter.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 017용 보조 진단: 스펙트럼 첨도(SK) · PSD 로그-로그 기울기 (스케일 불변)
fz, tz, Z = stft(Xc, fs=FS, nperseg=512, noverlap=256, boundary=None, padded=False, axis=-1)
P2 = np.abs(Z) ** 2
SK = (P2 ** 2).mean(-1) / (P2.mean(-1) ** 2 + 1e-30) - 2
fw, Pw = welch(Xc, fs=FS, nperseg=2048, axis=-1)
def slope(lo_, hi_):
    m = (fw >= lo_) & (fw < hi_); lf = np.log10(fw[m])
    return np.array([np.polyfit(lf, np.log10(Pw[i, m] + 1e-30), 1)[0] for i in range(len(y))])
S_lo, S_hi = slope(200, 2000), slope(2000, 12000)
bands = {"SK 2–4k": (2000, 4000), "SK 5.5–7.75k": (5500, 7750), "SK 8.0–8.7k": (8000, 8700), "SK 9–10k": (9000, 10000)}
D = pd.DataFrame({"파일": names, "코호트": [cohort(i) for i in range(len(y))], "RMS(g)": Xc.std(1), "기울기 0.2–2k": S_lo, "기울기 2–12k": S_hi})
for k, (a_, b_) in bands.items():
    m = (fz >= a_) & (fz <= b_); D[k] = SK[:, m].mean(1)
D.round(4).to_csv(os.path.join(HERE, "sk_slope_summary.csv"), index=False, encoding="utf-8-sig")
print(D[D["코호트"].isin(["오탐 경계 정상", "미탐 경계 고장"])].round(3).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(19, 5))
lo, hi = np.percentile(SK[GEN_N], [5, 95], axis=0)
ax[0].fill_between(fz / 1000, lo, hi, color="gray", alpha=0.35, label="일반 정상 5–95%")
for i in IDX_FP: ax[0].plot(fz / 1000, SK[i], color="tab:blue", lw=0.8)
for i in IDX_FN: ax[0].plot(fz / 1000, SK[i], color="tab:red", lw=0.8, label=f"고장 {names[i]}")
ax[0].set_yscale("symlog"); ax[0].set_xlim(0, 12.5); ax[0].set_xlabel("주파수 (kHz)"); ax[0].set_ylabel("스펙트럼 첨도 (가우시안=0)"); ax[0].legend(fontsize=7)
for g, c in COLOR.items():
    s = D[D["코호트"] == g]; big = g in ("오탐 경계 정상", "미탐 경계 고장")
    kw = dict(s=55 if big else 14, color=c, alpha=0.95 if big else 0.45, label=g)
    ax[1].scatter(s["기울기 0.2–2k"], s["기울기 2–12k"], **kw); ax[2].scatter(s["RMS(g)"], s["기울기 2–12k"], **kw)
    if big:
        for _, r in s.iterrows():
            ax[1].annotate(r["파일"], (r["기울기 0.2–2k"], r["기울기 2–12k"]), fontsize=7); ax[2].annotate(r["파일"], (r["RMS(g)"], r["기울기 2–12k"]), fontsize=7)
ax[1].set_xlabel("로그-로그 기울기 0.2–2 kHz"); ax[1].set_ylabel("기울기 2–12 kHz"); ax[2].set_xscale("log"); ax[2].set_xlabel("RMS (g)"); ax[2].set_ylabel("기울기 2–12 kHz")
ax[1].legend(fontsize=8)
fig.savefig(os.path.join(FIG, "sk_slope.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 합본 (휴대폰용)
import matplotlib.image as mpimg
parts = ["aligned_waveforms.png", "aligned_spectra.png", "aligned_scatter.png", "sk_slope.png"]
imgs = [mpimg.imread(os.path.join(FIG, p))[..., :3] for p in parts]
Wd = max(im.shape[1] for im in imgs)
big = np.concatenate([np.pad(im, ((0, 0), ((Wd - im.shape[1]) // 2, Wd - im.shape[1] - (Wd - im.shape[1]) // 2), (0, 0)), constant_values=1.0) for im in imgs])
plt.imsave(os.path.join(FIG, "aligned_all.png"), big); print("저장:", FIG)
