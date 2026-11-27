# %% [markdown]
# # 2-11 대화형 EDA (Jupyter 셀 방식)
# VS Code에서 `# %%` 셀을 하나씩 실행(Shift+Enter)하며 본다. 셀 하나 = 질문 하나.
# 커널: `qe-phm`

# %% 0. 준비 — 루트 찾기, 데이터 읽기
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import signal as sg
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").is_dir())
sys.path.insert(0, str(ROOT))
from src.data import load_dataset
from src.exp import OUTLIER_NORMALS
print(sorted(OUTLIER_NORMALS)[:3])

plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False,
                    
                     "figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})
FS = 25_000

# %% 1. 개요 — 파일 수, 길이, 라벨
D = ROOT / "2_data/TeamProject_train_변경"
X, y, names = load_dataset(D)
names = np.array(names)
print(X.shape, "정상", (y == 0).sum(), "고장", (y == 1).sum())

# %% 2. 파일별 기본 통계 (시간 영역) — 표
df = pd.DataFrame({"name": names, "y": y,
                   "mean": X.mean(1), "std": X.std(1), "peak": np.abs(X).max(1),
                   "kurt": pd.DataFrame(X.T).kurt().values})
df["crest"] = df["peak"] / df["std"]
df.groupby("y")[["mean", "std", "peak", "kurt", "crest"]].describe().T.round(3)

# %% 3. 정상 vs 고장 박스플롯
fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
for a, c in zip(ax, ["std", "peak", "kurt", "crest"]):
    a.boxplot([df[df.y == 0][c], df[df.y == 1][c]], tick_labels=["정상", "고장"])
    a.set_title(c)
plt.tight_layout()

# %% 4. 파형 비교 — 보고 싶은 파일 이름을 바꿔 가며
def wave(*ns, t0=0.0, t1=2.0):
    fig, ax = plt.subplots(len(ns), 1, figsize=(12, 2.2 * len(ns)), sharex=True)
    ax = np.atleast_1d(ax)
    for a, n in zip(ax, ns):
        i = list(names).index(n)
        s = slice(int(t0 * FS), int(t1 * FS))
        a.plot(np.arange(X.shape[1])[s] / FS, X[i, s], lw=.4)
        a.set_ylabel(f"{n}\n(y={y[i]})")
    ax[-1].set_xlabel("시간 (s)")
    plt.tight_layout()
wave(names[0], names[-1])

# %% 5. PSD (Welch) — 정상 평균 대역 vs 고장 개별
f, P = sg.welch(X, FS, nperseg=4096, axis=1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.semilogy(f, P[y == 0].mean(0), label="정상 평균", c="C0")
ax.semilogy(f, P[y == 1].mean(0), label="고장 평균", c="C1")
ax.set_xlabel("주파수 (Hz)"); ax.legend()
# 로그축 눈금은 수식(mathtext)을 쓰면 '−' 글리프 경고가 난다 → 일반 문자열로
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}"))
ax.yaxis.set_minor_formatter(NullFormatter())

# %% 6. 백색화 — 파일별 AR(32) 잔차 (200 Hz 고역통과 후)
# 원리: 과거 32점으로 다음 값을 선형 예측 → 실제 − 예측 = 잔차. 예측되는 공진·지속음은 지워지고
#       예측 안 되는 갑작스러운 성분(충격)만 남는다. 스펙트럼은 평평(백색)해진다. 파일마다 자기 신호로 맞춘다.
from scipy import linalg, stats
SOS_HP = sg.butter(4, 200, "hp", fs=FS, output="sos")

def ar_whiten(x, p=32):
    r = np.array([x[:len(x) - k] @ x[k:] for k in range(p + 1)])   # 필요한 지연 p+1개만 (전체 상관은 느림)
    a = linalg.solve_toeplitz(r[:p], r[1:p + 1])
    w = sg.lfilter(np.r_[1, -a], [1], x)
    w[:p] = 0.0
    return w

XH = sg.sosfiltfilt(SOS_HP, X - X.mean(1, keepdims=True), axis=1)   # 원신호(200 Hz HP)
XW = np.stack([ar_whiten(x) for x in XH])                           # 백색화 신호

def kurt(a): return stats.kurtosis(a, axis=1, fisher=False)          # 정규분포 = 3
df["kurt_hp"], df["kurt_w"] = kurt(XH), kurt(XW)
df["crest_hp"] = np.abs(XH).max(1) / XH.std(1)
df["crest_w"] = np.abs(XW).max(1) / XW.std(1)
df.groupby("y")[["kurt_hp", "kurt_w", "crest_hp", "crest_w"]].median().round(2)

# %% 7. 백색화 전/후 — 정상 vs 고장 파형 (이름만 바꿔 가며)
def wave_cmp(*ns, t0=0.0, t1=0.5):
    fig, ax = plt.subplots(len(ns), 2, figsize=(14, 2.2 * len(ns)), sharex=True)
    ax = np.atleast_2d(ax)
    for r, n in zip(ax, ns):
        i = list(names).index(n); s = slice(int(t0 * FS), int(t1 * FS)); t = np.arange(X.shape[1])[s] / FS
        r[0].plot(t, XH[i, s], lw=.4); r[1].plot(t, XW[i, s], lw=.4, c="C3")
        r[0].set_ylabel(f"{n} (y={y[i]})")
    ax[0, 0].set_title("원신호 (200 Hz HP)"); ax[0, 1].set_title("백색화 후 (AR 32 잔차)")
    ax[-1, 0].set_xlabel("시간 (s)"); ax[-1, 1].set_xlabel("시간 (s)")
    plt.tight_layout()
wave_cmp(names[0], names[1], names[-1], names[-2])

# %% 8. 스펙트럼이 평평해졌는가 + 첨도 변화
fw, PW = sg.welch(XW, FS, nperseg=4096, axis=1)
fh, PH = sg.welch(XH, FS, nperseg=4096, axis=1)
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for yy, c, lb in [(0, "C0", "정상"), (1, "C1", "고장")]:
    ax[0].semilogy(fh, np.median(PH[y == yy], 0), c=c, label=lb)
    ax[1].semilogy(fw, np.median(PW[y == yy], 0), c=c, label=lb)
ax[0].set_title("원신호 PSD (중앙값)"); ax[1].set_title("백색화 후 PSD (중앙값)")
for a in ax[:2]:
    a.set_xlabel("주파수 (Hz)"); a.legend()
    a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0e}")); a.yaxis.set_minor_formatter(NullFormatter())
ax[2].scatter(df.kurt_hp[y == 0], df.kurt_w[y == 0], s=14, label="정상")
ax[2].scatter(df.kurt_hp[y == 1], df.kurt_w[y == 1], s=22, c="C1", label="고장")
o = np.isin(names, list(OUTLIER_NORMALS))
ax[2].scatter(df.kurt_hp[o], df.kurt_w[o], s=60, marker="D", facecolors="none", edgecolors="k", label="이상치 정상 ◆")
ax[2].set_xscale("log"); ax[2].set_yscale("log")
ax[2].set_xlabel("첨도 (원신호)"); ax[2].set_ylabel("첨도 (백색화 후)"); ax[2].set_title("파일별 첨도 전/후"); ax[2].legend()
plt.tight_layout()

# %% 9. 정상 기준 백색화 A — 효섭 방식: 정상 파일 앞 0.8초로 AR(40) 계수 → 모든 파일에 적용
# 원리: 정상들의 평균 자기상관(정상 진동의 '규칙')으로 예측 필터를 만든다. 정상은 잘 지워지고(잔차 작음),
#       정상답지 않은 신호는 덜 지워진다. 효섭: AR_res_ratio(잔차 에너지 / 원 에너지) AUC 0.833.
# 공정성: 정상 파일 자신은 기준에서 뺀다(leave-one-out). 이상치 정상 3개는 기준 만들 때 제외(원칙 5), 점수엔 포함.
from sklearn.metrics import roc_auc_score
P_AR, N08 = 40, int(0.8 * FS)
out = np.isin(names, list(OUTLIER_NORMALS))
ref_ok = (y == 0) & ~out                                   # 기준을 만드는 정상 = 이상치 제외
def acf(x, p=P_AR): return np.array([x[:N08 - k] @ x[k:N08] for k in range(p + 1)])
R = np.stack([acf(x) for x in XH]); R = R / R[:, :1]       # 파일마다 에너지가 달라 r0로 나눠 동등하게
Rsum = R[ref_ok].sum(0)

res_A = np.zeros_like(XH)
for i in range(len(XH)):
    Ri = (Rsum - (R[i] if ref_ok[i] else 0)) / (ref_ok.sum() - ref_ok[i])   # LOO 평균 자기상관
    a = linalg.solve_toeplitz(Ri[:P_AR], Ri[1:P_AR + 1])
    res_A[i] = sg.lfilter(np.r_[1, -a], [1], XH[i]); res_A[i, :P_AR] = 0
df["ARratio"] = (res_A ** 2).mean(1) / (XH ** 2).mean(1)
df["kurt_A"] = kurt(res_A)
auc = lambda v: roc_auc_score(y, v)
print("AUC  AR_res_ratio %.3f | 잔차 첨도 %.3f | (참고) 원신호 첨도 %.3f | 자기 AR 잔차 첨도 %.3f" %
      (auc(df.ARratio), auc(df.kurt_A), auc(df.kurt_hp), auc(df.kurt_w)))
df.groupby("y")[["ARratio", "kurt_A"]].median().round(3)

# %% 10. 정상 기준 백색화 B — 정상 평균 PSD로 나누기 (주파수 영역)
# 원리: 각 주파수에서 '정상이면 이만큼' 기준(정상 PSD 중앙값, 로그 영역)으로 나눈다 → 비율 1(0 dB)이면 정상답다.
#       FFT에서 1/sqrt(기준 PSD)를 곱하면 정상 스펙트럼이 평평해진 시간 신호가 된다.
f_w, PW0 = sg.welch(XH, FS, nperseg=4096, axis=1)
LP = np.log10(PW0)
LRef = np.zeros_like(LP)
for i in range(len(LP)):
    m = ref_ok.copy(); m[i] = False                         # LOO
    LRef[i] = np.median(LP[m], 0)
DB = 10 * (LP - LRef)                                       # 정상 기준 대비 dB (0 = 정상 중앙)
df["dB_mean"] = DB[:, 3:].mean(1)                           # 전 대역 평균 dB (200 Hz 아래 제외한 전체)
df["dB_absmean"] = np.abs(DB[:, 3:]).mean(1)                # 방향 무관 이탈 크기

fx = np.fft.rfftfreq(X.shape[1], 1 / FS)
res_B = np.zeros_like(XH)
for i in range(len(XH)):
    Pi = 10 ** np.interp(fx, f_w, LRef[i])
    Z = np.fft.rfft(XH[i]) / np.sqrt(Pi)
    Z[fx < 200] = 0
    res_B[i] = np.fft.irfft(Z, X.shape[1])
df["kurt_B"] = kurt(res_B)
print("AUC  dB_mean %.3f | dB_absmean %.3f | 정상기준 FFT 백색화 첨도 %.3f" % (auc(df.dB_mean), auc(df.dB_absmean), auc(df.kurt_B)))
df.groupby("y")[["dB_mean", "dB_absmean", "kurt_B"]].median().round(2)

# %% 11. 그림 — 정상 대비 dB 곡선, 첨도 전/후 (정상 기준 두 방식)
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
for yy, c, lb in [(0, "C0", "정상"), (1, "C1", "고장")]:
    d = DB[y == yy]
    ax[0].plot(f_w, np.median(d, 0), c=c, label=lb + " 중앙값")
    ax[0].fill_between(f_w, np.percentile(d, 10, 0), np.percentile(d, 90, 0), color=c, alpha=.15)
ax[0].axhline(0, c="k", lw=.6); ax[0].set_xlabel("주파수 (Hz)"); ax[0].set_ylabel("정상 기준 대비 (dB)")
ax[0].set_title("정상 평균 PSD로 나눈 값 (10–90% 띠)"); ax[0].legend()
for a, v, t in [(ax[1], df.kurt_A, "A: 정상 AR(40) 잔차 첨도"), (ax[2], df.kurt_B, "B: 정상 PSD 백색화 첨도")]:
    a.scatter(df.kurt_hp[y == 0], v[y == 0], s=14, label="정상")
    a.scatter(df.kurt_hp[y == 1], v[y == 1], s=22, c="C1", label="고장")
    a.scatter(df.kurt_hp[out], v[out], s=60, marker="D", facecolors="none", edgecolors="k", label="이상치 정상")
    a.set_xscale("log"); a.set_yscale("log"); a.set_xlabel("첨도 (원신호)"); a.set_ylabel("첨도 (백색화 후)"); a.set_title(t); a.legend()
plt.tight_layout()

# %% 12. 파일별 정상 대비 dB 히트맵 — 행 = 파일(정상 위 · 고장 아래, 각각 이탈 크기순), 열 = 주파수
# 파랑 = 정상보다 약함, 빨강 = 정상보다 강함, 흰색 = 정상 수준. ◆ = 이상치 정상
order = np.r_[np.where(y == 0)[0][np.argsort(df.dB_absmean[y == 0].values)],
              np.where(y == 1)[0][np.argsort(df.dB_absmean[y == 1].values)]]
fig, ax = plt.subplots(figsize=(13, 9))
im = ax.imshow(DB[order], aspect="auto", cmap="RdBu_r", vmin=-30, vmax=30,
               extent=[0, FS / 2, len(order), 0], interpolation="nearest")
ax.axhline((y == 0).sum(), c="k", lw=1.5)
ax.set_yticks([(y == 0).sum() / 2, (y == 0).sum() + (y == 1).sum() / 2]); ax.set_yticklabels(["정상 120", "고장 30"])
for r, i in enumerate(order):
    if out[i]: ax.text(-150, r + .5, "◆", ha="right", va="center", fontsize=7)
    if y[i] == 1: ax.text(FS / 2 + 120, r + .5, names[i][-3:], va="center", fontsize=7)
ax.set_xlabel("주파수 (Hz)"); ax.set_title("정상 기준 대비 PSD (dB), 파일별")
ax.grid(False); plt.colorbar(im, ax=ax, label="dB", pad=.06, shrink=.7)
plt.tight_layout()
lowF = df[(y == 1)].sort_values("dB_absmean").head(6)[["name", "dB_absmean", "ARratio"]].round(2)
print(lowF.to_string())

# %% 13. 스펙트럴 첨도 SK(f) — 주파수마다 '에너지가 가끔 튀는가'
# SK(f) = <|S|^4> / <|S|^2>^2 − 2   (S = STFT, < > = 시간 프레임 평균)
#   가우스 잡음 = 0 · 일정한 지속음 = −1 · 가끔 터지는 충격 = 큰 양수
# 창 1024점(41 ms, 24 Hz 간격) · 홉 512. 대역은 라벨과 무관하게 1 kHz 고정 격자.
NSEG, HOP_ = 1024, 512
def sk_curve(x):
    f, _, Z = sg.stft(x, FS, nperseg=NSEG, noverlap=NSEG - HOP_, boundary=None, padded=False)
    P = np.abs(Z) ** 2
    return f, (P ** 2).mean(1) / (P.mean(1) ** 2 + 1e-30) - 2
fk, _ = sk_curve(XH[0])
SK = np.stack([sk_curve(x)[1] for x in XH])                    # (150, 주파수칸)
EDGES = np.arange(0, 12001, 1000); EDGES = np.r_[EDGES, 12500]
bands = [(a, b) for a, b in zip(EDGES[:-1], EDGES[1:])]
SKB = np.stack([SK[:, (fk >= max(a, 200)) & (fk < b)].mean(1) for a, b in bands], 1)   # (150, 13 대역)
df["SK_mean"] = SK[:, fk >= 200].mean(1)
df["SK_max"] = SK[:, fk >= 200].max(1)
df["SK_med"] = np.median(SK[:, fk >= 200], 1)
print("AUC  SK 평균 %.3f | SK 최대 %.3f | SK 중앙 %.3f | (참고) 원신호 첨도 %.3f" %
      (auc(df.SK_mean), auc(df.SK_max), auc(df.SK_med), auc(df.kurt_hp)))
print("대역별 AUC:", {f"{a//1000}-{b/1000:g}k": round(roc_auc_score(y, SKB[:, j]), 2) for j, (a, b) in enumerate(bands)})

# %% 14. 그림 — SK 곡선(정상 vs 고장) · 대역별 AUC
fig, ax = plt.subplots(1, 2, figsize=(15, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
for yy, c, lb in [(0, "C0", "정상"), (1, "C1", "고장")]:
    d = SK[y == yy]
    ax[0].plot(fk, np.median(d, 0), c=c, label=lb + " 중앙값")
    ax[0].fill_between(fk, np.percentile(d, 10, 0), np.percentile(d, 90, 0), color=c, alpha=.15)
ax[0].axhline(0, c="k", lw=.6); ax[0].set_ylim(-2, 25)
ax[0].set_xlabel("주파수 (Hz)"); ax[0].set_ylabel("스펙트럴 첨도"); ax[0].set_title("SK(f) 파일별 중앙값 (10–90% 띠)"); ax[0].legend()
aucs = [roc_auc_score(y, SKB[:, j]) for j in range(len(bands))]
ax[1].bar(range(len(bands)), aucs, color=["C3" if a > .5 else "C0" for a in aucs])
ax[1].axhline(.5, c="k", lw=.8); ax[1].set_ylim(0.3, 1)
ax[1].set_xticks(range(len(bands))); ax[1].set_xticklabels([f"{a//1000}–{b/1000:g}" for a, b in bands], rotation=45, fontsize=8)
ax[1].set_xlabel("대역 (kHz, 1 kHz 격자)"); ax[1].set_ylabel("AUC (고장 높음 = 1)"); ax[1].set_title("대역별 SK 평균의 AUC (탐색용)")
plt.tight_layout()

# %% 15. SK 요약이 기존 정보와 얼마나 겹치는가 + 이상치 정상 위치
from scipy.stats import spearmanr
cols = ["SK_mean", "kurt_hp", "dB_absmean", "ARratio", "std"]
print(df[cols].corr(method="spearman").round(2).to_string())
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, c2 in zip(ax, ["kurt_hp", "dB_absmean"]):
    a.scatter(df[c2][y == 0], df.SK_mean[y == 0], s=14, label="정상")
    a.scatter(df[c2][y == 1], df.SK_mean[y == 1], s=22, c="C1", label="고장")
    a.scatter(df[c2][out], df.SK_mean[out], s=60, marker="D", facecolors="none", edgecolors="k", label="이상치 정상")
    a.set_xlabel(c2); a.set_ylabel("SK 평균"); a.legend()
    if c2 == "kurt_hp": a.set_xscale("log")
plt.tight_layout()
print(df[out][["name", "SK_mean", "SK_max"]].round(2).to_string())
print(df.groupby("y")[["SK_mean", "SK_max", "SK_med"]].median().round(2).to_string())
