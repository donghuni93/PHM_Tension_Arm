# %% [markdown]
# 1. STFT 스펙트로그램 비교
# 질문: 오탐 경계 정상 파일(116·059·055·106)의 시간-주파수 구조가 (a) 대표 정상, (b) 놓친/저에너지 고장(022·014·017·002)과 어디가 닮고 어디가 다른가?
# 원리: 짧은 창마다 FFT → 가로 = 시간(0–2 s), 세로 = 주파수, 색 = log 파워(dB).
#   충격 하나 = 세로 줄무늬(모든 주파수에 동시에 에너지), 지속 공진 = 가로 띠, 울림(ring-down) = 줄무늬 끝이 주파수별로 늘어짐.
# 창 길이 트레이드오프: 짧은 창(256점, 10 ms)은 충격 시각이 선명, 긴 창(1024점, 41 ms)은 주파수 위치가 선명.
# 실행(PC): conda run -n qe-phm python "단계/2_EDA/아카이브/휴대폰신호해부/2-18_1119_휴대폰_원시신호분석/1_STFT/run.py"   (결과 그림 = 같은 폴더 figs/)
# 이 스크립트는 보기용이다. 피처화는 결과를 본 뒤 새 EXP로.

# %% 설정
import os, sys
import numpy as np
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
from scipy.signal import stft
from src.data import load_dataset

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000
X, y, names = load_dataset(os.path.join(_d, "data", "TeamProject_train_변경"))

# 비교 파일 (README의 EXP-034 · 2-16 기록 기준. 이름 끝 번호로 찾음, 라벨 지정)
FP_NORMAL = [116, 59, 55, 106]          # 정상인데 경계/오경보
FN_FAULT  = [22, 14, 17, 2]             # 놓친·저에너지 고장
def find(num, label):
    for i, (n, l) in enumerate(zip(names, y)):
        if l == label and int("".join(c for c in n if c.isdigit()) or -1) == num:
            return i
    raise KeyError(f"{num} (label {label}) 없음. names 예: {names[:3]}")
IDX_FP = [find(n, 0) for n in FP_NORMAL]
IDX_FN = [find(n, 1) for n in FN_FAULT]
# 대표 정상: 파일별 총 RMS가 정상 중앙값에 가장 가까운 3개 (라벨 외 정보로만 선택)
rms = X.std(1)
nor = np.where(y == 0)[0]
IDX_REF = list(nor[np.argsort(np.abs(rms[nor] - np.median(rms[nor])))[:3]])

# %% STFT (평균 제거 후 log10 파워, dB)
def spec(i, nper):
    x = X[i] - X[i].mean()
    f, t, Z = stft(x, fs=FS, nperseg=nper, noverlap=nper * 3 // 4, boundary=None, padded=False)
    return f, t, 10 * np.log10(np.abs(Z) ** 2 + 1e-15)

def label(i):
    return f"{'고장' if y[i] else '정상'} {names[i]}  (RMS {rms[i]:.3f} g)"

def grid(groups, nper, fname, fmax=12500, title=""):
    """groups = [(행 제목, [인덱스...])]. 모든 패널이 같은 색 범위(정상 전체의 1–99%)를 쓴다"""
    f, t, ref = spec(IDX_REF[0], nper)
    pool = np.concatenate([spec(i, nper)[2].ravel() for i in IDX_REF])
    vmin, vmax = np.percentile(pool, [1, 99.5])
    ncol = max(len(g[1]) for g in groups)
    fig, ax = plt.subplots(len(groups), ncol, figsize=(4.2 * ncol, 3.2 * len(groups)), squeeze=False, sharey=True)
    for r, (gt, idxs) in enumerate(groups):
        for c in range(ncol):
            a = ax[r, c]
            if c >= len(idxs): a.axis("off"); continue
            f, t, S = spec(idxs[c], nper)
            m = f <= fmax
            im = a.pcolormesh(t, f[m] / 1000, S[m], vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
            a.set_title(label(idxs[c]), fontsize=8)
            if c == 0: a.set_ylabel(f"{gt}\n주파수 (kHz)")
            if r == len(groups) - 1: a.set_xlabel("시간 (s)")
    fig.colorbar(im, ax=ax, label="dB (같은 색 범위)", shrink=0.6)
    fig.suptitle(title or f"STFT nperseg={nper} ({nper/FS*1000:.0f} ms, {FS/nper:.0f} Hz)")
    fig.savefig(os.path.join(FIG, fname), dpi=130, bbox_inches="tight"); plt.close(fig)

GROUPS = [("대표 정상", IDX_REF), ("오탐 경계 정상", IDX_FP), ("놓친·저에너지 고장", IDX_FN)]
grid(GROUPS, 256, "stft_256.png", title="STFT 짧은 창 256점 (10 ms): 충격 시각 선명")
grid(GROUPS, 1024, "stft_1024.png", title="STFT 긴 창 1024점 (41 ms): 주파수 위치 선명")

# %% 정상 중앙 스펙트로그램 대비 차이 (dB). 빨강 = 정상보다 센 곳, 파랑 = 약한 곳
def diff_grid(groups, nper, fname):
    S_nor = np.stack([spec(i, nper)[2] for i in nor])            # (정상 수, 주파수, 프레임)
    med = np.median(S_nor, axis=0)
    f, t, _ = spec(IDX_REF[0], nper)
    ncol = max(len(g[1]) for g in groups)
    fig, ax = plt.subplots(len(groups), ncol, figsize=(4.2 * ncol, 3.2 * len(groups)), squeeze=False, sharey=True)
    for r, (gt, idxs) in enumerate(groups):
        for c in range(ncol):
            a = ax[r, c]
            if c >= len(idxs): a.axis("off"); continue
            D = spec(idxs[c], nper)[2] - med
            m = f <= 12500
            im = a.pcolormesh(t, f[m] / 1000, D[m], vmin=-20, vmax=20, shading="auto", cmap="coolwarm")
            a.set_title(label(idxs[c]), fontsize=8)
            if c == 0: a.set_ylabel(f"{gt}\n주파수 (kHz)")
            if r == len(groups) - 1: a.set_xlabel("시간 (s)")
    fig.colorbar(im, ax=ax, label="파일 − 정상 중앙 스펙트로그램 (dB)", shrink=0.6)
    fig.suptitle(f"정상 중앙 대비 차이 (nperseg={nper}). 같은 프레임 위치끼리 뺀 것이라 충격 시각이 다르면 줄무늬가 어긋나 보임")
    fig.savefig(os.path.join(FIG, fname), dpi=130, bbox_inches="tight"); plt.close(fig)

diff_grid(GROUPS[1:], 512, "stft_512_diff.png")

# %% 시간 평균 프로파일: 주파수별로 프레임 평균/최대 (충격 지속성 vs 순간성)
f, t, _ = spec(0, 1024)
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
S_nor = np.stack([spec(i, 1024)[2] for i in nor])
lo, hi = np.percentile(S_nor.mean(2), [5, 95], axis=0)
mx_lo, mx_hi = np.percentile(S_nor.max(2), [5, 95], axis=0)
for a, fn, (l, h), tt in ((ax[0], lambda S: S.mean(1), (lo, hi), "프레임 평균 (지속 에너지)"),
                          (ax[1], lambda S: S.max(1), (mx_lo, mx_hi), "프레임 최댓값 (순간 충격)")):
    a.fill_between(f / 1000, l, h, color="gray", alpha=0.35, label="정상 5–95%")
    for i in IDX_FP: a.plot(f / 1000, fn(spec(i, 1024)[2]), color="tab:blue", lw=0.9, label=f"오탐 경계 {names[i]}")
    for i in IDX_FN: a.plot(f / 1000, fn(spec(i, 1024)[2]), color="tab:red", lw=0.9, label=f"고장 {names[i]}")
    a.set_title(tt); a.set_xlabel("주파수 (kHz)"); a.set_xlim(0, 12.5)
ax[0].set_ylabel("dB"); ax[1].legend(fontsize=7, ncol=2)
fig.savefig(os.path.join(FIG, "stft_profile.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
print("저장:", FIG)

# %% 4장을 한 장으로 합치기 (휴대폰에서 한 번에 보기용): figs/stft_all.png
import matplotlib.image as mpimg
parts = ["stft_256.png", "stft_1024.png", "stft_512_diff.png", "stft_profile.png"]
imgs = [mpimg.imread(os.path.join(FIG, p)) for p in parts]
# 세로로 쌓되 폭을 맞춘다 (가장 넓은 그림 기준, 모자란 폭은 흰색)
W = max(im.shape[1] for im in imgs)
rows = []
for im in imgs:
    pad = W - im.shape[1]
    rgb = im[..., :3]
    rows.append(np.pad(rgb, ((0, 0), (pad // 2, pad - pad // 2), (0, 0)), constant_values=1.0))
big = np.concatenate(rows, axis=0)
plt.imsave(os.path.join(FIG, "stft_all.png"), big)
print("합본:", os.path.join(FIG, "stft_all.png"), big.shape)
