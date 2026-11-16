# %% [markdown]
# # 2-2 파일별 FFT 150장 (2단계 EDA 조각 ② 준비)
# 원신호(평균만 제거) 2초 전체를 한 번에 FFT → 파일 하나 = 그림 하나 → `단계/2_EDA/2-2_1113_주파수개관/FFT_150/`
# - FFT 길이 50,000점 → 주파수 해상도 0.5 Hz, 0 ~ 12,500 Hz (나이퀴스트)
# - Hann 창을 곱해 누설을 줄이고, 진폭은 단측 진폭(g)으로 환산: 2|X(f)| / sum(w)
# - 파란 띠 = 정상 120개의 빈별 5–95% (25빈 = 12.5 Hz 이동 중앙값으로 매끄럽게, 읽기용)
# - 패널: 0–12.5 kHz 전체 / 0–300 Hz 저주파 / 5–12.5 kHz 고주파 (viz.psd_band와 같은 구간)
# - 제목: 진폭 상위 5개 피크 주파수 (서로 5 Hz 이상 떨어진 것, 2 Hz 미만 제외)

# %%
import sys
from pathlib import Path
ROOT_DIR = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
                if (p / "src" / "exp.py").exists())
sys.path.insert(0, str(ROOT_DIR))
from src.exp import *
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
from matplotlib.ticker import FuncFormatter
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")

HERE = ROOT_DIR / "단계" / "2_EDA" / "2-2_1113_주파수개관"
OUT = HERE / "FFT_150"; OUT.mkdir(exist_ok=True)
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#888888"
FS = 25000
RANGES = [(0, 12500, "0–12.5 kHz 전체"), (0, 300, "0–300 Hz 저주파"), (5000, 12500, "5–12.5 kHz 고주파")]

# %% FFT
X, y, names = load_train()
Xc = X - X.mean(1, keepdims=True)
N = X.shape[1]
w = np.hanning(N)
f = np.fft.rfftfreq(N, 1 / FS)
A = 2 * np.abs(np.fft.rfft(Xc * w, axis=1)) / w.sum()          # (150, 25001) 단측 진폭 g
A[:, 0] /= 2
logA = np.log10(A + 1e-12)
nor = logA[y == 0]
band_lo = 10 ** median_filter(np.percentile(nor, 5, 0), size=25)
band_hi = 10 ** median_filter(np.percentile(nor, 95, 0), size=25)
band_md = 10 ** median_filter(np.median(nor, 0), size=25)
np.savez_compressed(HERE / "figs" / "2-2_fft_amp.npz", f=f, A=A.astype(np.float32), y=y, names=np.array(names))


def top_peaks(a, k=5, fmin=2, sep_hz=5):
    order = np.argsort(a)[::-1]
    picked = []
    for j in order:
        if f[j] < fmin or any(abs(f[j] - f[p]) < sep_hz for p in picked):
            continue
        picked.append(j)
        if len(picked) == k:
            break
    return picked


# %% 파일별 그림
for i, n in enumerate(names):
    col = ORANGE if y[i] else BLUE
    lab = "고장" if y[i] else "정상"
    pk = top_peaks(A[i])
    fig, axs = plt.subplots(3, 1, figsize=(13, 9.5))
    for a, (lo, hi, title) in zip(axs, RANGES):
        m = (f >= lo) & (f <= hi)
        a.fill_between(f[m], band_lo[m], band_hi[m], color=BLUE, alpha=.15, lw=0, label="정상 5–95%")
        a.plot(f[m], band_md[m], color=BLUE, lw=.8, alpha=.7, label="정상 중앙값")
        a.plot(f[m], A[i, m], color=col, lw=.35 if hi - lo > 1000 else .8, label=n)
        for j in pk:
            if lo <= f[j] <= hi:
                a.plot(f[j], A[i, j], "v", ms=5, color=INK)
                a.annotate(f"{f[j]:.1f}", (f[j], A[i, j]), xytext=(3, 3), textcoords="offset points",
                           fontsize=7, color=INK)
        a.set_yscale("log"); a.set_xlim(lo, hi)
        a.yaxis.set_major_formatter(PLAIN)   # 로그축 지수의 마이너스 글꼴 깨짐 방지
        a.set_ylabel("진폭 (g)", fontsize=8); a.set_title(title, fontsize=9, loc="left")
        a.grid(alpha=.2); a.spines[["top", "right"]].set_visible(False); a.tick_params(labelsize=7)
    axs[0].legend(fontsize=7, frameon=False, ncol=3, loc="upper right")
    axs[-1].set_xlabel("Hz", fontsize=8)
    peaks_txt = " · ".join(f"{f[j]:.1f} Hz ({A[i, j]:.3g} g)" for j in pk)
    fig.suptitle(f"{n} ({lab})  FFT 진폭 스펙트럼 (평균 제거, Hann, 0.5 Hz 해상도)\n상위 피크: {peaks_txt}",
                 fontsize=10, x=.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / f"{n}.png", dpi=100)
    plt.close(fig)

with open(HERE / "figs" / "2-2_fft_log.txt", "w", encoding="utf-8") as fh:
    fh.write(f"{len(list(OUT.glob('*.png')))} 장 저장: {OUT}\n")
