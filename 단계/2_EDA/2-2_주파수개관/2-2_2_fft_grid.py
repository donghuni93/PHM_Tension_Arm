# %% [markdown]
# # 2-2 FFT 전체 한눈에 (정상 120 · 고장 30)
# `2-2_1_fft_150.py`가 저장한 진폭 배열(figs/2-2_fft_amp.npz)을 읽는다.
# - 작은 칸 하나 = 파일 하나. 25,001빈을 50 Hz 칸으로 묶어 칸별 RMS 진폭 sqrt(mean A²)로 줄인다 (대역 에너지 보존)
# - y축(로그)은 150개 전부 같게 → 칸끼리 크기 비교 가능
# - 회색 선 = 정상 120개 중앙값 (같은 방식으로 묶음)

# %%
import sys
from pathlib import Path
ROOT_DIR = next(p for p in [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
                if (p / "src" / "exp.py").exists())
sys.path.insert(0, str(ROOT_DIR))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.ticker import FuncFormatter, NullFormatter
PLAIN = FuncFormatter(lambda v, _: f"{v:g}")   # 로그축 지수의 마이너스 글꼴 깨짐 방지

HERE = ROOT_DIR / "단계" / "2_EDA" / "2-2_주파수개관"
FIGS = HERE / "figs"
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#222222", "#999999"
BOLD = ["abn_002", "abn_022", "abn_029", "nor_097", "nor_101", "nor_120", "abn_016"]

d = np.load(FIGS / "2-2_fft_amp.npz")
f, A, y, names = d["f"], d["A"].astype(float), d["y"], list(d["names"])
K = 100                                              # 0.5 Hz × 100 = 50 Hz 칸
nb = (len(f) - 1) // K
B = np.sqrt((A[:, 1:nb * K + 1] ** 2).reshape(len(A), nb, K).mean(2))   # DC 제외
fb = f[1:nb * K + 1].reshape(nb, K).mean(1) / 1000                      # kHz
med = np.median(B[y == 0], 0)
ylo, yhi = B.min() * 0.8, B.max() * 1.2


def grid(sel, ncol, color, title, fname):
    nrow = int(np.ceil(len(sel) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(2.3 * ncol, 1.35 * nrow), sharex=True, sharey=True)
    for a in axs.flat:
        a.set_axis_off()
    for k, (a, i) in enumerate(zip(axs.flat, sel)):
        a.set_axis_on()
        a.plot(fb, med, color=MUTED, lw=.7)
        a.plot(fb, B[i], color=color, lw=.7)
        a.set_yscale("log"); a.set_ylim(ylo, yhi); a.set_xlim(0, 12.5)
        a.yaxis.set_major_formatter(PLAIN); a.yaxis.set_minor_formatter(NullFormatter())
        a.tick_params(labelsize=5, length=2)
        a.spines[["top", "right"]].set_visible(False)
        if k % ncol:
            a.tick_params(labelleft=False)
        n = names[i]
        a.set_title(n, fontsize=6.5, loc="left", pad=2, fontweight="bold" if n in BOLD else "normal")
    fig.suptitle(title, fontsize=11, x=.01, ha="left")
    fig.tight_layout(h_pad=.4, w_pad=.3, rect=(0, 0, 1, 1 - .5 / (1.35 * nrow)))   # 제목 자리 0.5인치
    fig.savefig(FIGS / fname, dpi=110)
    plt.close(fig)


sub = "가로 0–12.5 kHz · 세로 50 Hz 칸별 RMS 진폭 (g, 로그, 150개 공통) · 회색 = 정상 중앙값"
grid(np.where(y == 0)[0], 10, BLUE, f"정상 120개 FFT — {sub}", "2-2_fft_all_normal.png")
grid(np.where(y == 1)[0], 6, ORANGE, f"고장 30개 FFT — {sub}", "2-2_fft_all_abnormal.png")
