# %% 2-1 전체 PSD: Train 150개(정상 120 · 고장 30), 평균 · 중앙값 Welch PSD
# 라벨은 색칠에만 쓴다. 대역 · 문턱을 고르지 않는 시각화 전용 스크립트.
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "2_data")):
    ROOT = os.path.dirname(ROOT)
D = os.path.join(ROOT, "2_data", "TeamProject_train_변경")
FS = 25000
NPERSEG = 5000          # 2 s 파일 안에서 0.2 s 창, 해상도 5 Hz
SMOOTH = 10             # 그림용 이동평균 10칸 = 50 Hz (선형 파워 도메인)

# %% 로드 + Welch
def load(group):
    fl = sorted(glob.glob(os.path.join(D, group, "*.csv")))
    P = []
    for f in fl:
        x = np.loadtxt(f); x = x - x.mean()          # 평균 제거만
        fr, p = welch(x, fs=FS, window="hann", nperseg=NPERSEG, noverlap=NPERSEG // 2)
        P.append(p)
    return fr, np.array(P)

fr, Pn = load("normal")
_, Pa = load("abnormal")
print("normal", Pn.shape, "abnormal", Pa.shape)

def smooth(v):
    # 가장자리는 창이 잘린 칸 수로 나눠 0 패딩으로 떨어지는 것을 막는다
    k = np.ones(SMOOTH)
    return np.convolve(v, k, mode="same") / np.convolve(np.ones_like(v), k, mode="same")

db = lambda v: 10 * np.log10(v)
f_k = fr / 1000

# 곡선: 평균(선형 평균 후 dB), 중앙값, 25–75 %
mean_n, mean_a = smooth(Pn.mean(0)), smooth(Pa.mean(0))
med_n, med_a = smooth(np.median(Pn, 0)), smooth(np.median(Pa, 0))
q_n = [smooth(np.percentile(Pn, q, 0)) for q in (25, 75)]
q_a = [smooth(np.percentile(Pa, q, 0)) for q in (25, 75)]
# 라벨 없는 바닥 참고선: 150개 전체의 하위 5 % (파일 사이)
allP = np.vstack([Pn, Pa]); floor = smooth(np.percentile(allP, 5, 0))
# 고장 파일 중 정상 중앙값보다 높은 비율 (500 Hz 구간 평균 파워 기준)
edges = np.arange(0, 12501, 500)
def binmean(P):
    return np.array([P[:, (fr >= a) & (fr < b + (1 if b == 12500 else 0))].mean(1)
                     for a, b in zip(edges[:-1], edges[1:])]).T
Bn, Ba = binmean(Pn), binmean(Pa)
frac = (Ba > np.median(Bn, 0)).mean(0)
centers = (edges[:-1] + edges[1:]) / 2000

# %% 그림
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_N, C_A = "#2a78d6", "#eb6834"      # 정상 파랑, 고장 주황 (카테고리 1·2)

fig, ax = plt.subplots(3, 1, figsize=(11, 10), sharex=True,
                       gridspec_kw={"height_ratios": [3, 3, 1.6]}, facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF)
    a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=10)

def label(a, x, y, txt, c, dy=0):
    a.annotate(txt, (x, y), xytext=(6, dy), textcoords="offset points",
               color=INK, fontsize=10, va="center",
               bbox=dict(boxstyle="square,pad=0.1", fc=SURF, ec="none"))

# (a) 평균
a = ax[0]
a.plot(f_k, db(mean_n), color=C_N, lw=1.6)
a.plot(f_k, db(mean_a), color=C_A, lw=1.6)
a.plot(f_k, db(floor), color=INK2, lw=1.0, ls=(0, (4, 3)))
a.set_title("(a) 평균 PSD — 극단 파일 몇 개에 끌려 올라갈 수 있다", loc="left", color=INK, fontsize=12)
label(a, 12.5, db(mean_n[-8]), "정상 120", C_N, 0); label(a, 12.5, db(mean_a[-8]), "고장 30", C_A, 0)
label(a, 9.0, db(floor[1800]) - 3, "하위 5 % 바닥(150개)", INK2, 0)

# (b) 중앙값 + 25–75 %
a = ax[1]
a.fill_between(f_k, db(q_n[0]), db(q_n[1]), color=C_N, alpha=0.18, lw=0)
a.fill_between(f_k, db(q_a[0]), db(q_a[1]), color=C_A, alpha=0.18, lw=0)
a.plot(f_k, db(med_n), color=C_N, lw=1.6)
a.plot(f_k, db(med_a), color=C_A, lw=1.6)
a.plot(f_k, db(floor), color=INK2, lw=1.0, ls=(0, (4, 3)))
a.set_title("(b) 중앙값 PSD와 파일 사이 25–75 % 띠", loc="left", color=INK, fontsize=12)
label(a, 12.5, db(med_n[-8]), "정상 120", C_N, 0); label(a, 12.5, db(med_a[-8]), "고장 30", C_A, 0)

# (c) 고장 파일 중 정상 중앙값 초과 비율
a = ax[2]
a.bar(centers, frac, width=0.42, color=C_A, edgecolor=SURF, lw=1.5)
a.axhline(0.5, color=INK2, lw=0.9, ls=(0, (4, 3)))
a.set_ylim(0, 1.05); a.set_yticks([0, 0.5, 1])
a.set_title("(c) 고장 파일 중 '정상 중앙값'보다 높은 비율 (500 Hz 구간)", loc="left", color=INK, fontsize=12)
a.set_xlabel("주파수 (kHz)", color=INK2)
a.annotate("0.5 = 정상과 같은 수준", (12.5, 0.5), xytext=(6, 0), textcoords="offset points",
           color=INK2, fontsize=9, va="center", annotation_clip=False)

ax[0].set_ylabel("PSD (dB re 1 g²/Hz)", color=INK2)
ax[1].set_ylabel("PSD (dB re 1 g²/Hz)", color=INK2)
ax[2].set_ylabel("비율", color=INK2)
ax[2].set_xlim(0, 12.5); ax[2].set_xticks(np.arange(0, 12.6, 1))
fig.text(0.01, 0.005, "Train 150개(정상 120 · 고장 30), 평균 제거 후 Welch(Hann, 0.2 s 창, 50 % 겹침, 해상도 5 Hz), 그림용 50 Hz 이동평균. "
         "점선 = 150개 하위 5 % 바닥.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 1))
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
fig.savefig(os.path.join(HERE, "figs", "psd_mean_median.png"), dpi=140, facecolor=SURF)

# %% 수치 (구간별)
lines = ["bin kHz | mean dB n/a | median dB n/a | frac abn > nor median"]
for i, c in enumerate(centers):
    m = (fr >= edges[i]) & (fr < edges[i + 1] + (1 if edges[i + 1] == 12500 else 0))
    lines.append(f"{edges[i]/1000:4.1f}-{edges[i+1]/1000:4.1f} | "
                 f"{db(Pn[:, m].mean()):6.1f} {db(Pa[:, m].mean()):6.1f} | "
                 f"{db(np.median(Bn[:, i])):6.1f} {db(np.median(Ba[:, i])):6.1f} | {frac[i]:.2f}")
open(os.path.join(HERE, "figs", "psd_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))
