# %% 2-8 val 배경 이동의 원인: "하중(노면 입력)이 커져서인가?"를 데이터로 시험
# 라벨은 집단 구분(정상/고장)에만 쓴다. val은 라벨 없이 본다. 모델 · 문턱은 바꾸지 않는다.
import glob, os
import numpy as np, pandas as pd
from scipy.signal import welch, butter, sosfiltfilt, find_peaks
from scipy.stats import spearmanr, mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "data")): ROOT = os.path.dirname(ROOT)
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True); FS = 25000
fl, grp = [], []
for g, p in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
    s = sorted(glob.glob(os.path.join(ROOT, "data", p, "*.csv"))); fl += s; grp += [g] * len(s)
names = [os.path.basename(f)[:-4] for f in fl]; grp = np.array(grp)
X = np.stack([np.loadtxt(f) for f in fl]); dc = X.mean(1); Xc = X - dc[:, None]
N, A, V = grp == "normal", grp == "abnormal", grp == "val"
db = lambda v: 10 * np.log10(v)
L = []

# %% A. 하중(저주파 입력) 크기
lp20 = sosfiltfilt(butter(4, 20, "lowpass", fs=FS, output="sos"), Xc, axis=1).std(1)
b200 = sosfiltfilt(butter(4, [20, 200], "bandpass", fs=FS, output="sos"), Xc, axis=1).std(1)
L.append("A. 하중 크기 (중앙값, g) | val − 정상 (dB, 파워)")
for nm, v in (("0–20 Hz RMS (출렁임 · 하중)", lp20), ("20–200 Hz RMS", b200)):
    L.append(f"{nm:28s} 정상 {np.median(v[N]):.4f} 고장 {np.median(v[A]):.4f} val {np.median(v[V]):.4f} | val−정상 {db(np.median(v[V]**2)/np.median(v[N]**2)):+.1f} dB, MWU p {mannwhitneyu(v[V], v[N]).pvalue:.3f}")
L.append(f"평균(DC, g) 중앙값: 정상 {np.median(dc[N]):+.3f} 고장 {np.median(dc[A]):+.3f} val {np.median(dc[V]):+.3f} | −0.5 g 아래 파일 수: 정상 {(dc[N]<-.5).sum()} 고장 {(dc[A]<-.5).sum()} val {(dc[V]<-.5).sum()}")

# %% B. 배경 PSD (구간 중앙값 Welch: 충격이 낀 구간의 영향을 줄임)
f, P = welch(Xc, fs=FS, nperseg=2048, average="median", detrend=False)
LPd = db(P)
med = lambda m: np.median(LPd[m], 0)
diff_vn, diff_an = med(V) - med(N), med(A) - med(N)
L.append("\nB. 배경 PSD 차이 (파일 중앙값 PSD 기준, dB) — 대역별")
edges = [200, 500, 1000, 1600, 2500, 3000, 4000, 5000, 5500, 6500, 7500, 8000, 9000, 10000, 11000]
for a, b in zip(edges[:-1], edges[1:]):
    m = (f >= a) & (f < b)
    L.append(f"{a/1000:5.1f}–{b/1000:<5.1f} kHz: val−정상 {diff_vn[m].mean():+5.1f} | 고장−정상 {diff_an[m].mean():+5.1f}")

# %% C. 하중이 배경을 끌어올리는가 (Train 정상 안에서의 관계 → val 예측)
def band_db(a, b): return db(P[:, (f >= a) & (f < b)].mean(1))
L.append("\nC. Train 정상 안에서 '하중(0–20 Hz RMS, dB) → 배경 대역 dB' 기울기와 val 예측")
x = db(lp20 ** 2)
for a, b in ((1000, 2500), (3000, 5000), (5500, 8000), (8000, 10000)):
    yb = band_db(a, b); r = spearmanr(x[N], yb[N])[0]; k, c = np.polyfit(x[N], yb[N], 1)
    pred = k * (np.median(x[V]) - np.median(x[N])); obs = np.median(yb[V]) - np.median(yb[N])
    # 하중을 맞춘 비교: val 각 파일을 같은 하중의 정상 예측치와 비교한 잔차
    resid = yb[V] - (k * x[V] + c)
    L.append(f"{a/1000:.1f}–{b/1000:.1f} kHz: ρ(하중, 배경) 정상 {r:+.2f}, 기울기 {k:.2f} dB/dB | val 하중 차이로 예측되는 이동 {pred:+.1f} dB vs 실제 {obs:+.1f} dB | 하중 보정 후 val 잔차 중앙 {np.median(resid):+.1f} dB")
x2 = db(b200 ** 2)
yb = band_db(3000, 5000); k, c = np.polyfit(x2[N], yb[N], 1)
L.append(f"(20–200 Hz RMS를 하중으로 쓸 때) 3–5 kHz: ρ {spearmanr(x2[N], yb[N])[0]:+.2f}, 예측 {k*(np.median(x2[V])-np.median(x2[N])):+.1f} dB vs 실제 {np.median(yb[V])-np.median(yb[N]):+.1f} dB")

# %% D. 모드 위치 (선형계에서는 하중이 커져도 고유진동수는 그대로)
def peak_in(a, b):
    m = (f >= a) & (f < b); return f[m][np.argmax(LPd[:, m], 1)]
L.append("\nD. 파일별 배경 PSD 최대 봉우리 위치 (중앙값 [IQR], Hz)")
for a, b, nm in ((5500, 8000, "본체 모드 5.5–8 kHz"), (2500, 4000, "2.5–4 kHz"), (1000, 1600, "1.0–1.6 kHz")):
    pk = peak_in(a, b)
    L.append(f"{nm:18s} 정상 {np.median(pk[N]):.0f} [{np.percentile(pk[N],25):.0f}, {np.percentile(pk[N],75):.0f}] | 고장 {np.median(pk[A]):.0f} | val {np.median(pk[V]):.0f} [{np.percentile(pk[V],25):.0f}, {np.percentile(pk[V],75):.0f}] | MWU p(val vs 정상) {mannwhitneyu(pk[V], pk[N]).pvalue:.3f}")
# 중앙 PSD에서의 봉우리 목록
def peaks(curve):
    m = (f >= 500) & (f < 11000); pk, pr = find_peaks(curve[m], prominence=1.5)
    return ", ".join(f"{f[m][i]/1000:.2f}" for i in pk)
L.append("중앙 PSD 봉우리(kHz, 돌출 ≥ 1.5 dB): 정상 " + peaks(med(N)) + " | val " + peaks(med(V)))

# %% E. 하중이 비슷한 파일끼리 비교 (하중 맞춤)
L.append("\nE. 하중(0–20 Hz RMS)이 비슷한 정상과 val만 비교 (val 하중 범위 안의 정상)")
lo_, hi_ = np.percentile(lp20[V], [25, 75]); m_n = N & (lp20 >= lo_) & (lp20 <= hi_); m_v = V & (lp20 >= lo_) & (lp20 <= hi_)
yb = band_db(3000, 5000)
L.append(f"하중 {lo_:.4f}–{hi_:.4f} g: 정상 {m_n.sum()}개, val {m_v.sum()}개 | 3–5 kHz 배경 val−정상 {np.median(yb[m_v]) - np.median(yb[m_n]):+.1f} dB")
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L))
pd.DataFrame(dict(file=names, group=grp, dc=dc, load20=lp20, b20_200=b200, bg3_5=band_db(3000, 5000), body_peak=peak_in(5500, 8000))).to_csv(os.path.join(FIG, "files.csv"), index=False)

# %% 그림: (a) 중앙 PSD 세 집단 (b) 하중 vs 3–5 kHz 배경
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C = {"normal": "#2a78d6", "abnormal": "#eb6834", "val": "#1baf7a"}; LAB = {"normal": "Train 정상 120", "abnormal": "Train 고장 30", "val": "val 40"}
fig, ax = plt.subplots(1, 2, figsize=(14, 5), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=.8); a.set_axisbelow(True); a.tick_params(colors=INK2)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
m = (f >= 200) & (f <= 11000)
for g in ("normal", "abnormal", "val"):
    ax[0].plot(f[m] / 1000, med(grp == g)[m], color=C[g], lw=2, label=LAB[g])
ax[0].set_xlabel("주파수 (kHz)", color=INK2); ax[0].set_ylabel("배경 PSD 중앙값 (dB, g²/Hz)", color=INK2)
ax[0].set_title("(a) 집단별 배경 PSD (구간 중앙값 Welch)", loc="left", color=INK); ax[0].legend(frameon=False)
yb = band_db(3000, 5000)
for g in ("normal", "abnormal", "val"):
    mm = grp == g; ax[1].scatter(db(lp20[mm] ** 2), yb[mm], s=18, color=C[g], alpha=.7, lw=0, label=LAB[g])
k, c = np.polyfit(db(lp20[N] ** 2), yb[N], 1); xx = np.linspace(db(lp20 ** 2).min(), db(lp20 ** 2).max(), 10)
ax[1].plot(xx, k * xx + c, color=C["normal"], lw=1.5, ls="--", label="정상의 하중–배경 관계(직선)")
ax[1].set_xlabel("하중 = 0–20 Hz 파워 (dB)", color=INK2); ax[1].set_ylabel("3–5 kHz 배경 (dB)", color=INK2)
ax[1].set_title("(b) 하중이 크면 배경도 큰가", loc="left", color=INK); ax[1].legend(frameon=False, fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "하중_배경.png"), dpi=140, facecolor=SURF); print("done")
