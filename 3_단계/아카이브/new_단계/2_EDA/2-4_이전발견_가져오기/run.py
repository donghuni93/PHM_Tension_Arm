# %% 2-4 이전 발견 가져오기: ① 울림 모드 경계(2-14) ② 8.3k 규칙(2-2)을 new_단계의 Train 150 + val 40(라벨 없이)에서 다시 확인
# 검출기 · 규칙 코드는 src의 것을 그대로 쓴다. 규칙 문턱은 바꾸지 않는다. val은 라벨 없이 위치 · 켜짐만 본다.
import glob, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.stats import mannwhitneyu

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.impacts import detect_impacts, highpass, _atoms50, FS
from src.final import _SEG, _peak83, kurt83, ring_table, P83_MIN, W83_MIN, K83_MIN, RES_FIT_MIN
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

# %% 로드 (train 150 + val 40)
def load(pattern):
    fl = sorted(glob.glob(pattern))
    return [os.path.basename(f)[:-4] for f in fl], [np.loadtxt(f) for f in fl]
dn, dx = load(os.path.join(ROOT, "2_data", "TeamProject_train_변경", "normal", "*.csv"))
da, dax = load(os.path.join(ROOT, "2_data", "TeamProject_train_변경", "abnormal", "*.csv"))
dv, dvx = load(os.path.join(ROOT, "2_data", "TeamProject_val", "val", "*.csv"))
names = dn + da + dv
grp = np.array(["normal"] * len(dn) + ["abnormal"] * len(da) + ["val"] * len(dv))
X = np.array(dx + dax + dvx)

# %% A. 울림 주파수 지도 (라벨 없음): 모든 검출의 울림 주파수, 파일마다 같은 무게
A50, F50 = _atoms50()
det_f = []      # 파일별 울림 주파수 배열
for x in X:
    xh = highpass(x)
    idx = detect_impacts(x, _hp=xh)["idx"]
    if len(idx) == 0:
        det_f.append(np.array([])); continue
    S = np.stack([xh[p:p + _SEG] for p in idx]); S /= (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    det_f.append(F50[np.linalg.norm(np.einsum("ks,ast->kat", S, A50), axis=2).argmax(1)])
edges = np.arange(1000, 12101, 100); ctr = (edges[:-1] + edges[1:]) / 2
def density(mask):
    h = np.zeros(len(ctr)); n = 0
    for f, m in zip(det_f, mask):
        if m and len(f):
            h += np.histogram(f, bins=edges)[0] / len(f); n += 1
    return h / max(n, 1)
allm = np.ones(len(X), bool)
D_all = density(allm); D_n = density(grp == "normal"); D_a = density(grp == "abnormal"); D_v = density(grp == "val")
sm = gaussian_filter1d(D_all, 2)      # 200 Hz
valleys, pr = find_peaks(-sm, prominence=0.02 * sm.max())
L = ["A) 라벨 없는 울림 주파수 지도: 190개 파일(학습 150 + val 40), 모든 검출, 파일당 같은 무게, 100 Hz 칸, 가우시안 200 Hz",
     "골짜기(국소 최소) 위치 kHz: " + ", ".join(f"{ctr[v]/1000:.2f}(돌출 {p/sm.max():.2f})" for v, p in zip(valleys, pr['prominences'])),
     "(참고) 2-14가 라벨 없이 찾은 경계: 1.6 · 4.5 · 5.5 · 8.0 kHz (6.3 kHz 골짜기는 합침)"]
peaks_, pp = find_peaks(sm, prominence=0.02 * sm.max())
L.append("봉우리 위치 kHz: " + ", ".join(f"{ctr[v]/1000:.2f}" for v in peaks_))

# 파일별 대역 충격 수 AUC (train 150, 라벨 사용)
tr = grp != "val"; yv = (grp == "abnormal")
bands = {"1.0-1.6k": (1000, 1600), "1.6-4.5k": (1600, 4500), "4.5-5.5k": (4500, 5500), "5.5-8k": (5500, 8000), "8-11k": (8000, 11000)}
L.append("\n파일당 충격 수(모든 검출) AUC (고장 > 정상, train 150) · 정상/고장 중앙값")
cnt = {}
for k, (a, b) in bands.items():
    c = np.array([((f >= a) & (f < b)).sum() for f in det_f]); cnt[k] = c
    u = mannwhitneyu(c[yv & tr], c[~yv & tr]).statistic / ((yv & tr).sum() * (~yv & tr).sum())
    L.append(f"{k:9s} AUC {u:.2f} | 정상 중앙 {np.median(c[grp=='normal']):.0f} · 고장 중앙 {np.median(c[grp=='abnormal']):.0f} · val 중앙 {np.median(c[grp=='val']):.0f}")

# %% B. 8.3k 규칙 (src.final 정의 그대로)
Xc = X - X.mean(1, keepdims=True)
P83, W83 = _peak83(Xc); K83 = kurt83(Xc)
_, RES, _ = ring_table(X)
spec_noK = (P83 >= P83_MIN) & (W83 >= W83_MIN)
spec = spec_noK & (K83 > K83_MIN)
res = RES >= RES_FIT_MIN
T = pd.DataFrame(dict(file=names, group=grp, P83=P83, W83=W83, K83=K83, RES=RES, spec_noK=spec_noK, spec=spec, res=res, rule=(spec | res)))
T.to_csv(os.path.join(HERE, "figs", "rule83_files.csv"), index=False)
L.append(f"\nB) 8.3k 규칙 재현 (P83 >= {P83_MIN}, W83 >= {W83_MIN} Hz, K83 > {K83_MIN}; 공진 규칙 RES >= {RES_FIT_MIN})")
for g in ("normal", "abnormal", "val"):
    t = T[T.group == g]
    L.append(f"{g:9s} n={len(t):3d} | 스펙트럼 규칙(K 없이) {t.spec_noK.sum():2d} · (K 포함) {t.spec.sum():2d} | 공진 규칙 {t.res.sum():2d} | OR {t.rule.sum():2d}")
L.append("켜진 파일 (K 포함 스펙트럼 규칙): " + ", ".join(T[T.spec].file))
L.append("켜진 파일 (공진 규칙): " + ", ".join(T[T.res].file))
L.append("K 없이만 켜진 파일: " + ", ".join(T[T.spec_noK & ~T.spec].file) + "  (K83 " + ", ".join(f"{k:.1f}" for k in T[T.spec_noK & ~T.spec].K83) + ")")
L.append("\n경계 근처 (규칙 문턱의 0.8–1.0배): " + "; ".join(f"{r.file} P{r.P83:.2f}/W{r.W83:.0f}/K{r.K83:.1f}" for r in T.itertuples()
         if (not r.spec) and r.P83 >= 0.8 * P83_MIN and r.W83 >= 0.8 * W83_MIN and r.K83 > K83_MIN))
open(os.path.join(HERE, "figs", "재확인_수치.txt"), "w", encoding="utf-8").write("\n".join(L))

# %% 그림
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
C_N, C_A, C_V = "#2a78d6", "#eb6834", "#6250d6"
fig, ax = plt.subplots(1, 3, figsize=(17, 5.2), facecolor=SURF)
for a in ax:
    a.set_facecolor(SURF); a.grid(True, color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    for s in ("left", "bottom"): a.spines[s].set_color(GRID)
    a.tick_params(colors=INK2, labelsize=10)
a = ax[0]
a.plot(ctr / 1000, sm, color=INK, lw=1.8)
for v in valleys: a.axvline(ctr[v] / 1000, color=INK2, lw=0.9, ls=(0, (4, 3)))
for k in (1.6, 4.5, 5.5, 8.0): a.axvline(k, color="#e34948", lw=1.0, alpha=.6)
a.set_title("(a) 라벨 없는 울림 주파수 지도 (190개)", loc="left", color=INK, fontsize=12)
a.set_xlabel("울림 주파수 (kHz)", color=INK2); a.set_ylabel("파일당 같은 무게의 비율", color=INK2)
a.annotate("점선 = 이번에 찾은 골짜기\n빨강 = 2-14 경계", xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top", color=INK2, fontsize=9)
a = ax[1]
for D_, c, lab in ((D_n, C_N, "정상 120"), (D_a, C_A, "고장 30"), (D_v, C_V, "val 40 (라벨 없음)")):
    a.plot(ctr / 1000, gaussian_filter1d(D_, 2), color=c, lw=1.8, label=lab)
a.set_title("(b) 집단별 울림 주파수", loc="left", color=INK, fontsize=12); a.set_xlabel("울림 주파수 (kHz)", color=INK2)
a.legend(frameon=False, labelcolor=INK2, fontsize=9)
a = ax[2]
for g, c, m in (("normal", C_N, "o"), ("abnormal", C_A, "o"), ("val", C_V, "D")):
    t = T[T.group == g]; a.scatter(t.P83, t.W83, s=26 if g != "val" else 30, color=c, alpha=.7, marker=m, lw=0, label={"normal": "정상", "abnormal": "고장", "val": "val"}[g])
a.axvline(P83_MIN, color=INK2, lw=0.9, ls=(0, (4, 3))); a.axhline(W83_MIN, color=INK2, lw=0.9, ls=(0, (4, 3)))
t = T[T.spec_noK & ~T.spec]
for r in t.itertuples(): a.annotate(r.file + f" K{r.K83:.1f}", (r.P83, r.W83), xytext=(6, 4), textcoords="offset points", fontsize=8.5, color=INK)
a.set_xlim(0, 1.6); a.set_ylim(0, 600)
a.set_title("(c) 8.3k 규칙 평면 (오른쪽 위 = 규칙 영역)", loc="left", color=INK, fontsize=12)
a.set_xlabel("P83 돌출 (dec)", color=INK2); a.set_ylabel("W83 폭 (Hz)", color=INK2); a.legend(frameon=False, labelcolor=INK2, fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(HERE, "figs", "재확인.png"), dpi=140, facecolor=SURF)
print("\n".join(L))
