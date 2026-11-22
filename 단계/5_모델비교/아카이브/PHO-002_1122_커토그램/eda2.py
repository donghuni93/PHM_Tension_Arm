# %% [markdown]
# PHO-002 보조 EDA (11-22 사용자 요청): 일반 커토그램 vs 서브세그먼트 중앙값 커토그램(4등분 · 8등분)
# 1. 파일별 최적 대역(중심 · 폭 · K)이 정상 120 / 고장 30에서 갈리는가 (Mann–Whitney p, AUC, 8 kHz 이상 비율)
# 2. 그 대역 포락선으로 다시 구한 충격 수 · 박자 S: 경계 고장(014 등)의 탐지력이 오르는가, 단발 오경보 후보(116 · 065 · 059)에 역효과가 없는가
# run.py가 만든 캐시(outputs/cache/pho002_kgram.npz: 일반 K, 4등분 중앙값 K)를 읽고 8등분만 새로 계산한다
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from src.exp import load_train
from src.features2 import lock_S
from src.impacts import detect_impacts
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); CACHE = os.path.join(_d, "outputs", "cache"); FS = 25000
LOG = open(os.path.join(HERE, "log_eda2.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(v) for v in a); print(s); LOG.write(s + "\n"); LOG.flush()
X, y, names = load_train(); names = [str(n) for n in names]; n = len(y); nor = y == 0
BANDS = [(fc - bw / 2, fc + bw / 2, fc, bw) for bw in (4000, 2000, 1000, 500, 250)
         for fc in np.arange(1000 + bw / 2, 12400 - bw / 2 + 1, bw / 2)]
B = np.array(BANDS); SOS = [butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos") for lo, hi, _, _ in BANDS]
kurt_env = lambda c: float(np.mean(np.abs(c) ** 4) / np.mean(np.abs(c) ** 2) ** 2 - 2)

z = np.load(os.path.join(CACHE, "pho002_kgram.npz")); KG, KM4 = z["K"], z["Km"]
p8 = os.path.join(CACHE, "pho002_kgram8.npz")
if os.path.exists(p8):
    KM8 = np.load(p8)["K"]
else:
    def k8(x):
        x = x - x.mean(); return [np.median([kurt_env(q) for q in np.array_split(hilbert(sosfiltfilt(s, x)), 8)]) for s in SOS]
    KM8 = np.array(Parallel(n_jobs=-1)(delayed(k8)(x) for x in X)); np.savez(p8, K=KM8)
METH = {"일반": KG, "중앙값 4등분": KM4, "중앙값 8등분": KM8}

z1 = np.load(os.path.join(CACHE, "pho001_train.npz"), allow_pickle=True); T = z1["T"].item()
n_old, S_old = np.rint(np.expm1(np.asarray(T["H1_nimp"], float))), np.asarray(T["L1_S"], float)

def redo(i, j):
    x = X[i]; xb = sosfiltfilt(SOS[j], x - x.mean()); t = detect_impacts(x, _hp=xb)["t"]
    return len(t), lock_S(x, t)[0]

FOC_A, FOC_N = ["abn_014", "abn_022", "abn_002", "abn_017"], ["nor_116", "nor_065", "nor_059", "nor_025"]
res = {}
say("=== 1. 최적 대역 분포 (정상 120 / 고장 30) ===")
for m, K in METH.items():
    best = K.argmax(1); fc, bw, Kx = B[best, 2], B[best, 3], K.max(1)
    pf, pb, pk = (mannwhitneyu(v[nor], v[~nor]).pvalue for v in (fc, bw, Kx))
    say(f"[{m}] 중심 중앙 정상 {np.median(fc[nor]) / 1e3:.2f} · 고장 {np.median(fc[~nor]) / 1e3:.2f} kHz (p {pf:.3f}, AUC {roc_auc_score(y, fc):.2f}) | "
        f"8 kHz 이상 정상 {(fc[nor] >= 8000).mean():.2f} · 고장 {(fc[~nor] >= 8000).mean():.2f} | 폭 중앙 정상 {np.median(bw[nor]):.0f} · 고장 {np.median(bw[~nor]):.0f} Hz (p {pb:.3f}) | "
        f"K 중앙 정상 {np.median(Kx[nor]):.2f} · 고장 {np.median(Kx[~nor]):.2f} (p {pk:.1e}, AUC {roc_auc_score(y, Kx):.2f})")
    RD = np.array(Parallel(n_jobs=-1)(delayed(redo)(i, best[i]) for i in range(n)))
    res[m] = dict(best=best, fc=fc, bw=bw, K=Kx, n=RD[:, 0], S=RD[:, 1])

say("\n=== 2. 다시 구한 충격 수 · 박자 S ===")
say(f"기존(1 kHz 고역): S AUC {roc_auc_score(y, S_old):.2f} · 정상 S 최대 {S_old[nor].max():.3f} · 그 위 고장 {int((S_old[~nor] > S_old[nor].max()).sum())}개 · 충격 수 AUC {roc_auc_score(y, n_old):.2f}")
for m, r in res.items():
    say(f"[{m}] S AUC {roc_auc_score(y, r['S']):.2f} · 정상 S 최대 {r['S'][nor].max():.3f} · 그 위 고장 {int((r['S'][~nor] > r['S'][nor].max()).sum())}개 · 충격 수 AUC {roc_auc_score(y, r['n']):.2f}")
pct = lambda v, i: (v[nor] < v[i]).mean() * 100
say("\n파일 | 기존: 충격 · S (정상 중 %) | 일반 · 4등분 · 8등분: 대역 · 충격 · S (정상 중 %)")
for fn in FOC_A + FOC_N:
    i = names.index(fn)
    s = f"{fn} | {int(n_old[i]):2d} · {S_old[i]:.3f} ({pct(S_old, i):3.0f}%)"
    for m, r in res.items():
        s += f" | {r['fc'][i] / 1e3:5.2f}±{r['bw'][i] / 2 / 1e3:.2f}k · {int(r['n'][i]):2d} · {r['S'][i]:.3f} ({pct(r['S'], i):3.0f}%)"
    say(s)
say("\n대역이 바뀐 파일 수 (일반 → 4등분 · 8등분): " + " · ".join(f"{int((res['일반']['best'] != res[m]['best']).sum())}" for m in ("중앙값 4등분", "중앙값 8등분")))

# 그림: 방식별 최적 중심 분포 + S 비교
fig, ax = plt.subplots(2, 3, figsize=(17, 9))
for k, (m, r) in enumerate(res.items()):
    bins = np.arange(1, 12.6, .5)
    ax[0, k].hist(r["fc"][nor] / 1e3, bins, density=True, alpha=.55, color="#2a78d6", label="정상 120")
    ax[0, k].hist(r["fc"][~nor] / 1e3, bins, density=True, alpha=.55, color="#eb6834", label="고장 30")
    for fn in FOC_N + FOC_A:
        i = names.index(fn); ax[0, k].axvline(r["fc"][i] / 1e3, color="#d62728" if fn in FOC_N else "k", lw=.8, ls=":")
    ax[0, k].set_title(f"{m}: 최적 대역 중심 (점선 = 관심 파일)"); ax[0, k].set_xlabel("kHz"); ax[0, k].legend()
    ax[1, k].scatter(S_old[nor], r["S"][nor], s=16, color="#2a78d6", alpha=.6, label="정상")
    ax[1, k].scatter(S_old[~nor], r["S"][~nor], s=30, color="#eb6834", alpha=.8, label="고장")
    for fn in FOC_A + FOC_N:
        i = names.index(fn); ax[1, k].annotate(fn.replace("abn_", "A").replace("nor_", "N"), (S_old[i], r["S"][i]), xytext=(3, 3), textcoords="offset points", fontsize=8)
    ax[1, k].plot([0, 1], [0, 1], "k--", lw=.6); ax[1, k].set_xlabel("기존 S (1 kHz 고역)"); ax[1, k].set_ylabel(f"{m} 대역 S"); ax[1, k].grid(alpha=.3); ax[1, k].legend()
fig.tight_layout(); fig.savefig(os.path.join(FIG, "methods_compare.png"), dpi=110)
say("완료")
