import sys, numpy as np, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
from scipy.signal import butter, sosfiltfilt, hilbert, stft
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from joblib import Parallel, delayed
from src.exp import load_train
FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; nor = y == 0
C = "/home/user/PHM_Tension_Arm/outputs/cache/"
T = np.load(C + "pho001_train.npz", allow_pickle=True)["T"].item()
K4 = np.log10(np.maximum(np.load(C + "pho002_kgram.npz")["Km"].max(1), 1e-3))
hp = butter(4, 1000, btype="high", fs=FS, output="sos"); b8 = butter(4, [8000, 12400], btype="band", fs=FS, output="sos"); lp = butter(4, 1000, fs=FS, output="sos")
def env1k(xf):
    e = sosfiltfilt(lp, np.abs(hilbert(xf)))[::25]; return e - e.mean()          # 1 kHz 표본
def acf_rep(e):
    e = e / (np.sqrt((e ** 2).sum()) + 1e-12); a = np.correlate(e, e, "full")[len(e) - 1:]
    return float(a[50:1201].max())
def mod_peak(e):
    F = np.abs(np.fft.rfft(e * np.hanning(len(e)))); f = np.fft.rfftfreq(len(e), 1 / 1000)
    m = (f >= 0.8) & (f <= 20); r = (f > 0.5) & (f <= 100)
    return float(np.log10(F[m].max() / (np.median(F[r]) + 1e-12)))
def feats(x):
    x = x - x.mean(); h = sosfiltfilt(hp, x); h8 = sosfiltfilt(b8, x)
    e, e8 = env1k(h), env1k(h8)
    f, t, Z = stft(h, fs=FS, nperseg=512, noverlap=256); P = np.abs(Z) ** 2; et = P[f >= 1000].sum(0); p = et / et.sum()
    Ht = float(-(p * np.log(p + 1e-30)).sum() / np.log(len(p)))
    tau = 10; r = np.sqrt(x[:-tau] ** 2 + x[tau:] ** 2)
    return [acf_rep(e), mod_peak(e), mod_peak(e8), Ht, float(r.max() / np.percentile(r, 95))]
Fm = np.array(Parallel(n_jobs=-1)(delayed(feats)(x) for x in X))
cols = ["ACF 반복성", "변조 1k+", "변조 8-12.5k", "시간 엔트로피", "위상 max/95%"]
ex = {"H1충격수": T["H1_nimp"], "H3첨도": T["H3_kurt_1-2k"], "H5첨도": T["H5_kurt_4-8k"], "박자S": T["L1_S"], "K4": K4}
print("피처 | 단독 AUC(고장↑) | 기존 피처와 |ρ| 최대")
for j, c in enumerate(cols):
    v = Fm[:, j]; rr = {k: spearmanr(v, np.asarray(w, float))[0] for k, w in ex.items()}; km = max(rr, key=lambda k: abs(rr[k]))
    print(f"{c:12s} | {roc_auc_score(y, v):.2f} | {km} {rr[km]:+.2f}  (" + " ".join(f"{k} {w:+.2f}" for k, w in rr.items()) + ")")
print("\n정상 중 백분위 (%)  | " + " | ".join(cols))
for fn in ["abn_002", "abn_014", "abn_017", "abn_022", "abn_020", "nor_116", "nor_065", "nor_059", "nor_025"]:
    i = names.index(fn); print(f"{fn} | " + " | ".join(f"{(Fm[nor, j] < Fm[i, j]).mean() * 100:5.0f}" for j in range(len(cols))))
print("\n중앙값 정상 / 고장: " + " · ".join(f"{c} {np.median(Fm[nor, j]):.3f}/{np.median(Fm[~nor, j]):.3f}" for j, c in enumerate(cols)))
