# F step5a: 8.3 kHz 좁은 순음 피처 (라벨 없이 정의). 8.0-8.6k Welch PSD 봉우리의 돌출(배경 대비 log10)과 반높이 폭(Hz).
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from scipy.signal import welch
from src.data import load_dataset, load_folder
HERE = os.path.dirname(os.path.abspath(__file__))
X, y, names = load_dataset(os.path.join(_d, "2_data", "TeamProject_train_변경"))
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val"))
XA = np.vstack([X, Xv]); NA = [os.path.splitext(os.path.basename(str(s)))[0] for s in list(names) + list(nv)]
rows = []
for nm, x in zip(NA, XA):
    x = x - x.mean()
    f, p = welch(x, fs=25000, nperseg=4096, noverlap=2048)
    ps = np.convolve(p, np.ones(3) / 3, mode="same")
    band = (f >= 8000) & (f <= 8600); bg_m = ((f >= 7300) & (f < 7900)) | ((f > 8800) & (f <= 9400))
    bg = np.median(ps[bg_m]) + 1e-18
    k0 = np.flatnonzero(band)[np.argmax(ps[band])]; pk = ps[k0]
    prom = np.log10(pk / bg)
    thr = np.sqrt(pk * bg)
    a = k0
    while a > 0 and ps[a - 1] > thr: a -= 1
    b = k0
    while b < len(ps) - 1 and ps[b + 1] > thr: b += 1
    w = (b - a + 1) * (f[1] - f[0])
    rows.append((nm, f[k0], prom, w if prom > 0.1 else 0.0))
T = pd.DataFrame(rows, columns=["file", "tone_f", "tone_prom", "tone_w"])
T.to_csv(os.path.join(HERE, "tone.csv"), index=False, encoding="utf-8-sig")
print(T.describe().to_string())
for n in ("abn_002","abn_006","abn_010","abn_029","abn_003","nor_019","nor_004","nor_092","nor_116"):
    print(T[T.file == n].to_string(header=False))
