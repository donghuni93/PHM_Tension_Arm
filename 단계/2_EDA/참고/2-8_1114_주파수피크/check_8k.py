# 2-8 보조: 8.0–8.6 kHz 정상 대비 초과(EXC 최대)가 정상과 고장에서 어떻게 분포하나. run.py와 같은 정의
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train
import numpy as np
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score
FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; nor = y == 0
Xc = X - X.mean(1, keepdims=True)
f, P = welch(Xc, fs=FS, nperseg=4096, detrend=False); L = np.log10(P + 1e-30)
m = f >= 200; fm = f[m]; Lm = L[:, m]
EXC = uniform_filter1d(Lm - np.median(Lm[nor], 0), 9, axis=1)
out = []
for lo, hi in [(8000, 8600), (9300, 9800), (10000, 10500), (6800, 7400)]:
    b = (fm >= lo) & (fm < hi); v = EXC[:, b].max(1)
    p95 = np.percentile(v[nor], 95)
    out.append(f"[{lo/1000:g}–{hi/1000:g} kHz 최대 초과] AUC {roc_auc_score(y, v):.2f} | 정상 p50 {np.median(v[nor]):.2f} p95 {p95:.2f} | 고장 p50 {np.median(v[~nor]):.2f} | 정상 95% 넘는 고장 {int((v[~nor] > p95).sum())}/30")
    top = np.argsort(v)[::-1][:15]
    out.append("   상위 15: " + ", ".join(f"{names[i]} {v[i]:.2f}" for i in top))
    for n in ["abn_002", "abn_029", "abn_003"]:
        i = names.index(n); out.append(f"   {n}: {v[i]:.2f} (정상 중 {(v[nor] < v[i]).mean()*100:.0f}%)")
open(os.path.join(HERE, "figs", "check_8k.txt"), "w", encoding="utf-8").write("\n".join(out))
