"""R 역할 1단계: train 150 + val 40의 원재료를 한 번 계산해 캐시한다 (라벨은 쓰지 않는다).
 - 새 충격 정의(src/impacts.py) 충격 목록 · 울림 (f, zeta, fit)
 - 8.3k 스펙트럼 돌출 PR(f) (Welch 8192, 55 Hz 평활 - 1.2 kHz 이동 중앙값) : src/final.signal_tables와 같은 식
 - 예전 정의 값(비교용): K1-K3 (200 Hz · 8 sigma), P83/W83/P86, RES(예전 충격 B 정의의 8.3k 공진 fit)
출력: code_R/out/raw.pkl
"""
import os, sys, pickle
here = os.path.dirname(os.path.abspath(__file__))
root = here
while not os.path.isdir(os.path.join(root, "src")):
    root = os.path.dirname(root)
sys.path.insert(0, root)
os.chdir(root)

import numpy as np
from scipy.signal import welch
from scipy.ndimage import uniform_filter1d, median_filter

from src.exp import load_train, ROOT
from src.data import load_folder
from src.features2 import cached_feat2
from src.impacts import all_impacts
from src.features import extract_features_knock
from src.final import ring_table

X, y, names = load_train()
Xv, nv = load_folder(os.path.join(ROOT, "2_data", "TeamProject_val", "val"))
XA = np.vstack([X, Xv])
nm = [str(n) for n in names] + [str(n) for n in nv]
split = np.array(["train"] * len(y) + ["val"] * len(nv))
yy = np.concatenate([y, -np.ones(len(nv), int)])
print("files", XA.shape, "normal", (y == 0).sum(), "fault", (y == 1).sum(), "val", len(nv))

imps = all_impacts(XA)
n_by = {"normal": sum(len(i["t"]) for i, s, l in zip(imps, split, yy) if s == "train" and l == 0),
        "val": sum(len(i["t"]) for i, s in zip(imps, split) if s == "val"),
        "fault": sum(len(i["t"]) for i, s, l in zip(imps, split, yy) if s == "train" and l == 1)}
print("충격 수", n_by)
assert n_by == {"normal": 854, "val": 247, "fault": 402}, n_by
for im in imps:
    im.pop("hp")

Tt, _, _ = cached_feat2("train", X)
Tv, _, _ = cached_feat2("val", Xv)
import pandas as pd
T2 = pd.concat([Tt, Tv], ignore_index=True)

Xc = XA - XA.mean(1, keepdims=True)
f, P8 = welch(Xc, fs=25000, nperseg=8192)
L = np.log10(P8)
df = f[1] - f[0]
Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
PR = Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest")


def band_peak(lo, hi):
    b = (f >= lo) & (f < hi); ia = np.flatnonzero(b)
    P, W = PR[:, b].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, b].argmax()]; h = PR[i, k] / 2; l = r = k
        while l > 0 and PR[i, l - 1] >= h: l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h: r += 1
        W[i] = f[r] - f[l]
    return P, W


P83, W83 = band_peak(8200, 8450)
P86, W86 = band_peak(8450, 8750)

K = np.array([[v for v in extract_features_knock(x).values()] for x in XA])
ring_old, RES, NIMP_old = ring_table(XA)

pickle.dump(dict(names=nm, split=split, y=yy, imps=imps, T2=T2, f=f, PR=PR, P83=P83, W83=W83, P86=P86, W86=W86,
                 K=K, RES=RES, ring_old=ring_old, NIMP_old=NIMP_old),
            open(os.path.join(here, "out", "raw.pkl"), "wb"))
print("saved")
