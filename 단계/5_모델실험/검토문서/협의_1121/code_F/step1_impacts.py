# F 피처 담당 step1: 새 충격 정의(src/impacts.py)로 train 150 + val 40의 충격 표(f, zeta, fit, 높이)를 만든다. 라벨은 split 열에 보고용으로만 붙임.
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from joblib import Parallel, delayed
from src.data import load_dataset, load_folder
from src.impacts import detect_impacts, impact_rings_v2, highpass
HERE = os.path.dirname(os.path.abspath(__file__))
X, y, names = load_dataset(os.path.join(_d, "data", "TeamProject_train_변경"))
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val"))
XA = np.vstack([X, Xv]); NA = [os.path.splitext(os.path.basename(str(s)))[0] for s in list(names) + list(nv)]
lab = list(y) + [-1] * len(Xv)
def work(x):
    xh = highpass(x); imp = detect_impacts(x, _hp=xh); r = impact_rings_v2(x, imp=imp, _hp=xh)
    n = len(r)
    return imp["t"][:n], imp["height_x"][:n], imp["height_g"][:n], r, len(imp["idx"])
if __name__ == "__main__":
    res = Parallel(n_jobs=4)(delayed(work)(x) for x in XA)
    rows = []; nimp = []
    for i, (t, hx, hg, r, n_all) in enumerate(res):
        nimp.append(n_all)
        for k in range(len(r)):
            rows.append((NA[i], lab[i], t[k], hx[k], hg[k], r[k, 0], r[k, 1], r[k, 2]))
    T = pd.DataFrame(rows, columns=["file", "y", "t", "height_x", "height_g", "f", "zeta", "fit"])
    T.to_csv(os.path.join(HERE, "impacts_new.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(dict(file=NA, y=lab, n_imp=nimp)).to_csv(os.path.join(HERE, "nimp.csv"), index=False, encoding="utf-8-sig")
    tot = dict(normal=int(sum(n for n, l in zip(nimp, lab) if l == 0)), val=int(sum(n for n, l in zip(nimp, lab) if l == -1)), fault=int(sum(n for n, l in zip(nimp, lab) if l == 1)))
    print(tot, len(T)); assert (tot["normal"], tot["val"], tot["fault"]) == (854, 247, 402), tot
