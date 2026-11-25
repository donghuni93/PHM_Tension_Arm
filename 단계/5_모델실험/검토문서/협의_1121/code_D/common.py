"""D 진단 공용: 루트 찾기, 데이터·표 캐시, A(b) 최종 모델. 라벨은 보고용으로만 쓴다."""
import os, sys, pickle
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
ROOT = _d; sys.path.insert(0, ROOT)
HERE = os.path.abspath(os.path.dirname(__file__)); CACHE = os.path.join(HERE, "_cache"); os.makedirs(CACHE, exist_ok=True)
import numpy as np, pandas as pd, joblib
from src.exp import load_train
from src.data import load_folder
from src import final

def log_to(name):
    f = open(os.path.join(HERE, name), "w", encoding="utf-8")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s); f.write(s + "\n"); f.flush()
    return P

def load_all():
    X, y, names = load_train(); names = [str(n) for n in names]
    Xv, nv = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val")); nv = [str(n) for n in nv]
    return X, np.asarray(y), names, Xv, nv

def load_FM():
    return joblib.load(os.path.join(ROOT, "outputs", "final", "final_model.joblib"))

def tables(FM, X, Xv):
    p = os.path.join(CACHE, "tables.pkl")
    if os.path.exists(p): return pickle.load(open(p, "rb"))
    Ttr, cols = final.signal_tables(X, FM.cols_); Tva, _ = final.signal_tables(Xv, FM.cols_)
    pickle.dump((Ttr, Tva, cols), open(p, "wb")); return Ttr, Tva, cols
