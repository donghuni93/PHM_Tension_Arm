# 오경보 파일들의 AE z (기준 A = EXP-034 b의 AE × ET 평면, OOF)를 확인하는 보조 스크립트 (run.py는 고치지 않음)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from src.exp2util import *
from src.expA import A as ACls
ns = NS(); A = ACls(ns.X, ns.y, ns.names); Z, rule = A.oof_plane()
nor = ns.y == 0
print("정상 AE z 분위 50/90/95/99:", np.percentile(Z[nor, 0], [50, 90, 95, 99]).round(2).tolist(), "· 고장 AE z 중앙", round(float(np.median(Z[~nor, 0])), 2))
for f in ["nor_065", "nor_116", "nor_059", "nor_095", "nor_110", "nor_037", "nor_025", "abn_002", "abn_017", "abn_022"]:
    i = ns.names.index(f); r_ae = (Z[nor, 0] >= Z[i, 0]).mean()
    print(f"{f}: AE z {Z[i,0]:.2f} (정상 중 이보다 높은 비율 {r_ae:.2f}) · ET z {Z[i,1]:.2f}")
