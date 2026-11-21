"""EXP-046 신규 피처 (사전등록_046). 파일마다 독립, 라벨 없음. 새 충격 정의(src.impacts) 위에서 계산.

hf_rep_{lo}  울림 f >= lo 인 충격만 시간순으로 놓고 간격 > 50 ms 면 새 사건 → log1p(사건 수), 해당 충격 없으면 0. lo 격자 6개 (선택은 모델 fit 안)
n1_reb       src.normal_feats.file_work 의 N1_reb 그대로
ring_{lo}_{hi}_{z}  f in [lo, hi) 이고 ζ <= zmax 인 울림의 최대 fit (없으면 0). 키 12개 = 사전 고정 격자 (키·문턱 선택은 모델 fit 안)

열 순서(고정): hf 6 | n1 1 | ring 12  → N_EX = 19
"""
import os

import numpy as np
import pandas as pd

from src.exp import ROOT
from src.impacts import FS, detect_impacts, highpass, impact_rings_v2
from src.normal_feats import file_work

HF_GRID = (7750, 8000, 8250, 8500, 8750, 9000)
GAP_EV = 0.05
RING_BANDS = ((8000, 8450), (8100, 8450), (8250, 8750), (7750, 9000))
RING_Z = (0.01, 0.02, 0.04)
RING_KEYS = [(lo, hi, z) for (lo, hi) in RING_BANDS for z in RING_Z]
HF_COLS = [f"hf_rep_{lo}" for lo in HF_GRID]
RING_COLS = [f"ring_{lo}_{hi}_{z}" for (lo, hi, z) in RING_KEYS]
COLS = HF_COLS + ["n1_reb"] + RING_COLS
N_EX = len(COLS)                      # 19
I_HF, I_N1, I_RING = 0, len(HF_GRID), len(HF_GRID) + 1
_CACHE = os.path.join(ROOT, "outputs", "cache")


def hf_rep(t, f, lo):
    """충격 시각 t(s, 오름차순)·울림 f(Hz) → log1p(사건 수). f >= lo 인 충격만, 간격 > 50 ms 면 새 사건"""
    t = np.asarray(t); f = np.asarray(f)
    if len(t) == 0 or len(f) != len(t):
        return 0.0
    tt = np.sort(t[f >= lo])
    if len(tt) == 0:
        return 0.0
    return float(np.log1p(1 + int((np.diff(tt) > GAP_EV).sum())))


def ring_fit_table(rings_list):
    """파일별 rings (k,3)[f, ζ, fit] 리스트 → (n, 12) 모든 후보 키의 fit_max (RING_KEYS 순서)"""
    out = np.zeros((len(rings_list), len(RING_KEYS)))
    for i, r in enumerate(rings_list):
        if len(r) == 0:
            continue
        for j, (lo, hi, z) in enumerate(RING_KEYS):
            m = (r[:, 0] >= lo) & (r[:, 0] < hi) & (r[:, 1] <= z + 1e-12)
            if m.any():
                out[i, j] = r[m, 2].max()
    return out


def compute_feat3(X):
    """신호 배열 → (DataFrame[COLS + n_imp], 충격 수 합)"""
    ts, rings, n1, nimp = [], [], [], []
    for x in np.asarray(X, float):
        xh = highpass(x)
        imp = detect_impacts(x, _hp=xh)
        r = impact_rings_v2(x, imp=imp, _hp=xh)
        ts.append(imp["t"]); rings.append(r); nimp.append(len(imp["t"]))
        n1.append(file_work(x)[0]["N1_reb"])
    hf = np.array([[hf_rep(t, r[:, 0] if len(r) else np.zeros(0), lo) for lo in HF_GRID] for t, r in zip(ts, rings)])
    D = pd.DataFrame(np.column_stack([hf, n1, ring_fit_table(rings)]), columns=COLS)
    D["n_imp"] = nimp
    return D


def cached_feat3(tag, X):
    """outputs/cache/feat3_{tag}.csv. 피처 함수를 고치면 지우고 다시"""
    p = os.path.join(_CACHE, f"feat3_{tag}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    D = compute_feat3(X)
    D.to_csv(p, index=False)
    return D
