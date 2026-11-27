# %% [markdown]
# EXP-053 (11-22) AE 입력 정규화 사전 점검 — "크기에 둔감한 AE"가 val 이동은 줄이고 train 구분력은 남기는가 (중첩 CV 전 거르기, H3)
# 후보 5개: J AE · 모양(구간 평균 제거, PHO-008) · 자기기준(파일 중앙 스펙트럼 제거, PHO-009) · 대역 수준 제거(5대역 평균, 새) · 포락 제거(1 kHz 이동 중앙값, 새)
# 잣대: (a) train OOF AUC(5겹, 정상만 학습) (b) val 중 train 정상 OOF 95 % 위 비율 · 중앙 z (c) 6-7 확실 정상 val 23개 (d) val vs train 정상 AUC
# 실행: conda run -n qe-phm python run.py → result.json
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore"); import json
import numpy as np
from scipy.ndimage import median_filter
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from src.experts import LP as LPt, segment_logpsd
from src.exp2util import NS
ns = NS(); y = ns.y; nor = y == 0; _, LPv = segment_logpsd(ns.Xv)
fb = np.arange(1025) * 25000 / 2048
EDGES = [0, 1000, 2000, 4000, 8000, 12501]; bid = np.digitize(fb, EDGES) - 1
def band_norm(L):
    out = L.copy()
    for b in range(len(EDGES) - 1):
        m = bid == b; out[..., m] -= L[..., m].mean(-1, keepdims=True)
    return out
V = {
    "AE (J)": lambda L: L,
    "모양 (구간 평균 제거)": lambda L: L - L.mean(-1, keepdims=True),
    "자기기준 (파일 중앙 스펙트럼 제거)": lambda L: L - np.median(L, axis=1, keepdims=True),
    "대역 수준 제거 (5대역 평균)": band_norm,
    "포락 제거 (1 kHz 이동 중앙값)": lambda L: L - median_filter(L, size=(1, 1, 81), mode="nearest"),
}
def fit(S):
    S = S.reshape(-1, S.shape[-1]); mu, sd = S.mean(0), S.std(0) + 1e-8
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3, max_iter=60, alpha=0.0, random_state=0).fit((S - mu) / sd, (S - mu) / sd)
    return mu, sd, m
def score(ae, L):
    mu, sd, m = ae; Z = (L - mu) / sd; R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape); return np.log10(((R - Z) ** 2).mean((1, 2)))
SURE_N = {9, 32, 33, 2, 22, 6, 38, 40, 8, 23, 34, 29, 28, 11, 26, 30, 15, 19, 14, 5, 3, 7, 20}
sn = np.array([int(v[-3:]) in SURE_N for v in ns.nv])
L = ["후보 | train OOF AUC(고장 vs 정상) | val 중 train정상 95% 위 | val 중앙 z | 확실정상 val 23 중앙 z · 95% 위 | AUC(val vs train정상)"]
for nm, f in V.items():
    Lt, Lv = f(LPt), f(LPv); oof = np.empty(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Lt, y):
        oof[b] = score(fit(Lt[a][nor[a]]), Lt[b])
    sv = score(fit(Lt[nor]), Lv)
    med = np.median(oof[nor]); sc = 1.4826 * np.median(np.abs(oof[nor] - med)); z = lambda v: (v - med) / sc
    p95 = np.percentile(oof[nor], 95)
    auc_v = roc_auc_score(np.r_[np.zeros(nor.sum()), np.ones(len(sv))], np.r_[oof[nor], sv])
    L.append(f"{nm} | {roc_auc_score(y, oof):.3f} | {np.mean(sv > p95):.0%} | {np.median(z(sv)):+.2f} | {np.median(z(sv[sn])):+.2f} · {np.mean(sv[sn] > p95):.0%} | {auc_v:.2f}")
    print(L[-1], flush=True)
json.dump({"mode": "사전 점검 (중첩 CV 아님, 5겹 OOF seed 0)", "rows": [r.split(" | ") for r in L]}, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
