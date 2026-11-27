# EXP-044 질문답용 (run.py는 고치지 않음): "10 kHz 위(10–12.5 kHz, 나이퀴스트까지) 정보가 필요한가?"
# (a) 대역별 단변량 분리력: 대역 평균 log PSD의 ROC-AUC (고장 vs 정상)  (b) 신호를 10 kHz 저역통과(8차 버터워스 양방향)한 뒤 피처 · AE · 규칙을 모두 다시 계산해 J와 같은 구성을 중첩 CV로 비교
# 같은 절차(바깥 5겹 × seed 5, 폴드마다 한 번 학습, 이진 판정 = 결합 OR E2 OR 8.3k)로 원래 신호와 저역통과 신호를 각각 잼. AE는 캐시를 쓰지 않고 새로 학습(캐시가 원본 신호 기준이라)
# 실행: conda run -n qe-phm python "3_단계/5_모델실험/4_J모델/EXP-044_AE추가/자료/분석/10k이상_제거.py"
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import roc_auc_score, fbeta_score
from sklearn.model_selection import StratifiedGroupKFold
import src.experts as E
from src.exp2util import *
from src.features2 import cached_feat2
from src.models2 import TwoAE
from src.viz import log_psd
HERE = os.path.dirname(os.path.abspath(__file__)); LOG = os.path.join(HERE, "figs", "10k이상_제거_log.txt"); out = []
def P(s): out.append(s); print(s, flush=True)
ns = NS(); y, names, g = ns.y, ns.names, ns.g; nor = y == 0; FS = 25000

# ---- (a) 대역별 분리력 (신호 그대로)
f, L = log_psd(ns.X)
P("(a) 대역 평균 log PSD의 ROC-AUC (고장 vs 정상, 0.5 = 분리 못 함)")
for lo, hi in ((200, 1000), (1000, 2000), (2000, 4000), (4000, 5500), (5500, 8000), (8000, 10000), (10000, 11500), (11500, 12500)):
    m = (f >= lo) & (f < hi); v = L[:, m].mean(1); P(f"    {lo:>5}–{hi:<5} Hz  AUC {roc_auc_score(y, v):.3f}  (정상 중앙 {np.median(v[nor]):.2f}, 고장 중앙 {np.median(v[~nor]):.2f})")
sd = L[nor].std(0); P("    정상 PSD가 10 kHz에서 12.5 kHz로 가며 얼마나 떨어지나(중앙 log10 PSD): " + ", ".join(f"{int(h)}Hz {np.median(L[nor][:, np.argmin(abs(f - h))]):.2f}" for h in (8000, 9000, 10000, 11000, 12000, 12400)))

# ---- (b) 같은 절차로 비교
def run(F, tag):
    pred = np.zeros((5, ns.n), bool); main = np.zeros((5, ns.n), bool)
    for s in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g):
            m = TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)).fit(F[tr][:, :-1], y[tr]); f_ = m.fires(F[te][:, :-1])
            main[s, te] = f_["main"]; pred[s, te] = f_["main"] | f_["E2"] | (F[te][:, -1] > .5)
    for nm, p in (("결합 + E2 + 8.3k", pred), ("결합만", main)):
        f2 = [fbeta_score(y, q, beta=2) for q in p]; mc = {names[i]: int(c) for i, c in enumerate(((y == 1) & ~p).sum(0)) if c >= 2}; fc = {names[i]: int(c) for i, c in enumerate(((y == 0) & p).sum(0)) if c >= 2}
        P(f"    {tag:20s} {nm:14s} F2 {np.mean(f2):.3f} ± {np.std(f2):.3f} · 재현율 {np.mean([q[y == 1].mean() for q in p]):.3f} · 오경보율 {np.mean([q[y == 0].mean() for q in p]):.3f} · 미탐(2회+) {mc} · 오경보(2회+) {fc}")
idx = np.arange(ns.n, dtype=float)
spec, res = rule83_flags(ns.X); F0 = np.column_stack([ns.M, idx, (spec | res).astype(float)])
P("\n(b) 같은 절차 비교 (바깥 5겹 × seed 5)")
run(F0, "원래 신호")                                              # AE 캐시 사용 (빠름)
X10 = sosfiltfilt(butter(8, 10000, btype="low", fs=FS, output="sos"), X := ns.X, axis=-1)
T10, L10, _ = cached_feat2("train_lp10k", X10)
M10 = np.hstack([T10[ns.FIXED].values, T10[["L1_S"]].values, L10]); spec10, res10 = rule83_flags(X10)
_, LP10 = E.segment_logpsd(X10); E.LP = LP10; E.ae_segerr = E.ae_segerr.func                    # 캐시 없이, 저역통과 신호의 스펙트럼으로 AE 학습
run(np.column_stack([M10, idx, (spec10 | res10).astype(float)]), "10 kHz 저역통과")
P(f"    8.3k 규칙이 켜진 train 고장: 원래 {[names[i][-3:] for i in np.flatnonzero((spec | res) & (y == 1))]} · 저역통과 {[names[i][-3:] for i in np.flatnonzero((spec10 | res10) & (y == 1))]}")
open(LOG, "w", encoding="utf-8").write("\n".join(out)); P("done")
