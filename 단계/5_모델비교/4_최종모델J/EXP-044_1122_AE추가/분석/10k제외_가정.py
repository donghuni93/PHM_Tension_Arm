# EXP-044 질문답용 (run.py는 고치지 않음): "8–12.5 kHz 옥타브 칸(= 10 kHz 대역) 피처를 안 본다면?" — ET 입력에서 H10(울림 8–12.5k) · H15(몫 8–12.5k)를 뺌
# 8.3k 규칙(8.0–8.7 kHz)은 그대로 둔다(규칙은 이 질문의 대상이 아님). 고정 격자(옥타브 4칸)라 9.5–12.5 kHz만 따로 뺄 수는 없다
# 실행: conda run -n qe-phm python "단계/5_모델비교/4_최종모델J/EXP-044_1122_AE추가/분석/10k제외_가정.py"
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from src.exp2util import *
from src.evaluate import nested_cv
from src.models2 import TwoAE
HERE = os.path.dirname(os.path.abspath(__file__))
HERE_EXP = os.path.dirname(HERE)   # EXP-044 폴더 (result.json · figs/logistic_plane.png 위치)
ns = NS(); y, names, g = ns.y, ns.names, ns.g
DROP = ["H10_ring_8-12.5k", "H15_share_8-12.5k"]; di = [ns.FIXED.index(c) for c in DROP]
fixed2 = tuple(c for c in ns.FIXED if c not in DROP); nf2 = ns.nf - len(DROP)
M2 = np.delete(ns.M, di, axis=1)
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); idx = np.arange(ns.n, dtype=float)
mk = lambda experts: (lambda: TwoAE("lin", experts, fixed=fixed2, nf=nf2, df=ns.df, peak=tuple(ns.PEAK)))
J = json.load(open(os.path.join(HERE_EXP, "result.json"), encoding="utf-8"))["configs"]
out = []
def rep(nm, R):
    mc = {names[i]: int(c) for i, c in enumerate(R["miss_counts"]) if c >= 2}; fc = {names[i]: int(c) for i, c in enumerate(R["fp_counts"]) if c >= 2}
    s = f"{nm:44s} F2 {R['f2'].mean():.3f} ± {R['f2'].std():.3f} · 재현율 {R['recall'].mean():.3f} · 오경보율 {R['fpr'].mean():.3f} · 미탐(2회+) {mc} · 오경보(2회+) {fc}"
    out.append(s); print(s, flush=True)
for k in ("J | W + AE + E2 + 8.3k (최종 후보)", "K | W + AE (전문가 · 규칙 없음)"):
    c = J["중첩: " + k]; s = f"[기준 EXP-044] {k:34s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} · 미탐 {c['miss_counts']} · 오경보 {c['fp_counts']}"; out.append(s); print(s)
rep("J' (8–12.5k 피처 제거) + E2 + 8.3k", nested_cv(np.column_stack([M2, idx, r83]), y, g, lambda: Add83(mk(("E2",))), n_jobs=-1))
rep("K' (8–12.5k 피처 제거) 결합만", nested_cv(np.column_stack([M2, idx]), y, g, mk(()), n_jobs=-1))
# ET 중요도 (전체 150 학습) — 뺀 두 피처가 원래 얼마나 쓰였나
m0 = TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)).fit(np.column_stack([ns.M, idx]), y)
imp = m0.m_.feature_importances_; cols = list(ns.FIXED) + list(ns.PEAK); order = np.argsort(imp)[::-1]
s = "ET 중요도(제거 전) 상위: " + ", ".join(f"{cols[i].split('_')[0]} {imp[i]:.3f}" for i in order[:8]) + " · H10 " + f"{imp[cols.index(DROP[0])]:.3f} · H15 {imp[cols.index(DROP[1])]:.3f} (순위 {list(order).index(cols.index(DROP[0])) + 1} · {list(order).index(cols.index(DROP[1])) + 1} / {len(cols)})"
out.append(s); print(s)
open(os.path.join(HERE, "figs", "10k제외_log.txt"), "w", encoding="utf-8").write("\n".join(out))
