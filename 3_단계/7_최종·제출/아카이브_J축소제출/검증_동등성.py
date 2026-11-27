# %% [markdown]
# 검증: 제출용 J-축소(src/final.py FinalJ)가 실험용 J-축소(EXP-050, src.exp2util.j_shrink)와 같은 판정을 내는가 (11-22)
# - 왜 필요한가: 중첩 CV 점수(0.977)와 val 판정(10/40)은 실험 코드로 잰 것이다. 제출은 final.py로 한다.
#   두 코드가 다르게 계산하면 "평가한 모델"과 "제출하는 모델"이 다른 것이 된다
# - 방법: 학습 150개 전체로 두 코드를 각각 학습 → 결합 입력 Z · 로지스틱 계수 · 문턱 · E2 문턱 · 축소 p0와
#   학습 150 · val 40의 결합 점수 · 최종 판정을 비교한다. final.py는 캐시를 쓰지 않고 원본 CSV 신호에서 피처를 새로 계산한다
# - 실험 쪽은 full 모드(ET 300, EXP-050과 같음)로 만든다 → 아래에서 QE_FULL=1을 import 전에 켠다
# 실행: conda run -n qe-phm python "3_단계/7_최종·제출/아카이브_J축소제출/검증_동등성.py"  → 결과/동등성_J.txt

# %%
import os, sys, warnings
os.environ["QE_FULL"] = "1"
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
warnings.filterwarnings("ignore")
import numpy as np
from src.data import load_dataset
from src.evaluate import ET_N
from src.exp2util import NS, rule83_flags, j_mats, j_shrink, ts_of
from src.experts import LP as LP_TR, segment_logpsd
from src.final import FinalJ, _fit_ae, _ae_score
assert ET_N == 300, "실험 쪽 ET 나무 수가 300이 아니다 (QE_FULL)"

# %% 실험 코드 (EXP-050 J-축소와 같은 조립)
ns = NS(); y = ns.y; nor = y == 0
spec, res = rule83_flags(ns.X, k_gate=True); sv, rv = rule83_flags(ns.Xv, k_gate=True)   # 11-23: 8.3k 규칙에 K83 > 3 조건 추가(EXP-054)
M, Mv = j_mats(ns, spec | res)
top = j_shrink(ns)().fit(M, y); ts, tf = ts_of(top)
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
ft, fv = top.fires(M), ts.fires_raw(tf(Mv), RAW)
E_tr = ft["main"] | ft["E2"] | ft["R83"]; E_v = fv["main"] | fv["E2"] | sv | rv

# %% 제출 코드 (원본 CSV에서)
X, y2, names = load_dataset(os.path.join(_d, "2_data", "TeamProject_train_변경"))
assert list(names) == list(ns.names) and (np.asarray(y2) == y).all(), "학습 파일 순서가 다르다"
m = FinalJ().fit(X, y2, names); c = m.core_
Dt = m.predict_table(X, names); Dv = m.predict_table(ns.Xv, ns.nv)

# %% 비교
mx = lambda a, b: float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))
L = ["[제출 J-축소 vs 실험 J-축소(EXP-050 조립, full)] 학습 150 전체 학습",
     f"  축소 p0 차이 최대 {mx(c.p0_, top.p0_):.2e}",
     f"  결합 입력 Z (ET OOF logit · 박자 z · AE z) 차이 최대 {mx(c.Z_, ts.Z_):.2e}  (열별 {[round(mx(c.Z_[:, k], ts.Z_[:, k]), 9) for k in range(3)]})",
     f"  로지스틱 계수 차이 최대 {mx(np.r_[c.c_.coef_[0], c.c_.intercept_], np.r_[ts.c_.coef_[0], ts.c_.intercept_]):.2e} · 결합 문턱 {c.tc_:.6f} vs {ts.tc_:.6f}",
     f"  E2 문턱 {c.t18_:.6f} vs {ts.t18_:.6f}",
     f"  학습 150 결합 점수 차이 최대 {mx(Dt.score, ft['score']):.2e} · 최종 판정 같음 {int((Dt.pred.values.astype(bool) == E_tr).sum())}/150",
     f"  val 40 결합 점수 차이 최대 {mx(Dv.score, fv['score']):.2e} · 최종 판정 같음 {int((Dv.pred.values.astype(bool) == E_v).sum())}/40",
     f"  val 고장 판정 수: 제출 {int(Dv.pred.sum())} · 실험 {int(E_v.sum())}",
     f"  val 고장 판정 파일(제출): {' '.join(f[-3:] for f in Dv.file[Dv.pred == 1])}",
     f"  판정이 다른 val: {' '.join(ns.nv[i][-3:] for i in np.flatnonzero(Dv.pred.values.astype(bool) != E_v)) or '없음'}"]
os.makedirs(os.path.join(HERE, "결과"), exist_ok=True)
open(os.path.join(HERE, "결과", "동등성_J.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("\n".join(L))
