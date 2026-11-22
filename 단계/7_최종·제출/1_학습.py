# %% [markdown]
# 7단계 ① 최종 모델 학습: 학습 150개 전체 → outputs/final/final_model.joblib + 결과/학습요약.txt
# - 구성 = J-축소 (EXP-050, 11-22 기준 구성): [ET(H 피처 · 울림 비율 축소) OOF logit, 박자 z, AE z] → 로지스틱 ≥ 문턱
#   OR E2(하중 대비 고주파) OR 8.3 kHz 규칙(스펙트럼 OR 공진). 실험 코드와 같은 판정인지는 검증_동등성.py (결과/동등성_J.txt)
# - 한 번만 돌린다. val·test를 본 뒤에 다시 학습하거나 문턱을 고치지 않는다 (단계/1_기준·기록/결정.md)
# - 예전 최종(EXP-034 c, FinalModel)은 src/final.py에 그대로 있다 (git 기록 · 비교용)
# 실행: conda run -n qe-phm python "단계/7_최종·제출/1_학습.py"
# %%
import os, sys
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import joblib
import numpy as np
from src.data import load_dataset
from src.final import FinalJ, J_ALPHA, J_ET_N, J_Q_E2, J_RING, P83_MIN, W83_MIN, RES_FIT_MIN
OUT = os.path.join(_d, "outputs", "final"); os.makedirs(OUT, exist_ok=True)
X, y, names = load_dataset(os.path.join(_d, "data", "TeamProject_train_변경"))   # 캐시가 아니라 원본 CSV에서 읽는다
m = FinalJ().fit(X, y, names)
joblib.dump(m, os.path.join(OUT, "final_model.joblib"))

# %% 학습 요약 (사람이 읽는 기록)
c = m.core_
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
w = c.c_.coef_[0]
L = [f"학습 파일 {len(y)}개 (정상 {(y == 0).sum()} · 고장 {(y == 1).sum()}) · 구성 J-축소 (EXP-050)", "",
     f"[결합] 로지스틱: 결정값 = {w[0]:+.3f} × ET logit {w[1]:+.3f} × 박자 z {w[2]:+.3f} × AE z {c.c_.intercept_[0]:+.3f} ≥ 문턱 {c.tc_:.3f} 이면 고장",
     f"  ET = ExtraTrees {J_ET_N}그루, 입력 = H 피처 13열 + 가장 솟은 좁은 봉우리 2열 · 결합에는 안쪽 4등분 OOF logit",
     f"  울림 비율 축소 (α = {J_ALPHA:g}): {', '.join(J_RING)} → (k + α·p0)/(n + α), p0 = {np.round(c.p0_, 3).tolist()} (학습 정상 충격의 대역 비율)",
     f"  박자 z = (L1_S − {c.zf_.med:.3f}) / {c.zf_.sc:.3f} · AE z = (AE 점수 − {c.aef_.med:.3f}) / {c.aef_.sc:.3f}  (학습 정상 중앙값 · 강건 퍼짐)",
     f"  문턱 = 학습 결합 점수에서 F2가 최대인 값 (J와 같은 절차)",
     "", f"[E2] H18(하중 대비 고주파) > {c.t18_:.3f} (학습 정상 {J_Q_E2:g} % 분위) 이고 충격 ≥ 1개 → 고장",
     f"[8.3 kHz] (스펙트럼) 돌출 ≥ {P83_MIN} dec 이고 폭 ≥ {W83_MIN:.0f} Hz  OR  (공진) 8.0–8.7 kHz · ζ ≤ 0.02 충격의 fit ≥ {RES_FIT_MIN} → 고장", ""]
D = m.predict_table(X, names)
D.to_csv(os.path.join(OUT, "학습파일_재대입표.csv"), index=False, encoding="utf-8-sig")
yb = y == 1
L += ["[재대입 확인] 학습한 150개를 그대로 다시 판정 (모델이 이미 본 파일이라 성능이 아님. 규칙이 어느 파일에 켜지는지 확인용. 성능은 EXP-050 중첩 CV 0.977 ± 0.010)",
      f"  8.3 kHz 스펙트럼 규칙: {' '.join(short(n) for n in D.file[D.spec83]) or '없음'} · 공진 규칙: {' '.join(short(n) for n in D.file[D.res83]) or '없음'}",
      f"  E2: {' '.join(short(n) for n in D.file[D.E2]) or '없음'}",
      f"  규칙이 켜진 정상: {' '.join(short(n) for n in D.file[(D.rule83 | D.E2) & ~yb]) or '없음'}",
      f"  놓친 고장: {' '.join(short(n) for n in D.file[(D.pred == 0) & yb]) or '없음'} · 오경보: {' '.join(short(n) for n in D.file[(D.pred == 1) & ~yb]) or '없음'}"]
os.makedirs(os.path.join(HERE, "결과"), exist_ok=True)
open(os.path.join(HERE, "결과", "학습요약.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
