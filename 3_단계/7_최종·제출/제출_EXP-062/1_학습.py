# %% [markdown]
# 최종 모델 학습 (EXP-062 구성 3, 11-25 사용자 확정)
# - 구성: J-축소 + K83 + 11 kHz 상한(fmax=11000: 충격 검출기 입력 1–11 kHz 대역통과 · 옥타브 마지막 칸 8–11 kHz · AE 입력 ≤ 11 kHz)
#   [ET(H 피처 13열 · 울림 비율 축소 + 봉우리 2열) OOF logit, 박자 z(L1_lock), AE z] → 로지스틱 ≥ 문턱
#   OR L2 위 꼬리 규칙(E2) OR 8.3 kHz 규칙(스펙트럼 · K83 OR 공진)
# - 평가(중첩 CV, 그룹 = 097·101·120 한 그룹, 이상치 정상 학습 포함): F2 0.971 ± 0.022 · 재현율 0.987 · 오경보율 0.023
#   (3_단계/5_모델비교/EXP-062_1125_11k상한·그룹CV)
# - 학습 150개 전체로 한 번만 학습한다. val · test를 본 뒤 다시 학습하거나 문턱을 고치지 않는다
# - 산출물은 모두 이 폴더 아래(모델/ · 결과/)에 만든다
# 실행(어디서 실행해도 같음): conda run -n qe-phm python "3_단계/7_최종·제출/제출_EXP-062/1_학습.py"
# %%
import os, sys
HERE = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")) and os.path.dirname(ROOT) != ROOT:   # src가 있는 상위 폴더 = 프로젝트 루트
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import joblib
import numpy as np
from src.data import load_dataset
from src.final import FinalJ, J_ALPHA, J_ET_N, J_Q_E2, J_RING, K83_MIN, P83_MIN, W83_MIN, RES_FIT_MIN

CONFIG = dict(fmax=11000)                     # EXP-062 구성 3. 나머지 옵션(events · atoms · res_new · cols · peak)은 기본값
MODEL_DIR = os.path.join(HERE, "모델"); RES_DIR = os.path.join(HERE, "결과")
os.makedirs(MODEL_DIR, exist_ok=True); os.makedirs(RES_DIR, exist_ok=True)

X, y, names = load_dataset(os.path.join(ROOT, "2_data", "TeamProject_train_변경"))       # 캐시가 아니라 원본 CSV
m = FinalJ(**CONFIG).fit(X, y, names)
joblib.dump(m, os.path.join(MODEL_DIR, "final_model_062.joblib"))

# %% 학습 요약 (사람이 읽는 기록)
c = m.core_; w = c.c_.coef_[0]
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
L = [f"학습 파일 {len(y)}개 (정상 {(y == 0).sum()} · 고장 {(y == 1).sum()}) · 최종 모델 EXP-062 구성 3 · 옵션 {CONFIG}", "",
     f"[결합] 로지스틱: 결정값 = {w[0]:+.3f} × ET logit {w[1]:+.3f} × 박자 z {w[2]:+.3f} × AE z {c.c_.intercept_[0]:+.3f} ≥ 문턱 {c.tc_:.3f} 이면 고장",
     f"  ET = ExtraTrees {J_ET_N}그루 · 결합에는 안쪽 4등분 OOF logit",
     f"  울림 비율 축소 (α = {J_ALPHA:g}): {', '.join(J_RING)}, p0 = {np.round(c.p0_, 3).tolist()}",
     f"  박자 z = (L1_lock − {c.zf_.med:.3f}) / {c.zf_.sc:.3f} · AE z = (AE 점수 − {c.aef_.med:.3f}) / {c.aef_.sc:.3f}",
     "", f"[L2 위 꼬리 규칙(E2)] L2_hfl > {c.t18_:.3f} (학습 정상 {J_Q_E2:g} % 분위) 이고 충격 ≥ 1개 → 고장",
     f"[8.3 kHz] 돌출 ≥ {P83_MIN} dec · 폭 ≥ {W83_MIN:.0f} Hz · K83 > {K83_MIN:g}  OR  8.0–8.7 kHz · ζ ≤ 0.02 충격 fit ≥ {RES_FIT_MIN} → 고장", ""]
D = m.predict_table(X, names)
D.to_csv(os.path.join(RES_DIR, "학습파일_재대입표.csv"), index=False, encoding="utf-8-sig")
yb = y == 1
L += ["[재대입 확인] 학습한 150개를 다시 판정(성능 아님, 규칙이 켜지는 파일 확인용. 성능은 중첩 CV 0.971 ± 0.022)",
      f"  8.3 kHz 스펙트럼: {' '.join(short(n) for n in D.file[D.spec83]) or '없음'} · 공진: {' '.join(short(n) for n in D.file[D.res83]) or '없음'}",
      f"  E2: {' '.join(short(n) for n in D.file[D.E2]) or '없음'}",
      f"  규칙이 켜진 정상: {' '.join(short(n) for n in D.file[(D.rule83 | D.E2) & ~yb]) or '없음'}",
      f"  놓친 고장: {' '.join(short(n) for n in D.file[(D.pred == 0) & yb]) or '없음'} · 오경보: {' '.join(short(n) for n in D.file[(D.pred == 1) & ~yb]) or '없음'}"]
open(os.path.join(RES_DIR, "학습요약.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
print("done")
