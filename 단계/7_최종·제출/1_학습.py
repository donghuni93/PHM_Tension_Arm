# %% [markdown]
# 7단계 ① 최종 모델 학습: 학습 150개 전체 → outputs/final/final_model.joblib + 학습요약.txt
# - 구성 = EXP-034 c (11-17 교체): AE + ET(ext150 + 울림 2열) → 로지스틱 + 유격 반복 규칙 + 8.3 kHz 규칙(스펙트럼 OR 공진) + 본체 모드 내림
# - 한 번만 돌린다. val·test를 본 뒤에 다시 학습하거나 문턱을 고치지 않는다 (단계/1_기준·기록/결정.md 'val 점수')
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
from sklearn.metrics import fbeta_score
from src.data import load_dataset
from src.final import FinalModel, P83_MIN, W83_MIN, R_MIN, RATIO_MIN, RES_FIT_MIN, BODY_MIN, BODY_NMIN

OUT = os.path.join(_d, "outputs", "final"); os.makedirs(OUT, exist_ok=True)
X, y, names = load_dataset(os.path.join(_d, "data", "TeamProject_train_변경"))   # 캐시가 아니라 원본 CSV에서 읽는다
m = FinalModel().fit(X, y, names)
joblib.dump(m, os.path.join(OUT, "final_model.joblib"))

# %% 학습 요약 (사람이 읽는 기록)
c = m.core_
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
L = [f"학습 파일 {len(y)}개 (정상 {(y == 0).sum()} · 고장 {(y == 1).sum()})", "",
     "[결합] 로지스틱 회귀: 결정값 = "
     f"{c.lr_.coef_[0, 0]:+.3f} × (스펙트럼 z) {c.lr_.coef_[0, 1]:+.3f} × (충격 z) {c.lr_.intercept_[0]:+.3f}",
     f"  스펙트럼(AE) z = (AE 점수 − {c.med_[0]:.3f}) / {c.sc_[0]:.3f} · 충격(ET) z = (ET 점수 − {c.med_[1]:.3f}) / {c.sc_[1]:.3f}  (정상 중앙값, 정상 90%선까지 거리)",
     "", f"[규칙 1] 8.3 kHz 모드: (스펙트럼) 돌출 ≥ {P83_MIN} dec 이고 폭 ≥ {W83_MIN:.0f} Hz  OR  (공진) 8.0–8.7 kHz · ζ ≤ 0.02 충격의 fit ≥ {RES_FIT_MIN}",
     f"[규칙 2] 유격 반복: 정상 중심 거리 r ≥ {R_MIN} 이고 2위/1위 ≥ {RATIO_MIN}",
     f"  정상 중심(중앙값) K1 {c.kmed_[0]:.3f} · K2 {c.kmed_[1]:.3f} / 퍼짐(MAD×1.4826) K1 {c.kmad_[0]:.3f} · K2 {c.kmad_[1]:.3f}",
     f"[내림] 본체 모드: 충격 ≥ {BODY_NMIN}개이고 5.5–7.75 kHz 울림 비율 ≥ {BODY_MIN} → 점수를 학습 정상 OOF 중앙값 {c.cap_:.3f} 이하로",
     f"[ET 입력] ext150 + 울림 2열 (2.125–4.125 kHz · 5.5–7.75 kHz로 울린 충격 비율)",
     f"[관찰용, 판정에 쓰지 않음] 주기 잠금 문턱 {c.lock_th_:.3f} · 큰 타격 정확히 1개",
     "", (f"[임계값] 고정 {m.thr_:.3f} (EXP-035, 11-17 사용자 결정). 참고: 분할 seed {list(m.thr_seeds)} 에서 고른 값 {np.round(m.thr_all_, 3).tolist()}, 이 최소–최대 사이 점수는 '경계'로 표시" if m.thr_fixed is not None
          else f"[임계값] 분할 seed {list(m.thr_seeds)} 에서 각각 {np.round(m.thr_all_, 3).tolist()} → 중앙값 {m.thr_:.3f}"),
     "  (결합 점수가 이 값 이상이면 고장. 규칙이 켜진 파일은 +100이라 항상 고장)", ""]
L.append("[학습 파일 OOF 성적] 각 파일을 '자기를 빼고 학습한 모델'로 채점 → 최종 임계값 적용. 임계값을 이 점수들에서 골랐고 분할마다 점수 눈금이 달라 seed별 값이 크게 다르다. 성능 추정값이 아니다 (추정값은 중첩 CV: 고정 1.0에서 0.957 ± 0.010, EXP-035 — 같은 데이터에서 고른 값이라 낙관적)")
for s, o in zip(m.thr_seeds, m.oof_):
    p = (o >= m.thr_).astype(int)
    L.append(f"  seed {s}: F2 {fbeta_score(y, p, beta=2):.3f} · 놓친 고장 {' '.join(short(names[i]) for i in np.flatnonzero((y == 1) & (p == 0))) or '없음'}"
             f" · 오경보 {' '.join(short(names[i]) for i in np.flatnonzero((y == 0) & (p == 1))) or '없음'}")
D = m.predict_table(X, names)
D.to_csv(os.path.join(OUT, "학습파일_재대입표.csv"), index=False, encoding="utf-8-sig")
L += ["", "[재대입 확인] 학습한 150개를 그대로 다시 판정 (모델이 이미 본 파일이라 성능이 아님. 규칙이 어느 파일에 켜지는지 확인용)",
      f"  8.3 kHz 스펙트럼 규칙 켜짐: {' '.join(short(n) for n in D.file[D.spec83])} · 공진 규칙 켜짐: {' '.join(short(n) for n in D.file[D.res83])}",
      f"  본체 모드 내림: 정상 {int((D.body & (y == 0)).sum())} · 고장 {' '.join(short(n) for n in D.file[D.body & (y == 1)]) or '0'}",
      f"  유격 반복 규칙 켜짐: {' '.join(short(n) for n in D.file[D.knock])}",
      f"  규칙이 켜진 정상: {' '.join(short(n) for n in D.file[(D.rule83 | D.knock) & (y == 0)]) or '없음'}",
      f"  [관찰] 주기 잠금 문턱 이상: {' '.join(short(n) for n in D.file[D.lock_obs]) or '없음'} · 큰 타격 1개: 정상 {int((D.single_hit_obs & (y == 0)).sum())} · 고장 {int((D.single_hit_obs & (y == 1)).sum())}"]
open(os.path.join(HERE, "학습요약.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
