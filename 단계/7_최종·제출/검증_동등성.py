# %% [markdown]
# 검증: 제출용 코드(src/final.py)가 실험용 코드(src/experts.py)와 같은 점수를 내는가
# - 왜 필요한가: 중첩 CV 점수(0.942)는 experts.py로 잰 것이다. 제출은 final.py로 한다.
#   두 코드가 다르게 계산하면 "평가한 모델"과 "제출하는 모델"이 다른 것이 된다
# - 방법: 같은 분할(바깥 5-fold seed 0의 fold 0·1)에서 학습 → 시험 파일 점수를 두 코드로 구해 차이의 최댓값을 본다
#   final.py는 캐시를 쓰지 않고 신호에서 피처를 새로 계산하므로, 피처 계산이 같은지도 함께 확인된다
# 실행: conda run -n qe-phm python "단계/7_최종·제출/검증_동등성.py"

# %%
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from src.exp import load_train
from src.experts import RuleOverride, SingleHitCap, Rule83, KnockRule, LockRule, ae_mean, et_logit
from src.final import Core, signal_tables, take

warnings.filterwarnings("ignore", category=ConvergenceWarning)
X, y, names = load_train()
g = np.arange(len(y)); F = g[:, None]
T, _ = signal_tables(X)                                        # 캐시 없이 새로 계산
B0 = dict(experts=(ae_mean, et_logit), combine="lr")
R83, KN, LK = Rule83(0.75, 150), KnockRule(2.0, 0.7), LockRule(0.1)
CASES = [("최종 구성 (8.3k + 유격)", RuleOverride(base_kw=B0, rule=[R83, KN]), dict()),
         ("+ 주기 잠금", RuleOverride(base_kw=B0, rule=[R83, KN, LK]), dict(use_lock=True)),
         ("+ 단일 타격 감점", SingleHitCap(base_kw=B0, rule=[R83, KN]), dict(use_cap=True))]
lines = []
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g))[:2]
for k, (tr, te) in enumerate(splits):
    for lab, ref, kw in CASES:
        a = ref.fit(F[tr], y[tr]).decision_function(F[te])
        b = Core(**kw).fit(take(T, tr), y[tr]).parts(take(T, te))["score"].values
        lines.append(f"fold {k} · {lab:22s} 점수 차이 최댓값 {np.abs(a - b).max():.2e} (시험 파일 {len(te)}개, 점수 범위 {a.min():.2f} ~ {a.max():.2f})")
ok = all(float(l.split("최댓값 ")[1].split(" ")[0]) < 1e-6 for l in lines)
lines.append("결과: " + ("같다 (차이 < 1e-6)" if ok else "다르다 — 원인을 찾기 전에는 제출 코드를 쓰지 않는다"))
open(os.path.join(HERE, "검증_동등성.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
