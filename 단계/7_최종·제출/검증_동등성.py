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

# EXP-034 c 참조 (EXP-034 run.py와 같은 정의). 울림 값은 2-16이 남긴 impacts.csv에서 읽는다 → final.py가 신호에서 새로 계산한 값과 비교됨
import pandas as pd
from sklearn.base import BaseEstimator
from src.exp import make_et, cached_table, feature_table_ext
IM = pd.read_csv(os.path.join(_d, "단계", "2_EDA", "2-7_충격울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
key = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
cnt = IM.groupby("file").size().reindex(key).fillna(0).values
frac = lambda lo, hi: np.nan_to_num(IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean().reindex(key).values.astype(float), nan=-1.0)
R24, RBODY = frac(2125, 4125), frac(5500, 7750)
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values
E = cached_table("ext", feature_table_ext, X).values


def ETX(tr, te):
    tr, te = np.array(tr), np.array(te)
    M = np.hstack([E, R24[:, None], RBODY[:, None]])
    p = np.clip(make_et(seed=0)().fit(M[tr], y[tr]).predict_proba(M[te])[:, 1], 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p))[:, None]


class Either:
    def __init__(self, a, b): self.a, self.b = a, b
    def __call__(self, te): return self.a(te) | self.b(te)


class BodyCap(BaseEstimator):
    def __init__(self, base_kw=None, rule=None): self.base_kw, self.rule = base_kw, rule
    def fit(self, F, yy):
        yy = np.asarray(yy); self.m_ = RuleOverride(self.base_kw, self.rule).fit(F, yy); st = self.m_.m_
        self.cap_ = float(np.median(st.m_.decision_function(st.oof_z_)[yy == 0])); return self
    def decision_function(self, F):
        s = self.m_.decision_function(F); te = np.asarray(F)[:, 0].astype(int)
        return np.where((RBODY[te] >= 0.9) & (cnt[te] >= 3) & (s < 50), np.minimum(s, self.cap_), s)


RS = lambda te: RES[list(te)] >= 0.7
B1 = dict(experts=(ae_mean, ETX), combine="lr")
CASES = [("최종 구성 EXP-034 c", BodyCap(B1, [KN, Either(R83, RS)]), dict()),
         ("예전 최종 EXP-025", RuleOverride(base_kw=B0, rule=[R83, KN]), dict(ring=False)),
         ("EXP-025 + 주기 잠금", RuleOverride(base_kw=B0, rule=[R83, KN, LK]), dict(use_lock=True, ring=False)),
         ("EXP-025 + 단일 타격 감점", SingleHitCap(base_kw=B0, rule=[R83, KN]), dict(use_cap=True, ring=False))]
# 울림 표 자체도 비교 (신호에서 새로 계산 vs 2-16 impacts.csv)
lines = [f"울림 표 차이 최댓값: 충격 수 {np.abs(T['NIMP'] - cnt).max():.0f} · 울림 2열 {np.abs(T['RING'] - np.c_[R24, RBODY]).max():.2e} · 공진 fit {np.abs(T['RES'] - RES).max():.2e}"]
splits = list(StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g))[:2]
for k, (tr, te) in enumerate(splits):
    for lab, ref, kw in CASES:
        a = ref.fit(F[tr], y[tr]).decision_function(F[te])
        b = Core(**kw).fit(take(T, tr), y[tr]).parts(take(T, te))["score"].values
        lines.append(f"fold {k} · {lab:22s} 점수 차이 최댓값 {np.abs(a - b).max():.2e} (시험 파일 {len(te)}개, 점수 범위 {a.min():.2f} ~ {a.max():.2f})")
ok = all(float(l.split("최댓값 ")[1].split(" ")[0]) < 1e-6 for l in lines if l.startswith("fold"))
lines.append("결과: " + ("같다 (차이 < 1e-6)" if ok else "다르다 — 원인을 찾기 전에는 제출 코드를 쓰지 않는다"))
open(os.path.join(HERE, "결과", "검증_동등성.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
sys.exit(0 if ok else 1)
