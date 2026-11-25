# %% [markdown]
# EXP-030 추가 확인 2 (11-16, run.py 기록 뒤, 독립 검토 지적): 임계값을 '간격'의 어디에 둘 것인가
# - 지적: best_f2_threshold는 후보가 점수값 자체라서, F2가 최대인 가장 낮은 임계값 = "잡히는 고장 중 점수가 가장 낮은 고장의 점수"가 된다.
#   즉 임계값이 고장 쪽 끝에 붙고(고장 쪽 여유 0), 그 아래 정상 최고점까지의 빈 간격은 전부 정상 쪽 여유가 된다.
#   F2는 미탐이 오경보의 4배 무게인데 여유는 반대로 준 셈이다
# - 비교 (같은 분할, 최종 구성 A, 안쪽 OOF에서는 세 위치의 F2가 똑같다 → 차이는 새 파일에서만 난다)
#   상단 = 지금 방식 (고장 최저점) · 중간 = 그 점수와 바로 아래 점수의 중점 · 하단 = 바로 아래 점수 바로 위
#   각각 안쪽 seed 1개 / 3개 중앙값
# - 채택 규칙 (결과를 보기 전에 고정): 지금 최종 정책(상단 · 3개 중앙값, 0.938 ± 0.014)보다 중첩 F2 평균이 0.014(sd) 넘게 높은 정책이 있으면
#   그중 가장 단순한 것으로 바꾼다. 없으면 지금 정책을 유지하고 결과만 기록한다
# 실행: conda run -n qe-phm python "단계/5_모델실험/아카이브/EXP-030_1116_최종후보고정_규칙감사/추가_임계값위치.py"

# %%
import os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import _oof_scores, _scores, best_f2_threshold, _collect
from src.experts import RuleOverride, Rule83, KnockRule, ae_mean, et_logit
from joblib import Parallel, delayed, parallel_config
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]
make = lambda: RuleOverride(base_kw=dict(experts=(ae_mean, et_logit), combine="lr"), rule=[Rule83(0.75, 150), KnockRule(2.0, 0.7)])


def three(yy, oof):
    """(상단, 중간, 하단) 임계값. 상단 = best_f2_threshold, 하단 = 바로 아래 점수 바로 위"""
    hi = best_f2_threshold(yy, oof)
    below = oof[oof < hi]
    lo = np.nextafter(below.max(), np.inf) if len(below) else hi
    return hi, (hi + lo) / 2, lo


def job(seed, tr, te):
    ts = [three(y[tr], _oof_scores(F[tr], y[tr], g[tr], make, StratifiedGroupKFold(4, shuffle=True, random_state=k)))
          for k in (seed, (seed + 1) % 5, (seed + 2) % 5)]
    return te, _scores(make().fit(F[tr], y[tr]), F[te]), np.array(ts)      # ts (안쪽 seed 3, 위치 3)


jobs = [(s, tr, te) for s in range(5) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
with parallel_config(backend="loky", inner_max_num_threads=1):
    out = Parallel(n_jobs=-1)(delayed(job)(*j) for j in jobs)

short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
L, R = [], {}
gap = np.array([ts[:, 0] - ts[:, 2] for _, _, ts in out])
L.append(f"간격(상단 − 하단) 크기: 중앙 {np.median(gap):.2f} · 5–95% {np.percentile(gap, 5):.2f}–{np.percentile(gap, 95):.2f} (결합 점수 단위, 분할 75개)")
for pi, pos in enumerate(("상단 (지금)", "중간", "하단")):
    for mode in ("seed 1개", "3개 중앙값"):
        thr = [ts[0, pi] if mode == "seed 1개" else float(np.median(ts[:, pi])) for _, _, ts in out]
        r = _collect([(te, (sc >= t).astype(int), t) for (te, sc, _), t in zip(out, thr)], y, range(5), 5)
        R[(pos, mode)] = r
        miss = " ".join(f"{short(names[i])}({c})" for i, c in enumerate(r["miss_counts"]) if c)
        fp = " ".join(f"{short(names[i])}({c})" for i, c in enumerate(r["fp_counts"]) if c)
        L.append(f"{pos:10s} · {mode:8s} F2 {r['f2'].mean():.3f} ± {r['f2'].std():.3f} · 재현율 {r['recall'].mean():.3f} · 오경보율 {r['fpr'].mean():.3f}"
                 f" | 놓침 {miss or '없음'} | 오경보 {fp or '없음'}")
base = R[("상단 (지금)", "3개 중앙값")]["f2"].mean()
better = [k for k, r in R.items() if r["f2"].mean() > base + 0.014]
L.append(f"판정: 지금 최종 정책 {base:.3f}. sd(0.014) 넘게 높은 정책 = {better or '없음'} → {'교체 검토' if better else '지금 정책 유지'}")
open(os.path.join(HERE, "figs", "threshold_position.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
