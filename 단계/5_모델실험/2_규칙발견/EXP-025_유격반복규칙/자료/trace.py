# %% [markdown]
# EXP-025 추적 (README 해석용, run.py와 같은 구성·분할) — run.py는 기록 후 고치지 않으므로 별도 파일
# 1. 규칙 관련 파일의 K1·K2·K3·r (150개 정상 중심, 설명용)
# 2. 중첩 CV를 같은 방식으로 다시 돌려 바깥 fold별 임계값·014 점수·오경보 정상을 기준(EXP-021)과 주 구성에서 비교
# 출력: figs/trace.txt
# 실행: conda run -n qe-phm python 단계/5_모델비교/2_규칙발견/EXP-025_1114_유격반복규칙/trace.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import _oof_scores, best_f2_threshold, _scores
from src.features import feature_table_knock
from src.experts import RuleOverride, Rule83, KnockRule, ae_mean, et_logit
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y, names = load_train(); g = np.arange(len(y)); F = g[:, None]
K = cached_table("knock", feature_table_knock, X)
BASE = dict(experts=(ae_mean, et_logit), combine="lr")
R83, KN = Rule83(0.75, 150), KnockRule(2.0, 0.7)
CFG = {"기준": dict(base_kw=BASE, rule=[R83]), "주": dict(base_kw=BASE, rule=[R83, KN])}
OUT = []
def log(*a): OUT.append(" ".join(str(v) for v in a)); print(*a, flush=True)
s = lambda n: n.replace("abn_", "A").replace("nor_", "N")

# %% 1. 규칙 관련 파일 값
kn = KnockRule(2.0, 0.7).fit(g); r = kn.r(g)
log("[1] r ≥ 2 파일 (150개 정상 중심): 이름 · 라벨 · K1 · K2 · K3(2위/1위) · r · 규칙")
for i in np.argsort(-r):
    if r[i] < 2: break
    log(f"  {names[i]} · {'고장' if y[i] else '정상'} · K1 {K.iloc[i, 0]:.2f} · K2 {K.iloc[i, 1]:+.2f} · K3 {K.iloc[i, 2]:.2f} · r {r[i]:.2f} · {'켜짐' if kn(np.array([i]))[0] else ''}")
for n in ("abn_022", "abn_005", "abn_024", "nor_025", "nor_055", "nor_116", "nor_059"):
    i = names.index(n)
    log(f"  (r < 2) {n} · K1 {K.iloc[i, 0]:.2f} · K2 {K.iloc[i, 1]:+.2f} · K3 {K.iloc[i, 2]:.2f} · r {r[i]:.2f}")

# %% 2. 바깥 fold별 추적 (nested_cv와 같은 분할·임계값 규칙)
WATCH = ["abn_014", "abn_017", "abn_005"]
log("\n[2] 바깥 fold별: seed · fold · 기준 임계값 → 주 임계값 · 감시 고장 점수(기준 → 주, 판정) · 그 fold 오경보 정상(기준 → 주)")
for seed in range(5):
    for k, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(F, y, g)):
        res = {}
        for c, kw in CFG.items():
            inner = StratifiedGroupKFold(4, shuffle=True, random_state=seed)
            oof = _oof_scores(F[tr], y[tr], g[tr], lambda kw=kw: RuleOverride(**kw), inner)
            t = best_f2_threshold(y[tr], oof)
            sc = _scores(RuleOverride(**kw).fit(F[tr], y[tr]), F[te])
            res[c] = (t, dict(zip([names[i] for i in te], sc)))
        w = [n for n in WATCH if n in res["기준"][1]]
        fp = {c: [s(n) for n, v in res[c][1].items() if n.startswith("nor") and v >= res[c][0]] for c in CFG}
        miss = {c: [s(n) for n, v in res[c][1].items() if n.startswith("abn") and v < res[c][0]] for c in CFG}
        wtxt = " · ".join(f"{s(n)} {res['기준'][1][n]:.3f}({'O' if res['기준'][1][n] >= res['기준'][0] else 'X'}) → "
                          f"{res['주'][1][n]:.3f}({'O' if res['주'][1][n] >= res['주'][0] else 'X'})" for n in w)
        log(f"  s{seed} f{k} · t {res['기준'][0]:.3f} → {res['주'][0]:.3f} · {wtxt or '-'} · "
            f"오경보 {' '.join(fp['기준']) or '없음'} → {' '.join(fp['주']) or '없음'} · 놓침 {' '.join(miss['기준']) or '없음'} → {' '.join(miss['주']) or '없음'}")

open(os.path.join(HERE, "figs", "trace.txt"), "w", encoding="utf-8").write("\n".join(OUT))
print("done")
