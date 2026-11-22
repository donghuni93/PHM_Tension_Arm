# %% [markdown]
# EXP-030 최종 후보 고정 — 규칙 감사(문턱 여유 · 측정 구간 민감도)와 임계값 정책
# - 목적: val 공개(11/19) 전에 최종 구성을 하나로 고정한다. 새 피처·새 규칙은 만들지 않는다 (있는 것 중에서 고르기만)
# - 사용자 기준 (11-16): 학습 데이터만 있는 상태라 과적합을 피해야 한다 → "명확한 도메인 규칙"만 남긴다
# - 규칙 채택 기준 (결과를 보기 전에 고정, 아래 코드가 이 기준대로 구성을 고른다)
#   C1 기전      : 물리 가설(H1–H6)에 연결되고 한 문장으로 말할 수 있다 (사람이 판단, 표에 적음)
#   C2 지지      : 규칙이 겨냥한 학습 파일이 3개 이상이다 (파일 1개를 보고 만든 규칙은 제외)
#   C3 문턱 여유 : 문턱을 ±20% 흔든 모든 조합에서 반대쪽 파일이 걸리지 않는다
#                  (올리는 규칙 = 정상이 켜지지 않음, 내리는 규칙 = 고장이 대상이 되지 않음)
#   C4 오류 방향 : '고장으로 올리는' 규칙이 틀리면 오경보 1건, '정상으로 내리는' 규칙이 틀리면 미탐 1건이다.
#                  F2 = 5TP / (5TP + 4FN + FP) 이므로 미탐 1건 = 오경보 4건의 무게.
#                  → 내리는 규칙은 측정 구간을 1.5초로 줄여 옮겨도 그 조건에 드는 고장 파일이 1개 이하여야 한다
# - 임계값 정책 (결과를 보기 전에 고정): 지금 = 안쪽 4-fold OOF(seed 1개)에서 F2 최대인 값.
#   대안 = 안쪽 분할 seed 5개로 각각 구한 값의 중앙값. 중첩 F2 차이가 sd 안이면 seed에 기대지 않는 중앙값을 최종에 쓴다
# - 구성 (모두 같은 분할)
#   L0 AE 단독 · L1 ET 단독 · L2 AE+ET 로지스틱(EXP-014) · L3 + 8.3k 규칙(EXP-021) · A + 유격 반복 규칙(EXP-025)
#   B = A + 단일 타격 감점(EXP-029 구성 3) · C = A + 주기 잠금 규칙(EXP-029 구성 6) · D = A + 둘 다
#   R = 규칙만 (8.3k 또는 유격이 켜지면 고장, 학습 모델 없음)
# - 이 PC는 캐시가 없어 처음부터 다시 계산한다 → L2·L3·A·B·C가 기록(0.857·0.892·0.942·0.954·0.966)과 맞는지가 재현 확인
# 실행: conda run -n qe-phm python "단계/5_모델비교/스크리닝/EXP-030_1116_최종후보고정_규칙감사/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, _oof_scores, _scores, best_f2_threshold, _collect
from src.experts import (RuleOverride, Rule83, KnockRule, LockRule, SingleHitCap, ScoreStack, ae_mean, et_logit,
                         ae_segerr, et_trees, warm_cache, peak83_table, _knock, _hits)
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from scipy.signal import butter, sosfiltfilt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN, FS = "#2a78d6", "#eb6834", 25000
ex = Experiment(__file__, "EXP-030", "최종 후보 고정 — 규칙 감사와 임계값 정책", ["H1", "H2", "H4"],
                member="이동훈", model="ET + 스펙트럼 AE → 로지스틱 + 채택 기준을 통과한 규칙",
                changes="새 피처 없음. 규칙 4개(8.3k · 유격 반복 · 주기 잠금 · 단일 타격 감점)를 지지 수·문턱 ±20%·측정 구간 민감도·오류 방향으로 감사해 최종 구성 고정",
                focus=("abn_022", "abn_005", "nor_025", "nor_116", "nor_055"))
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]; nor = y == 0
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
LOG = os.path.join(ex.figs, "audit.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


def who(mask):
    return " ".join(short(names[i]) for i in np.flatnonzero(mask)) or "없음"


FACT = (0.8, 0.9, 1.0, 1.1, 1.2)                               # 문턱 ±20% 격자

# %% 1. 규칙 감사 ① 8.3 kHz 모드 규칙 (올리는 규칙, H2′)
P83, W83, _ = peak83_table()
fire83 = (P83 >= 0.75) & (W83 >= 150)
grid83 = [(a, b, (P83 >= 0.75 * a) & (W83 >= 150 * b)) for a in FACT for b in FACT]
nor83 = np.any([m & nor for _, _, m in grid83], 0)
log("[8.3k 규칙] 기준 문턱에서 켜짐: 고장", who(fire83 & ~nor), "/ 정상", who(fire83 & nor))
log("  ±20% 25개 조합: 고장 수", min(int((m & ~nor).sum()) for *_, m in grid83), "–", max(int((m & ~nor).sum()) for *_, m in grid83),
    "· 한 번이라도 켜진 정상:", who(nor83))
near = np.argsort(-(np.minimum(P83 / 0.75, W83 / 150)) * nor)[:3]
log("  문턱에 가장 가까운 정상 (돌출 dec, 폭 Hz):", " · ".join(f"{short(names[i])} {P83[i]:.2f}, {W83[i]:.0f}" for i in near))

# %% 1. 규칙 감사 ② 유격 반복 규칙 (올리는 규칙, H1′)
kn = KnockRule(2.0, 0.7).fit(g)
r_all, K3 = kn.r(g), _knock()[:, 2]
fireK = (r_all >= 2.0) & (K3 >= 0.7)
gridK = [(a, b, (r_all >= 2.0 * a) & (K3 >= 0.7 * b)) for a in FACT for b in FACT]
norK = np.any([m & nor for _, _, m in gridK], 0)
log("[유격 반복 규칙] 기준 문턱에서 켜짐: 고장", who(fireK & ~nor), "/ 정상", who(fireK & nor))
log("  ±20% 25개 조합: 고장 수", min(int((m & ~nor).sum()) for *_, m in gridK), "–", max(int((m & ~nor).sum()) for *_, m in gridK),
    "· 한 번이라도 켜진 정상:", who(norK))
for a, b, m in gridK:
    if (m & nor).any(): log(f"    r ≥ {2.0 * a:.1f} · 2위/1위 ≥ {0.7 * b:.2f} → 정상 {who(m & nor)}")
near = np.argsort(-(np.minimum(r_all / 2.0, K3 / 0.7)) * nor)[:4]
log("  문턱에 가장 가까운 정상 (r, 2위/1위):", " · ".join(f"{short(names[i])} {r_all[i]:.2f}, {K3[i]:.2f}" for i in near))

# %% 1. 규칙 감사 ③ 주기 잠금 규칙 (올리는 규칙, H4)
S = _hits()["LOCK_S"].values
th_lock = S[nor].max() + 0.1
fireL = S >= th_lock
gridL = [(a, S >= th_lock * a) for a in FACT]
norL = np.any([m & nor for _, m in gridL], 0)
log(f"[주기 잠금 규칙] 문턱 {th_lock:.3f} (정상 최대 {S[nor].max():.3f} + 0.1) 켜짐: 고장", who(fireL & ~nor), "/ 정상", who(fireL & nor))
log("  ±20% 5개 문턱: 고장 수", min(int((m & ~nor).sum()) for _, m in gridL), "–", max(int((m & ~nor).sum()) for _, m in gridL),
    "· 한 번이라도 켜진 정상:", who(norL))
log("  S 상위: 고장", " · ".join(f"{short(names[i])} {S[i]:.2f}" for i in np.argsort(-S * ~nor)[:4]),
    "/ 정상", " · ".join(f"{short(names[i])} {S[i]:.2f}" for i in np.argsort(-S * nor)[:4]))
lockfold = {}
for s in range(5):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g):
        for i in te[LockRule(0.1).fit(tr)(te)]: lockfold[names[i]] = lockfold.get(names[i], 0) + 1
log("  폴드 안 문턱(seed 5 × 바깥 5)에서 켜진 횟수/5:", " ".join(f"{short(k)}({v})" for k, v in sorted(lockfold.items())))

# %% 1. 규칙 감사 ④ 단일 타격 감점 (내리는 규칙, H1′) — 문턱 격자 + 측정 구간 민감도
Hh = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=-1)


def big_hits(h, th=20.0):
    """src/features.py extract_features_hits와 같은 정의: σ = 1.4826·MAD, 사건 = |h| > 8σ(10 ms 안은 하나), 높이 ≥ th σ 개수"""
    sig = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if not len(idx): return 0
    gs = np.split(idx, np.flatnonzero(np.diff(idx) > int(0.010 * FS)) + 1)
    return int((np.array([np.abs(h[q]).max() for q in gs]) / sig >= th).sum())


n20 = np.array([big_hits(h) for h in Hh])
assert (n20 == _hits()["HIT_n20"].values).all(), "큰 타격 수가 피처 표와 다르다"
log("[단일 타격 감점] 큰 타격(≥ 20σ) 수 0 / 1 / 2 / 3 / 4+:",
    "정상", np.bincount(np.minimum(n20[nor], 4), minlength=5).tolist(), "· 고장", np.bincount(np.minimum(n20[~nor], 4), minlength=5).tolist())
one_fault_grid = {}
for th in (16, 18, 20, 22, 24):                                # 20σ ±20%
    c = np.array([big_hits(h, th) for h in Hh]); one_fault_grid[th] = (c == 1) & ~nor
    log(f"  문턱 {th}σ: '정확히 1개' 정상 {int(((c == 1) & nor).sum())} · 고장 {int(((c == 1) & ~nor).sum())} ({who((c == 1) & ~nor)})")
WIN = {"1.5초 (시작 0 · 0.25 · 0.5 s)": (37500, (0, 6250, 12500)), "1.0초 (시작 0 · 0.25 · 0.5 · 0.75 · 1.0 s)": (25000, (0, 6250, 12500, 18750, 25000))}
win_one = {}
for lab, (L, starts) in WIN.items():
    C = np.array([[big_hits(h[s:s + L]) for s in starts] for h in Hh])            # (파일, 창)
    any1 = (C == 1).any(1); win_one[lab] = any1
    log(f"  측정 구간 {lab}: 창 하나라도 '정확히 1개'인 파일 — 고장 {int((any1 & ~nor).sum())}/30 ({who(any1 & ~nor)}) · 정상 {int((any1 & nor).sum())}/120")
    log(f"    창 단위 비율: 고장 {(C[~nor] == 1).mean():.2f} · 정상 {(C[nor] == 1).mean():.2f}")
log("  0/30 관측일 때 실제 비율의 95% 상한 = 1 − 0.05^(1/30) =", round(1 - 0.05 ** (1 / 30), 3))

# %% 2. 채택 판정 (위 기준 C2–C4를 코드로)
any15 = win_one["1.5초 (시작 0 · 0.25 · 0.5 s)"]
CRIT = {
    "8.3 kHz 모드 규칙": dict(방향="올림", C2=int((fire83 & ~nor).sum()), C3=not nor83.any(), C4=True),
    "유격 반복 규칙": dict(방향="올림", C2=int((fireK & ~nor).sum()), C3=not norK.any(), C4=True),
    "주기 잠금 규칙": dict(방향="올림", C2=int((fireL & ~nor).sum()), C3=not norL.any(), C4=True),
    "단일 타격 감점": dict(방향="내림", C2=int(((n20 == 1) & nor).sum()), C3=not np.any(list(one_fault_grid.values())),
                     C4=int((any15 & ~nor).sum()) <= 1),
}
# [11-16 첫 실행의 감사 결과를 본 뒤 덧붙인 주석 — 선택 로직은 그대로다]
# 아래 FINAL 표는 처음부터 '결정 대기'였던 둘(단일 타격 감점·주기 잠금)만 고르게 짰다. 8.3k·유격 규칙은 11-14에
# 사용자가 '사후 문턱 예외'로 채택한 결정(단계/1_기준·기록/결정.md)이라 여기서 빼지 않는다. 그런데 머리말 기준은 규칙 4개 모두에
# 적용되는 것처럼 썼고, 두 기존 규칙도 C3(±20%)에서 탈락했다. 이 탈락은 숨기지 않고 '경고'로 기록한다 (README 해석 참고)
DECIDED = ("8.3 kHz 모드 규칙", "유격 반복 규칙")
for k, v in CRIT.items():
    v["채택"] = bool(v["C2"] >= 3 and v["C3"] and v["C4"])
    verdict = ("기준 통과" if v["채택"] else "기준 미달 → 경고 (11-14 채택 결정은 유지, 사용자 재확인 대상)") if k in DECIDED else ("채택" if v["채택"] else "제외")
    log(f"[판정] {k}: 방향 {v['방향']} · C2 지지 {v['C2']}개 {'통과' if v['C2'] >= 3 else '탈락'} · C3 ±20% {'통과' if v['C3'] else '탈락'} · "
        f"C4 {'통과' if v['C4'] else '탈락'} → {verdict}")
# 문턱을 10%만 흔들었을 때 (보고용): 가까운 쪽 3 × 3 조합
g10 = lambda grid: np.any([m & nor for a, b, m in grid if a in (0.9, 1.0, 1.1) and b in (0.9, 1.0, 1.1)], 0)
log("[보고] 문턱 ±10%에서 켜지는 정상: 8.3k", who(g10(grid83)), "· 유격", who(g10(gridK)))
ex.note("규칙 채택 판정 (C2 지지 수, C3 ±20%, C4 오류 방향)", "\n".join(f"{k}: {v}" for k, v in CRIT.items()))
USE_CAP, USE_LOCK = CRIT["단일 타격 감점"]["채택"], CRIT["주기 잠금 규칙"]["채택"]

# %% 3. 구성
R83, KN, LK = Rule83(0.75, 150), KnockRule(2.0, 0.7), LockRule(0.1)
B0 = dict(experts=(ae_mean, et_logit), combine="lr")
CONFIGS = {
    "L0 AE 단독": lambda: ScoreStack(experts=(ae_mean,), combine="col0"),
    "L1 ET 단독": lambda: ScoreStack(experts=(et_logit,), combine="col0"),
    "L2 AE+ET 로지스틱 (EXP-014)": lambda: ScoreStack(**B0),
    "L3 + 8.3k 규칙 (EXP-021)": lambda: RuleOverride(base_kw=B0, rule=[R83]),
    "A + 유격 반복 규칙 (EXP-025)": lambda: RuleOverride(base_kw=B0, rule=[R83, KN]),
    "B A + 단일 타격 감점": lambda: SingleHitCap(base_kw=B0, rule=[R83, KN]),
    "C A + 주기 잠금 규칙": lambda: RuleOverride(base_kw=B0, rule=[R83, KN, LK]),
    "D A + 감점 + 주기 잠금": lambda: SingleHitCap(base_kw=B0, rule=[R83, KN, LK]),
}
FINAL = {(False, False): "A + 유격 반복 규칙 (EXP-025)", (True, False): "B A + 단일 타격 감점",
         (False, True): "C A + 주기 잠금 규칙", (True, True): "D A + 감점 + 주기 잠금"}[(USE_CAP, USE_LOCK)]
log("[최종 구성]", FINAL)
warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
log("캐시 준비 끝")

# %% 4. 중첩 CV F2 (같은 분할)
res = {}
for nm in [FINAL] + [k for k in CONFIGS if k != FINAL]:
    res[nm] = nested_cv(F, y, g, CONFIGS[nm], n_jobs=-1)
    ex.record("중첩: " + nm, res[nm], names, main=(nm == FINAL))
    log(f"{nm:32s} F2 {res[nm]['f2'].mean():.3f} ± {res[nm]['f2'].std():.3f} · 재현율 {res[nm]['recall'].mean():.3f} · 오경보율 {res[nm]['fpr'].mean():.3f}")

# 규칙만 (학습 모델 없음): 8.3k 또는 유격이 켜지면 고장. 유격 규칙의 정상 중심은 폴드 안 학습 정상으로
f2r, rcr, fpr_ = [], [], []
for s in range(5):
    pred = np.zeros(len(y), int)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g):
        pred[te] = R83(te) | KnockRule(2.0, 0.7).fit(tr)(te)
    f2r.append(fbeta_score(y, pred, beta=2)); rcr.append(pred[~nor].mean()); fpr_.append(pred[nor].mean())
log(f"R 규칙만 (8.3k 또는 유격)          F2 {np.mean(f2r):.3f} ± {np.std(f2r):.3f} · 재현율 {np.mean(rcr):.3f} · 오경보율 {np.mean(fpr_):.3f}  (전부 고장 = 0.556)")
ex.note("규칙만 (학습 모델 없음)", f"F2 {np.mean(f2r):.3f} ± {np.std(f2r):.3f} · 재현율 {np.mean(rcr):.3f} · 오경보율 {np.mean(fpr_):.3f}")


# %% 5. 임계값 정책: 안쪽 seed 1개 vs 3개 중앙값 (최종 구성)
# (머리말에는 5개로 적었으나 계산 시간(AE 학습 약 2,000번 추가) 때문에 결과를 보기 전에 3개로 줄였다: 바깥 seed s → 안쪽 seed s, s+1, s+2)
def _multi(seed, tr, te, make):
    ts = [best_f2_threshold(y[tr], _oof_scores(F[tr], y[tr], g[tr], make, StratifiedGroupKFold(4, shuffle=True, random_state=k)))
          for k in (seed, (seed + 1) % 5, (seed + 2) % 5)]            # ts[0] = nested_cv와 같은 분할
    return seed, te, _scores(make().fit(F[tr], y[tr]), F[te]), ts


jobs = [(s, tr, te, CONFIGS[FINAL]) for s in range(5) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
with parallel_config(backend="loky", inner_max_num_threads=1):
    out = Parallel(n_jobs=-1)(delayed(_multi)(*j) for j in jobs)
single = _collect([(te, (sc >= ts[0]).astype(int), ts[0]) for s, te, sc, ts in out], y, range(5), 5)
median = _collect([(te, (sc >= np.median(ts)).astype(int), float(np.median(ts))) for s, te, sc, ts in out], y, range(5), 5)
assert np.allclose(single["f2"], res[FINAL]["f2"]), "seed 1개 정책이 nested_cv와 다르다"
ex.record("중첩: 최종 구성 · 임계값 = 안쪽 seed 3개 중앙값", median, names)
allts = np.array([ts for *_, ts in out])                       # (바깥 fold 25, 안쪽 seed 3)
log(f"[임계값 정책] seed 1개 F2 {single['f2'].mean():.3f} ± {single['f2'].std():.3f} (오경보율 {single['fpr'].mean():.3f}) → "
    f"3개 중앙값 F2 {median['f2'].mean():.3f} ± {median['f2'].std():.3f} (재현율 {median['recall'].mean():.3f} · 오경보율 {median['fpr'].mean():.3f})")
log(f"  바깥 fold 25개의 임계값 범위: seed 1개 {single['thresholds'].min():.2f}–{single['thresholds'].max():.2f} (sd {single['thresholds'].std():.2f}) · "
    f"중앙값 {median['thresholds'].min():.2f}–{median['thresholds'].max():.2f} (sd {median['thresholds'].std():.2f})")
log(f"  같은 학습 파일에서 안쪽 seed만 바꿨을 때 임계값 흔들림(fold 안 sd의 평균): {allts.std(1).mean():.2f}")
USE_MEDIAN = bool(median["f2"].mean() >= single["f2"].mean() - single["f2"].std())
log("[임계값 정책 판정]", "중앙값 채택 (차이가 sd 안 또는 개선)" if USE_MEDIAN else "seed 1개 유지 (중앙값이 sd 넘게 낮음)")
ex.note("임계값 정책", f"seed 1개 {single['f2'].mean():.3f} ± {single['f2'].std():.3f} · 3개 중앙값 {median['f2'].mean():.3f} ± {median['f2'].std():.3f} · "
                   f"채택 = {'중앙값' if USE_MEDIAN else 'seed 1개'}")
ex.note("최종 구성", FINAL)

# %% 그림 1 (핵심): 규칙 감사 — 문턱과 ±20% 띠, 가장 가까운 반대쪽 파일
fig, axs = plt.subplots(2, 2, figsize=(16, 12))
a = axs[0, 0]
for lab, m, c, mk in (("정상", nor, NOR, "s"), ("고장", ~nor, ABN, "o")):
    a.scatter(P83[m], np.minimum(W83[m], 600), color=c, s=26, alpha=.7, lw=0, marker=mk, label=lab)
a.add_patch(plt.Rectangle((0.75 * .8, 150 * .8), 0.75 * .4, 150 * .4, color="#999", alpha=.25, lw=0, label="문턱 ±20%"))
a.axvline(0.75, color="k", lw=1); a.axhline(150, color="k", lw=1)
for i in np.flatnonzero(fire83 | ((P83 >= 0.5) & nor)): a.annotate(short(names[i]), (P83[i], min(W83[i], 600)), xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
a.set_xlabel("8.20–8.45 kHz 봉우리 돌출 (dec)"); a.set_ylabel("봉우리 반높이 폭 (Hz, 600에서 자름)")
a.set_title(f"① 8.3 kHz 모드 규칙 (오른쪽 위 = 고장): 고장 {int((fire83 & ~nor).sum())} · 정상 {int((fire83 & nor).sum())}, ±20%에서 정상 {int(nor83.sum())}", fontsize=10); a.legend(fontsize=8); a.grid(alpha=.3)
a = axs[0, 1]
for lab, m, c, mk in (("정상", nor, NOR, "s"), ("고장", ~nor, ABN, "o")):
    a.scatter(np.minimum(r_all[m], 8), K3[m], color=c, s=26, alpha=.7, lw=0, marker=mk, label=lab)
a.add_patch(plt.Rectangle((2 * .8, .7 * .8), 2 * .4, .7 * .4, color="#999", alpha=.25, lw=0, label="문턱 ±20%"))
a.axvline(2, color="k", lw=1); a.axhline(.7, color="k", lw=1)
for i in np.flatnonzero((r_all >= 1.6) & (K3 >= .5)): a.annotate(short(names[i]), (min(r_all[i], 8), K3[i]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
a.set_xlabel("정상 중심 거리 r (8에서 자름)"); a.set_ylabel("2위/1위 타격 높이")
a.set_title(f"② 유격 반복 규칙 (오른쪽 위 = 고장): 고장 {int((fireK & ~nor).sum())} · 정상 {int((fireK & nor).sum())}, ±20%에서 정상 {int(norK.sum())}", fontsize=10); a.legend(fontsize=8); a.grid(alpha=.3)
a = axs[1, 0]
rng = np.random.default_rng(0)
a.scatter(rng.uniform(-.2, .2, nor.sum()), S[nor], s=16, color=NOR, alpha=.7, label="정상"); a.scatter(1 + rng.uniform(-.2, .2, (~nor).sum()), S[~nor], s=20, color=ABN, label="고장")
a.axhspan(th_lock * .8, th_lock * 1.2, color="#999", alpha=.25, lw=0, label="문턱 ±20%"); a.axhline(th_lock, color="k", lw=1)
for i in np.flatnonzero(S > .25): a.annotate(short(names[i]), (int(y[i]), S[i]), xytext=(10, 0), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
a.set_xticks([0, 1], ["정상", "고장"]); a.set_ylabel("주기 잠금 점수 S")
a.set_title(f"③ 주기 잠금 규칙 (선 위 = 고장): 고장 {int((fireL & ~nor).sum())} · 정상 {int((fireL & nor).sum())}, 폴드 안에서는 정상 {sum(k.startswith('nor') for k in lockfold)}개가 켜짐", fontsize=10); a.legend(fontsize=8); a.grid(alpha=.3)
a = axs[1, 1]
w, xs = .38, np.arange(5)
a.bar(xs - w / 2, np.bincount(np.minimum(n20[nor], 4), minlength=5) / nor.sum(), w, color=NOR, label="정상 (2초 전체)")
a.bar(xs + w / 2, np.bincount(np.minimum(n20[~nor], 4), minlength=5) / (~nor).sum(), w, color=ABN, label="고장 (2초 전체)")
a.axvspan(.5, 1.5, color="#999", alpha=.2, lw=0, label="감점 대상 = 정확히 1개")
a.set_xticks(xs, ["0", "1", "2", "3", "4+"]); a.set_xlabel("큰 타격(≥ 20σ) 수"); a.set_ylabel("파일 비율")
a.set_title(f"④ 단일 타격 감점 (1개면 정상 쪽으로 내림): 2초 전체에서 고장 0/30 · 1.5초 창으로 옮기면 고장 {int((any15 & ~nor).sum())}/30이 '1개'가 됨", fontsize=10); a.legend(fontsize=8); a.grid(alpha=.3, axis="y")
fig.tight_layout()
ex.save_fig(fig, "rule_audit", "규칙 4개의 문턱(검은 선)과 ±20% 띠(회색) 안에 반대쪽 파일이 들어오는가, 감점 조건이 측정 구간에 따라 고장에도 생기는가", key=True)

# %% 그림 2: logistic_plane — 최종 구성 (규칙으로 잡힌 고장은 뺌)
xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 400), np.linspace(-3, 4.5, 400)); G = np.c_[xx.ravel(), yy_.ravel()]
Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    Z[te] = ScoreStack(**B0).fit(F[tr], y[tr]).zscores(F[te])[:, :2]
    on = R83(te) | KnockRule(2.0, 0.7).fit(tr)(te)
    if USE_LOCK: on |= LockRule(0.1).fit(tr)(te)
    rule[te] = on
Zc = np.clip(Z, -3, 4.5)
D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(G).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(9.5, 9))
ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.6); ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.8, linestyles="--")
show = ~(rule & (y == 1))
for lab, sel, col, mk in (("정상", nor & show, NOR, "s"), ("고장 (규칙 밖)", ~nor & show, ABN, "o")):
    ax.scatter(Zc[sel, 0], Zc[sel, 1], color=col, s=28, lw=0, marker=mk, label=lab, alpha=.85)
if (rule & nor).any():
    ax.scatter(Zc[rule & nor, 0], Zc[rule & nor, 1], s=130, marker="D", facecolor="none", edgecolor="#7a3db8", lw=1.4, label="규칙이 켜진 정상 (오경보)")
c = ex.res["configs"]["중첩: " + FINAL]
wrong = [names.index(n) for n, k in {**c["miss_counts"], **c["fp_counts"]}.items() if k >= 2]
ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=90, marker="x", color="k", lw=1.8, label="자주 틀림 (5번 중 2번 이상)")
for i in wrong:
    k = c["miss_counts"].get(names[i]) or c["fp_counts"].get(names[i])
    ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {k}/5", Zc[i], xytext=(5, 4), textcoords="offset points", fontsize=8.5, color=ABN if y[i] else NOR, weight="bold")
gone = [short(names[i]) for i in np.flatnonzero(rule & (y == 1))]
ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (충격 피처 이상)")
ax.set_title(f"최종 구성: {FINAL}\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 규칙으로 잡혀 그림에서 뺀 고장 {len(gone)}개", fontsize=9.5)
ax.text(.98, .02, "뺀 고장 (규칙):\n" + " ".join(gone[:7]) + "\n" + " ".join(gone[7:]), transform=ax.transAxes, ha="right", va="bottom", fontsize=8, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "최종 구성의 AE z × ET z 평면(OOF seed 0). 규칙 밖 고장과 정상만 남김. 경계 근처에 남는 파일이 무엇인가")

# %% 그림 3: hard_files — 어려운 파일 위험도 (quick_cv seed 3), 열 = 후보 구성
HARD = ["abn_022", "abn_005", "abn_020", "abn_025", "abn_014", "abn_017", "abn_015", "abn_016", "abn_002",
        "nor_025", "nor_116", "nor_055", "nor_027", "nor_059", "nor_103", "nor_106", "nor_054", "nor_065"]
COLS = ["A + 유격 반복 규칙 (EXP-025)", "B A + 단일 타격 감점", "C A + 주기 잠금 규칙"]
M, oofs = [], {}
for nm in COLS:
    o = quick_cv(F, y, g, CONFIGS[nm], seeds=range(3))["oof"]; oofs[nm] = o
    pct = np.array([(o[:, nor] < o[:, [i]]).mean() for i in range(len(y))])
    M.append([1 - pct[names.index(h)] if h.startswith("abn") else pct[names.index(h)] for h in HARD])
M = np.array(M).T
fig, ax = plt.subplots(figsize=(8, 10))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h + f"  (큰 타격 {n20[names.index(h)]}개)" for h in HARD], fontsize=8)
ax.set_xticks(range(len(COLS)), ["A (EXP-025)", "B + 감점", "C + 주기 잠금"], fontsize=9)
ax.set_title("위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율\n(진할수록 틀리기 쉬움)", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "후보 A·B·C에서 어려운 파일의 위험도가 어떻게 달라지는가 (022, 단일 타격 정상 116·055, 103)")

# %% 그림 4: t-SNE (모든 파일)
E = cached_table("ext", feature_table_ext, X)
fig, _ = viz.tsne({"ext150 피처": E.values, "규칙 축 6개 (P83·W83·r·2위/1위·S·큰 타격 수)": np.column_stack([P83, W83, r_all, K3, S, np.log1p(n20)])},
                  y, names, mark=list(ex.focus) + ["abn_002", "abn_014", "abn_017", "nor_103"], score=oofs[COLS[0]][0].clip(-6, 6))
ex.save_fig(fig, "tsne", "규칙 축 공간에서 규칙이 잡는 고장이 따로 모이는가, 남은 오류 파일은 어디인가")

# %% 그림 5 (맨 끝): 남은 오류 파일 vs 정상 5–95%
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_missed", "최종 구성에서 남는 미탐 022·경계 005와 오경보 정상 025·116·055의 PSD가 정상 범위 밖으로 나가는 주파수가 있는가")
log("done")
