# %% [markdown]
# EXP-029 큰 타격 수 · 사건 울림 · 주기 잠금 — EXP-025에 2-14 · 2-15 · 2-13 정보를 더하면 ET와 최종 결과가 바뀌는가
# - 가설
#   H1′ (2-14): 고장은 바탕보다 훨씬 큰 타격이 반복된다. 큰 타격(≥ 20σ)이 딱 1개인 고장은 0/30, 정상은 38/120
#        → 지금 ET 입력(Peak·파고율·첨도)은 가장 큰 타격 하나가 값을 정해 단일 타격 정상(116 055 027)을 고장 쪽으로 민다
#   H1″ (2-15): 충격 수는 절대 크기(0.5 g)와 함께 볼 때만 의미, 정상 큰 사건은 높은 주파수(5–9.5 kHz)로 울리고 짧게 감쇠
#   H4  (2-13): 022 = 느린 하중 주기에 잠긴 충격 (주기 잠금 점수 S)
# - 새 피처 (src/features.py feature_table_hits, 라벨 무관)
#   HIT = [log1p(20σ 이상 타격 수), log10(4번째 타격/σ)] · EV = [log1p(0.5 g 이상 사건 수), 사건 주파수 중심 kHz, log10 감쇠 ms]
#   LOCK_S = 주기 잠금 점수 (2-13 기본 설정)
# - 구성 (모두 EXP-025 = AE + ET → 로지스틱 + 8.3k 규칙 + 유격 반복 규칙 위에서 하나씩)
#   0 기준 EXP-025
#   1 A-ET    : ET 입력 ext150 + HIT                  (큰 타격 수를 ET가 쓰게)
#   2 A-축    : 결합 로지스틱에 HIT 2축 추가
#   3 A-감점  : 큰 타격이 정확히 1개인 파일 점수를 학습 정상 OOF 중앙값 이하로 누름 (규칙 켜진 파일 제외)
#   4 B-ET    : ET 입력 ext150 + EV
#   5 B-축    : 결합 로지스틱에 EV 3축 추가
#   6 C-규칙  : 주기 잠금 S ≥ 학습 정상 최대 + 0.1 이면 무조건 고장 (폴드 안 문턱)
#   7 전부-ET : 1 + 4 + 3 + 6 (ET 입력 ext150 + HIT + EV, 단일 타격 감점, 주기 잠금 규칙)  ← 주 구성 (실행 전 지정)
#   8 전부-축 : 2 + 5 + 3 + 6
# - ET 단독 비교: ext150 / + HIT / + EV / + HIT + EV (quick_cv, 같은 ET 설정) + 새 열의 ET 중요도 순위
# - 주의: 20σ·0.5 g·k = 4는 2-14·2-15에서 데이터를 본 뒤 정한 값(라벨 섞음 없음, 하지만 선택 편향 가능).
#   LOCK_S는 022를 보고 만든 사후 정의(2-13). 감점 대상 "정확히 1개"는 2-14 표에서 고장 0개인 칸을 보고 정함 → 낙관적
# 실행: conda run -n qe-phm python "단계/5_모델실험/아카이브/EXP-029_1116_큰타격수·사건울림·주기잠금/run.py"

# %% 설정
import os, sys, warnings
from functools import partial
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from src.exp import *
from src.evaluate import quick_cv, separation
from src.experts import (RuleOverride, Rule83, KnockRule, LockRule, SingleHitCap, ScoreStack, ETLogitX,
                         ae_mean, et_logit, hit_axes, ev_axes, ae_segerr, et_trees, et_trees_x, warm_cache,
                         HIT_COLS, EV_COLS, _hits)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-029", "큰 타격 수 · 사건 울림 · 주기 잠금 — ET와 최종 결과 변화", ["H1", "H4"],
                member="이동훈", model="ET(+HIT·EV) + 스펙트럼 AE → 로지스틱 + 8.3k·유격·주기 잠금 규칙 + 단일 타격 감점",
                changes="EXP-025에 큰 타격 수(2-14)·사건 울림(2-15)·주기 잠금(2-13)을 ET 입력/결합 축/규칙으로 하나씩 추가",
                focus=("abn_022", "abn_020", "nor_025", "nor_116", "nor_055"))
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); nor = y == 0
SEEDS = range(3)
H = _hits(); H.index = names
E = cached_table("ext", feature_table_ext, X)
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")

R83, KN, LK = Rule83(0.75, 150), KnockRule(2.0, 0.7), LockRule(0.1)
ETH, ETE, ETHE = ETLogitX(HIT_COLS), ETLogitX(EV_COLS), ETLogitX(HIT_COLS + EV_COLS)
B0 = dict(experts=(ae_mean, et_logit), combine="lr")
CONFIGS = [
    ("0 기준 EXP-025", RuleOverride, dict(base_kw=B0, rule=[R83, KN])),
    ("1 A-ET 입력 + 큰 타격", RuleOverride, dict(base_kw=dict(experts=(ae_mean, ETH), combine="lr"), rule=[R83, KN])),
    ("2 A-축 큰 타격 2축", RuleOverride, dict(base_kw=dict(experts=(ae_mean, et_logit, hit_axes), combine="lr"), rule=[R83, KN])),
    ("3 A-감점 단일 타격", SingleHitCap, dict(base_kw=B0, rule=[R83, KN])),
    ("4 B-ET 입력 + 사건 울림", RuleOverride, dict(base_kw=dict(experts=(ae_mean, ETE), combine="lr"), rule=[R83, KN])),
    ("5 B-축 사건 울림 3축", RuleOverride, dict(base_kw=dict(experts=(ae_mean, et_logit, ev_axes), combine="lr"), rule=[R83, KN])),
    ("6 C-규칙 주기 잠금", RuleOverride, dict(base_kw=B0, rule=[R83, KN, LK])),
    ("7 전부-ET (주)", SingleHitCap, dict(base_kw=dict(experts=(ae_mean, ETHE), combine="lr"), rule=[R83, KN, LK])),
    ("8 전부-축", SingleHitCap, dict(base_kw=dict(experts=(ae_mean, et_logit, hit_axes, ev_axes), combine="lr"), rule=[R83, KN, LK])),
]
MAIN = CONFIGS[7][0]
make = {n: (lambda C=C, kw=kw: C(**kw)) for n, C, kw in CONFIGS}
ET_FNS = [et_trees, ae_segerr] + [partial(et_trees_x, extra=e) for e in (HIT_COLS, EV_COLS, HIT_COLS + EV_COLS)]
warm_cache(ET_FNS, seeds=SEEDS)
print("warm done", flush=True)

# %% 1. 큰 타격 수 표 (2-14 확인) + 큰 타격 0개 고장
n20 = H.HIT_n20.values
tab = ["큰 타격(≥ 20σ) 수: 0개 / 1개 / 2–3개 / 4개+"]
for lab, m in (("정상", nor), ("고장", ~nor)):
    c = n20[m]; tab.append(f"{lab}: {(c==0).sum()} / {(c==1).sum()} / {((c>=2)&(c<=3)).sum()} / {(c>=4).sum()}")
tab.append("큰 타격 1개 정상: " + " ".join(short(names[i]) for i in np.flatnonzero(nor & (n20 == 1))))
tab.append("큰 타격 0개 고장: " + " ".join(short(names[i]) for i in np.flatnonzero(~nor & (n20 == 0))))
ex.note("큰 타격 수 분포", "\n".join(tab))

# %% 2. ET 단독: 입력에 새 열을 넣으면 ET가 달라지는가
EV_ = {"ext150": E.values,
       "ext150 + 큰 타격": np.hstack([E.values, H[list(HIT_COLS)].values]),
       "ext150 + 사건 울림": np.hstack([E.values, H[list(EV_COLS)].values]),
       "ext150 + 큰 타격 + 사건 울림": np.hstack([E.values, H[list(HIT_COLS + EV_COLS)].values])}
rows, et_oof = [], {}
for nm, F in EV_.items():
    q = quick_cv(F, y, g, make_et(), seeds=SEEDS); s = separation(q["oof"], y, names); et_oof[nm] = q["oof"]
    rows.append(f"{nm:28s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 확실한 고장 {s['sure_a']} · "
                f"회색 고장 {' '.join(x[4:] for x in s['gray_a'])} · 위험 정상 {s['risk_n']:.1f} · "
                f"오경보 정상(116 055 027 025) 평균 순위 {np.mean([(q['oof'][:, nor] < q['oof'][:, [names.index(n)]]).mean() for n in ('nor_116','nor_055','nor_027','nor_025')]):.2f}")
    print(rows[-1], flush=True)
ex.note("ET 단독 quick_cv (seed 3)", "\n".join(rows))
# 새 열의 ET 중요도 (150개 전체 학습, 설명용)
Fall = EV_["ext150 + 큰 타격 + 사건 울림"]; cols = list(E.columns) + list(HIT_COLS + EV_COLS)
imp = make_et()().fit(Fall, y).feature_importances_; rk = (-imp).argsort().argsort() + 1
ex.note("ET 중요도 순위 (155개 중, 전체 학습)", " · ".join(f"{c} {rk[cols.index(c)]}위" for c in HIT_COLS + EV_COLS)
        + " | 상위 10: " + ", ".join(cols[j] for j in (-imp).argsort()[:10]))

# %% 3. quick_cv 분리 판정 (최종 구성)
qrows, q_oof = [], {}
for nm, C, kw in CONFIGS:
    q = quick_cv(g[:, None], y, g, make[nm], seeds=SEEDS); s = separation(q["oof"], y, names); q_oof[nm] = q["oof"]
    qrows.append(f"{nm:26s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 확실한 고장 {s['sure_a']} · "
                 f"회색 고장 {' '.join(x[4:] for x in s['gray_a'])} · 회색 정상 {len(s['gray_n'])} · 위험 정상 {s['risk_n']:.1f}")
    print(qrows[-1], flush=True)
ex.note("quick_cv 분리 판정 (seed 3)", "\n".join(qrows))

# %% 4. 폴드 안에서 규칙·감점이 켜진 파일 (seed 5 × 바깥 5-fold)
lock, cap = {}, {}
for s in range(5):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g):
        for i in te[LockRule(0.1).fit(tr)(te)]: lock[names[i]] = lock.get(names[i], 0) + 1
for i in np.flatnonzero(n20 == 1): cap[names[i]] = "정상" if nor[i] else "고장"
ex.note("주기 잠금 규칙이 켜진 파일 (횟수/5)", " ".join(f"{short(k)}({v})" for k, v in sorted(lock.items())) or "없음")
ex.note("단일 타격 감점 대상 (큰 타격 정확히 1개)", f"정상 {sum(v == '정상' for v in cap.values())}개 · 고장 {sum(v == '고장' for v in cap.values())}개")

# %% 그림 1 (핵심): 로지스틱 구분선 + 큰 타격 수 — x = 기준 결합 결정값(OOF seed 0), y = 큰 타격 수
D0 = np.empty(len(y))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    D0[te] = ScoreStack(**B0).fit(g[tr][:, None], y[tr]).decision_function(g[te][:, None])
jit = np.random.default_rng(0).uniform(-.25, .25, len(y))
fig, ax = plt.subplots(figsize=(10.5, 7))
for lab, s, c, mk in (("정상", nor, NOR, "s"), ("고장", ~nor, ABN, "o")):
    ax.scatter(np.clip(D0[s], -8, 8), np.minimum(n20[s], 8) + jit[s], color=c, s=26, alpha=.7, lw=0, marker=mk, label=lab)
one = n20 == 1
ax.scatter(np.clip(D0[one], -8, 8), 1 + jit[one], facecolor="none", edgecolor="k", lw=1, s=60, label="단일 타격 감점 대상 (큰 타격 1개)")
ax.axvline(0, color="k", lw=1.2, label="로지스틱 구분선 (결정값 0)")
ax.axhspan(.6, 1.4, color="#999", alpha=.12)
for i in np.flatnonzero(((D0 > -2) & (n20 <= 1)) | (~nor & (n20 == 0))):
    ax.annotate(short(names[i]), (np.clip(D0[i], -8, 8), min(n20[i], 8) + jit[i]), xytext=(4, 3), textcoords="offset points",
                fontsize=7.5, color=ABN if y[i] else NOR)
ax.set_xlim(-8, 8); ax.set_yticks(range(9), [str(k) for k in range(8)] + ["8+"]); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("기준 결합(AE + ET 로지스틱) 결정값, OOF seed 0"); ax.set_ylabel("큰 타격(≥ 20σ) 수 (세로 흔들림 = 겹침 방지)")
ax.set_title("구분선 오른쪽(고장 판정)에 있는 '큰 타격 1개' 파일은 모두 정상인가 — 감점이 겨냥하는 자리", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "boundary_lr", "구분선 오른쪽의 큰 타격 1개 파일(회색 띠)이 정상뿐인가, 큰 타격 0개 고장(002 015 016)은 어디에 있나", key=True)

# %% 그림 2: 점수 이동 — 기준 vs 주 구성 (quick_cv OOF 평균 순위)
def rank_n(o):  # 파일별 '정상 중 그 파일보다 낮은 비율' (seed 평균)
    return np.array([(o[:, nor] < o[:, [i]]).mean() for i in range(len(y))])
r0, r7 = rank_n(q_oof[CONFIGS[0][0]]), rank_n(q_oof[MAIN])
fig, ax = plt.subplots(figsize=(8.5, 8))
for lab, s, c, mk in (("정상", nor, NOR, "s"), ("고장", ~nor, ABN, "o")):
    ax.scatter(r0[s], r7[s], color=c, s=24 + 12 * np.minimum(n20[s], 6), alpha=.65, lw=0, marker=mk, label=lab)
ax.plot([0, 1], [0, 1], color="k", lw=.8)
for i in np.flatnonzero((np.abs(r7 - r0) > .08) | (nor & (r0 > .9)) | (~nor & (r0 < .97))):
    ax.annotate(short(names[i]), (r0[i], r7[i]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR)
ax.set_xlabel("기준 EXP-025: 정상 중 이 파일보다 낮은 비율 (OOF)"); ax.set_ylabel("주 구성 7: 같은 값")
ax.set_title("대각선 아래 = 주 구성에서 덜 이상해짐. 점 크기 = 큰 타격 수", fontsize=10); ax.grid(alpha=.3); ax.legend(fontsize=8)
fig.tight_layout()
ex.save_fig(fig, "score_shift", "오경보 정상(025 116 055)이 대각선 아래로 내려가는가, 고장이 내려가지 않는가")

# %% 그림 3: 새 축 지도 — 큰 타격 4위 × 사건 주파수 중심, 색 = 정상/고장, 테두리 = 주기 잠금 S > 정상 최대
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
a = axs[0]
for lab, s, c, mk in (("정상", nor, NOR, "s"), ("고장", ~nor, ABN, "o")):
    a.scatter(H.HIT_k4[s], H.EV_fc_khz[s], color=c, s=10 + 15 * H.EV_n05g_log[s] ** 2 * 4, alpha=.65, lw=0, marker=mk, label=lab)
for n in ("abn_005", "abn_008", "abn_010", "abn_014", "abn_022", "abn_020", "abn_018", "nor_001", "nor_025", "nor_116", "nor_055", "nor_086"):
    a.annotate(short(n), (H.HIT_k4[n], H.EV_fc_khz[n]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=ABN if n[0] == "a" else NOR)
a.axvline(H.HIT_k4[nor].max(), color="k", lw=.8, ls="--", label="정상 최대 (4번째 타격)")
a.set_xlabel("log10(4번째 큰 타격 / σ)"); a.set_ylabel("사건 주파수 중심 (kHz). 점 크기 = 0.5 g 이상 사건 수")
a.set_title("오른쪽 = 큰 타격 반복(2-14), 아래 = 낮은 주파수로 울림(2-15)", fontsize=10); a.grid(alpha=.3); a.legend(fontsize=8)
a = axs[1]
v = H.LOCK_S.values; rng = np.random.default_rng(1)
a.scatter(rng.uniform(-.2, .2, nor.sum()), v[nor], s=14, color=NOR, alpha=.7, label="정상"); a.scatter(1 + rng.uniform(-.2, .2, (~nor).sum()), v[~nor], s=18, color=ABN, label="고장")
a.axhline(v[nor].max() + .1, color="k", lw=.8, ls="--", label="정상 최대 + 0.1 (150개 기준, 설명용)")
for i in np.flatnonzero(v > .25): a.annotate(short(names[i]), (int(y[i]), v[i]), xytext=(8, 0), textcoords="offset points", fontsize=7.5)
a.set_xticks([0, 1], ["정상", "고장"]); a.set_ylabel("주기 잠금 점수 S (2-13)"); a.grid(alpha=.3); a.legend(fontsize=8)
a.set_title("주기 잠금 규칙: 정상 최대 + 0.1 위는 022뿐인가", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "new_axes", "새 축에서 고장·정상이 어디에 있나 (설명용, 150개 전체 기준)")

# %% 그림 4: t-SNE
fig, _ = viz.tsne({"ext150": E.values, "ext150 + 큰 타격 + 사건 울림": Fall,
                   "새 6열 (HIT·EV·LOCK)": H[list(HIT_COLS + EV_COLS) + ["LOCK_S"]].values},
                  y, names, mark=list(ex.focus) + ["abn_005", "abn_002", "abn_015", "abn_016"], score=q_oof[MAIN][0])
ex.save_fig(fig, "tsne", "새 열을 넣으면 고장이 더 모이는가, 오경보 정상·놓친 022는 어디인가")

# %% 중첩 CV F2 (모든 구성, 같은 분할)
warm_cache(ET_FNS, seeds=range(5), nested=True)
print("nested warm done", flush=True)
for nm, C, kw in [CONFIGS[7], CONFIGS[0]] + [c for c in CONFIGS if c[0] not in (MAIN, CONFIGS[0][0])]:
    r_ = nested_cv(g[:, None], y, g, make[nm], n_jobs=-1)
    ex.record("중첩: " + nm, r_, names, main=(nm == MAIN))

# %% 그림 5 (맨 끝): 놓친 파일 · 오경보 정상 vs 정상 5–95%
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_missed",
            "놓친 022·회색 020과 오경보 정상 025·116·055의 PSD가 정상 범위 밖으로 나가는 주파수가 있는가")
print("done", flush=True)
