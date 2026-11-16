# %% [markdown]
# EXP-016 회색지대용 새 정보 축 — AE 변형(A·C) · 비대칭 결합(D) · 충격 주기성(B) · 단발 사건(E)
# - 배경: EXP-014 로지스틱 결합 z(AE, ET)가 기준 (중첩 F2 0.857). EXP-015에서 SVM 경계로는 회색지대가 안 풀림 → 정보 문제
# - 1순위 놓친 고장, 2순위 오경보 정상. 처방은 파일 이름에서 출발했지만 정의는 라벨 없이 고정, 평가는 150개 전체
#   A (H1, 014·017): AE 구간 8개 오차를 평균 대신 최댓값 — 2초 중 1–2번 나는 충격이 평균에 묻히지 않게
#   C (H2·H3, 002·029 유지, 059·062 제거): 구간마다 평균 log PSD를 빼고 학습한 '모양' AE — 전체 높이(게인) 제거
#   D (017): 두 z 중 큰 값을 3번째 입력으로 — 한쪽 증거가 아주 강하면 고장 (파라미터 +1)
#   B (H4·H1, 022·005): 포락선 주기성, 충격 간격 변동계수, 저주파(<20 Hz) 정점 동기, 저주파 비율
#   E (H5, 오경보 025·055·116·086): 사건 수, 두 번째/첫 번째 사건 높이, 큰 사건 잔향 비(2-6 사후 정의), 큰 사건 높이
# - 평가: 탐색이라 quick_cv(5-fold × seed 3, PR-AUC). 기준보다 sd 넘게 좋은 구성만 중첩 CV
# - 결합기는 모두 EXP-014와 같은 L2 로지스틱(C=1, 균형). 입력만 바꾼다
# 실행: conda run -n qe-phm python 단계/5_모델비교/기각/EXP-016_1114_회색지대축/run.py

# %% 설정
import os, sys, warnings, functools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..", "..")))
from src.exp import *
from src.evaluate import quick_cv
from src.features import feature_table_impact
from src.experts import (ScoreStack, ae_mean, ae_max, ae_shape_mean, ae_shape_max, ae_bins, et_logit, et_trees,
                         ae_segerr, imp_period, imp_single, warm_cache, LP)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-016", "회색지대용 새 정보 축 — AE 변형 · 비대칭 결합 · 충격 주기성 · 단발 사건",
                ["H1", "H2", "H3", "H4", "H5"], member="이동훈", model="ET + AE 변형 + 충격 피처 → 정상 기준 z → L2 로지스틱",
                changes="EXP-014 결합의 입력에 AE 집계·모양 변형, max 열, 충격 주기성(B)·단발 사건(E) 피처를 하나씩 추가", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
IMP = cached_table("impact", feature_table_impact, X)

MISS = ["abn_002", "abn_005", "abn_014", "abn_017", "abn_022", "abn_029"]
FP = ["nor_008", "nor_025", "nor_027", "nor_039", "nor_050", "nor_054", "nor_055", "nor_059", "nor_062", "nor_065",
      "nor_073", "nor_086", "nor_095", "nor_106", "nor_116"]
HARD = MISS + FP

n = warm_cache([ae_segerr, functools.partial(ae_segerr, shape=True), et_trees], seeds=SEEDS)
print(f"캐시 완료: 분할 {n}개", flush=True)


# %% 그림 1 (먼저 보는 EDA 확인): 새 충격 피처 8개 — 정상 5–95% 띠 위에 고장, 어려운 파일 표시
cols = list(IMP.columns.drop("B2_env_acf_lag_s"))
fig, axs = plt.subplots(2, 4, figsize=(18, 8))
rj = np.random.default_rng(0)
for a, c in zip(axs.ravel(), cols):
    v = IMP[c].values
    lo, hi = np.percentile(v[y == 0], [5, 95])
    a.axhspan(lo, hi, color=NOR, alpha=.12)
    xs = np.where(y == 1, 1, 0) + rj.uniform(-.18, .18, len(y))
    a.scatter(xs[y == 0], v[y == 0], s=12, color=NOR, alpha=.5, lw=0)
    a.scatter(xs[y == 1], v[y == 1], s=18, color=ABN, lw=0)
    for nm in HARD:
        i = names.index(nm)
        a.annotate(nm.replace("abn_", "A").replace("nor_", "N"), (xs[i], v[i]), fontsize=6.5, xytext=(3, 0),
                   textcoords="offset points", color=ABN if y[i] else NOR)
    from sklearn.metrics import roc_auc_score
    a.set_title(f"{c}\nAUC {roc_auc_score(y, v):.2f} (설명용)", fontsize=9)
    a.set_xticks([0, 1]); a.set_xticklabels(["정상", "고장"]); a.grid(alpha=.3, axis="y")
fig.suptitle("충격 반복(B)·단발 사건(E) 피처. 파란 띠 = 정상 5–95%. 번호 = 이전 실험에서 틀린 파일", fontsize=11)
fig.tight_layout()
ex.save_fig(fig, "impact_features", "놓친 고장(A002·005·014·017·022·029)이 띠 밖으로 나가는 피처가 있는가, "
            "오경보 정상(N025·055·116·086 …)이 반대쪽으로 갈리는 피처가 있는가")


# %% quick_cv
Z0 = (ae_mean, et_logit)
CONFIGS = [("0 기준: z(AE 평균, ET) (EXP-014)", dict(experts=Z0)),
           ("A AE 구간 최댓값", dict(experts=(ae_max, et_logit))),
           ("C 모양 AE (게인 제거)", dict(experts=(ae_shape_mean, et_logit))),
           ("A+C 모양 AE 구간 최댓값", dict(experts=(ae_shape_max, et_logit))),
           ("D 기준 + max(z) 열 (비대칭)", dict(experts=Z0, add_max=True)),
           ("B 기준 + 충격 주기성 4", dict(experts=Z0 + (imp_period,))),
           ("E 기준 + 단발 사건 4", dict(experts=Z0 + (imp_single,))),
           ("B+E 기준 + 충격 8", dict(experts=Z0 + (imp_period, imp_single))),
           ("전부: 모양 AE 최댓값 + ET + 충격 8 + max(z)", dict(experts=(ae_shape_max, et_logit, imp_period, imp_single),
                                                       add_max=True))]
nor = y == 0
R, rows = {}, []
for name, kw in CONFIGS:
    q = quick_cv(g[:, None], y, g, lambda kw=kw: ScoreStack(combine="lr", **kw), seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    R[name] = q
    rows.append(f"{name:42s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · "
                f"ROC-AUC {q['roc_auc'].mean():.3f} ± {q['roc_auc'].std():.3f}")
    print(rows[-1], flush=True)
ex.note("quick_cv 결과 (5-fold × seed 3, 임계값 없음)", "\n".join(rows))
cfg = [c[0] for c in CONFIGS]
for c in cfg:
    pm = R[c]["pct"]
    ex.note(f"{c.split()[0]} 정상 90% 선 아래 고장 · 정상 상위 5%",
            f"고장: {' '.join(names[i].replace('abn_', '') for i in np.flatnonzero((y == 1) & (pm < .9)))}\n"
            f"정상: {' '.join(names[i].replace('nor_', '') for i in np.flatnonzero((y == 0) & (pm > .95)))}")


# %% 그림 2: 어려운 파일의 위험도 (고장 = 1 − 순위, 정상 = 순위). 기준 대비 옅어지면 개선
M = np.array([[1 - R[c]["pct"][names.index(nm)] if nm.startswith("abn") else R[c]["pct"][names.index(nm)]
               for c in cfg] for nm in HARD])
fig, ax = plt.subplots(figsize=(12, 9))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD))); ax.set_yticklabels([("놓침 " if n.startswith("abn") else "오경보 ") + n for n in HARD], fontsize=8)
ax.set_xticks(range(len(cfg))); ax.set_xticklabels([c.split()[0] for c in cfg], fontsize=9)
ax.set_title("어려운 파일의 위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율. "
             "열 이름 = 구성 기호", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "처방별로 겨냥한 파일이 옅어지는가 (A: 014·017, C: 002·029·059·062, D: 017, B: 022·005, "
            "E: 025·055·116·086), 다른 파일이 진해지지는 않는가")


# %% 그림 3: t-SNE (매 실험 기본) — 출력 끝단(AE bin 오차 + ET 나무) / + 충격 피처 / '전부' 구성의 결합 z
AEo, ETo, ZA = np.empty((len(y), LP.shape[-1])), np.empty((len(y), 300)), None
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    a, b = tuple(tr.tolist()), tuple(te.tolist())
    AEo[te], ETo[te] = ae_bins(a, b), et_trees(a, b)
    z = ScoreStack(combine="lr", **CONFIGS[-1][1]).fit(g[tr][:, None], y[tr]).zscores(g[te][:, None])
    ZA = np.empty((len(y), z.shape[1])) if ZA is None else ZA
    ZA[te] = z
std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
blk = lambda *Bs: np.hstack([std(B) / np.sqrt(B.shape[1]) for B in Bs])
IM8 = IMP[cols].values
best = max(cfg[1:], key=lambda c: R[c]["pr_auc"].mean())
fig, _ = viz.tsne({"출력 끝단: AE bin 오차 + ET 나무 (EXP-014 공간)": blk(AEo, ETo),
                   "출력 끝단 + 충격 피처 8 (블록 같은 비중)": blk(AEo, ETo, IM8),
                   "'전부' 구성의 결합 z (11차원, OOF)": std(ZA)},
                  y, names, mark=HARD, score=R[best]["oof"][0], scale=False)
ex.save_fig(fig, "tsne", f"충격 피처를 더하면 놓친 고장이 고장 무리로 옮겨 가는가. 2행 색 = PR-AUC 최고 구성({best.split()[0]})의 OOF 점수(seed 0)")


# %% 그림 4: 놓친 고장 vs 정상 5–95% (매 실험 기본) — PR-AUC 최고 구성 기준
missed = [names[i] for i in np.flatnonzero((y == 1) & (R[best]["pct"] < .9))]
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title=f"최고 구성({best.split()[0]})이 정상 90% 선 아래로 둔 고장"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)
