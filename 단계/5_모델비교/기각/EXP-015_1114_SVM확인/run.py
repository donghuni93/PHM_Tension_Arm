# %% [markdown]
# EXP-015 SVM 확인 — SVM 하나로(25년 3조 방식) / 결합 경계를 SVM으로 / 출력 끝단 위 SVM
# - 질문 (사용자): 25년 3조처럼 SVM만으로 고장·정상을 가를 수 있나? EXP-014 회색지대를 SVM 경계가 가르나?
# - 예상 (실행 전): 회색지대는 경계 모양이 아니라 정보 부족 문제라 SVM으로 크게 나아지지 않는다
#   (002·029가 059·062와 같은 자리, 022는 정상 한가운데 — EXP-014 boundary_2d·tsne)
# - 구성 (SVM은 C=1, gamma='scale', 균형 가중 고정 — 튜닝하면 선택 편향)
#   0 기준: EXP-014 로지스틱 결합 z(AE, ET)
#   1·2 SVM 단독: ext150 → 표준화 → RBF / 선형 (25년 3조 피처+SVM은 EXP-003에서 0.538)
#   3·4 결합 경계: z(AE, ET) 위 선형 / RBF SVM
#   5·6 출력 끝단: AE bin별 오차(1025) + ET 나무별 확률(300), 블록 같은 비중 → RBF / 선형 SVM
# - 평가: 탐색이라 quick_cv(5-fold × seed 3, PR-AUC). 기준보다 sd 넘게 좋은 구성만 중첩 CV
# 실행: conda run -n qe-phm python 단계/5_모델비교/기각/EXP-015_1114_SVM확인/run.py

# %% 설정
import os, sys, warnings
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..", "..")))
from src.exp import *
from src.evaluate import quick_cv
from src.experts import ScoreStack, ae_mean, ae_bins, et_logit, et_trees, ae_segerr, warm_cache, E, LP
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-015", "SVM 확인 — SVM 단독(25년 방식) · 결합 경계 SVM · 출력 끝단 SVM", ["H1", "H2", "H3"],
                member="이동훈", model="SVM (RBF·선형, C=1 고정)",
                changes="EXP-014 결합의 경계·입력을 SVM으로 바꿔 비교 (모델만 바꿈)", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)

ET_MISS = ["abn_002", "abn_022", "abn_029"]
AE_MISS = ["abn_005", "abn_014", "abn_017", "abn_022"]
ET_FP = ["nor_008", "nor_025", "nor_027", "nor_050", "nor_054", "nor_055", "nor_073", "nor_086", "nor_095", "nor_116"]
AE_FP = ["nor_025", "nor_039", "nor_055", "nor_059", "nor_062", "nor_106", "nor_116"]
HARD = sorted(set(ET_MISS + AE_MISS)) + sorted(set(ET_FP + AE_FP + ["nor_065"]))

n = warm_cache([ae_segerr, et_trees], seeds=SEEDS)
print(f"캐시 완료: 분할 {n}개", flush=True)

# %% quick_cv
svm = lambda kernel: (lambda: make_pipeline(StandardScaler(), SVC(kernel=kernel, C=1.0, gamma="scale", class_weight="balanced")))
Z2 = (ae_mean, et_logit)
ENDS, BLK = (ae_bins, et_trees), [LP.shape[-1], 300]
CONFIGS = [("0 기준: 로지스틱 결합 z(AE, ET) (EXP-014)", g[:, None], lambda: ScoreStack(Z2, "lr")),
           ("1 SVM 단독: ext150 → RBF (25년 방식)", E, svm("rbf")),
           ("2 SVM 단독: ext150 → 선형", E, svm("linear")),
           ("3 결합 경계: z(AE, ET) → 선형 SVM", g[:, None], lambda: ScoreStack(Z2, "svm_lin")),
           ("4 결합 경계: z(AE, ET) → RBF SVM", g[:, None], lambda: ScoreStack(Z2, "svm_rbf")),
           ("5 출력 끝단: AE bin 오차 + ET 나무 → RBF SVM", g[:, None],
            lambda: ScoreStack(ENDS, "svm_rbf", norm=False, blocks=BLK)),
           ("6 출력 끝단: AE bin 오차 + ET 나무 → 선형 SVM", g[:, None],
            lambda: ScoreStack(ENDS, "svm_lin", norm=False, blocks=BLK))]
nor = y == 0
R, rows = {}, []
for name, F, make in CONFIGS:
    q = quick_cv(F, y, g, make, seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    R[name] = q
    rows.append(f"{name:44s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · "
                f"ROC-AUC {q['roc_auc'].mean():.3f} ± {q['roc_auc'].std():.3f}")
    print(rows[-1], flush=True)
ex.note("quick_cv 결과 (5-fold × seed 3, 임계값 없음)", "\n".join(rows))

cfg = [c[0] for c in CONFIGS]
tab = ["파일        " + " | ".join(f"{c[:6]:6s}" for c in cfg)]
for nm in HARD:
    i = names.index(nm)
    tab.append(f"{nm:10s}  " + " | ".join(f"{R[c]['pct'][i]:6.2f}" for c in cfg))
ex.note("어려운 파일의 정상 중 순위 (고장은 1에 가까울수록, 정상은 0에 가까울수록 좋음)", "\n".join(tab))
for c in cfg:
    pm = R[c]["pct"]
    ex.note(f"{c[:2].strip()} 정상 90% 선 아래 고장 · 정상 상위 5%",
            f"고장: {' '.join(names[i].replace('abn_', '') for i in np.flatnonzero((y == 1) & (pm < .9)))}\n"
            f"정상: {' '.join(names[i].replace('nor_', '') for i in np.flatnonzero((y == 0) & (pm > .95)))}")


# %% 그림 1: 2차원 z에서 경계 모양 — 로지스틱 직선 vs 선형 SVM vs RBF SVM (seed 0 OOF z에 전부 맞춘 예시)
Zall = np.empty((len(y), 2))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    Zall[te] = ScoreStack(Z2, "mean").fit(g[tr][:, None], y[tr]).zscores(g[te][:, None])
LO, HI = -3, 4
gx, gy = np.meshgrid(np.linspace(LO, HI, 300), np.linspace(LO, HI, 300))
G = np.c_[gx.ravel(), gy.ravel()]
fig, ax = plt.subplots(figsize=(9.5, 8))
zc = np.clip(Zall, LO, HI)
ax.scatter(*zc[y == 0].T, s=22, color=NOR, alpha=.55, lw=0, label="정상")
ax.scatter(*zc[y == 1].T, s=34, color=ABN, lw=0, label="고장")
for mdl, ls, lab in ((LogisticRegression(C=1.0, class_weight="balanced"), "-", "로지스틱"),
                     (SVC(kernel="linear", C=1.0, class_weight="balanced"), "--", "선형 SVM"),
                     (SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"), ":", "RBF SVM")):
    d = mdl.fit(Zall, y).decision_function(G).reshape(gx.shape)
    cs = ax.contour(gx, gy, d, levels=[0], colors="k", linestyles=ls, linewidths=1.3)
    ax.plot([], [], "k", ls=ls, label=f"{lab} 경계 (결정값 0)")
for nm in HARD:
    i = names.index(nm)
    ax.annotate(nm.replace("abn_", "A").replace("nor_", "N"), zc[i], xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=ABN if y[i] else NOR)
ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("AE z (1 = 정상 90%선)"); ax.set_ylabel("ET z (1 = 정상 90%선)")
ax.set_title("2차원 z (OOF seed 0)에 세 경계를 모두 맞춘 예시. 회색지대 파일을 RBF가 휘어서 가르는가", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "boundary_svm", "RBF SVM 경계가 직선과 다르게 휘는 곳이 회색지대(A002·A029 ↔ N059·N062, A017, A022)를 "
            "가르는가, 아니면 점 몇 개를 둘러싸는가(과적합)")


# %% 그림 2: t-SNE (매 실험 기본) — SVM 단독의 입력(ext150), 출력 끝단(AE+ET). 2행 색 = 각 공간 SVM의 OOF 점수
AEo, ETo = np.empty((len(y), LP.shape[-1])), np.empty((len(y), 300))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    a, b = tuple(tr.tolist()), tuple(te.tolist())
    AEo[te], ETo[te] = ae_bins(a, b), et_trees(a, b)
std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
ENDV = np.hstack([std(AEo) / np.sqrt(AEo.shape[1]), std(ETo) / np.sqrt(ETo.shape[1])])
fig, _ = viz.tsne({"SVM 단독 입력: ext150": std(E), "출력 끝단: AE bin 오차 + ET 나무 (블록 같은 비중)": ENDV},
                  y, names, mark=HARD, score=R[cfg[5]]["oof"][0], scale=False)
ex.save_fig(fig, "tsne", "SVM이 보는 공간에서 고장이 모이는가. 2행 색 = 구성 5(출력 끝단 RBF SVM)의 OOF 점수(seed 0)")


# %% 그림 3: 놓친 고장 vs 정상 5–95% (매 실험 기본) — 가장 PR-AUC가 높은 SVM 구성 기준
best = max(cfg[1:], key=lambda c: R[c]["pr_auc"].mean())
missed = [names[i] for i in np.flatnonzero((y == 1) & (R[best]["pct"] < .9))]
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title=f"SVM 최고 구성({best[:2].strip()})이 정상 90% 선 아래로 둔 고장"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)
