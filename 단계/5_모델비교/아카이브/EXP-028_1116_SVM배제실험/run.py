# %% [markdown]
# EXP-028 SVM 배제 실험 — 지금 피처로 못 가르는 정상을 학습에서 빼면 SVM 경계가 어떻게 바뀌나
# - 질문 (사용자 11-16): SVM이 안 된 것은 지금 피처로 구분할 수 없는 것까지 가르려 했기 때문 아닌가?
#   정상 025·116·106·059 등을 학습에서 빼고 SVM을 만들면? (그 조건들은 나중에 피처를 더해 가른다)
# - 방법: 결합 구조는 EXP-025 그대로(AE·ET z 2축 + 8.3k 규칙 + 유격 규칙), 경계만 RBF SVM(C=1, γ=scale 고정)
#   배제 = **학습에서만** 뺀다 (src/experts.py ExcludeFit). 뺀 파일도 바깥·안쪽 fold에서 채점·평가된다
# - 배제 세트 (실험 전 지정)
#   A 없음 · B 사용자 목록 025 059 106 116 · C N-A형(큰 사건 한 번) 025 055 116
#   D 2-11 경계 정상 전체 025 027 055 059 060 062 073 106 116 · E D + 고장 022(정상 무리 안 고장)
# - 주 구성 = 2 (B, 사용자 제안). 예상(실행 전): 고장처럼 보이는 정상을 빼면 경계가 그쪽으로 밀려 들어가
#   뺀 정상은 오히려 더 확실한 오경보가 된다. 배제 목록은 라벨·CV 오류를 보고 고른 것이라 사후 선택
# 실행: conda run -n qe-phm python 단계/5_모델비교/아카이브/EXP-028_1116_SVM배제실험/run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv
from src.experts import (ScoreStack, RuleOverride, Rule83, KnockRule, ExcludeFit, ae_mean, et_logit,
                         ae_segerr, et_trees, warm_cache)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-028", "SVM 배제 실험 — 못 가르는 정상을 학습에서 빼면 경계가 어떻게 바뀌나", ["H1", "H5"],
                member="이동훈", model="ET + 스펙트럼 AE → RBF SVM(C=1) + 8.3k 규칙 + 유격 규칙, 학습에서 정상 일부 배제",
                changes="EXP-025 경계를 RBF SVM으로, 회색 정상(025 059 106 116 등)을 학습에서만 뺌 (평가는 150개 전부)",
                focus=("nor_025", "nor_059", "nor_106", "nor_116"))
X, y, names = load_train()
g = np.arange(len(y))
nor = y == 0
idx = lambda *ns: tuple(names.index(n) for n in ns)
SETS = {
    "A 없음": (),
    "B 사용자 025 059 106 116": idx("nor_025", "nor_059", "nor_106", "nor_116"),
    "C N-A형 025 055 116": idx("nor_025", "nor_055", "nor_116"),
    "D 경계 정상 9개": idx(*[f"nor_{k:03d}" for k in (25, 27, 55, 59, 60, 62, 73, 106, 116)]),
    "E D + 고장 022": idx(*[f"nor_{k:03d}" for k in (25, 27, 55, 59, 60, 62, 73, 106, 116)], "abn_022"),
}
Z2 = (ae_mean, et_logit)
RULES = [Rule83(0.75, 150), KnockRule(2.0, 0.7)]
svm_rules = lambda: RuleOverride(dict(experts=Z2, combine="svm_rbf"), RULES)
CONFIGS = [("0 기준: EXP-025 로지스틱 (배제 없음)", lambda: RuleOverride(dict(experts=Z2, combine="lr"), RULES), ())]
CONFIGS += [(f"{k + 1} SVM 배제 {s}", (lambda e=e: ExcludeFit(svm_rules, e)), e) for k, (s, e) in enumerate(SETS.items())]
MAIN = CONFIGS[2][0]
warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
print("cache ok", flush=True)


# %% 중첩 CV F2 (배제 파일도 평가에 포함)
R = {}
for name, make, e in CONFIGS:
    R[name] = nested_cv(g[:, None], y, g, make, n_jobs=-1)
    row = ex.record("중첩: " + name, R[name], names, main=(name == MAIN))
    fp = " ".join(f"{short(names[i])} {R[name]['fp_counts'][i] if y[i] == 0 else R[name]['miss_counts'][i]}/5" for i in e)
    ex.note(f"배제 파일 오류 횟수: {name}", fp or "—")


# %% quick_cv (seed 3) — 순위 지표 + 배제 파일의 정상 중 백분위
cfg = [c[0] for c in CONFIGS]
Q, rows = {}, []
for name, make, e in CONFIGS:
    q = quick_cv(g[:, None], y, g, make, seeds=range(3))
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    Q[name] = q
    rows.append(f"{name:36s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f}")
    print(rows[-1], flush=True)
ex.note("quick_cv PR-AUC (seed 3)", "\n".join(rows))
ALLX = sorted(set(sum(SETS.values(), ())), key=lambda i: (y[i], names[i]))
tab = ["파일     " + " | ".join(f"{c[:2].strip():>5s}" for c in cfg)]
for i in ALLX:
    tab.append(f"{short(names[i]):7s}  " + " | ".join(f"{Q[c]['pct'][i]:5.2f}" for c in cfg))
ex.note("배제 후보 파일의 정상 중 백분위 (quick_cv seed 3 평균, 1 = 모든 정상보다 높음 = 고장 쪽)", "\n".join(tab))


# %% 공통 좌표: AE·ET z 2축 (seed 0 바깥 5-fold OOF, 배제 없이)
Z = np.empty((len(y), 2))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    Z[te] = ScoreStack(Z2, "mean").fit(g[tr][:, None], y[tr]).zscores(g[te][:, None])
fired = np.zeros(len(y), bool)
for r in RULES:
    fired |= (r.fit(g) if hasattr(r, "fit") else r)(g)
np.save(os.path.join(ex.figs, "z2_seed0.npy"), Z)


# %% 그림 1 (핵심): 배제 세트별 SVM 경계 — 같은 z 평면에 학습(배제 뺀 점)으로 맞춘 예시
LO, HI = -3, 6
gx, gy = np.meshgrid(np.linspace(LO, HI, 300), np.linspace(LO, HI, 300))
G = np.c_[gx.ravel(), gy.ravel()]
zc = np.clip(Z, LO, HI)
LABEL = set(ALLX) | set(np.flatnonzero((y == 1) & ~fired & (Z.sum(1) < 2)))
fig, axs = plt.subplots(2, 3, figsize=(19, 12.5))
for ax, (name, make, e) in zip(axs.ravel(), CONFIGS):
    keep = ~np.isin(g, e)
    m = (LogisticRegression(C=1.0, class_weight="balanced") if "로지스틱" in name
         else SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")).fit(Z[keep], y[keep])
    d = m.decision_function(G).reshape(gx.shape)
    ax.contourf(gx, gy, d, levels=[0, 1e9], colors=[ABN], alpha=.10)
    ax.contour(gx, gy, d, levels=[0], colors="k", linewidths=1.4)
    ax.contour(gx, gy, d, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    ax.scatter(*zc[nor & keep].T, s=20, color=NOR, alpha=.55, lw=0, label="정상 (학습)")
    ax.scatter(*zc[~nor & keep & ~fired].T, s=30, color=ABN, lw=0, label="고장 (학습)")
    ax.scatter(*zc[~nor & keep & fired].T, s=30, color=ABN, lw=0, marker="D", alpha=.45, label="고장 · 규칙이 잡음")
    if hasattr(m, "support_"):
        sv = np.flatnonzero(keep)[m.support_]
        ax.scatter(*zc[sv].T, s=70, facecolor="none", edgecolor="#555", lw=.6, label="서포트 벡터")
    if len(e):
        ex_ = np.array(e)
        ax.scatter(*zc[ex_].T, s=80, marker="X", color="k", lw=0, label="학습에서 뺀 파일")
    for i in LABEL:
        ax.annotate(short(names[i]), zc[i], xytext=(4, 3), textcoords="offset points", fontsize=7.5,
                    color=ABN if y[i] else NOR, weight="bold" if i in e else "normal")
    r = R[name]
    fp_ex = sum(r["fp_counts"][i] for i in e if y[i] == 0)
    ax.set_title(f"{name}\n중첩 F2 {r['f2'].mean():.3f} ± {r['f2'].std():.3f} · 오경보율 {r['fpr'].mean():.3f}"
                 + (f" · 뺀 정상 오경보 {fp_ex}회/{5 * sum(y[i] == 0 for i in e)}" if len(e) else ""), fontsize=9.5)
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.grid(alpha=.25)
    ax.set_xlabel("AE z (스펙트럼 모양 이상, 1 = 정상 90%선)"); ax.set_ylabel("ET z (충격 피처 이상)")
axs[0, 0].legend(fontsize=7.5, loc="upper left")
axs[0, 1].legend(fontsize=7.5, loc="upper left")
fig.suptitle("배제 세트별 경계 (seed 0 OOF z에 학습 점만으로 맞춘 예시). 실선 = 결정값 0, 점선 = 여백 ±1, 주황 영역 = 고장 판정, X = 학습에서 뺀 파일",
             fontsize=11)
fig.tight_layout()
ex.save_fig(fig, "svm_exclusion", "정상을 빼면 경계가 그 자리로 들어가 뺀 정상(X)이 고장 영역에 들어가는가. 제목 = 150개 전부로 평가한 중첩 F2", key=True)


# %% 그림 2: 파일별 오류 횟수 (배제 후보 + 어느 구성이든 2회+)
err = {c: np.where(y == 1, R[c]["miss_counts"], R[c]["fp_counts"]) for c in cfg}
rows_i = sorted(set(ALLX) | {i for i in range(len(y)) if max(err[c][i] for c in cfg) >= 2}, key=lambda i: (-y[i], names[i]))
M = np.array([[err[c][i] for c in cfg] for i in rows_i])
fig, ax = plt.subplots(figsize=(11, 0.36 * len(rows_i) + 2.4))
ax.imshow(M, cmap="Blues", vmin=0, vmax=5, aspect="auto")
for a in range(M.shape[0]):
    for b in range(M.shape[1]):
        i = rows_i[a]
        mark = "×" if i in CONFIGS[b][2] else ""
        if M[a, b] or mark:
            ax.text(b, a, f"{M[a, b]}{mark}", ha="center", va="center", fontsize=9, color="white" if M[a, b] >= 3 else "#222")
ax.set_xticks(range(len(cfg)))
ax.set_xticklabels([c.replace(" 배제 ", "\n배제 ").replace("기준: ", "기준\n") for c in cfg], fontsize=7.5)
ax.set_yticks(range(len(rows_i)))
ax.set_yticklabels([("놓침 " if y[i] else "오경보 ") + short(names[i]) for i in rows_i], fontsize=8)
for t, i in zip(ax.get_yticklabels(), rows_i):
    t.set_color(ABN if y[i] else NOR)
ax.set_xticks(np.arange(-.5, len(cfg)), minor=True); ax.set_yticks(np.arange(-.5, len(rows_i)), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
ax2 = ax.secondary_xaxis("top"); ax2.set_xticks(range(len(cfg))); ax2.set_xticklabels([f"F2 {R[c]['f2'].mean():.3f}" for c in cfg], fontsize=8)
ax.set_title("파일별 오류 횟수 (seed 5번 중). × = 그 구성에서 학습에서 뺀 파일", fontsize=10, pad=24)
fig.tight_layout()
ex.save_fig(fig, "error_files", "뺀 정상(×)의 오경보가 줄었는가 늘었는가, 대신 다른 파일이 틀리기 시작했는가")


# %% 그림 3: 배제 후보 파일의 정상 중 백분위 변화 (quick_cv)
fig, ax = plt.subplots(figsize=(11, 5))
for k, c in enumerate(cfg):
    v = [Q[c]["pct"][i] for i in ALLX]
    ax.plot(range(len(ALLX)), v, "o-", lw=1, ms=6, label=c, alpha=.85)
ax.set_xticks(range(len(ALLX))); ax.set_xticklabels([short(names[i]) for i in ALLX], fontsize=8)
ax.axhline(.9, color="k", lw=.7, ls="--"); ax.set_ylim(0, 1.02); ax.grid(alpha=.3)
ax.set_ylabel("정상 중 백분위 (1 = 모든 정상보다 고장 쪽)"); ax.legend(fontsize=7.5, loc="lower left")
ax.set_title("배제 후보 파일의 점수 위치 (quick_cv OOF, seed 3 평균). 정상은 낮을수록, 고장 022는 높을수록 좋다", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "excluded_rank", "학습에서 뺀 정상이 정상 무리 안쪽(낮은 백분위)으로 내려가는가, 아니면 더 고장 쪽으로 올라가는가")


# %% 그림 4: t-SNE (매 실험 기본) — ext150 입력, 색 = 주 구성 OOF 점수(seed 0)
fig, _ = viz.tsne({"ext150 (ET 입력)": cached_table("ext", feature_table_ext, X).values, "AE·ET z 2축": Z}, y, names,
                  mark=[names[i] for i in ALLX], score=np.clip(Q[MAIN]["oof"][0], -5, 5))
ex.save_fig(fig, "tsne", "뺀 정상 025·059·106·116이 정상 무리 가장자리인가 고장 무리 안인가. 2행 색 = 주 구성(B 배제) OOF 점수")


# %% 그림 5: 놓친 고장 vs 정상 5–95% (매 실험 기본) — 주 구성
missed = [names[i] for i in np.flatnonzero(R[MAIN]["miss_counts"] >= 3)]
if missed:
    f, L = viz.log_psd(X)
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title="B 배제 SVM이 자주 놓친 고장"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)
