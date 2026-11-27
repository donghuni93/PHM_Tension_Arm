# %% [markdown]
# EXP-026 SVM 경계 + 하이퍼파라미터 튜닝 — 규칙 대신 규칙 축을 연속 입력으로 (6축)
# - 질문 (사용자 11-16): 로지스틱·SVM을 더 튜닝하면 나아지나? 지금 피처로 SVM을 다시 돌리면?
# - 바꾼 것: 결합 경계 (EXP-025는 로지스틱 C=1 고정 + 손으로 정한 규칙 2개)
#   6축 = AE z · ET z · 유격 r · 2위/1위 사건(K3) · 8.3k 돌출(P83) · 8.3k 폭(W83). 모두 학습 정상 기준 z (ScoreStack)
#   하이퍼파라미터(C, γ)는 바깥 fold 학습 파일 안에서 4-fold PR-AUC로 고름 (src/experts.py TUNE_GRID) → 선택 편향 없음
# - 구성 (주 구성 실행 전 지정 = 4)
#   0 기준 EXP-025: 로지스틱(z 2축, C=1) + 8.3k 규칙 + 유격 규칙
#   1 RBF SVM 튜닝(z 2축) + 같은 규칙 2개 — 경계만 교체
#   2 로지스틱 튜닝(6축, L2, C) — 규칙 없음
#   3 선형 SVM 튜닝(6축, C) — 규칙 없음
#   4 RBF SVM 튜닝(6축, C·γ) — 규칙 없음 (주)
#   5 RBF SVM 고정(6축, C=1, γ=scale) — 튜닝 효과 확인
# - 예상 (실행 전): EXP-025 README2 참고 표에서 6축 로지스틱·RBF가 PR-AUC 0.945로 같았다 → 경계 모양보다 정보가 결정.
#   규칙(문턱 0.7·2·0.75·150 Hz)을 모델이 스스로 배우게 하면 사후 문턱 의존이 줄지만, 규칙만큼 날카롭게는 못 할 수 있다
# - 주의: 규칙 축 정의(대역 8.20–8.45 kHz, K1·K2)는 라벨을 본 뒤 만든 축이라 6축 구성도 낙관적이다 (문턱은 모델이 배움)
# 실행: conda run -n qe-phm python 3_단계/5_모델비교/스크리닝/EXP-026_SVM튜닝6축/run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, separation
from src.experts import (ScoreStack, RuleOverride, Rule83, KnockRule, ae_mean, et_logit, knock_axes, peak83_axes,
                         ae_segerr, et_trees, warm_cache)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ex = Experiment(__file__, "EXP-026", "SVM 경계 + 튜닝 — 규칙 대신 규칙 축을 연속 입력으로 (6축)", ["H1", "H2", "H3"],
                member="이동훈", model="ET + 스펙트럼 AE + 규칙 축 4개 → RBF SVM (C·γ 안쪽 CV 튜닝)",
                changes="EXP-025의 '로지스틱 C=1 + 손 규칙 2개'를 6축 연속 입력 + 튜닝된 경계(로지스틱·선형/RBF SVM)로 교체",
                focus=("abn_022", "nor_025", "nor_055", "nor_116"))
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
Z2 = (ae_mean, et_logit)
Z6 = (ae_mean, et_logit, knock_axes, peak83_axes)
AX6 = ["AE z", "ET z", "유격 r", "2위/1위", "8.3k 돌출", "8.3k 폭"]
RULES = [Rule83(0.75, 150), KnockRule(2.0, 0.7)]
CONFIGS = [
    ("0 기준: EXP-025 (로지스틱 z2 + 규칙 2개)", lambda: RuleOverride(dict(experts=Z2, combine="lr"), RULES)),
    ("1 RBF SVM 튜닝 z2 + 규칙 2개", lambda: RuleOverride(dict(experts=Z2, combine="svm_rbf_cv"), RULES)),
    ("2 로지스틱 튜닝 6축", lambda: ScoreStack(Z6, "lr_cv")),
    ("3 선형 SVM 튜닝 6축", lambda: ScoreStack(Z6, "svm_lin_cv")),
    ("4 RBF SVM 튜닝 6축 (주)", lambda: ScoreStack(Z6, "svm_rbf_cv")),
    ("5 RBF SVM 고정 6축 (C=1, γ=scale)", lambda: ScoreStack(Z6, "svm_rbf_fix")),
]
MAIN = CONFIGS[4][0]
warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
print("cache ok", flush=True)


# %% quick_cv 분리 판정 (seed 3) + 골라진 하이퍼파라미터 (seed 0 바깥 5-fold)
Q, rows = {}, []
for name, make in CONFIGS:
    q = quick_cv(g[:, None], y, g, make, seeds=SEEDS)
    s = separation(q["oof"], y, names)
    Q[name] = q
    rows.append(f"{name:36s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · ROC-AUC {q['roc_auc'].mean():.3f} · "
                f"확실한 고장 {s['sure_a']} · 회색 고장 {' '.join(x[4:] for x in s['gray_a'])} · 위험 정상 {s['risk_n']:.1f}")
    print(rows[-1], flush=True)
ex.note("quick_cv 분리 판정 (seed 3)", "\n".join(rows))

prm = {}
for name, make in CONFIGS:
    if "튜닝" not in name:
        continue
    ps = []
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        m = make().fit(g[tr][:, None], y[tr])
        mm = m.m_.m_ if isinstance(m, RuleOverride) else m.m_
        ps.append({k.split("__")[1]: v for k, v in mm.best_params_.items()})
    prm[name] = ps
    print(name, ps, flush=True)
ex.note("골라진 하이퍼파라미터 (seed 0 바깥 fold 5개)", "\n".join(f"{k}: {v}" for k, v in prm.items()))


# %% 중첩 CV F2 (모든 구성)
R = {}
for name, make in CONFIGS:
    R[name] = nested_cv(g[:, None], y, g, make, n_jobs=-1)
    ex.record("중첩: " + name, R[name], names, main=(name == MAIN))


# %% 그림 1 (핵심): 파일별 오류 횟수 — 어느 구성이 어떤 파일을 틀리는가
cfg = [c[0] for c in CONFIGS]
err = {c: np.where(y == 1, R[c]["miss_counts"], R[c]["fp_counts"]) for c in cfg}
idx = [i for i in range(len(y)) if max(err[c][i] for c in cfg) >= 2]
idx = sorted(idx, key=lambda i: (-y[i], names[i]))
M = np.array([[err[c][i] for c in cfg] for i in idx])
fig, ax = plt.subplots(figsize=(10, 0.36 * len(idx) + 2.2))
ax.imshow(M, cmap="Blues", vmin=0, vmax=5, aspect="auto")
for r in range(M.shape[0]):
    for c in range(M.shape[1]):
        if M[r, c]:
            ax.text(c, r, str(M[r, c]), ha="center", va="center", fontsize=9, color="white" if M[r, c] >= 3 else "#222")
ax.set_xticks(range(len(cfg)))
ax.set_xticklabels([c.split(" (")[0].replace(" 튜닝", "\n튜닝").replace(" 고정", "\n고정") for c in cfg], fontsize=8)
ax.set_yticks(range(len(idx)))
ax.set_yticklabels([("놓침 " if y[i] else "오경보 ") + short(names[i]) for i in idx], fontsize=8)
for t, i in zip(ax.get_yticklabels(), idx):
    t.set_color("#eb6834" if y[i] else "#2a78d6")
ax.set_xticks(np.arange(-.5, len(cfg)), minor=True); ax.set_yticks(np.arange(-.5, len(idx)), minor=True)
ax.grid(which="minor", color="white", lw=2); ax.tick_params(which="minor", length=0)
f2s = [f"F2 {R[c]['f2'].mean():.3f}" for c in cfg]
ax2 = ax.secondary_xaxis("top"); ax2.set_xticks(range(len(cfg))); ax2.set_xticklabels(f2s, fontsize=8)
ax.set_title("파일별 오류 횟수 (seed 5번 중, 2회+ 파일만) — 위 = 중첩 F2", fontsize=10, pad=24)
fig.tight_layout()
ex.save_fig(fig, "error_files", "구성별로 같은 파일을 틀리는가(정보 한계) 아니면 다른 파일을 틀리는가(경계 차이). 셀 = seed 5번 중 틀린 횟수", key=True)


# %% 6축 z (seed 0 OOF, 그림용)
Z = np.empty((len(y), 6))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    Z[te] = ScoreStack(Z6, "mean").fit(g[tr][:, None], y[tr]).zscores(g[te][:, None])
np.save(os.path.join(ex.figs, "z6_seed0.npy"), Z)

# %% 그림 2: 6축 각각에서 오경보 정상·놓친 고장의 위치 (정상 5–95% 띠)
hard = [names.index(n) for n in ("abn_022", "abn_014", "abn_017", "nor_025", "nor_055", "nor_116", "nor_059")]
fig, axs = plt.subplots(1, 6, figsize=(17, 4.6), sharey=True)
for j, ax in enumerate(axs):
    v = Z[:, j]
    lo, hi = np.percentile(v[y == 0], [5, 95])
    ax.axvspan(lo, hi, color="#2a78d6", alpha=.12, lw=0)
    jit = np.random.default_rng(0).uniform(-.25, .25, len(y))
    ax.scatter(v[y == 0], 0 + jit[y == 0], s=14, color="#2a78d6", alpha=.5, lw=0)
    ax.scatter(v[y == 1], 1 + jit[y == 1], s=18, color="#eb6834", alpha=.7, lw=0)
    for i in hard:
        ax.scatter(v[i], y[i] + jit[i], s=60, facecolor="none", edgecolor="k", lw=1)
        ax.annotate(short(names[i]), (v[i], y[i] + jit[i]), xytext=(3, 4), textcoords="offset points", fontsize=7)
    ax.set_title(AX6[j], fontsize=9); ax.set_xscale("symlog"); ax.grid(alpha=.25)
axs[0].set_yticks([0, 1]); axs[0].set_yticklabels(["정상", "고장"])
fig.suptitle("6축 z (seed 0 OOF, 1 = 정상 90%선). 파란 띠 = 정상 5–95%. 테두리 = 자주 틀리는 파일", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "axes6", "자주 틀리는 파일(022·025·055·116 등)이 6축 중 어느 축에서라도 정상 띠 밖으로 나가는가")


# %% 그림 3: t-SNE (매 실험 기본) — 6축 입력, 색 = 주 구성 OOF 점수(seed 0)
fig, _ = viz.tsne({"6축 z (AE·ET·r·K3·P83·W83)": Z}, y, names, mark=[names[i] for i in hard], score=Q[MAIN]["oof"][0])
ex.save_fig(fig, "tsne", "6축 공간에서 022가 정상 무리 안인가, 오경보 정상 025·055·116이 고장 쪽인가. 2행 색 = RBF SVM 튜닝 OOF 점수")


# %% 그림 4: 놓친 고장 vs 정상 5–95% (매 실험 기본) — 주 구성
missed = [names[i] for i in np.flatnonzero(R[MAIN]["miss_counts"] >= 3)]
if missed:
    f, L = viz.log_psd(X)
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title="RBF SVM 튜닝 6축이 자주 놓친 고장"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)
