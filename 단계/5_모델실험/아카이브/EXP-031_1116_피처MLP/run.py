# %% [markdown]
# EXP-031 피처 입력 신경망(MLP) 스크리닝 — ET 자리를 신경망으로 바꾸거나 더하면 회색지대가 줄어드는가
# - 배경: 사용자 요청(11-16) "피처를 이용한 딥러닝 모델 검토". EXP-019·022는 원신호·스펙트로그램 모델 8종을 봤고 피처 MLP는 없었다
# - 가설: 없음(모델 교체 확인). 6-1에서 본 대로 남은 오류는 입력에 없는 정보(반복성·울리는 주파수·시점) 때문이므로
#   같은 ext150 입력의 모델만 바꿔서는 달라지지 않을 것으로 예상한다 (EXP-004·015와 같은 결론이 나오는지)
# - 모델 (하이퍼파라미터는 실행 전 고정, 튜닝 없음): ext150 → 부호 유지 log1p → 학습 쪽 표준화 → MLP(64-32, ReLU, L2 0.01, 500회)
#   고장을 4배로 복제해 균형, 초기값 5개 평균(배깅). 폴드 안에서만 학습
# - 구성 (quick_cv, 5-fold × seed 3, 기준과 같은 분할)
#   0 기준 EXP-025 (AE + ET → 로지스틱 + 8.3k·유격 규칙) · 1 교체 (AE + MLP) · 2 추가 (AE + ET + MLP) · 단독 비교 ET vs MLP (규칙 없음)
# - 통과 기준 (실행 전 고정, EXP-019와 같은 방식): 기준 대비 PR-AUC가 sd 넘게 오르거나, 분리 판정에서 "확실한 고장"이 늘고
#   "위험 정상"이 늘지 않을 것. 통과한 구성만 중첩 CV를 돌린다
# 실행: conda run -n qe-phm python "단계/5_모델실험/스크리닝/EXP-031_1116_피처MLP/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, separation
from src.experts import RuleOverride, Rule83, KnockRule, ScoreStack, ae_mean, et_logit, mem, E as EXT
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neural_network import MLPClassifier
warnings.filterwarnings("ignore", category=ConvergenceWarning)

NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-031", "피처 입력 신경망(MLP) 스크리닝 — ET 교체·추가", [],
                member="이동훈", model="스펙트럼 AE + MLP(ext150) → 로지스틱 + 8.3k·유격 규칙",
                changes="EXP-025의 ET 자리에 피처 입력 MLP(64-32)를 교체 또는 추가 (모델만 바꿈, 입력 피처는 같음)",
                focus=("abn_022", "abn_005", "nor_025", "nor_116", "nor_055"))
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]; nor = y == 0
SEEDS = range(3)
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
EL = np.sign(EXT) * np.log1p(np.abs(EXT))                       # 꼬리가 긴 피처(첨도 등)를 눌러 둔다 (라벨 무관)


@mem.cache
def mlp_proba(tr, te, n_bag=5):
    """ext150 MLP의 고장 확률 (te,). 표준화·학습은 tr로만"""
    tr, te = np.array(tr), np.array(te)
    mu, sd = EL[tr].mean(0), EL[tr].std(0) + 1e-12
    Ztr, Zte, ytr = (EL[tr] - mu) / sd, (EL[te] - mu) / sd, y[tr]
    rep = np.concatenate([np.arange(len(ytr)), np.repeat(np.flatnonzero(ytr == 1), 3)])
    ps = [MLPClassifier((64, 32), alpha=1e-2, max_iter=500, random_state=b).fit(Ztr[rep], ytr[rep]).predict_proba(Zte)[:, 1] for b in range(n_bag)]
    return np.mean(ps, 0).astype(np.float32)


def mlp_logit(tr, te):
    p = np.clip(mlp_proba(tr, te), 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))[:, None].astype(float)


R83, KN = Rule83(0.75, 150), KnockRule(2.0, 0.7)
BASE, REPL, ADD = "0 기준 EXP-025 (AE + ET)", "1 교체 (AE + MLP)", "2 추가 (AE + ET + MLP)"
CONFIGS = {
    BASE: lambda: RuleOverride(base_kw=dict(experts=(ae_mean, et_logit), combine="lr"), rule=[R83, KN]),
    REPL: lambda: RuleOverride(base_kw=dict(experts=(ae_mean, mlp_logit), combine="lr"), rule=[R83, KN]),
    ADD: lambda: RuleOverride(base_kw=dict(experts=(ae_mean, et_logit, mlp_logit), combine="lr"), rule=[R83, KN]),
    "단독 ET (규칙 없음)": lambda: ScoreStack(experts=(et_logit,), combine="col0"),
    "단독 MLP (규칙 없음)": lambda: ScoreStack(experts=(mlp_logit,), combine="col0"),
}
HARD = ["abn_022", "abn_005", "abn_025", "abn_020", "abn_015", "abn_021", "nor_025", "nor_116", "nor_055", "nor_059", "nor_106", "nor_001", "nor_054", "nor_027"]
rows, Q, RANK = [], {}, {}
for nm, mk in CONFIGS.items():
    q = quick_cv(F, y, g, mk, seeds=SEEDS); s = separation(q["oof"], y, names); Q[nm] = (q, s)
    pct = np.array([(q["oof"][:, nor] < q["oof"][:, [i]]).mean() for i in range(len(y))])
    RANK[nm] = [1 - pct[names.index(h)] if h.startswith("abn") else pct[names.index(h)] for h in HARD]
    rows.append(f"{nm:26s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 확실한 고장 {s['sure_a']} · 회색 고장 {' '.join(x[4:] for x in s['gray_a'])}"
                f" · 회색 정상 {len(s['gray_n'])} · 위험 정상 {s['risk_n']:.1f}")
ex.note("quick_cv 분리 판정 (seed 3)", "\n".join(rows))
ex.note("어려운 파일 위험도 (열 = 기준 · 교체 · 추가 · ET 단독 · MLP 단독)",
        "\n".join(f"{short(h)}: " + " · ".join(f"{RANK[k][i]:.2f}" for k in CONFIGS) for i, h in enumerate(HARD)))
b_q, b_s = Q[BASE]
passed = [nm for nm in (REPL, ADD)
          if Q[nm][0]["pr_auc"].mean() > b_q["pr_auc"].mean() + b_q["pr_auc"].std()
          or (Q[nm][1]["sure_a"] > b_s["sure_a"] and Q[nm][1]["risk_n"] <= b_s["risk_n"])]
ex.note("통과 판정", "통과: " + (", ".join(passed) or "없음"))

# %% 그림 1 (핵심): logistic_plane — 기준(AE z × ET z) | 교체(AE z × MLP z). 규칙으로 잡힌 고장은 뺀다
xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300)); G = np.c_[xx.ravel(), yy_.ravel()]
fig, axs = plt.subplots(1, 2, figsize=(16, 8))
for ax, (title, second, lab2) in zip(axs, (("기준 EXP-025", et_logit, "ET z (충격 피처, 나무)"), ("교체: MLP", mlp_logit, "MLP z (같은 피처, 신경망)"))):
    Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae_mean, second), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        rule[te] = R83(te) | KnockRule(2.0, 0.7).fit(tr)(te)
    Zc = np.clip(Z, -3, 4.5); show = ~(rule & (y == 1))
    Dm = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(G).reshape(xx.shape)
    ax.contourf(xx, yy_, Dm, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy_, Dm, levels=[0], colors="k", linewidths=1.5)
    for lab, sel, col, mk_ in (("정상", nor & show, NOR, "s"), ("고장 (규칙 밖)", ~nor & show, ABN, "o")):
        ax.scatter(Zc[sel, 0], Zc[sel, 1], color=col, s=26, lw=0, marker=mk_, label=lab, alpha=.85)
    for h in HARD:
        i = names.index(h)
        if show[i]: ax.annotate(short(h), Zc[i], xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel(lab2); ax.set_title(f"{title} (OOF seed 0, 실선 = 설명용 경계 0)", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "ET를 MLP로 바꾸면 경계 근처 파일(정상 025 116 055 059, 고장 022 005)의 위치가 달라지는가", key=True)

# %% 그림 2: hard_files
M = np.array([RANK[k] for k in CONFIGS]).T
fig, ax = plt.subplots(figsize=(10, 7.5))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h for h in HARD], fontsize=8)
ax.set_xticks(range(len(CONFIGS)), ["기준", "교체", "추가", "ET 단독", "MLP 단독"], fontsize=9)
ax.set_title("위험도 (quick_cv seed 3). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율 (진할수록 틀리기 쉬움)", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "MLP로 바꾸거나 더하면 어려운 파일의 위험도가 옅어지는가")

# %% 그림 3·4: t-SNE, 남은 오류 PSD
fig, _ = viz.tsne({"ext150 (log1p)": EL}, y, names, mark=list(ex.focus) + ["nor_059"], score=Q["단독 MLP (규칙 없음)"][0]["oof"][0])
ex.save_fig(fig, "tsne", "MLP 점수가 높은 정상은 피처 공간에서 고장 옆에 있는가")
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_missed", "남은 오류 파일의 PSD (EXP-030과 같은 파일)")

# %% 중첩 CV: 통과한 구성만
if passed:
    from src.experts import warm_cache, ae_segerr, et_trees
    warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
    for nm in [BASE] + passed:
        ex.record("중첩: " + nm, nested_cv(F, y, g, CONFIGS[nm], n_jobs=1), names, main=(nm == passed[0]))
