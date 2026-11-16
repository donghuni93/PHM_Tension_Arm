# %% [markdown]
# EXP-025 유격 반복 규칙 — 정상 중심에서 멀고(r ≥ 2) 타격 크기가 고르면(2위/1위 ≥ 0.7) 무조건 고장
# - 출발: 2-12 (유격 반복 · 하중 대비 고주파). 사용자 결정 11-14: "r > 2 이고 R ≥ 0.7이면 무조건 고장"
# - 가설 H1′ (유격 타격): 간극이 있으면 하중 크기와 무관하게 타격이 나고, 같은 간극·같은 질량이 부딪치므로 크기가 고르다
#   K1 유격 반복 = log10(두 번째로 큰 사건 / σ) · K2 하중 대비 고주파 = log10(저주파 RMS / σ) · K3 = 2위/1위 사건 높이
#   r = K1·K2를 학습 정상의 중앙값·MAD로 맞춘 z의 RMS (폴드 안)
# - 구성 (주 구성 실행 전 지정 = 1)
#   0 기준 EXP-021 (AE + ET → 로지스틱 + 8.3k 규칙)
#   1 + 유격 반복 규칙 r ≥ 2 & K3 ≥ 0.7 (주)
# - 주의: 문턱 r = 2, K3 = 0.7은 2-12에서 라벨을 본 뒤 정한 사후 값 (결정.md '사후 문턱 예외') → 점수는 낙관적.
#   축 K1·K2도 014를 분석하는 중에 나옴(선택 편향). val은 확인용으로만, val 보고 문턱 고치지 않음
# 실행: conda run -n qe-phm python 단계/5_모델비교/스크리닝/EXP-025_1114_유격반복규칙/run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, separation
from src.features import feature_table_knock
from src.experts import RuleOverride, Rule83, KnockRule, ae_mean, et_logit, ae_segerr, et_trees, warm_cache
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-025", "유격 반복 규칙 — 정상 중심에서 멀고 타격 크기가 고르면 고장", ["H1"],
                member="이동훈", model="ET + 스펙트럼 AE → 로지스틱 + 8.3k 규칙 + 유격 반복 규칙",
                changes="EXP-021에 '정상 중심 거리 r ≥ 2(유격 반복·하중 대비 고주파) 이고 2위/1위 사건 ≥ 0.7이면 고장' 규칙 추가",
                focus=("abn_014", "abn_017", "abn_022"))
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
nor = y == 0
K = cached_table("knock", feature_table_knock, X)
BASE = dict(experts=(ae_mean, et_logit), combine="lr")
R83, KN = Rule83(0.75, 150), KnockRule(2.0, 0.7)
CONFIGS = [("0 기준: EXP-021", dict(base_kw=BASE, rule=[R83])),
           ("1 + 유격 반복 규칙 (주)", dict(base_kw=BASE, rule=[R83, KN]))]
MAIN = CONFIGS[1][0]
warm_cache([ae_segerr, et_trees], seeds=SEEDS)
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")


# %% 그림 1: 규칙 지도 (150개 정상으로 중심을 잡은 설명용. 평가에서는 폴드 안 학습 정상으로)
kn = KnockRule(2.0, 0.7).fit(g)
Z = (K.values[:, :2] - kn.med_) / kn.mad_
r_all, R3 = kn.r(g), K["K3_top2_ratio"].values
fire = kn(g)
fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
ax = axs[0]
for lab, s, c, mk in (("정상", nor, NOR, "s"), ("고장", ~nor, ABN, "o")):
    ax.scatter(Z[s, 1], Z[s, 0], color=c, s=26, alpha=.6, lw=0, marker=mk, label=lab)
ax.scatter(Z[fire, 1], Z[fire, 0], s=80, facecolor="none", edgecolor="k", lw=1.3, label="규칙 켜짐 (r ≥ 2 & 2위/1위 ≥ 0.7)")
th = np.linspace(0, 2 * np.pi, 200)
ax.plot(2 * np.sqrt(2) * np.cos(th), 2 * np.sqrt(2) * np.sin(th), "k--", lw=.8)
for i in np.flatnonzero(r_all >= 2):
    ax.annotate(short(names[i]), (Z[i, 1], Z[i, 0]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR)
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
ax.set_xlabel("K2 하중 대비 고주파 (정상 기준 z)"); ax.set_ylabel("K1 유격 반복 (정상 기준 z)")
ax.set_title("정상 중심 거리 r = 2 (점선) 밖 + 타격 크기가 고르면 고장", fontsize=10)
ax = axs[1]
o = r_all >= 2
for lab, s, c, mk in (("정상", o & nor, NOR, "s"), ("고장", o & ~nor, ABN, "o")):
    ax.scatter(r_all[s], R3[s], color=c, s=40, alpha=.8, lw=0, marker=mk, label=lab)
for i in np.flatnonzero(o):
    ax.annotate(short(names[i]), (r_all[i], R3[i]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
ax.axhline(.7, color="k", ls="--", lw=.8, label="2위/1위 = 0.7")
ax.set_xlabel("정상 중심 거리 r"); ax.set_ylabel("K3 2위/1위 사건 높이"); ax.grid(alpha=.3); ax.legend(fontsize=8)
ax.set_title("r ≥ 2 파일만: 2위/1위 ≥ 0.7 위에 고장만 있는가", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "rule_map", "r ≥ 2 밖이고 2위/1위 ≥ 0.7인 영역(검은 테두리)에 고장만 들어가는가, 014·017이 들어가는가")
ex.note("규칙이 켜지는 파일 (150개 정상 중심, 설명용)", " ".join(short(names[i]) for i in np.flatnonzero(fire)))


# %% 그림 2: t-SNE (매 실험 기본) — K 공간
fig, _ = viz.tsne({"K1·K2·K3 (유격 반복)": K.values}, y, names, mark=["abn_014", "abn_017", "abn_022", "nor_009", "nor_054", "nor_086"])
ex.save_fig(fig, "tsne", "유격 반복 공간에서 014·017이 고장 무리에 있는가, 이웃 정상 009·054·086은 어디인가")


# %% 규칙이 폴드 안에서 켜진 파일 (seed 5 × 바깥 5-fold)
fired = {}
for s in range(5):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g):
        for i in te[KnockRule(2.0, 0.7).fit(tr)(te)]:
            fired[names[i]] = fired.get(names[i], 0) + 1
ex.note("유격 규칙이 켜진 파일 (seed 5 × 바깥 5-fold, 횟수/5)", " ".join(f"{short(k)}({v})" for k, v in sorted(fired.items())))
print("fired:", fired, flush=True)


# %% quick_cv 분리 판정
rows = []
for name, kw in CONFIGS:
    q = quick_cv(g[:, None], y, g, lambda kw=kw: RuleOverride(**kw), seeds=SEEDS)
    s = separation(q["oof"], y, names)
    rows.append(f"{name:30s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 확실한 고장 {s['sure_a']} · "
                f"회색 고장 {' '.join(x[4:] for x in s['gray_a'])} · 위험 정상 {s['risk_n']:.1f}")
    print(rows[-1], flush=True)
ex.note("quick_cv 분리 판정 (seed 3)", "\n".join(rows))


# %% 중첩 CV F2
warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
for name, kw in (CONFIGS[1], CONFIGS[0]):
    r_ = nested_cv(g[:, None], y, g, lambda kw=kw: RuleOverride(**kw), n_jobs=-1)
    ex.record("중첩: " + name, r_, names, main=(name == MAIN))
print("done", flush=True)
