# %% [markdown]
# 유의미 실험마다 "로지스틱 최종 경계" 그림 (사용자 11-14) → 각 실험 폴더 figs/logistic_plane.png
# - 평면: 기준 결합의 두 전문가 점수 (가로 AE z, 세로 ET z), OOF seed 0 (그 파일을 학습에 쓰지 않은 모델의 점수)
# - 경계: 그 평면에 로지스틱을 맞춘 결정값 0선(실선)과 여백 ±1(점선). 설명용 — 실제 판정 임계값은 fold마다 안쪽 CV로 고른다
#   EXP-012(AE 단독)는 AE z 한 축 로지스틱 = 세로선. EXP-020은 추가 축이 평면 밖이라 EXP-014 경계를 그린다
# - 규칙: 그 실험에 들어 있는 규칙이 켜진 파일 (보라 마름모 = 8.3 kHz 모드 규칙, 초록 원 = 유격 반복 규칙) → 경계와 무관하게 고장
# - 틀린 파일: 그 실험 result.json 대표 구성에서 seed 5번 중 3번 이상 틀린 파일 (검은 X + 이름)
# 실행: conda run -n qe-phm python 단계/5_모델비교/fig_logistic_plane.py

# %% 설정
import os, sys, json, warnings
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.experts import ScoreStack, KnockRule, Rule83, ae_mean, et_logit
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y, names = load_train(); g = np.arange(len(y)); F = g[:, None]
NOR, ABN = "#2a78d6", "#eb6834"
s = lambda n: n.replace("abn_", "A").replace("nor_", "N")
Z = np.empty((len(y), 2))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    Z[te] = ScoreStack(experts=(ae_mean, et_logit), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
Zc = np.clip(Z, -3, 4.5)
R83 = Rule83(0.75, 150)(g); KN = KnockRule(2.0, 0.7).fit(g)(g)

EXPS = [("EXP-012_1113_스펙트럼AE_tSNE", "AE 단독: AE z 한 축 로지스틱 (세로선)", "ae", ()),
        ("EXP-014_1114_ET·AE결합", "AE + ET 로지스틱 결합", "both", ()),
        ("EXP-020_1114_8k봉우리축", "EXP-014 경계 (이 실험의 8–9 kHz 축은 평면 밖)", "both", ()),
        ("EXP-021_1114_8k모드규칙", "EXP-014 경계 + 8.3 kHz 모드 규칙", "both", ("r83",)),
        ("EXP-025_1114_유격반복규칙", "EXP-014 경계 + 8.3 kHz 모드 규칙 + 유격 반복 규칙 (최종)", "both", ("r83", "knock"))]
xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 400), np.linspace(-3, 4.5, 400))
G = np.c_[xx.ravel(), yy_.ravel()]

# %% 실험마다 그림
for folder, title, kind, rules in EXPS:
    d = os.path.join(HERE, "base", folder)
    r = json.load(open(os.path.join(d, "result.json"), encoding="utf-8"))
    c = r["configs"][r.get("main") or next(iter(r["configs"]))]
    miss = [n for n, k in c["miss_counts"].items() if k >= 3]; fp = [n for n, k in c["fp_counts"].items() if k >= 3]
    cols = [0] if kind == "ae" else [0, 1]
    m = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc[:, cols], y)
    D = m.decision_function(G[:, cols]).reshape(xx.shape)
    on = np.zeros(len(y), bool)
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.6)
    ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.8, linestyles="--")
    for lab, sel, col, mk in (("정상", y == 0, NOR, "s"), ("고장", y == 1, ABN, "o")):
        ax.scatter(Zc[sel, 0], Zc[sel, 1], color=col, s=26, lw=0, marker=mk, label=lab, alpha=.85)
    if "r83" in rules:
        ax.scatter(Zc[R83, 0], Zc[R83, 1], s=150, marker="D", facecolor="none", edgecolor="#7a3db8", lw=1.3, label="8.3 kHz 모드 규칙 켜짐 (무조건 고장)")
        on |= R83
    if "knock" in rules:
        ax.scatter(Zc[KN, 0], Zc[KN, 1], s=110, facecolor="none", edgecolor="#1a9e5a", lw=1.5, label="유격 반복 규칙 켜짐 (무조건 고장)")
        on |= KN
    wrong = [names.index(n) for n in miss + fp]
    if wrong:
        ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=90, marker="x", color="k", lw=1.8, label="이 실험이 자주 틀림 (3/5 이상)")
    for i in wrong:
        ax.annotate(s(names[i]) + (" 미탐" if y[i] else " 오경보"), Zc[i], xytext=(5, 4), textcoords="offset points",
                    fontsize=8.5, color=ABN if y[i] else NOR, weight="bold")
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (충격 피처 이상)")
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{r['id']} 로지스틱 최종 경계 — {title}\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 자주 놓침 {' '.join(c['missed']) or '없음'} · 자주 오경보 {len(fp)}개", fontsize=9.5)
    fig.tight_layout(); fig.savefig(os.path.join(d, "figs", "logistic_plane.png"), dpi=80); plt.close(fig)
    print(f"{r['id']}: 놓침 {' '.join(s(n) for n in miss)} · 오경보 {' '.join(s(n) for n in fp)} · 규칙 켜짐 {int(on.sum())}", flush=True)
print("done")
