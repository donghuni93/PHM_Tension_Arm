# %% [markdown]
# EXP-029 기본 그림 보충 (11-16, 사용자): run.py 기록 후라 결과는 그대로 두고 그림만 추가
# ① logistic_plane (핵심 그림): 두 전문가 z 평면(가로 AE z, 세로 ET z, OOF seed 0) + 로지스틱 경계(실선 0, 점선 ±1)
#    - 왼쪽 = 기준 EXP-025 (ET = ext150), 오른쪽 = 주 구성 7 (ET = ext150 + 큰 타격 + 사건 울림)
#    - **규칙(8.3k · 유격 · 주기 잠금)으로 잡힌 고장은 그림에서 뺀다** (경계와 무관하게 고장이므로). 규칙이 켜진 정상은 오류라 남긴다
#    - 경계 = 그 평면 점 150개 전체로 맞춘 로지스틱(설명용, 실제 모델도 규칙 고장을 포함해 학습). 회색 고리 = 단일 타격 감점 대상
# ③ hard_files: 어려운 파일의 위험도 (quick_cv seed 3 평균), 열 = 구성
# t-SNE는 run.py 그림(모든 파일 포함)을 그대로 쓴다
# 실행: conda run -n qe-phm python "단계/5_모델비교/유의미/EXP-029_1116_큰타격수·사건울림·주기잠금/figs_기본.py"

# %% 설정
import os, sys, warnings
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
_d = HERE
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from src.exp import *
from src.evaluate import quick_cv
from src.experts import (RuleOverride, Rule83, KnockRule, LockRule, SingleHitCap, ScoreStack, ETLogitX,
                         ae_mean, et_logit, HIT_COLS, EV_COLS, _hits)
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(os.path.join(HERE, "run.py"), "EXP-029", "큰 타격 수 · 사건 울림 · 주기 잠금 — ET와 최종 결과 변화", ["H1", "H4"],
                member="이동훈", model="ET(+HIT·EV) + 스펙트럼 AE → 로지스틱 + 8.3k·유격·주기 잠금 규칙 + 단일 타격 감점",
                changes="EXP-025에 큰 타격 수(2-14)·사건 울림(2-15)·주기 잠금(2-13)을 ET 입력/결합 축/규칙으로 하나씩 추가",
                focus=("abn_022", "abn_020", "nor_025", "nor_116", "nor_055"))
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]; nor = y == 0
H = _hits(); n20 = H.HIT_n20.values
s = lambda n: n.replace("abn_", "A").replace("nor_", "N")
ETHE = ETLogitX(HIT_COLS + EV_COLS)
PLANES = [("기준 EXP-025 (ET = ext150)", (ae_mean, et_logit), False, "중첩: 0 기준 EXP-025"),
          ("주 구성 7 (ET = ext150 + 큰 타격 + 사건 울림)", (ae_mean, ETHE), True, "중첩: 7 전부-ET (주)")]
res = ex.res["configs"]

# %% ① logistic_plane
xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 400), np.linspace(-3, 4.5, 400)); G = np.c_[xx.ravel(), yy_.ravel()]
fig, axs = plt.subplots(1, 2, figsize=(17, 8.5))
for ax, (title, experts, main, key) in zip(axs, PLANES):
    Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=experts, combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        on = Rule83(0.75, 150)(te) | KnockRule(2.0, 0.7).fit(tr)(te)
        if main: on |= LockRule(0.1).fit(tr)(te)
        rule[te] = on
    Zc = np.clip(Z, -3, 4.5)
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(G).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.6)
    ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.8, linestyles="--")
    show = ~(rule & (y == 1))                                  # 규칙으로 잡힌 고장은 뺀다
    for lab, sel, col, mk in (("정상", nor & show, NOR, "s"), ("고장 (규칙 밖)", ~nor & show, ABN, "o")):
        ax.scatter(Zc[sel, 0], Zc[sel, 1], color=col, s=28, lw=0, marker=mk, label=lab, alpha=.85)
    rn = rule & nor
    if rn.any():
        ax.scatter(Zc[rn, 0], Zc[rn, 1], s=130, marker="D", facecolor="none", edgecolor="#7a3db8", lw=1.4, label="규칙이 켜진 정상 (오경보)")
    if main:
        cp = (n20 == 1) & show
        ax.scatter(Zc[cp, 0], Zc[cp, 1], s=80, facecolor="none", edgecolor="#555", lw=1, label="단일 타격 감점 대상 (큰 타격 1개)")
    c = res[key]
    wrong = [names.index(n) for n, k in {**c["miss_counts"], **c["fp_counts"]}.items() if k >= 2]
    ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=90, marker="x", color="k", lw=1.8, label="자주 틀림 (5번 중 2번 이상)")
    for i in wrong:
        k = c["miss_counts"].get(names[i]) or c["fp_counts"].get(names[i])
        ax.annotate(f"{s(names[i])} {'미탐' if y[i] else '오경보'} {k}/5", Zc[i], xytext=(5, 4), textcoords="offset points",
                    fontsize=8.5, color=ABN if y[i] else NOR, weight="bold")
    for nm in ("nor_116", "nor_055", "nor_027", "abn_015", "abn_016"):
        i = names.index(nm)
        if i not in wrong and show[i]:
            ax.annotate(s(nm), Zc[i], xytext=(5, -9), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR)
    gone = [s(names[i]) for i in np.flatnonzero(rule & (y == 1))]
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (충격 피처 이상)"); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"{title}\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 규칙으로 잡혀 그림에서 뺀 고장 {len(gone)}개", fontsize=9.5)
    ax.text(.98, .02, "뺀 고장 (규칙):\n" + " ".join(gone[:7]) + "\n" + " ".join(gone[7:]), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
fig.tight_layout(rect=(0, 0, 1, .96))
ex.save_fig(fig, "logistic_plane", "규칙 밖 고장과 정상만 남긴 AE z × ET z 평면. 기준 → 주 구성에서 오경보 116·055가 경계 밖으로 빠지는가, 022가 규칙으로 빠지는가", key=True)

# %% ③ hard_files: 어려운 파일 위험도 (quick_cv seed 3)
B0 = dict(experts=(ae_mean, et_logit), combine="lr"); R83, KN, LK = Rule83(0.75, 150), KnockRule(2.0, 0.7), LockRule(0.1)
CFG = [("0 기준", RuleOverride, dict(base_kw=B0, rule=[R83, KN])),
       ("1 ET+큰타격", RuleOverride, dict(base_kw=dict(experts=(ae_mean, ETLogitX(HIT_COLS)), combine="lr"), rule=[R83, KN])),
       ("3 감점", SingleHitCap, dict(base_kw=B0, rule=[R83, KN])),
       ("6 주기잠금", RuleOverride, dict(base_kw=B0, rule=[R83, KN, LK])),
       ("7 전부(주)", SingleHitCap, dict(base_kw=dict(experts=(ae_mean, ETHE), combine="lr"), rule=[R83, KN, LK]))]
HARD = ["abn_022", "abn_005", "abn_020", "abn_025", "abn_014", "abn_017", "abn_015", "abn_016", "abn_002",
        "nor_025", "nor_116", "nor_055", "nor_027", "nor_059", "nor_103", "nor_106", "nor_054", "nor_065"]
M = []
for nm, C, kw in CFG:
    o = quick_cv(F, y, g, lambda C=C, kw=kw: C(**kw), seeds=range(3))["oof"]
    pct = np.array([(o[:, nor] < o[:, [i]]).mean() for i in range(len(y))])     # 정상 중 이 파일보다 낮은 비율
    M.append([1 - pct[names.index(h)] if h.startswith("abn") else pct[names.index(h)] for h in HARD])
    print(nm, "done", flush=True)
M = np.array(M).T
fig, ax = plt.subplots(figsize=(9.5, 10))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h + f"  (큰 타격 {n20[names.index(h)]}개)" for h in HARD], fontsize=8)
ax.set_xticks(range(len(CFG)), [c[0] for c in CFG], fontsize=9)
ax.set_title("위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율\n(진할수록 틀리기 쉬움)", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "감점으로 116·055·027·065가 옅어지는가, 주기 잠금으로 022가 0이 되는가, 103이 진해지는가")

# 핵심 그림 = logistic_plane 하나 (boundary_lr은 보조로 내림)
ex.res["key_figs"] = ["logistic_plane"]
ex.res["figs"]["boundary_lr"] = "(보조) 기준 결정값 × 큰 타격 수: 큰 타격 1개 줄(회색 띠)에 고장이 없는가, 큰 타격 0개 고장(002 015 016)은 어디인가"
ex._write()
print("done", flush=True)
