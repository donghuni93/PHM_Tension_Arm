# %% [markdown]
# EXP-025 그림 errors — 최종 모델이 왜 이 파일들만 틀리나 (run.py는 기록 후 고치지 않으므로 별도 파일)
# 왼쪽: 파일별로 seed 5번 중 틀린 횟수, 기준 EXP-021 vs 최종 EXP-025 (result.json)
# 오른쪽: seed마다 (점수 − 그 바깥 fold 임계값). 0 위 = 고장 판정. 규칙(+100)이 켜진 파일은 맨 위 띠에 모음
#         고장이 0 아래 = 미탐, 정상이 0 위 = 오경보. nested_cv와 같은 분할·임계값 규칙으로 다시 계산
# 실행: conda run -n qe-phm python 단계/5_모델비교/2_규칙발견/EXP-025_1114_유격반복규칙/fig_errors.py

# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import _oof_scores, best_f2_threshold, _scores
from src.experts import RuleOverride, Rule83, KnockRule, ae_mean, et_logit
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y, names = load_train(); g = np.arange(len(y)); F = g[:, None]
NOR, ABN = "#2a78d6", "#eb6834"
s = lambda n: n.replace("abn_", "A").replace("nor_", "N")
KW = dict(base_kw=dict(experts=(ae_mean, et_logit), combine="lr"), rule=[Rule83(0.75, 150), KnockRule(2.0, 0.7)])

# %% seed별 (점수 − 임계값)
CACHE_M = os.path.join(HERE, "figs", "errors_margin.npy")         # 있으면 다시 계산하지 않음 (지우면 다시 계산)
M = np.load(CACHE_M) if os.path.exists(CACHE_M) else np.full((5, len(y)), np.nan)
for seed in ([] if os.path.exists(CACHE_M) else range(5)):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(F, y, g):
        oof = _oof_scores(F[tr], y[tr], g[tr], lambda: RuleOverride(**KW), StratifiedGroupKFold(4, shuffle=True, random_state=seed))
        t = best_f2_threshold(y[tr], oof)
        M[seed, te] = _scores(RuleOverride(**KW).fit(F[tr], y[tr]), F[te]) - t
np.save(os.path.join(HERE, "figs", "errors_margin.npy"), M)

# %% 틀린 횟수 (result.json)
r25 = json.load(open(os.path.join(HERE, "result.json"), encoding="utf-8"))
c25 = r25["configs"][r25["main"]]
r21 = json.load(open(os.path.join(HERE, "..", "EXP-021_1114_8k모드규칙", "result.json"), encoding="utf-8"))
c21 = r21["configs"][r21["main"]]
cnt = lambda c, n: c["miss_counts"].get(n, 0) + c["fp_counts"].get(n, 0)
ERR = sorted({n for c in (c21, c25) for n in list(c["miss_counts"]) + list(c["fp_counts"])},
             key=lambda n: (n.startswith("nor"), -cnt(c25, n), -cnt(c21, n), n))

# %% 그림
CAP = 4.0                                                  # 규칙이 켜진 파일(+100)은 이 위 띠에 모음
fig, (a1, a2) = plt.subplots(1, 2, figsize=(18, 8.5), gridspec_kw=dict(width_ratios=[1, 1.5]))
yy = np.arange(len(ERR))
a1.barh(yy + .2, [cnt(c21, n) for n in ERR], height=.4, color="#bbb", label="기준 EXP-021")
a1.barh(yy - .2, [cnt(c25, n) for n in ERR], height=.4, color=[ABN if n.startswith("abn") else NOR for n in ERR], label="최종 EXP-025")
a1.axvline(3, color="k", ls="--", lw=.8)
a1.set_yticks(yy); a1.set_yticklabels([("미탐 " if n.startswith("abn") else "오경보 ") + s(n) for n in ERR], fontsize=8.5)
a1.invert_yaxis(); a1.set_xlim(0, 5.3); a1.set_xlabel("seed 5번 중 틀린 횟수 (점선 3 = '자주 틀림')"); a1.grid(alpha=.3, axis="x")
a1.legend(fontsize=8, loc="lower right"); a1.set_title("파일별 틀린 횟수: 기준 → 최종", fontsize=10)

rng = np.random.default_rng(0)
for k in range(5):
    v = np.clip(M[k], -6, None); v = np.where(v > CAP, CAP + .4 + rng.uniform(-.2, .2, len(v)), v)
    jit = k + rng.uniform(-.28, .28, len(y))
    for lab, sel, c, mk in (("정상", y == 0, NOR, "s"), ("고장", y == 1, ABN, "o")):
        a2.scatter(jit[sel], v[sel], color=c, s=14, alpha=.45, lw=0, marker=mk, label=lab if k == 0 else None)
    wrong = np.flatnonzero(((y == 1) & (M[k] < 0)) | ((y == 0) & (M[k] >= 0)))
    for i in wrong:
        a2.scatter(jit[i], v[i], s=70, facecolor="none", edgecolor="k", lw=1.2)
        a2.annotate(s(names[i]), (jit[i], v[i]), xytext=(5, 0), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
a2.axhline(0, color="k", lw=1.2); a2.axhspan(CAP + .1, CAP + .7, color="#eee", zorder=0)
a2.text(4.45, CAP + .4, "규칙 켜짐 (+100)", fontsize=8, va="center", ha="right")
a2.set_xticks(range(5)); a2.set_xticklabels([f"seed {k}" for k in range(5)]); a2.set_ylim(-6.3, CAP + .9)
a2.set_ylabel("점수 - 그 fold 임계값  (0 위 = 고장 판정, -6 아래는 -6에 모음)")
a2.grid(alpha=.3, axis="y"); a2.legend(fontsize=8, loc="lower right")
a2.set_title("seed마다 0선을 넘은 파일 = 틀림 (검은 원). 고장은 0 아래면 미탐, 정상은 0 위면 오경보", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "errors.png"), dpi=80)

# %% 수치
out = []
for k in range(5):
    w = np.flatnonzero(((y == 1) & (M[k] < 0)) | ((y == 0) & (M[k] >= 0)))
    out.append(f"seed {k}: " + " · ".join(f"{s(names[i])} {M[k, i]:+.2f}" for i in w))
for n in ("abn_022", "abn_005", "nor_025", "nor_116", "nor_055", "nor_059"):
    out.append(f"{n} 여유(점수 − 임계값) seed별: " + " ".join(f"{M[k, names.index(n)]:+.2f}" for k in range(5)))
open(os.path.join(HERE, "figs", "errors.txt"), "w", encoding="utf-8").write("\n".join(out))
print("done")
