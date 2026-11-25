# %% [markdown]
# EXP-025 그림 boundary_lr (EXP-021과 같은 형식) — run.py는 기록 후 고치지 않으므로 별도 파일
# x = 기준 결합(AE + ET 로지스틱) 결정값, OOF seed 0 · y = 정상 중심 거리 r (유격 반복·하중 대비 고주파)
# 점 크기 = 2위/1위 사건 높이(K3) · 검은 테두리 = 유격 반복 규칙 켜짐 (r ≥ 2 & K3 ≥ 0.7)
# 실행: conda run -n qe-phm python 단계/5_모델비교/2_규칙발견/EXP-025_1114_유격반복규칙/fig_boundary.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.features import feature_table_knock
from src.experts import ScoreStack, KnockRule, Rule83, ae_mean, et_logit
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore", category=ConvergenceWarning)

X, y, names = load_train(); g = np.arange(len(y))
NOR, ABN = "#2a78d6", "#eb6834"
K3 = cached_table("knock", feature_table_knock, X)["K3_top2_ratio"].values
BASE = dict(experts=(ae_mean, et_logit), combine="lr")
D0 = np.empty(len(y))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    D0[te] = ScoreStack(**BASE).fit(g[tr][:, None], y[tr]).decision_function(g[te][:, None])
kn = KnockRule(2.0, 0.7).fit(g); r = kn.r(g); fired = kn(g)

# %% 그림
HARD = ["abn_014", "abn_017", "abn_022", "abn_005", "abn_001", "abn_009", "abn_002", "abn_029",
        "nor_009", "nor_054", "nor_086", "nor_025", "nor_055", "nor_116", "nor_059", "nor_106"]
fig, ax = plt.subplots(figsize=(10, 8))
for lab, s, c, mk in (("정상", y == 0, NOR, "s"), ("고장", y == 1, ABN, "o")):
    ax.scatter(np.clip(D0[s], -8, 8), r[s], s=12 + 110 * K3[s], color=c, alpha=.6, lw=0, marker=mk, label=lab)
ax.scatter(np.clip(D0[fired], -8, 8), r[fired], s=12 + 110 * K3[fired], facecolor="none", edgecolor="k", lw=1.3,
           label="유격 반복 규칙 켜짐 (무조건 고장)")
f83 = Rule83(0.75, 150)(g)
ax.scatter(np.clip(D0[f83], -8, 8), r[f83], s=160, marker="D", facecolor="none", edgecolor="#7a3db8", lw=1.3,
           label="8.3 kHz 모드 규칙 켜짐 (EXP-021, 무조건 고장)")
ax.axvline(0, color="k", lw=1.2, label="로지스틱 구분선 (결정값 0)")
ax.axhline(2, color="k", lw=.8, ls="--")
for i in set(np.flatnonzero(fired)) | {names.index(n) for n in HARD}:
    ax.annotate(names[i].replace("abn_", "A").replace("nor_", "N"), (np.clip(D0[i], -8, 8), r[i]), xytext=(4, 3),
                textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR)
ax.set_xlim(-8, 8); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("기준 결합(AE + ET 로지스틱) 결정값, OOF seed 0")
ax.set_ylabel("정상 중심 거리 r (유격 반복 · 하중 대비 고주파). 점 크기 = 2위/1위 사건 높이")
ax.set_title("로지스틱 구분선 왼쪽(정상 판정)인데 규칙이 켜져 고장으로 바뀌는 파일 = 규칙의 이득", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "boundary_lr.png"), dpi=80)
log = [f"구분선 왼쪽(결정값 < 0)인데 규칙 켜짐: {' '.join(names[i] for i in np.flatnonzero(fired & (D0 < 0)))}",
       f"구분선 오른쪽 정상(결정값 > 0): {' '.join(names[i] for i in np.flatnonzero((y == 0) & (D0 > 0)))}",
       f"규칙 켜짐 파일의 결정값: " + " · ".join(f"{names[i]} {D0[i]:.2f}" for i in np.flatnonzero(fired))]
open(os.path.join(HERE, "figs", "boundary_lr.txt"), "w", encoding="utf-8").write("\n".join(log))
print("done")
