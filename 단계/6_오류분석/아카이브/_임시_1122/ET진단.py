# 임시 (11-22 밤): ET 진단 — J의 ET(고정 H 열 + 봉우리 H16 · H17, ExtraTrees 300 · leaf 2 · sqrt · balanced_subsample)
# 그림7 = 피처별 기여: (왼) 한 열 빼고 다시 학습했을 때 폴드 밖 PR-AUC 변화, 폴드 10개(5겹 × seed 2) 각각 점 + 바닥 잡음(ET seed만 바꾼 차이)
#                     (오) 폴드 밖 순열 중요도(PR-AUC 하락), 폴드 10개 각각 — 폴드마다 일관된가
# 그림8 = 겹침: ET 열 + 박자 S + AE z의 스피어만 상관(정상 · 고장 함께), 군집 순서. AE와 겹치는 ET 열이 어느 것인가
# 라벨은 학습 · 채점에만, 폴드 밖으로 잰다. 경계 파일은 보지 않는다
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.metrics import average_precision_score as ap, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from src.exp import *
from src.exp2util import *
from src.features2 import PeakRef
from src.models2 import H_FOLD
HERE = os.path.dirname(os.path.abspath(__file__))
ns = NS(); y, g = ns.y, ns.g; nor = y == 0
FX, PK = list(ns.FIXED), list(ns.PEAK); COLS = FX + PK
out = []; P = lambda s="": (out.append(str(s)), print(s))
P(f"ET 입력 {len(COLS)}열: {', '.join(COLS)}")


def et_fold(tr, te, drop=None, seed=0):
    """폴드 안: 봉우리 기준을 학습 정상으로 맞추고 ET 학습 → 폴드 밖 확률 (te,), 학습 모델, 폴드 밖 행렬"""
    pr = PeakRef(ns.df).fit(ns.L[tr][y[tr] == 0]); pk = pr.transform(ns.L)[:, [H_FOLD.index(p) for p in PK]]
    H = np.column_stack([ns.T[FX].values, pk]); keep = [j for j, c in enumerate(COLS) if c != drop]
    m = make_et(n=300, seed=seed)().fit(H[tr][:, keep], y[tr])
    return m.predict_proba(H[te][:, keep])[:, 1], m, H[te][:, keep]


SPL = [(s, tr, te) for s in range(2) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
base, noise, drop, perm = [], [], {c: [] for c in COLS}, {c: [] for c in COLS}
rs = np.random.RandomState(0)
for s, tr, te in SPL:
    p0, m0, Hte = et_fold(tr, te, seed=0); b = ap(y[te], p0); base.append(b)
    noise.append(ap(y[te], et_fold(tr, te, seed=1)[0]) - b)                    # 바닥 잡음: ET seed만 바꿈
    for j, c in enumerate(COLS):
        drop[c].append(ap(y[te], et_fold(tr, te, drop=c, seed=0)[0]) - b)
        dd = []
        for _ in range(10):
            Hp = Hte.copy(); Hp[:, j] = rs.permutation(Hp[:, j]); dd.append(ap(y[te], m0.predict_proba(Hp)[:, 1]) - b)
        perm[c].append(float(np.mean(dd)))
P(f"기준 ET 폴드 밖 PR-AUC {np.mean(base):.3f} ± {np.std(base):.3f} (폴드 10개). 바닥 잡음(ET seed만 바꿈) sd {np.std(noise):.3f}, |평균| {abs(np.mean(noise)):.3f}")
D = pd.DataFrame({"열": COLS, "빼기_평균": [np.mean(drop[c]) for c in COLS], "빼기_sd": [np.std(drop[c]) for c in COLS],
                  "빼기_하락폴드": [int(np.sum(np.array(drop[c]) < 0)) for c in COLS], "순열_평균": [np.mean(perm[c]) for c in COLS],
                  "순열_하락폴드": [int(np.sum(np.array(perm[c]) < -1e-9)) for c in COLS]}).sort_values("빼기_평균")
P(D.round(4).to_string(index=False))

# %% 그림7
order = D["열"].tolist(); yy = np.arange(len(order))
fig, axs = plt.subplots(1, 2, figsize=(17, .45 * len(order) + 2.2), sharey=True)
ns_sd = np.std(noise)
for ax, Dd, ttl, xl in ((axs[0], drop, "한 열 빼고 다시 학습: 폴드 밖 PR-AUC 변화 (− = 그 열이 필요)", "PR-AUC 변화 (열 뺀 ET − 기준 ET)"),
                        (axs[1], perm, "순열 중요도: 그 열을 섞었을 때 PR-AUC 변화 (− = 모델이 그 열을 씀)", "PR-AUC 변화 (열 섞음 − 기준)")):
    ax.axvspan(-2 * ns_sd, 2 * ns_sd, color="#ddd", alpha=.7, label=f"바닥 잡음 ±2 sd (ET seed만 바꿈, sd {ns_sd:.3f})")
    for r, c in enumerate(order):
        v = np.array(Dd[c]); ax.scatter(v, np.full(len(v), r) + rs.uniform(-.15, .15, len(v)), s=16, color="#555", alpha=.7, lw=0)
        ax.scatter(v.mean(), r, s=90, marker="D", color="#d62728" if v.mean() < -2 * ns_sd / np.sqrt(len(v)) else ("#2a78d6" if v.mean() > 2 * ns_sd / np.sqrt(len(v)) else "#999"), zorder=3)
    ax.axvline(0, color="#000", lw=1); ax.set_title(ttl, fontsize=9.5, loc="left"); ax.set_xlabel(xl); ax.grid(axis="x", alpha=.3)
axs[0].set_yticks(yy); axs[0].set_yticklabels([f"{c}  (하락 {D.set_index('열').빼기_하락폴드[c]}/10)" for c in order], fontsize=8.5)
axs[0].legend(fontsize=8, loc="lower right")
fig.suptitle(f"ET 피처 진단 — 폴드 10개(5겹 × seed 2), 기준 PR-AUC {np.mean(base):.3f}. ◆ 빨강 = 평균이 잡음보다 확실히 아래(필요) · 파랑 = 확실히 위(빼면 나음) · 회색 = 구분 안 됨",
             x=.01, ha="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림7_ET피처_기여.png"), dpi=110); plt.close(fig)

# %% 그림8: 겹침 (ET 열 + 박자 S + AE z), AE z = J의 OOF(seed 0) 값을 6-3과 같은 방식으로 다시 구하지 않고 ae_map_cols의 평균 열 사용
pr = PeakRef(ns.df).fit(ns.L[nor]); pk = pr.transform(ns.L)[:, [H_FOLD.index(p) for p in PK]]
V = pd.DataFrame(np.column_stack([ns.T[FX].values, pk]), columns=COLS); V["L1_S (박자)"] = ns.T["L1_S"].values
V["AE (복원 오차 평균, OOF)"] = ae_map_cols(ns)[:, 0]
R = spearmanr(V.values).correlation; o = leaves_list(linkage(1 - np.abs(R[np.triu_indices(len(R), 1)]), "average")); lab = list(V.columns)
fig, axs = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw=dict(width_ratios=[1.35, 1]))
ax = axs[0]; im = ax.imshow(R[np.ix_(o, o)], cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(o))); ax.set_yticks(range(len(o))); ax.set_xticklabels([lab[i] for i in o], rotation=60, ha="right", fontsize=8); ax.set_yticklabels([lab[i] for i in o], fontsize=8)
for a_ in range(len(o)):
    for b_ in range(len(o)):
        if a_ != b_ and abs(R[o[a_], o[b_]]) >= .5: ax.text(b_, a_, f"{R[o[a_], o[b_]]:.1f}", ha="center", va="center", fontsize=6.5, color="white" if abs(R[o[a_], o[b_]]) > .75 else "#222")
fig.colorbar(im, ax=ax, shrink=.7, label="스피어만 상관 (150개)"); ax.set_title("ET 열 · 박자 · AE 사이 겹침 (|ρ| ≥ 0.5만 숫자, 군집 순서)", fontsize=9.5, loc="left")
ax = axs[1]
ia = lab.index("AE (복원 오차 평균, OOF)"); ra = pd.Series(R[ia], index=lab).drop(lab[ia]).sort_values()
mr = D.set_index("열").빼기_평균.reindex(ra.index)
ax.barh(range(len(ra)), ra.values, color=["#d62728" if (not np.isnan(v) and v < -2 * ns_sd / np.sqrt(10)) else "#999" for v in mr.values])
ax.set_yticks(range(len(ra))); ax.set_yticklabels(ra.index, fontsize=8.5); ax.axvline(0, color="#000", lw=1); ax.grid(axis="x", alpha=.3)
ax.set_xlabel("AE와의 스피어만 상관"); ax.set_title("AE와 겹치는 ET 열 (빨강 = 그림7에서 '필요'로 나온 열)", fontsize=9.5, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림8_ET피처_겹침.png"), dpi=110); plt.close(fig)
P(); P("AE와 상관: " + " · ".join(f"{k} {v:+.2f}" for k, v in ra.sort_values(key=np.abs, ascending=False).items()))
open(os.path.join(HERE, "ET진단_log.txt"), "w", encoding="utf-8").write("\n".join(out))
