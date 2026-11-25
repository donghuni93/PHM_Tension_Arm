# %% [markdown]
# EXP-020 8–9 kHz 봉우리 축 — 정상 봉우리 덮개를 넘는 봉우리를 EXP-014 결합에 한 열 더한다
# - 출발: EXP-014 README2 (8–9 kHz 봉우리만 본 그림). 큰 봉우리가 8.20–8.45 kHz면 고장(5/6), 8.45–8.75 kHz면 정상(8/8).
#   지금 모델은 이 위치 정보를 못 쓴다: ET B04_ratio_08125는 8.125–8.75 kHz를 한 칸으로 묶고, AE는 1025칸 평균
# - 가설 H2′ (2-9): 고장에서 8.3 kHz 모드가 크게 들뜬다. 정상은 8.6–8.7 kHz 모드가 흔하다
# - 주 구성 (실행 전 지정): 정상 봉우리 덮개 초과 PeakExceed(8–9 kHz, q90)
#   학습 폴드 정상만으로 주파수별 돌출 90%선(덮개) → 파일의 (돌출 − 덮개) 최댓값. 대역 경계를 손으로 정하지 않는다
#   단, 8–9 kHz 창 자체는 2-9·2-10·README2를 본 뒤 고른 것 → 점수는 낙관적일 수 있다
# - 비교: [사후] 두 모드 상대 크기(8.20–8.45 − 8.45–8.75 kHz, 라벨 보고 정한 경계) / 일반화: 같은 덮개를 2–12 kHz 전체에
# - 평가: quick_cv(5-fold × seed 3)로 비교, 주 구성은 중첩 CV F2 (기준 EXP-014 0.857 ± 0.031)
# - 통과 기준 (EXP-019와 같음, 실행 전 고정): ① PR-AUC ≥ 기준 + 2 × 기준 sd  ② 기준 pct < 0.90 회색 고장 2개 이상이 ≥ 0.90, 회색 정상 평균 pct 상승 없음
# 실행: conda run -n qe-phm python 단계/5_모델비교/아카이브/EXP-020_1114_8k봉우리축/run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv
from src.experts import (ScoreStack, ae_mean, ae_bins, et_logit, et_trees, ae_segerr, warm_cache, LP,
                         PeakExceed, mode_contrast, prominence)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-020", "8–9 kHz 봉우리 축 — 정상 봉우리 덮개 초과를 ET·AE 결합에 추가", ["H2"],
                member="이동훈", model="ET + 스펙트럼 AE + 8–9 kHz 봉우리 축 → 정상 기준 z → L2 로지스틱",
                changes="EXP-014 결합에 '학습 정상의 주파수별 돌출 90%선을 넘는 봉우리 최댓값(8–9 kHz)' 한 열 추가", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
nor = y == 0
GRAY_A = ["abn_002", "abn_005", "abn_014", "abn_017", "abn_022", "abn_025", "abn_029"]
GRAY_N = ["nor_025", "nor_027", "nor_055", "nor_059", "nor_060", "nor_062", "nor_073", "nor_086", "nor_106", "nor_116"]
HARD = GRAY_A + ["abn_006", "abn_010", "abn_030"] + GRAY_N + ["nor_019", "nor_045", "nor_009", "nor_066", "nor_021", "nor_092"]

n = warm_cache([ae_segerr, et_trees], seeds=SEEDS)
print(f"캐시 완료: 분할 {n}개", flush=True)


# %% quick_cv
Z0 = (ae_mean, et_logit)
MAIN_AX = PeakExceed(8000, 9000, 90)
CONFIGS = [("0 기준: 로지스틱(AE, ET) (EXP-014)", Z0),
           ("1 + 8–9 kHz 정상 덮개 초과 (주)", Z0 + (MAIN_AX,)),
           ("2 [사후] + 두 모드 상대 크기 (8.20–8.45 − 8.45–8.75)", Z0 + (mode_contrast,)),
           ("3 일반화: + 2–12 kHz 정상 덮개 초과", Z0 + (PeakExceed(2000, 12000, 90),))]
R, rows = {}, []
for name, exps in CONFIGS:
    q = quick_cv(g[:, None], y, g, lambda exps=exps: ScoreStack(exps, combine="lr"), seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    R[name] = q
    rows.append(f"{name:50s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · ROC-AUC {q['roc_auc'].mean():.3f} ± {q['roc_auc'].std():.3f}")
    print(rows[-1], flush=True)
cfg = [c[0] for c in CONFIGS]
MAIN = cfg[1]
b = R[cfg[0]]
for c in cfg[1:]:
    q = R[c]
    up = [n for n in GRAY_A if b["pct"][names.index(n)] < .9 <= q["pct"][names.index(n)]]
    dn_n = np.mean([q["pct"][names.index(n)] for n in GRAY_N]) - np.mean([b["pct"][names.index(n)] for n in GRAY_N])
    ok1 = q["pr_auc"].mean() >= b["pr_auc"].mean() + 2 * b["pr_auc"].std()
    ok2 = len(up) >= 2 and dn_n <= 0
    rows.append(f"  {c.split()[0]} 통과 기준: ① {'통과' if ok1 else '미달'} (차 {q['pr_auc'].mean() - b['pr_auc'].mean():+.3f}) · "
                f"② {'통과' if ok2 else '미달'} (0.90 넘은 회색 고장 {len(up)}개 {' '.join(u[4:] for u in up)}, 회색 정상 평균 pct 변화 {dn_n:+.3f})")
ex.note("quick_cv 결과 (5-fold × seed 3, 임계값 없음) · 통과 기준", "\n".join(rows))
for c in cfg:
    pm = R[c]["pct"]
    ex.note(f"{c.split()[0]} 정상 90% 선 아래 고장 · 정상 상위 5%",
            f"고장: {' '.join(names[i].replace('abn_', '') for i in np.flatnonzero((y == 1) & (pm < .9)))}\n"
            f"정상: {' '.join(names[i].replace('nor_', '') for i in np.flatnonzero((y == 0) & (pm > .95)))}")


# %% 그림 1: 로지스틱 구분선 — x = AE·ET 부분 점수(계수 가중합), y = 8k 축 z. 결정값 0 선은 이 평면에서 정확히 직선
P2, W = np.empty((len(y), 2)), []
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    m = ScoreStack(CONFIGS[1][1], combine="lr").fit(g[tr][:, None], y[tr])
    lr = m.m_[-1]; Zt = m.zscores(g[te][:, None])
    P2[te, 0] = Zt[:, :2] @ lr.coef_[0][:2]; P2[te, 1] = Zt[:, 2]
    W.append((lr.coef_[0], lr.intercept_[0]))
c8 = np.mean([w[0][2] for w in W]); b0 = np.mean([w[1] for w in W]); cA = np.mean([w[0][0] for w in W]); cE = np.mean([w[0][1] for w in W])
LO = (-6, -3); HI = (8, 8)
pc = np.column_stack([np.clip(P2[:, 0], LO[0], HI[0]), np.clip(P2[:, 1], LO[1], HI[1])])
fig, ax = plt.subplots(figsize=(10, 8))
ax.scatter(*pc[y == 0].T, s=20, color=NOR, alpha=.55, lw=0, label="정상")
ax.scatter(*pc[y == 1].T, s=32, color=ABN, lw=0, label="고장")
ys = np.linspace(LO[1], HI[1], 50)
ax.plot(-c8 * ys - b0, ys, color="k", lw=1.3, label=f"로지스틱 구분선 (결정값 0): x + {c8:.2f}·y + {b0:.2f} = 0")
ax.axvline(-b0, color="k", lw=.8, ls=":", label="8k 축이 없을 때 위치 (y = 0 기준)")
for nm in HARD:
    i = names.index(nm)
    ax.annotate(nm.replace("abn_", "A").replace("nor_", "N"), pc[i], xytext=(4, 3), textcoords="offset points",
                fontsize=7.5, color=ABN if y[i] else NOR)
ax.set_xlim(LO[0], HI[0]); ax.set_ylim(LO[1], HI[1]); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper right")
ax.set_xlabel(f"AE·ET 부분 점수 = {cA:.2f}·z_AE + {cE:.2f}·z_ET (폴드 평균 계수)")
ax.set_ylabel("8–9 kHz 정상 덮개 초과 z (1 = 정상 90%선)")
ax.set_title("로지스틱 구분선 (OOF seed 0). 위로 갈수록 정상에 드문 8–9 kHz 봉우리. 선의 기울기 = 8k 축이 경계를 미는 정도", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "boundary_lr", "A002·A029가 위로 올라가 구분선을 넘는가, 8.6–8.7 kHz 봉우리 정상(N045·009·066·021)은 아래에 남는가")


# %% 그림 2: t-SNE — 출력 끝단(AE+ET) / + 8–9 kHz 돌출 곡선 (블록 같은 비중). 2행 색 = 주 구성 OOF 점수
A, T = np.empty((len(y), LP.shape[-1])), np.empty((len(y), 300))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    a, bb = tuple(tr.tolist()), tuple(te.tolist())
    A[te], T[te] = ae_bins(a, bb), et_trees(a, bb)
f, PR = prominence()
K = PR[:, (f >= 8000) & (f <= 9000)]
std = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-12)
blk = lambda *Bs: np.hstack([std(B) / np.sqrt(B.shape[1]) for B in Bs])
fig, _ = viz.tsne({"출력 끝단: AE bin 오차 + ET 나무 (EXP-014)": blk(A, T),
                   "출력 끝단 + 8–9 kHz 돌출 곡선": blk(A, T, K)},
                  y, names, mark=HARD, score=R[MAIN]["oof"][0], scale=False)
ex.save_fig(fig, "tsne", "8–9 kHz 돌출을 더하면 A002·A029가 고장 무리로 옮겨 가는가. 2행 색 = 주 구성 OOF 점수(seed 0)")


# %% 그림 3: hard_files — 기준 / 주 / 사후 / 일반화 + 8k 축 단독
q8 = quick_cv(g[:, None], y, g, lambda: ScoreStack((MAIN_AX,), combine="col0"), seeds=SEEDS)
q8["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q8["oof"]], 0)
ex.note("8–9 kHz 덮개 초과 축 단독", f"PR-AUC {q8['pr_auc'].mean():.3f} ± {q8['pr_auc'].std():.3f} · ROC-AUC {q8['roc_auc'].mean():.3f}")
cols = [(c.split(":")[0].split(" (")[0].replace("[사후] ", "[사후]\n").replace("일반화", "일반화\n"), R[c]) for c in cfg] + [("8k 축 단독", q8)]
M = np.array([[1 - q["pct"][names.index(nm)] if nm.startswith("abn") else q["pct"][names.index(nm)] for _, q in cols] for nm in HARD])
fig, ax = plt.subplots(figsize=(11, 11))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < .7 else "w")
lab = [("놓침 " if n.startswith("abn") else "오경보 ") + n for n in HARD]
lab = [l.replace("놓침 abn_006", "참고 abn_006 (8.3k 봉우리)").replace("놓침 abn_010", "참고 abn_010 (8.3k)").replace("놓침 abn_030", "참고 abn_030 (8.3k)")
       for l in lab]
lab = [l if not any(k in l for k in ("019", "045", "009", "066", "021", "092")) else l.replace("오경보", "참고 8k 정상") for l in lab]
ax.set_yticks(range(len(HARD))); ax.set_yticklabels(lab, fontsize=8)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for c, _ in cols], fontsize=8)
ax.set_title("어려운 파일 위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "8k 축으로 002·029가 옅어지는가, 8.6–8.7 kHz 봉우리 정상(045·009·066·021)과 8.25 kHz 정상(019)·092가 진해지지 않는가")


# %% 보조: 놓친 고장 PSD (7.5–9.5 kHz 확대 포함)
missed = [names[i] for i in np.flatnonzero((y == 1) & (R[MAIN]["pct"] < .9))]
fL, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(fL, L, y, names, missed, ranges=((0, 12500), (7500, 9500)), title="주 구성이 정상 90% 선 아래로 둔 고장"),
                "psd_missed", "(보조) 놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)


# %% 최종 확인: 주 구성 중첩 CV F2 (기준은 같은 코드로 0.857 ± 0.031, EXP-017에서 재확인)
warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
r = nested_cv(g[:, None], y, g, lambda: ScoreStack(CONFIGS[1][1], combine="lr"), n_jobs=-1)
ex.record("중첩: " + MAIN, r, names, main=True)
print("nested done", flush=True)
