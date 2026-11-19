# %% [markdown]
# EXP-021 8.3 kHz 모드 규칙 — 8.20–8.45 kHz에 큰 돌출이 있으면 무조건 고장
# - 출발: 2-11 README2 (큰 봉우리가 8.20–8.45 kHz면 고장 5/6). 사용자 제안: "이 영역에 돌출이 있으면 무조건 고장으로 판별"
# - 실행 전 확인(사전 탐색, 11-14): 돌출만 보면 nor_019(0.96)와 abn_029(1.03)의 차이가 0.07 dec뿐이다.
#   봉우리 폭은 갈린다: 고장 006·010·002·029·030·007 = 198–275 Hz (넓음), 정상 019·004 = 92 Hz (좁음)
#   → 물리 해석: 고장의 8.3 kHz 공진은 충격으로 들떠 감쇠가 큰 넓은 봉우리, 정상은 좁고 꾸준한 순음 (H2′)
# - 구성 (주 구성은 실행 전 지정 = 2)
#   0 기준 EXP-014 (AE + ET → 로지스틱)
#   1 사용자 규칙: 8.20–8.45 kHz 돌출 ≥ 0.75 dec → 무조건 고장
#   2 규칙 + 폭: 돌출 ≥ 0.75 dec 이고 반높이 폭 ≥ 150 Hz → 무조건 고장 (주)
#   3 참고: EXP-020 [사후] 두 모드 상대 크기를 선형 입력으로 (규칙 아님)
# - 대역 경계·0.75 dec·150 Hz는 모두 그림을 본 뒤 정한 값 (결정.md '대역 예외') → 점수는 낙관적
# - 평가: quick_cv로 비교 + 규칙 구성(1·2)은 중첩 CV F2 (기준 0.857 ± 0.031)
# 실행: conda run -n qe-phm python 단계/5_모델비교/스크리닝/EXP-021_1114_8k모드규칙/run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv
from src.experts import (ScoreStack, RuleOverride, Rule83, peak83_table, ae_mean, ae_bins, et_logit, et_trees, ae_segerr,
                         mode_contrast, warm_cache, LP)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-021", "8.3 kHz 모드 규칙 — 8.20–8.45 kHz 넓은 봉우리면 무조건 고장", ["H2"],
                member="이동훈", model="ET + 스펙트럼 AE → 로지스틱, + 8.3 kHz 모드 규칙(돌출·폭) 무조건 고장",
                changes="EXP-014 결합 위에 '8.20–8.45 kHz 돌출 ≥ 0.75 dec 이고 폭 ≥ 150 Hz면 고장' 규칙을 덮어씀", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
nor = y == 0
P83, W83, P86 = peak83_table()
HARD = ["abn_002", "abn_029", "abn_006", "abn_010", "abn_030", "abn_007", "abn_005", "abn_014", "abn_017", "abn_022",
        "nor_019", "nor_004", "nor_092", "nor_070", "nor_025", "nor_055", "nor_059", "nor_062", "nor_106", "nor_116",
        "nor_045", "nor_009", "nor_066"]

n = warm_cache([ae_segerr, et_trees], seeds=SEEDS)
print(f"캐시 완료: 분할 {n}개", flush=True)

BASE = dict(experts=(ae_mean, et_logit), combine="lr")
R1, R2 = Rule83(0.75, 0), Rule83(0.75, 150)
CONFIGS = [("0 기준: EXP-014 (AE + ET → 로지스틱)", dict(base_kw=BASE, rule=None)),
           ("1 사용자 규칙: 8.3k 돌출 ≥ 0.75 → 고장", dict(base_kw=BASE, rule=R1)),
           ("2 규칙 + 폭: 돌출 ≥ 0.75 & 폭 ≥ 150 Hz → 고장 (주)", dict(base_kw=BASE, rule=R2)),
           ("3 참고: + 두 모드 상대 크기 선형 입력 (EXP-020)", dict(base_kw=dict(experts=(ae_mean, et_logit, mode_contrast), combine="lr"), rule=None))]
MAIN = CONFIGS[2][0]

fire = lambda r: [names[i] for i in np.flatnonzero(r(g))]
ex.note("규칙이 켜지는 파일 (150개 전체, 학습 없음)",
        f"1 사용자 규칙: {' '.join(fire(R1))}\n2 규칙 + 폭: {' '.join(fire(R2))}\n"
        f"(폭 조건으로 빠진 파일: {' '.join(sorted(set(fire(R1)) - set(fire(R2))))})")


# %% quick_cv
R, rows = {}, []
for name, kw in CONFIGS:
    q = quick_cv(g[:, None], y, g, lambda kw=kw: RuleOverride(**kw), seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    R[name] = q
    rows.append(f"{name:46s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · ROC-AUC {q['roc_auc'].mean():.3f} ± {q['roc_auc'].std():.3f}")
    print(rows[-1], flush=True)
ex.note("quick_cv 결과 (5-fold × seed 3, 임계값 없음)", "\n".join(rows))
cfg = [c[0] for c in CONFIGS]
for c in cfg:
    pm = R[c]["pct"]
    ex.note(f"{c.split()[0]} 정상 90% 선 아래 고장 · 정상 상위 5%",
            f"고장: {' '.join(names[i].replace('abn_', '') for i in np.flatnonzero((y == 1) & (pm < .9)))}\n"
            f"정상: {' '.join(names[i].replace('nor_', '') for i in np.flatnonzero((y == 0) & (pm > .95)))}")


# %% 그림 1: 규칙 지도 — 8.3 kHz 돌출 vs 봉우리 폭. 규칙 영역 = 오른쪽 위 상자
fig, ax = plt.subplots(figsize=(10, 7))
for lab, s, c, mk in (("정상", y == 0, NOR, "s"), ("고장", y == 1, ABN, "o")):
    ax.scatter(P83[s], W83[s], color=c, s=26, alpha=.7, lw=0, marker=mk, label=lab)
ax.axvline(.75, color="k", lw=1, ls="--", label="사용자 규칙: 돌출 ≥ 0.75 dec")
ax.fill_between([.75, 1.6], 150, 450, color=ABN, alpha=.08, label="규칙 + 폭: 폭 ≥ 150 Hz (주)")
for i in np.flatnonzero(P83 >= .55):
    ax.annotate(names[i].replace("abn_", "A").replace("nor_", "N"), (P83[i], W83[i]), xytext=(4, 3),
                textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
ax.set_xlim(-.05, 1.4); ax.set_ylim(0, 450); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("8.20–8.45 kHz 최대 돌출 (dec)"); ax.set_ylabel("그 봉우리의 반높이 폭 (Hz)")
ax.set_title("8.3 kHz 모드 규칙. 돌출만으로는 N019·N004가 고장 사이에 끼고, 폭(좁은 순음 vs 넓은 들뜸)으로 빠진다", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "rule_map", "규칙 영역(오른쪽 위)에 고장만 들어가는가. 폭 조건이 N019·N004(좁은 순음)를 빼는가")


# %% 그림 2: 로지스틱 구분선 + 규칙 — x = 기준 결합 결정값(OOF seed 0), y = 8.3k 돌출, 점 크기 = 폭
D0 = np.empty(len(y))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    D0[te] = ScoreStack(**BASE).fit(g[tr][:, None], y[tr]).decision_function(g[te][:, None])
fired = R2(g)
fig, ax = plt.subplots(figsize=(10, 8))
for lab, s, c, mk in (("정상", y == 0, NOR, "s"), ("고장", y == 1, ABN, "o")):
    ax.scatter(np.clip(D0[s], -8, 8), P83[s], s=10 + W83[s] / 4, color=c, alpha=.6, lw=0, marker=mk, label=lab)
ax.scatter(np.clip(D0[fired], -8, 8), P83[fired], s=10 + W83[fired] / 4, facecolor="none", edgecolor="k", lw=1.2,
           label="규칙 + 폭이 켜진 파일 (무조건 고장)")
ax.axvline(0, color="k", lw=1.2, label="로지스틱 구분선 (결정값 0)")
ax.axhline(.75, color="k", lw=.8, ls="--")
for nm in HARD:
    i = names.index(nm)
    ax.annotate(nm.replace("abn_", "A").replace("nor_", "N"), (np.clip(D0[i], -8, 8), P83[i]), xytext=(4, 3),
                textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR)
ax.set_xlim(-8, 8); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("기준 결합(AE + ET 로지스틱) 결정값, OOF seed 0"); ax.set_ylabel("8.20–8.45 kHz 최대 돌출 (dec). 점 크기 = 봉우리 폭")
ax.set_title("로지스틱 구분선 왼쪽(정상 판정)인데 규칙이 켜져 고장으로 바뀌는 파일 = 규칙의 이득", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "boundary_lr", "구분선 왼쪽에 있던 A002(·A029)가 규칙으로 고장이 되는가, 규칙이 정상을 끌어올리지 않는가")


# %% 그림 3: t-SNE — 출력 끝단 / + 8.3k 모드 특징(돌출·폭·8.6k 돌출). 2행 색 = 주 구성 OOF 점수
A, T = np.empty((len(y), LP.shape[-1])), np.empty((len(y), 300))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    a, b = tuple(tr.tolist()), tuple(te.tolist())
    A[te], T[te] = ae_bins(a, b), et_trees(a, b)
std = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-12)
blk = lambda *Bs: np.hstack([std(B) / np.sqrt(B.shape[1]) for B in Bs])
K = np.column_stack([P83, W83, P86])
fig, _ = viz.tsne({"출력 끝단: AE bin 오차 + ET 나무 (EXP-014)": blk(A, T),
                   "출력 끝단 + 8.3k 모드 (돌출·폭·8.6k 돌출)": blk(A, T, K)},
                  y, names, mark=HARD, score=np.clip(R[MAIN]["oof"][0], -10, 10), scale=False)
ex.save_fig(fig, "tsne", "8.3k 모드 특징을 더하면 A002·A029가 고장 무리 쪽으로 가는가. 2행 색 = 주 구성 OOF 점수(±10에서 자름, 규칙 +100)")


# %% 그림 4: hard_files
M = np.array([[1 - R[c]["pct"][names.index(nm)] if nm.startswith("abn") else R[c]["pct"][names.index(nm)] for c in cfg] for nm in HARD])
fig, ax = plt.subplots(figsize=(10, 11))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD)))
ax.set_yticklabels([("고장 " if n.startswith("abn") else "정상 ") + n + f"  (8.3k {P83[names.index(n)]:.2f}, 폭 {W83[names.index(n)]:.0f})" for n in HARD], fontsize=7.5)
ax.set_xticks(range(len(cfg))); ax.set_xticklabels([c.split(":")[0] for c in cfg], fontsize=9)
ax.set_title("위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율. 열: 0 기준 · 1 규칙 · 2 규칙+폭 · 3 두 모드 선형", fontsize=8.5)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "규칙으로 002·029가 0이 되는가, 규칙 1에서 N019(·N004)가 1로 올라가고 규칙 2에서는 그대로인가")


# %% 보조: 놓친 고장 PSD
missed = [names[i] for i in np.flatnonzero((y == 1) & (R[MAIN]["pct"] < .9))]
fL, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(fL, L, y, names, missed, ranges=((0, 12500), (7500, 9500)), title="주 구성이 정상 90% 선 아래로 둔 고장"),
                "psd_missed", "(보조) 놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)


# %% 중첩 CV F2: 기준(같은 코드) · 규칙 1 · 규칙 2(주)
warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
for name, kw in (CONFIGS[2], CONFIGS[1], CONFIGS[0]):
    r = nested_cv(g[:, None], y, g, lambda kw=kw: RuleOverride(**kw), n_jobs=-1)
    ex.record("중첩: " + name, r, names, main=(name == MAIN))
print("nested done", flush=True)
