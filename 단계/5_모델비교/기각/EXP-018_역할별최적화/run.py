# %% [markdown]
# EXP-018 두 전문가를 역할에 맞게 각각 최적화 → EXP-014 로지스틱 결합
# - 배경: EXP-014~017은 결합 규칙만 바꿨고 효과 없음. 두 전문가는 기본값 그대로였다
# - 역할과 목표 지표 (실행 전 지정)
#   AE = 특이성 ("스펙트럼 모양 이상"이 정상에서 드물어야): 오경보율 10% 이하 부분 AUC (sklearn max_fpr=0.1)
#     · 표준화: std (EXP-012) | robust (bin별 중앙값·MAD, 척도 하한 = bin 척도 중앙값) — 2-10: 059·062는 엉뚱한 bin에서 오차
#     · 집계: mean | top0.05 | top0.01 (상위 5%·1% bin 오차 평균) — 2-10: 002·029는 8.25 kHz에 몰림, 059·062는 흩어짐
#   ET = 충격형 고장 재현율: PR-AUC
#     · min_samples_leaf {1, 2, 5} × max_features {sqrt, 0.3}
# - 선택 규칙 (실행 전 지정): 기본값보다 그 설정의 sd 넘게 좋을 때만 기본값을 버린다. 격자가 작아도 선택 편향은 남는다 → 기록
# - 평가: 전문가·결합 비교는 quick_cv(5-fold × seed 3). 고른 결합은 중첩 CV F2 (기준 0.857 ± 0.031)
# 실행: conda run -n qe-phm python 단계/5_모델비교/기각/EXP-018_역할별최적화/run.py

# %% 설정
import os, sys, warnings, functools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..", "..")))
from src.exp import *
from src.evaluate import quick_cv
from src.experts import (ScoreStack, AEScore, ETScore, ae_segerr, ae_segerr_robust, et_trees, et_trees_p, warm_cache, LP)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-018", "두 전문가 역할별 최적화 (AE 특이성 · ET 재현율) → 로지스틱 결합", ["H1", "H2", "H3"],
                member="이동훈", model="ET + 스펙트럼 AE (각각 최적화) → 정상 기준 z → L2 로지스틱",
                changes="AE 표준화·오차 집계를 부분 AUC로, ET 잎 크기·분할 피처 수를 PR-AUC로 각각 고른 뒤 EXP-014 결합", focus=())
ex.res["figs"], ex.res["notes"], ex.res["configs"] = {}, {}, {}   # 11-14 그림 구성 변경: 이전 실행 기록 초기화 (README 메모는 유지)
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
MISS = ["abn_002", "abn_005", "abn_014", "abn_017", "abn_022", "abn_029"]
FP = ["nor_008", "nor_019", "nor_025", "nor_027", "nor_039", "nor_050", "nor_054", "nor_055", "nor_059", "nor_062",
      "nor_065", "nor_073", "nor_086", "nor_092", "nor_095", "nor_106", "nor_116"]
HARD = MISS + FP
nor = y == 0

AE_GRID = [AEScore(s, a) for s in ("std", "robust") for a in ("mean", "top0.05", "top0.01")]
ET_GRID = [ETScore(l, m) for l in (1, 2, 5) for m in ("sqrt", 0.3)]
fns = [ae_segerr, ae_segerr_robust, et_trees] + [functools.partial(et_trees_p, leaf=e.leaf, mf=e.mf)
                                                for e in ET_GRID if (e.leaf, e.mf) != (2, "sqrt")]
n = warm_cache(fns, seeds=SEEDS)
print(f"캐시 완료: 분할 {n}개", flush=True)


def qcv(experts, **kw):
    q = quick_cv(g[:, None], y, g, lambda: ScoreStack(experts, **kw), seeds=SEEDS)
    q["pauc"] = np.array([roc_auc_score(y, o, max_fpr=0.1) for o in q["oof"]])
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    return q


# %% 전문가별 최적화 (단독 점수, quick_cv)
RA = {repr(a): qcv((a,), combine="col0") for a in AE_GRID}
RE = {repr(e): qcv((e,), combine="col0") for e in ET_GRID}
rows = ["AE (목표: 부분 AUC@오경보율 10%)"]
for k, q in RA.items():
    rows.append(f"  {k:22s} 부분AUC {q['pauc'].mean():.3f} ± {q['pauc'].std():.3f} · PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f}")
rows.append("ET (목표: PR-AUC)")
for k, q in RE.items():
    rows.append(f"  {k:22s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 부분AUC {q['pauc'].mean():.3f} ± {q['pauc'].std():.3f}")
print("\n".join(rows), flush=True)


def pick(R, grid, metric, default):
    """기본값보다 그 설정의 sd 넘게 좋을 때만 바꾼다 (실행 전 지정 규칙)"""
    d = R[repr(default)][metric].mean()
    best = max(grid, key=lambda a: R[repr(a)][metric].mean())
    gain = R[repr(best)][metric].mean() - d
    return (best if gain > R[repr(best)][metric].std() else default), gain


AE0, ET0 = AEScore(), ETScore()
AEs, gA = pick(RA, AE_GRID, "pauc", AE0)
ETs, gE = pick(RE, ET_GRID, "pr_auc", ET0)
rows += ["", f"선택: AE = {AEs!r} (격자 최고와 기본값 차 {gA:+.3f}) · ET = {ETs!r} (차 {gE:+.3f})"]
ex.note("전문가별 단독 성능 (quick_cv 5-fold × seed 3)", "\n".join(rows))

# AE 특이성 확인: 002·029(8.25 kHz) vs AE 오경보 정상의 정상 중 순위
lines = ["AE 설정          " + "  ".join(f"{n.replace('abn_', 'A').replace('nor_', 'N'):>5s}" for n in
                                    ["abn_002", "abn_029", "abn_005", "abn_014", "nor_059", "nor_062", "nor_039", "nor_106", "nor_025", "nor_116"])]
for k, q in RA.items():
    lines.append(f"{k:16s} " + "  ".join(f"{q['pct'][names.index(n)]:5.2f}" for n in
                                         ["abn_002", "abn_029", "abn_005", "abn_014", "nor_059", "nor_062", "nor_039", "nor_106", "nor_025", "nor_116"]))
ex.note("AE 설정별 정상 중 순위 (고장은 높을수록, 정상은 낮을수록 좋음)", "\n".join(lines))


# %% 결합 (EXP-014와 같은 로지스틱). 선택 결과가 기본값이면 같은 모델이 겹치므로 겹치지 않는 구성만 계산
AEb = max(AE_GRID, key=lambda a: RA[repr(a)]["pauc"].mean() if repr(a) != repr(AE0) else -1)   # 기본값 제외 격자 최고
ETb = max(ET_GRID, key=lambda e: RE[repr(e)]["pr_auc"].mean() if repr(e) != repr(ET0) else -1)
COMB = [(f"최종: {AEs!r} + {ETs!r}" + (" (= EXP-014)" if (repr(AEs), repr(ETs)) == (repr(AE0), repr(ET0)) else ""), (AEs, ETs))]
for a, e in ((AE0, ETb), (AEb, ET0)):
    if (repr(a), repr(e)) != (repr(AEs), repr(ETs)):
        COMB.append((f"참고(채택 안 함): {a!r} + {e!r}", (a, e)))
RC, rows = {}, []
for name, exps in COMB:
    RC[name] = qcv(exps, combine="lr")
    q = RC[name]
    rows.append(f"{name:52s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 부분AUC {q['pauc'].mean():.3f} ± {q['pauc'].std():.3f}")
    print(rows[-1], flush=True)
ex.note("결합 quick_cv (로지스틱)", "\n".join(rows))
MAIN = COMB[0][0]
pm = RC[MAIN]["pct"]
ex.note("최종 결합: 정상 90% 선 아래 고장 · 정상 상위 5%",
        f"고장: {' '.join(names[i].replace('abn_', '') for i in np.flatnonzero((y == 1) & (pm < .9)))}\n"
        f"정상: {' '.join(names[i].replace('nor_', '') for i in np.flatnonzero((y == 0) & (pm > .95)))}")


# %% 그림 1: 로지스틱 구분선 — 최종 두 전문가의 z 평면 (OOF, quick_cv seed 0과 같은 분할)
Zall = np.empty((len(y), 2)); W = []
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    m = ScoreStack((AEs, ETs), combine="lr").fit(g[tr][:, None], y[tr])
    Zall[te] = m.zscores(g[te][:, None]); W.append(m.m_[-1])
coef = np.mean([w.coef_[0] for w in W], 0); icpt = np.mean([w.intercept_[0] for w in W])
LO, HI = -3, 4
zc = np.clip(Zall, LO, HI)
fig, ax = plt.subplots(figsize=(9.5, 8))
ax.scatter(*zc[y == 0].T, s=22, color=NOR, alpha=.55, lw=0, label="정상")
ax.scatter(*zc[y == 1].T, s=34, color=ABN, lw=0, label="고장")
ax.axvline(1, color=NOR, lw=.7, ls=":"); ax.axhline(1, color=NOR, lw=.7, ls=":")
xs = np.linspace(LO, HI, 50)
ax.plot(xs, -(coef[0] * xs + icpt) / coef[1], color="k", lw=1.3, label="로지스틱 구분선 (결정값 0, 폴드 평균 계수)")
for nm in HARD:
    i = names.index(nm)
    ax.annotate(nm.replace("abn_", "A").replace("nor_", "N"), zc[i], xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=ABN if y[i] else NOR)
ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel(f"{AEs!r} z (스펙트럼 모양 이상, 1 = 정상 90%선)"); ax.set_ylabel(f"{ETs!r} z (충격 피처 고장 정도, 1 = 정상 90%선)")
ax.set_title(f"로지스틱 구분선 (OOF seed 0). 계수 AE {coef[0]:+.2f} · ET {coef[1]:+.2f} · 절편 {icpt:+.2f}. "
             f"점선 = 각 전문가 정상 90%선", fontsize=9)
fig.tight_layout()
ex.save_fig(fig, "boundary_lr", "로지스틱 구분선 근처(회색지대)에 어떤 파일이 있는가. 번호 = 이전 실험에서 틀린 파일")


# %% 그림 2: t-SNE — AE 출력 끝 / ET 출력 끝 / 결합 출력 끝. 2행 색 = 최종 결합 OOF 점수 (seed 0)
A, T = np.empty((len(y), LP.shape[-1])), np.empty((len(y), 300))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    a, b = tuple(tr.tolist()), tuple(te.tolist())
    A[te], T[te] = np.log10(AEs.bins(a, b) + 1e-12), ETs.trees(a, b)
std = lambda M: (M - M.mean(0)) / (M.std(0) + 1e-12)
fig, _ = viz.tsne({f"AE 출력 끝: bin별 복원 오차 ({AEs!r})": std(A),
                   f"ET 출력 끝: 나무별 고장 확률 ({ETs!r})": std(T),
                   "결합: AE 출력 끝 + ET 출력 끝 (블록 같은 비중)": np.hstack([std(A) / np.sqrt(A.shape[1]), std(T) / np.sqrt(T.shape[1])])},
                  y, names, mark=HARD, score=RC[MAIN]["oof"][0], scale=False)
ex.save_fig(fig, "tsne", "각 전문가의 출력 끝단과 결합에서 고장이 모이는가, 회색지대 파일은 어디에 있는가. 2행 색 = 최종 결합 OOF 점수")


# %% 그림 3: 어려운 파일 위험도 — 겹치지 않는 열만 (최종 결합 / 참고 결합 / AE 기본·격자 최고 / ET 기본·격자 최고)
cols = [("결합 최종", RC[MAIN])] + [("결합 참고\n" + n.split(": ", 1)[1].replace(" + ", "\n+ "), RC[n]) for n, _ in COMB[1:]]
cols += [(f"AE 단독\n{AE0!r}", RA[repr(AE0)]), (f"AE 단독\n{AEb!r}", RA[repr(AEb)]),
         (f"ET 단독\n{ET0!r}", RE[repr(ET0)]), (f"ET 단독\n{ETb!r}", RE[repr(ETb)])]
M = np.array([[1 - q["pct"][names.index(nm)] if nm.startswith("abn") else q["pct"][names.index(nm)] for _, q in cols] for nm in HARD])
fig, ax = plt.subplots(figsize=(13, 10))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD))); ax.set_yticklabels([("놓침 " if n.startswith("abn") else "오경보 ") + n for n in HARD], fontsize=8)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c for c, _ in cols], fontsize=7.5)
ax.set_title("어려운 파일의 위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율. "
             "'참고'는 선택 규칙상 채택하지 않은 격자 최고", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "AE와 ET가 서로 다른 파일을 틀리는가(보완성), 격자 최고 설정이 기본값과 어떤 파일에서 다른가")


# %% 보조 그림: 전문가별 격자 · 놓친 고장 PSD (매 실험 기본)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 5))
for a, R, metric, lab, sel in ((a1, RA, "pauc", "AE: 부분 AUC (오경보율 ≤ 10%)", repr(AEs)), (a2, RE, "pr_auc", "ET: PR-AUC", repr(ETs))):
    ks = list(R)
    m = [R[k][metric].mean() for k in ks]; s = [R[k][metric].std() for k in ks]
    a.bar(range(len(ks)), m, yerr=s, color=[ABN if k == sel else "#9aa5b1" for k in ks], capsize=3)
    a.set_xticks(range(len(ks))); a.set_xticklabels(ks, rotation=20, fontsize=8)
    a.set_ylim(min(m) - .05, max(m) + .03); a.set_title(f"{lab} — 주황 = 선택", fontsize=10); a.grid(alpha=.3, axis="y")
fig.tight_layout()
ex.save_fig(fig, "expert_grid", "(보조) 각 전문가의 설정이 자기 역할 지표를 sd 넘게 바꾸는가 (막대 = seed 3 평균 ± sd)")
missed = [names[i] for i in np.flatnonzero((y == 1) & (RC[MAIN]["pct"] < .9))]
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, title="최종 결합이 정상 90% 선 아래로 둔 고장"),
                "psd_missed", "(보조) 놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("done", flush=True)


# %% 최종 확인: 중첩 CV F2 (기본값과 같으면 생략 — EXP-014·017에서 0.857 ± 0.031)
if (repr(AEs), repr(ETs)) != (repr(AE0), repr(ET0)):
    nf = [ae_segerr, et_trees]
    if AEs.scale == "robust":
        nf.append(ae_segerr_robust)
    if (ETs.leaf, ETs.mf) != (2, "sqrt"):
        nf.append(functools.partial(et_trees_p, leaf=ETs.leaf, mf=ETs.mf))
    warm_cache(nf, seeds=range(5), nested=True)
    r = nested_cv(g[:, None], y, g, lambda: ScoreStack((AEs, ETs), combine="lr"), n_jobs=-1)
    ex.record("중첩: " + MAIN, r, names, main=True)
    print("nested done", flush=True)
else:
    ex.note("중첩 CV", "선택 규칙상 두 전문가 모두 기본값 유지 → 최종 결합은 EXP-014와 같음 (중첩 F2 0.857 ± 0.031)")
