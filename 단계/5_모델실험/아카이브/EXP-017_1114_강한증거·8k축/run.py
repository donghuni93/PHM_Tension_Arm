# %% [markdown]
# EXP-017 강한 증거 묶음 결합 + 8.25 kHz 대역 축 (2-10)
# - 문제: EXP-014 로지스틱 계수 1:1 → 사실상 평균. 한쪽만 강하게 고장이라는 파일(002·029 = AE, 017 = ET)이 희석된다.
#   OR(최댓값)은 0.889로 하락 — AE 점수가 비특이적이라(2-10: 002·029는 8.25 kHz, 059·062는 9.1·2.5 kHz에서 오차)
#   "AE만 강해도 고장"을 허용하면 059·062도 같이 올라온다
# - 처방 (사용자 11-14 + 2-10)
#   ① 강한 증거 묶음: 로지스틱 점수와 허용한 축마다 '정상 95%선을 넘은 정도'를 같은 척도(1 = 정상 95%선)로 맞추고 최댓값
#   ② AE 대신 특이적인 8.25 kHz 대역 축에 강한 증거를 허용. 대역 비율 = 200 Hz 칸 파워 / 200 Hz 이상 파워 (2-10: 분모가 핵심)
#      - 정직판: 200 Hz 고정 격자 61칸 중 학습 폴드 안 AUC 최고 칸 (원칙 3)
#      - 사후판: 8.2–8.4 kHz 직접 (2-10에서 그림 보고 정함 → 비교용, 채택 근거로 쓰지 않음)
# - 주 구성 (실행 전 지정): 3 = 로지스틱(AE, ET) ∨ 8k 격자 축. q = 95 고정 (7은 q 민감도 확인)
# - 평가: 구성 비교는 quick_cv. 주 구성은 중첩 CV F2까지 (EXP-014 기준 0.857 ± 0.031과 비교)
# 실행: conda run -n qe-phm python 단계/5_모델비교/아카이브/EXP-017_1114_강한증거·8k축/run.py

# %% 설정
import os, sys, warnings
sys.path.insert(0, next(_p for _p in (os.path.abspath(os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."), *[".."] * _k)) for _k in range(8)) if os.path.isdir(os.path.join(_p, "src"))))
from src.exp import *
from src.evaluate import quick_cv
from src.experts import (ScoreStack, ae_mean, ae_bins, et_logit, et_trees, ae_segerr, band_grid_top, band_grid_choice,
                         band_8k_post, warm_cache, LP, _band)
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-017", "강한 증거 묶음 결합 + 8.25 kHz 대역 축 (2-10)", ["H1", "H2", "H3"],
                member="이동훈", model="ET + AE → 로지스틱 ∨ 8 kHz 대역 축 (강한 증거 최댓값)",
                changes="EXP-014 로지스틱 점수와 200 Hz 격자 대역 비율 축(폴드 안 선택)을 정상 95%선 기준으로 맞춰 최댓값", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
MISS = ["abn_002", "abn_005", "abn_014", "abn_017", "abn_022", "abn_029"]
FP = ["nor_008", "nor_019", "nor_025", "nor_027", "nor_039", "nor_050", "nor_054", "nor_055", "nor_059", "nor_062",
      "nor_065", "nor_073", "nor_086", "nor_092", "nor_095", "nor_106", "nor_116"]
HARD = MISS + FP

n = warm_cache([ae_segerr, et_trees], seeds=SEEDS)
print(f"캐시 완료: 분할 {n}개", flush=True)


# %% quick_cv (열 순서: 0 AE z, 1 ET z, 2 대역 축 z)
Z0 = (ae_mean, et_logit)
ZG, ZP = Z0 + (band_grid_top,), Z0 + (band_8k_post,)
CONFIGS = [("0 기준: 로지스틱(AE, ET) (EXP-014)", dict(experts=Z0, combine="lr")),
           ("1 로지스틱(AE, ET, 8k 격자) — 선형 입력", dict(experts=ZG, combine="lr")),
           ("2 강한 증거: 로지스틱 ∨ AE ∨ ET (8k 없이)", dict(experts=Z0, combine="strong", lr_cols=[0, 1], or_cols=(0, 1))),
           ("3 강한 증거: 로지스틱 ∨ 8k 격자 (주)", dict(experts=ZG, combine="strong", lr_cols=[0, 1], or_cols=(2,))),
           ("4 강한 증거: 로지스틱 ∨ ET ∨ 8k 격자", dict(experts=ZG, combine="strong", lr_cols=[0, 1], or_cols=(1, 2))),
           ("5 [사후] 로지스틱(AE, ET, 8.2–8.4k)", dict(experts=ZP, combine="lr")),
           ("6 [사후] 강한 증거: 로지스틱 ∨ 8.2–8.4k", dict(experts=ZP, combine="strong", lr_cols=[0, 1], or_cols=(2,))),
           ("7 q 민감도: 3을 q = 99로", dict(experts=ZG, combine="strong", lr_cols=[0, 1], or_cols=(2,), q=99))]
MAIN = CONFIGS[3][0]
nor = y == 0
R, rows = {}, []
for name, kw in CONFIGS:
    q = quick_cv(g[:, None], y, g, lambda kw=kw: ScoreStack(**kw), seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    R[name] = q
    rows.append(f"{name:42s} PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · "
                f"ROC-AUC {q['roc_auc'].mean():.3f} ± {q['roc_auc'].std():.3f}")
    print(rows[-1], flush=True)
ex.note("quick_cv 결과 (5-fold × seed 3, 임계값 없음)", "\n".join(rows))
cfg = [c[0] for c in CONFIGS]
for c in cfg:
    pm = R[c]["pct"]
    ex.note(f"{c.split()[0]} 정상 90% 선 아래 고장 · 정상 상위 5%",
            f"고장: {' '.join(names[i].replace('abn_', '') for i in np.flatnonzero((y == 1) & (pm < .9)))}\n"
            f"정상: {' '.join(names[i].replace('nor_', '') for i in np.flatnonzero((y == 0) & (pm > .95)))}")

# 폴드 안에서 고른 격자 칸 (안정성 — 바깥 학습 15개 분할)
picks = [band_grid_choice(tr) for s in SEEDS for tr, _ in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
vals, cnt = np.unique(picks, return_counts=True)
ex.note("폴드 안에서 고른 200 Hz 격자 칸 (quick_cv 학습 분할 15개)", " · ".join(f"{v} ×{c}" for v, c in zip(vals, cnt)))


# %% 그림 1: 강한 증거 두 성분 — 로지스틱(x) vs 8k 격자 축(y). 1 = 정상 95%선. 최종 점수 = 둘 중 큰 값
P = np.empty((len(y), 2))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    P[te] = ScoreStack(**CONFIGS[3][1]).fit(g[tr][:, None], y[tr]).strong_parts(g[te][:, None])
LO, HI = -4, 8
pc = np.clip(P, LO, HI)
fig, ax = plt.subplots(figsize=(9.5, 8))
ax.scatter(*pc[y == 0].T, s=22, color=NOR, alpha=.55, lw=0, label="정상")
ax.scatter(*pc[y == 1].T, s=34, color=ABN, lw=0, label="고장")
for t, ls in ((1, ":"), (2, "--")):
    ax.plot([LO, t, t], [t, t, LO], color="k", lw=.9, ls=ls, label=f"max = {t} 경계 (예시 임계값)")
for nm in HARD:
    i = names.index(nm)
    ax.annotate(nm.replace("abn_", "A").replace("nor_", "N"), pc[i], xytext=(4, 3), textcoords="offset points",
                fontsize=8, color=ABN if y[i] else NOR)
ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.grid(alpha=.25); ax.legend(fontsize=8, loc="upper left")
ax.set_xlabel("로지스틱(AE, ET) 증거 (1 = 정상 95%선)"); ax.set_ylabel("8 kHz 대역 축 증거 (폴드 안 선택 칸, 1 = 정상 95%선)")
ax.set_title("강한 증거 묶음 (OOF seed 0). 경계는 ㄱ자: 어느 한쪽이라도 임계값을 넘으면 고장", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "strong_evidence", "A002·A029가 위쪽(8 kHz 축)으로 올라가 ㄱ자 경계를 넘는가, N059·N062는 아래에 남는가, "
            "새로 위로 올라오는 정상(N019·N092)은 몇 개인가")


# %% 그림 2: 어려운 파일의 위험도
M = np.array([[1 - R[c]["pct"][names.index(nm)] if nm.startswith("abn") else R[c]["pct"][names.index(nm)]
               for c in cfg] for nm in HARD])
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="k" if M[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD))); ax.set_yticklabels([("놓침 " if n.startswith("abn") else "오경보 ") + n for n in HARD], fontsize=8)
ax.set_xticks(range(len(cfg))); ax.set_xticklabels([c.split()[0] for c in cfg], fontsize=9)
ax.set_title("어려운 파일의 위험도 (quick_cv seed 3 평균). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "강한 증거 묶음(2·3·4)이 002·029·017을 옅게 하는가, 오경보(특히 019·092·059·062)를 진하게 하지 않는가")


# %% 그림 3: t-SNE (매 실험 기본) — 출력 끝단(AE+ET) / 출력 끝단 + 200 Hz 격자 대역 비율 61칸. 2행 색 = 주 구성 OOF 점수
AEo, ETo = np.empty((len(y), LP.shape[-1])), np.empty((len(y), 300))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    a, b = tuple(tr.tolist()), tuple(te.tolist())
    AEo[te], ETo[te] = ae_bins(a, b), et_trees(a, b)
std = lambda A: (A - A.mean(0)) / (A.std(0) + 1e-12)
blk = lambda *Bs: np.hstack([std(B) / np.sqrt(B.shape[1]) for B in Bs])
GR = _band()[[c for c in _band().columns if c.startswith("G")]].values
fig, _ = viz.tsne({"출력 끝단: AE bin 오차 + ET 나무 (EXP-014 공간)": blk(AEo, ETo),
                   "출력 끝단 + 200 Hz 격자 대역 비율 61칸 (블록 같은 비중)": blk(AEo, ETo, GR)},
                  y, names, mark=HARD, score=R[MAIN]["oof"][0], scale=False)
ex.save_fig(fig, "tsne", "대역 비율을 더하면 002·029가 고장 무리 쪽으로 가는가. 2행 색 = 주 구성(강한 증거 ∨ 8k)의 OOF 점수(seed 0)")


# %% 그림 4: 놓친 고장 vs 정상 5–95% (매 실험 기본) — 주 구성 기준
missed = [names[i] for i in np.flatnonzero((y == 1) & (R[MAIN]["pct"] < .9))]
f, L = viz.log_psd(X)
if missed:
    ex.save_fig(viz.psd_band(f, L, y, names, missed, ranges=((0, 12500), (7500, 9500)),
                             title=f"주 구성이 정상 90% 선 아래로 둔 고장"),
                "psd_missed", "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가 (아래: 7.5–9.5 kHz 확대)")
print("done", flush=True)


# %% 최종 확인: 주 구성 중첩 CV F2 (+ quick_cv에서 기준보다 sd 넘게 좋은 다른 구성)
base = R[cfg[0]]["pr_auc"]
extra = [c for c in cfg[1:] if c != MAIN and not c.split()[1].startswith("[")
         and R[c]["pr_auc"].mean() - base.mean() > max(R[c]["pr_auc"].std(), base.std(), 0.005)]
warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
print("중첩 캐시 완료", flush=True)
for name in [MAIN] + extra:
    kw = dict(CONFIGS)[name]
    r = nested_cv(g[:, None], y, g, lambda kw=kw: ScoreStack(**kw), n_jobs=-1)
    ex.record("중첩: " + name, r, names, main=(name == MAIN))
r0 = nested_cv(g[:, None], y, g, lambda: ScoreStack(**CONFIGS[0][1]), n_jobs=-1)
ex.record("중첩: " + cfg[0] + " (같은 코드로 재계산)", r0, names)
print("nested done", flush=True)
