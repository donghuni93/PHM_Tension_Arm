# %% [markdown]
# EXP-032 정상상태 AE — AE를 "타격이 없을 때의 울림" 전문가로 바꿔 ET와 역할을 나누면, 지금 구분하지 못하는 파일이 드러나는가
# - 문제: 정상 120개에서 AE z와 ET z의 상관 0.52 (EXP-029 평면, OOF seed 0). AE 입력이 1초 구간의 평균 Welch PSD라
#   큰 타격 한 번도 광대역으로 PSD를 올린다 → 단일 타격 정상(025 116 055)을 AE와 ET가 함께 고장 쪽으로 민다 (6-1 오경보 유형)
# - 가설: AE = H2(공진 구조)·H3(게인 무관 모양) 전문가, ET = H1(충격) 전문가로 나눈다.
#   Welch 프레임(2048점, 1초에 23개)을 평균 대신 중앙값으로 모으면 충격이 든 프레임 몇 개가 무시된다 → AE는 정상상태 스펙트럼만 본다
#   예상: (a) 정상의 AE–ET 상관 하락 (b) 단일 타격 정상의 AE z 하락 (c) 공진 구조가 바뀐 고장(8.3k 등)은 AE에 남음
# - 구성 (모두 EXP-025 = AE + ET → 로지스틱 + 8.3k 규칙 + 유격 반복 규칙, AE만 바꿈). 설정은 실행 전 고정, 라벨 무관
#   0 기준 EXP-025 (평균 Welch AE)
#   1 중앙값 Welch AE                ← 주 구성 (실행 전 지정)
#   2 중앙값 + 모양 (구간마다 평균 log PSD를 빼서 게인 제거, H3)
#   3 중앙값 + 병목 4 (128-4-128, 거친 스펙트럼 구조만 기억)
# - 핵심 산출물 = "구분하지 못하는 파일": 모든 AE 설정에서 중첩 CV 3/5 이상 틀리는 파일과, 그 파일이 어느 축에서도 정상 범위 안인지
# - 주의: 8.3k·유격 규칙 문턱은 사후(EXP-021·025). 이번 실험의 새 선택(중앙값 Welch·모양·병목)은 라벨을 보지 않고 정함
# 실행: conda run -n qe-phm python "단계/5_모델비교/유의미/EXP-032_1116_정상상태AE_역할분리/run.py"

# %% 설정
import os, sys, warnings
from functools import partial
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
try: sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception: pass
from src.exp import *
from src.evaluate import quick_cv, separation
from src.experts import (RuleOverride, Rule83, KnockRule, ScoreStack, AEVar, ae_mean, et_logit, ae_segerr, ae_segerr_v,
                         et_trees, warm_cache, logpsd_var, peak83_table, F_PSD, _hits)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

NOR, ABN = "#2a78d6", "#eb6834"
FOCUS = ("abn_022", "abn_005", "nor_025", "nor_116", "nor_055", "nor_059")
ex = Experiment(__file__, "EXP-032", "정상상태 AE — 충격을 뺀 스펙트럼으로 AE와 ET의 역할 나누기", ["H2", "H3", "H1"],
                member="이동훈", model="ET + 중앙값 Welch 스펙트럼 AE → 로지스틱 + 8.3k·유격 규칙",
                changes="EXP-025의 AE 입력을 1초 평균 Welch PSD → 중앙값 Welch PSD(충격 프레임 무시)로. 비교: + 모양(게인 제거), + 병목 4",
                focus=FOCUS)
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]; nor = y == 0
H = _hits(); n20 = H.HIT_n20.values.astype(int)
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
LPa, LPm = logpsd_var("mean"), logpsd_var("median")         # (파일, 8, 1025). 중앙값판은 여기서 한 번 계산해 캐시

R83, KN = Rule83(0.75, 150), KnockRule(2.0, 0.7)
AES = [("0 기준 EXP-025", ae_mean, "평균 Welch"),
       ("1 중앙값 Welch AE (주)", AEVar("median"), "중앙값 Welch"),
       ("2 중앙값 + 모양", AEVar("median", shape=True), "중앙값 + 모양"),
       ("3 중앙값 + 병목 4", AEVar("median", hidden=(128, 4, 128)), "중앙값 + 병목 4")]
MAIN, BASE = AES[1][0], AES[0][0]
make = {nm: (lambda ae=ae: RuleOverride(base_kw=dict(experts=(ae, et_logit), combine="lr"), rule=[R83, KN])) for nm, ae, _ in AES}
AE_FNS = [ae_segerr, partial(ae_segerr_v, psd="median"), partial(ae_segerr_v, psd="median", shape=True),
          partial(ae_segerr_v, psd="median", hidden=(128, 4, 128))]
warm_cache([et_trees] + AE_FNS, seeds=range(3))
print("warm done", flush=True)


def pct_n(v):
    """정상 중 이 값보다 낮은 비율 (파일별)"""
    return np.array([(v[nor] < v[i]).mean() for i in range(len(y))])


# %% 1. 역할: 두 전문가 z 평면 (OOF seed 0) — 정상 상관 · 단일 타격 정상의 AE z · 정상 95%선 기준 누가 잡나
PL, RULE = {}, np.zeros(len(y), bool)
for nm, ae, _ in AES:
    Z = np.empty((len(y), 2))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae, et_logit), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        if nm == BASE: RULE[te] = R83(te) | KnockRule(2.0, 0.7).fit(tr)(te)
    PL[nm] = Z
rows = []
for nm, _, lab in AES:
    Z = PL[nm]; a95, e95 = np.percentile(Z[nor, 0], 95), np.percentile(Z[nor, 1], 95)
    A, E_ = Z[:, 0] > a95, Z[:, 1] > e95
    ab, out = ~nor, ~nor & ~RULE
    one, many = nor & (n20 == 1), nor & (n20 != 1)
    rows.append(dict(구성=lab, 정상상관=round(np.corrcoef(Z[nor, 0], Z[nor, 1])[0, 1], 2),
                     단일타격정상_AEz중앙=round(float(np.median(Z[one, 0])), 2), 나머지정상_AEz중앙=round(float(np.median(Z[many, 0])), 2),
                     고장_AE만=int((ab & A & ~E_).sum()), 고장_ET만=int((ab & E_ & ~A).sum()), 고장_둘다=int((ab & A & E_).sum()),
                     고장_둘다아님=int((ab & ~A & ~E_).sum()), 규칙밖고장_둘다아님=" ".join(short(names[i]) for i in np.flatnonzero(out & ~A & ~E_)),
                     AE만_고장=" ".join(short(names[i]) for i in np.flatnonzero(ab & A & ~E_))))
ROLE = pd.DataFrame(rows)
ex.note("역할 표 (OOF seed 0, 정상 95%선 기준)", ROLE.to_string(index=False))
print(ROLE.to_string(index=False), flush=True)

# %% 2. quick_cv: AE 단독 · 결합 + 분리 판정 (seed 3)
q_oof, qrows = {}, []
for nm, ae, lab in AES:
    qa = quick_cv(F, y, g, lambda ae=ae: ScoreStack(experts=(ae,), combine="col0"), seeds=range(3))
    q = quick_cv(F, y, g, make[nm], seeds=range(3)); q_oof[nm] = q["oof"]; s = separation(q["oof"], y, names)
    qrows.append(f"{lab:14s} AE 단독 PR-AUC {qa['pr_auc'].mean():.3f} ± {qa['pr_auc'].std():.3f} | 결합 PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f}"
                 f" · 확실한 고장 {s['sure_a']} · 회색 고장 {' '.join(short(x) for x in s['gray_a'])} · 회색 정상 {len(s['gray_n'])} · 위험 정상 {s['risk_n']:.1f}")
    print(qrows[-1], flush=True)
ex.note("quick_cv (seed 3)", "\n".join(qrows))

# %% 그림 1 (핵심): logistic_plane — 기준 | 주 구성. 규칙으로 잡힌 고장은 뺀다
xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 400), np.linspace(-3, 4.5, 400)); G = np.c_[xx.ravel(), yy_.ravel()]


def plane_fig(keys, res=None):
    fig, axs = plt.subplots(1, len(keys), figsize=(8.5 * len(keys), 8.5), squeeze=False)
    for ax, nm in zip(axs[0], keys):
        Zc = np.clip(PL[nm], -3, 4.5)
        D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(G).reshape(xx.shape)
        ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
        ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.6)
        ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.8, linestyles="--")
        show = ~(RULE & (y == 1))
        for lab, sel, col, mk in (("정상", nor & show, NOR, "s"), ("고장 (규칙 밖)", ~nor & show, ABN, "o")):
            ax.scatter(Zc[sel, 0], Zc[sel, 1], color=col, s=28, lw=0, marker=mk, label=lab, alpha=.85)
        cp = nor & (n20 == 1)
        ax.scatter(Zc[cp, 0], Zc[cp, 1], s=80, facecolor="none", edgecolor="#555", lw=.9, label="큰 타격 1개 정상")
        title = f"{nm}  ·  정상 AE–ET 상관 {np.corrcoef(PL[nm][nor, 0], PL[nm][nor, 1])[0, 1]:.2f}"
        if res and nm in res:
            c = res[nm]; title += f"\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f}"
            wrong = [names.index(n) for n, k in {**c["miss_counts"], **c["fp_counts"]}.items() if k >= 2]
            ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=90, marker="x", color="k", lw=1.8, label="자주 틀림 (5번 중 2번 이상)")
            for i in wrong:
                k = c["miss_counts"].get(names[i]) or c["fp_counts"].get(names[i])
                ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {k}/5", Zc[i], xytext=(5, 4), textcoords="offset points",
                            fontsize=8.5, color=ABN if y[i] else NOR, weight="bold")
        for n in FOCUS:
            i = names.index(n)
            if show[i]: ax.annotate(short(n), Zc[i], xytext=(5, -10), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR)
        gone = [short(names[i]) for i in np.flatnonzero(RULE & (y == 1))]
        ax.text(.98, .02, "뺀 고장 (규칙):\n" + " ".join(gone[:8]) + "\n" + " ".join(gone[8:]), transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
        ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
        ax.set_xlabel("AE z (스펙트럼 이상)"); ax.set_ylabel("ET z (충격 피처 이상)"); ax.legend(fontsize=8, loc="upper left")
        ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    return fig


# %% 그림 2: 역할 분리 — 네 AE 설정의 정상 점 (색 = 큰 타격 수). 상관이 내려가고 큰 타격 1개 정상이 왼쪽으로 가는가
fig, axs = plt.subplots(1, 4, figsize=(22, 5.8))
cmap = {0: "#9ab7d8", 1: "#d6452a", 2: "#555"}
for ax, (nm, _, lab) in zip(axs, AES):
    Z = np.clip(PL[nm], -3, 4.5)
    for k, txt in ((0, "정상 · 큰 타격 0개"), (1, "정상 · 큰 타격 1개"), (2, "정상 · 큰 타격 2개+")):
        sel = nor & (np.minimum(n20, 2) == k)
        ax.scatter(Z[sel, 0], Z[sel, 1], s=22, color=cmap[k], label=txt, alpha=.85)
    ax.scatter(Z[~nor, 0], Z[~nor, 1], s=26, facecolor="none", edgecolor=ABN, lw=.8, label="고장")
    for n in ("nor_025", "nor_116", "nor_055", "nor_059", "abn_022"):
        i = names.index(n); ax.annotate(short(n), Z[i], xytext=(4, 3), textcoords="offset points", fontsize=7.5)
    ax.axvline(np.percentile(PL[nm][nor, 0], 95), color="k", lw=.7, ls=":"); ax.axhline(np.percentile(PL[nm][nor, 1], 95), color="k", lw=.7, ls=":")
    ax.set_title(f"{lab}: 정상 상관 {np.corrcoef(PL[nm][nor, 0], PL[nm][nor, 1])[0, 1]:.2f}", fontsize=10)
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.grid(alpha=.3); ax.set_xlabel("AE z"); ax.set_ylabel("ET z")
axs[0].legend(fontsize=7.5, loc="upper left")
fig.suptitle("점선 = 정상 95%선. 빨간 점(큰 타격 1개 정상)이 AE z 방향으로 내려오면 AE가 충격을 덜 본다는 뜻", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "role_split", "AE 설정마다 정상의 AE z–ET z 상관과 큰 타격 1개 정상(빨강)의 AE z가 내려가는가")

# %% 그림 3: 평균 vs 중앙값 Welch — 큰 타격이 든 구간에서 PSD가 얼마나 들뜨나 (라벨 무관: 두 PSD 차이가 가장 큰 구간)
gap = (LPa - LPm).mean(-1)                                    # (파일, 8): 평균 − 중앙값 (dec, 주파수 평균)
fig, axs = plt.subplots(2, 3, figsize=(18, 8.5), sharex=True)
show_files = ["nor_025", "nor_116", "nor_055", "abn_022", "abn_005", "abn_010"]
ma = np.median(LPa[nor].mean(1), 0); mm = np.median(LPm[nor].mean(1), 0)
for ax, n in zip(axs.ravel(), show_files):
    i = names.index(n); k = int(gap[i].argmax())
    ax.fill_between(F_PSD, np.percentile(LPm[nor, k], 5, 0), np.percentile(LPm[nor, k], 95, 0), color=NOR, alpha=.18, label="정상 5–95% (중앙값 Welch, 같은 구간)")
    ax.plot(F_PSD, LPa[i, k], color="#d6452a", lw=.8, label="평균 Welch (기존 AE 입력)")
    ax.plot(F_PSD, LPm[i, k], color="k", lw=.8, label="중앙값 Welch (새 AE 입력)")
    ax.set_title(f"{n} ({'고장' if y[i] else '정상'}) · 구간 {k + 1}/8 · 평균−중앙값 {gap[i, k]:.2f} dec · 큰 타격 {n20[i]}개", fontsize=9)
    ax.grid(alpha=.3); ax.set_ylabel("log10 PSD")
for ax in axs[1]: ax.set_xlabel("주파수 (Hz)")
axs[0, 0].legend(fontsize=7.5)
fig.suptitle("평균 Welch는 타격 한 번에 광대역으로 들뜬다(빨강). 중앙값 Welch(검정)는 그 구간에서도 타격이 없을 때의 울림 구조만 남는다", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "mean_vs_median", "타격이 든 구간에서 평균 Welch(빨강)는 정상 띠 위로 들뜨고 중앙값 Welch(검정)는 띠 안으로 돌아오는가 (정상 025·116·055), 고장은 중앙값으로도 띠 밖에 남는가")
ex.note("평균−중앙값 차이 (구간 최대, dec)", f"정상 중앙 {np.median(gap[nor].max(1)):.3f} · 고장 중앙 {np.median(gap[~nor].max(1)):.3f} · "
        f"큰 타격 1개 정상 중앙 {np.median(gap[nor & (n20 == 1)].max(1)):.3f} · 큰 타격 0개 정상 중앙 {np.median(gap[nor & (n20 == 0)].max(1)):.3f}")

# %% 그림 4: t-SNE (모든 파일)
fig, _ = viz.tsne({"평균 Welch log PSD (구간 평균)": LPa.mean(1), "중앙값 Welch log PSD (구간 평균)": LPm.mean(1),
                   "주 구성 z (AE z, ET z)": PL[MAIN]}, y, names, mark=list(FOCUS) + ["abn_002", "abn_029"], score=q_oof[MAIN][0])
ex.save_fig(fig, "tsne", "중앙값 Welch 공간에서 고장이 더 모이는가, 오경보 정상·022는 어디인가")

# %% 중첩 CV F2 (모든 구성, 같은 분할)
warm_cache([et_trees] + AE_FNS, seeds=range(5), nested=True)
print("nested warm done", flush=True)
for nm, _, _ in [AES[1], AES[0], AES[2], AES[3]]:
    ex.record("중첩: " + nm, nested_cv(F, y, g, make[nm], n_jobs=-1), names, main=(nm == MAIN))
res = {k.replace("중첩: ", ""): v for k, v in ex.res["configs"].items()}

fig = plane_fig([BASE, MAIN], res)
ex.save_fig(fig, "logistic_plane", "규칙 밖 고장과 정상만 남긴 AE z × ET z 평면. 중앙값 AE에서 큰 타격 1개 정상(회색 고리)이 왼쪽으로 빠지고 오경보 ×가 줄어드는가", key=True)

# %% 5. 구분하지 못하는 파일 — 모든 AE 설정에서 3/5 이상 틀림 / 어떤 설정에서만 맞힘. 축별 정상 중 위치
err = pd.DataFrame({lab: [(res[nm]["miss_counts"] if y[i] else res[nm]["fp_counts"]).get(names[i], 0) for i in range(len(y))]
                    for nm, _, lab in AES}, index=names)
always = err.index[(err >= 3).all(1)].tolist()
some = err.index[(err >= 3).any(1) & ~(err >= 3).all(1)].tolist()
anyerr = err.index[(err >= 1).any(1)].tolist()
P83, W83, _ = peak83_table(); kn = KnockRule(2.0, 0.7).fit(g)
AX = pd.DataFrame({"AE z 평균 Welch": PL[BASE][:, 0], "AE z 중앙값 Welch": PL[MAIN][:, 0], "AE z 중앙값+모양": PL[AES[2][0]][:, 0],
                   "ET z": PL[BASE][:, 1], "8.3k 돌출": P83, "8.3k 폭": W83, "유격 r": kn.r(g), "큰 타격 수": n20.astype(float),
                   "주기 잠금 S": H.LOCK_S.values}, index=names)
PCT = AX.apply(lambda c: pd.Series(pct_n(c.values), index=names))
tab = pd.concat([err.add_prefix("틀림 "), AX.round(2), PCT.round(2).add_prefix("정상중 ")], axis=1).loc[anyerr]
tab.insert(0, "라벨", ["고장" if n[0] == "a" else "정상" for n in tab.index])
tab.to_csv(os.path.join(ex.figs, "unresolved.csv"), encoding="utf-8-sig")
ex.note("구분하지 못하는 파일 (모든 AE 설정에서 3/5 이상)", " ".join(short(n) for n in always) or "없음")
ex.note("AE 설정에 따라 갈리는 파일 (어떤 설정에서는 3/5 이상, 다른 설정에서는 아님)", " ".join(short(n) for n in some) or "없음")
ex.note("틀린 횟수 표 (중첩 seed 5번 중)", err.loc[anyerr].to_string())
print("구분 못 함:", always, "| 갈림:", some, flush=True)

# %% 그림 5: 틀린 파일의 축별 위치 (정상 중 백분위). 정상은 1에 가까울수록, 고장은 0에 가까울수록 그 축이 틀리게 민다
fig, ax = plt.subplots(figsize=(13, 0.42 * len(anyerr) + 2.5))
M = PCT.loc[anyerr].values
im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="w" if abs(M[i, j] - .5) > .35 else "k")
lab_rows = [f"{'고장' if n[0] == 'a' else '정상'} {short(n)} · 틀림 " + "/".join(str(v) for v in err.loc[n].values) + (" ★" if n in always else "") for n in anyerr]
ax.set_yticks(range(len(anyerr)), lab_rows, fontsize=8); ax.set_xticks(range(M.shape[1]), PCT.columns, rotation=30, ha="right", fontsize=8.5)
ax.set_title("틀린 파일의 축별 '정상 중 백분위' (1 = 정상 최대 이상, 빨강). 틀림 = 구성 0/1/2/3의 5번 중 횟수, ★ = 모든 설정에서 3/5 이상\n"
             "고장인데 모든 칸이 파랑·흰색 = 지금 축으로는 정상과 같다 / 정상인데 빨간 칸 = 그 축이 오경보를 만든다", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "unresolved_axes", "구분하지 못하는 파일이 어느 축에서 정상과 같고, 오경보 정상은 어느 축이 고장 쪽으로 미는가")

# %% 그림 6: hard_files (quick_cv seed 3 위험도)
HARD = ["abn_022", "abn_005", "abn_020", "abn_025", "abn_014", "abn_017", "abn_015", "abn_016", "abn_002", "abn_029",
        "nor_025", "nor_116", "nor_055", "nor_059", "nor_062", "nor_027", "nor_106", "nor_054", "nor_039", "nor_065"]
Mh = []
for nm, _, _ in AES:
    o = q_oof[nm]; pc = np.array([(o[:, nor] < o[:, [i]]).mean() for i in range(len(y))])
    Mh.append([1 - pc[names.index(h)] if h[0] == "a" else pc[names.index(h)] for h in HARD])
Mh = np.array(Mh).T
fig, ax = plt.subplots(figsize=(9.5, 10.5))
im = ax.imshow(Mh, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(Mh.shape[0]):
    for j in range(Mh.shape[1]):
        ax.text(j, i, f"{Mh[i, j]:.2f}", ha="center", va="center", fontsize=8, color="k" if Mh[i, j] < .7 else "w")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h + f" (큰 타격 {n20[names.index(h)]}개)" for h in HARD], fontsize=8)
ax.set_xticks(range(len(AES)), [a[2] for a in AES], fontsize=9)
ax.set_title("위험도 (quick_cv seed 3). 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 정상 비율 (진할수록 틀리기 쉬움)", fontsize=9)
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "단일 타격 정상(025 116 055)이 옅어지는가, 그 대가로 진해지는 고장이 있는가")

# %% 그림 7 (맨 끝): 놓친 파일 · 오경보 정상 vs 정상 5–95%
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, list(FOCUS)), "psd_missed", "놓친 022·005와 오경보 정상 025·116·055·059의 PSD가 정상 범위 밖으로 나가는 주파수가 있는가")
print("done", flush=True)
