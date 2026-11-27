# %% [markdown]
# EXP-023 회색 고장 014·017·022를 정상과 가르는 모델 찾기 (모델 단위)
# - 기준 = EXP-021 (AE + ET → 로지스틱 + 8.3k 규칙), 중첩 F2 0.892 ± 0.015. 이 셋이 늘 놓침(017·022 5/5, 014 4/5)
# - 관찰 (EXP-019·022): 세 파일은 서로 다른 이유로 회색이다
#   014 = 이미 정상 중 0.95(임계값 근처) · 017 = AE가 정상으로 봄(AE를 빼면 0.9+) · 022 = 원신호 모델만 0.92–0.93
#   → 다수 고장과 닮았나(지도 학습)보다 '정상 무리에서 국소적으로 떨어졌나'(정상만 학습)가 맞을 수 있다
# - 후보 단일 점수
#   A EXP-019 후보 8종 (MiniRocket·1D-CNN·합성곱AE·DeepSAD·2D-CNN·AR잔차·MSPC·CatBoost)
#   B 정상만 학습하는 국소 이상: {MiniRocket PPV, AR 잔차 피처, ext150} × {kNN 거리, LOF, Isolation Forest} (기본값 고정)
# - 결합 (실행 전 고정): 로지스틱에 섞으면 희석됨(EXP-019·022) → 기존 결합은 그대로 두고 후보는 '강한 증거'로만:
#   점수 = max(로지스틱 e, 후보 e), e = 학습 정상 OOF의 (s − 중앙값)/(99% − 중앙값) → 후보가 정상 99%선을 넘을 때만 영향 + 8.3k 규칙
# - 판단: ① 단일 점수의 세 파일 정상 중 백분위(OOF) ② 결합의 분리 판정(확실한 고장·회색 고장·위험 정상)
#   중첩 CV 대상 (실행 전 고정): 결합 중 회색 고장 수 < 기준 이고 위험 정상 ≤ 기준 + 1 인 것, 세 파일 개선 수 순 상위 2개
# - 주의: 세 파일을 보고 고르는 실험이다 → 고른 구성의 점수는 낙관적. 중첩 CV는 안쪽에서 임계값만 고르므로 선택 편향은 남는다
# 실행: conda run -n qe-phm python 3_단계/5_모델비교/스크리닝/EXP-023_1114_회색고장전용모델/run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, separation
from src.experts import ScoreStack, RuleOverride, Rule83, ae_mean, et_logit, ae_segerr, et_trees, warm_cache
import src.candidates as C
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
ex = Experiment(__file__, "EXP-023", "회색 고장 014·017·022 전용 모델 찾기 (국소 이상 + 강한 증거 결합)", ["H1", "H2", "H3"],
                member="이동훈", model="EXP-021 + 후보 단일 점수(정상만 학습 국소 이상 포함)를 강한 증거(max, 정상 99%선)로",
                changes="EXP-021 결합은 그대로, 후보가 정상 99%선을 넘을 때만 고장으로 올림", focus=())
X, y, names = load_train()
g = np.arange(len(y))
SEEDS = range(3)
nor = y == 0
RULE = Rule83(0.75, 150)
TGT = ["abn_014", "abn_017", "abn_022"]
it = [names.index(n) for n in TGT]
FA = ["nor_025", "nor_055", "nor_116", "nor_059"]          # EXP-021 자주 오경보
SINGLES = [("MiniRocket", C.minirocket_ridge), ("1D-CNN", C.cnn1d_mil), ("합성곱AE", C.convae_stft), ("DeepSAD", C.deepsad_stft),
           ("2D-CNN", C.cnn2d_mil), ("AR잔차", C.ar_resid_lr), ("MSPC", C.mspc_ext), ("CatBoost", C.catboost_ext)]
SINGLES += [(f"정상만 {r}·{m}", C.oc(r, m)) for r in ("mr", "ar", "ext") for m in ("knn", "lof", "iforest")]

warm_cache([ae_segerr, et_trees], seeds=SEEDS)
for nm, fn in SINGLES:
    warm_cache([fn], seeds=SEEDS, n_jobs=1 if nm in ("1D-CNN", "합성곱AE", "DeepSAD", "2D-CNN") else 8)
print("캐시 완료", flush=True)


def qcv(make):
    q = quick_cv(g[:, None], y, g, make, seeds=SEEDS)
    q["pct"] = np.mean([(o[nor][None, :] < o[:, None]).mean(1) for o in q["oof"]], 0)
    q["sep"] = separation(q["oof"], y, names)
    return q


# %% ① 단일 점수: 세 파일이 정상 중 몇 번째인가 (OOF, seed 3 평균)
S = {"AE (기준 성분)": qcv(lambda: ScoreStack((ae_mean,), combine="col0")),
     "ET (기준 성분)": qcv(lambda: ScoreStack((et_logit,), combine="col0"))}
for nm, fn in SINGLES:
    S[nm] = qcv(lambda fn=fn: ScoreStack((fn,), combine="col0"))
rows = [f"{'단일 점수':18s} PR-AUC   " + "  ".join(n[4:] for n in TGT) + "   셋 다 0.90+ ?   (정상 중 백분위, 높을수록 정상과 멀리)"]
for k, q in S.items():
    p = q["pct"][it]
    rows.append(f"{k:18s} {q['pr_auc'].mean():.3f}    " + "  ".join(f"{v:.2f}" for v in p) + f"   {'예' if (p >= .9).all() else ''}")
ex.note("① 단일 점수 — 회색 고장 정상 중 백분위 (quick_cv OOF, seed 3 평균)", "\n".join(rows))
print("\n".join(rows), flush=True)


# %% ② 결합: EXP-021 + 후보 강한 증거 (max, 정상 99%선)
def make_strong(fn):
    return lambda: RuleOverride(base_kw=dict(experts=(ae_mean, et_logit, fn), combine="strong", lr_cols=[0, 1], or_cols=[2], q=99), rule=RULE)


R = {"기준 EXP-021": qcv(lambda: RuleOverride(base_kw=dict(experts=(ae_mean, et_logit), combine="lr"), rule=RULE))}
for nm, fn in SINGLES:
    R[f"+ {nm}"] = qcv(make_strong(fn))
B = R["기준 EXP-021"]
rows = [f"{'결합':20s} PR-AUC  확실한고장  회색고장  위험정상   " + "  ".join(n[4:] for n in TGT) + "  " +
        "  ".join(n[4:] for n in FA) + "  (고장 = 정상 중 백분위↑ 좋음, 정상 = ↓ 좋음)"]
CAND = []
for k, q in R.items():
    s = q["sep"]
    p = q["pct"]
    better = int(sum(p[i] > B["pct"][i] + .02 for i in it))
    ok = k != "기준 EXP-021" and len(s["gray_a"]) < len(B["sep"]["gray_a"]) and s["risk_n"] <= B["sep"]["risk_n"] + 1
    if ok:
        CAND.append((better, s["sure_a"], k))
    rows.append(f"{k:20s} {q['pr_auc'].mean():.3f}   {s['sure_a']:4d}       {len(s['gray_a']):3d}     {s['risk_n']:5.1f}    " +
                "  ".join(f"{p[i]:.2f}" for i in it) + "  " + "  ".join(f"{p[names.index(n)]:.2f}" for n in FA) +
                f"   {'후보' if ok else ''}  회색: {' '.join(x[4:] for x in s['gray_a'])}")
ex.note("② 결합 — EXP-021 + 후보 강한 증거 (분리 판정 + 회색 고장·자주 오경보 정상의 정상 중 백분위)", "\n".join(rows))
print("\n".join(rows), flush=True)


# %% 그림: 단일 점수의 세 파일 백분위 · 결합의 분리 지표
fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 8))
ks = list(S)
P = np.array([S[k]["pct"][it] for k in ks])
for j, (n, c) in enumerate(zip(TGT, ("#eb6834", "#a23b72", "#2a78d6"))):
    a1.scatter(P[:, j], range(len(ks)), color=c, s=40, label=n)
a1.axvline(.9, color="k", lw=.8, ls="--")
a1.set_yticks(range(len(ks))); a1.set_yticklabels(ks, fontsize=8); a1.invert_yaxis(); a1.set_xlim(0, 1.02)
a1.set_xlabel("정상 중 백분위 (1 = 모든 정상보다 높음)"); a1.legend(fontsize=8); a1.grid(alpha=.3)
a1.set_title("① 단일 점수: 회색 고장 셋이 정상에서 얼마나 떨어지는가", fontsize=10)
kr = list(R)
sa = [R[k]["sep"]["sure_a"] for k in kr]; ga = [len(R[k]["sep"]["gray_a"]) for k in kr]; rn = [R[k]["sep"]["risk_n"] for k in kr]
yy = np.arange(len(kr))
a2.barh(yy - .25, sa, .25, color="#eb6834", label="확실한 고장 (/30)")
a2.barh(yy, ga, .25, color="#9aa5b1", label="회색 고장")
a2.barh(yy + .25, rn, .25, color="#2a78d6", label="위험 정상")
for v, c in ((sa[0], "#eb6834"), (ga[0], "#9aa5b1"), (rn[0], "#2a78d6")):
    a2.axvline(v, color=c, lw=.8, ls="--")
a2.set_yticks(yy); a2.set_yticklabels(kr, fontsize=8); a2.invert_yaxis(); a2.legend(fontsize=8); a2.grid(alpha=.3, axis="x")
a2.set_title("② EXP-021 + 후보 강한 증거: 분리 지표 (점선 = 기준)", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "gray_targets", "어떤 모델이 014·017·022를 정상 무리 밖(0.9+)으로 보내는가, 강한 증거로 붙이면 회색 고장이 줄고 오경보는 늘지 않는가")


# %% 중첩 CV (조건 통과 결합, 세 파일 개선 수 → 확실한 고장 수 순 상위 2개). 기준 0.892 ± 0.015
CAND = [k for _, _, k in sorted(CAND, key=lambda t: (-t[0], -t[1]))[:2]]
ex.note("중첩 CV 대상", ", ".join(CAND) or "없음")
if CAND:
    warm_cache([ae_segerr, et_trees], seeds=range(5), nested=True)
    for k in CAND:
        fn = dict(SINGLES)[k[2:]]
        warm_cache([fn], seeds=range(5), nested=True, n_jobs=1 if k[2:] in ("1D-CNN", "합성곱AE", "DeepSAD", "2D-CNN") else 8)
        r = nested_cv(g[:, None], y, g, make_strong(fn), n_jobs=-1)
        ex.record("중첩: EXP-021 " + k, r, names, main=(k == CAND[0]))
print("done", flush=True)
