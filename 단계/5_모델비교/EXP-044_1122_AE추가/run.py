# %% [markdown]
# EXP-044 (2차 설계, 11-22) G(EXP-043)에 스펙트럼 AE z를 세 번째 결합 입력으로 추가
# - J = TwoAE(ET 안쪽 OOF logit, 박자 z, AE z → 로지스틱) OR E2 OR 8.3k 규칙 (최종 후보). 비교: K = 같은 3입력 결합만, L = J에서 8.3k만 뺌
# - 문턱 · E2 · 8.3k는 EXP-043과 같음. AE = src.experts.ae_mean (정상만 학습, 안쪽 OOF)
# 실행: conda run -n qe-phm python "단계/5_모델비교/EXP-044_1122_AE추가/run.py"   (코드 · 출력은 이 파일 하나, 11-22 사용자)

# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp2util import *
from src.models2 import TwoAE
from src.evaluate import best_f2_threshold
from src.experts import LP as LP_TR, segment_logpsd
from src.final import _fit_ae, _ae_score
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
EXP043 = os.path.join(_d, "단계", "5_모델비교", "base", "EXP-043_1121_E1폐기_8.3k규칙")
ex = Experiment(__file__, "EXP-044", "G + 스펙트럼 AE (세 번째 결합 입력)", ["H3", "H2", "H1"],
                member="이동훈", model="ET OOF logit + 박자 z + AE z → 로지스틱 OR E2 OR 8.3k 규칙",
                changes="G(EXP-043)에 AE z를 결합 입력으로 추가 (src.models2.TwoAE)",
                key_changes=["G(EXP-043)에 스펙트럼 AE z를 세 번째 결합 입력으로 추가", "문턱 · E2 · 8.3k 규칙은 EXP-043과 같음"])
ns = NS(); names, y, g = ns.names, ns.y, ns.g; nor = y == 0
r43 = json.load(open(os.path.join(EXP043, "result.json"), encoding="utf-8"))["configs"]
for k, nm in (("W | 2단계 승자 C-lin (EXP-041 값)", "W | 2단계 승자 C-lin (EXP-041 값)"), ("G | W + E2 + 8.3k (최종)", "G | W + E2 + 8.3k (EXP-043 값)"), ("I | W + E1 + E2 + 8.3k (E1 폐기 전)", "I | W + E1 + E2 + 8.3k (E1 폐기 전)")):
    ex.res["configs"]["중첩: " + nm] = dict(r43["중첩: " + k], mid=True)
ex._write()
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res).astype(float); r83v = (specv | resv).astype(float)
idx = np.arange(ns.n, dtype=float); M_ae = np.column_stack([ns.M, idx]); M_rule = np.column_stack([ns.M, idx, r83])
mkAE = lambda experts=(): (lambda: TwoAE("lin", experts, fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))

# %% 중첩 CV (J가 대표, K · L은 중간 단계 결과)
for nm, F_, mk_, main in (("J | W + AE + E2 + 8.3k (최종 후보)", M_rule, lambda: Add83(mkAE(("E2",))), True), ("K | W + AE (전문가 · 규칙 없음)", M_ae, mkAE(()), False), ("L | W + AE + E2 (8.3k 없음)", M_ae, mkAE(("E2",)), False)):
    c = ex.record("중첩: " + nm, nested_cv(F_, y, g, mk_, n_jobs=-1), names, main=main, mid=not main)
J = ex.res["configs"]["중첩: J | W + AE + E2 + 8.3k (최종 후보)"]

# %% OOF seed 0 · 전체 학습 · ET 단독 경계
def oof_ae(seed=0):
    o = dict(a=np.empty(ns.n), z=np.empty(ns.n), ae=np.empty(ns.n), score=np.empty(ns.n), main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool), R83=np.zeros(ns.n, bool))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(g, y, g):
        f = Add83(mkAE(("E2",))).fit(M_rule[tr], y[tr]).fires(M_rule[te])
        for k in o: o[k][te] = f[k]
    return o


fo = oof_ae(); a, aez, zz = fo["a"], fo["ae"], fo["z"]; caught = fo["main"] | fo["E2"] | fo["R83"]
mJ = Add83(mkAE(("E2",))).fit(M_rule, y); mm = mJ.m_
t_et = float(best_f2_threshold(y, mm.Z_[:, 0]))                       # ET 단독 경계: 학습 OOF ET logit에서 F2 최대 문턱
et_on = a > t_et; et_miss, et_fp = int((~nor & ~et_on).sum()), int((nor & et_on).sum())
cat = np.where(fo["R83"], "8.3k", np.where(fo["E2"], "E2", np.where(fo["main"], "결합", "—")))

# %% 핵심 그림: ET × AE 평면 + ET 단독 경계 + 결합 경계 + 최종 판정 (데이터 범위로 크롭)
x0, x1, y0, y1 = a.min() - .3, a.max() + .3, aez.min() - .3, aez.max() + .3
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = mm._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(11, 8.2))
ax.axvspan(t_et, x1, color="#fbe3d6", alpha=.45, zorder=0); ax.axvline(t_et, color="#7a3db8", lw=2.2, label=f"ET 단독 경계 (ET ≥ {t_et:.2f}): 놓침 {et_miss} · 오경보 {et_fp}")
ax.contour(xx, yy, D, levels=[mm.tc_], colors="k", linewidths=1.3, linestyles="--"); ax.plot([], [], "k--", lw=1.3, label="결합 경계 (박자 z = 0 단면)")
ax.scatter(a[nor & ~caught], aez[nor & ~caught], color=NOR, s=34, marker="s", lw=0, alpha=.75, label="정상 → 정상 판정")
ax.scatter(a[nor & caught], aez[nor & caught], color=NOR, s=100, marker="s", edgecolor="#d62728", lw=2, label="정상 → 오경보 (최종)")
for k, c in (("결합", "#e8590c"), ("E2", "#7a3db8"), ("8.3k", "#000000")):
    m = (cat == k) & ~nor; ax.scatter(a[m], aez[m], s=90 if k == "결합" else 150, color=ABN, edgecolor=c, lw=.5 if k == "결합" else 2.4, label=f"고장 → {k}이 잡음 ({int(m.sum())})")
big = ~nor & (zz > .5); ax.scatter(a[big], aez[big], s=240, facecolor="none", edgecolor="#1f9d55", lw=1.6, label=f"고장 · 박자 z > 0.5 ({int(big.sum())})")
for f_ in ("abn_002", "abn_017", "abn_022", "nor_116", "nor_059", "nor_065"):
    i = names.index(f_); ax.annotate(short(f_), (a[i], aez[i]), xytext=(8, 8), textcoords="offset points", fontsize=9, color=ABN if y[i] else NOR, weight="bold")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xlabel("ET 점수 (logit, OOF)"); ax.set_ylabel("AE z (OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"J — 중첩 F2 {J['f2']:.3f} ± {J['f2_sd']:.3f} · 재현율 {J['recall']:.3f} · 오경보율 {J['fpr']:.3f} (OOF seed 0: 고장 {int(caught[~nor].sum())}/30 · 오경보 {int(caught[nor].sum())}/120)", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "ET × AE 평면 (OOF seed 0). 보라 세로선 = ET 단독 경계, 점선 = 결합 경계, 색 테두리 = 고장을 잡은 부품", key=True)

# %% 오류 분석 표 (md, 가로): 우리 판단(정답) · 모델 판단
cnt = np.array([(5 - J["miss_counts"].get(n_, 0)) if y[i] else J["fp_counts"].get(n_, 0) for i, n_ in enumerate(names)])
files = ["abn_002", "abn_017", "abn_022"] + [n_ for n_ in names if n_.startswith("nor") and J["fp_counts"].get(n_, 0) >= 1]
ex_ = lambda arr, fmt: {n_: fmt(arr[names.index(n_)]) for n_ in files}
tab = error_md_table(names, y, cnt, files, extra={"ET 점수": ex_(a, lambda v: f"{v:.2f}"), "ET 단독 판정": ex_(et_on, lambda v: "고장" if v else "정상"), "박자 z": ex_(zz, lambda v: f"{v:.1f}"),
                                                    "AE z": ex_(aez, lambda v: f"{v:.2f}"), "켠 부품": ex_(cat, str)})
ex.save_md("오류분석_표", f"# EXP-044 오류 분석 (중첩 CV seed 5개, 판정 횟수 / OOF seed 0 근거)\n\n우리 판단 = 정답 라벨, 모델 판단 = 최종 판정(결합 OR E2 OR 8.3k).\n\n{tab}\n")

# %% val 판정 (train 150 전체 학습, 제출 파일 아님): md + 그림
ae_tr = _fit_ae(LP_TR[y == 0]); _, LPv = segment_logpsd(ns.Xv); mm.ae_override_ = _ae_score(ae_tr, LPv)
Mv = np.column_stack([ns.Mv, np.zeros(len(ns.Mv)), r83v]); fv = mJ.fires(Mv); pv = mJ.decision_function(Mv) >= .5
catv = np.where(fv["R83"], "8.3k", np.where(fv["E2"], "E2", np.where(fv["main"], "결합", "—")))
fig, ax = plt.subplots(figsize=(11, 8.2))
ax.axvspan(t_et, max(x1, fv["a"].max() + .3), color="#fbe3d6", alpha=.45, zorder=0); ax.axvline(t_et, color="#7a3db8", lw=2.2, label=f"ET 단독 경계 (ET ≥ {t_et:.2f})")
ax.scatter(a[nor], aez[nor], color="#c8c8c8", s=24, marker="s", lw=0, label="train 정상 (OOF)"); ax.scatter(a[~nor], aez[~nor], color="#f2b8a0", s=28, lw=0, label="train 고장 (OOF)")
ax.scatter(fv["a"][~pv], fv["ae"][~pv], color=NOR, s=70, marker="^", edgecolor="white", lw=.6, label=f"val → 정상 판정 ({int((~pv).sum())})")
ax.scatter(fv["a"][pv], fv["ae"][pv], color=ABN, s=95, marker="^", edgecolor="black", lw=1, label=f"val → 고장 판정 ({int(pv.sum())})")
for i in range(len(pv)):
    if pv[i] or abs(fv["a"][i] - t_et) < .5: ax.annotate(ns.nv[i][-3:], (fv["a"][i], fv["ae"][i]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=ABN if pv[i] else NOR)
ax.set_xlabel("ET 점수 (logit)"); ax.set_ylabel("AE z (val은 val 신호로 계산, 학습 정상 기준)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"val 40개 판정 — 고장 {int(pv.sum())}개 (ET 단독 경계로는 {int((fv['a'] > t_et).sum())}개)", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "val_판정", "val 40개를 train OOF 위에 겹침. 보라 세로선 = ET 단독 경계")
V = pd.DataFrame(dict(파일=[n_[-3:] for n_ in ns.nv], 판정=np.where(pv, "고장", "정상"), ET=fv["a"].round(2), ET단독=np.where(fv["a"] > t_et, "고장", "정상"), AE_z=fv["ae"].round(2), 박자z=fv["z"].round(1), 켠부품=catv))
ex.save_md("val_판정", f"# EXP-044 val 판정 (train 150 전체 학습, 제출 파일 아님)\n\n고장 판정 {int(pv.sum())}개: {' '.join(V.파일[pv])}\n\n![val](figs/val_판정.png)\n\n{md_table(V)}\n")
# %% t-SNE (라벨 색, train 150): 새 피처 17열을 정상 기준 강건 표준화 → 2차원. README 그림
from sklearn.manifold import TSNE
Tn = ns.T.copy(); pk = PeakRef(ns.df).fit(ns.L[nor]).transform(ns.L); Tn["H16_peak_h"], Tn["H17_peak_w"] = pk[:, 0], pk[:, 1]
Zs = np.column_stack([robust_z(Tn[c].values[nor])(Tn[c].values) for c in ns.cols]); best = None
for s_ in range(3):
    t_ = TSNE(2, perplexity=30, init="random", random_state=s_, learning_rate="auto"); Em = t_.fit_transform(Zs)
    if best is None or t_.kl_divergence_ < best[0]: best = (t_.kl_divergence_, Em)
Em = best[1]; fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(*Em[nor].T, s=28, color=NOR, alpha=.6, lw=0, label="정상 (라벨)"); ax.scatter(*Em[~nor].T, s=50, color=ABN, lw=0, label="고장 (라벨)")
ax.scatter(*Em[nor & caught].T, s=130, facecolor="none", edgecolor="#d62728", lw=1.6, label="J 오경보 (seed 0)")
for f_ in ("abn_002", "abn_017", "abn_022", "nor_116", "nor_065", "nor_059"):
    i = names.index(f_); ax.annotate(short(f_), Em[i], xytext=(4, 4), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
ax.set_xticks([]); ax.set_yticks([]); ax.legend(fontsize=9); ax.set_title(f"t-SNE (라벨 색, 새 피처 {len(ns.cols)}열, perplexity 30, KL {best[0]:.2f})", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "tsne_라벨", "라벨(정상/고장)로 색을 입힌 t-SNE. 빨간 원 = J 오경보")
print("done")
