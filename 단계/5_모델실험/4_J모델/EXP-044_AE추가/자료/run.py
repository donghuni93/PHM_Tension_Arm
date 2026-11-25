# %% [markdown]
# EXP-044 (2차 설계, 11-22) G(EXP-043)에 스펙트럼 AE z를 세 번째 결합 입력으로 추가
# - J = TwoAE(ET 안쪽 OOF logit, 박자 z, AE z → 로지스틱) OR E2 OR 8.3k 규칙 (최종 후보). 비교: K = 같은 3입력 결합만, L = J에서 8.3k만 뺌
# - 문턱 · E2 · 8.3k는 EXP-043과 같음. AE = src.experts.ae_mean (정상만 학습, 안쪽 OOF)
# 실행: conda run -n qe-phm python "단계/5_모델실험/4_J모델/EXP-044_AE추가/자료/run.py"   (코드 · 출력은 이 파일 하나, 11-22 사용자)

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
EXP043 = find_exp("EXP-043")
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
c_ = mm.c_.coef_[0]; SH = c_[1] / c_[0]; ap = a + SH * zz              # 점 위치 = 실제 판정 위치 (ET + 박자 z 환산). ET 단독 경계(보라선)는 원래 ET 값 기준
def tails(ax, raw, eff, yv, color, lo=.3):                           # 박자 z 때문에 옮겨 온 점의 원래(박자 z 0) 위치와 이동선
    m = np.abs(eff - raw) >= lo
    ax.hlines(yv[m], raw[m], eff[m], color=color, lw=1, ls=":", alpha=.7, zorder=1); ax.scatter(raw[m], yv[m], s=9, color=color, zorder=1)
x0, x1, y0, y1 = ap.min() - .3, ap.max() + .3, aez.min() - .3, aez.max() + .3
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = mm._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(11, 8.2))
ax.contourf(xx, yy, D, levels=[-1e9, mm.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.85, zorder=0); ax.axvline(t_et, color="#7a3db8", lw=2.6, label=f"ET 단독 경계 (ET ≥ {t_et:.2f}): 놓침 {et_miss} · 오경보 {et_fp}")
ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#dbe8f7", label="배경 = 결합(로지스틱) 판단: 정상")); ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#fbe3d6", label="배경 = 결합(로지스틱) 판단: 고장 (점 위치에 박자 z 반영)"))
ax.scatter(ap[nor & ~caught], aez[nor & ~caught], color=NOR, s=34, marker="s", lw=0, alpha=.75, label="정상 → 정상 판정")
ax.scatter(ap[nor & caught], aez[nor & caught], color=NOR, s=100, marker="s", edgecolor="#d62728", lw=2, label="정상 → 오경보 (최종)")
for k, c in (("결합", "#e8590c"), ("E2", "#7a3db8"), ("8.3k", "#000000")):
    m = (cat == k) & ~nor; ax.scatter(ap[m], aez[m], s=90 if k == "결합" else 150, color=ABN, edgecolor=c, lw=.5 if k == "결합" else 2.4, label=f"고장 → {k}이 잡음 ({int(m.sum())})")
big = ~nor & (zz > .5); ax.scatter(ap[big], aez[big], s=240, facecolor="none", edgecolor="#1f9d55", lw=1.6, label=f"고장 · 박자 z > 0.5 ({int(big.sum())})")
for f_ in ("abn_002", "abn_017", "abn_022", "nor_116", "nor_059", "nor_065"):
    i = names.index(f_); ax.annotate(short(f_), (ap[i], aez[i]), xytext=(8, 8), textcoords="offset points", fontsize=9, color=ABN if y[i] else NOR, weight="bold")
tails(ax, a, ap, aez, "#777")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xlabel("ET 점수 (logit, OOF) + 박자 z 환산  (점 위치 = 실제 판정 위치, 점선 꼬리 = ET 값 위치에서 옮겨 온 거리)"); ax.set_ylabel("AE z (OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"J — 중첩 F2 {J['f2']:.3f} ± {J['f2_sd']:.3f} · 재현율 {J['recall']:.3f} · 오경보율 {J['fpr']:.3f} (OOF seed 0: 고장 {int(caught[~nor].sum())}/30 · 오경보 {int(caught[nor].sum())}/120)", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "점 위치 = ET + 박자 z 환산(점선 꼬리 = ET 값 위치). ET × AE 평면 (OOF seed 0). 배경 = 결합(로지스틱) 판단, 보라 세로선 = ET 단독 판단 경계, 색 테두리 = 고장을 잡은 부품", key=True)

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
fvp = fv["a"] + SH * fv["z"]
xv0, xv1, yv0, yv1 = min(x0, fvp.min() - .3), max(x1, fvp.max() + .3), min(y0, fv["ae"].min() - .3), max(y1, fv["ae"].max() + .3); xxv, yyv = np.meshgrid(np.linspace(xv0, xv1, 300), np.linspace(yv0, yv1, 300))
ax.contourf(xxv, yyv, mm._comb_score(xxv.ravel(), np.zeros(xxv.size), yyv.ravel()).reshape(xxv.shape), levels=[-1e9, mm.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.85, zorder=0); ax.axvline(t_et, color="#7a3db8", lw=2.6, label=f"ET 단독 경계 (ET ≥ {t_et:.2f})")
ax.scatter(ap[nor], aez[nor], color="#c8c8c8", s=24, marker="s", lw=0, label="train 정상 (OOF)"); ax.scatter(ap[~nor], aez[~nor], color="#f2b8a0", s=28, lw=0, label="train 고장 (OOF)")
ax.scatter(fvp[~pv], fv["ae"][~pv], color=NOR, s=70, marker="^", edgecolor="white", lw=.6, label=f"val → 정상 판정 ({int((~pv).sum())})")
ax.scatter(fvp[pv], fv["ae"][pv], color=ABN, s=95, marker="^", edgecolor="black", lw=1, label=f"val → 고장 판정 ({int(pv.sum())})")
for i in range(len(pv)):
    if pv[i] or abs(fvp[i] - t_et) < .5: ax.annotate(ns.nv[i][-3:], (fvp[i], fv["ae"][i]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=ABN if pv[i] else NOR)
tails(ax, a, ap, aez, "#bbb"); tails(ax, fv["a"], fvp, fv["ae"], "#7a3db8")
ax.set_xlim(xv0, xv1); ax.set_ylim(yv0, yv1); ax.set_xlabel("ET 점수 (logit) + 박자 z 환산  (점 위치 = 실제 판정 위치, 점선 꼬리 = ET 값 위치에서 옮겨 온 거리)"); ax.set_ylabel("AE z (val은 val 신호로 계산, 학습 정상 기준)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"val 40개 판정 — 고장 {int(pv.sum())}개 (ET 단독 경계로는 {int((fv['a'] > t_et).sum())}개)", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "val_판정", "점 위치 = ET + 박자 z 환산(점선 꼬리 = ET 값 위치). val 40개를 train OOF 위에 겹침. 배경 = 결합(로지스틱) 판단, 보라 세로선 = ET 단독 판단 경계")
V = pd.DataFrame(dict(파일=[n_[-3:] for n_ in ns.nv], 판정=np.where(pv, "고장", "정상"), ET=fv["a"].round(2), ET단독=np.where(fv["a"] > t_et, "고장", "정상"), AE_z=fv["ae"].round(2), 박자z=fv["z"].round(1), 켠부품=catv))
ex.save_md("val_판정", f"# EXP-044 val 판정 (train 150 전체 학습, 제출 파일 아님)\n\n고장 판정 {int(pv.sum())}개: {' '.join(V.파일[pv])}\n\n![val](figs/val_판정.png)\n\n{md_table(V)}\n")
# %% t-SNE 기본 (4-10 방식, 핵심 그림): H 열 + 박자 S + AE 5열, map_scale, 146개 (059 · 065 · 116 · 022 제외)
# 11-22 사용자 결정으로 예전 그림(robust_z: H9 척도가 MAD≈0이라 z 126, 지도가 한 줄로 늘어섬)을 교체. 함수 src.exp2util.tsne_default
fig, ti = tsne_default(ns, mark=np.flatnonzero(nor & caught), mark_label="J 오경보 (OOF seed 0)", title="EXP-044 J")
ex.save_fig(fig, "tsne_라벨", f"기본 t-SNE(4-10): 왼쪽 라벨 없음, 오른쪽 라벨 색. 빨간 원 = J 오경보. 폴드 밖 5-NN AUC {ti['auc']:.3f}", key=True)

# %% 기본 t-SNE + val (11-22 사용자 요청, README2 그림): 기본 공간에 val 40개. 왼쪽 = 가까운 train 5개 위치, 오른쪽 = 함께 다시 펼침. 함수 src.exp2util.tsne_val_default
fig, tv = tsne_val_default(ns, pv, title="EXP-044 J 판정")
ex.save_fig(fig, "tsne_val포함", "기본 t-SNE + val 40개 (보라 ▲ = J 판정, 진한 = 고장). 빨간 원 = train에서 먼 val (가장 가까운 train 거리가 train끼리 거리의 95 % 위)")
V["이웃고장(5)"], V["최근접거리_백분위"], V["최대z열"], V["최대z"] = tv["이웃고장"].values, tv["거리백분위"].values, tv["최대z열"].values, tv["최대z"].values
ex.save_md("val_판정", f"# EXP-044 val 판정 (train 150 전체 학습, 제출 파일 아님)\n\n고장 판정 {int(pv.sum())}개: {' '.join(V.파일[pv])}\n\n![val](figs/val_판정.png)\n\n"
           f"이웃고장(5) = 기본 t-SNE 공간(t-SNE 전)에서 가까운 train 5개 중 고장 수. 최근접거리_백분위 = 가장 가까운 train까지 거리가 train끼리 최근접 거리 분포의 몇 %인가(95 넘으면 train에서 먼 값). 최대z = 정상 기준 눈금에서 가장 벗어난 열\n\n{md_table(V)}\n")
print("done")
