# %% [markdown]
# EXP-046 (2차 설계, 11-22) EXP-044 J에서 이상치 정상 3개(N059 · N065 · N116)를 학습에서 제외 — 사용자 결정
# - 이 셋은 데이터로 구분할 수 없는 정상으로 보고 학습에서만 뺀다. 평가 · 결과 계산(F2 · 오경보)에는 포함, 그림에는 따로 표시 (src.exp.OUTLIER_NORMALS)
# - 구성은 EXP-044 J와 같음: ET + 박자 z + AE z → 로지스틱 OR E2 OR 8.3k. 빠지는 것은 학습 행뿐 (AE · E2 문턱 · 정상 기준 z도 이 3개 없이 계산)
# 실행: conda run -n qe-phm python "단계/5_모델비교/EXP-046_1122_이상치정상_학습제외/run.py"   (코드 · 출력은 이 파일 하나)

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
from joblib import Parallel, delayed, parallel_config
from sklearn.manifold import TSNE
from sklearn.metrics import fbeta_score, recall_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
EXP044 = os.path.join(_d, "단계", "5_모델비교", "EXP-044_1122_AE추가")
ex = Experiment(__file__, "EXP-046", "J에서 이상치 정상 3개를 학습에서 제외", ["H3", "H2", "H1"],
                member="이동훈", model="ET + 박자 z + AE z → 로지스틱 OR E2 OR 8.3k (정상 059 · 065 · 116 학습 제외)",
                changes="EXP-044 J에서 이상치 정상 3개를 학습에서만 뺌 (평가에는 포함)",
                key_changes=["정상 059 · 065 · 116을 학습에서 제외 (평가 · 결과 계산에는 포함)", "나머지는 EXP-044 J와 같음"])
ns = NS(); names, y, g = ns.names, ns.y, ns.g; nor = y == 0
OUT = outlier_idx(names); isout = np.zeros(ns.n, bool); isout[OUT] = True; keep = ~isout
ex.res["configs"]["중첩: J | EXP-044 값 (이상치 포함 학습)"] = dict(json.load(open(os.path.join(EXP044, "result.json"), encoding="utf-8"))["configs"]["중첩: J | W + AE + E2 + 8.3k (최종 후보)"], mid=True)
ex._write()
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res); r83v = (specv | resv)
Fn = np.column_stack([ns.M, np.arange(ns.n, dtype=float)])
mkx = lambda: ExcludeTrain(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)), OUT)


# %% 중첩 CV (같은 분할: 바깥 5겹 × seed 5개, 폴드마다 한 번 학습. 판정 = 결합 OR E2 OR 8.3k)
def task(seed, tr, te):
    f = mkx().fit(Fn[tr], y[tr]).fires(Fn[te])
    return seed, te, f["main"] | f["E2"] | r83[te], {k: f[k] for k in ("a", "z", "ae", "main", "E2")}


jobs = [(s, tr, te) for s in range(5) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
with parallel_config(backend="loky", inner_max_num_threads=1):
    R = Parallel(n_jobs=-1)(delayed(task)(*j) for j in jobs)
PRED = np.zeros((5, ns.n), bool); fo = {k: np.zeros(ns.n) for k in ("a", "z", "ae")}; fo.update(main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool))
for s, te, p, f in R:
    PRED[s, te] = p
    if s == 0:
        for k in fo: fo[k][te] = f[k]


def as_nested(P, yy):
    return dict(f2=np.array([fbeta_score(yy, p, beta=2) for p in P]), recall=np.array([recall_score(yy, p) for p in P]), precision=np.array([precision_score(yy, p, zero_division=0) for p in P]),
                fpr=np.array([p[yy == 0].mean() for p in P]), miss_counts=((yy == 1) & ~P).sum(0), fp_counts=((yy == 0) & P).sum(0))


K = ex.record("중첩: K | J, 이상치 3개 학습 제외 (150개 평가)", as_nested(PRED, y), names, main=True)
ex.record("중첩: 참고 | 같은 예측, 이상치 3개를 평가에서도 뺌 (147개)", as_nested(PRED[:, keep], y[keep]), [n_ for n_, k_ in zip(names, keep) if k_], mid=True)

# %% 핵심 그림: ET × AE 평면 + ET 단독 경계 + 결합 경계 + 최종 판정 (이상치 3개는 ◆로 따로)
a, aez, zz = fo["a"], fo["ae"], fo["z"]; caught = fo["main"] | fo["E2"] | r83
mex = mkx().fit(Fn, y); mm = mex.m_
t_et = float(best_f2_threshold(mm.y_, mm.Z_[:, 0]))                      # ET 단독 경계: 학습 OOF ET logit에서 F2 최대 문턱
et_on = a > t_et; et_miss, et_fp = int((~nor & ~et_on).sum()), int((nor & et_on).sum())
cat = np.where(r83, "8.3k", np.where(fo["E2"], "E2", np.where(fo["main"], "결합", "—")))
c_ = mm.c_.coef_[0]; SH = c_[1] / c_[0]; ap = a + SH * zz              # 점 위치 = 실제 판정 위치 (ET + 박자 z 환산). ET 단독 경계(보라선)는 원래 ET 값 기준
def tails(ax, raw, eff, yv, color, lo=.3):                           # 박자 z 때문에 옮겨 온 점의 원래(박자 z 0) 위치와 이동선
    m = np.abs(eff - raw) >= lo
    ax.hlines(yv[m], raw[m], eff[m], color=color, lw=1, ls=":", alpha=.7, zorder=1); ax.scatter(raw[m], yv[m], s=9, color=color, zorder=1)
x0, x1, y0, y1 = ap.min() - .3, ap.max() + .3, aez.min() - .3, aez.max() + .3
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = mm._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(11, 8.2))
ax.contourf(xx, yy, D, levels=[-1e9, mm.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.85, zorder=0); ax.axvline(t_et, color="#7a3db8", lw=2.6, label=f"ET 단독 경계 (ET ≥ {t_et:.2f}): 놓침 {et_miss} · 오경보 {et_fp} (이상치 포함)")
ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#dbe8f7", label="배경 = 결합(로지스틱) 판단: 정상")); ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#fbe3d6", label="배경 = 결합(로지스틱) 판단: 고장 (점 위치에 박자 z 반영)"))
n1 = nor & keep
ax.scatter(ap[n1 & ~caught], aez[n1 & ~caught], color=NOR, s=34, marker="s", lw=0, alpha=.75, label="정상 → 정상 판정")
ax.scatter(ap[n1 & caught], aez[n1 & caught], color=NOR, s=100, marker="s", edgecolor="#d62728", lw=2, label="정상 → 오경보 (최종)")
for k, c in (("결합", "#e8590c"), ("E2", "#7a3db8"), ("8.3k", "#000000")):
    m = (cat == k) & ~nor; ax.scatter(ap[m], aez[m], s=90 if k == "결합" else 150, color=ABN, edgecolor=c, lw=.5 if k == "결합" else 2.4, label=f"고장 → {k}이 잡음 ({int(m.sum())})")
ax.scatter(ap[isout], aez[isout], s=190, marker="D", color="#9aa0a6", edgecolor=["#d62728" if c else "#444" for c in caught[isout]], lw=2.2, label="이상치 정상 059 · 065 · 116 (학습 제외, 빨간 테두리 = 오경보)")
for i in OUT: ax.annotate(short(names[i]), (ap[i], aez[i]), xytext=(8, 8), textcoords="offset points", fontsize=9, color="#444", weight="bold")
for f_ in ("abn_002", "abn_017", "abn_022"):
    i = names.index(f_); ax.annotate(short(f_), (ap[i], aez[i]), xytext=(8, 8), textcoords="offset points", fontsize=9, color=ABN, weight="bold")
tails(ax, a, ap, aez, "#777")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xlabel("ET 점수 (logit, OOF) + 박자 z 환산  (점 위치 = 실제 판정 위치, 점선 꼬리 = ET 값 위치에서 옮겨 온 거리)"); ax.set_ylabel("AE z (OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"K — 중첩 F2 {K['f2']:.3f} ± {K['f2_sd']:.3f} · 재현율 {K['recall']:.3f} · 오경보율 {K['fpr']:.3f} (OOF seed 0: 고장 {int(caught[~nor].sum())}/30 · 오경보 {int(caught[nor].sum())}/120)", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "점 위치 = ET + 박자 z 환산(점선 꼬리 = ET 값 위치). ET × AE 평면 (OOF seed 0). 배경 = 결합(로지스틱) 판단, 보라 세로선 = ET 단독 판단 경계, ◆ = 학습에서 뺀 이상치 정상 3개", key=True)

# %% 오류 분석 표 (md, 가로): 우리 판단(정답) · 모델 판단
cnt = np.array([(5 - K["miss_counts"].get(n_, 0)) if y[i] else K["fp_counts"].get(n_, 0) for i, n_ in enumerate(names)])
files = ["abn_002", "abn_017", "abn_022"] + list(OUTLIER_NORMALS) + [n_ for n_ in names if n_.startswith("nor") and n_ not in OUTLIER_NORMALS and K["fp_counts"].get(n_, 0) >= 1]
ex_ = lambda arr, fmt: {n_: fmt(arr[names.index(n_)]) for n_ in files}
tab = error_md_table(names, y, cnt, files, extra={"학습": {n_: ("제외" if n_ in OUTLIER_NORMALS else "사용") for n_ in files}, "ET 점수": ex_(a, lambda v: f"{v:.2f}"), "ET 단독 판정": ex_(et_on, lambda v: "고장" if v else "정상"),
                                                    "박자 z": ex_(zz, lambda v: f"{v:.1f}"), "AE z": ex_(aez, lambda v: f"{v:.2f}"), "켠 부품": ex_(cat, str)})
ex.save_md("오류분석_표", f"# EXP-046 오류 분석 (중첩 CV seed 5개 판정 횟수 / OOF seed 0 근거)\n\n우리 판단 = 정답 라벨, 모델 판단 = 최종 판정(결합 OR E2 OR 8.3k). 이상치 3개는 학습에서만 제외, 여기 판정에는 포함.\n\n{tab}\n")

# %% val 판정 (학습 = 이상치 3개 뺀 147개, 제출 파일 아님): md + 그림
ae_tr = _fit_ae(LP_TR[(y == 0) & keep]); _, LPv = segment_logpsd(ns.Xv); mm.ae_override_ = _ae_score(ae_tr, LPv)
fv = mex.fires(np.column_stack([ns.Mv, np.zeros(len(ns.Mv))])); pv = fv["main"] | fv["E2"] | r83v
catv = np.where(r83v, "8.3k", np.where(fv["E2"], "E2", np.where(fv["main"], "결합", "—")))
fig, ax = plt.subplots(figsize=(11, 8.2))
fvp = fv["a"] + SH * fv["z"]
xv0, xv1, yv0, yv1 = min(x0, fvp.min() - .3), max(x1, fvp.max() + .3), min(y0, fv["ae"].min() - .3), max(y1, fv["ae"].max() + .3); xxv, yyv = np.meshgrid(np.linspace(xv0, xv1, 300), np.linspace(yv0, yv1, 300))
ax.contourf(xxv, yyv, mm._comb_score(xxv.ravel(), np.zeros(xxv.size), yyv.ravel()).reshape(xxv.shape), levels=[-1e9, mm.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.85, zorder=0); ax.axvline(t_et, color="#7a3db8", lw=2.6, label=f"ET 단독 경계 (ET ≥ {t_et:.2f})")
ax.scatter(ap[nor & keep], aez[nor & keep], color="#c8c8c8", s=24, marker="s", lw=0, label="train 정상 (OOF)"); ax.scatter(ap[~nor], aez[~nor], color="#f2b8a0", s=28, lw=0, label="train 고장 (OOF)")
ax.scatter(ap[isout], aez[isout], color="#9aa0a6", s=90, marker="D", label="이상치 정상 3개")
ax.scatter(fvp[~pv], fv["ae"][~pv], color=NOR, s=70, marker="^", edgecolor="white", lw=.6, label=f"val → 정상 판정 ({int((~pv).sum())})")
ax.scatter(fvp[pv], fv["ae"][pv], color=ABN, s=95, marker="^", edgecolor="black", lw=1, label=f"val → 고장 판정 ({int(pv.sum())})")
for i in range(len(pv)):
    if pv[i] or abs(fvp[i] - t_et) < .5: ax.annotate(ns.nv[i][-3:], (fvp[i], fv["ae"][i]), xytext=(5, 5), textcoords="offset points", fontsize=8, color=ABN if pv[i] else NOR)
tails(ax, a, ap, aez, "#bbb"); tails(ax, fv["a"], fvp, fv["ae"], "#7a3db8")
ax.set_xlim(xv0, xv1); ax.set_ylim(yv0, yv1); ax.set_xlabel("ET 점수 (logit) + 박자 z 환산  (점 위치 = 실제 판정 위치, 점선 꼬리 = ET 값 위치에서 옮겨 온 거리)"); ax.set_ylabel("AE z (val은 val 신호로 계산)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"val 40개 판정 — 고장 {int(pv.sum())}개 (ET 단독 경계로는 {int((fv['a'] > t_et).sum())}개)", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "val_판정", "점 위치 = ET + 박자 z 환산(점선 꼬리 = ET 값 위치). val 40개를 train OOF 위에 겹침. 배경 = 결합(로지스틱) 판단, 보라 세로선 = ET 단독 판단 경계")
V = pd.DataFrame(dict(파일=[n_[-3:] for n_ in ns.nv], 판정=np.where(pv, "고장", "정상"), ET=fv["a"].round(2), ET단독=np.where(fv["a"] > t_et, "고장", "정상"), AE_z=fv["ae"].round(2), 박자z=fv["z"].round(1), 켠부품=catv))
ex.save_md("val_판정", f"# EXP-046 val 판정 (학습 = 이상치 3개 뺀 147개, 제출 파일 아님)\n\n고장 판정 {int(pv.sum())}개: {' '.join(V.파일[pv])}\n\n![val](figs/val_판정.png)\n\n{md_table(V)}\n")

# %% t-SNE 기본 (4-10 방식, 핵심 그림): H 열 + 박자 S + AE 5열, map_scale, 146개 (이상치 3개 · 022는 지도에서 제외)
# 11-22 사용자 결정으로 예전 그림(robust_z 눈금)을 교체. 함수 src.exp2util.tsne_default
fig, ti = tsne_default(ns, mark=np.flatnonzero(nor & caught), mark_label="오경보 (OOF seed 0)", title="EXP-046 이상치 학습 제외")
ex.save_fig(fig, "tsne_라벨", f"기본 t-SNE(4-10): 왼쪽 라벨 없음, 오른쪽 라벨 색. 빨간 원 = 오경보. 폴드 밖 5-NN AUC {ti['auc']:.3f}", key=True)
print("done")
