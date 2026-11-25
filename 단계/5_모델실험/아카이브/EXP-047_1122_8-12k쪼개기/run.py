# %% [markdown]
# EXP-047 (11-22) J(EXP-044)에서 ET 입력의 8–12.5 kHz 옥타브 두 열(H10 울림 비율 · H15 파워 몫)을 8–10 · 10–12.5 kHz 4열로 쪼갬 — 방법 ②
# - 가설 H2′: 고장은 8–10 kHz 금속 모드(8.3 · 9.5 kHz)가 들뜬다. 4-10 진단에서 섞인 고장 009 · 018 · 013은 옆 정상보다 8–9 kHz +0.77 dec, 11–12.5 kHz −1.02 dec → 한 칸이면 상쇄
# - 경계 10 kHz는 실행 전 고정(고르지 않음). 정의 src.features2.hi_split_table. 나머지(박자 z · AE z · 로지스틱 · E2 · 8.3k)는 J와 같음
# - 대표 구성 S = 이상치 포함 학습(J와 같은 조건). CLAUDE.md 기본(이상치 3개 학습 제외)과 다른 이유: 비교 기준 J가 포함 학습이고, EXP-046에서 제외가 0.961로 더 낮았음. 제외판(S-x)도 함께 잰다
# 실행: conda run -n qe-phm python "단계/5_모델실험/아카이브/EXP-047_1122_8-12k쪼개기/run.py"   (코드 · 출력은 이 파일 하나)

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
from src.features2 import PeakRef, cached_hi_split, HI_SPLIT
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from sklearn.metrics import fbeta_score, recall_score, precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
B5 = os.path.join(_d, "단계", "5_모델실험")
ex = Experiment(__file__, "EXP-047", "8–12.5 kHz 옥타브를 8–10 · 10–12.5 kHz로 쪼갬", ["H2", "H3", "H1"],
                member="이동훈", model="ET(H10 · H15 → 8–10 · 10–12.5k 4열) + 박자 z + AE z → 로지스틱 OR E2 OR 8.3k",
                changes="J(EXP-044)의 ET 입력에서 H10 · H15를 8–10 · 10–12.5 kHz 울림 비율 · 파워 몫 4열로 교체",
                key_changes=["ET 입력 H10(울림 8–12.5k) · H15(몫 8–12.5k) → H10a/H10b(울림 8–10k / 10–12.5k) · H15a/H15b(몫 8–10k / 10–12.5k)", "경계 10 kHz는 실행 전 고정", "나머지는 J와 같음 (이상치 포함 학습)"])
ns = NS(); names, y, g = ns.names, ns.y, ns.g; nor = y == 0
OUT = outlier_idx(names); isout = np.zeros(ns.n, bool); isout[OUT] = True; keep = ~isout
for src_, k, nm in (("EXP-044_1122_AE추가", "중첩: J | W + AE + E2 + 8.3k (최종 후보)", "J | EXP-044 값 (기준, 이상치 포함 학습)"),
                    ("EXP-046_1122_이상치정상_학습제외", "중첩: K | J, 이상치 3개 학습 제외 (150개 평가)", "J-x | EXP-046 값 (이상치 3개 학습 제외)")):
    ex.res["configs"]["중첩: " + nm] = dict(json.load(open(os.path.join(find_exp(src_), "result.json"), encoding="utf-8"))["configs"][k], mid=True)
ex._write()

SP, SPv = cached_hi_split("train", ns.X), cached_hi_split("val", ns.Xv)
DROPC = ["H10_ring_8-12.5k", "H15_share_8-12.5k"]
FX = [c for c in ns.FIXED if c not in DROPC] + HI_SPLIT
Hs = pd.concat([ns.T, SP], axis=1); Hsv = pd.concat([ns.Tv, SPv], axis=1)
M = np.hstack([Hs[FX].values, ns.T[["L1_S"]].values, ns.L]); Mv = np.hstack([Hsv[FX].values, ns.Tv[["L1_S"]].values, ns.Lv])
spec, res = rule83_flags(ns.X); specv, resv = rule83_flags(ns.Xv); r83 = (spec | res); r83v = (specv | resv)
Fn = np.column_stack([M, np.arange(ns.n, dtype=float)])
mkS = lambda: TwoAE("lin", ("E2",), fixed=tuple(FX), nf=len(FX), df=ns.df, peak=tuple(ns.PEAK))
mkSx = lambda: ExcludeTrain(mkS, OUT)


# %% 중첩 CV (바깥 5겹 × seed 5개, 폴드마다 한 번 학습. 판정 = 결합 OR E2 OR 8.3k)
def task(mk, seed, tr, te):
    f = mk().fit(Fn[tr], y[tr]).fires(Fn[te])
    return seed, te, f["main"] | f["E2"] | r83[te], {k: f[k] for k in ("a", "z", "ae", "main", "E2")}


def as_nested(P, yy):
    return dict(f2=np.array([fbeta_score(yy, p, beta=2) for p in P]), recall=np.array([recall_score(yy, p) for p in P]), precision=np.array([precision_score(yy, p, zero_division=0) for p in P]),
                fpr=np.array([p[yy == 0].mean() for p in P]), miss_counts=((yy == 1) & ~P).sum(0), fp_counts=((yy == 0) & P).sum(0))


jobs = [(s, tr, te) for s in range(5) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
RUN = {}
for nm, mk in (("S", mkS), ("S-x", mkSx)):
    with parallel_config(backend="loky", inner_max_num_threads=1):
        R = Parallel(n_jobs=-1)(delayed(task)(mk, *j) for j in jobs)
    PRED = np.zeros((5, ns.n), bool); fo = {k: np.zeros(ns.n) for k in ("a", "z", "ae")}; fo.update(main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool))
    for s, te, p, f in R:
        PRED[s, te] = p
        if s == 0:
            for k in fo: fo[k][te] = f[k]
    RUN[nm] = (PRED, fo)
Sres = ex.record("중첩: S | J에서 H10 · H15 → 8–10 · 10–12.5k 4열 (이상치 포함 학습)", as_nested(RUN["S"][0], y), names, main=True)
ex.record("중첩: S-x | S, 이상치 3개 학습 제외 (150개 평가)", as_nested(RUN["S-x"][0], y), names, mid=True)

# 폴드 안 안정성: 학습 폴드마다 새 4열(과 원래 2열)의 단변량 AUC — 방향이 바뀌지 않는지
stab = []
for c in HI_SPLIT + DROPC:
    v = Hs[c].values; au = [roc_auc_score(y[tr], v[tr]) for _, tr, _ in jobs]
    stab.append(dict(열=c, AUC_전체=round(roc_auc_score(y, v), 3), 폴드_최소=round(min(au), 3), 폴드_최대=round(max(au), 3),
                     val중앙_정상백분위=round(float(np.mean(v[nor] <= np.median(Hsv[c].values)) * 100)), val_정상95위초과=f"{np.mean(Hsv[c].values > np.percentile(v[nor], 95)) * 100:.0f} %",
                     고장_정상95위초과=f"{np.mean(v[~nor] > np.percentile(v[nor], 95)) * 100:.0f} %"))
mfull = mkS().fit(Fn, y); imp = dict(zip(list(FX) + list(ns.PEAK), mfull.m_.feature_importances_))
for r in stab: r["ET중요도"] = round(float(imp.get(r["열"], np.nan)), 3) if r["열"] in imp else "—"
ex.save_md("피처점검", "# EXP-047 새 4열 점검 (단변량 AUC: 학습 폴드 25개의 최소 · 최대, val 이동, ET 중요도(전체 학습))\n\n원래 2열(H10 · H15)은 비교용으로 함께 적는다(S에서는 빠짐).\n\n" + md_table(pd.DataFrame(stab)) + "\n")

# %% 핵심 그림: ET × AE 평면 + ET 단독 경계 + 결합 경계 + 최종 판정 (이상치 3개 ◆)
PRED, fo = RUN["S"]; a, aez, zz = fo["a"], fo["ae"], fo["z"]; caught = fo["main"] | fo["E2"] | r83
mm = mfull
t_et = float(best_f2_threshold(mm.y_, mm.Z_[:, 0])); et_on = a > t_et; et_miss, et_fp = int((~nor & ~et_on).sum()), int((nor & et_on).sum())
cat = np.where(r83, "8.3k", np.where(fo["E2"], "E2", np.where(fo["main"], "결합", "—")))
c_ = mm.c_.coef_[0]; SH = c_[1] / c_[0]; ap = a + SH * zz              # 점 위치 = 실제 판정 위치 (ET + 박자 z 환산). ET 단독 경계(보라선)는 원래 ET 값 기준
def tails(ax, raw, eff, yv, color, lo=.3):                           # 박자 z 때문에 옮겨 온 점의 원래(박자 z 0) 위치와 이동선
    m = np.abs(eff - raw) >= lo
    ax.hlines(yv[m], raw[m], eff[m], color=color, lw=1, ls=":", alpha=.7, zorder=1); ax.scatter(raw[m], yv[m], s=9, color=color, zorder=1)
x0, x1, y0, y1 = ap.min() - .3, ap.max() + .3, aez.min() - .3, aez.max() + .3
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = mm._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(11, 8.2))
ax.contourf(xx, yy, D, levels=[-1e9, mm.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.85, zorder=0); ax.axvline(t_et, color="#7a3db8", lw=2.6, label=f"ET 단독 경계 (ET ≥ {t_et:.2f}): 놓침 {et_miss} · 오경보 {et_fp}")
ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#dbe8f7", label="배경 = 결합(로지스틱) 판단: 정상")); ax.add_patch(plt.Rectangle((0, 0), 0, 0, color="#fbe3d6", label="배경 = 결합(로지스틱) 판단: 고장 (점 위치에 박자 z 반영)"))
n1 = nor & keep
ax.scatter(ap[n1 & ~caught], aez[n1 & ~caught], color=NOR, s=34, marker="s", lw=0, alpha=.75, label="정상 → 정상 판정")
ax.scatter(ap[n1 & caught], aez[n1 & caught], color=NOR, s=100, marker="s", edgecolor="#d62728", lw=2, label="정상 → 오경보 (최종)")
for k, c in (("결합", "#e8590c"), ("E2", "#7a3db8"), ("8.3k", "#000000")):
    m = (cat == k) & ~nor; ax.scatter(ap[m], aez[m], s=90 if k == "결합" else 150, color=ABN, edgecolor=c, lw=.5 if k == "결합" else 2.4, label=f"고장 → {k}이 잡음 ({int(m.sum())})")
ax.scatter(ap[isout], aez[isout], s=190, marker="D", color="#9aa0a6", edgecolor=["#d62728" if c else "#444" for c in caught[isout]], lw=2.2, label="이상치 정상 059 · 065 · 116 (빨간 테두리 = 오경보)")
for i in OUT: ax.annotate(short(names[i]), (ap[i], aez[i]), xytext=(8, 8), textcoords="offset points", fontsize=9, color="#444", weight="bold")
for f_ in ("abn_002", "abn_017", "abn_022", "abn_009", "abn_018", "abn_013"):
    i = names.index(f_); ax.annotate(short(f_), (ap[i], aez[i]), xytext=(8, 8), textcoords="offset points", fontsize=9, color=ABN, weight="bold")
tails(ax, a, ap, aez, "#777")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xlabel("ET 점수 (logit, OOF) + 박자 z 환산  (점 위치 = 실제 판정 위치, 점선 꼬리 = ET 값 위치에서 옮겨 온 거리)"); ax.set_ylabel("AE z (OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
ax.set_title(f"S — 중첩 F2 {Sres['f2']:.3f} ± {Sres['f2_sd']:.3f} · 재현율 {Sres['recall']:.3f} · 오경보율 {Sres['fpr']:.3f} (OOF seed 0: 고장 {int(caught[~nor].sum())}/30 · 오경보 {int(caught[nor].sum())}/120)", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "점 위치 = ET + 박자 z 환산(점선 꼬리 = ET 값 위치). ET × AE 평면 (OOF seed 0). 배경 = 결합(로지스틱) 판단, 보라 세로선 = ET 단독 판단 경계, ◆ = 이상치 정상 3개", key=True)

# %% 오류 분석 표 (md, 가로)
cnt = np.array([(5 - Sres["miss_counts"].get(n_, 0)) if y[i] else Sres["fp_counts"].get(n_, 0) for i, n_ in enumerate(names)])
files = ["abn_002", "abn_009", "abn_013", "abn_017", "abn_018", "abn_022"] + [n_ for n_ in names if n_.startswith("abn") and Sres["miss_counts"].get(n_, 0) >= 1 and n_ not in ("abn_002", "abn_009", "abn_013", "abn_017", "abn_018", "abn_022")] \
    + [n_ for n_ in names if n_.startswith("nor") and Sres["fp_counts"].get(n_, 0) >= 1]
ex_ = lambda arr, fmt: {n_: fmt(arr[names.index(n_)]) for n_ in files}
tab = error_md_table(names, y, cnt, files, extra={"ET 점수": ex_(a, lambda v: f"{v:.2f}"), "ET 단독 판정": ex_(et_on, lambda v: "고장" if v else "정상"), "박자 z": ex_(zz, lambda v: f"{v:.1f}"),
                                                    "AE z": ex_(aez, lambda v: f"{v:.2f}"), "켠 부품": ex_(cat, str), "몫 8–10k": ex_(Hs["H15a_share_8-10k"].values, lambda v: f"{v:.2f}"), "몫 10–12.5k": ex_(Hs["H15b_share_10-12.5k"].values, lambda v: f"{v:.2f}")})
ex.save_md("오류분석_표", f"# EXP-047 오류 분석 (중첩 CV seed 5개 판정 횟수 / OOF seed 0 근거)\n\n우리 판단 = 정답 라벨, 모델 판단 = 최종 판정(결합 OR E2 OR 8.3k). 009 · 018 · 013 = 이 실험이 겨냥한 섞인 고장.\n\n{tab}\n")

# %% val 판정 (train 150 전체 학습, 제출 파일 아님)
ae_tr = _fit_ae(LP_TR[y == 0]); _, LPv = segment_logpsd(ns.Xv); mm.ae_override_ = _ae_score(ae_tr, LPv)
fv = mm.fires(np.column_stack([Mv, np.zeros(len(Mv))])); pv = fv["main"] | fv["E2"] | r83v
catv = np.where(r83v, "8.3k", np.where(fv["E2"], "E2", np.where(fv["main"], "결합", "—")))
j44 = {r.split("|")[1].strip(): r.split("|")[2].strip() for r in open(os.path.join(find_exp("EXP-044"), "val_판정.md"), encoding="utf-8").read().splitlines() if r.startswith("| 0")}
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
V = pd.DataFrame(dict(파일=[n_[-3:] for n_ in ns.nv], 판정=np.where(pv, "고장", "정상"), J판정=[j44.get(n_[-3:], "?") for n_ in ns.nv], ET=fv["a"].round(2), ET단독=np.where(fv["a"] > t_et, "고장", "정상"),
                      AE_z=fv["ae"].round(2), 박자z=fv["z"].round(1), 켠부품=catv))
diff = V[V.판정 != V.J판정].파일.tolist()
ex.save_md("val_판정", f"# EXP-047 val 판정 (train 150 전체 학습, 제출 파일 아님)\n\n고장 판정 {int(pv.sum())}개: {' '.join(V.파일[pv])} · J(EXP-044)와 다른 파일: {' '.join(diff) if diff else '없음'}\n\n![val](figs/val_판정.png)\n\n{md_table(V)}\n")

# %% t-SNE 기본 (4-10 방식, 핵심 그림): 이 실험의 ET 입력 열(쪼갠 4열 포함) + 봉우리 + 박자 S + AE 5열, map_scale, 146개. 함수 src.exp2util.tsne_default
fig, ti = tsne_default(ns, T=Hs, cols=FX + list(ns.PEAK) + ["L1_S"], mark=np.flatnonzero(nor & caught), mark_label="S 오경보 (OOF seed 0)", title="EXP-047 S (8–12.5k 쪼갬)")
ex.save_fig(fig, "tsne_라벨", f"기본 t-SNE(4-10): 왼쪽 라벨 없음, 오른쪽 라벨 색. 빨간 원 = S 오경보. 폴드 밖 5-NN AUC {ti['auc']:.3f}", key=True)

# %% 기본 t-SNE + val (11-22 사용자 요청): 기본 공간(이 실험의 열)에 val 40개. 왼쪽 = 가까운 train 5개 위치, 오른쪽 = 함께 다시 펼침. 함수 src.exp2util.tsne_val_default
fig, tv = tsne_val_default(ns, pv, T=Hs, Tv=Hsv, cols=FX + list(ns.PEAK) + ["L1_S"], title="EXP-047 S 판정")
ex.save_fig(fig, "tsne_val포함", "기본 t-SNE + val 40개 (보라 ▲ = S 판정, 진한 = 고장). 빨간 원 = train에서 먼 val (가장 가까운 train 거리가 train끼리 거리의 95 % 위)")
V["이웃고장(5)"], V["최근접거리_백분위"], V["최대z열"], V["최대z"] = tv["이웃고장"].values, tv["거리백분위"].values, tv["최대z열"].values, tv["최대z"].values
ex.save_md("val_판정", f"# EXP-047 val 판정 (train 150 전체 학습, 제출 파일 아님)\n\n고장 판정 {int(pv.sum())}개: {' '.join(V.파일[pv])} · J(EXP-044)와 다른 파일: {' '.join(diff) if diff else '없음'}\n\n![val](figs/val_판정.png)\n\n"
           f"이웃고장(5) = 기본 t-SNE 공간(t-SNE 전)에서 가까운 train 5개 중 고장 수. 최근접거리_백분위 = 가장 가까운 train까지 거리가 train끼리 최근접 거리 분포의 몇 %인가(95 넘으면 train에서 먼 값). 최대z = 정상 기준 눈금에서 가장 벗어난 열\n\n{md_table(V)}\n")
print("done")
