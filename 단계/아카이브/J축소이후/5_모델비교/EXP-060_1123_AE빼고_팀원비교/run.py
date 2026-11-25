# %% [markdown]
# EXP-060 J-축소에서 AE를 빼고(ET + 박자 → 로지스틱 OR E2 OR 8.3k) val을 팀원 판정과 비교 (4-12 비교 · EXP-059 후속)
# - 질문: val AE 점수가 통째로 올라가 있다(val AE z 중앙값 1.75, 40개 중 38개 > 1). AE를 빼면 우리 val 판정이 팀원 쪽으로 가나, 학습 성능은 얼마나 잃나
# - 구성 (같은 바깥 분할, quick 기본, 이상치 학습 제외 없음 = 제출 J-축소와 같은 조건)
#   J-축소       제출 구성 (8.3k 규칙 K83 > 3 포함)
#   J-축소 − AE  같은 구성에서 결합 입력의 AE z만 뺌 (TwoStack use_ae=False). 박자 z · E2 · 8.3k 규칙은 그대로
# - 팀원 val(갱신된 8.3k 규칙 반영, EXP-059와 같음): 효섭 13 · 한동훈 14(= 효섭 + 006) · 소린 16
# 실행: conda run -n qe-phm python run.py  (기록용 QE_FULL=1)

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import best_f2_threshold
from src.exp2util import NS, Add83, rule83_flags, NOR, ABN, tsne_default, j_mats, RING_COLS
from src.models2 import TwoStack, RingShrink
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
HS13 = {4, 9, 12, 16, 27, 29, 31, 32, 34, 35, 36, 38, 39}                 # 효섭 제출 14에서 001(K83로 정상) 뺌
DH14 = HS13 | {6}                                                          # 한동훈 V14 = 효섭 + 006 (K83 반영)
SR16 = HS13 | {6, 8, 26}                                                   # 소린 M1 (K83 반영)

ex = Experiment(__file__, "EXP-060", "AE 빼고 팀원과 val 비교", ["H1", "H2"],
                member="이동훈", model="J-축소 · J-축소 − AE (ET + 박자 → 로지스틱 OR E2 OR 8.3k)",
                changes="J-축소의 결합 입력에서 AE z만 뺌. 같은 분할, val을 팀원(효섭 · 한동훈 · 소린, K83 반영) 판정과 비교",
                key_changes=["결합 입력 [ET, 박자, AE] → [ET, 박자] (AE 제외)", "E2 · 8.3k(K83) 규칙 · 울림 비율 축소 그대로", "이상치 학습 제외 없음(제출 J-축소와 같은 조건)"])
ns = NS(); y, g, names = ns.y, ns.g, np.array(ns.names); nor = y == 0; nv = [f"val_{n[-3:]}" for n in ns.nv]
short = lambda n: n.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
vnum = np.array([int(n[-3:]) for n in nv])
spec, res = rule83_flags(ns.X, k_gate=True); sv, rv = rule83_flags(ns.Xv, k_gate=True); r83, r83v = spec | res, sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
F0 = list(ns.FIXED); RI = [F0.index(c) for c in RING_COLS if c in F0]; I1 = F0.index("H1_nimp")
def build(use_ae):
    base = lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(F0), nf=len(F0), df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=use_ae))
    return lambda: RingShrink(base, RI, I1, 3.0)
CFG = [("J-축소", True), ("J-축소 − AE", False)]; CN = [c[0] for c in CFG]

# %% 중첩 CV (같은 분할)
RES = {}
for nm, ae in CFG:
    M, _ = j_mats(ns, r83, F0); RES[nm] = nested_cv(M, y, g, build(ae), n_jobs=-1); ex.record(nm, RES[nm], names, main=(nm == "J-축소 − AE"), mid=(nm == "J-축소"))

# %% 전체 학습 모델 → val
FIT, VAL = {}, {}
M, Mv = j_mats(ns, r83, F0)
for nm, ae in CFG:
    top = build(ae)().fit(M, y); ts, tf = top.m_.m_, top._tf
    ft = top.fires(M); fv = ts.fires_raw(tf(Mv), RAW); comb = ts.c_.decision_function(ts.Z_) >= ts.tc_
    FIT[nm] = dict(ts=ts, comb=comb, fin=comb | ft["E2"] | r83, fv=fv, E2v=fv["E2"]); VAL[nm] = fv["main"] | fv["E2"] | r83v
ex.note("결합 계수 · 문턱", {nm: dict(계수=np.round(FIT[nm]["ts"].c_.coef_[0], 3).tolist(), 문턱=round(float(FIT[nm]["ts"].tc_), 3)) for nm in CN})
ex.note("val AE z (학습 정상 기준)", dict(중앙값=round(float(np.median(FIT["J-축소"]["fv"]["ae"])), 2), 개수_1초과=int((FIT["J-축소"]["fv"]["ae"] > 1).sum())))

# %% 오류 분석 표 (학습, 가로)
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm in CN}; NSD = len(RES[CN[0]]["f2"])
files = [names[i] for i in range(ns.n) if any((C[nm][i] > 0) if y[i] == 0 else (C[nm][i] < NSD) for nm in CN)]
ex.save_md("오류분석_표", f"모델 판단 행 = J-축소 − AE, 아래 행 = J-축소(AE 포함)의 고장 판정 횟수 / seed {NSD}\n\n"
           + error_md_table(names, y, C["J-축소 − AE"], files, n_seeds=NSD, extra={"J-축소 (고장 판정)": {f: int(C["J-축소"][list(names).index(f)]) for f in files}}))

# %% val 판정 표 (같은 판정끼리 묶음) + 일치 수
COLS = {"J-축소 (AE 포함, 제출)": VAL["J-축소"], "J-축소 − AE": VAL["J-축소 − AE"],
        "효섭": np.isin(vnum, list(HS13)), "한동훈": np.isin(vnum, list(DH14)), "소린": np.isin(vnum, list(SR16))}
lab = list(COLS); Mx = np.array([COLS[k] for k in lab]).astype(int)
agree = {f"{a}–{b}": int((COLS[a] == COLS[b]).sum()) for i, a in enumerate(lab) for b in lab[i + 1:]}
def group_name(p): return "모두 고장" if p.all() else "모두 정상" if not p.any() else "고장: " + " · ".join(l for l, b in zip(lab, p) if b)
order = np.array(sorted(range(40), key=lambda i: (-Mx[:, i].sum(), tuple(-Mx[:, i]), vnum[i])))
V = pd.DataFrame({"val": [f"{v:03d}" for v in vnum], "묶음 (고장이라고 본 모델)": [group_name(Mx[:, i]) for i in range(40)],
                  **{k: np.where(v, "고장", "정상") for k, v in COLS.items()},
                  **{f"{nm} 결합−문턱": np.round(FIT[nm]["fv"]["score"] - FIT[nm]["ts"].tc_, 2) for nm in CN},
                  "AE z": np.round(FIT["J-축소"]["fv"]["ae"], 2), "8.3k": np.where(r83v, "켜짐", ""), "E2": np.where(FIT["J-축소"]["E2v"], "켜짐", "")}).iloc[order]
split = V[V[lab].nunique(axis=1) > 1]
ex.note("val 일치 수 (40개 중)", agree); ex.note("val 고장 수", {k: int(v.sum()) for k, v in COLS.items()})
ex.save_md("val_판정", "val 고장 수: " + " · ".join(f"{k} {int(v.sum())}" for k, v in COLS.items()) + "\n\n일치 수(40개 중): " + " · ".join(f"{k} {v}" for k, v in agree.items())
           + "\n\n팀원 판정은 갱신된 8.3k 규칙(K83 > 3) 반영. 표는 같은 판정끼리 묶음(고장이라고 본 모델이 많은 묶음부터)\n\n"
           + f"판정이 갈린 val\n\n{md_table(split)}\n\n전체\n\n{md_table(V)}")
print(agree, flush=True)

# %% 핵심 그림 1: 판정이 갈린 val (행 = 파일, 열 = 모델, 같은 판정 묶음끼리)
FULL = ["J-축소\n(AE 포함,\n제출)", "J-축소\nAE 제외", "효섭\n(RF 배깅)", "한동훈\n(ET)", "소린\n(RF + ET)"]
sp = [o for o in order if 0 < Mx[:, o].sum() < Mx.shape[0]]
allf = [vnum[o] for o in order if Mx[:, o].all()]; nonef = [o for o in order if not Mx[:, o].any()]
fig, ax = plt.subplots(figsize=(13, .52 * len(sp) + 2.6))
for r, o in enumerate(sp):
    for c in range(Mx.shape[0]):
        f = Mx[c, o] == 1
        ax.add_patch(plt.Rectangle((c, r), .94, .86, color=ABN if f else "#eef1f5", lw=0))
        ax.text(c + .47, r + .43, "고장" if f else "정상", ha="center", va="center", fontsize=10, color="white" if f else "#9aa3ae", weight="bold" if f else "normal")
    ax.text(-.12, r + .43, f"val_{vnum[o]:03d}", ha="right", va="center", fontsize=10, color=INK)
r0 = 0
for r in range(1, len(sp) + 1):
    if r == len(sp) or tuple(Mx[:, sp[r]]) != tuple(Mx[:, sp[r - 1]]):
        names_f = [lab[c].split(" (")[0].replace(" − AE", " AE 제외") for c in range(Mx.shape[0]) if Mx[c, sp[r0]]]
        ax.text(Mx.shape[0] + .15, (r0 + r - .14) / 2, "고장: " + " · ".join(names_f), ha="left", va="center", fontsize=10, color=INK)
        if r < len(sp): ax.axhline(r - .07, color=INK, lw=1.2)
        r0 = r
for c, nm in enumerate(FULL): ax.text(c + .47, -.25, nm, ha="center", va="bottom", fontsize=9.5, color=INK, weight="bold")
ax.axvline(1.97, color=INK, lw=.8, ls=":", ymin=0, ymax=1)
ax.set_xlim(-1.3, Mx.shape[0] + 4.6); ax.set_ylim(len(sp) + .1, -2.2); ax.axis("off")
ax.text(-1.3, -2.9, f"AE를 뺀 우리 모델 vs 팀원 — 판정이 갈린 val {len(sp)}개", fontsize=13, weight="bold", color=INK)
ax.text(-1.3, -2.35, f"나머지 {40 - len(sp)}개는 다섯 모델이 모두 같음: 모두 고장 {len(allf)}개 ({' · '.join(f'{v:03d}' for v in allf)}) · 모두 정상 {len(nonef)}개.  "
        "팀원 판정은 갱신된 8.3k 규칙(K83 > 3) 반영", fontsize=9, color=MUTED)
fig.tight_layout(); ex.save_fig(fig, "val_판정지도", "AE 포함 · 제외 우리 모델과 팀원 세 명의 val 판정 (갈린 파일만, 같은 판정끼리)", key=True)

# %% 핵심 그림 2: logistic_plane (왼쪽 J-축소 = ET·박자 × AE z, 오른쪽 J-축소 − AE = ET logit × 박자 z). 위 학습, 아래 val
def plane(ax, nm, val=False):
    F = FIT[nm]; ts = F["ts"]; c, b, tc = ts.c_.coef_[0], ts.c_.intercept_[0], ts.tc_; Z = ts.Z_; t_et = float(best_f2_threshold(y, Z[:, 0]))
    has_ae = len(c) == 3
    if val: fv = F["fv"]; et, zb, ae, fin = fv["a"], fv["z"], fv["ae"], VAL[nm]
    else: et, zb, ae, fin = Z[:, 0], Z[:, 1], (Z[:, 2] if has_ae else None), F["fin"]
    if has_ae: x, yy, ylab, xlab = et + c[1] / c[0] * zb, ae, "AE z", f"ET logit + ({c[1] / c[0]:.2f})·박자 z"
    else: x, yy, ylab, xlab = et, zb, "박자 z", "ET logit"
    x0, x1 = min(x.min(), t_et) - .6, max(x.max(), t_et) + .6; y0, y1 = yy.min() - .6, yy.max() + .6
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    dec = (c[0] * gx + c[2] * gy + b) if has_ae else (c[0] * gx + c[1] * gy + b)
    ax.contourf(gx, gy, (dec >= tc).astype(float), levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    col = np.where(fin, ABN, NOR)
    if val:
        ax.scatter(x, yy, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in range(len(x)):
            if VAL[nm][i] != COLS["효섭"][i]: ax.annotate(short(nv[i]), (x[i], yy[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
    else:
        show = ~((y == 1) & ~F["comb"] & fin)
        for msk, mkr, s_ in ((show & (y == 0), "s", 22), (show & (y == 1), "o", 40)):
            ax.scatter(x[msk], yy[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & (fin != (y == 1))): ax.annotate(short(names[i]), (x[i], yy[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2); ax.set_xlabel(xlab, color=MUTED); ax.set_ylabel(ylab, color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f}" + (f" · val 고장 {int(fin.sum())}/40 (굵은 이름 = 효섭과 다름)" if val else " · 학습 150"), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")
fig, axs = plt.subplots(2, 2, figsize=(18, 13))
for k, nm in enumerate(CN): plane(axs[0, k], nm); plane(axs[1, k], nm, val=True)
fig.suptitle("logistic_plane — 위 = 학습 150(안쪽 OOF 입력), 아래 = val 40. 배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 색 = 최종 판정(규칙 포함). 규칙만으로 잡힌 고장은 뺌", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .96)); ex.save_fig(fig, "logistic_plane", "AE 포함 · 제외 결합 경계 (학습 · val)", key=True)

# %% t-SNE 기본 (핵심 그림): AE를 빼서 판정이 달라진 학습 파일 표시
K = NSD // 2 + 1; wrong = {nm: (C[nm] >= K) != (y == 1) for nm in CN}
mk = np.flatnonzero(wrong["J-축소"] != wrong["J-축소 − AE"])
fig, ti = tsne_default(ns, mark=mk, mark_label="AE 포함 · 제외 중 한쪽만 자주 틀린 파일", title="EXP-060 (J 입력 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10). 빨간 원 = AE를 빼면 판정이 달라지는(한쪽만 자주 틀린) 학습 파일", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
