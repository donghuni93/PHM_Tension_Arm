# %% [markdown]
# EXP-058 모드 경계로 대역 교체 + 소린 비율 — 공유 대역 제외 (연결부 · 본체 · 고주파 3칸)
# - 가설: H1 · H2. 옥타브 경계(1 · 2 · 4 · 8k)를 라벨 없이 찾은 울림 모드 경계(1.0 · 1.6 · 4.5 · 5.5 · 8.0 · 12.5k, 2-14)로 바꾸면
#   칸 하나가 한 가지 의미(공유 · 연결부 · 전이 · 본체 · 고주파)를 가져 ET 근거가 깨끗해진다
# - ET 입력: H3 · H5 · H7–H10 · H12–H15 → 모드 칸 첨도 5 · 울림 비율 5(축소 α = 3 유지) · 에너지 몫 5 + 소린 비율 2 (src.features2.mode_band_table)
# - EXP-057과 다른 점 하나: 정상 · 고장이 같이 우는 공유(1.0–1.6k) · 전이(4.5–5.5k) 칸의 첨도 · 울림 비율 · 에너지 몫을 뺀다 (ET 22 → 16열)
# - 같은 분할 quick, 이상치 학습 제외 없음. 057도 같은 실행에서 다시 돌려 나란히 비교(mid)
# 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import best_f2_threshold
from src.exp2util import NS, Add83, rule83_flags, NOR, ABN, tsne_default, j_mats, ts_of, RING_COLS
from src.models2 import TwoStack, RingShrink
from src.features2 import mode_band_table, MODE_TAG
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
ex = Experiment(__file__, "EXP-058", "모드 경계로 대역 교체 + 소린 비율 — 공유 제외", ["H1", "H2"],
                member="이동훈", model="J-축소 (ET 입력 = 모드 경계 3칸(연결부 · 본체 · 고주파) 첨도 · 울림 비율 · 에너지 몫 + 소린 비율 2열) + 박자 z + AE z → 로지스틱 OR E2 OR 8.3k",
                changes="옥타브 경계 → 울림 모드 경계 3칸(공유 · 전이 제외, 2-14 라벨 없음), 소린 비율 2열 추가. ET 15 → 16열. quick",
                key_changes=["첨도 H3 · H5 → 모드 칸 3개 (연결부 · 본체 · 고주파)", "울림 비율 H7–H10 → 모드 칸 3개 (축소 유지)", "에너지 몫 H12–H15 → 모드 칸 3개", "소린 비율 2열: 연결부 ÷ 본체 · 고주파 ÷ 본체", "EXP-057(5칸)과 다른 점 = 공유 · 전이 칸 제외"])
ns = NS(); y, g, names = ns.y, ns.g, np.array(ns.names); nor = y == 0; nv = [f"val_{n[-3:]}" for n in ns.nv]
ns.T = pd.concat([ns.T.reset_index(drop=True), cached_table("modeband_train", mode_band_table, ns.X)], axis=1)
ns.Tv = pd.concat([ns.Tv.reset_index(drop=True), cached_table("modeband_val", mode_band_table, ns.Xv)], axis=1)
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83, r83v = spec | res, sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
OUT = [names.tolist().index(n) for n in OUTLIER_NORMALS]
short = lambda n: n.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
KEEP = ["H1_nimp", "H2_ratio21", "H18_hfl"]                       # 대역과 무관한 H 열 (H16 · H17 봉우리는 TwoStack이 따로)
def mode_cols(tags): return [f"MK_{t}" for t in tags] + [f"MR_{t}" for t in tags] + [f"MS_{t}" for t in tags] + ["SO_연결부÷본체", "SO_고주파÷본체"]
F0 = list(ns.FIXED)
T3 = [t for t in MODE_TAG if not t.startswith(("공유", "전이"))]
CFG = [("J-축소", F0, RING_COLS), ("057 공유 포함 (5칸)", KEEP + mode_cols(MODE_TAG), [f"MR_{t}" for t in MODE_TAG]),
       ("058 공유 제외 (3칸)", KEEP + mode_cols(T3), [f"MR_{t}" for t in T3])]
MAIN = CFG[2][0]


def build(fixed, ring):
    """J-축소와 같은 조립(RingShrink(Add83(TwoStack lin + E2 + AE))), 축소할 울림 열만 지정"""
    nf = len(fixed)
    base = lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(fixed), nf=nf, df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True))
    ri = [fixed.index(c) for c in ring if c in fixed]; i1 = fixed.index("H1_nimp")
    return lambda: RingShrink(base, ri, i1, 3.0)


from sklearn.metrics import roc_auc_score
ex.note("단독 분리도 |AUC−0.5|×2 (학습 150)", {c: round(abs(roc_auc_score(y, ns.T[c]) - .5) * 2, 3) for c in mode_cols(MODE_TAG)})

# %% 중첩 CV (quick, 같은 분할)
RES = {}
for nm, fx, rg in CFG:
    M, _ = j_mats(ns, r83, fx); RES[nm] = nested_cv(M, y, g, build(fx, rg), n_jobs=-1)
    ex.record(nm, RES[nm], names, main=(nm == MAIN), mid=(nm != MAIN))

# %% 전체 학습 모델 · val
FIT = {}
for nm, fx, rg in CFG:
    M, Mv = j_mats(ns, r83, fx); top = build(fx, rg)().fit(M, y); ts, tf = ts_of(top)
    ft = top.fires(M); fv = ts.fires_raw(tf(Mv), RAW); comb = ts.c_.decision_function(ts.Z_) >= ts.tc_
    FIT[nm] = dict(ts=ts, comb=comb, fin=comb | ft["E2"] | r83, fv=fv, vfin=fv["main"] | fv["E2"] | r83v)
ex.note("결합 계수 [ET, 박자, AE] · 문턱", {nm: dict(계수=np.round(FIT[nm]["ts"].c_.coef_[0], 3).tolist(), 문턱=round(float(FIT[nm]["ts"].tc_), 3)) for nm, _, _ in CFG})

# %% 오류 분석 표 (가로): 어느 구성에서든 한 번이라도 틀린 파일
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm, _, _ in CFG}; NSD = len(RES[MAIN]["f2"])
files = [names[i] for i in range(ns.n) if any((C[nm][i] < NSD) if y[i] == 1 else (C[nm][i] > 0) for nm in C)]
extra = {f"{nm} ({NSD}번 중 고장)": {f: int(C[nm][names.tolist().index(f)]) for f in files} for nm, _, _ in CFG if nm != MAIN}
ex.save_md("오류분석_표", f"모델 판단 행 = {MAIN}. 아래 행 = 같은 분할에서 다른 구성의 고장 판정 횟수\n\n" + error_md_table(names, y, C[MAIN], files, n_seeds=NSD, extra=extra))

# %% val 판정
V = pd.DataFrame({"파일": [n[-3:] for n in nv], **{nm: np.where(FIT[nm]["vfin"], "고장", "정상") for nm, _, _ in CFG},
                  **{f"{nm} 결합−문턱": np.round(FIT[nm]["fv"]["score"] - FIT[nm]["ts"].tc_, 2) for nm, _, _ in CFG}})
d = V[V[[nm for nm, _, _ in CFG]].nunique(axis=1) > 1]
ex.save_md("val_판정", "val 고장 판정 수: " + " · ".join(f"{nm} {int(FIT[nm]['vfin'].sum())}/40" for nm, _, _ in CFG) + f"\n\n구성 사이에 판정이 다른 파일\n\n{md_table(d)}\n\n전체\n\n{md_table(V)}")


# %% 핵심 그림: logistic_plane (배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 = 최종 판정)
def plane(ax, nm, val=False):
    F = FIT[nm]; ts = F["ts"]; c, b, tc = ts.c_.coef_[0], ts.c_.intercept_[0], ts.tc_; Z = ts.Z_; t_et = float(best_f2_threshold(y, Z[:, 0]))
    if val: fv = F["fv"]; et, zb, y3, fin = fv["a"], fv["z"], fv["ae"], F["vfin"]
    else: et, zb, y3, fin = Z[:, 0], Z[:, 1], Z[:, 2], F["fin"]
    x = et + c[1] / c[0] * zb
    x0, x1 = min(x.min(), t_et) - .6, max(x.max(), t_et) + .6; y0, y1 = y3.min() - .6, y3.max() + .6
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    ax.contourf(gx, gy, (c[0] * gx + c[2] * gy + b >= tc).astype(float), levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    show = np.ones(len(x), bool)
    if not val: show &= ~((y == 1) & ~F["comb"] & fin)
    for i in np.flatnonzero(show & (np.abs(x - et) >= .3)): ax.plot([et[i], x[i]], [y3[i], y3[i]], ls=":", color="#888", lw=.8, zorder=1)
    col = np.where(fin, ABN, NOR); base = FIT["J-축소"]
    if val:
        ax.scatter(x, y3, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in np.flatnonzero(fin != base["vfin"]): ax.annotate(short(nv[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=8.5, weight="bold")
    else:
        o = np.zeros(len(x), bool); o[OUT] = True
        for msk, mkr, s_ in ((show & (y == 0) & ~o, "s", 22), (show & (y == 1), "o", 40), (show & o, "D", 60)):
            ax.scatter(x[msk], y3[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & ((fin != (y == 1)) | o)): ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f} · 계수 ET {c[0]:.2f} · AE {c[2]:.2f}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116")]
for val, nmf, cap in ((False, "logistic_plane", "기준 J-축소와 새 구성의 결합 경계 (학습 150)"), (True, "val_판정", "같은 경계 위의 val 40. 굵은 이름 = 기준과 판정이 다른 파일")):
    fig, axs = plt.subplots(1, len(CFG), figsize=(8 * len(CFG), 7.5))
    for ax, (nm, _, _) in zip(np.atleast_1d(axs), CFG): plane(ax, nm, val)
    fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
    fig.suptitle(("val 40 — " if val else "학습 150 (안쪽 OOF 입력 · 전체 학습 결합) — ") + "배경 = 결합 판단, 세로 점선 = ET 단독 판단" + ("" if val else ". 규칙만으로 잡힌 고장은 뺌"), x=.01, ha="left", fontsize=10, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, nmf, cap, key=not val)

# %% t-SNE 기본 (핵심 그림, 새 구성의 입력 공간)
fpM = np.flatnonzero((y == 0) & (C[MAIN] >= NSD // 2 + 1))
cols = list(CFG[-1][1]) + list(ns.PEAK) + ["L1_S"]
fig, ti = tsne_default(ns, T=ns.T, cols=cols, mark=fpM, mark_label=f"오경보 ({NSD}번 중 과반)", title=f"{ex.id} 입력 공간")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10), 새 구성 입력. 빨간 원 = 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
