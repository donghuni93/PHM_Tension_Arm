# %% [markdown]
# EXP-051 J-축소 + 배깅: J-축소(EXP-050)를 층화 80 % 부분표본 10개로 학습해 평균
# - 가설(H1 · H2 그대로, 모델 쪽): J-축소의 남은 흔들림은 결합기(로지스틱 w · 문턱)와 ET가 학습 표본에 따라 움직이는 데서 나온다
#   → 부분표본 여러 개로 학습해 평균하면(배깅) 경계 파일 판정이 안정되고, 점수가 오를 수도 있다
# - 배깅 방식(실행 전 고정, 팀원 신효섭 RF 배깅과 같은 틀): StratifiedShuffleSplit(10, train_size=0.8, random_state=0), bag b의 ET seed = b
#   판정 = 평균 여유(각 bag의 결합 점수 − 그 bag의 문턱) ≥ 0  OR  E2 과반(bag 10개 중 5개 이상)  OR  8.3k 규칙
# - 기준 = J-축소(EXP-050, α = 3), 같은 바깥 분할. 이상치 학습 제외는 쓰지 않는다(EXP-050과 같은 조건)
# - 잣대: ① 중첩 F2(주) ② 흔들림: train = seed마다 판정이 갈리는 파일 수, val = 바깥 5폴드 모델 + 전체 모델(6개)의 val 최종 판정이 갈리는 파일 수
#   ③ val 판정 수(보조) ④ (배깅만) bag 하나하나의 val 고장 수 범위 = 평균이 무엇을 눌렀는지
# 실행: conda run -n qe-phm python run.py (quick 선별) · QE_FULL=1 (정식)

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import FULL, best_f2_threshold
from src.exp2util import NS, Add83, rule83_flags, NOR, ABN, tsne_default
from src.models2 import TwoStack
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit
from joblib import Parallel, delayed
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
ALPHA, NB, FRAC = 3.0, 10, 0.8                           # 실행 전 고정
RING = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]

ex = Experiment(__file__, "EXP-051", "J-축소 + 배깅 (부분표본 80 % × 10)", ["H1", "H2"],
                member="이동훈", model="J-축소(ET + 박자 z + AE z → 로지스틱, 울림 비율 축소 α=3) × 배깅 10 (층화 80 %), 평균 여유 OR E2 과반 OR 8.3k",
                changes="EXP-050 J-축소를 층화 80 % 부분표본 10개로 학습해 평균 여유로 판정(신효섭 RF 배깅과 같은 틀). 같은 분할",
                key_changes=["배깅 10 (StratifiedShuffleSplit 80 %, bag b의 ET seed = b)", "판정 = 평균 여유 ≥ 0 OR E2 과반 OR 8.3k", "기준 = J-축소(EXP-050)"])
ns = NS(); y, g, names = ns.y, ns.g, np.array(ns.names); nor = y == 0; nv = [f"val_{n[-3:]}" for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83, r83v = spec | res, sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
OUT = [names.tolist().index(n) for n in OUTLIER_NORMALS]
short = lambda n: n.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
F0 = list(ns.FIXED)
M = np.column_stack([ns.T[F0].values, ns.T[["L1_S"]].values, ns.L, np.arange(ns.n, dtype=float), r83.astype(float)])
Mv = np.column_stack([ns.Tv[F0].values, ns.Tv[["L1_S"]].values, ns.Lv])

class Shrink:
    """EXP-050과 같다: ET 입력의 울림 비율 열을 충격 수로 축소한 뒤 안쪽 모델(Add83 → TwoStack)에 넘긴다. p0는 fit의 학습 정상에서"""
    def __init__(self, mk, ring_idx, i1, alpha=ALPHA): self.mk, self.ring_idx, self.i1, self.alpha = mk, list(ring_idx), i1, alpha
    def _tf(self, F):
        F = np.array(F, float); n = np.rint(np.expm1(F[:, self.i1])); has = n > 0
        R = F[np.ix_(has, self.ring_idx)]; k = R * n[has, None]
        F[np.ix_(has, self.ring_idx)] = (k + self.alpha * self.p0_) / (n[has, None] + self.alpha); return F
    def fit(self, F, y):
        F = np.asarray(F, float); y = np.asarray(y); n = np.rint(np.expm1(F[:, self.i1])); m = (y == 0) & (n > 0)
        self.p0_ = (F[np.ix_(m, self.ring_idx)] * n[m, None]).sum(0) / n[m].sum()
        self.m_ = self.mk().fit(self._tf(F), y); return self
    def fires(self, F): return self.m_.fires(self._tf(F))
    def decision_function(self, F): return self.m_.decision_function(self._tf(F))
    def __getattr__(self, k):
        if k in ("m_", "mk") or k.startswith("__"): raise AttributeError(k)
        return getattr(self.m_, k)

def shrink_mk(seed=0):
    base = lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(F0), nf=len(F0), df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True, seed=seed))
    return lambda: Shrink(base, [F0.index(c) for c in RING], F0.index("H1_nimp"))
ts_of = lambda top: (top.m_.m_, top._tf)                 # (TwoStack, 입력 변환)

class Bag:
    """J-축소 NB개를 층화 FRAC 부분표본으로 학습. 판정 = 평균 여유 ≥ 0 OR E2 과반 OR 8.3k (이진)"""
    def __init__(self, nb=NB, frac=FRAC): self.nb, self.frac = nb, frac
    def fit(self, F, y):
        F = np.asarray(F, float); y = np.asarray(y)
        sp = StratifiedShuffleSplit(self.nb, train_size=self.frac, random_state=0)
        self.ms_ = [shrink_mk(b)().fit(F[s], y[s]) for b, (s, _) in enumerate(sp.split(F, y))]; return self
    def _agg(self, fs):
        marg = np.array([f["score"] - ts_of(m)[0].tc_ for f, m in zip(fs, self.ms_)]); e2 = np.array([f["E2"] for f in fs])
        return dict(main=marg.mean(0) >= 0, E2=e2.mean(0) >= .5, margin=marg.mean(0), margins=marg, e2s=e2)
    def fires(self, F):
        fs = [m.fires(F) for m in self.ms_]; out = self._agg(fs); out["R83"] = fs[0]["R83"]; return out
    def fires_val(self, Mv):
        return self._agg([ts_of(m)[0].fires_raw(ts_of(m)[1](Mv), RAW) for m in self.ms_])
    def decision_function(self, F):
        f = self.fires(F); return (f["main"] | f["E2"] | f["R83"]).astype(float)

def val_single(top):
    ts, tf = ts_of(top); fv = ts.fires_raw(tf(Mv), RAW)
    return dict(main=fv["score"] >= ts.tc_, E2=fv["E2"], margin=fv["score"] - ts.tc_)

CFG = [("J-축소", shrink_mk(0), False), ("J-축소+배깅", lambda: Bag(), True)]

# %% 중첩 CV (같은 분할)
RES = {}
for nm, mk, bag in CFG:
    RES[nm] = nested_cv(M, y, g, mk, n_jobs=-1)
    ex.record(nm, RES[nm], names, main=bag, mid=not bag); print(nm, RES[nm]["f2"].mean(), flush=True)
NS_ = len(RES["J-축소"]["f2"])

# %% 전체 모델 · val · 흔들림 (바깥 5폴드 모델 + 전체 모델의 val 판정)
splits = [np.flatnonzero(np.isin(np.arange(ns.n), tr)) for tr, _ in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(M, y, g)] + [np.arange(ns.n)]
FIT, WOB = {}, {}
for nm, mk, bag in CFG:
    models = Parallel(n_jobs=6)(delayed(lambda tr: mk().fit(M[tr], y[tr]))(tr) for tr in splits)
    V = [m.fires_val(Mv) if bag else val_single(m) for m in models]
    VF = np.array([v["main"] | v["E2"] | r83v for v in V])     # (6, 40) 최종 판정
    full = models[-1]; fv = V[-1]
    cnt = np.where(y == 1, NS_ - RES[nm]["miss_counts"], RES[nm]["fp_counts"])
    w = dict(train_seed마다_갈림=int(((cnt > 0) & (cnt < NS_)).sum()), train_갈리는_파일=[short(names[i]) for i in np.flatnonzero((cnt > 0) & (cnt < NS_))],
             val_6모델_갈림=int((VF.min(0) != VF.max(0)).sum()), val_갈리는_파일=[short(nv[i]) for i in np.flatnonzero(VF.min(0) != VF.max(0))],
             val_고장수_6모델=[int(v) for v in VF.sum(1)])
    if bag:
        vb = np.array([(fv["margins"][b] >= 0) | fv["e2s"][b] | r83v for b in range(NB)])
        w["bag하나씩_val고장수"] = [int(v) for v in vb.sum(1)]
        w["bag별_w_AE"] = [round(float(ts_of(m)[0].c_.coef_[0][2]), 2) for m in full.ms_]
        w["bag별_문턱"] = [round(float(ts_of(m)[0].tc_), 2) for m in full.ms_]
    else:
        w["w_AE"] = round(float(ts_of(full)[0].c_.coef_[0][2]), 2); w["문턱"] = round(float(ts_of(full)[0].tc_), 2)
    FIT[nm] = dict(full=full, fv=fv, vfin=VF[-1], cnt=cnt); WOB[nm] = w; print(nm, w, flush=True)
ex.note("흔들림 · val · bag별 결합기", WOB)

# %% 오류 분석 표 (가로)
C = {nm: FIT[nm]["cnt"] for nm, _, _ in CFG}
files = [names[i] for i in range(ns.n) if any((C[nm][i] > 0) != (y[i] == 1) for nm in C)]
extra = {f"J-축소 ({NS_}번 중 고장)": {f: int(C["J-축소"][names.tolist().index(f)]) for f in files}}
ex.save_md("오류분석_표", f"모델 판단 행 = J-축소+배깅. 아래 행 = 같은 분할에서 J-축소의 고장 판정 횟수\n\n" + error_md_table(names, y, C["J-축소+배깅"], files, n_seeds=NS_, extra=extra))

# %% val 판정
Vt = pd.DataFrame({"파일": [n[-3:] for n in nv], **{nm: np.where(FIT[nm]["vfin"], "고장", "정상") for nm, _, _ in CFG},
                   **{f"{nm} 여유": np.round(FIT[nm]["fv"]["margin"], 2) for nm, _, _ in CFG}, "8.3k": np.where(r83v, "켜짐", "")})
d = Vt[Vt["J-축소"] != Vt["J-축소+배깅"]]
ex.save_md("val_판정", "val 고장 판정 수: " + " · ".join(f"{nm} {int(FIT[nm]['vfin'].sum())}/40" for nm, _, _ in CFG)
           + "\n\n여유 = 결합 점수 − 문턱(배깅은 bag 10개 평균). 규칙(E2 · 8.3k)이 켜지면 여유와 무관하게 고장\n\n"
           + f"두 구성의 판정이 다른 파일\n\n{md_table(d) if len(d) else '(없음)'}\n\n전체\n\n{md_table(Vt)}")
fig, ax = plt.subplots(figsize=(15, 4.8)); o = np.argsort(FIT["J-축소"]["fv"]["margin"]); xs = np.arange(40)
for nm, mk_, off in (("J-축소", "s", -.15), ("J-축소+배깅", "^", .15)):
    mg = FIT[nm]["fv"]["margin"][o]; fin = FIT[nm]["vfin"][o]
    ax.scatter(xs + off, mg, marker=mk_, s=46, c=np.where(fin, ABN, NOR), edgecolor="k", lw=.5, label=nm, zorder=3)
ax.axhline(0, color=INK, lw=1); ax.set_xticks(xs); ax.set_xticklabels([nv[i][-3:] for i in o], rotation=90, fontsize=8)
ax.set_ylabel("여유 (결합 점수 − 문턱)", color=MUTED); ax.grid(alpha=.25)
ax.legend(handles=[Line2D([], [], marker="s", ls="", mfc="#bbb", mec="k", label="J-축소"), Line2D([], [], marker="^", ls="", mfc="#bbb", mec="k", label="J-축소+배깅 (bag 평균)"),
                   Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="최종 고장 (규칙 포함)"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="최종 정상")], fontsize=8.5, loc="upper left")
ax.set_title("val 40 — 여유(가로 = J-축소 여유 순). 0 위 = 결합 고장, 규칙이 켜진 파일은 여유가 음수여도 고장(주황)", loc="left", fontsize=10)
fig.tight_layout(); ex.save_fig(fig, "val_판정", "val 40의 여유: J-축소 vs 배깅 평균")

# %% 핵심 그림: logistic_plane
# 좌표 = J-축소 전체 모델의 안쪽 OOF 결합 입력(가로 = ET logit + (c1/c0)·박자 z, 세로 = AE z). 점 색 = 중첩 CV seed 과반 판정(그 구성)
# 왼쪽 배경 = J-축소 결합 판단. 오른쪽 배경 = bag 10개 결합기의 평균 여유 부호(각 bag의 박자 몫은 J-축소 비율로 근사), 회색 선 = bag 하나하나의 경계
tsJ = ts_of(FIT["J-축소"]["full"])[0]; Z = tsJ.Z_; cJ, bJ, tcJ = tsJ.c_.coef_[0], tsJ.c_.intercept_[0], tsJ.tc_
xJ = Z[:, 0] + cJ[1] / cJ[0] * Z[:, 1]; aeJ = Z[:, 2]; t_et = float(best_f2_threshold(y, Z[:, 0]))
fJ = FIT["J-축소"]["full"].fires(M); comb = (tsJ.c_.decision_function(Z) >= tcJ); rule_only = (y == 1) & ~comb & (fJ["E2"] | r83)
x0, x1, y0, y1 = min(xJ.min(), t_et) - .6, max(xJ.max(), t_et) + .6, aeJ.min() - .6, aeJ.max() + .6
gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
BAGC = [(ts_of(m)[0].c_.coef_[0], ts_of(m)[0].c_.intercept_[0], ts_of(m)[0].tc_) for m in FIT["J-축소+배깅"]["full"].ms_]
fig, axs = plt.subplots(1, 2, figsize=(18, 7.2))
for ax, nm in zip(axs, ("J-축소", "J-축소+배깅")):
    if nm == "J-축소": G = cJ[0] * gx + cJ[2] * gy + bJ - tcJ
    else:
        G = np.mean([c[0] * gx + c[2] * gy + b - t for c, b, t in BAGC], 0)
        for c, b, t in BAGC: ax.contour(gx, gy, c[0] * gx + c[2] * gy + b - t, levels=[0], colors="#777", linewidths=.7, alpha=.6, zorder=1)
    ax.contourf(gx, gy, (G >= 0).astype(float), levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    fin = FIT[nm]["cnt"] > NS_ / 2; col = np.where(fin, ABN, NOR); show = ~rule_only; o_ = np.zeros(ns.n, bool); o_[OUT] = True
    for msk, mkr, s_ in ((show & (y == 0) & ~o_, "s", 22), (show & (y == 1), "o", 40), (show & o_, "D", 60)):
        ax.scatter(xJ[msk], aeJ[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
    gal = (FIT[nm]["cnt"] > 0) & (FIT[nm]["cnt"] < NS_)
    for i in np.flatnonzero(show & (gal | (fin != (y == 1)))):
        ax.annotate(f"{short(names[i])} {FIT[nm]['cnt'][i]}/{NS_}", (xJ[i], aeJ[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f} · seed마다 갈리는 train {WOB[nm]['train_seed마다_갈림']} · val 6모델 갈림 {WOB[nm]['val_6모델_갈림']}", loc="left", fontsize=10)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2); ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel(f"ET logit + ({cJ[1] / cJ[0]:.2f})·박자 z  (J-축소 좌표)", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정 (seed 과반)"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상"), Line2D([], [], color="#777", label="bag 하나하나의 경계")]
fig.legend(handles=leg, ncol=6, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 좌표 = J-축소 전체 모델의 안쪽 OOF 입력. 배경 = 결합 판단(오른쪽 = bag 10개 평균, 근사). 이름 = seed마다 갈리는 파일 · 틀린 파일(숫자 = 고장 판정 횟수). 규칙만으로 잡힌 고장은 뺌",
             x=.01, ha="left", fontsize=9.5, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "J-축소 vs J-축소+배깅의 결합 경계 (오른쪽 회색 선 = bag 10개 각각)", key=True)

# %% t-SNE 기본 (핵심 그림)
fpB = np.flatnonzero((y == 0) & (FIT["J-축소+배깅"]["cnt"] > NS_ / 2))
fig, ti = tsne_default(ns, mark=fpB, mark_label="배깅 오경보 (seed 과반)", title="EXP-051 (J 입력 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10). 빨간 원 = 배깅 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
