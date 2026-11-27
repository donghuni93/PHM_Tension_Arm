# %% [markdown]
# EXP-050 울림 비율 축소 × H10 빼기 (2×2): J · J−H10 · J-축소 · J−H10+축소
# - 두 세션 협의 설계(11-22, claude-70 경계 파일 진단 6-4 · claude-af ET 진단 _임시_1122). 같은 바깥 분할, full(seed 5 · ET 300)
# - 가설
#   축소(H1 · H2): 충격이 적을 때 울림 주파수 비율(H7–H10)은 믿기 어렵다 → 비율을 충격 수에 맞춰 정상 기준 쪽으로 당긴다
#     H_b' = (k_b + α·p0_b) / (n + α),  n = round(expm1(H1_nimp)), k_b = H_b · n, p0_b = 폴드 안 학습 정상의 모든 충격에서 b 대역 비율
#     α = 3 (실행 전 고정, 다른 값은 돌리지 않는다). 충격 0개 파일은 기존 −1 유지
#     ※ 축소의 동기는 val 분포 관찰(충격 1–3개 val 9/17이 4 kHz 아래로 절반 이상 울림, 라벨 안 씀)에서 나왔다 → 주 잣대는 train 중첩 CV, val은 보조
#   H10 빼기(claude-af): H10(8–12.5k 울림 비율)은 빼도 ET 손해가 없고(그림7) AE와 겹치며(ρ +0.55) 지속음 주파수에 값이 좌우된다 → 유형 B(N065 · V021) · 결합기 w_AE 흔들림
# - 잣대: ① 중첩 F2(주) ② 흔들림 (a) 폴드 경계 5개 + 전체 경계에서 판정이 바뀌는 파일 수(규칙 꺼진 파일) (b) 부트스트랩 300회 w_AE · 문턱 5–95 %
#   ③ 경계 13개 · 충격 4–6개 고장의 이동 ④ val 판정 수(보조). (a)(b)는 _임시_1122/결합기흔들림.py와 같은 정의(import하지 않고 옮김)
# - 이상치 학습 제외는 쓰지 않는다(J · EXP-049와 같은 조건 비교)
# 실행: QE_FULL=1 conda run -n qe-phm python run.py

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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore"); assert FULL, "QE_FULL=1 로 실행"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
ALPHA = 3.0                                              # 실행 전 고정
RING = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]
ex = Experiment(__file__, "EXP-050", "울림 비율 축소 × H10 빼기 (2×2)", ["H1", "H2"],
                member="이동훈", model="ET(H 피처, 울림 비율 축소 · H10 제외 조합) + 박자 z + AE z → 로지스틱 OR E2 OR 8.3k",
                changes="J의 ET 입력에서 울림 비율 H7–H10을 충격 수로 축소(α=3 사전 고정) · H10 제외, 2×2. 같은 분할 full",
                key_changes=["울림 비율 축소 (k + 3·p0)/(n + 3), p0 = 폴드 안 학습 정상", "H10 제외", "두 변경의 2×2", "흔들림 잣대 (a) 판정 바뀌는 파일 · (b) 부트스트랩 w_AE"])
ns = NS(); y, g, names = ns.y, ns.g, np.array(ns.names); nor = y == 0; nv = [f"val_{n[-3:]}" for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83, r83v = spec | res, sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
OUT = [names.tolist().index(n) for n in OUTLIER_NORMALS]
short = lambda n: n.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
BOUND = ["abn_009", "nor_065", "nor_037", "nor_095", "nor_110", "nor_106", "nor_073", "val_025", "val_018", "val_021", "val_012", "val_036", "val_037"]
NIMP = np.rint(np.expm1(ns.T["H1_nimp"].values)).astype(int)
F46 = [names[i] for i in range(ns.n) if y[i] == 1 and 4 <= NIMP[i] <= 6]

def mats(fixed):
    M = np.column_stack([ns.T[fixed].values, ns.T[["L1_S"]].values, ns.L, np.arange(ns.n, dtype=float), r83.astype(float)])
    Mv = np.column_stack([ns.Tv[fixed].values, ns.Tv[["L1_S"]].values, ns.Lv])
    return M, Mv

class Shrink:
    """ET 입력의 울림 비율 열을 충격 수로 축소한 뒤 안쪽 모델(Add83 → TwoStack)에 넘긴다. p0는 fit의 학습 정상에서"""
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

def build(fixed, shrink):
    nf = len(fixed)
    base = lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(fixed), nf=nf, df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True))
    if not shrink: return base
    ri = [fixed.index(c) for c in RING if c in fixed]; i1 = fixed.index("H1_nimp")
    return lambda: Shrink(base, ri, i1)
def ts_of(top):                                          # (TwoStack, 입력 변환)
    return (top.m_.m_, top._tf) if isinstance(top, Shrink) else (top.m_, lambda F: np.asarray(F, float))

F0 = list(ns.FIXED); F_H10 = [c for c in F0 if c != "H10_ring_8-12.5k"]
CFG = [("J", F0, False), ("J−H10", F_H10, False), ("J-축소", F0, True), ("J−H10+축소", F_H10, True)]

# %% 중첩 CV (full, 같은 분할)
RES = {}
for nm, fx, sh in CFG:
    M, _ = mats(fx); RES[nm] = nested_cv(M, y, g, build(fx, sh), n_jobs=-1)
    ex.record(nm, RES[nm], names, main=(nm == "J−H10+축소"), mid=(nm != "J−H10+축소"))

# %% 전체 학습 모델 · val · 흔들림 (a)(b)
FIT, WOB = {}, {}
for nm, fx, sh in CFG:
    M, Mv = mats(fx); top = build(fx, sh)().fit(M, y); ts, tf = ts_of(top)
    ft = top.fires(M); fv = ts.fires_raw(tf(Mv), RAW)
    comb = ts.c_.decision_function(ts.Z_) >= ts.tc_
    FIT[nm] = dict(top=top, ts=ts, tf=tf, comb=comb, fin=comb | ft["E2"] | r83, fv=fv, vfin=fv["main"] | fv["E2"] | r83v, p0=getattr(top, "p0_", None))
    # (a) 폴드 경계 5개 + 전체 경계 → 같은 좌표에서 여유
    WS = []; Xtr = np.zeros((ns.n, 3)); rule_tr = np.zeros(ns.n, bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(M, y, g):
        mf = build(fx, sh)().fit(M[tr], y[tr]); tsf, tff = ts_of(mf); f = mf.fires(M[te])
        Xtr[te] = np.column_stack([f["a"], f["z"], f["ae"]]); rule_tr[te] = f["E2"] | f["R83"]
        WS.append((tsf.c_.coef_[0], tsf.c_.intercept_[0] - tsf.tc_))
    WS.append((ts.c_.coef_[0], ts.c_.intercept_[0] - ts.tc_))
    Xv = np.column_stack([fv["a"], fv["z"], fv["ae"]]); rule_v = fv["E2"] | r83v
    Mt = np.array([Xtr @ w + b for w, b in WS]); Mvv = np.array([Xv @ w + b for w, b in WS])
    flip_t = (Mt.min(0) < 0) & (Mt.max(0) >= 0) & ~rule_tr; flip_v = (Mvv.min(0) < 0) & (Mvv.max(0) >= 0) & ~rule_v
    # (b) 부트스트랩 300회 (전체 모델 안쪽 OOF 결합 입력)
    rs = np.random.RandomState(0); Z, yz = ts.Z_, ts.y_; wae, thr, nvf = [], [], []
    for _ in range(300):
        i = rs.randint(0, len(yz), len(yz))
        if len(np.unique(yz[i])) < 2: continue
        lr = LogisticRegression(C=1.0, class_weight="balanced").fit(Z[i], yz[i]); t = float(best_f2_threshold(yz[i], lr.decision_function(Z[i])))
        wae.append(lr.coef_[0][2]); thr.append(t); nvf.append(int((lr.decision_function(Xv) >= t).sum()))
    q = lambda v: [round(float(np.percentile(v, 5)), 2), round(float(np.percentile(v, 95)), 2)]
    WOB[nm] = dict(바뀌는_train=int(flip_t.sum()), 바뀌는_val=int(flip_v.sum()),
                  train_파일=[short(names[i]) for i in np.flatnonzero(flip_t)], val_파일=[short(nv[i]) for i in np.flatnonzero(flip_v)],
                  w_AE_5_95=q(wae), 문턱_5_95=q(thr), 결합만_val고장_5_95=q(nvf), 계수=np.round(ts.c_.coef_[0], 3).tolist(),
                  p0=None if FIT[nm]["p0"] is None else np.round(FIT[nm]["p0"], 3).tolist())
    print(nm, WOB[nm], flush=True)
ex.note("흔들림 (a)(b) · 결합 계수 · p0", WOB)

# %% 경계 13개 · 충격 4–6개 고장 이동: train = 중첩 5번 중 고장 판정 횟수, val = 전체 모델 최종 판정
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm, _, _ in CFG}
rows = []
for f in BOUND + [x for x in F46 if x not in BOUND]:
    if f.startswith("val"):
        k = nv.index(f); rows.append({"파일": short(f), "구분": "val (경계)", **{nm: ("고장" if FIT[nm]["vfin"][k] else "정상") for nm, _, _ in CFG}})
    else:
        i = names.tolist().index(f); tag = ("고장" if y[i] else "정상") + (" (경계)" if f in BOUND else "") + (f" · 충격 {NIMP[i]}개" if f in F46 else "")
        rows.append({"파일": short(f), "구분": tag, **{nm: f"고장 {C[nm][i]}/5" for nm, _, _ in CFG}})
ex.save_md("경계파일_이동", "train = 중첩 CV seed 5번 중 고장 판정 횟수, val = train 150 전체 학습 모델의 최종 판정(규칙 포함)\n\n" + md_table(pd.DataFrame(rows)))

# %% 오류 분석 표 (가로)
files = [names[i] for i in range(ns.n) if any((C[nm][i] > 0) != (y[i] == 1) for nm in C)]
extra = {f"{nm} (5번 중 고장)": {f: int(C[nm][names.tolist().index(f)]) for f in files} for nm in ("J−H10", "J-축소", "J−H10+축소")}
ex.save_md("오류분석_표", "모델 판단 행 = J. 아래 세 행 = 같은 분할에서 다른 구성의 고장 판정 횟수\n\n" + error_md_table(names, y, C["J"], files, n_seeds=5, extra=extra))

# %% val 판정
V = pd.DataFrame({"파일": [n[-3:] for n in nv], **{nm: np.where(FIT[nm]["vfin"], "고장", "정상") for nm, _, _ in CFG},
                  **{f"{nm} 결합점수−문턱": np.round(FIT[nm]["fv"]["score"] - FIT[nm]["ts"].tc_, 2) for nm, _, _ in CFG}})
d = V[(V[[nm for nm, _, _ in CFG]].nunique(axis=1) > 1)]
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
    mv = np.abs(x - et) >= .3
    for i in np.flatnonzero(show & mv): ax.plot([et[i], x[i]], [y3[i], y3[i]], ls=":", color="#888", lw=.8, zorder=1)
    col = np.where(fin, ABN, NOR)
    if val:
        ax.scatter(x, y3, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in range(len(x)):
            if nv[i] in BOUND or fin[i] != FIT["J"]["vfin"][i]:
                ax.annotate(short(nv[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold" if fin[i] != FIT["J"]["vfin"][i] else "normal")
    else:
        o = np.zeros(len(x), bool); o[OUT] = True
        for msk, mkr, s_ in ((show & (y == 0) & ~o, "s", 22), (show & (y == 1), "o", 40), (show & o, "D", 60)):
            ax.scatter(x[msk], y3[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & ((fin != (y == 1)) | np.isin(names, BOUND))):
            ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    r = ex.res["configs"][nm]; w = WOB[nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f} · 바뀌는 파일 train {w['바뀌는_train']} · val {w['바뀌는_val']}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")
leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116")]
fig, axs = plt.subplots(2, 2, figsize=(18, 13))
for ax, (nm, _, _) in zip(axs.ravel(), CFG): plane(ax, nm)
fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합). 배경 = 결합 판단, 세로 점선 = ET 단독 판단. 이름 = 경계 파일 · 틀린 파일. 규칙만으로 잡힌 고장은 뺌", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .96)); ex.save_fig(fig, "logistic_plane", "J · J−H10 · J-축소 · J−H10+축소의 결합 경계 (2×2)", key=True)
fig, axs = plt.subplots(2, 2, figsize=(18, 13))
for ax, (nm, _, _) in zip(axs.ravel(), CFG): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 결합 경계 위의 val (train 150 전체 학습). 굵은 글씨 = J와 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .96)); ex.save_fig(fig, "val_판정", "네 구성의 val 판정")

# %% t-SNE 기본 (핵심 그림, J 입력 공간)
fpJ = np.flatnonzero((y == 0) & (RES["J"]["fp_counts"] >= 3))
fig, ti = tsne_default(ns, mark=fpJ, mark_label="J 오경보 (5번 중 3회+)", title="EXP-050 (J 입력 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10). 빨간 원 = J 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
