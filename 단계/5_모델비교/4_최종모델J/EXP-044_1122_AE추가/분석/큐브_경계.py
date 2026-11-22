# EXP-044 질문답용 (run.py는 고치지 않음): ① ET × AE 평면을 데이터 범위로 크롭 ② 박자 z를 세 번째 축으로 해서 "큐브 자르듯" 축에 평행한 경계로 나누면 어떤가
# 큐브 규칙(B1) = (ET ≥ t_e AND AE ≥ t_a) OR 박자 z ≥ t_z, 문턱 3개는 fit 행의 안쪽 OOF 점수에서 F2 최대로 (격자 탐색). 트리(B2) = 깊이 3 결정 트리(축에 평행한 경계의 일반형)
# 둘 다 J와 같게 E2 · 8.3k를 OR로 붙여 결합 방식만 바꾼다. 실행: conda run -n qe-phm python "단계/5_모델비교/4_최종모델J/EXP-044_1122_AE추가/분석/큐브_경계.py"
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt
from sklearn.metrics import fbeta_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier
from src.exp import make_et
from src.exp2util import *
from src.evaluate import nested_cv
from src.models2 import TwoAE, lg
from src.features2 import PeakRef, robust_z
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figs")
HERE_EXP = os.path.dirname(HERE)   # EXP-044 폴더 (result.json · figs/logistic_plane.png 위치)
ns = NS(); y, names, g = ns.y, ns.names, ns.g; nor = y == 0


class BoxAE(TwoAE):
    """TwoAE의 결합 로지스틱 자리를 축에 평행한 경계로 바꾼다. kind = 'box' | 'tree'"""
    def __init__(self, comb="lin", experts=(), fixed=(), nf=0, df=1.0, peak=(), q_exp=99.0, seed=0, scale_thr=1.0, kind="box"):
        super().__init__(comb, experts, fixed, nf, df, peak, q_exp, seed, scale_thr); self.kind = kind

    def _box(self, Z, te, ta, tz): return ((Z[:, 0] >= te) & (Z[:, 2] >= ta)) | (Z[:, 1] >= tz)

    def fit(self, F, yy):
        from src.experts import ae_mean
        F, yy = np.asarray(F, float), np.asarray(yy)
        idx = F[:, -1].astype(int); F = F[:, :-1]; self.fit_idx_ = idx
        self.pr_ = PeakRef(self.df).fit(F[yy == 0, self.nf + 1:])
        H, pk, S = self._parts(F); self.zf_ = robust_z(S[yy == 0]); z = self.zf_(S)
        oof, ae = np.empty(len(yy)), np.empty(len(yy))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(H, yy):
            oof[b] = lg(make_et(seed=self.seed)().fit(H[a], yy[a]).predict_proba(H[b])[:, 1]); ae[b] = ae_mean(idx[a], idx[b])[:, 0]
        self.m_ = make_et(seed=self.seed)().fit(H, yy)
        self.aef_ = robust_z(ae[yy == 0]); aez = self.aef_(ae); Z = np.column_stack([oof, z, aez])
        if self.kind == "tree":
            self.tree_ = DecisionTreeClassifier(max_depth=3, min_samples_leaf=3, class_weight="balanced", random_state=0).fit(Z, yy)
        else:
            ce = np.unique(np.percentile(Z[:, 0], np.linspace(5, 98, 40))); ca = np.unique(np.percentile(Z[:, 2], np.linspace(5, 98, 40)))
            cz = np.r_[np.unique(np.percentile(z[z > 0], np.linspace(0, 100, 12))) if (z > 0).any() else [], np.inf]
            best = (-1, 1, None)
            for tz in cz:
                for te in ce:
                    for ta in ca:
                        p = self._box(Z, te, ta, tz); f2 = fbeta_score(yy, p, beta=2, zero_division=0); fpr = p[yy == 0].mean()
                        if (f2, -fpr) > (best[0], -best[1]): best = (f2, fpr, (te, ta, tz))
            self.box_ = best[2]
        self.i1_ = self.fixed.index("H1_nimp") if "H1_nimp" in self.fixed else None
        self.t16_ = self._thr(pk[yy == 0, 0]); self.t18_ = self._thr(H[yy == 0, self.fixed.index("H18_hfl")])
        return self

    def _main(self, a, z, aez):
        Z = np.column_stack([a, z, aez])
        return self.tree_.predict(Z).astype(bool) if self.kind == "tree" else self._box(Z, *self.box_)

    def fires(self, F):
        from src.experts import ae_mean
        F = np.asarray(F, float); te = F[:, -1].astype(int); F = F[:, :-1]
        H, pk, S = self._parts(F); a, z = lg(self.m_.predict_proba(H)[:, 1]), self.zf_(S)
        raw = getattr(self, "ae_override_", None); aez = self.aef_(raw if raw is not None else ae_mean(self.fit_idx_, te)[:, 0])
        m = self._main(a, z, aez); out = dict(main=m, a=a, z=z, ae=aez, score=m.astype(float))
        if "E2" in self.experts: out["E2"] = (H[:, self.fixed.index("H18_hfl")] > self.t18_) & (H[:, self.i1_] >= np.log(2) - 1e-9)
        return out


spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float); idx = np.arange(ns.n, dtype=float)
F_rule = np.column_stack([ns.M, idx, r83])
mk = lambda kind: (lambda: Add83(lambda: BoxAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), kind=kind)))
r44 = json.load(open(os.path.join(HERE_EXP, "result.json"), encoding="utf-8"))["configs"]
J = r44["중첩: J | W + AE + E2 + 8.3k (최종 후보)"]
rows = {"J | 로지스틱 (EXP-044)": J}
for nm, kind in (("B1 | 큐브: (ET∧AE) ∨ 박자 z", "box"), ("B2 | 깊이 3 트리", "tree")):
    R = nested_cv(F_rule, y, g, mk(kind), n_jobs=-1)
    rows[nm] = dict(f2=float(R["f2"].mean()), sd=float(R["f2"].std()), rec=float(R["recall"].mean()), fpr=float(R["fpr"].mean()),
                    miss={names[i]: int(c) for i, c in enumerate(R["miss_counts"]) if c >= 2}, fp={names[i]: int(c) for i, c in enumerate(R["fp_counts"]) if c >= 2})
    print(nm, json.dumps(rows[nm], ensure_ascii=False), flush=True)

# ---- 전체 150 학습 → 큐브 문턱 · 그림 (OOF seed 0 점은 box 모델로)
mb = BoxAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), kind="box").fit(np.column_stack([ns.M, idx]), y)
te_, ta_, tz_ = mb.box_; print("큐브 문턱 (전체 150): ET ≥", round(te_, 2), "AE z ≥", round(ta_, 2), "박자 z ≥", tz_)
o = dict(a=np.empty(ns.n), z=np.empty(ns.n), ae=np.empty(ns.n), main=np.zeros(ns.n, bool))
Fn = np.column_stack([ns.M, idx])
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    f = BoxAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), kind="box").fit(Fn[tr], y[tr]).fires(Fn[te])
    for k in o: o[k][te] = f[k]
a, aez, zz = o["a"], o["ae"], o["z"]
# ① 크롭된 ET × AE 평면 (J와 같은 OOF 점, 경계는 J의 로지스틱 박자 z = 0 단면 + 큐브 사각형)
mJ = Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK))).fit(np.column_stack([ns.M, idx, r83]), y).m_
x0, x1, y0, y1 = a.min() - .3, a.max() + .3, aez.min() - .3, aez.max() + .3
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
D = mJ._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(13, 8.2))
ax.contourf(xx, yy, D, levels=[-1e9, mJ.tc_, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.7); ax.contour(xx, yy, D, levels=[mJ.tc_], colors="k", linewidths=1.6)
ax.add_patch(plt.Rectangle((te_, ta_), x1 - te_, y1 - ta_, fill=False, edgecolor="#2ca02c", lw=2.2, ls="--", label=f"큐브 문턱 (ET ≥ {te_:.2f} 그리고 AE z ≥ {ta_:.2f})"))
big = zz > 0.5
ax.scatter(a[nor & ~big], aez[nor & ~big], color=NOR, s=40, marker="s", lw=0, alpha=.75, label="정상 (박자 z ≤ 0.5)"); ax.scatter(a[nor & big], aez[nor & big], color=NOR, s=70, marker="^", lw=0, label="정상 (박자 z > 0.5)")
ax.scatter(a[~nor & ~big], aez[~nor & ~big], color=ABN, s=55, lw=0, label="고장 (박자 z ≤ 0.5)"); ax.scatter(a[~nor & big], aez[~nor & big], color=ABN, s=90, marker="^", lw=0, label="고장 (박자 z > 0.5)")
lab = [names.index(f) for f in ("abn_002", "abn_017", "abn_022", "abn_014", "nor_116", "nor_059", "nor_065", "nor_110", "nor_095", "nor_037", "nor_073", "nor_025")]
for k, i in enumerate(lab): ax.annotate(f"{short(names[i])} (z {zz[i]:.1f})", (a[i], aez[i]), xytext=(8, 8 + 9 * (k % 3)), textcoords="offset points", fontsize=9, color=ABN if y[i] else NOR, weight="bold")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xlabel("ET 점수 (logit, OOF seed 0)"); ax.set_ylabel("AE z (정상 기준, OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=9, loc="upper left")
ax.set_title("ET × AE 평면 (데이터 범위로 크롭). 파랑/주황 배경 = J의 로지스틱 경계(박자 z = 0 단면), 초록 점선 = 큐브 규칙의 ET·AE 사각형", fontsize=10.5); fig.tight_layout()
fig.savefig(os.path.join(OUT, "ET_AE_평면_크롭.png"), dpi=120); plt.close(fig)
# ② 3차원: 박자 z를 세 번째 축. 큐브 문턱 면 표시
fig = plt.figure(figsize=(16, 7.5))
for k, (el, az) in enumerate(((22, -62), (18, 28))):
    ax = fig.add_subplot(1, 2, k + 1, projection="3d"); zc = np.clip(zz, -1, 10)
    ax.scatter(a[nor], aez[nor], zc[nor], color=NOR, s=14, alpha=.5); ax.scatter(a[~nor], aez[~nor], zc[~nor], color=ABN, s=26)
    zt = tz_ if np.isfinite(tz_) else 10
    G1, G2 = np.meshgrid([te_, a.max() + .3], [ta_, aez.max() + .3]); ax.plot_surface(G1, G2, np.full_like(G1, 0) + (-1), alpha=.0)
    for xs in ([te_, te_], ): pass
    ax.plot([te_, te_], [ta_, ta_], [-1, 10], color="#2ca02c", lw=1.5)
    Y_, Z_ = np.meshgrid([ta_, aez.max() + .3], [-1, 10]); ax.plot_surface(np.full_like(Y_, te_), Y_, Z_, color="#2ca02c", alpha=.12)
    X_, Z_ = np.meshgrid([te_, a.max() + .3], [-1, 10]); ax.plot_surface(X_, np.full_like(X_, ta_), Z_, color="#2ca02c", alpha=.12)
    if np.isfinite(tz_):
        X_, Y_ = np.meshgrid([a.min() - .3, a.max() + .3], [aez.min() - .3, aez.max() + .3]); ax.plot_surface(X_, Y_, np.full_like(X_, tz_), color="#7a3db8", alpha=.15)
    for f in ("abn_022", "abn_014", "abn_002", "abn_017"):
        i = names.index(f); ax.text(a[i], aez[i], zc[i], short(f), fontsize=9, color=ABN)
    ax.set_xlabel("ET"); ax.set_ylabel("AE z"); ax.set_zlabel("박자 z (-1~10로 자름)"); ax.view_init(el, az)
fig.suptitle("ET × AE × 박자 z 3차원. 초록 면 = ET·AE 문턱(둘 다 넘으면 고장), 보라 면 = 박자 z 문턱(넘으면 고장)", x=.01, ha="left", fontsize=11); fig.tight_layout()
fig.savefig(os.path.join(OUT, "큐브_경계_3d.png"), dpi=110); plt.close(fig)
json.dump(dict(rows=rows, box=[float(te_), float(ta_), float(tz_) if np.isfinite(tz_) else None]), open(os.path.join(HERE, "figs", "큐브_결과.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("done")
