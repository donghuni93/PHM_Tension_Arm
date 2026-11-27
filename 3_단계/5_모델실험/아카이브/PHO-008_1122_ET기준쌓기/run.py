# %% [markdown]
# PHO-008 (휴대폰 세션, 11-22) ET 기준 쌓기: J(ET H15 · 박자 · AE → 로지스틱 OR E2 OR 8.3k)에 정보가 다른 1단 전문가를 더한다
# - 후보(점수가 아니라 정보가 다른지로 고름, 실행 전 고정): ext(ET on ext150) · shape(모양 AE, 게인 제거) · mspc(ext150 정상 PCA T²+SPE) · knn(H 피처 정상 5-NN 거리)
# - 2단 = 로지스틱(L2, C=1, 균형). 규칙(E2 · 8.3k)은 J와 같음. 클래스: src.models2.TwoStack
# - 판정 규칙(사전 고정): quick(seed 2 · ET 100)에서 J보다 +0.010 이상 → full(QE_FULL=1, seed 5 · ET 300)로 확인
#   채택 = full F2 ≥ 0.977 + 0.010, 또는 재현율 1.000 유지 + 116 · 065 · 059 중 하나를 확실히(오경보 0–1/5) 내림
# 실행: python run.py (quick 단계) → QE_FULL=1 python run.py (full 단계: quick에서 통과한 구성만)
# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import FULL, MODE
from src.exp2util import NS, Add83, rule83_flags, short, NOR, ABN
from src.models2 import TwoStack
from sklearn.model_selection import StratifiedGroupKFold
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__))
ex = Experiment(__file__, "PHO-008", "ET 기준 쌓기 (정보가 다른 1단 전문가 추가)", ["H1", "H2", "H3"],
                member="이동훈", model="J + 1단 전문가(ext · shape · mspc · knn) → 로지스틱 OR E2 OR 8.3k",
                changes="J의 2단 로지스틱 입력에 ext150 ET · 모양 AE · 정상 PCA · 정상 kNN 점수를 하나씩 / 모두 추가",
                key_changes=["1단 전문가 후보 4개(ext · shape · mspc · knn)를 하나씩, 그리고 모두", "2단 로지스틱 · 규칙 · 문턱은 J와 같음", "quick 선별 → full 확인 (사전 고정 규칙)"])
ns = NS(); y, g, names = ns.y, ns.g, ns.names; nor = y == 0
spec, res = rule83_flags(ns.X); M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)])
mk = lambda extra: (lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), extra=extra)))
CFG = [("J", ()), ("J + ext", ("ext",)), ("J + shape", ("shape",)), ("J + mspc", ("mspc",)), ("J + knn", ("knn",)), ("J + 넷 다", ("ext", "shape", "mspc", "knn"))]
tag = "full" if FULL else "quick"

# %% 중첩 CV
if not FULL:
    run = CFG
else:
    q = ex.res.get("notes", {}).get("quick_pass", []); run = [c for c in CFG if c[0] in ["J"] + q]
    print("full 대상:", [c[0] for c in run])
for nm, extra in run:
    ex.record(f"{tag}: {nm}", nested_cv(M, y, g, mk(extra), n_jobs=-1), names, main=False, mid=not (FULL and nm != "J"))
C = ex.res["configs"]
if not FULL:
    j = C["quick: J"]["f2"]; passed = [nm for nm, _ in CFG[1:] if C[f"quick: {nm}"]["f2"] >= j + 0.010]
    ex.note("quick_pass", passed); print("quick 통과(J +0.010 이상):", passed or "없음")
    best = max(CFG[1:], key=lambda c: C[f"quick: {c[0]}"]["f2"])[0]; ex.res["main"] = f"quick: {best}"; ex._write()
else:
    full_rows = [k for k in C if k.startswith("full:") and k != "full: J"]
    if full_rows: ex.res["main"] = max(full_rows, key=lambda k: C[k]["f2"]); ex._write()

# %% 핵심 그림 (quick 단계에서만): logistic_plane — J vs J + 넷 다 (ET × AE, OOF seed 0, 규칙으로 잡힌 고장 제외)
if not FULL:
    def oof(extra, seed=0):
        o = dict(a=np.empty(ns.n), ae=np.empty(ns.n), main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool), R83=np.zeros(ns.n, bool))
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(g, y, g):
            f = mk(extra)().fit(M[tr], y[tr]).fires(M[te])
            for k in o: o[k][te] = f[k]
        return o
    fig, axs = plt.subplots(1, 2, figsize=(18, 7.5), sharey=True)
    for ax, (nm, extra) in zip(axs, (CFG[0], CFG[-1])):
        o = oof(extra); a, ae = o["a"], o["ae"]; rule = o["E2"] | o["R83"]; fin = o["main"] | rule; show = ~(~nor & rule)
        m = mk(extra)().fit(M, y).m_; x0, x1, y0, y1 = a.min() - .3, a.max() + .3, ae.min() - .3, ae.max() + .3
        xx, yy = np.meshgrid(np.linspace(x0, x1, 250), np.linspace(y0, y1, 250))
        D = (m._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()) >= m.tc_).reshape(xx.shape).astype(int)
        ax.contourf(xx, yy, D, levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0); ax.contour(xx, yy, D, levels=[.5], colors="#333", linewidths=1.1, linestyles="--")
        ax.scatter(a[nor & ~fin], ae[nor & ~fin], color=NOR, s=28, marker="s", lw=0, alpha=.75, label="정상 → 정상")
        ax.scatter(a[nor & fin], ae[nor & fin], color=NOR, s=95, marker="s", edgecolor="#c00000", lw=2, label=f"정상 → 오경보 ({int((nor & fin).sum())})")
        ax.scatter(a[~nor & show & fin], ae[~nor & show & fin], color=ABN, s=55, lw=0, label="고장 → 고장 (결합)")
        ax.scatter(a[~nor & ~fin], ae[~nor & ~fin], color=ABN, s=120, marker="X", edgecolor="k", label=f"고장 → 놓침 ({int((~nor & ~fin).sum())})")
        for f_ in ("nor_116", "nor_065", "nor_059", "abn_022"):
            i = names.index(f_); ax.annotate(short(f_), (a[i], ae[i]), xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold")
        r = C[f"quick: {nm}"]; ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.25); ax.set_xlabel("ET 점수 (logit, OOF)"); ax.legend(fontsize=8.5, loc="upper left")
        ax.set_title(f"{nm} — quick F2 {r['f2']:.3f} · 오경보율 {r['fpr']:.3f} (배경 = 다른 입력 0 단면, 규칙 고장 제외)", fontsize=10.5)
    axs[0].set_ylabel("AE z (OOF)"); fig.tight_layout()
    ex.save_fig(fig, "logistic_plane", "ET × AE 평면 (OOF seed 0). 왼쪽 J, 오른쪽 J + 전문가 넷. 배경 = 결합 판단 영역(박자 · 새 전문가 z = 0 단면)", key=True)
    m_all = mk(CFG[-1][1])().fit(M, y).m_
    ex.note("2단 계수 (J + 넷 다, 전체 학습)", dict(zip(["ET", "박자", "AE", "ext", "shape", "mspc", "knn"], np.round(m_all.c_.coef_[0], 3).tolist())))
print("done", MODE)
