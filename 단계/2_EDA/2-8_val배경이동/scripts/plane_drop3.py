import os, sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
import numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "WenQuanYi Zen Hei"):
    if any(f == x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
import src.exp2util as U, src.experts as EXP
_lf = U.load_folder; U.load_folder = lambda p, *a, **k: _lf(p[:-4] if p.endswith(os.sep + "val") else p, *a, **k)   # 이 컨테이너의 val 경로
from src.exp2util import NS, rule83_flags, Add83, NOR, ABN, short
from src.models2 import TwoAE
from src.evaluate import best_f2_threshold
from sklearn.model_selection import StratifiedGroupKFold
plt.rcParams["font.family"] = "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
ns = NS(); names, y, g = ns.names, ns.y, ns.g; nor = y == 0
spec, res = rule83_flags(ns.X); r83 = (spec | res).astype(float)
idx = np.arange(ns.n, dtype=float); M_rule = np.column_stack([ns.M, idx, r83])
mk = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
DROP = {names.index(k) for k in ("nor_116", "nor_065", "nor_059")}
_ae = EXP.ae_mean
def run(drop):
    EXP.ae_mean = (lambda tr, te: _ae(np.array([i for i in tr if i not in DROP]), te)) if drop else _ae
    o = dict(a=np.empty(ns.n), ae=np.empty(ns.n), z=np.empty(ns.n), main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool), R83=np.zeros(ns.n, bool))
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        f = mk().fit(M_rule[tr], y[tr]).fires(M_rule[te])
        for k in o: o[k][te] = f[k]
    m = mk().fit(M_rule, y).m_
    EXP.ae_mean = _ae
    return o, m
fig, axs = plt.subplots(1, 2, figsize=(18, 8))
for ax, drop, ttl in zip(axs, (False, True), ("J (EXP-044 그대로)", "J, AE 학습에서 정상 116·065·059 제외")):
    o, m = run(drop); a, aez, zz = o["a"], o["ae"], o["z"]; caught = o["main"] | o["E2"] | o["R83"]
    cat = np.where(o["R83"], "8.3k", np.where(o["E2"], "E2", np.where(o["main"], "결합", "—")))
    t_et = float(best_f2_threshold(y, m.Z_[:, 0]))
    x0, x1, y0, y1 = -4.5, 4.5, -2.5, 6.5
    xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = m._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
    ax.axvline(t_et, color="#7a3db8", lw=2, label=f"ET 단독 경계 (ET ≥ {t_et:.2f})")
    ax.contour(xx, yy, D, levels=[m.tc_], colors="k", linewidths=1.3, linestyles="--"); ax.plot([], [], "k--", label="결합 경계 (박자 z = 0)")
    ax.scatter(a[nor & ~caught], aez[nor & ~caught], color=NOR, s=30, marker="s", lw=0, alpha=.7, label="정상 → 정상")
    ax.scatter(a[nor & caught], aez[nor & caught], color=NOR, s=100, marker="s", edgecolor="#d62728", lw=2, label=f"정상 → 오경보 ({int((nor & caught).sum())})")
    for k, c in (("결합", "#e8590c"), ("E2", "#7a3db8"), ("8.3k", "#000000")):
        mm = (cat == k) & ~nor; ax.scatter(a[mm], aez[mm], s=90 if k == "결합" else 150, color=ABN, edgecolor=c, lw=.5 if k == "결합" else 2.4, label=f"고장 → {k} ({int(mm.sum())})")
    miss = ~nor & ~caught; ax.scatter(a[miss], aez[miss], s=150, color="white", edgecolor=ABN, lw=2.2, label=f"고장 → 놓침 ({int(miss.sum())})")
    for f_ in ("abn_002", "abn_017", "abn_022", "nor_116", "nor_059", "nor_065"):
        i = names.index(f_); ax.annotate(short(f_), (a[i], aez[i]), xytext=(7, 7), textcoords="offset points", fontsize=10, color=ABN if y[i] else NOR, weight="bold")
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left"); ax.set_xlabel("ET 점수 (logit, OOF)"); ax.set_ylabel("AE z (OOF)")
    ax.set_title(f"{ttl}\nOOF seed 0: 고장 {int(caught[~nor].sum())}/30 · 오경보 {int(caught[nor].sum())}/120", fontsize=11)
    print(ttl, "| 고장", int(caught[~nor].sum()), "오경보", int(caught[nor].sum()), [names[i] for i in np.flatnonzero(nor & caught)], "| AE z", {short(k): round(float(aez[names.index(k)]), 2) for k in ("nor_116", "nor_065", "nor_059")})
fig.tight_layout(); fig.savefig(sys.argv[1], dpi=110)
