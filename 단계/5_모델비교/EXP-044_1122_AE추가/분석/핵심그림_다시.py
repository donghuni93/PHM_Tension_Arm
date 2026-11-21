# EXP-044 핵심 그림(logistic_plane.png)을 다시 그린다: 오른쪽을 데이터 범위로 크롭하고 최종 판정(J)으로 색 입힘. run.py는 고치지 않음 (같은 파일 이름으로 덮어써서 README 링크 유지)
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedGroupKFold
from src.exp2util import *
from src.models2 import TwoAE
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figs")
HERE_EXP = os.path.dirname(HERE)   # EXP-044 폴더 (result.json · figs/logistic_plane.png 위치)
ns = NS(); y, names, g = ns.y, ns.names, ns.g; nor = y == 0
res = json.load(open(os.path.join(HERE_EXP, "result.json"), encoding="utf-8")); cfg = res["configs"]; J = cfg["중첩: J | W + AE + E2 + 8.3k (최종 후보)"]
spec, rs = rule83_flags(ns.X); r83 = (spec | rs).astype(float); idx = np.arange(ns.n, dtype=float); F = np.column_stack([ns.M, idx, r83])
mkJ = lambda: Add83(lambda: TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)))
o = dict(a=np.empty(ns.n), z=np.empty(ns.n), ae=np.empty(ns.n), main=np.zeros(ns.n, bool), E2=np.zeros(ns.n, bool), R83=np.zeros(ns.n, bool))
for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
    f = mkJ().fit(F[tr], y[tr]).fires(F[te])
    for k in o: o[k][te] = f[k]
a, aez, zz = o["a"], o["ae"], o["z"]; caught = o["main"] | o["E2"] | o["R83"]
cat = np.where(o["R83"], "8.3k 규칙", np.where(o["E2"], "E2", np.where(o["main"], "결합", "못 잡음")))
mJ = mkJ().fit(F, y).m_
x0, x1, y0, y1 = a.min() - .3, a.max() + .3, aez.min() - .3, aez.max() + .3
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300)); D = mJ._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()).reshape(xx.shape)
fig, ax = plt.subplots(figsize=(11, 8.2))
ax.contour(xx, yy, D, levels=[mJ.tc_], colors="k", linewidths=1.3, linestyles="--")
ax.scatter(a[nor & ~caught], aez[nor & ~caught], color=NOR, s=36, marker="s", lw=0, alpha=.75, label="정상 → 정상 판정")
ax.scatter(a[nor & caught], aez[nor & caught], color=NOR, s=100, marker="s", edgecolor="#d62728", lw=2, label="정상 → 오경보 (최종)")
for k, c in {"결합": "#e8590c", "E2": "#7a3db8", "8.3k 규칙": "#000000"}.items():
    m = (cat == k) & ~nor; ax.scatter(a[m], aez[m], s=90 if k == "결합" else 150, color=ABN, edgecolor=c, lw=.5 if k == "결합" else 2.4, label=f"고장 → {k}이 잡음 ({int(m.sum())})")
big = ~nor & (zz > .5); ax.scatter(a[big], aez[big], s=240, facecolor="none", edgecolor="#1f9d55", lw=1.6, label=f"고장 · 박자 z > 0.5 ({int(big.sum())})")
for f_ in ("abn_002", "abn_017", "abn_022", "nor_116", "nor_059", "nor_065"):
    i = names.index(f_); ax.annotate(f"{short(f_)} (z {zz[i]:.1f})", (a[i], aez[i]), xytext=(8, 8), textcoords="offset points", fontsize=9, color=ABN if y[i] else NOR, weight="bold")
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_xlabel("ET 점수 (logit, OOF)"); ax.set_ylabel("AE z (정상 기준, OOF)"); ax.grid(alpha=.3); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title(f"J 최종 판정 (결합 OR E2 OR 8.3k) — 점선 = 결합 경계(박자 z = 0 단면)\n중첩 F2 {J['f2']:.3f} ± {J['f2_sd']:.3f} · 재현율 {J['recall']:.3f} · 오경보율 {J['fpr']:.3f} · seed 0에서 고장 {int(caught[~nor].sum())}/30", fontsize=9.5)
fig.suptitle("logistic_plane — 최종 후보 J (ET 점수 × AE z, 데이터 범위로 크롭, 최종 판정으로 색). OOF 5-fold seed 0", x=.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, .95))
fig.savefig(os.path.join(HERE_EXP, "figs", "logistic_plane.png"), dpi=80); plt.close(fig)
res["figs"]["logistic_plane"] = "최종 후보 J의 평면 (데이터 범위로 크롭, 최종 판정으로 색: 8.3k 검은 테두리 · E2 보라 · 박자 z > 0.5 초록 원). 점선 = 결합 경계의 박자 z = 0 단면"
LABS = ["W | 2단계 승자 C-lin (EXP-041 값)", "G | W + E2 + 8.3k (EXP-043 값)", "K | W + AE (전문가 · 규칙 없음)", "L | W + AE + E2 (8.3k 없음)", "J | W + AE + E2 + 8.3k (최종 후보)"]
cfg.pop("중첩: A | EXP-034 b (EXP-040 값)", None)                                   # 44를 기준으로 삼으므로 EXP-034 b(A) 비교는 뺀다
bad = sorted({k for c in LABS for k in list(cfg["중첩: " + c]["miss_counts"]) + list(cfg["중첩: " + c]["fp_counts"]) if max(cfg["중첩: " + c]["miss_counts"].get(k, 0), cfg["중첩: " + c]["fp_counts"].get(k, 0)) >= 3})
class _Ex: pass
ex_ = _Ex(); ex_.res = res
hf = hard_fig(ns, ex_, LABS, bad); hf.savefig(os.path.join(HERE_EXP, "figs", "hard_files.png"), dpi=80); plt.close(hf)
res["figs"]["hard_files"] = "어느 파일이 W · G · K · L · J에서 맞고 틀리는가"
json.dump(res, open(os.path.join(HERE_EXP, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1); print("ok")
