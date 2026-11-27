# EXP-043 해석용: ET 점수만으로 판정했을 때 맞고 틀리는 파일과, 최종 G(박자 z · E2 · 8.3k)가 그 오류를 어디서 고치는가. run.py는 고치지 않음
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt
from src.exp2util import *
from src.evaluate import best_f2_threshold
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); EXP041 = find_exp("EXP-041")
ns = NS(); y, names, nor = ns.y, ns.names, ns.y == 0
W = json.load(open(os.path.join(EXP041, "result.json"), encoding="utf-8"))["notes"]["선택"]["winner"]
spec, res = rule83_flags(ns.X); r83 = spec | res
ns.M = np.column_stack([ns.M, r83.astype(float)])
fo = oof_fires(ns, lambda: Add83(ns.mk(W, ("E2",))), seed=0)
a = fo["a"]; thr = float(best_f2_threshold(y, a)); et_on = a > thr
final = fo["main"] | fo["E2"] | r83
# 고장을 ET가 놓쳤을 때 누가 살렸나
why = np.array(["" for _ in y], dtype=object)
for i in np.flatnonzero(~nor & ~et_on):
    why[i] = "박자 z·결합" if fo["main"][i] else ("E2" if fo["E2"][i] else ("8.3k 규칙" if r83[i] else "못 살림"))
COL = {"박자 z·결합": "#1f9d55", "E2": "#7a3db8", "8.3k 규칙": "#000000", "못 살림": "#d62728"}
rng = np.random.RandomState(1); yj = np.where(nor, 0, 1) + rng.uniform(-.28, .28, len(y))
fig, ax = plt.subplots(figsize=(17, 8))
ax.axvspan(thr, a.max() + .5, color="#fbe3d6", alpha=.6, zorder=0); ax.axvspan(a.min() - .5, thr, color="#dbe8f7", alpha=.6, zorder=0)
ax.axvline(thr, color="k", ls="--", lw=1.4)
ax.scatter(a[nor & ~et_on], yj[nor & ~et_on], color=NOR, s=40, marker="s", lw=0, alpha=.75, label=f"정상 · ET도 정상 ({int((nor & ~et_on).sum())})")
ax.scatter(a[nor & et_on], yj[nor & et_on], color=NOR, s=90, marker="s", edgecolor="#d62728", lw=2, label=f"정상 · ET 오경보 ({int((nor & et_on).sum())})")
ax.scatter(a[~nor & et_on], yj[~nor & et_on], color=ABN, s=55, lw=0, alpha=.9, label=f"고장 · ET도 고장 ({int((~nor & et_on).sum())})")
for k, c in COL.items():
    m = why == k
    if m.any(): ax.scatter(a[m], yj[m], color=ABN, s=150, edgecolor=c, lw=2.6, label=f"고장 · ET 놓침 → {k}가 살림 ({int(m.sum())})")
# 라벨: ET가 틀린 파일 전부, 위/아래 번갈아
wrong = sorted(np.flatnonzero((nor & et_on) | (~nor & ~et_on)), key=lambda i: a[i])
offs = [(0, 26), (0, 46), (0, 66), (0, 86)]
for k, i in enumerate(wrong):
    up = not nor[i]; dy = (offs[k % 4][1] if up else -offs[k % 4][1])
    tag = short(names[i]) + ("" if nor[i] else f" ({why[i]})") + (" · G도 오경보" if nor[i] and final[i] else "")
    ax.annotate(tag, (a[i], yj[i]), xytext=(0, dy), textcoords="offset points", ha="center", fontsize=9.5, weight="bold", color=ABN if up else NOR,
                arrowprops=dict(arrowstyle="-", color="#888", lw=.6))
ax.set_yticks([0, 1]); ax.set_yticklabels(["정상 120", "고장 30"], fontsize=12); ax.set_ylim(-1.1, 2.2 if False else 2.1); ax.set_xlim(a.min() - .4, a.max() + .4)
ax.set_xlabel("ET 점수 (logit, 폴드 밖 OOF seed 0)"); ax.grid(alpha=.3, axis="x"); ax.legend(fontsize=10, loc="center left", bbox_to_anchor=(0, .5))
nfp, nfn = int((nor & et_on).sum()), int((~nor & ~et_on).sum())
ax.set_title(f"ET만 판정 (문턱 {thr:.2f}, F2 최대·사후): 고장 놓침 {nfn} · 정상 오경보 {nfp}  →  G(박자 z · E2 · 8.3k 추가): 놓침 {int((~nor & ~final).sum())} · 오경보 {int((nor & final).sum())}", fontsize=12, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "분포_ET판정.png"), dpi=125)
print("ET 문턱", round(thr, 3), "놓침", [(short(names[i]), why[i]) for i in np.flatnonzero(~nor & ~et_on)], "ET 오경보", [short(names[i]) for i in np.flatnonzero(nor & et_on)], "G 오경보", [short(names[i]) for i in np.flatnonzero(nor & final)], "G 놓침", [short(names[i]) for i in np.flatnonzero(~nor & ~final)])
