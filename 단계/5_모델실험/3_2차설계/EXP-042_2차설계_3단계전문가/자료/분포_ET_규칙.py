# EXP-042 해석용: ET 점수(가로) × 규칙 값(세로) 2차원. 세로 = 박자 z / H18(E2 값) / H16(E1 값). run.py는 고치지 않음
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, matplotlib.pyplot as plt
from src.exp2util import *
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(HERE, "figs")
ns = NS(); y, names, nor = ns.y, ns.names, ns.nor
W = json.load(open(os.path.join(find_exp("EXP-041"), "result.json"), encoding="utf-8"))["notes"]["선택"]["winner"]
fo = oof_fires(ns, ns.mk(W, ("E1", "E2")), seed=0); a = fo["a"]; z = fo["z"]
h18 = ns.T["H18_hfl"].values; h1 = ns.T["H1_nimp"].values
h16 = PeakRef(ns.df).fit(ns.L[nor]).transform(ns.L)[:, 0]          # 그림용: 전체 정상 기준 (문턱 검증 아님)
q18, q16 = np.percentile(h18[nor], 99), np.percentile(h16[nor], 99)
imp = h1 >= np.log(2) - 1e-9
panels = [("박자 z (S 강건 z)", np.clip(z, -3, 20), None, "박자 z"),
          ("H18 하중 대비 고주파 (E2 값)", h18, q18, "E2: H18 > 정상 99% & 충격≥1"),
          ("H16 가장 솟은 좁은 봉우리 (E1 값)", h16, q16, "E1: H16 > 정상 99%")]
hi = [short(names[i]) for i in np.flatnonzero(~nor)]
fig, axs = plt.subplots(1, 3, figsize=(24, 8.5))
for ax, (lab, v, q, nm) in zip(axs, panels):
    ax.scatter(a[nor], v[nor], color=NOR, s=34, marker="s", lw=0, alpha=.7, label="정상")
    ax.scatter(a[~nor & ~imp] if lab.startswith("H18") else a[~nor], v[~nor & ~imp] if lab.startswith("H18") else v[~nor], color=ABN, s=50, lw=0, label="고장")
    if lab.startswith("H18"): ax.scatter(a[~nor & imp], v[~nor & imp], color=ABN, s=70, marker="^", lw=0, label="고장 (충격≥1)"); ax.scatter(a[nor & imp], v[nor & imp], color="#1b4f9c", s=34, marker="^", lw=0, label="정상 (충격≥1)")
    if q is not None: ax.axhline(q, color="#7a3db8", ls="--", lw=1.3, label=f"정상 99% = {q:.2f}")
    ax.axvline(-.115, color="k", ls=":", lw=1, label="ET 문턱(사후) −0.12")
    for i in np.flatnonzero(~nor):                      # 고장 라벨: 문턱·ET 아래쪽(놓치기 쉬운) 것만 + 규칙으로 올라온 것
        if a[i] < 1.0 or (q is not None and v[i] > q):
            ax.annotate(short(names[i]), (a[i], v[i]), xytext=(5, 5), textcoords="offset points", fontsize=9, color=ABN, weight="bold")
    ax.set_xlabel("ET 점수 (logit, 폴드 밖 OOF)"); ax.set_ylabel(lab); ax.set_title(nm, fontsize=11); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("ET 점수 × 규칙 값 — 세로로 퍼지는가 (ET OOF seed 0, 규칙 문턱은 그림용으로 전체 정상 99%)", x=.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(os.path.join(OUT, "분포_ET_규칙.png"), dpi=120); plt.close(fig)
lo = [i for i in np.flatnonzero(~nor) if a[i] < -.115]
print("ET 문턱 아래 고장:", [(short(names[i]), round(float(a[i]), 2), round(float(h16[i]), 2), round(float(h18[i]), 2), int(imp[i]), round(float(z[i]), 2)) for i in lo])
print("H16 q99", q16, "H18 q99", q18)
print("H18>q99 & 충격≥1 : 고장", [short(names[i]) for i in np.flatnonzero(~nor & (h18 > q18) & imp)], "정상", [short(names[i]) for i in np.flatnonzero(nor & (h18 > q18) & imp)])
print("H16>q99: 고장", [short(names[i]) for i in np.flatnonzero(~nor & (h16 > q16))], "정상", [short(names[i]) for i in np.flatnonzero(nor & (h16 > q16))])
