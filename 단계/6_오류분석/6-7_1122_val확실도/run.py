# %% [markdown]
# 6-7 (11-22) val 40개 판정이 얼마나 확실한가 — 기준 구성 J-축소(EXP-050)
# - 전체 모델(train 150) 판정 + 중첩 CV 바깥 모델 25개(seed 0–4 × 바깥 5폴드, 각 120개 학습)가 val을 고장으로 본 횟수
# - 확실도 분류(실행 전 고정): 25개 모두 같은 판정 = 확실 / 1–24 = 흔들림. 규칙(8.3k · E2)으로만 고장인 파일은 따로 표시
# 실행: QE_FULL=1 conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedGroupKFold
from src.exp import md_table
from src.evaluate import FULL
from src.exp2util import NS, rule83_flags, NOR, ABN
from src.models2 import TwoStack
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
assert FULL, "QE_FULL=1로 실행"
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
INK, MUTED = "#1d2430", "#5b6575"
ALPHA = 3.0; RING = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]
ns = NS(); y, g, n = ns.y, ns.g, ns.n; nor = y == 0; nv = [v[-3:] for v in ns.nv]; F0 = list(ns.FIXED)
M = np.hstack([ns.T[F0].values, ns.T[["L1_S"]].values, ns.L]); Mv = np.hstack([ns.Tv[F0].values, ns.Tv[["L1_S"]].values, ns.Lv])
Fn = np.column_stack([M, np.arange(n, dtype=float)])
sv, rv = rule83_flags(ns.Xv); r83v = sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}


class Shrink:
    """EXP-050 J-축소와 같은 축소 + TwoStack(J)"""
    def __init__(self): self.ri, self.i1 = [F0.index(c) for c in RING], F0.index("H1_nimp")
    def _tf(self, F):
        F = np.array(F, float); k = np.rint(np.expm1(F[:, self.i1])); has = k > 0
        F[np.ix_(has, self.ri)] = (F[np.ix_(has, self.ri)] * k[has, None] + ALPHA * self.p0_) / (k[has, None] + ALPHA); return F
    def fit(self, F, y):
        F = np.asarray(F, float); k = np.rint(np.expm1(F[:, self.i1])); m = (y == 0) & (k > 0)
        self.p0_ = (F[np.ix_(m, self.ri)] * k[m, None]).sum(0) / k[m].sum()
        self.m_ = TwoStack("lin", ("E2",), fixed=tuple(F0), nf=len(F0), df=ns.df, peak=tuple(ns.PEAK), extra=(), use_ae=True).fit(self._tf(F), y); return self


def val_call(tr):
    sh = Shrink().fit(Fn[tr], y[tr]); ts = sh.m_; fv = ts.fires_raw(sh._tf(Mv), RAW)
    return fv["score"] - ts.tc_, fv["E2"]


splits = [tr for s in range(5) for tr, _ in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
out = Parallel(n_jobs=13)(delayed(val_call)(tr) for tr in splits + [np.arange(n)])
MG = np.array([o[0] for o in out[:-1]]); E2 = np.array([o[1] for o in out[:-1]]); mF, e2F = out[-1]
comb = MG >= 0; fin25 = comb | E2 | r83v[None, :]; votes = fin25.sum(0); cvotes = comb.sum(0)
finF = (mF >= 0) | e2F | r83v

def kind(k):
    if r83v[k] and not ((mF[k] >= 0) or e2F[k]): return "규칙(8.3k)만 고장"
    if votes[k] == 25: return "확실 고장"
    if votes[k] == 0: return "확실 정상"
    return "흔들림"
D = pd.DataFrame({"val": nv, "전체 모델": np.where(finF, "고장", "정상"), "여유(전체)": np.round(mF, 2), "바깥 모델 25개 중 고장": votes,
                  "그중 결합으로 고장": cvotes, "여유 범위(25개)": [f"{MG[:, k].min():+.2f} – {MG[:, k].max():+.2f}" for k in range(40)],
                  "규칙": np.where(r83v, "8.3k", np.where(e2F, "E2", "")), "분류": [kind(k) for k in range(40)]})
D = D.iloc[np.argsort(-votes * 100 - mF)]
S = D["분류"].value_counts()
open(os.path.join(HERE, "표_val확실도.md"), "w", encoding="utf-8").write(
    "# 6-7 val 40 확실도 (J-축소)\n\n"
    f"전체 모델 val 고장 {int(finF.sum())}/40. 바깥 모델 25개(seed 0–4 × 5폴드, 각 120개 학습)의 val 고장 수: {int(fin25.sum(1).min())}–{int(fin25.sum(1).max())}개 (중앙 {int(np.median(fin25.sum(1)))})\n\n"
    + " · ".join(f"{k} {v}" for k, v in S.items()) + "\n\n" + md_table(D) + "\n")

# %% 그림: 파일별 25개 모델의 여유 (상자) + 전체 모델 여유 (점)
o = np.argsort(np.median(MG, 0)); fig, ax = plt.subplots(figsize=(16, 5.4))
bp = ax.boxplot([MG[:, k] for k in o], positions=range(40), widths=.6, patch_artist=True, showfliers=False)
for k, b in zip(o, bp["boxes"]):
    kk = kind(k); b.set_facecolor({"확실 고장": "#f6c9b4", "확실 정상": "#cfe0f6", "흔들림": "#e7dcf6", "규칙(8.3k)만 고장": "#f6c9b4"}[kk]); b.set_edgecolor(MUTED)
ax.scatter(range(40), mF[o], marker="D", s=28, c=np.where(finF[o], ABN, NOR), edgecolor="k", lw=.5, zorder=4, label="전체 모델 여유 (색 = 최종 판정)")
for j, k in enumerate(o):
    if kind(k) == "흔들림": ax.text(j, MG[:, k].max() + .3, f"{votes[k]}/25", ha="center", fontsize=7.5, color=INK, weight="bold")
    if r83v[k]: ax.text(j, MG[:, k].min() - .7, "8.3k", ha="center", fontsize=7, color=ABN)
ax.axhline(0, color=INK, lw=1); ax.set_xticks(range(40)); ax.set_xticklabels([nv[k] for k in o], rotation=90, fontsize=8)
ax.set_ylabel("여유 (결합 점수 − 문턱)", color=MUTED); ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8.5, loc="upper left")
ax.set_title("val 40 — 바깥 모델 25개의 여유(상자: 사분위, 수염: 범위 안 1.5 IQR) · 숫자 = 흔들리는 파일의 25개 중 최종 고장 수. 상자 색: 주황 확실 고장 · 파랑 확실 정상 · 보라 흔들림", loc="left", fontsize=9.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "val_확실도.png"), dpi=150); plt.close(fig)
print("done", dict(S), flush=True)
