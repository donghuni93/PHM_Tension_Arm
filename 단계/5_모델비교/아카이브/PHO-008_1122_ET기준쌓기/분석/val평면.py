# PHO-008 보조: 구성 6개(J · +ext · +shape · +mspc · +knn · 넷 다)를 train 150 전체로 학습해 val 40개를 ET × AE 평면에 찍는다
# 배경 = 그 구성의 결합 판단 영역(박자 · 새 전문가 z = 0 단면). 점 색 = 최종 판단(새 전문가 · 박자 · 규칙 포함). QE_FULL=1 (ET 300)
# val 원점수: AE · ext(ET on ext150) · shape(모양 AE) · mspc는 train 150 전체로 학습한 모델로 계산, knn은 H로 직접 (11-22 척도 수정 후)
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.evaluate import FULL
from src.exp import make_et, CACHE
from src.features import feature_table_ext
from src.exp2util import NS, Add83, rule83_flags
from src.models2 import TwoStack, lg
from src.experts import LP as LP_TR, E, segment_logpsd
from src.final import _fit_ae, _ae_score
assert FULL, "QE_FULL=1 로 실행"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED = "#1d2430", "#5b6575"; C_N, C_A = "#2a78d6", "#e0622a"
ns = NS(); y = ns.y; nor = y == 0; nv = [n[-3:] for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83v = (sv | rv)
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)])
# val 원점수
p = os.path.join(CACHE, "feat_ext_val.csv")
Ev = (pd.read_csv(p) if os.path.exists(p) else feature_table_ext(ns.Xv).pipe(lambda d: (d.to_csv(p, index=False), d)[1])).values
_, LPv = segment_logpsd(ns.Xv)
raw = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
pe = np.clip(make_et(seed=0)().fit(E, y).predict_proba(Ev)[:, 1], 1 / 600, 1 - 1 / 600); raw["ext"] = np.log(pe / (1 - pe))
sh = lambda L: L - L.mean(-1, keepdims=True)
S = sh(LP_TR[nor]).reshape(-1, LP_TR.shape[-1]); mu, sd = S.mean(0), S.std(0) + 1e-8
mlp = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3, max_iter=60, alpha=0.0, random_state=0).fit((S - mu) / sd, (S - mu) / sd)
Zv = (sh(LPv) - mu) / sd; Rv = mlp.predict(Zv.reshape(-1, Zv.shape[-1])).reshape(Zv.shape); raw["shape"] = np.log10(((Rv - Zv) ** 2).mean((1, 2)))
N = E[nor]; sc = StandardScaler().fit(N); Zn = sc.transform(N); pca = PCA(0.9, random_state=0).fit(Zn)
def stat(Z): t = pca.transform(Z); return (t ** 2 / pca.explained_variance_).sum(1), ((Z - pca.inverse_transform(t)) ** 2).sum(1)
t2n, spn = stat(Zn); t2, sp = stat(sc.transform(Ev)); raw["mspc"] = np.log10(t2 / np.median(t2n) + sp / np.median(spn))
CFG = [("J", ()), ("J + ext", ("ext",)), ("J + shape", ("shape",)), ("J + mspc", ("mspc",)), ("J + knn", ("knn",)), ("J + 넷 다", ("ext", "shape", "mspc", "knn"))]
BORDER = ["027", "025", "018", "021", "012", "036", "009", "001", "031", "024", "035", "002"]
R = {}
for nm, ex in CFG:
    m = Add83(lambda ex=ex: TwoStack("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), extra=ex)).fit(M, y).m_
    f = m.fires_raw(ns.Mv, raw); fin = f["main"] | f["E2"] | r83v; R[nm] = (m, f, fin)
J_fin = R["J"][2]
allx = np.concatenate([R[k][1]["a"] for k in R]); ally = np.concatenate([R[k][1]["ae"] for k in R])
x0, x1, y0, y1 = allx.min() - .4, allx.max() + .4, min(ally.min(), -1) - .4, ally.max() + .4
fig, axs = plt.subplots(2, 3, figsize=(19, 11.5), sharex=True, sharey=True)
xx, yy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
for ax, (nm, ex) in zip(axs.ravel(), CFG):
    m, f, fin = R[nm]
    D = (m._comb_score(xx.ravel(), np.zeros(xx.size), yy.ravel()) >= m.tc_).reshape(xx.shape).astype(int)
    ax.contourf(xx, yy, D, levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0); ax.contour(xx, yy, D, levels=[.5], colors="#333", linewidths=1.1, linestyles="--")
    ax.scatter(f["a"], f["ae"], marker="^", s=80, c=np.where(fin, C_A, C_N), edgecolor="k", lw=.7, zorder=3)
    chg = fin != J_fin
    ax.scatter(f["a"][chg], f["ae"][chg], marker="o", s=260, facecolor="none", edgecolor="#1f9d55", lw=2, zorder=4)
    for i, t in enumerate(nv):
        if t in BORDER or chg[i]: ax.annotate(t, (f["a"][i], f["ae"][i]), xytext=(6, 5), textcoords="offset points", fontsize=8.5, weight="bold" if chg[i] else "normal", zorder=5)
    w = [nv[i] for i in np.flatnonzero(chg)]
    ax.set_title(f"{nm} — val 고장 {int(fin.sum())}/40" + (f" · J와 다름: {' '.join(w)}" if w else ""), fontsize=11.5, color=INK, loc="left")
    ax.grid(alpha=.25); ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
for ax in axs[1]: ax.set_xlabel("ET 점수 (logit)", color=MUTED)
for ax in axs[:, 0]: ax.set_ylabel("AE z (학습 정상 기준)", color=MUTED)
fig.legend(handles=[Patch(color="#d6e6fa", label="정상 판단 영역"), Patch(color="#fbe0d0", label="고장 판단 영역"),
                    Line2D([], [], marker="^", ls="", mfc=C_N, mec="k", ms=9, label="val → 정상"), Line2D([], [], marker="^", ls="", mfc=C_A, mec="k", ms=9, label="val → 고장"),
                    Line2D([], [], marker="o", ls="", mfc="none", mec="#1f9d55", mew=2, ms=13, label="J와 판단이 다른 파일")], ncol=5, frameon=False, loc="upper right", fontsize=10)
fig.suptitle("PHO-008 구성별 val 40개 판단 — ET × AE 평면 (train 150 전체 학습, ET 300). 배경 = 박자 · 새 전문가 z = 0 단면이라 점마다 실제 경계와 다를 수 있음",
             fontsize=11, color=MUTED, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(HERE, "val평면_구성별.png"), dpi=120); plt.close(fig)
for nm in R: print(nm, int(R[nm][2].sum()), [nv[i] for i in np.flatnonzero(R[nm][2] != J_fin)], "계수", np.round(R[nm][0].c_.coef_[0], 2))
