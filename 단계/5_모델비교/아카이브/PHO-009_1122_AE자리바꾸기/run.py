# %% [markdown]
# PHO-009 (휴대폰 세션, 11-22) AE 자리 바꾸기: J의 '정상에서 벗어난 정도' 입력(AE)을 val 측정 조건에 덜 흔들리는 것으로 교체
# - 동기: AE는 학습 정상 기준이라 val 배경(3–5 kHz +10.9 dB) 때문에 val 전체가 위로 밀림(val AE z 중앙 +1.75). 라벨 없이 본 val 이동: mspc +0.58 · ext +0.69 (기대 범위), AE · 모양 AE · knn은 +1.7
# - 구성: J(ET + 박자 + AE) · J′-mspc(ET + 박자 + 정상 PCA) · J′-self(ET + 박자 + 자기기준 AE) · J′-mspc+self
#   mspc = ext150을 학습 정상으로 표준화 → PCA(분산 90%) → T²(주성분 안 거리) + SPE(주성분 밖 잔차) (src.candidates.mspc_ext)
#   self = 구간 log PSD − 그 파일의 구간 중앙값 스펙트럼을 학습 정상으로 AE 학습 (src.experts.ae_self_mean) → 파일 전체 바탕은 상쇄
# - 평가: 중첩 CV seed 2 · ET 300(QE_FULL=1, seeds 명시) + val 판정 · val 입력 이동. 규칙(E2 · 8.3k) · 2단 로지스틱은 J와 같음
# 실행: QE_FULL=1 python run.py
# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import FULL
from src.exp2util import NS, Add83, rule83_flags, NOR, ABN
from src.models2 import TwoStack
from src.experts import LP as LP_TR, E, segment_logpsd
from src.final import _fit_ae, _ae_score
from src.features import feature_table_ext
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore"); assert FULL, "QE_FULL=1 로 실행"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); INK, MUTED = "#1d2430", "#5b6575"; C_N, C_A = "#2a78d6", "#e0622a"
ex = Experiment(__file__, "PHO-009", "AE 자리 바꾸기 (정상 PCA · 자기기준 AE)", ["H3"],
                member="이동훈", model="ET + 박자 + {AE | 정상 PCA | 자기기준 AE} → 로지스틱 OR E2 OR 8.3k",
                changes="J의 AE 입력을 val 측정 조건에 덜 흔들리는 정상 비교 모델(ext150 정상 PCA · 자기기준 AE)로 교체",
                key_changes=["AE 대신 mspc(ext150 정상 PCA T² + SPE)", "AE 대신 자기기준 AE(파일 자신의 중앙값 스펙트럼 대비)", "둘 다(AE 없이)", "중첩 CV seed 2 · ET 300"])
ns = NS(); y, g, names = ns.y, ns.g, ns.names; nor = y == 0; nv = [n[-3:] for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83v = sv | rv
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), (spec | res).astype(float)])
CFG = [("J (AE)", (), True), ("J′-mspc", ("mspc",), False), ("J′-self", ("self",), False), ("J′-mspc+self", ("mspc", "self"), False)]
mk = lambda extra, ua: (lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), extra=extra, use_ae=ua)))
LOG = []
def say(*a):
    s = " ".join(str(v) for v in a); print(s, flush=True); LOG.append(s)

# %% 중첩 CV (seed 2 · ET 300)
for nm, extra, ua in CFG:
    ex.record("중첩: " + nm, nested_cv(M, y, g, mk(extra, ua), seeds=range(2), n_jobs=-1), names, main=(nm == "J′-mspc"), mid=(nm == "J (AE)"))

# %% val 원점수 (train 150 전체로 학습한 모델)
p = os.path.join(CACHE, "feat_ext_val.csv"); Ev = (pd.read_csv(p) if os.path.exists(p) else feature_table_ext(ns.Xv).pipe(lambda d: (d.to_csv(p, index=False), d)[1])).values
_, LPv = segment_logpsd(ns.Xv)
raw = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
N = E[nor]; sc = StandardScaler().fit(N); Zn = sc.transform(N); pca = PCA(0.9, random_state=0).fit(Zn)
def stat(Z): t = pca.transform(Z); return (t ** 2 / pca.explained_variance_).sum(1), ((Z - pca.inverse_transform(t)) ** 2).sum(1)
t2n, spn = stat(Zn); t2, sp = stat(sc.transform(Ev)); raw["mspc"] = np.log10(t2 / np.median(t2n) + sp / np.median(spn))
say(f"정상 PCA: 학습 정상 120개 · ext150 {E.shape[1]}열 → 주성분 {pca.n_components_}개로 분산 90% (전체 학습 기준)")
selfL = lambda L: L - np.median(L, axis=1, keepdims=True)
S = selfL(LP_TR[nor]).reshape(-1, LP_TR.shape[-1]); mu, sd = S.mean(0), S.std(0) + 1e-8
mlp = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3, max_iter=60, alpha=0.0, random_state=0).fit((S - mu) / sd, (S - mu) / sd)
Zv = (selfL(LPv) - mu) / sd; Rv = mlp.predict(Zv.reshape(-1, Zv.shape[-1])).reshape(Zv.shape); raw["self"] = np.log10(((Rv - Zv) ** 2).mean((1, 2)))
R = {}
for nm, extra, ua in CFG:
    m = mk(extra, ua)().fit(M, y).m_; f = m.fires_raw(ns.Mv, raw); fin = f["main"] | f["E2"] | r83v; R[nm] = (m, f, fin)

# %% val 이동 (라벨 없음): 바꾼 입력의 val 중앙 · train 정상 95% 위 비율 (이동 없고 고장 20–25%면 약 20–25%)
say("\n입력 | train 정상(OOF) 중앙 · train 고장 중앙 · val 중앙 | val 중 train 정상 95% 위")
shift = {}
for nm, col in (("AE", ("J (AE)", 2)), ("mspc", ("J′-mspc", 2)), ("self", ("J′-self", 2))):
    m, f, _ = R[col[0]]; Zt = m.Z_[:, col[1]]; v = (f["ae"] if nm == "AE" else f["ex"][:, 0])
    shift[nm] = dict(nor=round(float(np.median(Zt[nor])), 2), abn=round(float(np.median(Zt[~nor])), 2), val=round(float(np.median(v)), 2), over95=round(float(np.mean(v > np.percentile(Zt[nor], 95))), 2))
    say(f"{nm:5s} | {shift[nm]['nor']:+.2f} · {shift[nm]['abn']:+.2f} · {shift[nm]['val']:+.2f} | {shift[nm]['over95']:.0%}")
ex.note("val 이동", shift)
J_fin = R["J (AE)"][2]; vres = {}
say("\n구성 | val 고장 판정 | J와 다른 파일 | 2단 계수")
for nm, _, _ in CFG:
    m, f, fin = R[nm]; d = [nv[i] for i in np.flatnonzero(fin != J_fin)]
    vres[nm] = dict(n=int(fin.sum()), faults=[nv[i] for i in np.flatnonzero(fin)], diff=d, coef=np.round(m.c_.coef_[0], 2).tolist())
    say(f"{nm} | {fin.sum()}/40 | {' '.join(d) or '—'} | {vres[nm]['coef']}")
ex.note("val 판정", vres)

# %% 핵심 그림: 결정축 관점 (모든 파일에 정확한 경계 = 세로선 0), 4구성
fig, axs = plt.subplots(2, 2, figsize=(18, 11))
BORDER = ["027", "025", "018", "021", "012", "036", "009", "001", "031", "024"]
for ax, (nm, extra, ua) in zip(axs.ravel(), CFG):
    m, f, fin = R[nm]; w, b, t = m.c_.coef_[0], m.c_.intercept_[0], m.tc_
    Zv_ = np.column_stack([f["a"], f["z"]] + ([f["ae"]] if ua else []) + ([f["ex"]] if len(extra) else []))
    xt, xv = m.Z_ @ w + b - t, Zv_ @ w + b - t
    u = w / np.linalg.norm(w); P = np.eye(len(w)) - np.outer(u, u); Rt = m.Z_ @ P; mu_ = Rt.mean(0); v1 = np.linalg.svd(Rt - mu_, full_matrices=False)[2][0]
    yt, yv = (Rt - mu_) @ v1, (Zv_ @ P - mu_) @ v1
    x0, x1 = min(xt.min(), xv.min()) - .5, max(xt.max(), xv.max()) + .5
    ax.axvspan(x0, 0, color="#d6e6fa", zorder=0); ax.axvspan(0, x1, color="#fbe0d0", zorder=0); ax.axvline(0, color=INK, lw=1.2, ls="--")
    ax.scatter(xt[nor], yt[nor], s=16, marker="s", color="#9db7d8", lw=0); ax.scatter(xt[~nor], yt[~nor], s=20, color="#f1b393", lw=0)
    ax.scatter(xv, yv, marker="^", s=80, c=np.where(fin, C_A, C_N), edgecolor="k", lw=.7, zorder=4)
    for i, t_ in enumerate(nv):
        if t_ in BORDER or fin[i] != J_fin[i]: ax.annotate(t_, (xv[i], yv[i]), xytext=(5, 5), textcoords="offset points", fontsize=8.5, weight="bold" if fin[i] != J_fin[i] else "normal", zorder=5)
    r = ex.res["configs"]["중첩: " + nm]; ax.set_xlim(x0, x1); ax.grid(alpha=.25)
    ax.set_title(f"{nm} — 중첩 F2 {r['f2']:.3f} · 오경보율 {r['fpr']:.3f} · val 고장 {int(fin.sum())}/40", fontsize=11.5, color=INK, loc="left")
    ax.set_xlabel("결정값 (0 = 경계)", color=MUTED); ax.set_ylabel("경계와 나란한 방향 (PCA 1축)", color=MUTED)
fig.legend(handles=[Line2D([], [], marker="s", ls="", color="#9db7d8", label="train 정상 (OOF)"), Line2D([], [], marker="o", ls="", color="#f1b393", label="train 고장 (OOF)"),
                    Line2D([], [], marker="^", ls="", mfc=C_N, mec="k", ms=9, label="val → 정상"), Line2D([], [], marker="^", ls="", mfc=C_A, mec="k", ms=9, label="val → 고장"),
                    Line2D([], [], ls="", label="굵은 글씨 = J와 판정이 다른 val")], ncol=5, frameon=False, loc="upper right", fontsize=10)
fig.suptitle("AE 자리 바꾸기 — 결정축 관점 (train 150 전체 학습, 규칙 E2 · 8.3k는 점 색에 포함)", fontsize=11, color=MUTED, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .95)); ex.save_fig(fig, "결정축_4구성", "가로 = 결정값(모든 입력 반영, 0 = 경계), 세로 = 경계와 나란한 방향. 왼쪽 위 J, 나머지 AE 자리 교체", key=True)
open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8").write("\n".join(LOG)); print("done")
