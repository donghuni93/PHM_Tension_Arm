# %% [markdown]
# EXP-049 정상 PCA(mspc) full 재확인: J · J′-mspc · G를 같은 분할, seed 5 · ET 300으로 비교
# - 가설: H3 (게인 · 측정 조건 차이) — 정상 PCA는 val 측정 조건에 덜 흔들린다(PHO-009: val 중 정상 95% 위 AE 40% → mspc 17%)
# - 바꾼 것: J의 AE z 자리에 ext150 정상 PCA(T² + SPE) z. 비교로 AE를 아예 뺀 G
# - 왜: PHO-009(seed 2)는 J′-mspc 0.974 > J 0.968이었지만, full 수치는 J 0.977 > G 0.959(EXP-043/044)로 방향이 반대. mspc 계수 −0.01
# - 이상치 정상 3개 학습 제외(기본 설정)는 쓰지 않는다: EXP-044 · PHO-009와 같은 조건에서 재확인하는 실험이라서
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
from src.experts import LP as LP_TR, E, segment_logpsd
from src.final import _fit_ae, _ae_score
from src.features import feature_table_ext
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore"); assert FULL, "QE_FULL=1 로 실행"
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
ex = Experiment(__file__, "EXP-049", "정상 PCA(mspc) full 재확인 — J · J′-mspc · G", ["H3"],
                member="이동훈", model="ET + 박자 + {AE | 정상 PCA | 없음} → 로지스틱 OR E2 OR 8.3k",
                changes="J의 AE z 자리에 ext150 정상 PCA(T² + SPE) z, 비교로 AE 없는 G. seed 5 · ET 300, 같은 분할 (PHO-009 재확인)",
                key_changes=["AE z → 정상 PCA z (J′-mspc)", "AE 없음 (G)", "full 모드 seed 5 · ET 300, 세 구성 같은 바깥 분할", "이상치 학습 제외 안 함(EXP-044 · PHO-009와 같은 조건)"])
ns = NS(); y, g, names = ns.y, ns.g, np.array(ns.names); nor = y == 0; nv = [n[-3:] for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83, r83v = spec | res, sv | rv
M = np.column_stack([ns.M, np.arange(ns.n, dtype=float), r83.astype(float)])
CFG = [("J (AE)", (), True), ("J′-mspc", ("mspc",), False), ("G (AE 없음)", (), False)]
mk = lambda extra, ua: (lambda: Add83(lambda: TwoStack("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK), extra=extra, use_ae=ua)))
OUT = [names.tolist().index(n) for n in OUTLIER_NORMALS]
short = lambda n: n.replace("nor_", "N").replace("abn_", "A")

# %% 중첩 CV (full: seed 5 · ET 300, 세 구성 같은 분할)
RES = {}
for nm, extra, ua in CFG:
    RES[nm] = nested_cv(M, y, g, mk(extra, ua), n_jobs=-1)
    ex.record(nm, RES[nm], names, main=(nm == "J′-mspc"), mid=(nm != "J′-mspc"))

# %% 전체 학습 모델 (그림 · val용) + val 원점수
p = os.path.join(CACHE, "feat_ext_val.csv"); Ev = (pd.read_csv(p) if os.path.exists(p) else feature_table_ext(ns.Xv).pipe(lambda d: (d.to_csv(p, index=False), d)[1])).values
_, LPv = segment_logpsd(ns.Xv)
raw = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
N_ = E[nor]; sc = StandardScaler().fit(N_); Zn = sc.transform(N_); pca = PCA(0.9, random_state=0).fit(Zn)
def stat(Z): t = pca.transform(Z); return (t ** 2 / pca.explained_variance_).sum(1), ((Z - pca.inverse_transform(t)) ** 2).sum(1)
t2n, spn = stat(Zn); t2, sp = stat(sc.transform(Ev)); raw["mspc"] = np.log10(t2 / np.median(t2n) + sp / np.median(spn))
FIT = {}
for nm, extra, ua in CFG:
    m = mk(extra, ua)().fit(M, y).m_
    ft = m.fires(M[:, :-1]); fv = m.fires_raw(ns.Mv, raw)
    comb = m.c_.decision_function(m.Z_) >= m.tc_                    # 그림의 점 위치(안쪽 OOF 입력)와 같은 입력으로 판정
    FIT[nm] = dict(m=m, comb=comb, e2=ft["E2"], fin=comb | ft["E2"] | r83,
                   vcomb=fv["main"], ve2=fv["E2"], vfin=fv["main"] | fv["E2"] | r83v, fv=fv)
coef = {nm: dict(zip(["ET", "박자"] + (["AE"] if ua else []) + list(extra), np.round(FIT[nm]["m"].c_.coef_[0], 3).tolist())) for nm, extra, ua in CFG}
ex.note("결합 계수 (전체 학습)", coef)
AEZ = FIT["J (AE)"]["m"].Z_[:, 2]                 # G 그림의 세로축(참고용, G는 안 씀)

# %% 핵심 그림: logistic_plane — 배경 = 결합 판단, 세로선 = ET 단독 판단, 점 = 최종 판정
def plane(ax, nm, val=False):
    F = FIT[nm]; m = F["m"]; c, b, tc = m.c_.coef_[0], m.c_.intercept_[0], m.tc_; Z = m.Z_
    has3 = Z.shape[1] >= 3; t_et = float(best_f2_threshold(y, Z[:, 0]))
    if val:
        fv = F["fv"]; et, zb = fv["a"], fv["z"]
        y3 = fv["ae"] if nm.startswith("J (") else fv["ex"][:, 0] if has3 else FIT["J (AE)"]["fv"]["ae"]
        fin, comb = F["vfin"], F["vcomb"]; lab = None
    else:
        et, zb = Z[:, 0], Z[:, 1]; y3 = Z[:, 2] if has3 else AEZ; fin, comb = F["fin"], F["comb"]; lab = y
    x = et + c[1] / c[0] * zb
    xs = np.r_[x, t_et]; x0, x1 = xs.min() - .6, xs.max() + .6; y0, y1 = y3.min() - .6, y3.max() + .6
    gx, gy = np.meshgrid(np.linspace(x0, x1, 300), np.linspace(y0, y1, 300))
    S = c[0] * gx + (c[2] * gy if has3 else 0) + b
    ax.contourf(gx, gy, (S >= tc).astype(float), levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    show = np.ones(len(x), bool)
    if lab is not None: show &= ~((lab == 1) & ~comb & fin)            # 규칙(E2 · 8.3k)만으로 잡힌 고장은 뺀다 (t-SNE에는 모두)
    mv = np.abs(x - et) >= .3
    for i in np.flatnonzero(show & mv): ax.plot([et[i], x[i]], [y3[i], y3[i]], ls=":", color="#888", lw=.8, zorder=1)
    ax.scatter(et[show & mv], y3[show & mv], s=10, facecolor="none", edgecolor="#888", lw=.6, zorder=1)
    col = np.where(fin, ABN, NOR)
    if lab is None:
        ax.scatter(x, y3, marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in range(len(x)):
            if fin[i] != FIT["J (AE)"]["vfin"][i] or nv[i] in ("027", "025", "018", "021", "001"):
                ax.annotate(nv[i], (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold" if fin[i] != FIT["J (AE)"]["vfin"][i] else "normal")
    else:
        o = np.zeros(len(x), bool); o[OUT] = True
        for msk, mkr, s_ in ((show & (lab == 0) & ~o, "s", 22), (show & (lab == 1), "o", 40), (show & o, "D", 60)):
            ax.scatter(x[msk], y3[msk], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & (fin != (lab == 1))): ax.annotate(short(names[i]), (x[i], y3[i]), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2)
    ax.set_xlabel(f"ET logit + ({c[1] / c[0]:.2f})·박자 z  (점선 꼬리 = 원래 ET 위치)", color=MUTED)
    ax.set_ylabel(("AE z" if nm.startswith("J (") else "정상 PCA z" if has3 else "AE z (참고: G는 안 씀)"), color=MUTED)
    r = ex.res["configs"][nm]
    ax.set_title(f"{nm} — F2 {r['f2']:.3f} ± {r['f2_sd']:.3f} · 계수 {coef[nm]}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")

leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116")]
fig, axs = plt.subplots(1, 3, figsize=(22, 7))
for ax, (nm, _, _) in zip(axs, CFG): plane(ax, nm)
fig.legend(handles=leg, ncol=5, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합). 배경 = 결합 판단(파랑 정상 · 주황 고장), 세로 점선 = ET 단독 판단. 규칙만으로 잡힌 고장은 뺌", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "J · J′-mspc · G의 결합 경계. 세로축 = 세 번째 입력(G는 AE z를 참고로만 표시)", key=True)

# %% 오류 분석 표 (가로): 세 구성의 seed 5번 중 고장 판정 횟수
def cnt(r): return np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
C = {nm: cnt(RES[nm]) for nm, _, _ in CFG}
files = [names[i] for i in range(ns.n) if any((C[nm][i] > 0) != (y[i] == 1) for nm in C)]
extra = {f"{nm} (5번 중 고장)": {f: int(C[nm][names.tolist().index(f)]) for f in files} for nm in ("J′-mspc", "G (AE 없음)")}
ex.save_md("오류분석_표", "모델 판단 행 = J (AE). 아래 두 행 = 같은 분할에서 J′-mspc · G의 고장 판정 횟수\n\n" + error_md_table(names, y, C["J (AE)"], files, n_seeds=5, extra=extra))

# %% val 판정 (제출 파일 아님)
V = pd.DataFrame({"파일": nv, **{nm: np.where(FIT[nm]["vfin"], "고장", "정상") for nm, _, _ in CFG},
                  "ET logit": np.round(FIT["J (AE)"]["fv"]["a"], 2), "박자 z": np.round(FIT["J (AE)"]["fv"]["z"], 2),
                  "AE z": np.round(FIT["J (AE)"]["fv"]["ae"], 2), "정상 PCA z": np.round(FIT["J′-mspc"]["fv"]["ex"][:, 0], 2),
                  "E2": FIT["J (AE)"]["ve2"].astype(int), "8.3k": r83v.astype(int)})
diff = V[(V["J (AE)"] != V["J′-mspc"]) | (V["J (AE)"] != V["G (AE 없음)"])]
head = " · ".join(f"{nm} {int(FIT[nm]['vfin'].sum())}/40" for nm, _, _ in CFG)
ex.save_md("val_판정", f"val 고장 판정 수: {head}\n\n구성 사이에 판정이 다른 파일\n\n{md_table(diff)}\n\n전체\n\n{md_table(V)}")
fig, axs = plt.subplots(1, 3, figsize=(22, 7))
for ax, (nm, _, _) in zip(axs, CFG): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 결합 경계 위의 val (train 150 전체 학습). 굵은 글씨 = J와 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .95)); ex.save_fig(fig, "val_판정", "세 구성의 val 판정. 세로축 = 세 번째 입력(G는 AE z 참고)")

# %% val 이동 (라벨 없음): 정상 95% 위 비율
sh = {}
for nm, col, v in (("AE", FIT["J (AE)"]["m"].Z_[:, 2], FIT["J (AE)"]["fv"]["ae"]), ("정상 PCA", FIT["J′-mspc"]["m"].Z_[:, 2], FIT["J′-mspc"]["fv"]["ex"][:, 0])):
    sh[nm] = dict(정상중앙=round(float(np.median(col[nor])), 2), 고장중앙=round(float(np.median(col[~nor])), 2), val중앙=round(float(np.median(v)), 2),
                 val_정상95위=round(float(np.mean(v > np.percentile(col[nor], 95))), 2))
ex.note("val 이동", sh)

# %% t-SNE 기본 (핵심 그림)
fpJ = np.flatnonzero((y == 0) & (RES["J (AE)"]["fp_counts"] >= 3))
fig, ti = tsne_default(ns, mark=fpJ, mark_label="J 오경보 (5번 중 3회+)", title="EXP-049 (입력 공간은 J와 같음)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10). 빨간 원 = J 오경보", key=True); ex.note("t-SNE", ti)
print("done", flush=True)
