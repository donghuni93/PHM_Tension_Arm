# %% [markdown]
# EXP-052 비선형 결합 재실험 (11-22) — 기준 J-축소. 설계는 같은 폴더 계획.md (실행 전 고정)
# - 폴드마다 J-축소(RingShrink → Add83 → TwoStack)를 한 번 학습해 안쪽 OOF 결합 입력 Z = [ET logit, 박자 z, AE z]를 만들고,
#   결합기만 바꿔 Z에 맞춘다. 판정 = 결합 ≥ 문턱 OR E2 OR 8.3k (J-축소와 같음)
# - 결합기 7종 × 용량 사다리(실행 전 고정, 안쪽 튜닝 없이 칸마다 전부 보고) × 입력 변환 2(원래 / 꼬리 누르기 sign·log1p)
#   문턱 = 결합기의 안쪽 4겹 OOF 점수에서 F2 최대. 단 "로지스틱 J 절차"는 J와 같게 학습 점수에서 문턱 (J-축소 재현 확인용)
# - 잣대 (판단기준 ①): ① 중첩 F2 ± sd ② 경계 모양(logistic_plane) ③ 오류 분석 표 ④ val 흔들림 (a) 폴드 경계 5개 + 전체 경계에서 판정이 바뀌는 파일 수
#   (b) 부트스트랩 300회 결합만 val 고장 수 5–95 %  (EXP-050과 같은 정의, 결합기만 바꿈)
# - 교체 규칙(실행 전 고정): ① J-축소보다 sd 넘게 좋고 ④ 흔들림 폭이 넓지 않을 때만 교체 논의. 아니면 로지스틱 유지
# - kNN은 뺌(11-22 사용자: 잘 되는 비선형만)
# 실행: QE_FULL=1 conda run -n qe-phm python run.py   (quick은 오류 확인용)

# %% 설정
import os, sys, time, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import FULL, N_SEEDS, best_f2_threshold
from src.exp2util import NS, rule83_flags, NOR, ABN, tsne_default, j_mats, j_shrink, ts_of
from src.final import _fit_ae, _ae_score
from src.experts import LP as LP_TR, segment_logpsd
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, recall_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVC
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
INK, MUTED = "#1d2430", "#5b6575"
ex = Experiment(__file__, "EXP-052", "비선형 결합 재실험 (기준 J-축소)", ["H1", "H2", "H3"],
                member="이동훈", model="J-축소의 결합기 교체: 로지스틱 · 가법 스플라인 · poly2 · SVM-RBF · 단조 부스팅 · MLP · ET 결합 (용량 사다리 × 입력 변환 2), OR E2 OR 8.3k",
                changes="EXP-045 재실험: 기준 J-축소 · 용량 사다리(튜닝 없이 전부 보고) · 꼬리 누르기 변환(022) · 단조 제약 · val 흔들림",
                key_changes=["기준 J → J-축소", "결합기마다 용량 사다리 (값 하나 → 2–6칸)", "입력 꼬리 누르기 sign·log1p (모든 결합기 공통)", "단조 제약 부스팅 · 가법 스플라인 추가, kNN 제외", "val 흔들림 (a)(b) 추가"])
ns = NS(); y, g, names = ns.y, ns.g, np.array(ns.names); nor = y == 0; nv = [f"val_{n[-3:]}" for n in ns.nv]
spec, res = rule83_flags(ns.X); sv, rv = rule83_flags(ns.Xv); r83, r83v = spec | res, sv | rv
_, LPv = segment_logpsd(ns.Xv); RAW = {"ae": _ae_score(_fit_ae(LP_TR[nor]), LPv)}
OUT = [names.tolist().index(n) for n in OUTLIER_NORMALS]
short = lambda n: n.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
M, Mv = j_mats(ns, r83); MK = j_shrink(ns)
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close(); T0 = time.time()


def log(*a):
    s = f"[{time.time() - T0:6.0f}s] " + " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


# %% 결합기 · 사다리 · 입력 변환 (실행 전 고정)
LADDER = [("로지스틱", "J 절차 (C=1, 학습 점수 문턱)", "lin", dict(C=1.0), True)]
LADDER += [("로지스틱", f"C={c}", "lin", dict(C=c), False) for c in (0.1, 1.0, 10.0)]
LADDER += [("가법 스플라인", f"매듭 {k} · C={c}", "spline", dict(k=k, C=c), False) for k in (4, 6) for c in (0.3, 3.0)]
LADDER += [("poly2", f"C={c}", "poly2", dict(C=c), False) for c in (0.1, 1.0)]
LADDER += [("SVM RBF", f"C={c} · γ={gm}", "svm", dict(C=c, gamma=gm), False) for c in (0.3, 1.0, 3.0) for gm in (0.1, 0.5)]
LADDER += [("단조 부스팅", f"깊이 {d} · 잎 {l}", "hgb", dict(d=d, l=l), False) for d in (2, 3) for l in (5, 15)]
LADDER += [("MLP", f"은닉 {h} · α={a}", "mlp", dict(h=h, a=a), False) for h in (8, 16) for a in (0.01, 0.1)]
LADDER += [("ET 결합", f"잎 {l}", "et", dict(l=l), False) for l in (3, 8)]
TF = {"원래": lambda Z: np.asarray(Z, float), "꼬리누름": lambda Z: np.sign(Z) * np.log1p(np.abs(Z))}


def make(kind, p):
    if kind == "lin": return LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000), "dec"
    if kind == "spline": return make_pipeline(StandardScaler(), SplineTransformer(n_knots=p["k"], degree=3, extrapolation="linear"),
                                              LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000)), "dec"
    if kind == "poly2": return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LogisticRegression(C=p["C"], class_weight="balanced", max_iter=5000)), "dec"
    if kind == "svm": return make_pipeline(StandardScaler(), SVC(kernel="rbf", C=p["C"], gamma=p["gamma"], class_weight="balanced")), "dec"
    if kind == "hgb": return HistGradientBoostingClassifier(max_depth=p["d"], learning_rate=0.05, max_iter=200, min_samples_leaf=p["l"], monotonic_cst=[1, 1, 1],
                                                            class_weight="balanced", early_stopping=False, random_state=0), "proba"
    if kind == "mlp": return make_pipeline(StandardScaler(), MLPClassifier((p["h"],), alpha=p["a"], max_iter=4000, random_state=0)), "proba"
    if kind == "et": return ExtraTreesClassifier(300, min_samples_leaf=p["l"], class_weight="balanced", random_state=0, n_jobs=1), "proba"
    raise ValueError(kind)


def score(m, how, Z): return m.decision_function(Z) if how == "dec" else m.predict_proba(Z)[:, 1]


def fit_comb(ci, Z, yy):
    """결합기 한 칸을 Z에 맞추고 문턱을 정한다. J 절차 = 학습 점수에서, 나머지 = 안쪽 4겹 OOF 점수에서 F2 최대"""
    _, _, kind, p, insample = LADDER[ci]
    if insample:
        m, how = make(kind, p); m.fit(Z, yy); return m, how, float(best_f2_threshold(yy, score(m, how, Z)))
    oof = np.empty(len(yy))
    for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(Z, yy):
        m, how = make(kind, p); m.fit(Z[a], yy[a]); oof[b] = score(m, how, Z[b])
    m, how = make(kind, p); m.fit(Z, yy); return m, how, float(best_f2_threshold(yy, oof))


CELLS = [(ci, tn) for ci in range(len(LADDER)) for tn in TF]
lab = lambda ci, tn: f"{LADDER[ci][0]} {LADDER[ci][1]} · {tn}"


# %% 1. 바깥 폴드마다 J-축소 학습 → 결합 입력 (무거운 부분, 폴드당 한 번)
def fold(seed, tr, te):
    top = MK().fit(M[tr], y[tr]); ts, _ = ts_of(top); f = top.fires(M[te])
    return dict(seed=seed, tr=tr, te=te, Ztr=ts.Z_, ytr=ts.y_, Zte=np.column_stack([f["a"], f["z"], f["ae"]]),
                rule=f["E2"] | f["R83"], mainJ=f["main"])


jobs = [(s, tr, te) for s in range(N_SEEDS) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(M, y, g)]
with parallel_config(backend="loky", inner_max_num_threads=1):
    FO = Parallel(n_jobs=-1)(delayed(fold)(*j) for j in jobs)
log(f"J-축소 바깥 폴드 {len(FO)}개 학습 끝 ({'full' if FULL else 'quick'})")
RULE = np.zeros((N_SEEDS, ns.n), bool); JP = np.zeros((N_SEEDS, ns.n), bool)
for fo in FO: RULE[fo["seed"], fo["te"]] = fo["rule"]; JP[fo["seed"], fo["te"]] = fo["mainJ"] | fo["rule"]


# %% 2. 결합기 사다리 (가벼운 부분)
def comb_eval(ci, tn, k):
    fo = FO[k]; T = TF[tn]; m, how, thr = fit_comb(ci, T(fo["Ztr"]), fo["ytr"])
    return ci, tn, k, score(m, how, T(fo["Zte"])) >= thr


with parallel_config(backend="loky", inner_max_num_threads=1):
    R = Parallel(n_jobs=-1)(delayed(comb_eval)(ci, tn, k) for ci, tn in CELLS for k in range(len(FO)))
MAIN = {c: np.zeros((N_SEEDS, ns.n), bool) for c in CELLS}
for ci, tn, k, d in R: MAIN[(ci, tn)][FO[k]["seed"], FO[k]["te"]] = d
log("사다리 끝")


def as_nested(P):
    d = dict(f2=np.array([fbeta_score(y, p, beta=2) for p in P]), recall=np.array([recall_score(y, p) for p in P]),
             precision=np.array([precision_score(y, p, zero_division=0) for p in P]), fpr=np.array([p[y == 0].mean() for p in P]))
    d["miss_counts"] = ((y == 1) & ~P).sum(0); d["fp_counts"] = ((y == 0) & P).sum(0); return d


BASE = "J-축소 (로지스틱 J 절차) — 기준"
NEST = {BASE: as_nested(JP)}; rb = ex.record(BASE, NEST[BASE], names)
e50 = json.load(open(os.path.join(find_exp("EXP-050"), "result.json"), encoding="utf-8"))["configs"]["J-축소"]
log(f"재현 확인: 이 실행 J-축소 {rb['f2']:.3f} ± {rb['f2_sd']:.3f} · EXP-050 J-축소 {e50['f2']:.3f} ± {e50['f2_sd']:.3f} ({e50.get('mode', '')})")
SUM = {}
for ci, tn in CELLS:
    nm = lab(ci, tn); NEST[nm] = as_nested(MAIN[(ci, tn)] | RULE); r = ex.record("사다리: " + nm, NEST[nm], names, mid=True)
    SUM[(ci, tn)] = r
FAMS = list(dict.fromkeys(c[0] for c in LADDER))
best_of = {}
for fam in FAMS:
    cand = [(ci, tn) for ci, tn in CELLS if LADDER[ci][0] == fam]
    best_of[fam] = max(cand, key=lambda c: (SUM[c]["f2"], -SUM[c]["f2_sd"]))
NL = [f for f in FAMS if f != "로지스틱"]
BEST = max((best_of[f] for f in NL), key=lambda c: (SUM[c]["f2"], -SUM[c]["f2_sd"]))
for fam in FAMS:
    c = best_of[fam]; nm = f"{fam} 최선 ({LADDER[c[0]][1]} · {c[1]})"; NEST[nm] = NEST[lab(*c)]
    ex.record(nm, NEST[nm], names, main=(c == BEST))
rows = []
for fam in FAMS:
    for ci, tn in CELLS:
        if LADDER[ci][0] != fam: continue
        r = SUM[(ci, tn)]; d = r["f2"] - rb["f2"]
        rows.append({"결합기": fam, "칸": LADDER[ci][1], "입력": tn, "F2": f"{r['f2']:.3f} ± {r['f2_sd']:.3f}", "− 기준": f"{d:+.3f}",
                     "판정": "sd 넘게 좋음" if d > rb["f2_sd"] else ("sd 안" if d > -rb["f2_sd"] else "sd 넘게 나쁨"),
                     "재현율": r["recall"], "오경보율": r["fpr"], "자주 놓침": " ".join(r["missed"]) or "—"})
ex.save_md("사다리_표", f"기준 = {BASE} {rb['f2']:.3f} ± {rb['f2_sd']:.3f}. 모든 칸 = 결합 ≥ 문턱 OR E2 OR 8.3k\n\n" + md_table(pd.DataFrame(rows)))
log("최선 비선형:", lab(*BEST), SUM[BEST]["f2"], SUM[BEST]["f2_sd"])

# %% 3. 사다리 그림 (핵심)
fig, ax = plt.subplots(figsize=(18, 6.5)); xs = np.arange(len(LADDER))
ax.axhspan(rb["f2"] - rb["f2_sd"], rb["f2"] + rb["f2_sd"], color="#9aa3af", alpha=.25, lw=0, label=f"기준 J-축소 {rb['f2']:.3f} ± {rb['f2_sd']:.3f}")
ax.axhline(rb["f2"], color=MUTED, lw=1)
for k, (tn, mk_, col) in enumerate((("원래", "o", "#2a78d6"), ("꼬리누름", "s", "#eb6834"))):
    m_ = [SUM[(ci, tn)]["f2"] for ci in range(len(LADDER))]; s_ = [SUM[(ci, tn)]["f2_sd"] for ci in range(len(LADDER))]
    ax.errorbar(xs + (k - .5) * .25, m_, yerr=s_, fmt=mk_, color=col, ms=6, capsize=2, lw=1, label=f"입력 {tn}")
edges = [i for i in range(1, len(LADDER)) if LADDER[i][0] != LADDER[i - 1][0]]
for e in edges: ax.axvline(e - .5, color="#ccc", lw=.8)
for fam in FAMS:
    idx = [i for i, c in enumerate(LADDER) if c[0] == fam]; ax.text(np.mean(idx), 1.005, fam, ha="center", va="bottom", fontsize=10, weight="bold", transform=ax.get_xaxis_transform())
ax.set_xticks(xs); ax.set_xticklabels([c[1] for c in LADDER], rotation=55, ha="right", fontsize=8)
ax.set_ylabel("중첩 F2 (+ E2 + 8.3k)"); ax.grid(axis="y", alpha=.3); ax.legend(loc="lower left", fontsize=9)
ax.set_title(f"결합기 × 용량 사다리 (중첩 CV, seed {N_SEEDS}). 회색 띠 = 기준 ± sd. 띠 위로 벗어나야 개선", loc="left", fontsize=10, color=MUTED, pad=22)
fig.tight_layout(); ex.save_fig(fig, "사다리", "결합기 7종 × 용량 사다리 × 입력 변환 2의 중첩 F2 (회색 띠 = J-축소 ± sd)", key=True)

# %% 4. 전체 학습 · val · 흔들림 (a)(b): 기준 + 결합기별 최선
top = MK().fit(M, y); ts, tf = ts_of(top); fv = ts.fires_raw(tf(Mv), RAW); ft = top.fires(M)
Zf, yf = ts.Z_, ts.y_; Xv = np.column_stack([fv["a"], fv["z"], fv["ae"]]); rule_v = fv["E2"] | r83v; rule_t = ft["E2"] | ft["R83"]
CAND = [("J-축소 (기준)", (0, "원래"))] + [(f"{fam} 최선", best_of[fam]) for fam in NL]
S0 = [fo for fo in FO if fo["seed"] == 0]; Xtr = np.zeros((ns.n, 3)); rtr = np.zeros(ns.n, bool)
for fo in S0: Xtr[fo["te"]] = fo["Zte"]; rtr[fo["te"]] = fo["rule"]


def boot_one(ci, tn, b):
    rs = np.random.RandomState(b); i = rs.randint(0, len(yf), len(yf))
    if len(np.unique(yf[i])) < 2: return None
    T = TF[tn]; m, how, thr = fit_comb(ci, T(Zf[i]), yf[i]); return int((score(m, how, T(Xv)) >= thr).sum())


FIT, WOB = {}, {}
q = lambda v: [int(np.percentile(v, 5)), int(np.percentile(v, 95))]
for nm, (ci, tn) in CAND:
    T = TF[tn]; m, how, thr = fit_comb(ci, T(Zf), yf)
    comb_t = score(m, how, T(Zf)) >= thr; comb_v = score(m, how, T(Xv)) >= thr
    FIT[nm] = dict(c=(ci, tn), m=m, how=how, thr=thr, comb=comb_t, fin=comb_t | rule_t, vfin=comb_v | rule_v)
    DT, DV = [], []                                          # (a) 폴드 결합기 5개 + 전체 결합기, 같은 좌표(Xtr · Xv)에서 판정
    for fo in S0:
        mf, hf, tf_ = fit_comb(ci, T(fo["Ztr"]), fo["ytr"]); DT.append(score(mf, hf, T(Xtr)) >= tf_); DV.append(score(mf, hf, T(Xv)) >= tf_)
    DT.append(score(m, how, T(Xtr)) >= thr); DV.append(comb_v)
    DT, DV = np.array(DT), np.array(DV)
    flip_t = (DT.min(0) != DT.max(0)) & ~rtr; flip_v = (DV.min(0) != DV.max(0)) & ~rule_v
    with parallel_config(backend="loky", inner_max_num_threads=1):
        B = [b for b in Parallel(n_jobs=-1)(delayed(boot_one)(ci, tn, b) for b in range(300)) if b is not None]
    WOB[nm] = dict(칸=lab(ci, tn), 바뀌는_train=int(flip_t.sum()), 바뀌는_val=int(flip_v.sum()),
                   train_파일=[short(names[i]) for i in np.flatnonzero(flip_t)], val_파일=[short(nv[i]) for i in np.flatnonzero(flip_v)],
                   결합만_val고장_5_95=q(B), val_고장=int(FIT[nm]["vfin"].sum()))
    log(nm, WOB[nm])
ex.note("흔들림 (a)(b) · val 고장 수", WOB)
f2_of = lambda nm: rb if nm == "J-축소 (기준)" else SUM[FIT[nm]["c"]]
wrows = [{"구성": nm, "칸": w["칸"], "중첩 F2": f"{f2_of(nm)['f2']:.3f} ± {f2_of(nm)['f2_sd']:.3f}",
          "판정 바뀌는 파일 train · val": f"{w['바뀌는_train']} · {w['바뀌는_val']}", "결합만 val 고장 5–95 %": f"{w['결합만_val고장_5_95'][0]}–{w['결합만_val고장_5_95'][1]}",
          "val 고장 (최종)": f"{w['val_고장']}/40", "바뀌는 val 파일": " ".join(w["val_파일"]) or "—"} for nm, w in WOB.items()]
ex.save_md("흔들림_표", "EXP-050과 같은 정의: (a) seed 0 폴드 결합기 5개 + 전체 결합기에서 판정이 갈리는 파일(규칙 켜진 파일 제외) (b) 부트스트랩 300회 결합만 val 고장 수\n\n" + md_table(pd.DataFrame(wrows)))

# %% 5. 오류 분석 표 (가로) · val 판정
cnt = lambda r: np.where(y == 1, len(r["f2"]) - r["miss_counts"], r["fp_counts"])
CB = cnt(NEST[BASE]); CC = {nm: cnt(NEST[lab(*c)]) for nm, c in CAND[1:]}
files = [names[i] for i in range(ns.n) if any((C_[i] > N_SEEDS / 2) != (y[i] == 1) or (0 < C_[i] < N_SEEDS) for C_ in [CB] + list(CC.values()))]
extra = {f"{nm} ({N_SEEDS}번 중 고장)": {f: int(CC[nm][names.tolist().index(f)]) for f in files} for nm in CC}
ex.save_md("오류분석_표", f"모델 판단 행 = 기준 J-축소. 아래 행 = 결합기별 최선의 고장 판정 횟수 (같은 분할)\n\n" + error_md_table(names, y, CB, files, n_seeds=N_SEEDS, extra=extra))
V = pd.DataFrame({"파일": [n[-3:] for n in nv], **{nm: np.where(F_["vfin"], "고장", "정상") for nm, F_ in FIT.items()}})
d = V[V[list(FIT)].nunique(axis=1) > 1]
ex.save_md("val_판정", "val 고장 판정 수: " + " · ".join(f"{nm} {int(F_['vfin'].sum())}/40" for nm, F_ in FIT.items()) + f"\n\n구성 사이에 판정이 다른 파일\n\n{md_table(d) if len(d) else '(없음)'}\n\n전체\n\n{md_table(V)}")

# %% 6. 핵심 그림: logistic_plane — 기준 J-축소 vs 최선 비선형 (ET × AE, 박자 z = 0 단면)
#   배경 = 결합 판단, 세로 점선 = ET 단독 판단, 점 색 = 최종 판정. 규칙만으로 잡힌 고장은 뺀다. 초록 고리 = 박자 z > 0.5 (단면 밖으로 올라간 점)
t_et = float(best_f2_threshold(yf, Zf[:, 0])); bn = [nm for nm, c in CAND if c == BEST][0]


def plane(ax, nm, val=False):
    F_ = FIT[nm]; ci, tn = F_["c"]; T = TF[tn]
    P = Xv if val else Zf; fin = F_["vfin"] if val else F_["fin"]
    x0, x1 = min(P[:, 0].min(), t_et) - .6, max(P[:, 0].max(), t_et) + .6; y0, y1 = P[:, 2].min() - .6, P[:, 2].max() + .6
    gx, gy = np.meshgrid(np.linspace(x0, x1, 260), np.linspace(y0, y1, 260)); G = np.column_stack([gx.ravel(), np.zeros(gx.size), gy.ravel()])
    S = (score(F_["m"], F_["how"], T(G)) >= F_["thr"]).reshape(gx.shape).astype(float)
    ax.contourf(gx, gy, S, levels=[-.5, .5, 1.5], colors=["#d6e6fa", "#fbe0d0"], zorder=0)
    ax.axvline(t_et, color=INK, lw=1.3, ls="--", label=f"ET 단독 판단 경계 ({t_et:.2f})")
    col = np.where(fin, ABN, NOR); big = P[:, 1] > .5
    if val:
        ax.scatter(P[:, 0], P[:, 2], marker="^", s=70, c=col, edgecolor="k", lw=.6, zorder=3)
        for i in range(len(P)):
            if fin[i] != FIT["J-축소 (기준)"]["vfin"][i]: ax.annotate(short(nv[i]), (P[i, 0], P[i, 2]), xytext=(4, 4), textcoords="offset points", fontsize=8, weight="bold")
    else:
        show = ~((y == 1) & ~F_["comb"] & fin); o = np.zeros(ns.n, bool); o[OUT] = True
        for msk, mkr, s_ in ((show & (y == 0) & ~o, "s", 22), (show & (y == 1), "o", 40), (show & o, "D", 60)):
            ax.scatter(P[msk, 0], P[msk, 2], marker=mkr, s=s_, c=col[msk], edgecolor="k", lw=.5, zorder=3)
        for i in np.flatnonzero(show & (fin != (y == 1))): ax.annotate(short(names[i]), (P[i, 0], P[i, 2]), xytext=(4, 4), textcoords="offset points", fontsize=7.5)
        big &= show
    ax.scatter(P[big, 0], P[big, 2], s=150, facecolor="none", edgecolor="#1f9d55", lw=1.3, zorder=4)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.2); ax.set_xlabel("ET 점수 (OOF logit)", color=MUTED); ax.set_ylabel("AE z", color=MUTED)
    w = WOB[nm]; r = SUM[F_["c"]] if nm != "J-축소 (기준)" else rb
    ax.set_title(f"{nm} — {w['칸']}\nF2 {r['f2']:.3f} ± {r['f2_sd']:.3f} · 바뀌는 파일 train {w['바뀌는_train']} · val {w['바뀌는_val']}" + (f" · val 고장 {int(fin.sum())}/40" if val else ""), fontsize=9.5, loc="left")
    ax.legend(fontsize=8, loc="upper left")


leg = [Line2D([], [], marker="o", ls="", mfc=ABN, mec="k", label="고장 판정"), Line2D([], [], marker="o", ls="", mfc=NOR, mec="k", label="정상 판정"),
       Line2D([], [], marker="s", ls="", color="#999", label="□ 정상 라벨"), Line2D([], [], marker="o", ls="", color="#999", label="○ 고장 라벨"),
       Line2D([], [], marker="D", ls="", color="#999", label="◆ 이상치 정상 059 · 065 · 116"), Line2D([], [], marker="o", ls="", mfc="none", mec="#1f9d55", label="박자 z > 0.5 (단면 밖)")]
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, ["J-축소 (기준)", bn]): plane(ax, nm)
fig.legend(handles=leg, ncol=6, frameon=False, loc="upper right", fontsize=9)
fig.suptitle("logistic_plane — 학습 150 (안쪽 OOF 입력 · 전체 학습 결합기), 박자 z = 0 단면. 배경 = 결합 판단, 세로 점선 = ET 단독 판단. 규칙만으로 잡힌 고장은 뺌", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "logistic_plane", "기준 J-축소 vs 최선 비선형 결합의 경계 (ET × AE, 박자 z = 0 단면)", key=True)
fig, axs = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, nm in zip(axs, ["J-축소 (기준)", bn]): plane(ax, nm, val=True)
fig.suptitle("val 40 — 같은 경계 위의 val (train 150 전체 학습). 굵은 글씨 = 기준과 판정이 다른 파일", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .93)); ex.save_fig(fig, "val_판정", "기준과 최선 비선형의 val 판정")
fig, axs = plt.subplots(2, 3, figsize=(24, 14))
for ax, nm in zip(axs.ravel(), [n for n, _ in CAND[1:]]): plane(ax, nm)
fig.suptitle("결합기별 최선의 경계 (학습 150, 박자 z = 0 단면)", x=.01, ha="left", fontsize=10, color=MUTED)
fig.tight_layout(rect=(0, 0, 1, .96)); ex.save_fig(fig, "결합기별_경계", "결합기별 최선 칸의 경계 6종")

# %% 7. 기본 t-SNE (핵심)
fpB = np.flatnonzero((y == 0) & (NEST[lab(*BEST)]["fp_counts"] > N_SEEDS / 2))
fig, ti = tsne_default(ns, mark=fpB, mark_label=f"최선 비선형 오경보 ({N_SEEDS}번 중 과반)", title="EXP-052 (J 입력 공간)")
ex.save_fig(fig, "tsne_라벨", "기본 t-SNE(4-10). 빨간 원 = 최선 비선형 오경보", key=True); ex.note("t-SNE", ti)
log("done")
