# %% [markdown]
# EXP-045 (2차 설계, 11-22) 결합 모델을 로지스틱 대신 비선형로 — 사용자 질문 (EXP-044 뒤)
# - 같은 3입력 Z = [ET 안쪽 OOF logit, 박자 z, AE z] (EXP-044 TwoAE가 만든 것)에 결합 모델만 바꾼다: 로지스틱(lin) · 2차 항 로지스틱(poly2) · RBF SVM · 부스팅(깊이 2) · kNN · 작은 MLP
# - 각 결합마다 (a) 단독(전문가 · 규칙 없음) (b) + E2 + 8.3k (OR)를 같은 중첩 CV 분할(바깥 5 × seed 5)에서 잰다. 하이퍼파라미터는 고정(실행 전), 문턱 = 학습 폴드 Z의 안쪽 4겹 OOF 점수에서 F2 최대
# - 선택 규칙(실행 전 고정): 비선형이 로지스틱(lin)보다 (b)에서 sd(lin의 sd)를 넘게 좋을 때만 교체를 논의한다. 아니면 lin 유지
# - 결합 입력 Z는 폴드 안에서만 만들어진다(TwoAE). 8.3k 규칙은 EXP-043과 같은 사후 문턱
# 실행: conda run -n qe-phm python "단계/5_모델비교/4_최종모델J/EXP-045_1122_비선형결합/run.py"

# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.exp2util import *
from src.expA import A as ACls
from src.models2 import TwoAE
from src.evaluate import best_f2_threshold
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_config
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, recall_score, precision_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
warnings.filterwarnings("ignore")
EXP044 = find_exp("EXP-044")
ex = Experiment(__file__, "EXP-045", "결합 모델 비교 — 로지스틱 vs 비선형 (같은 3입력)", ["H3", "H2"],
                member="이동훈", model="결합(ET OOF logit · 박자 z · AE z) = lin · poly2 · SVM-RBF · 부스팅 · kNN · MLP, 각각 단독 / + E2 + 8.3k",
                changes="EXP-044의 결합 로지스틱을 비선형 모델로 바꿈 (하이퍼파라미터 고정, 문턱 = 안쪽 OOF F2 최대)")
ns = NS(); names, y, g = ns.names, ns.y, ns.g
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


spec, res = rule83_flags(ns.X); r83 = (spec | res); idx = np.arange(ns.n, dtype=float); Fn = np.column_stack([ns.M, idx])


class Wt:                                              # 샘플 가중(균형)이 필요한 sklearn 모델을 래핑
    def __init__(self, m): self.m = m
    def fit(self, Z, yy): self.m.fit(Z, yy, sample_weight=compute_sample_weight("balanced", yy)); return self
    def score(self, Z): return self.m.predict_proba(Z)[:, 1]


class Pl:
    def __init__(self, m, how): self.m, self.how = m, how
    def fit(self, Z, yy): self.m.fit(Z, yy); return self
    def score(self, Z): return self.m.decision_function(Z) if self.how == "dec" else self.m.predict_proba(Z)[:, 1]


COMB = {"lin 로지스틱": lambda: Pl(LogisticRegression(C=1.0, class_weight="balanced"), "dec"),
        "poly2 2차항 로지스틱": lambda: Pl(make_pipeline(StandardScaler(), PolynomialFeatures(2), LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000)), "dec"),
        "svm RBF": lambda: Pl(make_pipeline(StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")), "dec"),
        "gb 부스팅(깊이 2)": lambda: Wt(GradientBoostingClassifier(n_estimators=100, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=0)),
        "knn k=7": lambda: Pl(make_pipeline(StandardScaler(), KNeighborsClassifier(7, weights="distance")), "proba"),
        "mlp (8)": lambda: Pl(make_pipeline(StandardScaler(), MLPClassifier((8,), alpha=1e-2, max_iter=3000, random_state=0)), "proba")}


def fit_comb(make, Z, yy):
    oof = np.empty(len(yy))
    for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(Z, yy): oof[b] = make().fit(Z[a], yy[a]).score(Z[b])
    return make().fit(Z, yy), float(best_f2_threshold(yy, oof))


def task(seed, tr, te):
    m = TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)).fit(Fn[tr], y[tr])
    f = m.fires(Fn[te]); Zte = np.column_stack([f["a"], f["z"], f["ae"]]); out = {}
    for nm, make in COMB.items():
        c, thr = fit_comb(make, m.Z_, y[tr]); main = c.score(Zte) >= thr
        out[nm] = (main, main | f["E2"] | r83[te])
    return seed, te, out, Zte


# %% 중첩 CV (같은 분할: 바깥 5겹 × seed 5개)
jobs = [(s, tr, te) for s in range(5) for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
with parallel_config(backend="loky", inner_max_num_threads=1):
    R = Parallel(n_jobs=-1)(delayed(task)(*j) for j in jobs)
PRED = {(nm, k): np.zeros((5, ns.n), bool) for nm in COMB for k in (0, 1)}; Z0 = np.zeros((ns.n, 3))
for s, te, out, Zte in R:
    for nm, (a_, b_) in out.items(): PRED[(nm, 0)][s, te] = a_; PRED[(nm, 1)][s, te] = b_
    if s == 0: Z0[te] = Zte


def as_nested(P):
    d = dict(f2=np.array([fbeta_score(y, p, beta=2) for p in P]), recall=np.array([recall_score(y, p) for p in P]), precision=np.array([precision_score(y, p, zero_division=0) for p in P]), fpr=np.array([p[y == 0].mean() for p in P]))
    d["miss_counts"] = ((y == 1) & ~P).sum(0); d["fp_counts"] = ((y == 0) & P).sum(0); return d


labs = {}
for nm in COMB:
    for k, tag in ((1, "+ E2 + 8.3k"), (0, "단독")):
        key = f"{nm} {tag}"; labs[(nm, k)] = key
        c = ex.record("중첩: " + key, as_nested(PRED[(nm, k)]), names, main=(nm.startswith("lin") and k == 1))
        log(f"{key:36s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} · 미탐 {c['miss_counts']} · 오경보(3회+) { {a: b for a, b in c['fp_counts'].items() if b >= 3} }")
j44 = json.load(open(os.path.join(EXP044, "result.json"), encoding="utf-8"))["configs"]["중첩: J | W + AE + E2 + 8.3k (최종 후보)"]
ex.res["configs"]["중첩: J | EXP-044 값 (TwoAE 로지스틱)"] = j44; ex._write()
base = ex.res["configs"]["중첩: " + labs[("lin 로지스틱", 1)]]
for nm in COMB:
    d = ex.res["configs"]["중첩: " + labs[(nm, 1)]]["f2"] - base["f2"]
    log(f"{nm:24s} (+E2+8.3k) − lin = {d:+.3f}  (lin sd {base['f2_sd']}) → {'sd 넘게 좋음' if d > base['f2_sd'] else ('sd 안' if d > -base['f2_sd'] else 'sd 넘게 나쁨')}")
cfgs = [labs[(nm, 1)] for nm in COMB]; cf = lambda k: ex.res["configs"]["중첩: " + k]
bad = sorted({k for c in cfgs for k in list(cf(c)["miss_counts"]) + list(cf(c)["fp_counts"]) if max(cf(c)["miss_counts"].get(k, 0), cf(c)["fp_counts"].get(k, 0)) >= 3})
ex.focus = bad[:12]

# %% 그림 1 (핵심): 결합 모델별 ET × AE 평면 (박자 z = 0 단면), 전체 150으로 학습한 경계 + OOF 점 (seed 0). 데이터 범위로 크롭
m_all = TwoAE("lin", ("E2",), fixed=tuple(ns.FIXED), nf=ns.nf, df=ns.df, peak=tuple(ns.PEAK)).fit(Fn, y); Zall = m_all.Z_
x0, x1, y0, y1 = Z0[:, 0].min() - .3, Z0[:, 0].max() + .3, Z0[:, 2].min() - .3, Z0[:, 2].max() + .3
xx, yy_ = np.meshgrid(np.linspace(x0, x1, 220), np.linspace(y0, y1, 220)); G = np.column_stack([xx.ravel(), np.zeros(xx.size), yy_.ravel()])
fig, axs = plt.subplots(2, 3, figsize=(21, 12.5)); nor = y == 0
for ax, (nm, make) in zip(axs.ravel(), COMB.items()):
    c, thr = fit_comb(make, Zall, y); S = c.score(G).reshape(xx.shape)
    ax.contourf(xx, yy_, S, levels=[-1e9, thr, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy_, S, levels=[thr], colors="k", linewidths=1.3)
    ax.scatter(Z0[nor, 0], Z0[nor, 2], color=NOR, s=22, marker="s", lw=0, alpha=.8); ax.scatter(Z0[~nor, 0], Z0[~nor, 2], color=ABN, s=28, lw=0)
    big = ~nor & (Z0[:, 1] > .5); ax.scatter(Z0[big, 0], Z0[big, 2], s=120, facecolor="none", edgecolor="#1f9d55", lw=1.3)
    r = ex.res["configs"]["중첩: " + labs[(nm, 0)]]; r1 = ex.res["configs"]["중첩: " + labs[(nm, 1)]]
    ax.set_title(f"{nm}\n단독 F2 {r['f2']:.3f} · +E2+8.3k F2 {r1['f2']:.3f} ± {r1['f2_sd']:.3f} (재현율 {r1['recall']:.3f} · 오경보율 {r1['fpr']:.3f})", fontsize=10)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.grid(alpha=.3); ax.set_xlabel("ET 점수 (OOF logit)"); ax.set_ylabel("AE z")
fig.suptitle("결합 모델별 경계 (박자 z = 0 단면, 초록 원 = 박자 z > 0.5인 고장). 점 = OOF seed 0. 최종 판정은 이 경계에 E2 · 8.3k가 OR로 더해진 것", x=.01, ha="left", fontsize=11); fig.tight_layout(rect=(0, 0, 1, .96))
ex.save_fig(fig, "logistic_plane", "결합 모델 6종의 ET × AE 평면 경계 (박자 z = 0 단면, 데이터 범위로 크롭)", key=True)
ex.save_fig(hard_fig(ns, ex, cfgs, bad), "hard_files", "어느 파일이 결합 모델별(+E2+8.3k)에서 맞고 틀리는가")

# %% psd (맨 끝). t-SNE는 입력이 EXP-044와 같아 생략
fq, Lp = viz.log_psd(ns.X)
ex.save_fig(viz.psd_band(fq, Lp, y, names, ex.focus), "psd_missed", "자주 틀린 파일의 PSD가 정상 5–95% 범위 밖으로 나가는 주파수")
log("done")
