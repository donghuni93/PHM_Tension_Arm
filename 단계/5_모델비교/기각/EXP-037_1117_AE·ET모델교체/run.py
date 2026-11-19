# %% [markdown]
# EXP-037 (11-17) AE · ET 자리의 모델만 바꿔도 지금 수준(EXP-034 c, 0.955 ± 0.007)이 나오는가
# - 바꾸지 않는 것: 입력(ET 자리 = ext150 + 울림 2열, AE 자리 = 1초 구간 8개 log Welch PSD 1025 bin),
#   폴드 안 z 정규화 → 로지스틱 결합, 규칙 3개(유격 반복 · 8.3k 스펙트럼 OR 공진 · 본체 모드 내림)
# - ET 자리 후보 (지도학습, 고장 확률 → logit): RF · CatBoost · HistGB · 로지스틱(L2)
# - AE 자리 후보 (정상 구간만 학습, 파일 점수 = log10 구간 평균): PCA16 복원오차 · 정상 z 거리(모델 없음) · kNN 거리 · IsolationForest
# - 설정값은 실행 전에 고정, 튜닝 없음. 이미 해 본 SVM(015·026·028) · MLP(031) · MiniRocket(019)는 제외
# - 단계 1 quick_cv(PR-AUC, seed 3) → 자리별 1등을 자동 선택 → 단계 2 중첩 CV: 기준 / ET 교체 / AE 교체 / 둘 다
# - 가설: 없음(모델만 바꿈). 규칙·입력은 EXP-034 c 그대로 (H1 · H2′)
# 실행: conda run -n qe-phm python "단계/5_모델비교/스크리닝/EXP-037_1117_AE·ET모델교체/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.experts import RuleOverride, Rule83, KnockRule, ScoreStack, ae_mean, ae_segerr, warm_cache, LP
from src.evaluate import quick_cv
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-037", "AE · ET 자리 모델 교체 — 다른 모델로 지금 수준(EXP-034 c)이 나오는가", ["H1", "H2"],
                member="이동훈", model="(AE 자리 후보) + (ET 자리 후보) → 로지스틱 + 유격 · 8.3k(스펙트럼 OR 공진) · 본체 내림 규칙",
                changes="EXP-034 c에서 입력·규칙은 그대로, AE 자리(PCA·z거리·kNN·IF)와 ET 자리(RF·CatBoost·HistGB·로지스틱)의 모델만 교체",
                focus=("abn_022", "abn_005", "nor_025", "nor_055", "nor_059", "nor_116", "abn_015", "abn_020", "abn_009"))
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]; nor = y == 0
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
E = cached_table("ext", feature_table_ext, X).values
IM = pd.read_csv(os.path.join(_d, "단계", "2_EDA", "8kHz공진", "2-16_1116_울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
key = [os.path.splitext(os.path.basename(n))[0] for n in names]
cnt = IM.groupby("file").size().reindex(key).fillna(0).values


def frac(lo, hi):
    b = IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean()
    return b.reindex(key).values.astype(float)


R24 = np.nan_to_num(frac(2125, 4125), nan=-1.0)
RBODY = np.nan_to_num(frac(5500, 7750), nan=-1.0)
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values
EX = np.hstack([E, R24[:, None], RBODY[:, None]])                    # ET 자리 입력 (EXP-034 ext+r2)
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


# %% 규칙 (EXP-034 c 그대로)
class Res83:
    def __init__(self, t=0.7): self.t = t
    def __repr__(self): return f"Res83(fit≥{self.t})"
    def __call__(self, te): return RES[list(te)] >= self.t


class Either:
    def __init__(self, a, b): self.a, self.b = a, b
    def __repr__(self): return f"{self.a} OR {self.b}"
    def __call__(self, te): return self.a(te) | self.b(te)


def body(te, th=0.9, nmin=3):
    te = list(te); return (RBODY[te] >= th) & (cnt[te] >= nmin)


class BodyCap(BaseEstimator):
    """RuleOverride + 본체 모드 내림 (EXP-034 c)"""
    def __init__(self, base_kw=None, rule=None, th=0.9): self.base_kw, self.rule, self.th = base_kw, rule, th
    def fit(self, F, yy):
        yy = np.asarray(yy)
        self.m_ = RuleOverride(self.base_kw, self.rule).fit(F, yy)
        st = self.m_.m_
        self.cap_ = float(np.median(st.m_.decision_function(st.oof_z_)[yy == 0]))
        return self
    def decision_function(self, F):
        s = self.m_.decision_function(F)
        te = np.asarray(F)[:, 0].astype(int)
        return np.where(body(te, self.th) & (s < 50), np.minimum(s, self.cap_), s)


RULES = [KnockRule(2.0, 0.7), Either(Rule83(0.75, 150), Res83(0.7))]


# %% ET 자리 후보: (tr, te) → (te, 1) logit 점수. 모두 입력 EX, 설정 고정
def _logit(p):
    p = np.clip(p, 1 / 600, 1 - 1 / 600); return np.log(p / (1 - p))[:, None]


class Sup:
    def __init__(self, kind="et"): self.kind = kind
    def __repr__(self): return f"Sup[{self.kind}]"
    def model(self):
        k = self.kind
        if k == "et":  return make_et(seed=0)()
        if k == "rf":  return RandomForestClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=1)
        if k == "hgb": return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, class_weight="balanced", random_state=0)
        if k == "cat":
            from catboost import CatBoostClassifier
            return CatBoostClassifier(iterations=300, depth=4, learning_rate=0.05, auto_class_weights="Balanced", random_seed=0, verbose=0, thread_count=1, allow_writing_files=False)
        if k == "lr":  return make_pipeline(StandardScaler(), LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000))
    def __call__(self, tr, te):
        tr, te = np.array(tr), np.array(te)
        m = self.model().fit(EX[tr], y[tr])
        if self.kind == "lr": return m.decision_function(EX[te])[:, None]
        return _logit(m.predict_proba(EX[te])[:, 1])


# %% AE 자리 후보: 정상(tr 중 y=0) 구간 스펙트럼만 학습 → te 파일 점수 log10(구간 평균) (te, 1)
class Unsup:
    def __init__(self, kind="pca"): self.kind = kind
    def __repr__(self): return f"Unsup[{self.kind}]"
    def __call__(self, tr, te):
        tr, te = np.array(tr), np.array(te)
        S = LP[tr[y[tr] == 0]].reshape(-1, LP.shape[-1])
        mu, sd = S.mean(0), S.std(0) + 1e-8
        Zs = (S - mu) / sd
        Z = ((LP[te] - mu) / sd).reshape(-1, LP.shape[-1])
        k = self.kind
        if k == "zdist":                                      # 모델 없음: 정상 평균에서 bin별 z 제곱 평균
            e = (Z ** 2).mean(1)
        elif k == "pca":                                      # 선형 AE: 병목 16 (AE와 같은 크기)
            p = PCA(16, random_state=0).fit(Zs)
            e = ((p.inverse_transform(p.transform(Z)) - Z) ** 2).mean(1)
        elif k == "knn":                                      # PCA 32 공간에서 정상 구간 5개까지 평균 거리
            p = PCA(32, random_state=0).fit(Zs)
            nn = NearestNeighbors(n_neighbors=5).fit(p.transform(Zs))
            e = nn.kneighbors(p.transform(Z))[0].mean(1)
        elif k == "if":                                       # PCA 32 공간 IsolationForest (높을수록 이상)
            p = PCA(32, random_state=0).fit(Zs)
            m = IsolationForest(n_estimators=300, random_state=0).fit(p.transform(Zs))
            e = np.exp(-m.score_samples(p.transform(Z)))       # 양수로 (log10 위해)
        return np.log10(e.reshape(len(te), -1).mean(1) + 1e-12)[:, None]


def mk(ae, et):
    return lambda: BodyCap(dict(experts=(ae, et), combine="lr"), RULES)


ET_C = {"ET (기준)": Sup("et"), "RF": Sup("rf"), "CatBoost": Sup("cat"), "HistGB": Sup("hgb"), "로지스틱": Sup("lr")}
AE_C = {"AE (기준)": ae_mean, "PCA16": Unsup("pca"), "z거리": Unsup("zdist"), "kNN": Unsup("knn"), "IF": Unsup("if")}
warm_cache([ae_segerr], seeds=range(5), nested=True)
log("AE 캐시 준비 끝")

# %% 단계 1: quick_cv (PR-AUC, 규칙 포함 c 전체). 한 자리만 바꾸고 다른 자리는 기준
Q = {}
for nm, f in ET_C.items():
    Q[("ET", nm)] = quick_cv(F, y, g, mk(ae_mean, f))
for nm, f in list(AE_C.items())[1:]:
    Q[("AE", nm)] = quick_cv(F, y, g, mk(f, Sup("et")))
Q[("AE", "AE (기준)")] = Q[("ET", "ET (기준)")]
# 단독 점수 (규칙 · 결합 없이 그 모델 하나): 그 모델이 혼자 얼마나 보는가
SOLO = {}
for slot, C in (("ET", ET_C), ("AE", AE_C)):
    for nm, f in C.items():
        SOLO[(slot, nm)] = quick_cv(F, y, g, lambda f=f: ScoreStack(experts=(f,), combine="col0"))
lines = ["자리 | 후보 | quick_cv PR-AUC (c 전체, 다른 자리 = 기준) | ROC-AUC | 단독 PR-AUC"]
for k, r in Q.items():
    s = SOLO[k]
    lines.append(f"{k[0]} | {k[1]} | {r['pr_auc'].mean():.3f} ± {r['pr_auc'].std():.3f} | {r['roc_auc'].mean():.3f} | {s['pr_auc'].mean():.3f} ± {s['pr_auc'].std():.3f}")
for l in lines: log(l)
best = {slot: max((k for k in Q if k[0] == slot and "기준" not in k[1]), key=lambda k: Q[k]["pr_auc"].mean())[1] for slot in ("ET", "AE")}
log("자리별 1등 (기준 제외):", best)
ex.note("quick_cv", "\n".join(lines) + f"\n\n자리별 1등 (기준 제외, 자동 선택): ET 자리 {best['ET']} · AE 자리 {best['AE']}")

# %% 단계 2: 중첩 CV
CONFIGS = {
    "기준 | AE + ET (EXP-034 c)": (ae_mean, ET_C["ET (기준)"]),
    f"ET 교체 | AE + {best['ET']}": (ae_mean, ET_C[best["ET"]]),
    f"AE 교체 | {best['AE']} + ET": (AE_C[best["AE"]], ET_C["ET (기준)"]),
    f"둘 다 | {best['AE']} + {best['ET']}": (AE_C[best["AE"]], ET_C[best["ET"]]),
}
MAIN = f"둘 다 | {best['AE']} + {best['ET']}"
res = {}
for nm, (a, e) in CONFIGS.items():
    res[nm] = nested_cv(F, y, g, mk(a, e), n_jobs=-1)
    ex.record("중첩: " + nm, res[nm], names, main=(nm == MAIN))
    c = ex.res["configs"]["중첩: " + nm]
    log(f"{nm:34s} F2 {res[nm]['f2'].mean():.3f} ± {res[nm]['f2'].std():.3f} · 재현율 {res[nm]['recall'].mean():.3f} · "
        f"오경보율 {res[nm]['fpr'].mean():.3f} · 미탐 {dict(sorted(c['miss_counts'].items()))} · 오경보 "
        f"{ {k: v for k, v in sorted(c['fp_counts'].items()) if v >= 2} }")


# %% 그림 1 (핵심): logistic_plane — 기준 | ET 교체 / AE 교체 | 둘 다
alli = np.arange(len(y))


def plane(ax, ae, et, title, xl, yl):
    Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae, et), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        on = KnockRule(2.0, 0.7).fit(tr)(te)
        for r in RULES[1:]: on |= r(te)
        rule[te] = on
    capped = body(alli) & ~rule
    Zc = np.clip(Z, -3, 4.5)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5); ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    show = ~(rule & (y == 1))
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
    ax.scatter(Zc[capped, 0], Zc[capped, 1], s=70, marker="o", facecolor="none", edgecolor="#2ca02c", lw=1.2, label="본체 모드 내림 대상")
    if (rule & nor).any():
        ax.scatter(Zc[rule & nor, 0], Zc[rule & nor, 1], s=110, marker="D", facecolor="none", edgecolor="#7a3db8", lw=1.3, label="규칙이 켜진 정상")
    c = ex.res["configs"]["중첩: " + title]
    wrong = [names.index(n) for n, k in {**c["miss_counts"], **c["fp_counts"]}.items() if k >= 2]
    ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=80, marker="x", color="k", lw=1.6, label="자주 틀림 (2/5+)")
    for i in wrong:
        k = c["miss_counts"].get(names[i]) or c["fp_counts"].get(names[i])
        ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {k}/5", Zc[i], xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=ABN if y[i] else NOR, weight="bold")
    gone = [short(names[i]) for i in np.flatnonzero(rule & (y == 1))]
    ax.text(.98, .02, f"규칙이 잡아 뺀 고장 {len(gone)}개:\n" + " ".join(gone[:6]) + ("\n" + " ".join(gone[6:]) if len(gone) > 6 else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel(f"{xl} z (스펙트럼 모양 이상)"); ax.set_ylabel(f"{yl} z (ext150 + 울림 2열)")
    ax.set_title(f"{title}\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f}", fontsize=9.5)


fig, axs = plt.subplots(2, 2, figsize=(16, 16))
for ax, (nm, (a, e)) in zip(axs.ravel(), CONFIGS.items()):
    xl = "AE" if a is ae_mean else best["AE"]; yl = "ET" if e is ET_C["ET (기준)"] else best["ET"]
    plane(ax, a, e, nm, xl, yl)
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("logistic_plane — 위: 기준 | ET 자리 교체 · 아래: AE 자리 교체 | 둘 다 교체 (OOF seed 0, 규칙 고장은 평면에서 뺌)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.98))
ex.save_fig(fig, "logistic_plane", "모델을 바꾸면 경계 근처 파일(022 · 005 · 025 · 116 · 059)의 위치가 달라지는가", key=True)

# %% 그림 2 (핵심): 스크리닝 막대
fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
for ax, slot in zip(axs, ("ET", "AE")):
    ks = [k for k in Q if k[0] == slot]; ks.sort(key=lambda k: "기준" not in k[1])
    v = [Q[k]["pr_auc"].mean() for k in ks]; s = [Q[k]["pr_auc"].std() for k in ks]
    sv = [SOLO[k]["pr_auc"].mean() for k in ks]
    xs = np.arange(len(ks))
    ax.bar(xs - .2, v, .4, yerr=s, capsize=3, color=["#999"] + ["#555"] * (len(ks) - 1), label="c 전체에 넣었을 때")
    ax.bar(xs + .2, sv, .4, color="#c9c9c9", label="단독 (규칙·결합 없이)")
    for xv, vv, ss in zip(xs, v, sv):
        ax.text(xv - .2, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=8); ax.text(xv + .2, ss, f"{ss:.3f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(v[0], color="k", lw=.8, ls="--")
    ax.set_xticks(xs, [k[1] for k in ks]); ax.set_ylim(0.5, 1.0); ax.grid(alpha=.3, axis="y")
    ax.set_title(f"{slot} 자리 후보 — quick_cv PR-AUC (seed 3, 점선 = 기준)", loc="left"); ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
ex.save_fig(fig, "screening", "자리마다 어떤 모델이 기준만큼 나오는가. 단독이 약해도 c 전체에서는 비슷할 수 있다", key=True)

# %% 그림 3: hard_files
labs = list(CONFIGS)
HARD = ["abn_022", "abn_005", "abn_002", "abn_029", "abn_015", "abn_020", "abn_009", "abn_017", "abn_025", "abn_028",
        "nor_025", "nor_055", "nor_059", "nor_116", "nor_019", "nor_092", "nor_054", "nor_106"]
M = np.array([[ex.res["configs"]["중첩: " + c]["miss_counts" if h.startswith("abn") else "fp_counts"].get(h, 0) for c in labs] for h in HARD])
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(M, cmap="Reds", vmin=0, vmax=5, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="w" if M[i, j] >= 3 else "k")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h for h in HARD], fontsize=8)
ax.set_xticks(range(len(labs)), [c.replace(" | ", "\n") for c in labs], fontsize=8)
ax.set_title("파일별 틀린 횟수 (중첩 CV seed 5번 중). 고장 = 미탐, 정상 = 오경보", loc="left")
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "모델을 바꿔도 같은 파일을 틀리는가(정보 한계) 다른 파일을 틀리는가(모델 차이)")

# %% 그림 4: t-SNE (입력 공간 2개 + 둘 다 교체 구성의 OOF 점수 색)
oof = Q[("ET", best["ET"])]["oof"][0]
PSDm = LP.mean(1)
fig, _ = viz.tsne({"ET 자리 입력 ext150 + 울림 2열": EX, "AE 자리 입력 log PSD (구간 평균)": PSDm}, y, names, mark=list(ex.focus), score=oof)
ex.save_fig(fig, "tsne", f"입력 공간마다 고장이 따로 모이는가. 2행 색 = ET 자리 {best['ET']} 구성의 OOF 점수(seed 0)")

# %% 그림 5 (맨 끝)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_missed", "문제 파일의 PSD가 정상 범위 밖으로 나가는 주파수")
log("done")
