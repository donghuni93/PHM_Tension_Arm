# %% [markdown]
# EXP-038 (11-17) 역할 분리 ET — 규칙이 잡는 고장을 ET·로지스틱 학습에서 뺀다
# - 출발: 사용자 목표(11-17) "과적합 없이 logistic_plane에서 고장과 정상을 더 멀리 떨어뜨리기"
#   파라미터 튜닝은 근거가 없다: EXP-018(AE·ET 역할별 최적화)·EXP-026(SVM 튜닝) 이득 없음,
#   11-17 진단(seed 10개): 평면 z 퍼짐 중 학습 운(AE seed·분할·ET 나무) 몫은 축마다 약 12%, 10 seed 평균도 분리 이득 없음
# - 가설: 지금 ET는 고장 30개 전체로 학습한다. 그중 약 13개는 규칙(8.3 kHz·유격 반복)이 어차피 고장으로 판정하고 평면에서도 빠진다.
#   ET가 그 고장(8.3 kHz 모드·유격 반복처럼 이미 해결된 차이)을 가르는 데 분할을 쓰면, 평면에 남는 고장 17개와 정상 사이의 경계는 덜 배운다.
#   규칙 고장을 학습에서 빼면 ET가 '규칙 밖 고장 vs 정상'에 집중해 정상의 ET z가 덜 퍼지고 고장 무리와의 간격이 벌어질 것 (H1·H2·H3 역할 분리)
# - 바꾼 것 하나: 학습에서 뺄 파일 = 바깥 학습 fold 안에서 규칙(유격 반복 · 8.3 kHz 스펙트럼 OR 공진)이 켜진 고장. 새 문턱·대역 없음
#   (유격 규칙의 정상 중심은 학습 fold 정상으로 fit, 8.3 kHz 두 규칙은 학습 없음). 채점은 모든 파일 그대로
# - 구성: c = 기준(EXP-034 c, 지금 최종) · r = 규칙 고장을 ET와 로지스틱 학습에서 뺌 (주) · r_et = ET 학습에서만 뺌 (로지스틱은 전체)
# - 분리 지표 (실행 전에 고정): 바깥 시험 점수에서 규칙이 켜진 파일(점수 > 50)을 뺀 나머지로 seed마다
#     PR-AUC · ROC-AUC · 간격 = (고장 10%선 − 정상 95%선) / 정상 sd · 겹친 파일 수 = (고장 10%선 이상인 정상) + (정상 95%선 이하인 고장)
#   + 평면(OOF seed 0)에서 정상 ET z의 sd. 마지막 확인은 중첩 F2 (fold마다 임계값 · 고정 1.0 둘 다)
# 실행: conda run -n qe-phm python "3_단계/5_모델실험/스크리닝/EXP-038_1117_역할분리ET/run.py"

# %% 설정
import os, sys, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import _oof_scores, _scores, best_f2_threshold, _collect
from src.experts import RuleOverride, Rule83, KnockRule, ScoreStack, ae_mean, ae_segerr, warm_cache
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed, parallel_config
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
FOCUS = ("abn_005", "abn_015", "abn_020", "abn_022", "abn_025", "nor_025", "nor_055", "nor_059", "nor_106", "nor_116")
ex = Experiment(__file__, "EXP-038", "역할 분리 ET — 규칙이 잡는 고장을 ET·로지스틱 학습에서 뺀다", ["H1", "H2", "H3"],
                member="이동훈", model="EXP-034 c, 단 ET·로지스틱은 규칙 밖 고장 + 정상으로만 학습",
                changes="학습 fold 안에서 규칙(유격 · 8.3 kHz)이 켜진 고장을 ET·로지스틱 학습에서 뺌. 새 문턱·대역 없음",
                focus=FOCUS)
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]; nor = y == 0; alli = np.arange(len(y))
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
key = [os.path.splitext(os.path.basename(n))[0] for n in names]
SEEDS, FIXED = range(5), 1.0
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


def who(m):
    return " ".join(short(names[i]) for i in np.flatnonzero(m)) or "없음"


# %% 입력: EXP-034 c와 같은 표 (2-16 충격 울림)
E = cached_table("ext", feature_table_ext, X).values
IM = pd.read_csv(os.path.join(_d, "3_단계", "2_EDA", "2-3_충격울림주파수", "자료", "figs", "impacts.csv"), encoding="utf-8-sig")
cnt = IM.groupby("file").size().reindex(key).fillna(0).values
frac = lambda lo, hi: np.nan_to_num(IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean().reindex(key).values.astype(float), nan=-1.0)
R24, RBODY = frac(2125, 4125), frac(5500, 7750)
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values
M = np.hstack([E, R24[:, None], RBODY[:, None]])
body = lambda te: (RBODY[list(te)] >= 0.9) & (cnt[list(te)] >= 3)


# %% 모델
class Res83:
    def __call__(self, te): return RES[list(te)] >= 0.7


class Either:
    def __init__(self, a, b): self.a, self.b = a, b
    def __call__(self, te): return self.a(te) | self.b(te)


RULES = lambda: [KnockRule(2.0, 0.7), Either(Rule83(0.75, 150), Res83())]


def rule_on(tr, te):
    """규칙 판정 (te,) bool. 유격 규칙의 정상 중심은 tr 정상으로 fit (폴드 안)"""
    return KnockRule(2.0, 0.7).fit(tr)(te) | Rule83(0.75, 150)(te) | Res83()(te)


class ETX:
    """ET 점수 = logit(나무 평균 확률), 입력 ext150 + 울림 2열. excl에 든 파일은 학습에서 뺀다 (채점은 그대로)"""
    def __init__(self, excl=()): self.excl = frozenset(excl)
    def __repr__(self): return f"ET[ext+r2, 뺌 {len(self.excl)}]"
    def __call__(self, tr, te):
        tr = np.array([i for i in tr if i not in self.excl]); te = np.array(te)
        p = np.clip(make_et(seed=0)().fit(M[tr], y[tr]).predict_proba(M[te])[:, 1], 1 / 600, 1 - 1 / 600)
        return np.log(p / (1 - p))[:, None]


class RoleSep(BaseEstimator):
    """EXP-034 c + 역할 분리. mode: 'none' = 기준 c · 'all' = 규칙 고장을 ET·로지스틱 학습에서 뺌 · 'et' = ET에서만 뺌
    규칙 고장 = 이 학습 파일들 안에서 rule_on이 켜진 고장. 규칙(+100)·본체 모드 내림은 기준과 같다"""
    def __init__(self, mode="none"): self.mode = mode
    def fit(self, Fm, yy):
        idx, yy = np.asarray(Fm)[:, 0].astype(int), np.asarray(yy)
        rf = rule_on(idx, idx) & (yy == 1)
        self.excl_ = idx[rf] if self.mode != "none" else np.array([], int)
        st_kw = dict(experts=(ae_mean, ETX(self.excl_)), combine="lr")
        keep = ~rf if self.mode == "all" else np.ones(len(idx), bool)
        self.m_ = RuleOverride(st_kw, RULES()).fit(np.asarray(Fm)[keep], yy[keep])
        # 유격 규칙의 정상 중심은 정상으로만 fit하므로 keep으로 고장을 빼도 같다
        st = self.m_.m_
        self.cap_ = float(np.median(st.m_.decision_function(st.oof_z_)[yy[keep] == 0]))
        return self
    def decision_function(self, Fm):
        s = self.m_.decision_function(Fm); te = np.asarray(Fm)[:, 0].astype(int)
        return np.where(body(te) & (s < 50), np.minimum(s, self.cap_), s)


CONFIGS = {"c | 기준 (EXP-034 c, 지금 최종)": "none", "r | 규칙 고장을 ET·로지스틱 학습에서 뺌": "all",
           "r_et | 규칙 고장을 ET 학습에서만 뺌": "et"}
MAIN = "r | 규칙 고장을 ET·로지스틱 학습에서 뺌"
on_all = rule_on(alli, alli)
log(f"[규칙 (150개 전체 기준)] 고장 {(on_all & ~nor).sum()}개: {who(on_all & ~nor)} | 정상 {who(on_all & nor)}")
warm_cache([ae_segerr], seeds=SEEDS, nested=True)


# %% 중첩 CV (EXP-035 · 036과 같은 분할) — 바깥 점수 + fold마다 고른 임계값
def job(mode, s, tr, te):
    make = lambda: RoleSep(mode)
    oof = _oof_scores(F[tr], y[tr], g[tr], make, StratifiedGroupKFold(4, shuffle=True, random_state=s))
    return te, _scores(make().fit(F[tr], y[tr]), F[te]), best_f2_threshold(y[tr], oof)


t0 = time.time()
OUTER = [(s, tr, te) for s in SEEDS for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
OUT, SC = {}, {}
for nm, mode in CONFIGS.items():
    with parallel_config(backend="loky", inner_max_num_threads=1):
        OUT[nm] = Parallel(n_jobs=-1)(delayed(job)(mode, *o) for o in OUTER)
    S = np.full((len(SEEDS), len(y)), np.nan)
    for (s, _, _), (te, sc, _) in zip(OUTER, OUT[nm]): S[s, te] = sc
    SC[nm] = S
    log(f"{nm} 계산 끝 {time.time() - t0:.0f}s")
np.savez(os.path.join(ex.figs, "outer_scores.npz"), **{k.split(" | ")[0]: v for k, v in SC.items()})


# %% 분리 지표 (미리 고정한 정의)
def sep_stats(S):
    rows = []
    for s in S:
        m = s < 50                                  # 규칙이 켜진 파일 제외
        sn, sa = s[m & nor], s[m & ~nor]
        a10, n95 = np.percentile(sa, 10), np.percentile(sn, 95)
        rows.append([average_precision_score(y[m], s[m]), roc_auc_score(y[m], s[m]), (a10 - n95) / sn.std(),
                     int((sn >= a10).sum() + (sa <= n95).sum()), int(m[~nor].sum())])
    return np.array(rows, float)


SEP = {}
for nm in CONFIGS:
    R = sep_stats(SC[nm]); SEP[nm] = R
    mu, sd = R.mean(0), R.std(0)
    log(f"[분리] {nm:38s} PR-AUC {mu[0]:.3f} ± {sd[0]:.3f} · ROC-AUC {mu[1]:.3f} ± {sd[1]:.3f} · 간격 {mu[2]:+.2f} ± {sd[2]:.2f} · 겹친 파일 {mu[3]:.1f} ± {sd[3]:.1f} (규칙 밖 고장 {mu[4]:.0f}개)")
    ex.note(f"분리 {nm.split(' | ')[0]}", dict(pr_auc=[float(mu[0]), float(sd[0])], roc_auc=[float(mu[1]), float(sd[1])],
                                              gap=[float(mu[2]), float(sd[2])], overlap=[float(mu[3]), float(sd[3])]))


def summarize(nm, r, tag):
    full = f"{tag}: {nm}"
    ex.record(full, r, names, main=(nm == MAIN and tag == "고정 1.0"))
    c = ex.res["configs"][full]
    log(f"{full:52s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} | 놓침 {dict(sorted(c['miss_counts'].items()))} | 오경보 {dict(sorted(c['fp_counts'].items()))}")


for nm in CONFIGS:
    summarize(nm, _collect([(te, (sc >= t).astype(int), t) for te, sc, t in OUT[nm]], y, list(SEEDS), 5), "fold마다")
for nm in CONFIGS:
    summarize(nm, _collect([(te, (sc >= FIXED).astype(int), FIXED) for te, sc, _ in OUT[nm]], y, list(SEEDS), 5), "고정 1.0")
log("경계 파일 바깥 점수 중앙값 (c / r / r_et): " + " | ".join(
    f"{short(k)} " + " / ".join(f"{np.median(SC[n][:, key.index(k)]):+.2f}" for n in CONFIGS) for k in FOCUS))


# %% 그림 1 (핵심): logistic_plane — c | r | r_et
def plane_z(mode):
    """OOF seed 0 평면 좌표 + 로지스틱 경계 학습에 쓴 파일 표시"""
    Z, rule, used = np.empty((len(y), 2)), np.zeros(len(y), bool), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        m = RoleSep(mode).fit(F[tr], y[tr])
        Z[te] = m.m_.m_.zscores(F[te])[:, :2]
        rule[te] = rule_on(tr, te)
    return Z, rule


PL = {}
for nm, mode in CONFIGS.items():
    PL[nm] = plane_z(mode)
    Z, rule = PL[nm]
    sh = ~(rule & ~nor)
    log(f"[평면 OOF seed 0] {nm:38s} 정상 ET z sd {Z[nor, 1].std():.2f} · AE z sd {Z[nor, 0].std():.2f} | 규칙 밖 고장 ET z 중앙값 {np.median(Z[sh & ~nor, 1]):+.2f}")


def plane(ax, nm):
    Z, rule = PL[nm]
    Zc = np.clip(Z, -3, 4.5); show = ~(rule & ~nor)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    fit_rows = show if CONFIGS[nm] == "all" else np.ones(len(y), bool)
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc[fit_rows], y[fit_rows]).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5); ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    cb = body(alli) & ~rule
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
    ax.scatter(Zc[cb, 0], Zc[cb, 1], s=70, marker="o", facecolor="none", edgecolor="#2ca02c", lw=1.0, label="본체 모드 내림 대상")
    c = ex.res["configs"]["고정 1.0: " + nm]
    wrong = [names.index(n) for n, k in {**c["miss_counts"], **c["fp_counts"]}.items() if k >= 1]
    ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=80, marker="x", color="k", lw=1.6, label="틀림 (고정 1.0, 1/5+)")
    for i in set(wrong) | {key.index(k) for k in FOCUS if not (rule[key.index(k)] and y[key.index(k)])}:
        k = c["miss_counts"].get(names[i]) or c["fp_counts"].get(names[i]) or 0
        ax.annotate(f"{short(names[i])}" + (f" {k}/5" if k else ""), Zc[i], xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=ABN if y[i] else NOR, weight="bold" if k else "normal")
    gone = [short(names[i]) for i in np.flatnonzero(rule & ~nor)]
    ax.text(.98, .02, f"규칙이 잡아 뺀 고장 {len(gone)}개:\n" + " ".join(gone[:7]) + ("\n" + " ".join(gone[7:]) if len(gone) > 7 else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (ext150 + 울림 2열)")
    mu = SEP[nm].mean(0)
    ax.set_title(f"{nm}\n고정 1.0: F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f}\n"
                 f"분리: PR-AUC {mu[0]:.3f} · 간격 {mu[2]:+.2f} · 겹침 {mu[3]:.1f} · 정상 ET z sd {PL[nm][0][nor, 1].std():.2f}", fontsize=9)


fig, axs = plt.subplots(1, 3, figsize=(24, 8.8))
for a, nm in zip(axs, CONFIGS): plane(a, nm)
axs[0].legend(fontsize=8, loc="upper left")
fig.suptitle("logistic_plane — 기준 | 규칙 고장을 ET·로지스틱 학습에서 뺌 | ET에서만 뺌 (OOF seed 0, 규칙 고장은 평면에서 뺌, 내림은 판정만 바꿈)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.96))
ex.save_fig(fig, "logistic_plane", "규칙 고장을 학습에서 빼면 정상 무리가 좁아지고 고장 무리와의 간격이 벌어지는가", key=True)

# %% 그림 2: 분리 지표 (seed 5개)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
labs = [n.split(" | ")[0] for n in CONFIGS]
for a, (j, t) in zip(axs, ((0, "PR-AUC (규칙 밖 파일, 높을수록 좋음)"), (2, "간격 = (고장 10%선 − 정상 95%선) / 정상 sd"), (3, "겹친 파일 수 (낮을수록 좋음)"))):
    for k, nm in enumerate(CONFIGS):
        v = SEP[nm][:, j]
        a.scatter(np.full(len(v), k) + np.linspace(-.12, .12, len(v)), v, color=f"C{k}", s=22)
        a.errorbar(k + .25, v.mean(), v.std(), fmt="o", color="k", ms=4, capsize=3)
    a.set_xticks(range(len(labs)), labs); a.set_title(t, fontsize=10, loc="left"); a.grid(alpha=.3, axis="y")
fig.tight_layout()
ex.save_fig(fig, "separation", "미리 정한 분리 지표 3개 (점 = 바깥 seed, 검은 점·막대 = 평균 ± sd)")

# %% 그림 3: 경계 파일 점수 (구성별)
fig, ax = plt.subplots(figsize=(12, 6))
for j, nm in enumerate(CONFIGS):
    for k, f in enumerate(FOCUS):
        i = key.index(f); v = SC[nm][:, i]
        if np.median(v) > 50: continue
        yy0 = k + (j - 1) * 0.22
        ax.plot([v.min(), v.max()], [yy0, yy0], color=f"C{j}", lw=2, alpha=.6); ax.scatter(np.median(v), yy0, color=f"C{j}", s=18, label=nm if k == 0 else None)
ax.set_yticks(range(len(FOCUS)), [("고장 " if f[0] == "a" else "정상 ") + f for f in FOCUS], fontsize=8)
ax.axvline(FIXED, color="#1a9e5a", lw=1.5, label="고정 임계값 1.0"); ax.set_xlim(-6, 8); ax.grid(alpha=.3, axis="x")
ax.set_xlabel("바깥 시험 점수 (점 = 5회 중앙, 선 = 최소–최대, 규칙으로 켜진 파일은 뺌)"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("경계 파일의 점수가 구성마다 어떻게 움직이나 — 정상은 왼쪽, 고장은 오른쪽이 좋음", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "edge_files", "구성별 경계 파일 바깥 점수")

# %% 그림 4: t-SNE (입력 공간 + 주 구성 점수 색) · 그림 5: psd_missed (맨 끝)
score = np.nanmedian(np.clip(SC[MAIN], -10, 10), 0)
fig, _ = viz.tsne({"ET 입력: ext150 + 울림 2열": M, "평면: AE z × ET z (r, OOF seed 0)": PL[MAIN][0]}, y, names, mark=list(FOCUS), score=score)
ex.save_fig(fig, "tsne", "입력 공간과 평면에서 경계 파일은 어디에 있는가 (2행 색 = r의 바깥 점수 중앙값)")
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, FOCUS), "psd_missed", "경계 파일의 PSD가 정상 범위 밖으로 나가는 주파수")
log(f"done {time.time() - t0:.0f}s")
