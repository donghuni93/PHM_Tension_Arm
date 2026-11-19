# %% [markdown]
# EXP-036 (3차, 11-17) 경계선 파일 — 9.5 kHz 단일 사건 내림 + 폴드 안 자동 내림 규칙(높은 정상 ↔ 고장을 가르는 물리 피처 1개)
# - 1차(figs/1차/): 9.5k 단일 사건 내림(e)으로 055 · 116 해소. n_ring 기각
# - 2차(figs/2차/): 결합에 min(AE z, ET z) → 005도 내려가 기각. e의 F2 평평 구간 0.68–1.09, 고정 0.85에서 005 · 025 5/5 (사후)
# - 3차 문제: 남은 오경보 054 · 059 · 027 · 039 · 065를 손으로 고른 규칙으로 가르면 같은 150개를 보고 고르는 사후 조정이 된다
#   → 규칙 찾기 자체를 학습 fold 안에서만 한다 (AutoCap). 중첩 CV 점수가 정직하고, fold마다 같은 피처가 뽑히는지로 신호인지 우연인지 본다
# - AutoCap.fit(학습 fold):
#   1) 기본 모델(e) 학습 → 안쪽 OOF 결합 점수
#   2) '높은 정상' = 학습 정상 중 OOF 점수 상위 k개 (이미 본체 · 9.5k 내림 대상이면 뺌)
#   3) 후보 물리 피처(아래 POOL)마다 학습 정상 중앙값 · MAD로 맞춘 값 v, 방향 ±. 학습 고장 **전부**(규칙 고장 포함)의 최대 v보다
#      margin(0.25 MAD) 넘게 높은 '높은 정상'이 min_cover개 이상이면 후보. 덮은 수 → 여유 순으로 1개 고름, 남은 높은 정상으로 1개 더(최대 2개)
#      문턱 = 가장 높은 고장과 덮인 정상 중 가장 낮은 값의 가운데
#   4) 시험 파일이 고른 규칙에 걸리면 본체 내림과 같이 정상 OOF 중앙값 이하로 (규칙으로 켜진 +100 파일은 그대로)
# - POOL: ext150 · phys(H2 · H3 모양 · H4) · impact(B · E) · knock(K1–K3) · hits(큰 타격 · 사건 울림) · 충격 울림 요약(2-16) · s95 · c95
# - 구성: c(지금 최종) · e(9.5k 내림) · x = e + AutoCap(k=5) · x8 = e + AutoCap(k=8)
#   · y = e + 고주파 충격 내림(고정): 1회차 실행에서 x · x8이 0.90 · 0.88로 실패(fold마다 다른 피처, 시험 fold에서 고장 015 5/5 내림)한 뒤,
#     150개 전체 AutoCap가 찾은 단 하나의 규칙(충격 중 9 kHz 이상으로 울린 비율 ≥ 0.72, 여유 3.5 MAD)을 고정해 확인. 사후
# - 판정: fold마다 / 고정 1.0 / 고정 0.85 (+ 고정 임계값 훑기)
# - 가설: H1 · H2 · H3 (어느 피처가 뽑히는지로 사후 해석)
# 실행: conda run -n qe-phm python "단계/5_모델비교/base/EXP-036_1117_경계파일_9.5k단일사건·울림다양성/run.py"

# %% 설정
import os, sys, time, warnings
from collections import Counter
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
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed, parallel_config
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
FOCUS = ("abn_005", "abn_025", "abn_015", "abn_020", "abn_022", "abn_014", "nor_025", "nor_054", "nor_055", "nor_059", "nor_060",
         "nor_106", "nor_116", "nor_103", "nor_027", "nor_039", "nor_065")
ex = Experiment(__file__, "EXP-036", "경계선 파일 — 9.5 kHz 단일 사건 내림 + 폴드 안 자동 내림 규칙", ["H2", "H1", "H3"],
                member="이동훈", model="EXP-034 c + 9.5 kHz 단일 사건 내림 + 폴드 안 자동 내림(높은 정상 ↔ 고장 피처 1–2개)",
                changes="9.25–10 kHz 에너지 몫 ≥ 10^−1 & 한 사건(±25 ms) 몫 ≥ 0.7 → 정상 쪽 내림 / 학습 fold에서 점수 높은 정상을 모든 고장과 가르는 물리 피처를 찾아 내림",
                focus=FOCUS)
X, y, names = load_train(); names = [str(n) for n in names]; Xf = np.asarray(X, float)
g = np.arange(len(y)); F = g[:, None]; nor = y == 0; alli = np.arange(len(y))
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
key = [os.path.splitext(os.path.basename(n))[0] for n in names]
FS, SEEDS = 25000, range(5)
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


def who(m):
    return " ".join(short(names[i]) for i in np.flatnonzero(m)) or "없음"


# %% 입력: EXP-034 c 표 + 9.5k (1차와 같음)
E = cached_table("ext", feature_table_ext, X).values
IM = pd.read_csv(os.path.join(_d, "단계", "2_EDA", "8kHz공진", "2-16_1116_울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
cnt = IM.groupby("file").size().reindex(key).fillna(0).values
frac = lambda lo, hi: np.nan_to_num(IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean().reindex(key).values.astype(float), nan=-1.0)
R24, RBODY = frac(2125, 4125), frac(5500, 7750)
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values
B95 = sosfiltfilt(butter(4, [9250, 10000], "bandpass", fs=FS, output="sos"), Xf, axis=1) ** 2
HP = sosfiltfilt(butter(4, 200, "highpass", fs=FS, output="sos"), Xf, axis=1) ** 2
c_ = np.cumsum(B95, 1); i_ = B95.argmax(1); r_ = np.arange(len(y)); W_ = int(0.025 * FS)
S95 = np.log10(c_[:, -1] / HP.sum(1))
C95 = (c_[r_, np.clip(i_ + W_, None, B95.shape[1] - 1)] - c_[r_, np.clip(i_ - W_, 0, None)]) / c_[:, -1]
one95 = lambda te: (S95[list(te)] >= -1.0) & (C95[list(te)] >= 0.7)
# 고주파 충격 내림 (3차 추가, 150개 전체 AutoCap가 찾은 규칙을 고정): 충격 중 9 kHz 이상으로 울린 비율 ≥ 0.72
# (문턱 0.5–1.0 모두 정상 043 · 059 · 065만, 고장 0. 1.5초 창 11개에서 고장 0 · 1.0초 창에서는 021 · 024 · 026이 일부 창에서 듦)
FHI = IM.assign(h=IM.f >= 9000).groupby("file").h.mean().reindex(key).fillna(0).values
fhi = lambda te: FHI[list(te)] >= 0.72
body = lambda te: (RBODY[list(te)] >= 0.9) & (cnt[list(te)] >= 3)

# 후보 피처 풀 (라벨 없이 만든 표만)
C = os.path.join(_d, "outputs", "cache")
tabs = {"ext": pd.read_csv(os.path.join(C, "feat_ext.csv")), "phys": pd.read_csv(os.path.join(C, "feat_phys.csv")),
        "impact": pd.read_csv(os.path.join(C, "feat_impact.csv")), "knock": pd.read_csv(os.path.join(C, "feat_knock.csv")),
        "hits": pd.read_csv(os.path.join(C, "feat_hits.csv"))}


def ring_summary(d):
    if len(d) == 0:
        return dict(rg_n=0, rg_nfreq=0, rg_zhi=0, rg_h2=0, rg_toph=0, rg_topz=0, rg_topfit=0, rg_flo=0, rg_fhi=0)
    h = d.height.values; top = d.loc[d.height.idxmax()]
    return dict(rg_n=len(d), rg_nfreq=(d.f / 250).round().nunique(), rg_zhi=(d.zeta >= 0.04).mean(), rg_h2=(np.sort(h)[-2] / h.max() if len(h) > 1 else 0),
                rg_toph=np.log10(top.height), rg_topz=top.zeta, rg_topfit=top.fit, rg_flo=(d.f < 2125).mean(), rg_fhi=(d.f >= 9000).mean())


RG = pd.DataFrame([ring_summary(IM[IM.file == k]) for k in key])
POOL = pd.concat([t.reset_index(drop=True) for t in tabs.values()] + [RG, pd.DataFrame(dict(s95=S95, c95=C95))], axis=1)
POOL = POOL.loc[:, POOL.std() > 1e-12]
PN, PV = list(POOL.columns), POOL.values.astype(float)
log(f"후보 피처 {len(PN)}개")


# %% 모델
class ETX:
    def __call__(self, tr, te):
        tr, te = np.array(tr), np.array(te)
        M = np.hstack([E, R24[:, None], RBODY[:, None]])
        p = np.clip(make_et(seed=0)().fit(M[tr], y[tr]).predict_proba(M[te])[:, 1], 1 / 600, 1 - 1 / 600)
        return np.log(p / (1 - p))[:, None]


class Res83:
    def __call__(self, te): return RES[list(te)] >= 0.7


class Either:
    def __init__(self, a, b): self.a, self.b = a, b
    def __call__(self, te): return self.a(te) | self.b(te)


class AutoCap(BaseEstimator):
    """기본(본체 내림 + 9.5k 내림) + 폴드 안에서 찾은 내림 규칙 0–max_rules개"""
    def __init__(self, one=True, k=0, min_cover=2, margin=0.25, max_rules=2, hi=False):
        self.one, self.k, self.min_cover, self.margin, self.max_rules, self.hi = one, k, min_cover, margin, max_rules, hi

    def fit(self, F, yy):
        yy = np.asarray(yy); idx = np.asarray(F)[:, 0].astype(int)
        self.m_ = RuleOverride(dict(experts=(ae_mean, ETX()), combine="lr"), [KnockRule(2.0, 0.7), Either(Rule83(0.75, 150), Res83())]).fit(F, yy)
        st = self.m_.m_; oof = st.m_.decision_function(st.oof_z_)
        self.cap_ = float(np.median(oof[yy == 0])); self.rules_ = []
        if self.k:
            V = PV[idx]; n = V[yy == 0]; med = np.median(n, 0); sc = 1.4826 * np.median(np.abs(n - med), 0)
            sc = np.where(sc > 1e-9, sc, n.std(0) + 1e-9)
            Z = (V - med) / sc
            capped = body(idx) | (one95(idx) if self.one else False)
            cand = np.flatnonzero((yy == 0) & ~capped)
            hard = list(cand[np.argsort(-oof[cand])[:self.k]])
            for _ in range(self.max_rules):
                best = None
                for j in range(Z.shape[1]):
                    for d in (1, -1):
                        v = d * Z[:, j]; fm = v[yy == 1].max(); hv = v[hard]; cov = hv > fm + self.margin
                        if cov.sum() >= self.min_cover:
                            sc_ = (int(cov.sum()), float(hv[cov].min() - fm))
                            if best is None or sc_ > best[0]: best = (sc_, j, d, (fm + hv[cov].min()) / 2, [hard[i] for i in np.flatnonzero(cov)])
                if best is None: break
                (_, gap), j, d, t, covd = best
                self.rules_.append(dict(feat=PN[j], j=j, d=d, thr_raw=med[j] + d * t * sc[j], gap=gap, covered=[idx[i] for i in covd]))
                hard = [h for h in hard if h not in covd]
                if len(hard) < self.min_cover: break
        return self

    def mined(self, te):
        te = np.asarray(te); m = np.zeros(len(te), bool)
        for r in self.rules_:
            m |= (r["d"] * (PV[te, r["j"]] - r["thr_raw"]) > 0)
        return m

    def decision_function(self, F):
        s = self.m_.decision_function(F); te = np.asarray(F)[:, 0].astype(int)
        m = body(te) | (one95(te) if self.one else False) | (fhi(te) if self.hi else False) | self.mined(te)
        return np.where(m & (s < 50), np.minimum(s, self.cap_), s)


CONFIGS = {"c | 기준 (EXP-034 c, 지금 최종)": lambda: AutoCap(one=False, k=0), "e | c + 9.5k 단일 사건 내림": lambda: AutoCap(one=True, k=0),
           "x | e + 폴드 안 자동 내림 (k=5)": lambda: AutoCap(one=True, k=5), "x8 | e + 폴드 안 자동 내림 (k=8)": lambda: AutoCap(one=True, k=8),
           "y | e + 고주파 충격 내림 (고정)": lambda: AutoCap(one=True, k=0, hi=True)}
MAIN = "y | e + 고주파 충격 내림 (고정)"
warm_cache([ae_segerr], seeds=SEEDS, nested=True)


# %% 중첩 CV — 바깥 점수 + fold마다 고른 임계값 + 바깥 모델이 찾은 규칙
def job(make, s, tr, te):
    oof = _oof_scores(F[tr], y[tr], g[tr], make, StratifiedGroupKFold(4, shuffle=True, random_state=s))
    m = make().fit(F[tr], y[tr])
    return te, _scores(m, F[te]), best_f2_threshold(y[tr], oof), m.rules_


t0 = time.time()
OUTER = [(s, tr, te) for s in SEEDS for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
OUT, SC, RULES = {}, {}, {}
for nm, make in CONFIGS.items():
    with parallel_config(backend="loky", inner_max_num_threads=1):
        res = Parallel(n_jobs=-1)(delayed(job)(make, *o) for o in OUTER)
    OUT[nm] = [r[:3] for r in res]; RULES[nm] = [r[3] for r in res]
    S = np.full((len(SEEDS), len(y)), np.nan)
    for (s, _, _), (te, sc, _) in zip(OUTER, OUT[nm]): S[s, te] = sc
    SC[nm] = S
    log(f"{nm} 계산 끝 {time.time() - t0:.0f}s")
np.savez(os.path.join(ex.figs, "outer_scores.npz"), **{k.split(" | ")[0]: v for k, v in SC.items()})


def summarize(nm, r, tag):
    full = f"{tag}: {nm}"
    ex.record(full, r, names, main=(nm == MAIN and tag == "fold마다"))
    c = ex.res["configs"][full]
    log(f"{full:48s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} | 놓침 {dict(sorted(c['miss_counts'].items()))} | 오경보 {dict(sorted(c['fp_counts'].items()))}")


fixed = lambda nm, t: _collect([(te, (sc >= t).astype(int), t) for te, sc, _ in OUT[nm]], y, list(SEEDS), 5)
for nm in CONFIGS:
    summarize(nm, _collect([(te, (sc >= t).astype(int), t) for te, sc, t in OUT[nm]], y, list(SEEDS), 5), "fold마다")
for t in (1.0, 0.85):
    for nm in CONFIGS:
        summarize(nm, fixed(nm, t), f"고정 {t}")
for nm in CONFIGS:
    S = SC[nm]; cand = np.unique(np.round(S[S < 50], 4)); f2m = np.array([fixed(nm, t)["f2"].mean() for t in cand])
    top = cand[f2m >= f2m.max() - 0.005]
    log(f"  [{nm}] 고정 최적값(사후) {cand[f2m.argmax()]:.3f} → F2 {f2m.max():.3f} | F2 ≥ 최대−0.005 구간 {top.min():.2f}–{top.max():.2f}")

# 바깥 fold 25개에서 뽑힌 규칙의 안정성
for nm in ("x | e + 폴드 안 자동 내림 (k=5)", "x8 | e + 폴드 안 자동 내림 (k=8)"):
    cf = Counter(r["feat"] for rs in RULES[nm] for r in rs); cv = Counter(short(names[i]) for rs in RULES[nm] for r in rs for i in r["covered"])
    log(f"[{nm}] 규칙 수 fold당 {np.mean([len(rs) for rs in RULES[nm]]):.1f} · 뽑힌 피처(25 fold 중) {dict(cf.most_common(10))}")
    log(f"   덮인 높은 정상 {dict(cv.most_common(12))}")
    caught = Counter()
    for (s, tr, te), (te2, sc, _), rs in zip(OUTER, OUT[nm], RULES[nm]):
        m = AutoCap(); m.rules_ = rs
        on = m.mined(te)
        for i in np.asarray(te)[on]: caught[("고장 " if y[i] else "정상 ") + short(names[i])] += 1
    log(f"   시험 fold에서 자동 규칙에 걸린 파일 {dict(sorted(caught.items(), key=lambda v: -v[1]))}")

# 150개 전체로 찾은 규칙 (참고)
full = AutoCap(one=True, k=5).fit(F, y)
log("[150개 전체, k=5] " + " | ".join(f"{r['feat']} {'≥' if r['d'] > 0 else '≤'} {r['thr_raw']:.3g} (여유 {r['gap']:.2f} MAD, 덮음 {' '.join(short(names[i]) for i in r['covered'])})" for r in full.rules_))
log("  전체 규칙에 걸리는 파일: " + who(full.mined(alli)))
log("경계 파일 바깥 점수 중앙 (c / e / x / x8): " + " | ".join(f"{short(k)} " + " / ".join(f"{np.median(SC[n][:, key.index(k)]):+.2f}" for n in CONFIGS) for k in FOCUS))

# %% 그림 1 (핵심): 고정 임계값 훑기
labs = list(CONFIGS)
GRID = np.round(np.arange(-1.5, 2.51, 0.05), 3)
fig, ax = plt.subplots(figsize=(11, 5))
for j, nm in enumerate(labs):
    sw = np.array([fixed(nm, t)["f2"] for t in GRID])
    ax.plot(GRID, sw.mean(1), color=f"C{j}", lw=1.6, label=nm); ax.fill_between(GRID, sw.mean(1) - sw.std(1), sw.mean(1) + sw.std(1), color=f"C{j}", alpha=.12, lw=0)
for t in (1.0, 0.85): ax.axvline(t, color="#1a9e5a", lw=1, ls=":")
ax.set_ylim(0.85, 1.0); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower left")
ax.set_xlabel("고정 임계값 (로지스틱 로짓, 점선 = 0.85 · 1.0)"); ax.set_ylabel("중첩 F2 (seed 5개 평균 ± sd)")
ax.set_title("임계값을 하나로 고정할 때 — 꼭대기가 넓고 평평할수록 새 데이터에서 덜 흔들림", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "threshold_sweep", "구성별 고정 임계값 F2 곡선 (꼭대기 높이와 평평한 폭)", key=True)

# %% 그림 2 (핵심): 자동 규칙의 안정성 — 어느 피처가 몇 fold에서 뽑혔나, 시험 fold에서 무엇을 내렸나
nm = "x | e + 폴드 안 자동 내림 (k=5)"
cf = Counter(r["feat"] for rs in RULES[nm] for r in rs).most_common(12)
fig, axs = plt.subplots(1, 2, figsize=(16, 5.5))
a = axs[0]; a.barh(range(len(cf)), [v for _, v in cf], color="#555"); a.set_yticks(range(len(cf)), [k for k, _ in cf], fontsize=8); a.invert_yaxis()
a.set_xlabel("뽑힌 바깥 fold 수 (25개 중)"); a.set_title("① 학습 fold마다 고른 내림 피처 — 한 피처가 반복되면 신호, 흩어지면 우연", loc="left", fontsize=10); a.grid(alpha=.3, axis="x")
caught = Counter()
for (s, tr, te), rs in zip(OUTER, RULES[nm]):
    m = AutoCap(); m.rules_ = rs
    for i in np.asarray(te)[m.mined(te)]: caught[names[i]] += 1
ck = sorted(caught.items(), key=lambda v: -v[1])[:20]
a = axs[1]; a.barh(range(len(ck)), [v for _, v in ck], color=[ABN if k.startswith("abn") else NOR for k, _ in ck])
a.set_yticks(range(len(ck)), [short(k) for k, _ in ck], fontsize=8); a.invert_yaxis(); a.grid(alpha=.3, axis="x")
a.set_xlabel("시험 fold에서 자동 규칙에 걸린 횟수 (seed 5번 중)"); a.set_title("② 시험 파일 중 내려간 파일 — 주황(고장)이 있으면 위험", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "autocap_rules", "폴드 안에서 찾은 내림 규칙이 fold마다 같은 피처인가, 시험 fold에서 고장을 내리지 않는가", key=True)


# %% 그림 3 (핵심): logistic_plane — e | x (내림 대상 표시: x는 150개 전체로 찾은 규칙)
def plane(ax, title, mined):
    Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae_mean, ETX()), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        rule[te] = KnockRule(2.0, 0.7).fit(tr)(te) | Rule83(0.75, 150)(te) | Res83()(te)
    Zc = np.clip(Z, -3, 4.5)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5); ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    show = ~(rule & (y == 1))
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
    cb = body(alli) & ~rule; c1 = one95(alli) & ~rule; cm = mined & ~rule
    ax.scatter(Zc[cb, 0], Zc[cb, 1], s=70, marker="o", facecolor="none", edgecolor="#2ca02c", lw=1.0, label="본체 모드 내림")
    ax.scatter(Zc[c1, 0], Zc[c1, 1], s=150, marker="o", facecolor="none", edgecolor="#7a3db8", lw=1.8, label="9.5k 단일 사건 내림")
    if cm.any(): ax.scatter(Zc[cm, 0], Zc[cm, 1], s=200, marker="D", facecolor="none", edgecolor="#d62728", lw=1.5, label="고주파 충격 내림 (충격의 ≥ 72%가 9 kHz 이상)")
    c = ex.res["configs"]["고정 0.85: " + title]
    wrong = [names.index(n) for n, k in {**c["miss_counts"], **c["fp_counts"]}.items() if k >= 1]
    ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=80, marker="x", color="k", lw=1.6, label="틀림 (고정 0.85, 1/5+)")
    for i in set(wrong) | {key.index(k) for k in FOCUS if not (rule[key.index(k)] and y[key.index(k)])}:
        k = c["miss_counts"].get(names[i]) or c["fp_counts"].get(names[i]) or 0
        ax.annotate(f"{short(names[i])}" + (f" {k}/5" if k else ""), Zc[i], xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=ABN if y[i] else NOR, weight="bold" if k else "normal")
    gone = [short(names[i]) for i in np.flatnonzero(rule & (y == 1))]
    ax.text(.98, .02, f"규칙이 잡아 뺀 고장 {len(gone)}개:\n" + " ".join(gone[:7]) + ("\n" + " ".join(gone[7:]) if len(gone) > 7 else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (ext150 + 울림 2열)")
    cf_ = ex.res["configs"]["fold마다: " + title]
    ax.set_title(f"{title}\nfold마다 F2 {cf_['f2']:.3f} ± {cf_['f2_sd']:.3f} · 고정 0.85 F2 {c['f2']:.3f} (오경보율 {c['fpr']:.3f})", fontsize=9.5)


fig, axs = plt.subplots(1, 2, figsize=(17, 8.6))
plane(axs[0], "e | c + 9.5k 단일 사건 내림", np.zeros(len(y), bool))
plane(axs[1], MAIN, fhi(alli))
axs[0].legend(fontsize=8, loc="upper left"); axs[1].legend(fontsize=8, loc="upper left")
fig.suptitle("logistic_plane — 9.5k 내림 | + 고주파 충격 내림 (OOF seed 0, 내림은 평면 위치를 바꾸지 않고 판정만 내림)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
ex.save_fig(fig, "logistic_plane", "고주파 충격 내림 대상(빨간 마름모)이 오경보 059 · 065와 겹치는가, 남은 오경보 054 · 027 · 039와 경계 고장 005 · 025는 어디에 있는가", key=True)

# %% 그림 4: 경계 파일 점수 · 그림 5: t-SNE · 그림 6: psd_missed (맨 끝)
fig, ax = plt.subplots(figsize=(13, 8))
for j, nm in enumerate(labs):
    for k, f in enumerate(FOCUS):
        i = key.index(f); v = SC[nm][:, i]
        if np.median(v) > 50: continue
        yy0 = k + (j - 1.5) * 0.18
        ax.plot([v.min(), v.max()], [yy0, yy0], color=f"C{j}", lw=2, alpha=.6); ax.scatter(np.median(v), yy0, color=f"C{j}", s=18, label=nm if k == 0 else None)
ax.set_yticks(range(len(FOCUS)), [("고장 " if f[0] == "a" else "정상 ") + f for f in FOCUS], fontsize=8)
ax.axvline(0.85, color="#1a9e5a", lw=1.5, label="고정 0.85"); ax.set_xlim(-6, 6); ax.grid(alpha=.3, axis="x")
ax.set_xlabel("바깥 시험 점수 (점 = 5회 중앙, 선 = 최소–최대, 규칙으로 켜진 파일은 뺌)"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("경계 파일의 점수 — 정상은 왼쪽, 고장은 오른쪽이 좋음", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "edge_files", "구성별 경계 파일 바깥 점수")
Ps = (PV - PV.mean(0)) / (PV.std(0) + 1e-9)
fig, _ = viz.tsne({f"후보 피처 풀 {len(PN)}개 (표준화)": Ps}, y, names, mark=list(FOCUS))
ex.save_fig(fig, "tsne", "자동 규칙이 고르는 피처 풀에서 경계 파일은 어디에 있는가")
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, FOCUS), "psd_missed", "경계 파일의 PSD가 정상 범위 밖으로 나가는 주파수")
log(f"done {time.time() - t0:.0f}s")
