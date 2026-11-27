# %% [markdown]
# EXP-036 (11-17) 경계선 파일 — 9.5 kHz 단일 사건 내림 · 울림 주파수 다양성
# - 출발: 사용자 목표(11-17) "고장 005 · 025를 올리거나, 오경보 정상 025 054 055 059 060 106 116을 내리는 방법"
#   경계 파일 = EXP-035 바깥 점수 중앙값이 임계값(1.0) 근처인 파일 (정상 0–1.1, 고장 1.1–2.8)
# - 단서 1 (사용자, 2-0_2 결론 4): 정상 055 · 116은 H3 9.5–9.75 kHz 모양이 독보적. 그러나 L1 고장(003 007 013 026)과 020도 높다
#   → 9.25–10 kHz 대역 에너지가 '한 사건에 몰렸나(정상: 외부 타격 한 번이 9.5 kHz 모드를 울림)' vs
#     '여러 충격에 퍼졌나(고장: 충격마다 고주파로 울림, H2)'로 가른다
#   s95 = log10(9.25–10 kHz 에너지 / 200 Hz 이상 에너지), c95 = 그 대역 에너지 중 최대 순간 ±25 ms 안의 몫
#   내림 [9.5k 단일 사건]: s95 ≥ −1.0 & c95 ≥ 0.7 → 결합 점수를 학습 정상 OOF 중앙값 이하로 (본체 모드 내림과 같은 방식)
#   (학습 150개 탐색: s95 −1.2~−0.8 × c95 0.7~0.85 모든 조합에서 정상 055 · 116만, 고장 0. 9.5k 몫이 높은 고장의 c95 ≤ 0.43)
# - 단서 2 (경계 구간 탐색): 고장은 울림 주파수 종류가 많다 (경계 21개에서 AUC 0.86). 유격·손상 부품이 여러 모드를 울림 (H1)
#   ET 입력에 1열: n_ring = 충격들이 울린 서로 다른 주파수(250 Hz 격자) 개수 (2-16 impacts.csv, 충격 0개 = 0)
# - 구성: c(기준, EXP-034 c = 지금 최종) · e = c + 9.5k 단일 사건 내림 · f = c + ET n_ring · g = e + f
# - 판정: ① fold마다 안쪽 OOF로 임계값 (지금 검증 방식) ② 제출용 고정 1.0 (EXP-035 결정) 둘 다
# - 9.5k 규칙의 측정 구간 민감도: 1.5초 창 4개(시작 0 · 0.17 · 0.33 · 0.5초)에서 켜지는 파일
# - 주의: 대역(9.25–10 kHz)은 2-0_2에서 라벨을 본 뒤 고른 칸, 문턱도 사후. 지지 파일이 정상 2개뿐
# - 가설: H2(고주파 링잉 = 고장, 사건 하나 = 정상) · H1(울림 다양성)
# 실행: conda run -n qe-phm python "3_단계/5_모델실험/스크리닝/EXP-036_1117_경계파일_9.5k단일사건·울림다양성/run.py"

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
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed, parallel_config
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
FOCUS = ("abn_005", "abn_025", "abn_015", "abn_020", "abn_022", "abn_014", "nor_025", "nor_054", "nor_055", "nor_059", "nor_060", "nor_106", "nor_116", "nor_103")
ex = Experiment(__file__, "EXP-036", "경계선 파일 — 9.5 kHz 단일 사건 내림 · 울림 주파수 다양성", ["H2", "H1"],
                member="이동훈", model="EXP-034 c + 9.5 kHz 단일 사건 내림 (+ ET 울림 다양성 1열)",
                changes="9.25–10 kHz 에너지 몫 ≥ 10^−1 & 한 사건(±25 ms) 몫 ≥ 0.7 → 정상 쪽 내림 / ET에 울림 주파수 종류 수",
                focus=FOCUS)
X, y, names = load_train(); names = [str(n) for n in names]; Xf = np.asarray(X, float)
g = np.arange(len(y)); F = g[:, None]; nor = y == 0; alli = np.arange(len(y))
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
key = [os.path.splitext(os.path.basename(n))[0] for n in names]
FS, SEEDS, FIXED = 25000, range(5), 1.0
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


def who(m):
    return " ".join(short(names[i]) for i in np.flatnonzero(m)) or "없음"


# %% 입력: EXP-034 c와 같은 표 + 새 값 (s95 · c95 · n_ring)
E = cached_table("ext", feature_table_ext, X).values
IM = pd.read_csv(os.path.join(_d, "3_단계", "2_EDA", "2-3_충격울림주파수", "자료", "figs", "impacts.csv"), encoding="utf-8-sig")
cnt = IM.groupby("file").size().reindex(key).fillna(0).values
frac = lambda lo, hi: np.nan_to_num(IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean().reindex(key).values.astype(float), nan=-1.0)
R24, RBODY = frac(2125, 4125), frac(5500, 7750)
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values
NRING = IM.assign(fb=(IM.f / 250).round()).groupby("file").fb.nunique().reindex(key).fillna(0).values.astype(float)

SOS95 = butter(4, [9250, 10000], "bandpass", fs=FS, output="sos")
SOSHP = butter(4, 200, "highpass", fs=FS, output="sos")
B95, HP = sosfiltfilt(SOS95, Xf, axis=1) ** 2, sosfiltfilt(SOSHP, Xf, axis=1) ** 2


def s95c95(a=0, b=None, w=0.025):
    """구간 [a, b) 샘플에서 s95 = log10(9.25–10 kHz 에너지 / 200 Hz 이상 에너지), c95 = 대역 에너지 중 최대 순간 ±w초 몫"""
    Z = B95[:, a:b]; c = np.cumsum(Z, 1); W = int(w * FS)
    i = Z.argmax(1); lo = np.clip(i - W, 0, None); hi = np.clip(i + W, None, Z.shape[1] - 1)
    r = np.arange(len(i))
    return np.log10(c[:, -1] / HP[:, a:b].sum(1)), (c[r, hi] - c[r, lo]) / c[:, -1]


S95, C95 = s95c95()
S95_MIN, C95_MIN = -1.0, 0.7
one95 = lambda te, s=S95_MIN, c=C95_MIN: (S95[list(te)] >= s) & (C95[list(te)] >= c)
body = lambda te: (RBODY[list(te)] >= 0.9) & (cnt[list(te)] >= 3)

on = one95(alli); log(f"[9.5k 단일 사건] 정상 {who(on & nor)} · 고장 {who(on & ~nor)}")
for s, c in ((-1.2, 0.56), (-0.8, 0.84), (-1.2, 0.84), (-0.8, 0.56)):
    o = one95(alli, s, c); log(f"  문턱 s95 ≥ {s} · c95 ≥ {c}: 정상 {who(o & nor)} · 고장 {who(o & ~nor)}")
hi95 = S95 >= -1.3
log("  9.5k 몫 상위(s95 ≥ −1.3) 파일의 c95: " + " ".join(f"{short(names[i])}({C95[i]:.2f})" for i in np.flatnonzero(hi95)))
# 측정 구간 민감도: 1.5초 창 4개
WIN = [(int(t * FS), int(t * FS) + int(1.5 * FS)) for t in (0, 0.17, 0.33, 0.5)]
WON = np.array([((lambda sc: (sc[0] >= S95_MIN) & (sc[1] >= C95_MIN))(s95c95(a, b))) for a, b in WIN])
log(f"  1.5초 창 4개: 한 번이라도 켜짐 정상 {who(WON.any(0) & nor)} · 고장 {who(WON.any(0) & ~nor)} | 4개 모두 켜짐 {who(WON.all(0))}")
log(f"[n_ring] 정상 중앙 {np.median(NRING[nor]):.0f} · 고장 중앙 {np.median(NRING[~nor]):.0f} | 경계 파일: " +
    " ".join(f"{short(k)}={NRING[key.index(k)]:.0f}" for k in FOCUS))


# %% 모델 (EXP-034 c 정의 + 선택)
class ETX:
    def __init__(self, nring=False): self.nring = nring
    def __repr__(self): return f"ET[ext+r2{'+n_ring' if self.nring else ''}]"
    def __call__(self, tr, te):
        tr, te = np.array(tr), np.array(te)
        M = np.hstack([E, R24[:, None], RBODY[:, None]] + ([NRING[:, None]] if self.nring else []))
        p = np.clip(make_et(seed=0)().fit(M[tr], y[tr]).predict_proba(M[te])[:, 1], 1 / 600, 1 - 1 / 600)
        return np.log(p / (1 - p))[:, None]


class Res83:
    def __call__(self, te): return RES[list(te)] >= 0.7


class Either:
    def __init__(self, a, b): self.a, self.b = a, b
    def __call__(self, te): return self.a(te) | self.b(te)


class Cap(BaseEstimator):
    """RuleOverride + 내림: cap 조건(본체 모드 · 9.5k 단일 사건)인 파일은 학습 정상 OOF 결합 점수 중앙값 이하로. 규칙으로 켜진 파일(+100)은 그대로"""
    def __init__(self, base_kw=None, rule=None, one=False): self.base_kw, self.rule, self.one = base_kw, rule, one
    def fit(self, F, yy):
        yy = np.asarray(yy); self.m_ = RuleOverride(self.base_kw, self.rule).fit(F, yy); st = self.m_.m_
        self.cap_ = float(np.median(st.m_.decision_function(st.oof_z_)[yy == 0])); return self
    def decision_function(self, F):
        s = self.m_.decision_function(F); te = np.asarray(F)[:, 0].astype(int)
        m = body(te) | (one95(te) if self.one else False)
        return np.where(m & (s < 50), np.minimum(s, self.cap_), s)


RULES = lambda: [KnockRule(2.0, 0.7), Either(Rule83(0.75, 150), Res83())]
mk = lambda nring, one: (lambda: Cap(dict(experts=(ae_mean, ETX(nring)), combine="lr"), RULES(), one))
CONFIGS = {"c | 기준 (EXP-034 c, 지금 최종)": mk(False, False), "e | c + 9.5k 단일 사건 내림": mk(False, True),
           "f | c + ET n_ring": mk(True, False), "g | e + f": mk(True, True)}
MAIN = "e | c + 9.5k 단일 사건 내림"
warm_cache([ae_segerr], seeds=SEEDS, nested=True)


# %% 중첩 CV (EXP-035와 같은 분할) — 바깥 점수 + fold마다 고른 임계값
def job(make, s, tr, te):
    oof = _oof_scores(F[tr], y[tr], g[tr], make, StratifiedGroupKFold(4, shuffle=True, random_state=s))
    return te, _scores(make().fit(F[tr], y[tr]), F[te]), best_f2_threshold(y[tr], oof)


t0 = time.time()
OUTER = [(s, tr, te) for s in SEEDS for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
OUT, SC = {}, {}
for nm, make in CONFIGS.items():
    with parallel_config(backend="loky", inner_max_num_threads=1):
        OUT[nm] = Parallel(n_jobs=-1)(delayed(job)(make, *o) for o in OUTER)
    S = np.full((len(SEEDS), len(y)), np.nan)
    for (s, _, _), (te, sc, _) in zip(OUTER, OUT[nm]): S[s, te] = sc
    SC[nm] = S
    log(f"{nm} 계산 끝 {time.time() - t0:.0f}s")
np.savez(os.path.join(ex.figs, "outer_scores.npz"), **{k.split(" | ")[0]: v for k, v in SC.items()})


def summarize(nm, r, tag):
    full = f"{tag}: {nm}"
    ex.record(full, r, names, main=(nm == MAIN and tag == "고정 1.0"))
    c = ex.res["configs"][full]
    log(f"{full:46s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} | 놓침 {dict(sorted(c['miss_counts'].items()))} | 오경보 {dict(sorted(c['fp_counts'].items()))}")


for nm in CONFIGS:
    summarize(nm, _collect([(te, (sc >= t).astype(int), t) for te, sc, t in OUT[nm]], y, list(SEEDS), 5), "fold마다")
for nm in CONFIGS:
    summarize(nm, _collect([(te, (sc >= FIXED).astype(int), FIXED) for te, sc, _ in OUT[nm]], y, list(SEEDS), 5), "고정 1.0")
for nm in CONFIGS:
    S = SC[nm]; cand = np.unique(np.round(S[S < 50], 4))
    f2m = np.array([_collect([(te, (sc >= t).astype(int), t) for te, sc, _ in OUT[nm]], y, list(SEEDS), 5)["f2"].mean() for t in cand])
    log(f"  [{nm}] 고정 최적값(사후) {cand[f2m.argmax()]:.3f} → F2 {f2m.max():.3f}")
log("경계 파일 바깥 점수 (중앙 · 최소–최대): " + " | ".join(
    f"{short(k)} " + " / ".join(f"{np.median(SC[n][:, key.index(k)]):+.2f}" for n in CONFIGS) for k in FOCUS))

# %% 그림 1 (핵심): 9.5k 단일 사건 규칙의 축
fig, axs = plt.subplots(1, 2, figsize=(17, 6.5))
a = axs[0]
for lab, c, nm in ((0, NOR, "정상"), (1, ABN, "고장")):
    m = y == lab; a.scatter(S95[m], C95[m], s=24, color=c, alpha=.7, label=nm)
for i in np.flatnonzero((S95 >= -1.6) | np.isin(key, FOCUS)):
    a.annotate(short(names[i]), (S95[i], C95[i]), xytext=(4, 3), textcoords="offset points", fontsize=7.5, color=ABN if y[i] else NOR)
a.axvline(S95_MIN, color="k", lw=1); a.axhline(C95_MIN, color="k", lw=1)
a.axvspan(-1.2, -0.8, color="#999", alpha=.15, lw=0); a.axhspan(0.56, 0.84, color="#999", alpha=.15, lw=0, label="문턱 ±20% 범위")
a.set_xlabel("s95 = log10(9.25–10 kHz 에너지 / 200 Hz 이상 에너지)"); a.set_ylabel("c95 = 9.25–10 kHz 에너지 중 최대 순간 ±25 ms 몫")
a.set_title("① 오른쪽 위 = 9.5 kHz 에너지가 많고 한 사건에 몰림 → 정상 쪽으로 내림", loc="left", fontsize=10); a.legend(fontsize=8, loc="lower left")
a = axs[1]
t = np.arange(B95.shape[1]) / FS
for k, (f, c) in enumerate((("nor_116", NOR), ("nor_055", NOR), ("abn_020", ABN), ("abn_013", ABN), ("abn_007", ABN), ("abn_026", ABN))):
    i = key.index(f); env = np.sqrt(np.convolve(B95[i], np.ones(250) / 250, "same")); env = env / env.max()
    a.plot(t, env + 1.2 * k, color=c, lw=.7); a.text(2.01, 1.2 * k + .2, f"{short(f)}  c95 {C95[i]:.2f} · s95 {S95[i]:.2f}", fontsize=8, color=c)
a.set_yticks([]); a.set_xlim(0, 2.45); a.set_xlabel("시간 (s)")
a.set_title("② 9.25–10 kHz 포락선(10 ms 평균, 파일마다 최대 = 1): 정상은 한 번, 고장은 여러 번", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "rule95_axes", "9.5 kHz 몫이 높은 파일 중 정상(한 사건)과 고장(여러 충격)이 c95로 갈리는가, 문턱 ±20% 여유", key=True)


# %% 그림 2 (핵심): logistic_plane — 기준 c | 채택 후보
def plane(ax, nring, one, title):
    Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae_mean, ETX(nring)), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        o = KnockRule(2.0, 0.7).fit(tr)(te) | Rule83(0.75, 150)(te) | Res83()(te)
        rule[te] = o
    cb = body(alli) & ~rule; c1 = (one95(alli) & ~rule) if one else np.zeros(len(y), bool)
    Zc = np.clip(Z, -3, 4.5)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5); ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    show = ~(rule & (y == 1))
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
    ax.scatter(Zc[cb, 0], Zc[cb, 1], s=70, marker="o", facecolor="none", edgecolor="#2ca02c", lw=1.0, label="본체 모드 내림 대상")
    if c1.any(): ax.scatter(Zc[c1, 0], Zc[c1, 1], s=150, marker="o", facecolor="none", edgecolor="#7a3db8", lw=1.8, label="9.5k 단일 사건 내림 대상")
    c = ex.res["configs"]["고정 1.0: " + title]
    wrong = [names.index(n) for n, k in {**c["miss_counts"], **c["fp_counts"]}.items() if k >= 1]
    ax.scatter(Zc[wrong, 0], Zc[wrong, 1], s=80, marker="x", color="k", lw=1.6, label="틀림 (고정 1.0, 1/5+)")
    for i in set(wrong) | {key.index(k) for k in FOCUS if not (rule[key.index(k)] and y[key.index(k)])}:
        k = c["miss_counts"].get(names[i]) or c["fp_counts"].get(names[i]) or 0
        ax.annotate(f"{short(names[i])}" + (f" {k}/5" if k else ""), Zc[i], xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=ABN if y[i] else NOR, weight="bold" if k else "normal")
    gone = [short(names[i]) for i in np.flatnonzero(rule & (y == 1))]
    ax.text(.98, .02, f"규칙이 잡아 뺀 고장 {len(gone)}개:\n" + " ".join(gone[:7]) + ("\n" + " ".join(gone[7:]) if len(gone) > 7 else ""),
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (ext150 + 울림 2열" + (" + n_ring)" if nring else ")"))
    ax.set_title(f"{title}\n고정 1.0: F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f}", fontsize=9.5)


fig, axs = plt.subplots(1, 2, figsize=(17, 8.6))
plane(axs[0], False, False, "c | 기준 (EXP-034 c, 지금 최종)")
plane(axs[1], False, True, MAIN)
axs[0].legend(fontsize=8, loc="upper left"); axs[1].legend(fontsize=8, loc="upper left")
fig.suptitle("logistic_plane — 기준 | + 9.5k 단일 사건 내림 (OOF seed 0, 내림은 평면 위치를 바꾸지 않고 판정만 내림)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97))
ex.save_fig(fig, "logistic_plane", "9.5k 내림 대상(보라 원)이 오경보 055 · 116과 겹치는가, 경계 고장 005 · 025는 어디에 있는가", key=True)

# %% 그림 3: 경계 파일 점수 (구성별)
labs = list(CONFIGS)
fig, ax = plt.subplots(figsize=(13, 7))
for j, nm in enumerate(labs):
    for k, f in enumerate(FOCUS):
        i = key.index(f); v = SC[nm][:, i]
        if np.median(v) > 50: continue
        yy0 = k + (j - 1.5) * 0.18
        ax.plot([v.min(), v.max()], [yy0, yy0], color=f"C{j}", lw=2, alpha=.6); ax.scatter(np.median(v), yy0, color=f"C{j}", s=18, label=nm if k == 0 else None)
ax.set_yticks(range(len(FOCUS)), [("고장 " if f[0] == "a" else "정상 ") + f for f in FOCUS], fontsize=8)
ax.axvline(FIXED, color="#1a9e5a", lw=1.5, label="고정 임계값 1.0"); ax.set_xlim(-6, 6); ax.grid(alpha=.3, axis="x")
ax.set_xlabel("바깥 시험 점수 (점 = 5회 중앙, 선 = 최소–최대, 규칙으로 켜진 파일은 뺌)"); ax.legend(fontsize=8, loc="lower right")
ax.set_title("경계 파일의 점수가 구성마다 어떻게 움직이나 — 정상은 왼쪽, 고장은 오른쪽이 좋음", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "edge_files", "구성별 경계 파일 바깥 점수")

# %% 그림 4: t-SNE · 그림 5: psd_missed (맨 끝)
Es = (E - E.mean(0)) / (E.std(0) + 1e-9)
R3 = np.column_stack([R24, RBODY, NRING, S95, C95]); R3s = (R3 - R3.mean(0)) / (R3.std(0) + 1e-9)
fig, _ = viz.tsne({"ext150 + 울림 2열 + n_ring · s95 · c95 (×4 가중)": np.hstack([Es, R3s * 4])}, y, names, mark=list(FOCUS))
ex.save_fig(fig, "tsne", "새 값을 더한 입력 공간에서 경계 파일은 어디에 있는가")
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, FOCUS), "psd_missed", "경계 파일의 PSD가 정상 범위 밖으로 나가는 주파수")
log(f"done {time.time() - t0:.0f}s")
