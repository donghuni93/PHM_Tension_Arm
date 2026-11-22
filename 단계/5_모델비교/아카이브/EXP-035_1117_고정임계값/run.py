# %% [markdown]
# EXP-035 (11-17) 고정 임계값 — 지금 검증 방식(중첩 CV seed 5 × 바깥 5 × 안쪽 4) 그대로, 제출용 임계값만 '고정 최적값'으로
# - 출발: 나영님 검증(참고자료/팀원 공유 정보/1116_임계값비교). 나영님은 fold마다 고른 임계값이 흔들려서 임계값을 고정했다
# - 사용자 결정(11-17): 검증 방식은 바꾸지 않는다. 고정 임계값(나영님 방식의 E)의 값만 0이 아니라 최적값으로 고른다
# - 방법: EXP-034 c를 지금과 같은 중첩 CV로 돌려 바깥 시험 점수를 모은다 → 임계값 t를 고정했을 때 seed 5개 평균 F2가 최대인 t
#   (같으면 가장 낮은 t = 재현율 우선, best_f2_threshold와 같은 규칙). 비교: 지금 방식(fold마다 안쪽 OOF로 고름) · 고정 0 · 최종 모델 1.373
# - 주의: 고정값을 학습 150개의 바깥 점수를 보고 고르므로 그 F2는 사후(낙관적)다. 고른 값 주변이 평평한지(±0.25)를 함께 본다
# 실행: conda run -n qe-phm python "단계/5_모델비교/아카이브/EXP-035_1117_고정임계값/run.py"

# %% 설정
import os, sys, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import _oof_scores, _scores, best_f2_threshold, _collect
from src.experts import RuleOverride, Rule83, KnockRule, ae_mean
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
from joblib import Parallel, delayed, parallel_config
import matplotlib.pyplot as plt
import pandas as pd

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-035", "고정 임계값 — 지금 검증 방식 그대로, 제출 임계값만 고정 최적값 (EXP-034 c)", ["-"],
                member="이동훈", model="EXP-034 c 그대로 (AE + ET(ext150 + 울림 2열) → 로지스틱 + 유격 + 8.3k(스펙트럼 OR 공진) + 본체 내림)",
                changes="제출용 임계값: fold마다 고르기 → 하나로 고정(중첩 CV 바깥 점수에서 F2 최대인 값). 나영님 '고정 임계값' 방식",
                focus=("abn_022", "abn_005", "abn_025", "abn_020", "nor_116", "nor_059", "nor_055", "nor_054", "nor_106"))
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
SEEDS, FINAL_THR = range(5), 1.373
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


# %% EXP-034 c 정의 (EXP-034 run.py와 같음)
E = cached_table("ext", feature_table_ext, X).values
IM = pd.read_csv(os.path.join(_d, "단계", "2_EDA", "2-7_충격울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
key = [os.path.splitext(os.path.basename(n))[0] for n in names]
cnt = IM.groupby("file").size().reindex(key).fillna(0).values
frac = lambda lo, hi: np.nan_to_num(IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean().reindex(key).values.astype(float), nan=-1.0)
R24, RBODY = frac(2125, 4125), frac(5500, 7750)
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values
M_ET = np.hstack([E, R24[:, None], RBODY[:, None]])


def ETX(tr, te):
    tr, te = np.array(tr), np.array(te)
    p = np.clip(make_et(seed=0)().fit(M_ET[tr], y[tr]).predict_proba(M_ET[te])[:, 1], 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p))[:, None]


class Res83:
    def __call__(self, te): return RES[list(te)] >= 0.7


class Either:
    def __init__(self, a, b): self.a, self.b = a, b
    def __call__(self, te): return self.a(te) | self.b(te)


class BodyCap(BaseEstimator):
    def __init__(self, base_kw=None, rule=None): self.base_kw, self.rule = base_kw, rule
    def fit(self, F, yy):
        yy = np.asarray(yy); self.m_ = RuleOverride(self.base_kw, self.rule).fit(F, yy); st = self.m_.m_
        self.cap_ = float(np.median(st.m_.decision_function(st.oof_z_)[yy == 0])); return self
    def decision_function(self, F):
        s = self.m_.decision_function(F); te = np.asarray(F)[:, 0].astype(int)
        return np.where((RBODY[te] >= 0.9) & (cnt[te] >= 3) & (s < 50), np.minimum(s, self.cap_), s)


make = lambda: BodyCap(dict(experts=(ae_mean, ETX), combine="lr"), [KnockRule(2.0, 0.7), Either(Rule83(0.75, 150), Res83())])


# %% 지금과 같은 중첩 CV (nested_cv의 _outer_fold와 같은 분할) + 바깥 시험 점수 저장
def job(s, tr, te):
    oof = _oof_scores(F[tr], y[tr], g[tr], make, StratifiedGroupKFold(4, shuffle=True, random_state=s))
    return te, _scores(make().fit(F[tr], y[tr]), F[te]), best_f2_threshold(y[tr], oof)


t0 = time.time()
OUTER = [(s, tr, te) for s in SEEDS for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(g, y, g)]
with parallel_config(backend="loky", inner_max_num_threads=1):
    out = Parallel(n_jobs=-1)(delayed(job)(*o) for o in OUTER)
log(f"계산 {time.time() - t0:.0f}s")
S = np.full((len(SEEDS), len(y)), np.nan)
for (s, _, _), (te, sc, _) in zip(OUTER, out):
    S[s, te] = sc
np.save(os.path.join(ex.figs, "outer_scores.npy"), S)


def fixed(t):
    return _collect([(te, (sc >= t).astype(int), t) for te, sc, _ in out], y, list(SEEDS), 5)


# %% 고정 임계값 최적값: 후보 = 바깥 점수 값들 (best_f2_threshold와 같은 방식). 평균 F2 최대, 같으면 가장 낮은 값
lo_s = S[S < 50]
cand = np.unique(np.round(lo_s, 4))
f2m = np.array([fixed(t)["f2"].mean() for t in cand])
best = cand[int(np.argmax(f2m))]
# 같은 평균 F2를 내는 구간 (최적값이 평평한 언덕 어디쯤인가)
top = cand[np.isclose(f2m, f2m.max())]
r_sel = _collect([(te, (sc >= t).astype(int), t) for te, sc, t in out], y, list(SEEDS), 5)
CFG = {"지금 방식 (fold마다 안쪽 OOF로 고름)": r_sel, f"고정 최적값 {best:.3f}": fixed(best), "고정 0 (확률 0.5)": fixed(0.0),
       f"고정 {FINAL_THR} (지금 최종 모델)": fixed(FINAL_THR)}
for nm, r in CFG.items():
    ex.record(nm, r, names, main=nm.startswith("고정 최적값"))
    c = ex.res["configs"][nm]
    log(f"{nm:34s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} | 놓침 {dict(sorted(c['miss_counts'].items()))} | 오경보 {dict(sorted(c['fp_counts'].items()))}")
ths = np.array([t for _, _, t in out])
log(f"지금 방식이 고른 임계값 25개: 중앙 {np.median(ths):.3f} · 범위 {ths.min():.3f}–{ths.max():.3f}")
log(f"고정 최적값 {best:.3f} (평균 F2 {f2m.max():.3f}). 같은 F2를 내는 구간 {top.min():.3f}–{top.max():.3f}")
for d in (-0.25, -0.1, 0.1, 0.25):
    log(f"  최적값 {d:+.2f} ({best + d:.3f}): F2 {fixed(best + d)['f2'].mean():.3f}")

# 최적값 양옆에 있는 파일 (바깥 점수 5회 중앙값)
med = np.median(S, 0)
near = np.argsort(np.abs(med - best))[:12]
log("최적값 근처 파일 (5회 점수 중앙 · 최소–최대):\n" + "\n".join(
    f"  {short(names[i])} {'고장' if y[i] else '정상'} {med[i]:+.2f} ({S[:, i].min():+.2f} ~ {S[:, i].max():+.2f})" for i in sorted(near, key=lambda i: med[i])))

# %% 그림 1 (핵심): 고정 임계값 훑기
GRID = np.round(np.arange(-1.0, 2.51, 0.02), 3)
SW = np.array([fixed(t)["f2"] for t in GRID])
FP = np.array([[((S[s] >= t) & (y == 0)).sum() for s in SEEDS] for t in GRID]); MS = np.array([[((S[s] < t) & (y == 1)).sum() for s in SEEDS] for t in GRID])
fig, axs = plt.subplots(1, 2, figsize=(16, 5.6))
a = axs[0]; m, sd = SW.mean(1), SW.std(1)
a.plot(GRID, m, color="k", lw=1.8, label="고정 임계값의 중첩 F2 (seed 5개 평균)"); a.fill_between(GRID, m - sd, m + sd, color="#999", alpha=.25, lw=0, label="± sd")
a.axhline(r_sel["f2"].mean(), color="#7a3db8", ls="--", lw=1, label=f"지금 방식 {r_sel['f2'].mean():.3f}")
a.axvline(best, color="#1a9e5a", lw=1.6, label=f"고정 최적값 {best:.3f} → {f2m.max():.3f}")
a.axvline(FINAL_THR, color="#d62728", lw=1, ls=":", label=f"지금 최종 모델 {FINAL_THR}"); a.axvline(0, color=NOR, lw=1, ls=":", label="고정 0 (확률 0.5)")
a.scatter(ths, np.full(len(ths), a.get_ylim()[0] + .005), marker="|", s=80, color="#7a3db8", label="지금 방식이 fold마다 고른 값 25개")
a.set_xlabel("임계값 (로지스틱 로짓)"); a.set_ylabel("중첩 F2"); a.legend(fontsize=8, loc="lower left"); a.grid(alpha=.3)
a.set_title("(a) 임계값을 하나로 고정하면 — 이 곡선의 꼭대기를 고른 것이라 사후 값", loc="left", fontsize=10)
a = axs[1]
a.plot(GRID, FP.mean(1), color=NOR, label="오경보 수 (정상 120개 중)"); a.plot(GRID, MS.mean(1), color=ABN, label="놓친 고장 수 (30개 중)")
a.axvline(best, color="#1a9e5a", lw=1.6); a.axvline(FINAL_THR, color="#d62728", lw=1, ls=":"); a.axvline(0, color=NOR, lw=1, ls=":")
a.set_xlabel("임계값"); a.legend(fontsize=8); a.grid(alpha=.3); a.set_title("(b) 오경보와 놓침 (seed 5개 평균)", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "threshold_sweep", "임계값을 하나로 고정할 때 F2 · 오경보 · 놓침, 최적값과 지금 최종 모델 값의 위치", key=True)

# %% 그림 2 (핵심): 경계 근처 파일
idx = [i for i in np.argsort(med) if -1.5 < med[i] < 3.5]
fig, ax = plt.subplots(figsize=(12, max(4, .3 * len(idx))))
for k, i in enumerate(idx):
    c = ABN if y[i] else NOR
    ax.plot([S[:, i].min(), min(S[:, i].max(), 4)], [k, k], color=c, lw=2, alpha=.5); ax.scatter(med[i], k, color=c, s=30, zorder=3)
ax.set_yticks(range(len(idx)), [("고장 " if y[i] else "정상 ") + names[i] for i in idx], fontsize=8)
ax.axvline(best, color="#1a9e5a", lw=1.6, label=f"고정 최적값 {best:.3f}"); ax.axvline(FINAL_THR, color="#d62728", lw=1, ls=":", label=f"지금 최종 모델 {FINAL_THR}")
ax.axvline(0, color=NOR, lw=1, ls=":", label="고정 0")
ax.set_xlim(-1.6, 4); ax.set_xlabel("바깥 시험 점수 (점 = 5회 중앙값, 선 = 최소–최대)"); ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=.3, axis="x")
ax.set_title("경계 근처 파일 — 선 오른쪽 = 고장 판정 (규칙으로 켜진 파일 · 아주 낮은 파일은 뺌)", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "edge_files", "임계값 근처의 정상 · 고장과 분할에 따른 점수 흔들림", key=True)

# %% 그림 3 · 4: t-SNE · psd_missed (매 실험 기본)
Es = (E - E.mean(0)) / (E.std(0) + 1e-9)
R2 = np.column_stack([R24, RBODY]); R2s = (R2 - R2.mean(0)) / (R2.std(0) + 1e-9)
fig, _ = viz.tsne({"ext150 + 울림 2열(×4 가중)": np.hstack([Es, R2s * 4])}, y, names, mark=list(ex.focus))
ex.save_fig(fig, "tsne", "입력 공간에서 경계 근처 파일은 어디에 있는가")
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_missed", "경계 근처 파일의 PSD가 정상 범위 밖으로 나가는 주파수")
log(f"done {time.time() - t0:.0f}s")
