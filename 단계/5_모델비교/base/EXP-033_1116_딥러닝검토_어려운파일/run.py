# %% [markdown]
# EXP-033 딥러닝 검토 — 기준이 가르지 못하는 파일을 MLP · 1D-CNN · 2D-CNN이 가르는가
# - 질문 (사용자 11-16): "베이스라인 점수를 넘는가"가 아니라 "베이스라인이 분류하지 못하는 것을 구분할 수 있는가"
# - 기준 = 최종 구성 A (EXP-025/030: AE + ET → 로지스틱 + 8.3k 모드 규칙 + 유격 반복 규칙). 규칙 없는 L2(EXP-014)도 함께 본다
# - 어려운 파일 (실행 전 고정, EXP-025·030 기록에서)
#   ① A도 못 가르는 것: 고장 022 (놓침 5/5) · 정상 025 055 116 059 (오경보 3/5+)
#   ② 규칙으로만 잡히는 고장: 002 029 (8.3k 규칙) · 014 017 (유격 규칙). 규칙 문턱은 사후 → 학습 모델이 규칙 없이 가르면 근거가 단단해진다
# - 후보 (입력마다 하나, 설정은 실행 전 고정 · 튜닝 없음, src/candidates.py)
#   MLP     : AE와 같은 스펙트럼(1초 log PSD 8개 평균) → 지도 MLP. "같은 입력, 학습 방식만 다름"
#   1D-CNN  : 원신호 0.2초 창 MIL (EXP-019 cnn1d_mil 그대로). EXP-019에서 단독으로 022 0.92 · 002 0.99 · 025 정상 0.36을 본 모델
#   2D-CNN  : 2초 전체 log STFT 한 장 + 시간축 합성곱(약 1.3초 시야). EXP-019 cnn2d_mil(0.33초 구간)이 022를 못 본 원인 추정
#             (사건 사이 0.9초 관계를 구간이 자름, 딥러닝_검토 2절)을 바꾼 판. 비교용으로 EXP-019 구간판도 단독으로 다시 잰다
# - 넣는 방식: (a) 단독  (b) A에 한 열 추가 (AE + ET + 후보 → 로지스틱 + 같은 규칙 2개)
# - 평가: quick_cv와 같은 분할 (5-fold × seed 3), OOF 점수. 판정 (실행 전 고정)
#   고장 구분 = seed 3개 모두에서 그 고장 점수 > 정상 95% 선
#   정상 구분 = seed 3개 모두에서 그 정상 점수 < 고장 10% 선 (고장 90%보다 낮음. 무작위 모델이면 seed당 10% 확률)
#   새 오류 = 나머지 고장 25개 중 A에서는 구분됐는데 후보에서 구분 안 되는 수
#   유의미 = 어떤 후보·방식이든 (①에서 1개 이상) 또는 (②를 규칙 없이 2개 이상) 구분하고, 새 오류 ≤ 3
# - 가설: H1(타격 반복) · H2(공진) · H3(스펙트럼 모양) — 신경망이 사람이 만든 피처 밖의 신호 구조를 찾는가
# 실행: conda run -n qe-phm python 단계/5_모델비교/base/EXP-033_1116_딥러닝검토_어려운파일/run.py

# %% 설정
import os, sys, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, separation
from src.experts import RuleOverride, Rule83, KnockRule, ScoreStack, ae_mean, et_logit, ae_segerr, et_trees, warm_cache, LP
import src.candidates as C
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False

HARD1_A, HARD1_N = ["abn_022"], ["nor_025", "nor_055", "nor_116", "nor_059"]
HARD2_A = ["abn_002", "abn_029", "abn_014", "abn_017"]
HARD = HARD1_A + HARD2_A + HARD1_N
ex = Experiment(__file__, "EXP-033", "딥러닝 검토 — 기준이 못 가르는 파일을 MLP·1D-CNN·2D-CNN이 가르는가", ["H1", "H2", "H3"],
                member="이동훈", model="MLP(스펙트럼) · 1D-CNN(원신호 창 MIL) · 2D-CNN(2초 STFT 전체), 단독 / 최종 구성에 추가",
                changes="입력별 신경망 3종을 같은 분할로 학습해 어려운 파일 9개(022 · 규칙 의존 고장 4 · 오경보 정상 4)의 구분 여부 판정",
                focus=tuple(HARD))
X, y, names = load_train()
g = np.arange(len(y)); F = g[:, None]; nor = y == 0
SEEDS = range(3)
IDX = {n: names.index(n) for n in HARD}
EASY_A = [i for i in np.flatnonzero(y == 1) if names[i] not in HARD]
R83, KN = Rule83(0.75, 150), KnockRule(2.0, 0.7)
B0 = dict(experts=(ae_mean, et_logit), combine="lr")
warm_cache([ae_segerr, et_trees], seeds=SEEDS)
t0 = time.time()


def log(*a):
    print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)


class Direct(BaseEstimator):
    """전문가 함수 하나를 그대로 모델로: fit = 학습 파일 기억, 점수 = fn(학습, 시험). 단독 평가용 (안쪽 CV 없음)"""
    def __init__(self, fn=None): self.fn = fn
    def fit(self, F, yy): self.tr_ = tuple(np.asarray(F)[:, 0].astype(int).tolist()); return self
    def decision_function(self, F):
        return self.fn(self.tr_, tuple(np.asarray(F)[:, 0].astype(int).tolist()))[:, 0]


def judge(oof):
    """(seed, 파일) OOF → 파일별 구분 여부 (seed 모두) + 위험도(seed 평균: 고장 = 자기보다 높은 정상 비율, 정상 = 자기보다 낮은 고장 비율)"""
    ok = np.ones(len(y), bool); risk = np.zeros(len(y))
    for o in oof:
        n95, a10 = np.percentile(o[nor], 95), np.percentile(o[~nor], 10)
        ok &= np.where(nor, o < a10, o > n95)
        risk += np.where(nor, (o[~nor][None, :] < o[:, None]).mean(1), (o[nor][None, :] >= o[:, None]).mean(1)) / len(oof)
    return ok, risk


R = {}


def evaluate(name, make):
    q = quick_cv(F, y, g, make, seeds=SEEDS)
    ok, risk = judge(q["oof"])
    R[name] = dict(q=q, ok=ok, risk=risk, sep=separation(q["oof"], y, names))
    log(f"{name}: PR-AUC {q['pr_auc'].mean():.3f} ± {q['pr_auc'].std():.3f} · 구분된 어려운 파일 "
        f"{' '.join(n[4:] for n in HARD if ok[IDX[n]]) or '없음'}")
    return R[name]


# %% 1. 기준
evaluate("L2 AE+ET (규칙 없음)", lambda: ScoreStack(**B0))
evaluate("A 최종 (L2 + 규칙 2개)", lambda: RuleOverride(base_kw=B0, rule=[R83, KN]))
BASE = R["A 최종 (L2 + 규칙 2개)"]

# %% 2. 후보 단독 · 최종에 추가
CANDS = [("MLP", C.mlp_psd), ("1D-CNN", C.cnn1d_mil), ("2D-CNN", C.cnn2d_full)]
for nm, fn in CANDS:
    evaluate(f"{nm} 단독", lambda fn=fn: Direct(fn))
evaluate("2D-CNN 구간판(EXP-019) 단독", lambda: Direct(C.cnn2d_mil))
for nm, fn in CANDS:
    evaluate(f"A + {nm}", lambda fn=fn: RuleOverride(base_kw=dict(experts=(ae_mean, et_logit, fn), combine="lr"), rule=[R83, KN]))

# %% 3. 판정 표
base_ok = BASE["ok"]; l2_ok = R["L2 AE+ET (규칙 없음)"]["ok"]
rows, VERD = [], {}
hdr = f"{'구성':28s} PR-AUC         " + " ".join(f"{n[4:]:>5s}" for n in HARD) + "   새오류  확실고장 위험정상"
rows.append(hdr); rows.append("                                            " + "  ".join(["고장①"] + ["고장②"] * 4 + ["정상①"] * 4))
for k, r in R.items():
    ok, risk = r["ok"], r["risk"]
    cell = " ".join(f"{('●' if ok[IDX[n]] else ' ')}{risk[IDX[n]]:.2f}" for n in HARD)
    new_err = [names[i][4:] for i in EASY_A if base_ok[i] and not ok[i]]
    gain1 = [n[4:] for n in HARD1_A + HARD1_N if ok[IDX[n]] and not base_ok[IDX[n]]]
    # ②는 단독 구성에서만 본다 (추가 구성은 규칙이 이미 ②를 올림)
    gain2 = [n[4:] for n in HARD2_A if ok[IDX[n]] and not l2_ok[IDX[n]]] if k.endswith("단독") else []
    sig = (len(gain1) >= 1 or len(gain2) >= 2) and len(new_err) <= 3 and k not in ("L2 AE+ET (규칙 없음)", "A 최종 (L2 + 규칙 2개)")
    VERD[k] = dict(gain1=gain1, gain2=gain2, new_err=new_err, sig=bool(sig))
    rows.append(f"{k:28s} {r['q']['pr_auc'].mean():.3f} ± {r['q']['pr_auc'].std():.3f}  {cell}   {len(new_err):4d}   "
                f"{r['sep']['sure_a']:4d}    {r['sep']['risk_n']:5.1f}")
rows += ["", "● = 구분 (seed 3개 모두: 고장은 정상 95% 선 위, 정상은 고장 10% 선 아래). 숫자 = 위험도 (0 = 안전, 1 = 반대쪽 끝)",
         "새오류 = 나머지 고장 25개 중 A에서 구분됐는데 이 구성에서 안 되는 수", ""]
for k, v in VERD.items():
    if k.startswith(("L2", "A 최종")):
        continue
    rows.append(f"{k:28s} ①새로 구분 {' '.join(v['gain1']) or '—':16s} ②규칙 없이 {' '.join(v['gain2']) or '—':16s} "
                f"새오류 {' '.join(v['new_err']) or '—':20s} → {'유의미' if v['sig'] else '해당 없음'}")
SIG = [k for k, v in VERD.items() if v["sig"]]
rows += ["", f"판정: {'유의미 — ' + ', '.join(SIG) if SIG else '기각 — 기준이 못 가르는 파일을 새로 가르는 구성 없음'}"]
txt = "\n".join(rows); print(txt, flush=True)
ex.note("어려운 파일 구분 판정 (quick_cv OOF, seed 3)", txt)
ex.res["verdict"] = {k: v for k, v in VERD.items()}
ex.res["pr_auc"] = {k: [round(float(r["q"]["pr_auc"].mean()), 3), round(float(r["q"]["pr_auc"].std()), 3)] for k, r in R.items()}
ex.res["risk"] = {k: {n: round(float(r["risk"][IDX[n]]), 3) for n in HARD} for k, r in R.items()}

# %% 그림 1 (핵심): hard_files — 어려운 파일 위험도, 열 = 구성
cols = list(R)
M = np.array([[R[k]["risk"][IDX[n]] for k in cols] for n in HARD])
OK = np.array([[R[k]["ok"][IDX[n]] for k in cols] for n in HARD])
fig, ax = plt.subplots(figsize=(2.6 + 1.25 * len(cols), 6.4))
im = ax.imshow(M, cmap="Oranges", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{'●' if OK[i, j] else ''}{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                color="k" if M[i, j] < .6 else "w", fontweight="bold" if OK[i, j] else "normal")
for b in (len(HARD1_A) - .5, len(HARD1_A) + len(HARD2_A) - .5):
    ax.axhline(b, color="k", lw=.8)
ax.axvline(1.5, color="k", lw=1.2)
ax.set_yticks(range(len(HARD)))
ax.set_yticklabels([f"{n}  {'① 놓침' if n in HARD1_A else '② 규칙 의존' if n in HARD2_A else '① 오경보'}" for n in HARD], fontsize=9)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([k.replace(" (", "\n(").replace(" 단독", "\n단독") for k in cols], fontsize=8)
ax.set_title("어려운 파일 위험도 (quick_cv seed 3 평균, 0 = 안전 · 1 = 반대쪽 끝). ● = seed 3개 모두에서 구분\n"
             "고장: 정상 95% 선 위 · 정상: 고장 10% 선 아래. 세로선 왼쪽 = 기준", fontsize=9.5, loc="left")
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "기준이 못 가르는 파일 9개를 신경망 3종이 가르는가 (● = 구분)", key=True)

# %% 그림 2 (핵심): 기준 점수 × 후보 점수 평면 (OOF seed 0) — 후보 축이 기준 축과 다른 파일을 가르는가
base_o = R["L2 AE+ET (규칙 없음)"]["q"]["oof"][0]
zn = lambda o: (o - np.median(o[nor])) / (np.percentile(o[nor], 90) - np.median(o[nor]) + 1e-9)
fig, axs = plt.subplots(1, len(CANDS), figsize=(6.4 * len(CANDS), 6))
for ax, (nm, _) in zip(axs, CANDS):
    co = R[f"{nm} 단독"]["q"]["oof"][0]
    xs, ys = np.clip(zn(base_o), -3, 8), np.clip(zn(co), -3, 8)
    ax.scatter(xs[nor], ys[nor], s=16, c="#2a78d6", alpha=.55, lw=0, label="정상")
    ax.scatter(xs[~nor], ys[~nor], s=26, c="#eb6834", alpha=.8, lw=0, label="고장")
    for n in HARD:
        i = IDX[n]
        ax.scatter(xs[i], ys[i], s=110, facecolor="none", edgecolor="k", lw=1.1)
        ax.annotate(n.replace("abn_", "A").replace("nor_", "N"), (xs[i], ys[i]), fontsize=9, xytext=(5, 3), textcoords="offset points")
    for v in (np.percentile(xs[nor], 95),):
        ax.axvline(v, color="#6b6a64", ls="--", lw=.9)
    ax.axhline(np.percentile(ys[nor], 95), color="#6b6a64", ls="--", lw=.9)
    ax.set_xlabel("기준 L2 점수 (AE+ET 로지스틱, 정상 기준 z, 3 이상 자름)"); ax.set_ylabel(f"{nm} 단독 점수 (정상 기준 z)")
    ax.set_title(f"{nm}: 오른쪽 위 = 둘 다 고장 쪽. 점선 = 각 축 정상 95%\n왼쪽 위 고장 = 기준은 놓치고 {nm}만 잡음", fontsize=10, loc="left")
    ax.grid(alpha=.25); ax.legend(fontsize=8, frameon=False, loc="lower right")
fig.tight_layout()
ex.save_fig(fig, "logistic_plane", "기준 점수(가로) × 신경망 점수(세로), OOF seed 0. 동그라미 = 어려운 파일 9개", key=True)

# %% 그림 3: t-SNE — 신경망 입력 공간 (MLP 입력 스펙트럼, ext150), 2행 = 1D-CNN 단독 점수 색
E = cached_table("ext", feature_table_ext, X)
best = max(CANDS, key=lambda c: sum(R[f"{c[0]} 단독"]["ok"][IDX[n]] for n in HARD))[0]
fig, _ = viz.tsne({"MLP 입력: 평균 log PSD (1025칸)": LP.mean(1), "ext150 피처 (ET 입력)": E.values}, y, names, mark=HARD,
                  score=R[f"{best} 단독"]["q"]["oof"].mean(0))
ex.save_fig(fig, "tsne", f"입력 공간에서 어려운 파일의 위치와 {best} 단독 점수(2행 색)")

# %% 그림 4: 어려운 고장 PSD vs 정상 5–95% (매 실험 기본, 맨 끝)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, HARD1_A + HARD2_A), "psd_missed",
            "어려운 고장 5개의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
log("done")
