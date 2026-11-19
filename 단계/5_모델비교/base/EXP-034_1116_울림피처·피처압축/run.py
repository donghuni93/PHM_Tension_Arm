# %% [markdown]
# EXP-034 (3차, 11-16) 충격 울림으로 만든 규칙: 8.3 kHz 깨끗한 공진(올림) · 본체 모드 울림(내림) + ET 울림 2열
# - 1차(figs/1차/): 울림 3열 0.952 · 2차(figs/2차/): 2.25–4 kHz 1열 + 8.3k 울림 비율 규칙 → 002(충격 1개)를 울림으로 못 봄
# - 3차 = 2-16 '규칙 수치 도출'(2-16_규칙수치.py)의 두 규칙을 확인
#   올림 [공진] 8.0–8.7 kHz로 울린 충격 중 감쇠비 ζ ≤ 0.02 이고 감쇠 사인파 맞춤 fit ≥ 0.7 인 충격이 하나라도 있으면 고장
#        (학습 150개: 고장 002 003 006 010 029 · 정상 0, fit −20%에서도 정상 0)
#   내림 [본체] 충격 3개 이상이고 그중 5.5–7.75 kHz로 울린 비율 ≥ 0.9 이면 점수를 학습 정상 OOF 중앙값 이하로 (규칙으로 켜진 파일 제외)
#        (학습 150개: 정상 38 · 고장 0, 문턱 −20% · 1.5초 창에서도 고장 0)
# - ET 입력 울림 2열: ring_225_4k(2.125–4.125 kHz 비율) · ring_body(5.5–7.75 kHz 비율), 충격 0개 = −1
# - 구성: 기준(EXP-025) · a ET+울림 2열 · b a + 8.3k=스펙트럼 OR 공진 · c b + 본체 내림 · d a + 8.3k=공진만(스펙트럼 없이)
# - 가설: H1 · H2′. 대역·문턱은 라벨을 본 뒤 정함 = 사후
# 실행: conda run -n qe-phm python "단계/5_모델비교/base/EXP-034_1116_울림피처·피처압축/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.experts import RuleOverride, Rule83, KnockRule, ScoreStack, ae_mean, ae_segerr, warm_cache, peak83_table
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-034", "충격 울림 규칙 — 8.3 kHz 깨끗한 공진(올림) · 본체 모드 울림(내림)", ["H1", "H2"],
                member="이동훈", model="AE + ET(ext150 + 울림 2열) → 로지스틱 + 유격 규칙 + 8.3k(스펙트럼 OR 공진) + 본체 모드 내림",
                changes="2-16 규칙 수치: 8.0–8.7 kHz 깨끗한 공진 충격(ζ≤0.02, fit≥0.7) ≥ 1개 → 고장 / 충격 ≥3 & 5.5–7.75 kHz 울림 ≥ 0.9 → 정상 쪽. ET에 울림 2열",
                focus=("abn_022", "abn_005", "nor_025", "nor_055", "nor_059", "nor_116", "abn_002", "abn_029", "abn_030"))
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
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values   # 공진 충격의 최대 fit
P83, W83, _ = peak83_table()
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


def who(m):
    return " ".join(short(names[i]) for i in np.flatnonzero(m)) or "없음"


# %% 규칙
class Res83:
    """8.0–8.7 kHz 깨끗한 공진 충격(ζ ≤ 0.02, fit ≥ t) ≥ 1개 → 고장 (te,) bool. 학습 없음"""
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
    """RuleOverride + 본체 모드 내림: body()인 파일은 점수를 학습 정상 OOF 결합 점수 중앙값 이하로. 규칙으로 켜진 파일(+100)은 그대로"""
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


class ETX:
    """ET 점수 = logit(나무 평균 확률). inp: 'ext' | 'ext+r2'"""
    def __init__(self, inp="ext"): self.inp = inp
    def __repr__(self): return f"ET[{self.inp}]"
    def __call__(self, tr, te):
        tr, te = np.array(tr), np.array(te)
        M = np.hstack([E, R24[:, None], RBODY[:, None]]) if self.inp == "ext+r2" else E
        m = make_et(seed=0)().fit(M[tr], y[tr])
        p = np.clip(m.predict_proba(M[te])[:, 1], 1 / 600, 1 - 1 / 600)
        return np.log(p / (1 - p))[:, None]


S83, RS, KN = Rule83(0.75, 150), Res83(0.7), KnockRule(2.0, 0.7)
alli = np.arange(len(y))
for nm, r in (("스펙트럼", S83), ("공진", RS), ("스펙트럼 OR 공진", Either(S83, RS))):
    on = r(alli); log(f"[8.3k {nm}] 고장 {int((on & ~nor).sum())} ({who(on & ~nor)}) · 정상 {who(on & nor)}")
for t in (0.56, 0.63, 0.77, 0.84):
    on = Res83(t)(alli); log(f"  공진 fit 문턱 {t}: 고장 {int((on & ~nor).sum())} · 정상 {who(on & nor)}")
b = body(alli); log(f"[본체 내림] 해당: 정상 {int((b & nor).sum())} · 고장 {who(b & ~nor)} · 오경보 후보 중 해당 {who(b & np.isin(key, ['nor_025', 'nor_055', 'nor_059', 'nor_116']))}")
for th in (0.72, 0.81):
    bb = body(alli, th); log(f"  본체 문턱 {th}: 고장 {who(bb & ~nor)}")


def mk(inp, r83, cap=False):
    kw = dict(experts=(ae_mean, ETX(inp)), combine="lr"); rules = [KN] + ([r83] if r83 is not None else [])
    return (lambda: BodyCap(kw, rules)) if cap else (lambda: RuleOverride(base_kw=kw, rule=rules))


CONFIGS = {
    "기준 | ext150 · 스펙트럼 (EXP-025)": mk("ext", S83),
    "a | +울림 2열 · 스펙트럼": mk("ext+r2", S83),
    "b | +울림 2열 · 스펙트럼 OR 공진": mk("ext+r2", Either(S83, RS)),
    "c | b + 본체 모드 내림": mk("ext+r2", Either(S83, RS), cap=True),
    "d | +울림 2열 · 공진만": mk("ext+r2", RS),
}
MAIN = "c | b + 본체 모드 내림"
warm_cache([ae_segerr], seeds=range(5), nested=True)
log("AE 캐시 준비 끝")

# %% 중첩 CV
res = {}
for nm, f in CONFIGS.items():
    res[nm] = nested_cv(F, y, g, f, n_jobs=-1)
    ex.record("중첩: " + nm, res[nm], names, main=(nm == MAIN))
    c = ex.res["configs"]["중첩: " + nm]
    log(f"{nm:34s} F2 {res[nm]['f2'].mean():.3f} ± {res[nm]['f2'].std():.3f} · 재현율 {res[nm]['recall'].mean():.3f} · "
        f"오경보율 {res[nm]['fpr'].mean():.3f} · 미탐 {dict(sorted(c['miss_counts'].items()))} · 오경보3+ "
        f"{ {k: v for k, v in sorted(c['fp_counts'].items()) if v >= 3} }")


# %% 그림 1 (핵심): logistic_plane — 기준 | b / d | c
def plane(ax, inp, rules, cap, title):
    Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae_mean, ETX(inp)), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        on = KnockRule(2.0, 0.7).fit(tr)(te)
        for r in rules: on |= r(te)
        rule[te] = on
    capped = body(alli) & ~rule if cap else np.zeros(len(y), bool)
    Zc = np.clip(Z, -3, 4.5)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5); ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    show = ~(rule & (y == 1))
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
    if capped.any():
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
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (" + ("ext150" if inp == "ext" else "ext150 + 울림 2열") + ")")
    ax.set_title(f"{title}\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f}", fontsize=9.5)


fig, axs = plt.subplots(2, 2, figsize=(16, 16))
plane(axs[0, 0], "ext", [S83], False, "기준 | ext150 · 스펙트럼 (EXP-025)")
plane(axs[0, 1], "ext+r2", [S83, RS], False, "b | +울림 2열 · 스펙트럼 OR 공진")
plane(axs[1, 0], "ext+r2", [RS], False, "d | +울림 2열 · 공진만")
plane(axs[1, 1], "ext+r2", [S83, RS], True, "c | b + 본체 모드 내림")
axs[0, 0].legend(fontsize=8, loc="upper left"); axs[1, 1].legend(fontsize=8, loc="upper left")
fig.suptitle("logistic_plane — 위: 기준 | 스펙트럼 OR 공진 · 아래: 공진만(스펙트럼 규칙 없이) | + 본체 모드 내림 (OOF seed 0, 내림은 평면 위치를 바꾸지 않고 판정만 내림)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.98))
ex.save_fig(fig, "logistic_plane", "새 규칙으로 002 등이 평면에서 빠지는가, 본체 모드 내림 대상(초록 원)에 오경보 025가 들어가는가", key=True)

# %% 그림 2 (핵심): 두 규칙의 축과 여유
fig, axs = plt.subplots(1, 2, figsize=(17, 6.5))
a = axs[0]
B8 = IM[(IM.f >= 8000) & (IM.f < 8700)]
for lab, c, nm in ((0, NOR, "정상"), (1, ABN, "고장")):
    s = B8[B8.y == lab]
    a.scatter(s.fit, s.zeta, s=np.clip(s.height * 3, 15, 200), color=c, alpha=.75, edgecolor="k", lw=.5, label=f"{nm} 충격 {len(s)}개 (점 크기 = 높이)")
for _, r in B8.iterrows():
    if r.y == 0 or r.file in ("abn_002", "abn_030", "abn_013"):
        a.annotate(short(r.file), (r.fit, r.zeta), xytext=(5, 5), textcoords="offset points", fontsize=8)
a.axvline(0.7, color="k", lw=1); a.axvspan(0.56, 0.84, color="#999", alpha=.2, lw=0, label="fit 문턱 0.7 ±20%"); a.axhline(0.02, color="k", lw=.8, ls="--")
a.set_xlabel("감쇠 사인파 맞춤 fit"); a.set_ylabel("감쇠비 ζ (사전 격자 0.01 · 0.02 · 0.04 · 0.08)")
a.set_title("① 올림 [공진]: 8.0–8.7 kHz로 울린 충격 — 오른쪽 아래(깨끗하고 오래 울림) = 고장", loc="left", fontsize=10); a.legend(fontsize=8)
a = axs[1]
rng = np.random.default_rng(0)
m3 = cnt >= 3
for lab, c, nm in ((0, NOR, "정상"), (1, ABN, "고장")):
    m = (y == lab) & m3
    a.scatter(cnt[m] * np.exp(rng.uniform(-.06, .06, m.sum())), RBODY[m], s=24, color=c, alpha=.7, label=f"{nm} (충격 3개 이상)")
for f in ("nor_025", "nor_055", "nor_116", "abn_028", "abn_009", "abn_005", "abn_022"):
    i = key.index(f); a.annotate(short(f), (cnt[i], RBODY[i]), xytext=(5, -3), textcoords="offset points", fontsize=8)
a.axhline(0.9, color="k", lw=1); a.axhspan(0.72, 1.0, color="#999", alpha=.15, lw=0, label="문턱 0.9 → −20% (0.72)")
a.set_xscale("log"); a.set_xlabel("충격 수 (log)"); a.set_ylabel("5.5–7.75 kHz(본체 모드)로 울린 비율")
a.set_title("② 내림 [본체]: 선 위 = 충격이 모두 본체 모드로 울림 → 정상 쪽", loc="left", fontsize=10); a.legend(fontsize=8, loc="lower left")
fig.tight_layout()
ex.save_fig(fig, "ring_rules", "두 울림 규칙의 축 · 문턱 · ±20% 여유. 정상(파랑)과 고장(주황)이 문턱 근처에 있는가", key=True)

# %% 그림 3: 점수
labs = list(CONFIGS)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for a, (k2, nm) in zip(axs, (("f2", "중첩 F2"), ("recall", "재현율"), ("fpr", "오경보율"))):
    v = [res[k][k2].mean() for k in labs]; s = [res[k][k2].std() for k in labs]
    a.bar(range(len(labs)), v, yerr=s, color=["#999"] + ["#555"] * (len(labs) - 1), capsize=3)
    for xv, vv in enumerate(v): a.text(xv, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=8)
    a.set_xticks(range(len(labs)), [k.split(" | ")[0] for k in labs]); a.set_title(nm, loc="left"); a.grid(alpha=.3, axis="y")
    if k2 in ("f2", "recall"): a.set_ylim(0.8, 1.0)
fig.suptitle(" · ".join(labs), fontsize=8, x=0.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, 0.94))
ex.save_fig(fig, "scores", "구성별 중첩 F2 · 재현율 · 오경보율 (막대 = 평균, 선 = sd)")

# %% 그림 4: hard_files
HARD = ["abn_022", "abn_005", "abn_002", "abn_029", "abn_006", "abn_010", "abn_030", "abn_003", "abn_013", "abn_014", "abn_017", "abn_015", "abn_025", "abn_028", "abn_009",
        "nor_025", "nor_055", "nor_059", "nor_116", "nor_019", "nor_092", "nor_054", "nor_106"]
M = np.array([[ex.res["configs"]["중첩: " + c]["miss_counts" if h.startswith("abn") else "fp_counts"].get(h, 0) for c in labs] for h in HARD])
fig, ax = plt.subplots(figsize=(12, 9.5))
im = ax.imshow(M, cmap="Reds", vmin=0, vmax=5, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="w" if M[i, j] >= 3 else "k")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h + f"  (공진 fit {RES[key.index(h)]:.2f} · 본체 {RBODY[key.index(h)]:.2f} · 충격 {int(cnt[key.index(h)])})" for h in HARD], fontsize=8)
ax.set_xticks(range(len(labs)), [c.replace(" | ", "\n") for c in labs], fontsize=8)
ax.set_title("파일별 틀린 횟수 (중첩 CV seed 5번 중). 고장 = 미탐, 정상 = 오경보", loc="left")
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "어느 파일이 구성마다 맞고 틀리는가")

# %% 그림 5: t-SNE
Es = (E - E.mean(0)) / (E.std(0) + 1e-9)
R2 = np.column_stack([R24, RBODY]); R2s = (R2 - R2.mean(0)) / (R2.std(0) + 1e-9)
fig, _ = viz.tsne({"ext150": E, "ext150 + 울림 2열(×4 가중)": np.hstack([Es, R2s * 4])}, y, names, mark=list(ex.focus))
ex.save_fig(fig, "tsne", "입력 공간마다 고장이 정상과 따로 모이는가, 문제 파일은 어디에 있는가")

# %% 그림 6 (맨 끝)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_missed", "문제 파일의 PSD가 정상 범위 밖으로 나가는 주파수")
log("done")
