# %% [markdown]
# EXP-040 (2차 설계 1단계, 11-21) A 기준 재측정 · B ET 하나(H 피처 + 박자 S) · C H → ET + 박자 z → 로지스틱 결합
# - 명세: 단계/5_모델비교/검토문서/모델구성_계획_1121.md 4–6절 (실행 전 고정). 피처 = 4-7 최종 목록(단계/4_피처/4-7_1121_2차피처점검/최종피처.json)
# - A: EXP-034 b(= c에서 본체 모드 내림만 뺌, 사용자 결정 11-21) 코드를 그대로(복사) 같은 중첩 CV로 재측정. 기록 0.949 ± 0.007. c(0.955 ± 0.007)는 참고로 같이 잰다
# - B: ET(make_et 기본값, 튜닝 없음) 하나에 H 피처(+ H16–H17 폴드 안 계산) + L1 S. 규칙 없음
# - C: H 피처 → ET(안쪽 4겹 OOF logit) · L1 S → 박자 z(학습 정상 S의 강건 z) → 로지스틱(입력 2개). 박자 문턱(정상 99%)은 2단계용이라 안 씀
# - 폴드 의존(H16–H17, 박자 z, C의 로지스틱)은 모두 추정기 fit 안에서 학습 행으로. 임계값 = 안쪽 OOF F2 최대(nested_cv)
# - val(라벨 없음): B·C를 train 150으로 학습해 40개 채점만 (구성 선택에 안 씀, 제출 파일 없음)
# 실행: conda run -n qe-phm python "단계/5_모델실험/3_2차설계/EXP-040_2차설계_1단계/자료/run.py"

# %% 설정
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.experts import RuleOverride, Rule83, KnockRule, ScoreStack, ae_mean, ae_segerr, warm_cache, peak83_table
from src.data import load_folder
from src.features2 import cached_feat2, PeakRef, robust_z, H_FIXED, H_FOLD
from src.evaluate import best_f2_threshold
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN, PUR = "#2a78d6", "#eb6834", "#7B1FA2"
ex = Experiment(__file__, "EXP-040", "2차 설계 1단계 — A 기준 · B ET 하나 · C 저/고 분리 + 로지스틱", ["H1", "H2", "H3", "H4"],
                member="이동훈", model="A: EXP-034 b / B: ET(H 15열 + 박자 S) / C: ET(H) + 박자 z → 로지스틱",
                changes="2차 설계: 새 충격 정의(11-21) · 1 kHz 고역 H 블록(옥타브 4칸·울림 몫·PSD 몫·봉우리·HFL) + 박자 S. AE·규칙 없음 (B·C)")
X, y, names = load_train(); names = [str(n) for n in names]
Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val", "val")); nv = [str(n) for n in nv]
g = np.arange(len(y)); nor = y == 0; n = len(y)
short = lambda s: s.replace("abn_", "A").replace("nor_", "N")
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


# %% 피처 (4-7 최종 목록)
FJ = json.load(open(os.path.join(_d, "단계", "4_피처", "4-2_H피처목록", "자료", "최종피처.json"), encoding="utf-8"))
FIXED, PEAK = FJ["fixed"], FJ["hfold"]                      # 폴드 밖 H 피처, 폴드 안 H16–H17 중 남은 것
assert FJ["lock"] == ["L1_S"]
T, L, fr = cached_feat2("train", X); Tv, Lv, _ = cached_feat2("val", Xv)
df = float(fr[1] - fr[0]); nf = len(FIXED)
M = np.hstack([T[FIXED].values, T[["L1_S"]].values, L])      # 열 = [FIXED | S | log PSD(1 kHz 위)]
Mv = np.hstack([Tv[FIXED].values, Tv[["L1_S"]].values, Lv])
HCOLS = FIXED + PEAK
log("H 피처", len(HCOLS), HCOLS, "+ L1_S")


def lg(p):
    p = np.clip(p, 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p))


class Fold2(BaseEstimator):
    """B / C 추정기. 입력 행 = [FIXED | S | log PSD]. 학습 정상 의존 부분은 모두 fit 안에서 학습 행으로
    mode 'B': ET[H + S]  /  mode 'C': ET[H] OOF logit(안쪽 4겹) + 박자 z → 로지스틱(balanced, C=1)"""
    def __init__(self, mode="B", nf=nf, df=df, peak=tuple(PEAK), seed=0):
        self.mode, self.nf, self.df, self.peak, self.seed = mode, nf, df, peak, seed

    def _H(self, F):
        H = F[:, :self.nf]
        if self.peak:
            pk = self.pr_.transform(F[:, self.nf + 1:])
            H = np.hstack([H, pk[:, [H_FOLD.index(p) for p in self.peak]]])
        return H

    def fit(self, F, y):
        F, y = np.asarray(F, float), np.asarray(y)
        self.pr_ = PeakRef(self.df).fit(F[y == 0, self.nf + 1:])
        H, S = self._H(F), F[:, self.nf]
        if self.mode == "B":
            self.m_ = make_et(seed=self.seed)().fit(np.column_stack([H, S]), y)
        else:
            self.zf_ = robust_z(S[y == 0])
            oof = np.empty(len(y))
            for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(H, y):
                oof[b] = lg(make_et(seed=self.seed)().fit(H[a], y[a]).predict_proba(H[b])[:, 1])
            self.lr_ = LogisticRegression(C=1.0, class_weight="balanced").fit(np.column_stack([oof, self.zf_(S)]), y)
            self.m_ = make_et(seed=self.seed)().fit(H, y)
        return self

    def parts(self, F):
        """C: (ET logit, 박자 z). B: (ET logit, S 원값)"""
        F = np.asarray(F, float); H, S = self._H(F), F[:, self.nf]
        if self.mode == "B":
            return lg(self.m_.predict_proba(np.column_stack([H, S]))[:, 1]), S
        return lg(self.m_.predict_proba(H)[:, 1]), self.zf_(S)

    def decision_function(self, F):
        a, b = self.parts(F)
        if self.mode == "B":
            return a
        return self.lr_.decision_function(np.column_stack([a, b]))


# %% A: EXP-034 c 코드 (복사, 바꾸지 않음)
E = cached_table("ext", feature_table_ext, X).values
IM = pd.read_csv(os.path.join(_d, "단계", "2_EDA", "2-3_충격울림주파수", "자료", "figs", "impacts.csv"), encoding="utf-8-sig")
key = [os.path.splitext(os.path.basename(s))[0] for s in names]
cnt = IM.groupby("file").size().reindex(key).fillna(0).values


def frac(lo, hi):
    b = IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean()
    return b.reindex(key).values.astype(float)


R24 = np.nan_to_num(frac(2125, 4125), nan=-1.0); RBODY = np.nan_to_num(frac(5500, 7750), nan=-1.0)
RES = IM[(IM.f >= 8000) & (IM.f < 8700) & (IM.zeta <= 0.02)].groupby("file").fit.max().reindex(key).fillna(0).values


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
    def __init__(self, inp="ext"): self.inp = inp
    def __repr__(self): return f"ET[{self.inp}]"
    def __call__(self, tr, te):
        tr, te = np.array(tr), np.array(te)
        Mx = np.hstack([E, R24[:, None], RBODY[:, None]]) if self.inp == "ext+r2" else E
        m = make_et(seed=0)().fit(Mx[tr], y[tr])
        p = np.clip(m.predict_proba(Mx[te])[:, 1], 1 / 600, 1 - 1 / 600)
        return np.log(p / (1 - p))[:, None]


S83, RS, KN = Rule83(0.75, 150), Res83(0.7), KnockRule(2.0, 0.7)
_kwA = dict(experts=(ae_mean, ETX("ext+r2")), combine="lr")
mkA = lambda: RuleOverride(_kwA, [KN, Either(S83, RS)])                      # A = EXP-034 b (본체 모드 내림 없음, 사용자 결정 11-21)
mkAc = lambda: BodyCap(_kwA, [KN, Either(S83, RS)])                          # 참고: EXP-034 c = b + 본체 모드 내림
warm_cache([ae_segerr], seeds=range(5), nested=True)

# %% 중첩 CV (A · B · C 모두 같은 split)
FA = g[:, None]
CFG = {"A | EXP-034 b 재측정": (FA, mkA),
       "B | ET 하나 (H + 박자 S)": (M, lambda: Fold2("B")),
       "C | ET(H) + 박자 z → 로지스틱": (M, lambda: Fold2("C"))}
res = {}
for nm, (Fm, mk_) in CFG.items():
    res[nm] = nested_cv(Fm, y, g, mk_, n_jobs=-1)
    c = ex.record("중첩: " + nm, res[nm], names, main=nm.startswith("C"))
    log(f"{nm:34s} F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f} · F2(10%/40%) {c['f2_p10']:.3f}/{c['f2_p40']:.3f} · "
        f"미탐 {dict(sorted(c['miss_counts'].items()))} · 오경보 {dict(sorted(c['fp_counts'].items()))}")
res["Ac | EXP-034 c 재측정 (참고, A + 본체 모드 내림)"] = nested_cv(FA, y, g, mkAc, n_jobs=-1)
ex.record("중첩: Ac | EXP-034 c 재측정 (참고, A + 본체 모드 내림)", res["Ac | EXP-034 c 재측정 (참고, A + 본체 모드 내림)"], names)
ac = ex.res["configs"]["중첩: Ac | EXP-034 c 재측정 (참고, A + 본체 모드 내림)"]
log(f"Ac(c) F2 {ac['f2']:.3f} ± {ac['f2_sd']:.3f} 기록 0.955 ± 0.007 · 미탐 {ac['miss_counts']} · 오경보 {ac['fp_counts']}")
a, b, c_ = (ex.res["configs"]["중첩: " + k] for k in CFG)
sdm = float(np.mean([a["f2_sd"], b["f2_sd"], c_["f2_sd"]]))
log(f"A(b) 재측정 {a['f2']:.3f} ± {a['f2_sd']:.3f} vs 기록 0.949 ± 0.007 (차 {a['f2'] - 0.949:+.3f}, sd 안 = {abs(a['f2'] - 0.949) <= 0.007})")
log(f"B − A = {b['f2'] - a['f2']:+.3f} (A sd {a['f2_sd']:.3f}, B sd {b['f2_sd']:.3f}) · C − A = {c_['f2'] - a['f2']:+.3f} · C − B = {c_['f2'] - b['f2']:+.3f} (B sd {b['f2_sd']:.3f})")
ex.note("비교", dict(A_vs_record=round(a["f2"] - 0.949, 4), Ac_vs_record=round(ac["f2"] - 0.955, 4), B_minus_A=round(b["f2"] - a["f2"], 4), C_minus_A=round(c_["f2"] - a["f2"], 4), C_minus_B=round(c_["f2"] - b["f2"], 4),
                    sd=dict(A=a["f2_sd"], B=b["f2_sd"], C=c_["f2_sd"]), selection_rule="차 < sd 면 단순한 쪽 (B → C)",
                    C_ge_B=bool(c_["f2"] >= b["f2"])))
labs = list(CFG)
bad = sorted({k for r in ex.res["configs"].values() for k in list(r["miss_counts"]) + list(r["fp_counts"])
              if max(r["miss_counts"].get(k, 0), r["fp_counts"].get(k, 0)) >= 3})
ex.focus = bad[:12] if bad else list(ex.focus)

# %% 점수 평면 (OOF, 5-fold seed 0): 핵심 그림
def oof_parts(mk_, Fm):
    P = np.empty((n, 2)); sc = np.empty(n)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        m = mk_().fit(Fm[tr], y[tr]); P[te] = np.column_stack(m.parts(Fm[te])); sc[te] = m.decision_function(Fm[te])
    return P, sc


def wrong_of(cfg):
    r = ex.res["configs"]["중첩: " + cfg]
    return {names.index(k): v for k, v in {**r["miss_counts"], **r["fp_counts"]}.items() if v >= 2}


def plane_A(ax):
    """EXP-034 방식 AE z × ET z (A의 logistic_plane, 규칙으로 잡힌 고장은 뺌)"""
    Z, rule = np.empty((n, 2)), np.zeros(n, bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae_mean, ETX("ext+r2")), combine="lr").fit(FA[tr], y[tr]).zscores(FA[te])[:, :2]
        rule[te] = KnockRule(2.0, 0.7).fit(tr)(te) | S83(te) | RS(te)
    Zc = np.clip(Z, -3, 4.5)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5)
    ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    show = ~(rule & (y == 1))
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
    if (rule & nor).any(): ax.scatter(Zc[rule & nor, 0], Zc[rule & nor, 1], s=110, marker="D", facecolor="none", edgecolor="#7a3db8", lw=1.3, label="규칙이 켜진 정상")
    w = wrong_of(labs[0]); wi = list(w)
    ax.scatter(Zc[wi, 0], Zc[wi, 1], s=80, marker="x", color="k", lw=1.6, label="자주 틀림 (2/5+)")
    for i in wi: ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {w[i]}/5", Zc[i], xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR, weight="bold")
    gone = [short(names[i]) for i in np.flatnonzero(rule & (y == 1))]
    ax.text(.98, .02, f"규칙이 잡아 뺀 고장 {len(gone)}개:\n" + " ".join(gone), transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5, bbox=dict(facecolor="white", alpha=.85, edgecolor="#999"))
    ax.set_xlim(-3, 4.5); ax.set_ylim(-3, 4.5); ax.set_aspect("equal"); ax.grid(alpha=.3); ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (ext150 + 울림 2열)")
    ax.set_title(f"A | EXP-034 b 재측정 (규칙 2개 사후)\n중첩 F2 {a['f2']:.3f} ± {a['f2_sd']:.3f} · 재현율 {a['recall']:.3f} · 오경보율 {a['fpr']:.3f}", fontsize=9.5); ax.legend(fontsize=8, loc="upper left")


PC, SC = oof_parts(CFG[labs[2]][1], M)
def plane_C(ax):
    xc, yc = np.clip(PC[:, 0], -7, 7), np.clip(PC[:, 1], -3, 20)
    lr = LogisticRegression(C=1.0, class_weight="balanced").fit(np.column_stack([PC[:, 0], PC[:, 1]]), y)
    xx, yy_ = np.meshgrid(np.linspace(-7, 7, 300), np.linspace(-3, 20, 300))
    D = lr.decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8); ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5)
    ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    ax.scatter(xc[nor], yc[nor], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상"); ax.scatter(xc[~nor], yc[~nor], color=ABN, s=28, lw=0, alpha=.9, label="고장")
    w = wrong_of(labs[2]); wi = list(w)
    ax.scatter(xc[wi], yc[wi], s=80, marker="x", color="k", lw=1.6, label="자주 틀림 (2/5+)")
    for i in wi: ax.annotate(f"{short(names[i])} {'미탐' if y[i] else '오경보'} {w[i]}/5", (xc[i], yc[i]), xytext=(4, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR, weight="bold")
    for nm_ in ("abn_022", "abn_014"):
        i = names.index(nm_); ax.annotate(short(nm_), (xc[i], yc[i]), xytext=(-18, 6), textcoords="offset points", fontsize=8, color=ABN)
    ax.set_xlabel("H 점수 (ET logit, 폴드 밖 OOF)"); ax.set_ylabel("박자 z (S의 학습 정상 강건 z, 20으로 자름)"); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"C | ET(H) + 박자 z → 로지스틱 (규칙 없음, 고장 모두 표시)\n중첩 F2 {c_['f2']:.3f} ± {c_['f2_sd']:.3f} · 재현율 {c_['recall']:.3f} · 오경보율 {c_['fpr']:.3f}", fontsize=9.5)


fig, axs = plt.subplots(1, 2, figsize=(18, 8.2)); plane_A(axs[0]); plane_C(axs[1])
fig.suptitle("logistic_plane — 왼쪽 기준 A (AE z × ET z) | 오른쪽 C (H 점수 × 박자 z). OOF 5-fold seed 0, 배경 = 로지스틱 경계(0 실선, ±1 점선)", x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .96))
ex.save_fig(fig, "logistic_plane", "A는 AE z × ET z 평면에서 규칙이 고장 일부를 빼고, C는 H 점수 × 박자 z 평면에서 022·014가 위쪽에 따로 서는가", key=True)

# %% hard_files · scores
HARD = bad + [h for h in ("abn_022", "abn_014", "abn_017", "abn_002", "abn_015", "nor_116") if h not in bad]
Mh = np.array([[ex.res["configs"]["중첩: " + c]["miss_counts" if h.startswith("abn") else "fp_counts"].get(h, 0) for c in labs] for h in HARD])
fig, ax = plt.subplots(figsize=(8.5, 0.42 * len(HARD) + 2))
im = ax.imshow(Mh, cmap="Reds", vmin=0, vmax=5, aspect="auto")
for i in range(Mh.shape[0]):
    for j in range(Mh.shape[1]): ax.text(j, i, Mh[i, j], ha="center", va="center", fontsize=9, color="w" if Mh[i, j] >= 3 else "k")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h for h in HARD], fontsize=8); ax.set_xticks(range(3), [c.split(" | ")[0] for c in labs])
ax.set_title("파일별 틀린 횟수 (seed 5번 중). 고장 = 미탐, 정상 = 오경보", loc="left", fontsize=10); fig.colorbar(im, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "어느 파일이 A · B · C에서 맞고 틀리는가")
fig, axs = plt.subplots(1, 3, figsize=(15, 4.5))
for a_, (k2, nm) in zip(axs, (("f2", "중첩 F2"), ("recall", "재현율"), ("fpr", "오경보율"))):
    v = [res[k][k2].mean() for k in labs]; s = [res[k][k2].std() for k in labs]
    a_.bar(range(3), v, yerr=s, color=["#999", "#555", "#555"], capsize=3)
    for xv, vv in enumerate(v): a_.text(xv, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=8)
    a_.set_xticks(range(3), [k.split(" | ")[0] for k in labs]); a_.set_title(nm, loc="left"); a_.grid(alpha=.3, axis="y")
fig.tight_layout(); ex.save_fig(fig, "scores", "A · B · C 중첩 F2 · 재현율 · 오경보율 (막대 = 평균, 선 = sd)")

# %% val 채점 (train 150 전체로 학습, 라벨 없음) — 구성 선택에 쓰지 않는다
def thr_of(mk_, Fm, seeds=(0, 1, 2)):
    """임계값 = 학습 150을 4등분 OOF 점수에서 F2 최대 (final.py의 FinalModel 방식, seed 중앙값)"""
    t = []
    for s in seeds:
        o = np.empty(n)
        for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=s).split(g, y, g): o[te] = mk_().fit(Fm[tr], y[tr]).decision_function(Fm[te])
        t.append(best_f2_threshold(y, o))
    return float(np.median(t)), t


VAL = {}
for nm in labs[1:]:
    mk_ = CFG[nm][1]; t, ts = thr_of(mk_, M); m = mk_().fit(M, y); sv = m.decision_function(Mv); pa, pb = m.parts(Mv)
    VAL[nm] = dict(thr=t, thr_seeds=ts, score=sv, pred=(sv >= t).astype(int), part_a=pa, part_b=pb)
    log(f"val {nm}: 임계값 {t:.3f} (seed별 {np.round(ts, 3).tolist()}) → 판정 {int(VAL[nm]['pred'].sum())}/40")
import joblib
FM = joblib.load(os.path.join(_d, "outputs", "final", "final_model.joblib"))
DA = FM.predict_table(Xv, nv); predAc = DA["pred"].values.astype(int)              # final_model = EXP-034 c 설정 (본체 모드 내림 포함)
assert [str(f) for f in DA["file"]] == nv
# 내림 전 판정(= b) 복원: Core.parts의 score = lr + 100·rule83 + 100·knock (use_lock·use_cap 꺼짐) 에서 내림이 body 파일의 점수를 min(score, cap)으로 바꾼다.
# predict_table의 lr · rule83 · knock 열로 내림 전 점수를 다시 만들고 같은 임계값(FM.thr_)으로 자른다. src/final.py는 고치지 않음
pre = DA["lr"].values + 100.0 * DA["rule83"].values + 100.0 * DA["knock"].values
nb_ = ~DA["body"].values.astype(bool)
assert np.allclose(pre[nb_], DA["score"].values[nb_]), "내림 안 걸린 파일은 내림 전 점수 = 최종 점수여야 한다"
predA = (pre >= FM.thr_).astype(int)
log(f"val A (EXP-034 b = 내림 전, 임계값 {FM.thr_}) 판정 {int(predA.sum())}/40: " + " ".join(nv[i][-3:] for i in np.flatnonzero(predA)))
log(f"val Ac (EXP-034 c, 최종 모델 그대로) 판정 {int(predAc.sum())}/40: " + " ".join(nv[i][-3:] for i in np.flatnonzero(predAc)) + f" · 내림이 바꾼 파일: {' '.join(nv[i][-3:] for i in np.flatnonzero(predA != predAc)) or '없음'}")
pB, pC = VAL[labs[1]]["pred"], VAL[labs[2]]["pred"]
vt = pd.DataFrame(dict(file=nv, A=predA, Ac=predAc, B=pB, C=pC, B_score=VAL[labs[1]]["score"].round(3), C_score=VAL[labs[2]]["score"].round(3), C_ET_logit=VAL[labs[2]]["part_a"].round(2),
                       박자z=VAL[labs[2]]["part_b"].round(2), S=Tv.L1_S.values.round(3)))
vt.to_csv(os.path.join(ex.dir, "val_판정.csv"), index=False, encoding="utf-8-sig")
lst = lambda p: " ".join(nv[i][-3:] for i in np.flatnonzero(p))
ov = {f"A∩B": int((predA & pB).sum()), "A∩C": int((predA & pC).sum()), "B∩C": int((pB & pC).sum()), "A∪B∪C": int((predA | pB | pC).sum()), "A만": lst(predA & ~pB & ~pC),
      "B만": lst(pB & ~predA & ~pC), "C만": lst(pC & ~predA & ~pB), "셋 다": lst(predA & pB & pC)}
top = vt.sort_values("박자z", ascending=False).head(5)
VALJ = dict(A=dict(n=int(predA.sum()), files=lst(predA)), B=dict(n=int(pB.sum()), thr=VAL[labs[1]]["thr"], files=lst(pB)), C=dict(n=int(pC.sum()), thr=VAL[labs[2]]["thr"], files=lst(pC)),
            overlap=ov, beat_z_top=[(r.file[-3:], float(r.박자z), float(r.S)) for r in top.itertuples()],
            Ac=dict(n=int(predAc.sum()), files=lst(predAc), changed_by_cap=lst(predA != predAc)),
            Ac_expected_record="001 004 013 016 017 018 024 027 031 035 039", Ac_matches_record=lst(predAc) == "001 004 013 016 017 018 024 027 031 035 039")
ex.note("val", VALJ); log("val 겹침", ov, "박자 z 상위", VALJ["beat_z_top"], "c 기록과 같음", VALJ["Ac_matches_record"])

# %% B의 ET 피처 중요도 (train 150 전체)
mB = Fold2("B").fit(M, y)
pk_all = mB.pr_.transform(L)
cols_B = FIXED + PEAK + ["L1_S"]
imp = mB.m_.feature_importances_
o = np.argsort(imp)
fig, ax = plt.subplots(figsize=(8, 0.35 * len(cols_B) + 1.5)); ax.barh(range(len(cols_B)), imp[o], color="#555")
ax.set_yticks(range(len(cols_B)), [cols_B[i] for i in o], fontsize=8); ax.set_xlabel("ET 피처 중요도 (MDI, train 150 전체)")
ax.set_title("B의 ET 피처 중요도 (참고: MDI는 연속 값 열에 치우침)", loc="left", fontsize=10); fig.tight_layout()
ex.save_fig(fig, "importance_B", "B의 ET가 어떤 H 피처·박자 S를 쓰는가")

# %% t-SNE (입력 공간별, train + val, 이상 점수 색: train = C의 OOF 점수, val = train 전체로 학습한 C 점수)
def stdz(Atr, Ava):
    Z, Zv = np.empty(Atr.shape), np.empty(Ava.shape)
    for j in range(Atr.shape[1]):
        f = robust_z(Atr[nor, j]); Z[:, j] = f(Atr[:, j]); Zv[:, j] = f(Ava[:, j])
    return Z, Zv


from sklearn.manifold import TSNE
NEWC = json.load(open(os.path.join(_d, "단계", "4_피처", "4-2_H피처목록", "자료", "result.json"), encoding="utf-8"))["cols_new"]
pk_v = PeakRef(df).fit(L[nor]).transform(Lv)
Tn, Tnv = T.copy(), Tv.copy(); Tn["H16_peak_h"], Tn["H17_peak_w"] = pk_all[:, 0], pk_all[:, 1]; Tnv["H16_peak_h"], Tnv["H17_peak_w"] = pk_v[:, 0], pk_v[:, 1]
Ev_ = cached_table("ext_val", feature_table_ext, Xv)
views = {f"새 피처 {len(NEWC)}열 (H + L1 S)": stdz(Tn[NEWC].values, Tnv[NEWC].values),
         "ext150 (A의 ET 입력)": stdz(E, Ev_.reindex(columns=cached_table("ext", feature_table_ext, X).columns, fill_value=0.0).values)}
sc_all = np.concatenate([SC, VAL[labs[2]]["score"]])
fig, axs = plt.subplots(2, 2, figsize=(14, 12.4))
for j, (title, (Z, Zv)) in enumerate(views.items()):
    best = None
    for s in range(5):
        t = TSNE(2, perplexity=30, init="random", random_state=s, learning_rate="auto"); Em = t.fit_transform(np.vstack([Z, Zv]))
        if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, Em)
    Em = best[1]; Et, Ev = Em[:n], Em[n:]
    a_ = axs[0, j]; a_.scatter(*Et[nor].T, s=20, color=NOR, alpha=.55, lw=0, label="train 정상"); a_.scatter(*Et[~nor].T, s=36, color=ABN, lw=0, label="train 고장")
    a_.scatter(*Ev.T, s=45, color=PUR, marker="^", edgecolor="white", lw=.4, label="val")
    for h in bad + ["abn_022", "abn_014"]:
        i = names.index(h); a_.annotate(short(h), Et[i], xytext=(3, 3), textcoords="offset points", fontsize=7, color=ABN if y[i] else NOR)
    a_.set_title(f"{title} · 라벨 색 (KL 최소 {best[0]:.2f})", fontsize=10); a_.legend(fontsize=8)
    a_ = axs[1, j]; sc = a_.scatter(*Em.T, c=sc_all, cmap="viridis", s=22, lw=0); fig.colorbar(sc, ax=a_, shrink=.8, label="C의 이상 점수")
    a_.scatter(*Et[~nor].T, s=55, facecolor="none", edgecolor=ABN, lw=1); a_.set_title(f"{title} · 색 = C 이상 점수 (주황 테두리 = 고장)", fontsize=10)
for a_ in axs.ravel(): a_.set_xticks([]); a_.set_yticks([])
fig.suptitle("t-SNE (perplexity 30, 강건 표준화, train 150 + val 40 함께 임베딩). 모양은 perplexity에 따라 바뀌므로 4-7의 5-NN 숫자를 함께 본다", x=.01, ha="left", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, .96)); ex.save_fig(fig, "tsne", "새 피처 공간과 ext150 공간에서 고장이 따로 모이는가 · val이 train 분포 안인가 · C 점수와 배치")

# %% psd_missed (맨 끝)
f, Lp = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, Lp, y, names, ex.focus), "psd_missed", "자주 틀린 파일의 PSD가 정상 5–95% 범위 밖으로 나가는 주파수")
log("done")
