# %% [markdown]
# EXP-034 (교체판, 11-16) 충격 울림 피처 + 8.3 kHz 규칙 v2
# - 1차(같은 날, figs/1차/): 울림 3열(2–5 kHz 비율 등)을 ET에 넣음 → 오경보 감소, F2 +0.010(sd 안), 8.3k 규칙을 빼면 002·029 미탐, 022 그대로.
#   ext 압축만 0.916(하락), 압축+울림 0.937(기준 수준). 사용자 검토 후 아래로 교체:
#   ① 2 kHz 칸은 정상도 울린다(파일 단위 정상 16 · 고장 15) → 2.25–4 kHz만 쓴다(≥ 0.2: 고장 12 · 정상 1)
#   ② 충격 0개 파일을 0으로 채우면 '고장 쪽' 값과 섞인다 → −1로 따로 표시
#   ③ 002·029 등은 충격이 8.25 kHz로 울린다(2-16). 8 kHz대 울림을 ET 입력이 아니라 8.3k 규칙 쪽으로 넣는다
# - 가설: H1(유격 타격 → 2.25–4 kHz 울림) · H2′(타격이 8.3 kHz 모드를 울림)
# - ET 입력: ① ext150 (기준) ② ext150 + ring_225_4k (충격 중 2.125–4.125 kHz로 울린 비율, 충격 0개 = −1)
# - 8.3k 규칙 (유격 반복 규칙은 모든 구성에 있음)
#   없음 · 스펙트럼(현행: 8.20–8.45 kHz 돌출 ≥ 0.75 & 폭 ≥ 150 Hz) · 울림(충격 중 8.0–8.7 kHz로 울린 비율 ≥ 0.1, 충격 2개 이상)
#   · OR(둘 중 하나) · AND(둘 다)
# - 충격·울림 주파수 = 2-16 정의(figs/impacts.csv를 읽음, 라벨 무관). 대역 경계 2.125–4.125 · 8.0–8.7 kHz는 라벨을 본 뒤 정함 = 사후
# 실행: conda run -n qe-phm python "단계/5_모델비교/유의미/EXP-034_1116_울림피처·피처압축/run.py"

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.experts import RuleOverride, Rule83, KnockRule, ScoreStack, ae_mean, ae_segerr, warm_cache, peak83_table
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore", category=ConvergenceWarning)
NOR, ABN = "#2a78d6", "#eb6834"
ex = Experiment(__file__, "EXP-034", "충격 울림 피처(2.25–4 kHz) + 8.3 kHz 규칙 v2(충격 울림)", ["H1", "H2"],
                member="이동훈", model="AE + ET(ext150 + 2.25–4 kHz 울림) → 로지스틱 + 유격 규칙 + 8.3k 규칙(5가지)",
                changes="ET 입력에 충격 중 2.25–4 kHz로 울린 비율 1열(2-16) 추가. 8.3k 규칙에 '충격이 8.0–8.7 kHz로 울린 비율 ≥ 0.1'을 단독·OR·AND로 비교",
                focus=("abn_022", "abn_005", "nor_025", "nor_055", "nor_059", "nor_116", "abn_002", "abn_029", "abn_030"))
X, y, names = load_train(); names = [str(n) for n in names]
g = np.arange(len(y)); F = g[:, None]; nor = y == 0
short = lambda n: n.replace("abn_", "A").replace("nor_", "N")
E = cached_table("ext", feature_table_ext, X).values
IM = pd.read_csv(os.path.join(_d, "단계", "2_EDA", "유의미", "2-16_1116_울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
key = [os.path.splitext(os.path.basename(n))[0] for n in names]
cnt = IM.groupby("file").size().reindex(key).fillna(0).values


def frac(lo, hi):
    """파일마다 충격 중 [lo, hi) Hz로 울린 비율. 충격 0개 = NaN"""
    b = IM.assign(b=(IM.f >= lo) & (IM.f < hi)).groupby("file").b.mean()
    return b.reindex(key).values.astype(float)


R24 = np.nan_to_num(frac(2125, 4125), nan=-1.0)              # ET 입력 (충격 0개 = −1)
R8 = np.nan_to_num(frac(8000, 8700), nan=0.0)                 # 8.3k 규칙 v2 (충격 0개 = 0 → 꺼짐)
P83, W83, _ = peak83_table()
LOG = os.path.join(ex.figs, "log.txt"); open(LOG, "w", encoding="utf-8").close()


def log(*a):
    s = " ".join(str(v) for v in a)
    open(LOG, "a", encoding="utf-8").write(s + "\n")
    try: print(s, flush=True)
    except Exception: pass


def who(m):
    return " ".join(short(names[i]) for i in np.flatnonzero(m)) or "없음"


# %% 규칙: 8.3k v2 (학습 없음, 파일마다 고정값)
class Ring83:
    """충격이 8.0–8.7 kHz로 울린 비율 ≥ t 이고 충격 ≥ n_min개 → 고장 (te,) bool"""
    def __init__(self, t=0.1, n_min=2): self.t, self.n_min = t, n_min
    def __repr__(self): return f"Ring83(≥{self.t}, n≥{self.n_min})"
    def __call__(self, te):
        te = list(te); return (R8[te] >= self.t) & (cnt[te] >= self.n_min)


class Both:
    """두 규칙 결합. how='or' | 'and'"""
    def __init__(self, a, b, how="or"): self.a, self.b, self.how = a, b, how
    def __repr__(self): return f"{self.a} {self.how.upper()} {self.b}"
    def __call__(self, te):
        x, z = self.a(te), self.b(te)
        return (x | z) if self.how == "or" else (x & z)


S83, RG = Rule83(0.75, 150), Ring83(0.1, 2)
RULE83 = {"없음": None, "스펙트럼(현행)": S83, "울림": RG, "OR": Both(S83, RG, "or"), "AND": Both(S83, RG, "and")}
alli = np.arange(len(y))
for k, r in RULE83.items():
    if r is None: continue
    on = r(alli)
    log(f"[8.3k 규칙 {k}] 켜짐: 고장 {int((on & ~nor).sum())} ({who(on & ~nor)}) · 정상 {int((on & nor).sum())} ({who(on & nor)})")
for f in (0.8, 1.2):
    on = Ring83(0.1 * f, 2)(alli)
    log(f"  울림 문턱 ×{f}: 고장 {int((on & ~nor).sum())} · 정상 {who(on & nor)}")
log(f"  정상의 8.0–8.7 kHz 울림 비율 최대: {R8[nor].max():.2f} ({who(nor & (R8 == R8[nor].max()))})")
log(f"ET 입력 ring_225_4k ≥ 0.2: 고장 {int(((R24 >= 0.2) & ~nor).sum())} · 정상 {who((R24 >= 0.2) & nor)} · 충격 0개(−1) = 정상 {int(((R24 < 0) & nor).sum())} · 고장 {int(((R24 < 0) & ~nor).sum())}")


# %% 전문가: ET (입력 선택)
class ETX:
    """ET 점수 = logit(나무 평균 확률). inp: 'ext' | 'ext+r24'"""
    def __init__(self, inp="ext"): self.inp = inp
    def __repr__(self): return f"ET[{self.inp}]"
    def __call__(self, tr, te):
        tr, te = np.array(tr), np.array(te)
        M = np.hstack([E, R24[:, None]]) if self.inp == "ext+r24" else E
        m = make_et(seed=0)().fit(M[tr], y[tr])
        p = np.clip(m.predict_proba(M[te])[:, 1], 1 / 600, 1 - 1 / 600)
        return np.log(p / (1 - p))[:, None]


KN = KnockRule(2.0, 0.7)
def make(inp, rk):
    rules = [KN] + ([RULE83[rk]] if RULE83[rk] is not None else [])
    return lambda: RuleOverride(base_kw=dict(experts=(ae_mean, ETX(inp)), combine="lr"), rule=rules)


INPUTS = {"① ext150": "ext", "② ext150+울림": "ext+r24"}
CONFIGS = {"① ext150 | 스펙트럼(현행)": make("ext", "스펙트럼(현행)"), "① ext150 | 없음": make("ext", "없음")}
for rk in RULE83:
    CONFIGS[f"② ext150+울림 | {rk}"] = make("ext+r24", rk)
MAIN = "② ext150+울림 | 스펙트럼(현행)"
warm_cache([ae_segerr], seeds=range(5), nested=True)
log("AE 캐시 준비 끝")

# %% 중첩 CV (같은 분할)
res = {}
for nm, mk in CONFIGS.items():
    res[nm] = nested_cv(F, y, g, mk, n_jobs=-1)
    ex.record("중첩: " + nm, res[nm], names, main=(nm == MAIN))
    c = ex.res["configs"]["중첩: " + nm]
    log(f"{nm:28s} F2 {res[nm]['f2'].mean():.3f} ± {res[nm]['f2'].std():.3f} · 재현율 {res[nm]['recall'].mean():.3f} · "
        f"오경보율 {res[nm]['fpr'].mean():.3f} · 미탐 {dict(sorted(c['miss_counts'].items()))} · 오경보3+ "
        f"{ {k: v for k, v in sorted(c['fp_counts'].items()) if v >= 3} }")


# %% 그림 1 (핵심): logistic_plane 2 × 2 — 행 = 8.3k 규칙(현행 / 없음), 열 = ET 입력
def plane(ax, inp, rk, title):
    Z, rule = np.empty((len(y), 2)), np.zeros(len(y), bool)
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(g, y, g):
        Z[te] = ScoreStack(experts=(ae_mean, ETX(inp)), combine="lr").fit(F[tr], y[tr]).zscores(F[te])[:, :2]
        on = KnockRule(2.0, 0.7).fit(tr)(te)
        if RULE83[rk] is not None: on |= RULE83[rk](te)
        rule[te] = on
    Zc = np.clip(Z, -3, 4.5)
    xx, yy_ = np.meshgrid(np.linspace(-3, 4.5, 300), np.linspace(-3, 4.5, 300))
    D = LogisticRegression(C=1.0, class_weight="balanced").fit(Zc, y).decision_function(np.c_[xx.ravel(), yy_.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy_, D, levels=[-1e9, 0, 1e9], colors=["#dbe8f7", "#fbe3d6"], alpha=.8)
    ax.contour(xx, yy_, D, levels=[0], colors="k", linewidths=1.5); ax.contour(xx, yy_, D, levels=[-1, 1], colors="k", linewidths=.7, linestyles="--")
    show = ~(rule & (y == 1))
    ax.scatter(Zc[nor & show, 0], Zc[nor & show, 1], color=NOR, s=22, marker="s", lw=0, alpha=.8, label="정상")
    ax.scatter(Zc[~nor & show, 0], Zc[~nor & show, 1], color=ABN, s=26, lw=0, alpha=.9, label="고장 (규칙 밖)")
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
    ax.set_xlabel("AE z (스펙트럼 모양 이상)"); ax.set_ylabel("ET z (" + ("ext150" if inp == "ext" else "ext150 + 2.25–4 kHz 울림") + ")")
    ax.set_title(f"{title}\n중첩 F2 {c['f2']:.3f} ± {c['f2_sd']:.3f} · 재현율 {c['recall']:.3f} · 오경보율 {c['fpr']:.3f}", fontsize=9.5)


fig, axs = plt.subplots(2, 2, figsize=(16, 16))
for r, rk in enumerate(["스펙트럼(현행)", "없음"]):
    for cc, (ik, iv) in enumerate(INPUTS.items()):
        plane(axs[r, cc], iv, rk, f"{ik} | {rk}")
axs[0, 0].legend(fontsize=8, loc="upper left")
fig.suptitle("logistic_plane — 위: 8.3k 규칙 있음(현행) / 아래: 8.3k 규칙 뺌 · 왼쪽: ET 입력 ext150 / 오른쪽: ext150 + 2.25–4 kHz 울림 (OOF seed 0)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.98))
ex.save_fig(fig, "logistic_plane", "울림 1열을 넣으면 경계 근처 파일이 맞는 쪽으로 옮겨 가는가, 8.3k 규칙을 빼면 어떤 고장이 평면으로 돌아오는가", key=True)

# %% 그림 2 (핵심): 8.3k 규칙 v2 — 스펙트럼 봉우리 × 충격 울림, 문턱과 ±20% 띠
fig, axs = plt.subplots(1, 2, figsize=(17, 7))
a = axs[0]
for lab, m, c, mk in (("정상", nor, NOR, "s"), ("고장", ~nor, ABN, "o")):
    a.scatter(R8[m], P83[m], color=c, s=np.where(cnt[m] >= 2, 30, 10), alpha=.7, lw=0, marker=mk, label=lab + " (작은 점 = 충격 0–1개)")
s_on, r_on = S83(alli), RG(alli)
a.scatter(R8[s_on], P83[s_on], s=170, facecolors="none", edgecolors="#2ca02c", lw=1.8, label="스펙트럼 규칙 켜짐 (+ 폭 ≥ 150 Hz)")
a.axvspan(0.08, 0.12, color="#999", alpha=.25, lw=0, label="울림 문턱 0.1 ±20%"); a.axvline(0.1, color="k", lw=1)
a.axhspan(0.6, 0.9, color="#2ca02c", alpha=.08, lw=0); a.axhline(0.75, color="#2ca02c", lw=1, ls="--")
for i in np.flatnonzero(s_on | r_on | ((R8 > 0) & nor) | ((P83 >= 0.6) & nor)):
    a.annotate(short(names[i]), (R8[i], P83[i]), xytext=(5, 3), textcoords="offset points", fontsize=8, color=ABN if y[i] else NOR)
a.set_xlabel("충격 중 8.0–8.7 kHz로 울린 비율 (울림 규칙 축)"); a.set_ylabel("8.20–8.45 kHz 스펙트럼 돌출 P83 (현행 규칙 축)")
a.set_title("① 두 근거: 오른쪽(울림) · 위(스펙트럼). 정상은 어느 쪽 문턱에 얼마나 가까운가", loc="left", fontsize=10); a.legend(fontsize=8, loc="center right"); a.grid(alpha=.3)
a = axs[1]
rows = [k for k in RULE83 if RULE83[k] is not None]
on = np.array([[RULE83[k](alli)[i] for k in rows] for i in range(len(y))])
show = np.flatnonzero(on.any(1))
M = on[show].astype(int)
a.imshow(M, cmap="Greens", aspect="auto", vmin=0, vmax=1.4)
for i in range(M.shape[0]):
    for j in range(M.shape[1]): a.text(j, i, "●" if M[i, j] else "", ha="center", va="center", color="k")
a.set_yticks(range(len(show)), [("고장 " if y[i] else "정상 ") + names[i] + f"  (울림 {R8[i]:.2f} · P83 {P83[i]:.2f} · 폭 {W83[i]:.0f})" for i in show], fontsize=8)
a.set_xticks(range(len(rows)), rows)
a.set_title("② 규칙별로 켜지는 파일 (학습 150개 전체)", loc="left", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "rule83_v2", "8.3k 규칙을 스펙트럼 · 충격 울림 · OR · AND로 정의할 때 켜지는 파일과 정상과의 여유", key=True)

# %% 그림 3: 구성별 점수
labs = list(CONFIGS)
fig, axs = plt.subplots(1, 3, figsize=(18, 5))
for a, (k2, nm) in zip(axs, (("f2", "중첩 F2"), ("recall", "재현율"), ("fpr", "오경보율"))):
    v = [res[k][k2].mean() for k in labs]; s = [res[k][k2].std() for k in labs]
    cols = ["#999" if k.startswith("①") else "#555" for k in labs]
    a.bar(range(len(labs)), v, yerr=s, color=cols, capsize=3)
    for xv, vv in enumerate(v): a.text(xv, vv, f"{vv:.3f}", ha="center", va="bottom", fontsize=7.5)
    a.set_xticks(range(len(labs)), [k.replace(" | ", "\n") for k in labs], fontsize=7.5); a.set_title(nm, loc="left"); a.grid(alpha=.3, axis="y")
    if k2 in ("f2", "recall"): a.set_ylim(0.75, 1.0)
fig.tight_layout()
ex.save_fig(fig, "scores", "ET 입력 · 8.3k 규칙 정의별 중첩 F2 · 재현율 · 오경보율 (막대 = 평균, 선 = sd)")

# %% 그림 4: hard_files
HARD = ["abn_022", "abn_005", "abn_002", "abn_029", "abn_006", "abn_010", "abn_030", "abn_003", "abn_013", "abn_014", "abn_017", "abn_015", "abn_025",
        "nor_025", "nor_055", "nor_059", "nor_116", "nor_019", "nor_092", "nor_090", "nor_054", "nor_106"]
HARD = [h for h in HARD if h in names]
M = np.array([[ex.res["configs"]["중첩: " + c]["miss_counts" if h.startswith("abn") else "fp_counts"].get(h, 0) for c in labs] for h in HARD])
fig, ax = plt.subplots(figsize=(13, 9))
im = ax.imshow(M, cmap="Reds", vmin=0, vmax=5, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9, color="w" if M[i, j] >= 3 else "k")
ax.set_yticks(range(len(HARD)), [("고장 " if h[0] == "a" else "정상 ") + h + f"  (2.25–4k {R24[names.index(h)]:.2f} · 8k 울림 {R8[names.index(h)]:.2f})" for h in HARD], fontsize=8)
ax.set_xticks(range(len(labs)), [c.replace(" | ", "\n") for c in labs], fontsize=8)
ax.set_title("파일별 틀린 횟수 (중첩 CV seed 5번 중). 고장 = 미탐, 정상 = 오경보 · 괄호 = 울림 비율 (−1 = 충격 0개)", loc="left")
fig.colorbar(im, ax=ax, shrink=.6); fig.tight_layout()
ex.save_fig(fig, "hard_files", "어느 파일이 구성마다 맞고 틀리는가")

# %% 그림 5: t-SNE
Es = (E - E.mean(0)) / (E.std(0) + 1e-9)
fig, _ = viz.tsne({"ext150": E, "ext150 + 2.25–4 kHz 울림(×5 가중)": np.hstack([Es, (R24[:, None] - R24.mean()) / (R24.std() + 1e-9) * 5])},
                  y, names, mark=list(ex.focus))
ex.save_fig(fig, "tsne", "입력 공간마다 고장이 정상과 따로 모이는가, 문제 파일은 어디에 있는가")

# %% 그림 6 (맨 끝)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, ex.focus), "psd_missed", "문제 파일의 PSD가 정상 범위 밖으로 나가는 주파수")
log("done")
