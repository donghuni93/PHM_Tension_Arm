# %% [markdown]
# 4-14 (11-23) 묶음 균형 · 대역 맞춤 — ET 단독 OOF 진단 (모델 실험 아님, 결합 · AE · 규칙 없음)
# 4-13: 같은 개념의 열이 늘면 그 개념의 표가 커진다(057 첨도 2 → 5열). 2-15: 피처 종류마다 잘 갈리는 대역이 다르다.
# 구성 (ET 입력, 모두 H2 · H18 · 봉우리 H16 · H17 포함)
#   C0 기준 J-축소 15열                      C1 + 056 RM 3열 (참고)
#   C2 균형-선택: 묶음(첨도 · 울림 · 몫 · 소린 비 · 충격 수)마다 2열, 기존 + 모드 열 풀에서 폴드 안 분리도로 (둘째는 첫째와 |ρ| < 0.8)
#   C3 균형-요약: 묶음마다 PCA 2성분 (폴드 안 정상 기준 표준화, 라벨 안 씀)
#   C4 대역 맞춤-고정: 2-15 격자에서 피처 종류(A 울림 비율 · B 직후 몫 · C 첨도 · D 전체 몫)마다 대역 2개 (150개 전체로 고름 = 낙관적)
#   C5 대역 맞춤-폴드 안: C4와 같은 규칙을 학습 폴드에서만 다시 고름 (정직한 값)
# 지표: OOF PR-AUC(주) · 분리도 · 기준 ET가 헷갈리는 파일 안 분리도 · 초점 파일 점수 · 묶음 중요도 · 폴드마다 고른 대역
# 같은 5겹 분할, ET 300 · seed 5. 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from collections import Counter
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats
from src.exp import cached_table
from src.exp2util import NS, j_mats
from src.features2 import PeakRef, H_FOLD, mode_band_table, MODE_TAG
from src.impacts import ring_mode_table
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT, flush=True)
INK, MUTED, ABN, NOR = "#1d2430", "#5b6575", "#eb6834", "#2a78d6"

ns = NS(); y = ns.y; names = list(ns.names); n = len(y)
ns.T = pd.concat([ns.T.reset_index(drop=True), cached_table("ringmode_train", ring_mode_table, ns.X), cached_table("modeband_train", mode_band_table, ns.X)], axis=1)
ns.Tv = pd.concat([ns.Tv.reset_index(drop=True), cached_table("ringmode_val", ring_mode_table, ns.Xv), cached_table("modeband_val", mode_band_table, ns.Xv)], axis=1)
T = ns.T
GZ = np.load(os.path.join(_d, "단계", "2_EDA", "2-15_1123_대역경계지도", "figs", "격자피처.npz"))
assert list(GZ["names"]) == names
GRID = {"A": (GZ["VA"], [tuple(b) for b in GZ["BA"]]), "B": (GZ["VB"], [tuple(b) for b in GZ["BA"]]),
        "C": (GZ["VC"], [tuple(b) for b in GZ["BC"]]), "D": (GZ["VD"], [tuple(b) for b in GZ["BC"]])}
GNAME = {"A": "울림 비율", "B": "직후 몫", "C": "첨도", "D": "전체 몫"}
CNT = GZ["cnt"].astype(float)                                    # 파일당 충격 수 (2-14 캐시, 울림 비율 축소에 씀)
short = lambda s: s.replace("nor_", "N").replace("abn_", "A")
sep = lambda v, yy: abs(roc_auc_score(yy, v) - .5) * 2
def sep_cols(V, yy):
    R = stats.rankdata(V, axis=0); n1 = yy.sum(); n0 = len(yy) - n1
    return np.abs((R[yy == 1].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0) - .5) * 2

# 개념 묶음 (기존 + 모드 열 풀)
POOL = {"첨도": ["H3_kurt_1-2k", "H5_kurt_4-8k"] + [f"MK_{t}" for t in MODE_TAG],
        "울림 비율": ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"] + [f"MR_{t}" for t in MODE_TAG],
        "에너지 몫": ["H12_share_1-2k", "H13_share_2-4k", "H14_share_4-8k", "H15_share_8-12.5k"] + [f"MS_{t}" for t in MODE_TAG],
        "직후 비 (소린)": ["RM_e_joint", "RM_e_high", "SO_연결부÷본체", "SO_고주파÷본체"],
        "충격 수": ["H1_nimp", "RM_n_joint"]}
RINGS = set(POOL["울림 비율"]); OTHER = ["H2_ratio21", "H18_hfl"]
def grp_of(c):
    for g, cs in POOL.items():
        if c in cs: return g
    if c.endswith((" PC1", " PC2")): return c[:-4]
    if c.startswith("A:"): return "울림 비율"
    if c.startswith("B:"): return "직후 비 (소린)"
    if c.startswith("C:"): return "첨도"
    if c.startswith("D:"): return "에너지 몫"
    return "봉우리" if c in H_FOLD else "기타"

# %% 공통 변환 (폴드 안 정상 기준)
def shrink(V, tr):
    """울림 비율 열 축소 (k + 3·p0)/(n + 3), 충격 없음은 그대로 (J-축소와 같은 식)"""
    V = np.array(V, float); has = CNT > 0; m = (y[tr] == 0) & has[tr]
    k = V * CNT[:, None]; p0 = k[tr][m].sum(0) / CNT[tr][m].sum()
    V[has] = (k[has] + 3.0 * p0) / (CNT[has, None] + 3.0); return V
def peaks(tr):
    M, _ = j_mats(ns, np.zeros(n, bool), ["H1_nimp"]); F = M[:, 2:-2]
    return PeakRef(ns.df).fit(F[tr][y[tr] == 0]).transform(F)[:, [H_FOLD.index(p) for p in ns.PEAK]]
def tab(cols, tr):
    V = T[cols].values.astype(float); ri = [i for i, c in enumerate(cols) if c in RINGS]
    if ri: V[:, ri] = shrink(V[:, ri], tr)
    return V

def base_cols(extra=()):
    F0 = list(ns.FIXED) + list(extra)
    return lambda tr: (np.hstack([tab(F0, tr), peaks(tr)]), F0 + list(ns.PEAK))

def pick2(V, cols, tr, rho_max=.8, overlap=None):
    """학습 폴드 분리도 1위 + (|ρ| < rho_max, 겹침 조건) 중 1위"""
    s = sep_cols(V[tr], y[tr]); o = np.argsort(-s); a = o[0]
    for b in o[1:]:
        if abs(stats.spearmanr(V[tr, a], V[tr, b])[0]) < rho_max and (overlap is None or overlap(cols[a], cols[b])): return [a, b]
    return [a]

def balanced_select(tr):
    X, C = [], []
    for g, cs in POOL.items():
        V = tab(cs, tr); k = pick2(V, cs, tr) if g != "충격 수" else [0, 1]
        X.append(V[:, k]); C += [cs[i] for i in k]
    return np.hstack(X + [tab(OTHER, tr), peaks(tr)]), C + OTHER + list(ns.PEAK)

def balanced_pca(tr):
    X, C = [], []
    for g, cs in POOL.items():
        V = tab(cs, tr); N = V[tr][y[tr] == 0]; med = np.median(N, 0); sc = 1.4826 * np.median(np.abs(N - med), 0)
        sc = np.where(sc > 1e-9, sc, N.std(0) + 1e-9); Z = (V - med) / sc
        _, _, Vt = np.linalg.svd(Z[tr] - Z[tr].mean(0), full_matrices=False)
        X.append((Z - Z[tr].mean(0)) @ Vt[:2].T); C += [f"{g} PC1", f"{g} PC2"]
    return np.hstack(X + [tab(OTHER, tr), peaks(tr)]), C + OTHER + list(ns.PEAK)

def nonover(a, b):   # 두 대역이 겹치는 길이 ≤ 작은 대역의 25 %
    (a0, a1), (b0, b1) = a, b; ov = max(0, min(a1, b1) - max(a0, b0)); return ov <= .25 * min(a1 - a0, b1 - b0)
def tailored(sel_idx=None):
    def f(tr):
        idx = np.arange(n) if sel_idx == "all" else tr
        X, C = [], []
        for k, (V, B) in GRID.items():
            W = shrink(V, tr) if k == "A" else V
            p = pick2(W, B, idx, overlap=nonover); X.append(W[:, p]); C += [f"{k}:{B[i][0]}–{B[i][1]}" for i in p]
        return np.hstack(X + [tab(["H1_nimp"] + OTHER, tr), peaks(tr)]), C + ["H1_nimp"] + OTHER + list(ns.PEAK)
    return f

CFG = {"C0 기준 J-축소": base_cols(), "C1 + 056 RM 3열": base_cols(["RM_n_joint", "RM_e_joint", "RM_e_high"]),
       "C2 균형-선택 (묶음당 2열)": balanced_select, "C3 균형-요약 (묶음 PCA 2)": balanced_pca,
       "C4 대역 맞춤-고정 (낙관적)": tailored("all"), "C5 대역 맞춤-폴드 안": tailored()}
FOCUS = ["abn_022", "abn_028", "abn_002", "nor_025", "nor_027", "nor_054", "nor_002", "nor_110", "nor_037"]

# %% OOF (같은 분할, ET 300 · seed 5)
ET = lambda s: ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=s, n_jobs=-1)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y))
OOF, PICK, IMP, NCOL = {}, {}, {}, {}
for cn, mk in CFG.items():
    P = np.zeros((5, n)); picks = []
    for tr, te in FOLDS:
        X, cols = mk(tr); picks.append(cols)
        for s in range(5): P[s, te] = ET(s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    OOF[cn] = P; PICK[cn] = picks; NCOL[cn] = len(picks[0])
    al = np.arange(n); X, cols = mk(al)
    imp = pd.Series(np.mean([ET(s).fit(X, y).feature_importances_ for s in range(5)], 0), index=cols)
    IMP[cn] = imp.groupby(imp.index.map(grp_of)).sum()

b0 = OOF["C0 기준 J-축소"].mean(0)
HARD = ((y == 1) & (b0 <= np.quantile(b0[y == 1], 1 / 3))) | ((y == 0) & (b0 >= np.quantile(b0[y == 0], .9)))
ROWS = []
for cn, P in OOF.items():
    m = P.mean(0); ap = [average_precision_score(y, p) for p in P]
    row = {"구성": cn, "열": NCOL[cn], "PR-AUC": f"{np.mean(ap):.3f} ± {np.std(ap):.3f}", "분리도": round(sep(m, y), 3), "헷갈림 분리도": round(sep(m[HARD], y[HARD]), 3)}
    row.update({short(f): round(m[names.index(f)], 2) for f in FOCUS}); ROWS.append(row)
R = pd.DataFrame(ROWS); say("ET 단독 OOF (seed 5 · ET 300, 같은 분할). 헷갈림 = C0 기준이 헷갈리는 고장 10 · 정상 12"); say(R.to_string(index=False))
R.to_csv(os.path.join(FIG, "구성비교.csv"), index=False, encoding="utf-8-sig")
say("\n묶음 중요도 (전체 학습, seed 5 평균)")
say(pd.DataFrame(IMP).fillna(0).round(3).T.to_string())
for cn in ("C2 균형-선택 (묶음당 2열)", "C5 대역 맞춤-폴드 안"):
    say(f"\n[{cn}] 폴드마다 고른 열 (5겹 중 횟수): " + " · ".join(f"{c} {k}" for c, k in Counter(c for p in PICK[cn] for c in p).most_common() if c not in OTHER + list(ns.PEAK) + ["H1_nimp"]))
say("\n[C4 대역 맞춤-고정] 고른 대역: " + " · ".join(c for c in PICK["C4 대역 맞춤-고정 (낙관적)"][0] if ":" in c))

# %% 그림 1: 구성 비교 (PR-AUC · 헷갈림 분리도) + 초점 파일
fig, axs = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1, 1.4]))
ax = axs[0]; yy = np.arange(len(R))[::-1]
apm = [float(s.split(" ± ")[0]) for s in R["PR-AUC"]]; aps = [float(s.split(" ± ")[1]) for s in R["PR-AUC"]]
ax.errorbar(apm, yy, xerr=aps, fmt="o", color=INK, capsize=3, label="PR-AUC (seed 5 sd)")
ax.axvline(apm[0], color=MUTED, ls=":", lw=1); ax.set_yticks(yy); ax.set_yticklabels([f"{c} ({k}열)" for c, k in zip(R.구성, R.열)], fontsize=9)
ax2 = ax.twiny(); ax2.scatter(R["헷갈림 분리도"], yy, marker="D", color=ABN, label="헷갈리는 파일 안 분리도"); ax2.set_xlim(0, .6)
ax.set_xlabel("PR-AUC (검정 ●)", color=INK); ax2.set_xlabel("헷갈리는 파일 안 분리도 (주황 ◆)", color=ABN); ax.grid(alpha=.2)
ax = axs[1]; FF = [short(f) for f in FOCUS]; M = R[FF].values.astype(float)
im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]): ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if abs(M[i, j] - .5) > .3 else INK)
ax.set_xticks(range(len(FF))); ax.set_xticklabels([f"{f}\n({'고장' if f[0] == 'A' else '정상'})" for f in FF], fontsize=8.5); ax.set_yticks(range(len(R))); ax.set_yticklabels(R.구성, fontsize=8.5)
ax.set_title("초점 파일의 ET OOF 점수 — 고장(A)은 빨강, 정상(N)은 파랑이 좋다", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_구성비교.png"), dpi=140); plt.close(fig)

# %% 그림 2: 묶음 중요도
GR = ["첨도", "울림 비율", "에너지 몫", "직후 비 (소린)", "충격 수", "봉우리", "기타"]; GC = ["#2a78d6", "#eb6834", "#8a63d2", "#e8443a", "#3aa57a", "#f2c14e", "#b7bec9"]
fig, ax = plt.subplots(figsize=(12, 4.8)); left = np.zeros(len(CFG))
for gname, c in zip(GR, GC):
    v = np.array([IMP[cn].get(gname, 0) for cn in CFG]); ax.barh(list(CFG), v, left=left, color=c, label=gname)
    for i, (l, w) in enumerate(zip(left, v)):
        if w > .04: ax.text(l + w / 2, i, f"{w:.2f}", ha="center", va="center", fontsize=8, color="white")
    left += v
ax.invert_yaxis(); ax.set_xlim(0, 1); ax.legend(ncol=7, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.08))
ax.set_title("묶음 중요도 (합 = 1) — 균형 구성은 묶음이 고르게 나눠 갖는가", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_묶음중요도.png"), dpi=140); plt.close(fig)
OUT.close()
