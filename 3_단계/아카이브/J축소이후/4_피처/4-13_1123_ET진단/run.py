# %% [markdown]
# 4-13 (11-23) ET 진단 — 의미 있는 피처를 넣었는데 왜 성능이 떨어지나 (모델 실험 없음, ET 단독 OOF만)
# 구성: 기준 J-축소 ET 입력 · 056 추가(+RM 3열) · 057(모드 5칸 + 소린 2열). ET 입력 = J-축소와 같다(울림 비율 축소 α=3, 봉우리 H16 · H17, 폴드 안 정상 기준)
# 1 중복: 새 열과 기존 열의 Spearman |ρ| 최대
# 2 새 정보: 기준 ET가 헷갈리는 파일(점수 낮은 고장 · 점수 높은 정상)만 놓고 분리도
# 3 ET 안: 열 묶음별 중요도, 초점 파일을 고장으로 판단한 나무가 경로에서 쓴 열
# 4 나무 수: 같은 폴드, ET seed 5개 → 나무 100 · 300 · 500 · 1000에서 파일 점수의 seed 간 sd와 판정 뒤집힘
# 실행: conda run -n qe-phm python run.py

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
from src.exp import cached_table
from src.exp2util import NS, j_mats, RING_COLS
from src.features2 import PeakRef, H_FOLD, mode_band_table, MODE_TAG
from src.impacts import ring_mode_table
from src.evaluate import best_f2_threshold
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT, flush=True)
NOR, ABN, INK, MUTED = "#2a78d6", "#eb6834", "#1d2430", "#5b6575"

ns = NS(); y = ns.y; names = list(ns.names); n = len(y)
ns.T = pd.concat([ns.T.reset_index(drop=True), cached_table("ringmode_train", ring_mode_table, ns.X),
                  cached_table("modeband_train", mode_band_table, ns.X)], axis=1)
ns.Tv = pd.concat([ns.Tv.reset_index(drop=True), cached_table("ringmode_val", ring_mode_table, ns.Xv),
                   cached_table("modeband_val", mode_band_table, ns.Xv)], axis=1)   # j_mats가 val 행렬도 만들어서 (여기선 안 씀)
short = lambda s: s.replace("nor_", "N").replace("abn_", "A")
F0 = list(ns.FIXED); RM = ["RM_n_joint", "RM_e_joint", "RM_e_high"]
KEEP = ["H1_nimp", "H2_ratio21", "H18_hfl"]
MC = [f"MK_{t}" for t in MODE_TAG] + [f"MR_{t}" for t in MODE_TAG] + [f"MS_{t}" for t in MODE_TAG] + ["SO_연결부÷본체", "SO_고주파÷본체"]
CFG = {"기준 J-축소": (F0, RING_COLS), "056 추가": (F0 + RM, RING_COLS), "057 모드 5칸": (KEEP + MC, [f"MR_{t}" for t in MODE_TAG])}
FOCUS = ["abn_022", "abn_028", "abn_005", "nor_025", "nor_027", "nor_073", "nor_054", "nor_002", "nor_110"]
sep = lambda v, yy=y: abs(roc_auc_score(yy, v) - .5) * 2

def group(c):   # 열 → 개념 묶음
    if c.startswith(("H3_", "H4_", "H5_", "H6_", "MK_")) or "kurt" in c: return "첨도"
    if c.startswith(("H7_", "H8_", "H9_", "H10_", "MR_")) or "ring" in c: return "울림 비율"
    if c.startswith(("H12_", "H13_", "H14_", "H15_", "MS_")) or "share" in c: return "에너지 몫"
    if c.startswith(("SO_", "RM_e")): return "충격 직후 비 (소린)"
    if c in ("H1_nimp", "RM_n_joint"): return "충격 수"
    if c in H_FOLD: return "봉우리"
    return "기타"

# %% ET 입력 (J-축소와 같은 변환, 폴드 안 정상 기준)
def et_input(fixed, ring, tr, te):
    M, _ = j_mats(ns, np.zeros(n, bool), fixed); nf = len(fixed); F = M[:, :-2]
    ri, i1 = [fixed.index(c) for c in ring], fixed.index("H1_nimp")
    cnt = np.rint(np.expm1(F[:, i1])); m = (y[tr] == 0) & (cnt[tr] > 0)
    p0 = (F[tr][m][:, ri] * cnt[tr][m][:, None]).sum(0) / cnt[tr][m].sum()
    H = F[:, :nf].copy(); has = cnt > 0
    H[np.ix_(has, ri)] = (H[np.ix_(has, ri)] * cnt[has, None] + 3.0 * p0) / (cnt[has, None] + 3.0)
    pr = PeakRef(ns.df).fit(F[tr][y[tr] == 0, nf + 1:]); pk = pr.transform(F[:, nf + 1:])[:, [H_FOLD.index(p) for p in ns.PEAK]]
    X = np.hstack([H, pk]); return X[tr], X[te], fixed + list(ns.PEAK)

ET = lambda N, s: ExtraTreesClassifier(N, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=s, n_jobs=-1)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y))
NT, SEEDS, NS_ = (100, 300, 500, 1000), range(5), 1000

# %% OOF: 같은 폴드 · ET seed 5개 · 나무 1000 (앞 N그루 = 나무 N개 숲)
TP, VOTE = {}, {}
for cn, (fx, rg) in CFG.items():
    P = np.zeros((len(SEEDS), NS_, n)); paths = {f: [] for f in FOCUS}
    for s in SEEDS:
        for tr, te in FOLDS:
            Xtr, Xte, cols = et_input(fx, rg, tr, te); m = ET(NS_, s).fit(Xtr, y[tr])
            P[s][:, te] = np.stack([t.predict_proba(Xte)[:, 1] for t in m.estimators_])
            if s == 0:
                for f in FOCUS:
                    i = names.index(f)
                    if i in te:
                        xi = Xte[list(te).index(i)][None]
                        for t in m.estimators_:
                            nodes = t.decision_path(xi).indices; ft = t.tree_.feature[nodes]; ft = ft[ft >= 0]
                            paths[f].append((t.predict_proba(xi)[0, 1] > .5, [cols[k] for k in ft]))
    TP[cn] = P; VOTE[cn] = (paths, cols)
    say(f"[{cn}] ET 입력 {len(cols)}열, max_features = {int(np.sqrt(len(cols)))} → 한 갈림에서 특정 열이 후보에 들 확률 {int(np.sqrt(len(cols))) / len(cols):.3f}")

base = TP["기준 J-축소"].mean((0, 1))                    # 기준 ET OOF 점수 (seed · 나무 평균)

# %% 1 중복 · 2 새 정보
Xall = ns.T; OLD = F0
hard_f = (y == 1) & (base <= np.quantile(base[y == 1], 1 / 3)); hard_n = (y == 0) & (base >= np.quantile(base[y == 0], .9))
HARD = hard_f | hard_n
say(f"\n헷갈리는 파일: 고장 {hard_f.sum()} (기준 점수 하위 1/3) · 정상 {hard_n.sum()} (상위 10%)")
say("  고장: " + " ".join(short(names[i]) for i in np.where(hard_f)[0])); say("  정상: " + " ".join(short(names[i]) for i in np.where(hard_n)[0]))
R1 = []
for c in OLD + RM + MC:
    v = Xall[c].values; rho = [abs(spearmanr(v, Xall[o].values)[0]) if o != c else 0 for o in OLD]; k = int(np.nanargmax(rho))
    R1.append(dict(열=c, 묶음=group(c), 새열=c not in OLD, 전체분리도=round(sep(v), 3), 헷갈림분리도=round(sep(v[HARD], y[HARD]), 3),
                   기존과최대ρ=round(rho[k], 2), 가장닮은기존열=OLD[k]))
R1 = pd.DataFrame(R1); say("\n1 · 2 열별: 전체 분리도 · 헷갈리는 파일 안 분리도 · 기존 열과 최대 |ρ|"); say(R1.to_string(index=False))
R1.to_csv(os.path.join(FIG, "열별_중복·새정보.csv"), index=False, encoding="utf-8-sig")
say(f"\n기준 ET 점수와 고장 안 Spearman: " + " · ".join(f"{c} {spearmanr(Xall[c].values[y == 1], base[y == 1])[0]:+.2f}" for c in RM + ["SO_연결부÷본체", "SO_고주파÷본체"]))

# %% 3 ET 안: 묶음 중요도 (전체 학습, seed 5 평균) · 초점 파일 경로
IMP = {}
for cn, (fx, rg) in CFG.items():
    al = np.arange(n); imps = []
    for s in SEEDS:
        Xtr, _, cols = et_input(fx, rg, al, al); imps.append(ET(1000, s).fit(Xtr, y).feature_importances_)
    imp = pd.Series(np.mean(imps, 0), index=cols); IMP[cn] = imp
    say(f"\n[{cn}] 중요도 상위 8: " + " · ".join(f"{c} {v:.3f}" for c, v in imp.sort_values(ascending=False).head(8).items()))
    say("  묶음 합: " + " · ".join(f"{g} {v:.2f}" for g, v in imp.groupby(imp.index.map(group)).sum().sort_values(ascending=False).items()))
say("\n3 초점 파일 (폴드 밖, seed 0, 나무 1000): 고장 투표 비율 · 고장 투표 나무의 경로에 가장 많이 나온 묶음")
ROWS3 = []
for f in FOCUS:
    row = dict(파일=short(f), 라벨="고장" if f.startswith("abn") else "정상")
    for cn in CFG:
        paths, cols = VOTE[cn]; pv = paths[f]; fv = [p for v, p in pv if v]
        g = pd.Series([group(c) for p in fv for c in p]).value_counts(normalize=True)
        row[cn] = f"{len(fv) / len(pv):.2f} ({', '.join(f'{k} {v:.0%}' for k, v in g.head(2).items())})"
    ROWS3.append(row)
R3 = pd.DataFrame(ROWS3); say(R3.to_string(index=False))

# %% 4 나무 수
say("\n4 나무 수: 파일 점수(확률)의 ET seed 간 sd 중앙값 · 판정 뒤집힘(기준 1000그루 평균 F2 문턱, seed 5개 판정이 갈린 파일 수)")
R4 = []
for cn in CFG:
    P = TP[cn]; thr = best_f2_threshold(y, P.mean((0, 1)))
    for N in NT:
        S = P[:, :N].mean(1)                                   # seed × 파일
        sd = S.std(0); flip = ((S >= thr).any(0) & (S < thr).any(0))
        R4.append(dict(구성=cn, 나무=N, sd중앙=round(np.median(sd), 4), sd_초점=round(np.median(sd[[names.index(f) for f in FOCUS]]), 4),
                       뒤집힘=int(flip.sum()), 뒤집힌파일=" ".join(short(names[i]) for i in np.where(flip)[0])))
R4 = pd.DataFrame(R4); say(R4.to_string(index=False))

# %% 그림 1: 새 열의 중복 · 새 정보
NEW = R1[R1.새열].reset_index(drop=True); OLDR = R1[~R1.새열]
fig, axs = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
ax = axs[0]; x = np.arange(len(NEW))
ax.bar(x - .2, NEW.전체분리도, .4, color=INK, label="전체 150개 분리도"); ax.bar(x + .2, NEW.헷갈림분리도, .4, color=ABN, label="헷갈리는 파일 안 분리도")
ax.set_xticks(x); ax.set_xticklabels(NEW.열, rotation=60, ha="right", fontsize=8); ax.legend(fontsize=9); ax.grid(axis="y", alpha=.2)
ax.set_title("새 열: 전체로는 잘 갈려도(검정) 기준 ET가 헷갈리는 파일에서는 약하다(주황)", loc="left", fontsize=10.5)
ax = axs[1]
for d, c, lab in ((OLDR, MUTED, "기존 열"), (NEW, ABN, "새 열")):
    ax.scatter(d.기존과최대ρ, d.헷갈림분리도, s=45, color=c, label=lab, edgecolor="white")
for r in NEW.itertuples():
    if r.헷갈림분리도 > .35 or r.기존과최대ρ > .85: ax.annotate(r.열, (r.기존과최대ρ, r.헷갈림분리도), fontsize=7.5, xytext=(3, 3), textcoords="offset points")
ax.set_xlabel("기존 열과 최대 |ρ| (오른쪽 = 이미 있는 정보)", color=MUTED); ax.set_ylabel("헷갈리는 파일 안 분리도 (위 = 새로 도움)", color=MUTED)
ax.legend(fontsize=9); ax.grid(alpha=.2); ax.set_title("도움이 되려면 왼쪽 위 (새 정보 + 헷갈리는 파일을 가름)", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_중복·새정보.png"), dpi=140); plt.close(fig)

# %% 그림 2: 묶음 중요도
GR = ["첨도", "울림 비율", "에너지 몫", "충격 직후 비 (소린)", "충격 수", "봉우리", "기타"]; GC = ["#2a78d6", "#eb6834", "#8a63d2", "#e8443a", "#3aa57a", "#f2c14e", "#b7bec9"]
fig, ax = plt.subplots(figsize=(11, 4.2)); left = np.zeros(len(CFG))
for gname, c in zip(GR, GC):
    v = np.array([IMP[cn].groupby(IMP[cn].index.map(group)).sum().get(gname, 0) for cn in CFG])
    ax.barh(list(CFG), v, left=left, color=c, label=gname)
    for i, (l, w) in enumerate(zip(left, v)):
        if w > .04: ax.text(l + w / 2, i, f"{w:.2f}", ha="center", va="center", fontsize=8.5, color="white")
    left += v
ax.invert_yaxis(); ax.legend(ncol=4, fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.12)); ax.set_xlim(0, 1)
ax.set_title("ET 중요도를 개념 묶음별로 (합 = 1) — 열이 늘어난 묶음이 중요도를 가져간다", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_묶음중요도.png"), dpi=140); plt.close(fig)

# %% 그림 3: 나무 수
fig, axs = plt.subplots(1, 2, figsize=(13, 4.2))
for cn, c in zip(CFG, (INK, ABN, "#8a63d2")):
    d = R4[R4.구성 == cn]; axs[0].plot(d.나무, d.sd중앙, "o-", color=c, label=cn); axs[1].plot(d.나무, d.뒤집힘, "o-", color=c, label=cn)
axs[0].set_title("파일 점수의 ET seed 간 sd (중앙값)", loc="left", fontsize=10.5); axs[1].set_title("seed 5개 판정이 갈린 파일 수", loc="left", fontsize=10.5)
for ax in axs: ax.set_xlabel("나무 수", color=MUTED); ax.grid(alpha=.2); ax.legend(fontsize=9); ax.set_xticks(NT)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_나무수.png"), dpi=140); plt.close(fig)
OUT.close()
