# %% [markdown]
# 4-15 (11-24) 피처 풀 선별 — ext150 · D+ · 우리 모드 피처에서 '정상을 뜻하는 피처'와 '고장을 뜻하는 피처'를 찾는다 (모델 실험 아님)
# 기준 = 채택 구성 J-축소 + RM 3열(EXP-056 추가)의 ET 입력 18열 + 봉우리 2
# 단계 1 (전체 풀, 라벨은 학습 150만):
#   분리도 |AUC−0.5|×2 · 방향(고장이 크면 +) · 고장 꼬리(고장 중 정상 95 % 선 너머 비율) · 정상 꼬리(정상 중 고장 5 % 선 너머 = 고장이 거의 없는 쪽 비율)
#   헷갈리는 파일 안 분리도(기준 ET OOF로 고른 고장 10 · 정상 12, 4-13과 같은 규칙) · 기존 ET 입력과 최대 |ρ| · val 이동도(참고)
# 단계 2 (관문, 미리 정함): 분리도 ≥ 0.45 또는 꼬리 ≥ 0.30 · 기존과 |ρ| < 0.8 · 헷갈림 분리도 ≥ 0.30 → 후보끼리 |ρ| ≥ 0.8이면 헷갈림 분리도 높은 하나
# 단계 3: 후보 하나씩 기준 ET 입력에 더해 ET OOF(같은 5겹, ET 300 · seed 5): PR-AUC · 헷갈림 분리도 · 초점 파일. 통과 후보 함께 더하기
# 실행: conda run -n qe-phm python run.py (첫 실행은 D+ 물리 · 규칙 블록 계산 몇 분, 이후 figs/*.csv 캐시)

# %% 설정
import os, sys, warnings, json
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
from src.exp import cached_table
from src.exp2util import NS, j_mats, RING_COLS
from src.features import feature_table_ext
from src.features2 import PeakRef, H_FOLD, mode_band_table
from src.impacts import ring_mode_table
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT, flush=True)

ns = NS(); y = ns.y; names = list(ns.names); n = len(y); nor = y == 0
RM = ["RM_n_joint", "RM_e_joint", "RM_e_high"]; BASE = list(ns.FIXED) + RM
T = pd.concat([ns.T.reset_index(drop=True), cached_table("ringmode_train", ring_mode_table, ns.X), cached_table("modeband_train", mode_band_table, ns.X)], axis=1)
Tv = pd.concat([ns.Tv.reset_index(drop=True), cached_table("ringmode_val", ring_mode_table, ns.Xv), cached_table("modeband_val", mode_band_table, ns.Xv)], axis=1)
ns.T, ns.Tv = T, Tv                                               # j_mats가 ns.T · ns.Tv에서 읽음
E = cached_table("ext", feature_table_ext, ns.X); Ev = cached_table("ext_val", feature_table_ext, ns.Xv).reindex(columns=E.columns, fill_value=0.0)

# D+ 고유 블록 (울림 4 · 물리 10 · 규칙 5), 원값 (분위수 변환 전: 순위 기반 지표와 ET는 단조 변환에 무관)
DC = os.path.join(FIG, "dplus_train.csv"), os.path.join(FIG, "dplus_val.csv")
if not os.path.exists(DC[0]):
    from src.final import signal_tables
    from src.dplus import _blocks
    for X, p in ((ns.X, DC[0]), (ns.Xv, DC[1])):
        Tt, _ = signal_tables(X); r, u, ph = _blocks(Tt, X)
        pd.concat([r.add_prefix("D_"), ph.add_prefix("D_"), u.add_prefix("D_")], axis=1).to_csv(p, index=False, encoding="utf-8-sig")
Dp, Dpv = pd.read_csv(DC[0]), pd.read_csv(DC[1])

OURS = [c for c in T.columns if c.startswith(("MK_", "MR_", "MS_", "SO_")) or c == "RM_n_joint22"]
POOL = pd.concat([E.add_prefix("E:"), Dp, T[OURS].add_prefix("O:")], axis=1)
POOLv = pd.concat([Ev.add_prefix("E:"), Dpv, Tv[OURS].add_prefix("O:")], axis=1)
POOL = POOL.loc[:, POOL.nunique() > 3]; POOLv = POOLv[POOL.columns]
SRC = lambda c: {"E:": "ext150", "D_": "D+", "O:": "우리(모드)"}[c[:2]]
say(f"풀: {POOL.shape[1]}열 (ext150 {sum(c.startswith('E:') for c in POOL)} · D+ {sum(c.startswith('D_') for c in POOL)} · 우리 모드 {sum(c.startswith('O:') for c in POOL)})")

# %% 기준 ET 입력 · OOF
def et_input(fixed, tr, extra=None):
    """J-축소와 같은 변환(울림 비율 축소 α=3, 봉우리 PeakRef, 폴드 안 정상 기준) + extra 열(원값)"""
    M, _ = j_mats(ns, np.zeros(n, bool), fixed); nf = len(fixed); F = M[:, :-2]
    ri, i1 = [fixed.index(c) for c in RING_COLS], fixed.index("H1_nimp")
    cnt = np.rint(np.expm1(F[:, i1])); m = (y[tr] == 0) & (cnt[tr] > 0)
    p0 = (F[tr][m][:, ri] * cnt[tr][m][:, None]).sum(0) / cnt[tr][m].sum()
    H = F[:, :nf].copy(); has = cnt > 0
    H[np.ix_(has, ri)] = (H[np.ix_(has, ri)] * cnt[has, None] + 3.0 * p0) / (cnt[has, None] + 3.0)
    pk = PeakRef(ns.df).fit(F[tr][y[tr] == 0, nf + 1:]).transform(F[:, nf + 1:])[:, [H_FOLD.index(p) for p in ns.PEAK]]
    X = np.hstack([H, pk] + ([np.asarray(extra, float).reshape(n, -1)] if extra is not None else []))
    return X
ET = lambda s: ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=s, n_jobs=-1)
FOLDS = list(StratifiedKFold(5, shuffle=True, random_state=0).split(np.zeros(n), y))
def oof(extra=None):
    P = np.zeros((5, n))
    for tr, te in FOLDS:
        X = et_input(BASE, tr, extra)
        for s in range(5): P[s, te] = ET(s).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return P
P0 = oof(); b0 = P0.mean(0)
HARD = ((y == 1) & (b0 <= np.quantile(b0[y == 1], 1 / 3))) | ((y == 0) & (b0 >= np.quantile(b0[y == 0], .9)))
short = lambda s: s.replace("nor_", "N").replace("abn_", "A")
say("기준(J-축소 + RM) ET OOF PR-AUC %.3f · 헷갈리는 파일: " % np.mean([average_precision_score(y, p) for p in P0]) + " ".join(short(names[i]) for i in np.where(HARD)[0]))
np.save(os.path.join(FIG, "base_oof.npy"), P0)

# %% 단계 1: 전체 풀 지표
def auc(v, yy): return roc_auc_score(yy, v)
BASEV = T[BASE].values
rows = []
for c in POOL.columns:
    v = POOL[c].values.astype(float); v = np.where(np.isfinite(v), v, np.nanmedian(v))
    a = auc(v, y); d = 1 if a >= .5 else -1; w = d * v                 # w: 고장 쪽이 크도록 뒤집은 값
    ft = (w[y == 1] > np.quantile(w[nor], .95)).mean()                 # 고장 꼬리
    nt = (w[nor] < np.quantile(w[y == 1], .05)).mean()                 # 정상 꼬리
    ah = auc(v[HARD], y[HARD]); rho = np.abs([stats.spearmanr(v, BASEV[:, j])[0] for j in range(BASEV.shape[1])])
    k = int(np.nanargmax(rho)); vv = np.where(np.isfinite(POOLv[c].values.astype(float)), POOLv[c].values.astype(float), np.nanmedian(v))
    sh = abs(auc(np.r_[v[nor], vv], np.r_[np.zeros(nor.sum()), np.ones(len(vv))]) - .5) * 2
    rows.append(dict(피처=c, 출처=SRC(c), 분리도=abs(a - .5) * 2, 방향="고장↑" if d > 0 else "고장↓", 고장꼬리=ft, 정상꼬리=nt,
                     헷갈림분리도=abs(ah - .5) * 2, 헷갈림방향같음=(ah >= .5) == (a >= .5), 기존최대ρ=rho[k], 닮은기존=BASE[k], val이동=sh))
S = pd.DataFrame(rows).sort_values("분리도", ascending=False)
S.round(3).to_csv(os.path.join(FIG, "풀_전체지표.csv"), index=False, encoding="utf-8-sig")
say(f"\n단계 1: 전체 {len(S)}열. 분리도 ≥ 0.45: {(S.분리도 >= .45).sum()} · 고장 꼬리 ≥ 0.3: {(S.고장꼬리 >= .3).sum()} · 정상 꼬리 ≥ 0.3: {(S.정상꼬리 >= .3).sum()}")

# %% 단계 2: 관문
g1 = (S.분리도 >= .45) | (S.고장꼬리 >= .30) | (S.정상꼬리 >= .30)
g2 = S.기존최대ρ < .8; g3 = (S.헷갈림분리도 >= .30) & S.헷갈림방향같음
S["관문"] = np.select([~g1, ~g2, ~g3], ["① 약함", "② 기존과 중복", "③ 헷갈리는 파일 못 가름"], "통과")
say("관문별: " + " · ".join(f"{k} {v}" for k, v in S.관문.value_counts().items()))
C = S[S.관문 == "통과"].sort_values("헷갈림분리도", ascending=False)
keep = []
for c in C.피처:
    if all(abs(stats.spearmanr(POOL[c], POOL[k])[0]) < .8 for k in keep): keep.append(c)
S.loc[S.피처.isin(C.피처) & ~S.피처.isin(keep), "관문"] = "④ 후보끼리 중복"
say(f"후보 {len(keep)}개: " + ", ".join(keep))
say("\n" + S[S.피처.isin(keep)].round(3).to_string(index=False))

# %% 단계 3: 하나씩 더하기 + 함께 더하기 (ET OOF)
FOCUS = ["abn_022", "abn_028", "abn_009", "abn_005", "nor_025", "nor_027", "nor_054", "nor_002", "nor_110", "nor_037"]
def summ(P, label):
    m = P.mean(0); ap = [average_precision_score(y, p) for p in P]
    r = dict(구성=label, PR_AUC=np.mean(ap), PR_sd=np.std(ap), 헷갈림분리도=abs(auc(m[HARD], y[HARD]) - .5) * 2)
    r.update({short(f): m[names.index(f)] for f in FOCUS}); return r
R3 = [summ(P0, "기준 (J-축소 + RM)")]; OOFS = {}
for c in keep:
    P = oof(POOL[c].values); OOFS[c] = P; R3.append(summ(P, f"+ {c}"))
R3 = pd.DataFrame(R3); base_ap = R3.PR_AUC[0]
R3["ΔPR"] = R3.PR_AUC - base_ap; R3["Δ헷갈림"] = R3.헷갈림분리도 - R3.헷갈림분리도[0]
say("\n단계 3: 하나씩 더하기 (ET OOF, seed 5 · ET 300)"); say(R3.round(3).to_string(index=False))
ok = [c for c in keep if (R3.set_index("구성").loc[f"+ {c}", "ΔPR"] >= 0) and (R3.set_index("구성").loc[f"+ {c}", "Δ헷갈림"] >= 0)]
say(f"\n하나씩 더해 PR-AUC · 헷갈림 분리도가 둘 다 떨어지지 않은 후보: {ok}")
if len(ok) >= 2:
    P = oof(POOL[ok].values); r = summ(P, "+ 통과 전부"); say(pd.DataFrame([r]).round(3).to_string(index=False)); R3 = pd.concat([R3, pd.DataFrame([r])])
if all(c in keep for c in ("D_K1", "D_ring_peak_rms")):   # 게인 의존 열(P02_peak, 절대 g) 없이 D+ 두 열만
    r = summ(oof(POOL[["D_K1", "D_ring_peak_rms"]].values), "+ D_K1 + D_ring_peak_rms"); say(pd.DataFrame([r]).round(3).to_string(index=False)); R3 = pd.concat([R3, pd.DataFrame([r])])
say("\n후보끼리 Spearman ρ"); say(POOL[keep].corr(method="spearman").round(2).to_string())
R3.round(4).to_csv(os.path.join(FIG, "단계3_더하기.csv"), index=False, encoding="utf-8-sig")
json.dump(dict(keep=keep, ok=ok, hard=[names[i] for i in np.where(HARD)[0]]), open(os.path.join(FIG, "선별.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
POOL[keep].assign(file=names, y=y).to_csv(os.path.join(FIG, "후보_train.csv"), index=False, encoding="utf-8-sig")
POOLv[keep].assign(file=[f"val_{v[-3:]}" for v in ns.nv]).to_csv(os.path.join(FIG, "후보_val.csv"), index=False, encoding="utf-8-sig")
OUT.close()
