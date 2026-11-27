# %% [markdown]
# EXP-027 Random Forest 피처 중요도 — 뽑은 피처가 정말 유의미한가
# - 질문: 지금까지 만든 손 피처 229개(ext150 + 물리 67 + 충격 9 + 유격 3) 중 무엇이 고장을 가르는가,
#   그 중요도가 우연(라벨과 무관한 피처도 나무에서 쓰이면 중요도가 생김)보다 큰가, 그 피처만으로 충분한가
# - 가설: H1(P02·E01·S01·K), H2(H2_peaky·H3_shape), H3(정규화 모양), H4(H4_*, 충격 B1–B5), H5(E1–E4)를 그룹으로 묶어 한꺼번에 본다
# - 방법 (같은 RF: 1000그루, class_weight=balanced_subsample, min_samples_leaf=2, max_features=sqrt)
#   1 MDI(불순도 감소) 중요도 — 150개 전체 학습, 설명용
#   2 귀무 중요도 검정 (Altmann 2010): 라벨을 200번 섞어 같은 RF를 학습 → 피처별 귀무 분포 → p값 → BH-FDR
#   3 단변량 검정: Mann-Whitney U(= AUC) p값 → BH-FDR. 모델 없이도 차이가 있는가
#   4 OOF 순열 중요도: 5-fold × seed 3, 학습 폴드로 RF 학습 → 시험 폴드에서 피처(또는 상관 군집 전체)를 섞어 PR-AUC 하락
#      상관이 큰 피처는 하나만 섞으면 다른 피처가 대신해 0이 나오므로 상관 군집(|ρ| > 0.7) 단위가 주 결과
#   5 가설 그룹 순열 중요도: 그룹 전체를 섞었을 때 PR-AUC 하락
#   6 안정성: 폴드 안 RF 15개에서 상위 20 안에 든 횟수
#   7 쓸모: 폴드 안 RF 중요도 상위 k개(5·10·20·40)만 ET에 넣은 quick_cv vs 전체
# - 주의: MDI·귀무 검정은 150개 전체에서 계산한 설명용 수치(성능 주장 아님). 성능 관련 수치는 모두 폴드 밖(4·5·7)
#   상관 군집은 라벨 없이 정함. 임계값을 고르지 않으므로 중첩 CV는 하지 않는다 (피처 진단 실험)
# 실행: conda run -n qe-phm python 3_단계/5_모델비교/아카이브/EXP-027_1116_RF피처중요도/run.py

# %% 설정
import os, sys, re, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
from src.exp import *
from src.evaluate import quick_cv, separation
from src.features import feature_table_impact, feature_table_knock
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

warnings.filterwarnings("ignore")
NOR, ABN, SIG, NS = "#2a78d6", "#eb6834", "#c0392b", "#9aa5b1"
ex = Experiment(__file__, "EXP-027", "Random Forest 피처 중요도 — 뽑은 피처가 정말 유의미한가", ["H1", "H2", "H3", "H4", "H5"],
                member="이동훈", model="Random Forest (중요도 진단) · ExtraTrees (상위 k 확인)",
                changes="손 피처 229개의 RF 중요도(MDI·OOF 순열·군집·가설 그룹)와 귀무 중요도 검정·단변량 FDR",
                focus=("abn_002", "abn_022", "abn_029"))
X, y, names = load_train()
g = np.arange(len(y))
OUT = ex.dir
N_NULL, N_TREES, SEEDS = 200, 1000, range(3)

tabs = [cached_table("ext", feature_table_ext, X), cached_table("phys", feature_table_phys, X),
        cached_table("impact", feature_table_impact, X), cached_table("knock", feature_table_knock, X)]
F = pd.concat(tabs, axis=1)
F = F.loc[:, F.std() > 0]
cols = list(F.columns)
A = F.values.astype(float)
print("피처", A.shape, flush=True)


def group_of(c):
    """피처 → 가설 그룹 (3_단계/1_기준·기록/물리가설.md)"""
    rules = [(r"^P02_", "P02 대역 충격 (H1)"), (r"^E01_", "E01 대역 포락선 (H1)"), (r"^S01_", "S01 대역 시간변동 (H1)"),
             (r"^K\d_", "K 유격 반복 (H1′)"), (r"^(E\d_|H1_)", "E 단발·잔향·충격수 (H1·H5)"),
             (r"^H2_", "H2 공진 날카로움 (H2)"), (r"^H3_", "H3 스펙트럼 모양 (H2·H3)"),
             (r"^(H4_|B\d_)", "H4 반전·주기 (H4)"), (r"^B0\d_", "대역 에너지 비율 (일반)"),
             (r"^F\d", "주파수 요약 (일반)"), (r"^[TW]\d", "시간·윈도우 통계 (일반)")]
    for p, gname in rules:
        if re.match(p, c):
            return gname
    return "기타"


grp = np.array([group_of(c) for c in cols])
GROUPS = list(dict.fromkeys(grp))


def make_rf(seed=0, n=N_TREES, n_jobs=-1):
    return RandomForestClassifier(n, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt",
                                  random_state=seed, n_jobs=n_jobs)


# %% 0 RF가 쓸 만한 모델인가 (중요도는 잘 맞히는 모델에서만 의미가 있다)
rf_q = lambda: make_rf(0, 500, 1)
r_rf = quick_cv(A, y, g, rf_q, seeds=SEEDS)
ext_idx = [cols.index(c) for c in tabs[0].columns if c in cols]
r_et_ext = quick_cv(A[:, ext_idx], y, g, make_et(), seeds=SEEDS)
r_et_all = quick_cv(A, y, g, make_et(), seeds=SEEDS)
qc = {"RF 229개": r_rf, "ET ext150 (기준선 EXP-002)": r_et_ext, "ET 229개": r_et_all}
for k, r in qc.items():
    print(f"{k:28s} PR-AUC {r['pr_auc'].mean():.3f} ± {r['pr_auc'].std():.3f} ROC {r['roc_auc'].mean():.3f}", flush=True)
sep_rf = separation(r_rf["oof"], y, names)
miss_rank = {names[i]: v for i, v in r_rf["miss_rank"].items()}
worst = sorted(miss_rank, key=miss_rank.get, reverse=True)[:4]
print("RF가 가장 못 잡는 고장", [(w, miss_rank[w]) for w in worst], flush=True)


# %% 1 MDI 중요도 (150개 전체, seed 10개 평균 — 설명용)
mdi_s = np.array([make_rf(s).fit(A, y).feature_importances_ for s in range(10)])
mdi, mdi_sd = mdi_s.mean(0), mdi_s.std(0)


# %% 2 귀무 중요도 검정 (라벨 섞기 200번)
rf0 = make_rf(0).fit(A, y)
act = rf0.feature_importances_
rng = np.random.default_rng(0)
null = np.array([make_rf(1000 + b).fit(A, rng.permutation(y)).feature_importances_ for b in range(N_NULL)])
p_null = (1 + (null >= act).sum(0)) / (1 + N_NULL)
max_null = null.max(1)                           # 섞은 라벨에서 '가장 중요한 피처'의 중요도 (가족 전체 기준)
p_fwer = (1 + (max_null[:, None] >= act).sum(0)) / (1 + N_NULL)


def bh(p):
    """Benjamini–Hochberg FDR q값"""
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[o] = np.minimum(q, 1)
    return out


q_null = bh(p_null)


# %% 3 단변량 (Mann-Whitney U = AUC)
auc = np.array([roc_auc_score(y, A[:, j]) for j in range(A.shape[1])])
p_mw = np.array([mannwhitneyu(A[y == 1, j], A[y == 0, j]).pvalue for j in range(A.shape[1])])
q_mw = bh(p_mw)


# %% 상관 군집 (라벨 없음): Spearman |ρ| > 0.7 이면 같은 군집 (평균 연결, 거리 1−|ρ| ≤ 0.3)
rho = np.abs(spearmanr(A).correlation)
rho = np.nan_to_num(rho, nan=0.0); np.fill_diagonal(rho, 1)
Z = linkage(squareform(1 - rho, checks=False), "average")
clu = fcluster(Z, t=0.3, criterion="distance")
CL = sorted(set(clu))
print("상관 군집 수", len(CL), flush=True)


# %% 4·5 OOF 순열 중요도 (피처 · 상관 군집 · 가설 그룹)
UNITS = ([("f", [j]) for j in range(len(cols))] + [("c", list(np.flatnonzero(clu == c))) for c in CL]
         + [("g", list(np.flatnonzero(grp == gname))) for gname in GROUPS])
N_REP = 3


def fold_task(seed, tr, te):
    m = make_rf(seed, 500, 1).fit(A[tr], y[tr])
    base = m.predict_proba(A[te])[:, 1]
    r = np.random.default_rng(seed * 100 + te[0])
    out = np.empty((len(UNITS), N_REP, len(te)))
    for u, (_, idx) in enumerate(UNITS):
        for k in range(N_REP):
            B = A[te].copy()
            p = r.permutation(len(te))
            B[:, idx] = B[p][:, idx]               # 군집·그룹은 같은 순열로 함께 섞어 군집 안 관계를 유지
            out[u, k] = m.predict_proba(B)[:, 1]
    top = np.argsort(m.feature_importances_)[::-1][:20]
    return seed, te, base, out, top, m.feature_importances_


jobs = [(s, tr, te) for s in SEEDS for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(A, y, g)]
res = Parallel(n_jobs=min(15, os.cpu_count()))(delayed(fold_task)(*j) for j in jobs)
oof_b = np.zeros((len(SEEDS), len(y))); oof_p = np.zeros((len(SEEDS), len(UNITS), N_REP, len(y)))
top_count = np.zeros(len(cols)); fold_imp = []
for s, te, base, out, top, fi in res:
    oof_b[s, te] = base; oof_p[s, :, :, te] = np.moveaxis(out, 2, 0)
    top_count[top] += 1; fold_imp.append(fi)
ap_b = np.array([average_precision_score(y, o) for o in oof_b])
drop = np.array([[ap_b[s] - np.mean([average_precision_score(y, oof_p[s, u, k]) for k in range(N_REP)])
                  for s in range(len(SEEDS))] for u in range(len(UNITS))])      # (unit, seed)
nF, nC = len(cols), len(CL)
perm_f, perm_c, perm_g = drop[:nF], drop[nF:nF + nC], drop[nF + nC:]


# %% 표 저장 (자료)
T = pd.DataFrame(dict(feature=cols, group=grp, cluster=clu, mdi=mdi, mdi_sd=mdi_sd, mdi_rank=(-mdi).argsort().argsort() + 1,
                      null_p=p_null, null_q=q_null, null_p_fwer=p_fwer, null_95=np.percentile(null, 95, 0),
                      auc=auc, auc_dir=np.where(auc >= .5, "+", "-"), mw_p=p_mw, mw_q=q_mw,
                      perm_drop=perm_f.mean(1), perm_drop_sd=perm_f.std(1), top20_in_folds=top_count.astype(int)))
T = T.sort_values("mdi", ascending=False).reset_index(drop=True)
T.to_csv(os.path.join(OUT, "importance_table.csv"), index=False, encoding="utf-8-sig")
C = []
for k, c in enumerate(CL):
    idx = np.flatnonzero(clu == c)
    rep = idx[np.argmax(mdi[idx])]
    C.append(dict(cluster=c, n=len(idx), representative=cols[rep], groups=" · ".join(sorted(set(grp[idx]))),
                  mdi_sum=mdi[idx].sum(), perm_drop=perm_c[k].mean(), perm_drop_sd=perm_c[k].std(),
                  n_sig_fdr=int((q_null[idx] < .05).sum()), members=" ".join(cols[i] for i in idx[:12]) + (" …" if len(idx) > 12 else "")))
C = pd.DataFrame(C).sort_values("perm_drop", ascending=False).reset_index(drop=True)
C.to_csv(os.path.join(OUT, "cluster_table.csv"), index=False, encoding="utf-8-sig")
G = pd.DataFrame([dict(group=gname, n=int((grp == gname).sum()), mdi_sum=mdi[grp == gname].sum(),
                       mdi_per_feat=mdi[grp == gname].mean(), n_sig_fdr=int((q_null[grp == gname] < .05).sum()),
                       n_sig_mw=int((q_mw[grp == gname] < .05).sum()),
                       perm_drop=perm_g[k].mean(), perm_drop_sd=perm_g[k].std()) for k, gname in enumerate(GROUPS)])
G = G.sort_values("perm_drop", ascending=False).reset_index(drop=True)
G.to_csv(os.path.join(OUT, "group_table.csv"), index=False, encoding="utf-8-sig")


# %% 7 쓸모: 폴드 안 RF 중요도 상위 k → ET
class TopKRF(BaseEstimator):
    """학습 폴드로 RF(500) 중요도를 매겨 상위 k개만 ET에 넣는다 (선택이 폴드 안)"""
    def __init__(self, k=10): self.k = k
    def fit(self, Xf, yf):
        imp = make_rf(0, 500, 1).fit(Xf, yf).feature_importances_
        self.cols_ = np.argsort(imp)[::-1][:self.k]
        self.m_ = make_et()().fit(Xf[:, self.cols_], yf); return self
    def predict_proba(self, Xf): return self.m_.predict_proba(Xf[:, self.cols_])


topk = {}
for k in (5, 10, 20, 40):
    topk[k] = quick_cv(A, y, g, (lambda k=k: TopKRF(k)), seeds=SEEDS)
    print(f"상위 {k:3d}  PR-AUC {topk[k]['pr_auc'].mean():.3f} ± {topk[k]['pr_auc'].std():.3f}", flush=True)
# 비교: 가설 그룹 하나만으로
grp_only = {gname: quick_cv(A[:, grp == gname], y, g, make_et(), seeds=SEEDS) for gname in GROUPS}


# %% 요약 수치 → result.json
summ = dict(
    n_features=len(cols), n_clusters=len(CL),
    quick_cv={k: f"{r['pr_auc'].mean():.3f} ± {r['pr_auc'].std():.3f} (ROC {r['roc_auc'].mean():.3f})" for k, r in qc.items()},
    topk_et={str(k): f"{r['pr_auc'].mean():.3f} ± {r['pr_auc'].std():.3f}" for k, r in topk.items()},
    group_only_et={k: f"{r['pr_auc'].mean():.3f} ± {r['pr_auc'].std():.3f}" for k, r in grp_only.items()},
    n_sig_null_fdr05=int((q_null < .05).sum()), n_sig_null_fwer05=int((p_fwer < .05).sum()),
    n_sig_mw_fdr05=int((q_mw < .05).sum()),
    n_sig_both=int(((q_null < .05) & (q_mw < .05)).sum()),
    rf_separation=dict(sure_a=sep_rf["sure_a"], sure_n=sep_rf["sure_n"], gray_a=sep_rf["gray_a"]),
    rf_worst_missed={w: miss_rank[w] for w in worst},
    max_null_mdi_95=float(np.percentile(max_null, 95)), uniform_mdi=1 / len(cols),
    top20_mdi=T.head(20)[["feature", "mdi", "null_q", "null_p_fwer", "auc", "perm_drop", "top20_in_folds"]].round(4).to_dict("records"),
    spearman_mdi_vs_absauc=float(spearmanr(mdi, np.abs(auc - .5)).correlation),
)
ex.res["summary"] = summ
ex.res["missed"] = worst
ex._write()
import json
json.dump(summ, open(os.path.join(OUT, "summary_utf8.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


# %% 그림 1 (핵심): 실제 중요도 vs 섞은 라벨 중요도 — 상위 30
top = T.head(30).iloc[::-1]
ix = [cols.index(c) for c in top.feature]
fig, ax = plt.subplots(figsize=(11, 10))
yy = np.arange(len(top))
ax.fill_betweenx(yy, np.percentile(null[:, ix], 5, 0), np.percentile(null[:, ix], 95, 0), color=NS, alpha=.35,
                 step="mid", label="섞은 라벨 200번의 중요도 5–95%")
ax.axvline(np.percentile(max_null, 95), color="k", ls="--", lw=.8, label="섞은 라벨에서 '1등 피처' 중요도 95% (가족 전체 기준)")
col = [SIG if q < .05 else NS for q in top.null_q]
ax.barh(yy, top.mdi, xerr=top.mdi_sd, color=col, alpha=.85, height=.6, label="실제 MDI (seed 10 평균 ± sd), 빨강 = FDR q < 0.05")
ax.set_yticks(yy); ax.set_yticklabels([f"{f}  [{gname.split(' (')[0]}]" for f, gname in zip(top.feature, top.group)], fontsize=8)
ax.set_xlabel("MDI 중요도 (불순도 감소 합, 전체 = 1)")
ax.set_title("RF 중요도 상위 30: 우연(섞은 라벨)보다 큰가", fontsize=11)
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=.3)
fig.tight_layout()
ex.save_fig(fig, "null_importance", "막대(실제 중요도)가 회색 띠(라벨을 섞었을 때 같은 피처가 얻는 중요도)와 점선(섞은 라벨에서 1등이 얻는 중요도)을 넘는가", key=True)


# %% 그림 2 (핵심): 가설 그룹·상관 군집 순열 중요도 (폴드 밖)
fig, axs = plt.subplots(1, 2, figsize=(17, 7.5))
ax = axs[0]
Gs = G.iloc[::-1]
ax.barh(range(len(Gs)), Gs.perm_drop, xerr=Gs.perm_drop_sd, color=[ABN if "일반" not in s else NS for s in Gs.group])
ax.set_yticks(range(len(Gs))); ax.set_yticklabels([f"{s} ({n}개)" for s, n in zip(Gs.group, Gs.n)], fontsize=9)
ax.axvline(0, color="k", lw=.6); ax.grid(axis="x", alpha=.3)
ax.set_xlabel("그룹 전체를 섞었을 때 OOF PR-AUC 하락 (seed 3 평균 ± sd)")
ax.set_title(f"가설 그룹 순열 중요도 (기준 OOF PR-AUC {ap_b.mean():.3f})", fontsize=10)
ax = axs[1]
Cs = C.head(15).iloc[::-1]
ax.barh(range(len(Cs)), Cs.perm_drop, xerr=Cs.perm_drop_sd, color=ABN)
ax.set_yticks(range(len(Cs))); ax.set_yticklabels([f"{r} 외 {n - 1}개" if n > 1 else r for r, n in zip(Cs.representative, Cs.n)], fontsize=8)
ax.axvline(0, color="k", lw=.6); ax.grid(axis="x", alpha=.3)
ax.set_xlabel("군집 전체를 섞었을 때 OOF PR-AUC 하락")
ax.set_title(f"상관 군집(|ρ| > 0.7, {len(CL)}개) 순열 중요도 상위 15 · 이름 = 군집 안 MDI 1위", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "group_cluster_perm", "폴드 밖에서 그 묶음을 망가뜨리면 성능이 얼마나 떨어지는가 (회색 = 가설 없는 일반 피처)", key=True)


# %% 그림 3: 세 가지 중요도가 같은 말을 하는가 (MDI vs 단변량 AUC vs 순열)
fig, axs = plt.subplots(1, 2, figsize=(15, 6))
ax = axs[0]
sig = (q_null < .05)
for gname in GROUPS:
    m = grp == gname
    ax.scatter(np.abs(auc[m] - .5) + .5, mdi[m], s=22, alpha=.75, label=gname)
for j in np.argsort(mdi)[::-1][:10]:
    ax.annotate(cols[j], (abs(auc[j] - .5) + .5, mdi[j]), fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.axhline(np.percentile(max_null, 95), color="k", ls="--", lw=.8)
ax.axvline(.5 + (np.sort(np.abs(auc - .5))[::-1][(q_mw < .05).sum() - 1] if (q_mw < .05).any() else .5), color="gray", ls=":", lw=.8)
ax.set_xlabel("단변량 AUC (방향 무시 = max(AUC, 1−AUC)), 점선 = 단변량 FDR 0.05 경계"); ax.set_ylabel("MDI")
ax.set_title(f"MDI vs 단변량 AUC (Spearman {summ['spearman_mdi_vs_absauc']:.2f})", fontsize=10)
ax.legend(fontsize=7, ncol=2); ax.grid(alpha=.3)
ax = axs[1]
ax.scatter(mdi, perm_f.mean(1), s=18, c=[SIG if s else NS for s in sig])
for j in np.argsort(mdi)[::-1][:10]:
    ax.annotate(cols[j], (mdi[j], perm_f[j].mean()), fontsize=7, xytext=(3, 2), textcoords="offset points")
ax.axhline(0, color="k", lw=.6)
ax.set_xlabel("MDI"); ax.set_ylabel("피처 하나 순열 PR-AUC 하락 (폴드 밖)")
ax.set_title("피처 하나를 섞어도 거의 안 떨어진다 = 상관 피처가 대신함 (빨강 = 귀무 FDR q < 0.05)", fontsize=10)
ax.grid(alpha=.3)
fig.tight_layout()
ex.save_fig(fig, "agreement", "MDI·단변량 AUC·개별 순열 중요도가 같은 피처를 가리키는가")


# %% 그림 4: 안정성 + 상위 k
fig, axs = plt.subplots(1, 2, figsize=(16, 6.5))
ax = axs[0]
o = np.argsort(top_count)[::-1][:30][::-1]
ax.barh(range(len(o)), top_count[o], color=[SIG if q_null[j] < .05 else NS for j in o])
ax.set_yticks(range(len(o))); ax.set_yticklabels([cols[j] for j in o], fontsize=8)
ax.axvline(len(jobs), color="k", ls="--", lw=.8)
ax.set_xlabel(f"폴드 안 RF {len(jobs)}개 중 상위 20에 든 횟수"); ax.set_title("중요도 순위 안정성", fontsize=10); ax.grid(axis="x", alpha=.3)
ax = axs[1]
ks = list(topk)
mu = [topk[k]["pr_auc"].mean() for k in ks]; sd = [topk[k]["pr_auc"].std() for k in ks]
ax.errorbar(ks, mu, yerr=sd, marker="o", color=ABN, label="폴드 안 RF 상위 k → ET")
for lab, r, c in (("ET 229개 전체", r_et_all, "k"), ("ET ext150 (기준선)", r_et_ext, NOR), ("RF 229개", r_rf, "gray")):
    ax.axhspan(r["pr_auc"].mean() - r["pr_auc"].std(), r["pr_auc"].mean() + r["pr_auc"].std(), color=c, alpha=.12)
    ax.axhline(r["pr_auc"].mean(), color=c, lw=1, label=f"{lab} {r['pr_auc'].mean():.3f}")
ax.set_xscale("log"); ax.set_xticks(ks); ax.set_xticklabels(ks)
ax.set_xlabel("k (피처 수)"); ax.set_ylabel("quick_cv PR-AUC"); ax.legend(fontsize=8); ax.grid(alpha=.3)
ax.set_title("중요한 피처만으로 충분한가", fontsize=10)
fig.tight_layout()
ex.save_fig(fig, "stability_topk", "왼쪽: 학습 데이터가 바뀌어도 같은 피처가 상위에 오는가 / 오른쪽: 상위 k개만 써도 전체만큼 맞히는가")


# %% 그림 5: 놓친 고장이 중요 피처에서 정상 범위 안에 있는가
top15 = list(T.feature.head(15))
fp_n = [names[i] for i in np.argsort(np.where(y == 0, r_rf["oof"].mean(0), -1))[::-1][:3]]
fig = viz.feature_percentile(F, y, names, worst + fp_n, top15)
ex.save_fig(fig, "missed_percentile", f"RF가 가장 못 잡은 고장 {', '.join(worst)}와 점수가 가장 높은 정상 {', '.join(fp_n)}이 상위 15 피처에서 정상 몇 백분위인가")


# %% 그림 6: t-SNE (매 실험 기본)
top20i = [cols.index(c) for c in T.feature.head(20)]
fig, _ = viz.tsne({"229개 전체": A, "MDI 상위 20 (전체 데이터로 고름, 설명용)": A[:, top20i]}, y, names,
                  mark=worst + fp_n, score=r_rf["oof"].mean(0))
ex.save_fig(fig, "tsne", "중요 피처 공간에서 고장이 따로 모이는가, RF가 놓친 파일은 어디에 있는가 (색 = RF OOF 점수)")


# %% 그림 7: 놓친 파일 PSD vs 정상 5–95% (매 실험 기본, 맨 끝)
f, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(f, L, y, names, worst), "psd_missed", "RF가 가장 못 잡은 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")
print("완료", flush=True)
