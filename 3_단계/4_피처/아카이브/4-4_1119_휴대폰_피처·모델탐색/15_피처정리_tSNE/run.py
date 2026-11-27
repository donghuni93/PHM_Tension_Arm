# %% [markdown]
# 15. 피처 정리 → t-SNE: 고장과 정상이 가장 잘 갈리는 정리된 피처 집합을 정직하게 찾는다
# 정직한 평가: 피처 순위·개수(k)는 항상 학습 폴드 안에서만 정하고, 시험 폴드 점수(로지스틱 PR-AUC · AUC)와 t-SNE 이웃 순도로 비교한다.
#   (전체 150개로 피처를 고른 뒤 그린 t-SNE는 분리가 과장되므로, 그림 제목에 '폴드 밖(OOF) 순도'를 따로 적는다)
# 정리 순서: ① 분위수 변환(학습 150개로 fit) ② 라벨 없이 상관 군집(|ρ| ≥ 0.85)당 대표 1개 ③ (폴드 안) 단변량 AUC 상위 k개 ④ k를 OOF로 고름
# 후보 묶음: A = ext150 전체 · B = 대표만(라벨 없는 정리) · C = B + 울림 4열(src.final.ring_table) · 각각 상위 k
# val(보라)은 임베딩에 함께 넣되 라벨을 쓰지 않는다(분포 이동 확인용).
# 실행: python "3_단계/4_피처/4-4_1119_휴대폰_피처·모델탐색/15_피처정리_tSNE/run.py"

# %% 설정
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.final import ring_table
from matplotlib import font_manager as _fm  # src.viz가 글꼴을 덮어쓰므로 src 임포트 뒤에 설정
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train()
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val"))
F = cached_table("ext", feature_table_ext, X); Fv = feature_table_ext(Xv)
def ringdf(Z):
    r, res, n = ring_table(Z); return pd.DataFrame({"ring24": r[:, 0], "ring_body": r[:, 1], "res83": res, "nimp": n})
RG, RGv = ringdf(X), ringdf(Xv)
MARK = [("nor", 116), ("nor", 59), ("nor", 55), ("nor", 106), ("nor", 25), ("abn", 22), ("abn", 14), ("abn", 17), ("abn", 2)]
mark = [i for i, n in enumerate(names) if any(n.startswith(p) and int(n.split("_")[-1]) == k for p, k in MARK)]
print("train", F.shape, "val", Fv.shape, "| 고장", int(y.sum()))

# %% 1) 변환 + 라벨 없는 정리
F = F.loc[:, F.nunique() > 3].copy(); Fv = Fv[F.columns]
qt = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(F.values)
Z, Zv = qt.transform(F.values), qt.transform(Fv.values)
C = np.abs(pd.DataFrame(Z).corr(method="spearman").fillna(0).values)
lk = linkage(squareform(1 - C, checks=False), "average"); cl = fcluster(lk, 0.15, "distance")
reps = [int(np.where(cl == c)[0][np.argmax(C[np.ix_(cl == c, cl == c)].mean(1))]) for c in np.unique(cl)]
print(f"ext {Z.shape[1]}열 → 상관 군집 {len(reps)}개 (대표만)")
cols = list(F.columns); R = pd.DataFrame(qt.transform(F.values)[:, reps], columns=[cols[i] for i in reps])
ZB, ZBv = Z[:, reps], Zv[:, reps]
rq = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(RG.values)
ZR, ZRv = rq.transform(RG.values), rq.transform(RGv.values)
SETS = {"A ext150 전체": (Z, Zv, cols), "B 대표만": (ZB, ZBv, [cols[i] for i in reps]),
        "C 대표 + 울림4": (np.hstack([ZB, ZR]), np.hstack([ZBv, ZRv]), [cols[i] for i in reps] + list(RG.columns))}

# %% 2) 폴드 안 순위 + k 선택 (로지스틱 PR-AUC / AUC)
def rank_auc(Ztr, ytr):
    return np.array([abs(roc_auc_score(ytr, Ztr[:, j]) - 0.5) for j in range(Ztr.shape[1])])
KS = [4, 6, 8, 12, 16, 24, 32, 48, 10 ** 6]
def cv_eval(Zs, n_rep=5):
    out = {k: [] for k in KS}
    for r in range(n_rep):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Zs, y):
            o = np.argsort(-rank_auc(Zs[tr], y[tr]))
            for k in KS:
                s = o[:min(k, Zs.shape[1])]; sc = StandardScaler().fit(Zs[tr][:, s])
                m = LogisticRegression(C=0.3, class_weight="balanced", max_iter=2000).fit(sc.transform(Zs[tr][:, s]), y[tr])
                p = m.decision_function(sc.transform(Zs[te][:, s]))
                out[k].append((average_precision_score(y[te], p), roc_auc_score(y[te], p)))
    return {k: (np.mean([a for a, _ in v]), np.std([a for a, _ in v]), np.mean([b for _, b in v])) for k, v in out.items()}
CV = {n: cv_eval(s[0]) for n, s in SETS.items()}
rows = [(n, "전체" if k >= 10 ** 5 else k, *v) for n, d in CV.items() for k, v in d.items()]
T = pd.DataFrame(rows, columns=["집합", "k", "PR-AUC", "sd", "AUC"]); T.round(3).to_csv(os.path.join(HERE, "cv_by_k.csv"), index=False, encoding="utf-8-sig")
print(T.round(3).to_string(index=False))
best = max(((n, k) for n, d in CV.items() for k in d), key=lambda t: (CV[t[0]][t[1]][0], -min(t[1], 1000)))
bn, bk = best
# 1-SE 규칙: 최고 PR-AUC − sd 이상이면서 가장 작은 k
top = CV[bn][bk]; cand = [k for k in KS if CV[bn][k][0] >= top[0] - top[1]]; bk1 = min(cand)
print("최고:", bn, bk, round(top[0], 3), "| 1-SE:", bn, bk1, round(CV[bn][bk1][0], 3))

# %% 3) 선택 집합(전체 150개로 순위, 설명·시각화용) + t-SNE
Zs_, Zsv_, cn = SETS[bn]; o = np.argsort(-rank_auc(Zs_, y)); sel = o[:min(bk1, Zs_.shape[1])]
Sel = pd.DataFrame({"피처": [cn[j] for j in sel], "단변량 |AUC−0.5|": rank_auc(Zs_, y)[sel].round(3)}); Sel.to_csv(os.path.join(HERE, "selected_features.csv"), index=False, encoding="utf-8-sig")
print("선택 피처 (집합 %s, k=%d):\n" % (bn, len(sel)), Sel.to_string(index=False))
def embed(A, seeds=(0, 1, 2, 3, 4), perp=20):
    best_ = None
    for s in seeds:
        t = TSNE(2, perplexity=perp, init="pca", random_state=s, learning_rate="auto"); E = t.fit_transform(A)
        if best_ is None or t.kl_divergence_ < best_[0]: best_ = (t.kl_divergence_, E)        # 라벨 없이 KL 최소 seed
    return best_[1]
def purity(E, lab, test_idx, ref_idx, k=5):
    nb = NearestNeighbors(n_neighbors=k).fit(E[ref_idx]).kneighbors(E[test_idx], return_distance=False)
    return lab[ref_idx][nb].mean()
def purity_loo(E, lab, k=5):
    f_ = np.where(lab == 1)[0]; nb = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E[f_], return_distance=False)[:, 1:]; return lab[nb].mean()
def honest_purity(Zs, selector, seed=0):
    P = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Zs, y):
        s = selector(Zs[tr], y[tr]); E = embed(Zs[:, s], seeds=(0,))
        f_te = te[y[te] == 1]; P.append(purity(E, y, f_te, tr))
    return float(np.mean(P))
sel_k = lambda k: (lambda Ztr, ytr: np.argsort(-rank_auc(Ztr, ytr))[:k])
views = {f"A ext150 전체 ({SETS['A ext150 전체'][0].shape[1]}열)": (SETS["A ext150 전체"][0], SETS["A ext150 전체"][1], lambda Ztr, ytr: np.arange(Ztr.shape[1])),
         f"B 상관 정리 ({ZB.shape[1]}열, 라벨 없음)": (ZB, ZBv, lambda Ztr, ytr: np.arange(Ztr.shape[1])),
         f"{bn} 상위 {len(sel)}개": (Zs_, Zsv_, sel_k(len(sel)))}
fig, ax = plt.subplots(1, 3, figsize=(21, 6.6)); NOR, ABN, PUR = "#2a78d6", "#eb6834", "#7B1FA2"
res_rows = []
for a, (title, (Ztr, Zva, selector)) in zip(ax, views.items()):
    is_sel = title.startswith(bn)
    cols_idx = sel if is_sel else np.arange(Ztr.shape[1])
    A = np.vstack([Ztr[:, cols_idx], Zva[:, cols_idx]]); E = embed(A); Et, Ev = E[:len(y)], E[len(y):]
    pin = purity_loo(Et, y); phon = honest_purity(Ztr, selector)
    a.scatter(*Et[y == 0].T, s=22, color=NOR, alpha=.55, lw=0, label="train 정상"); a.scatter(*Et[y == 1].T, s=34, color=ABN, alpha=.85, lw=0, label="train 고장")
    a.scatter(*Ev.T, s=60, color=PUR, edgecolor="white", lw=0.6, zorder=5, label="val 40 (라벨 없음)")
    for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Et[i], fontsize=7, color="k")
    a.set_title(f"{title}\n이웃 순도 in-sample {pin:.2f} · 폴드 밖(OOF) {phon:.2f}", fontsize=10); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=7, loc="best")
    res_rows.append((title, pin, phon))
base = y.mean(); fig.suptitle(f"t-SNE (perplexity 20, 라벨 없는 KL 최소 seed). 고장 비율 {base:.2f} = 무작위 수준의 순도", fontsize=11)
fig.savefig(os.path.join(FIG, "tsne_compare.png"), dpi=125, bbox_inches="tight"); plt.close(fig)
pd.DataFrame(res_rows, columns=["패널", "in-sample 순도", "OOF 순도"]).round(3).to_csv(os.path.join(HERE, "tsne_purity.csv"), index=False, encoding="utf-8-sig")
print(pd.DataFrame(res_rows, columns=["패널", "in-sample 순도", "OOF 순도"]).round(3).to_string(index=False))

# %% 4) k 곡선 그림
fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
for n, d in CV.items():
    xs = [min(k, Z.shape[1]) if k < 10 ** 5 else (SETS[n][0].shape[1]) for k in KS]
    ax[0].errorbar(xs, [d[k][0] for k in KS], yerr=[d[k][1] for k in KS], marker="o", capsize=3, label=n); ax[1].plot(xs, [d[k][2] for k in KS], marker="o", label=n)
for a, t in zip(ax, ("로지스틱 PR-AUC (폴드 안 순위·k, 5회×5겹)", "AUC")): a.set_xscale("log"); a.set_xlabel("선택한 피처 수 k"); a.set_title(t); a.legend(fontsize=8)
fig.savefig(os.path.join(FIG, "cv_curve.png"), dpi=120, bbox_inches="tight"); plt.close(fig)
print("저장:", FIG)

# %% 5) 피처 가중: 상위 k개 자르기보다 열마다 가중치를 주는 방식이 t-SNE에서 더 잘 갈리는가 (폴드 안에서만 가중치 계산)
from sklearn.ensemble import ExtraTreesClassifier
ZC, ZCv, cnC = SETS["C 대표 + 울림4"]
def w_uniform(Ztr, ytr): return np.ones(Ztr.shape[1])
def w_top(k): return lambda Ztr, ytr: (np.argsort(np.argsort(-rank_auc(Ztr, ytr))) < k).astype(float)
def w_auc_soft(Ztr, ytr): a = rank_auc(Ztr, ytr); w = np.clip(a - 0.10, 0, None); return w / (w.max() + 1e-12)
def w_auc_sq(Ztr, ytr): a = rank_auc(Ztr, ytr); return (a / (a.max() + 1e-12)) ** 2
def w_et(Ztr, ytr):
    m = ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, random_state=0, n_jobs=-1).fit(Ztr, ytr); w = np.sqrt(m.feature_importances_); return w / (w.max() + 1e-12)
def w_l1(Ztr, ytr):
    m = LogisticRegression(penalty="l1", solver="liblinear", C=0.15, class_weight="balanced", max_iter=2000).fit(Ztr, ytr); w = np.abs(m.coef_[0]); return w / (w.max() + 1e-12)
def w_combo(Ztr, ytr): return np.sqrt(w_auc_soft(Ztr, ytr) * w_et(Ztr, ytr))
WM = {"균등 (66열)": w_uniform, "AUC 상위 12": w_top(12), "AUC 소프트 가중": w_auc_soft, "AUC² 가중": w_auc_sq, "ExtraTrees 중요도": w_et, "L1 로지스틱": w_l1, "AUC×ET": w_combo}
def oof_purity(Zs, wf, seeds=(0, 1, 2), perp=20):
    P = []
    for sd in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(Zs, y):
            w = wf(Zs[tr], y[tr]); A = Zs * w
            E = TSNE(2, perplexity=perp, init="pca", random_state=0, learning_rate="auto").fit_transform(A[:, w > 0])
            f_te = te[y[te] == 1]; P.append(purity(E, y, f_te, tr))
    return float(np.mean(P)), float(np.std(P) / np.sqrt(len(P)))
PU = {n: oof_purity(ZC, f) for n, f in WM.items()}
Pt = pd.DataFrame([(n, *v) for n, v in PU.items()], columns=["가중 방식", "OOF 이웃 순도", "표준오차"]).sort_values("OOF 이웃 순도", ascending=False)
Pt.round(3).to_csv(os.path.join(HERE, "weighting_oof_purity.csv"), index=False, encoding="utf-8-sig"); print(Pt.round(3).to_string(index=False))
# 가중치 문턱 t와 perplexity 소격자 (같은 폴드에서 고르므로 OOF 순도는 약간 낙관적일 수 있음)
def w_soft(t):
    def f(Ztr, ytr):
        a = rank_auc(Ztr, ytr); w = np.clip(a - t, 0, None); return w / (w.max() + 1e-12)
    return f
GRID = {(t, pp): oof_purity(ZC, w_soft(t), perp=pp) for t in (0.10, 0.15, 0.20) for pp in (15, 20, 30)}
Gt = pd.DataFrame([(t, pp, *v) for (t, pp), v in GRID.items()], columns=["문턱 t", "perplexity", "OOF 이웃 순도", "표준오차"]).sort_values("OOF 이웃 순도", ascending=False)
Gt.round(3).to_csv(os.path.join(HERE, "grid_oof_purity.csv"), index=False, encoding="utf-8-sig"); print(Gt.round(3).to_string(index=False))
(bt, PERP) = (Gt.iloc[0]["문턱 t"], int(Gt.iloc[0]["perplexity"])); WM["AUC 소프트(최적 t=%.2f)" % bt] = w_soft(bt); PU["AUC 소프트(최적 t=%.2f)" % bt] = GRID[(bt, PERP)]
bw = "AUC 소프트(최적 t=%.2f)" % bt; print("최종 설정:", bw, "perplexity", PERP)

# %% 6) 최고 가중 방식으로 최종 t-SNE (전체 150개로 가중치 계산 = 시각화용, 제목에 OOF 순도 병기)
w = WM[bw](ZC, y); keep = w > 0; A = np.vstack([ZC, ZCv]) * w
E = embed(A[:, keep], perp=PERP); Et, Ev = E[:len(y)], E[len(y):]
pin = purity_loo(Et, y); phon = PU[bw][0]
W = pd.DataFrame({"피처": cnC, "가중치": w.round(3)}).sort_values("가중치", ascending=False); W.to_csv(os.path.join(HERE, "feature_weights.csv"), index=False, encoding="utf-8-sig")
print("가중치 상위 15:\n", W.head(15).to_string(index=False), "\n가중치>0 열:", int(keep.sum()), "/", len(cnC))
fig, ax = plt.subplots(1, 2, figsize=(17, 7.2))
a = ax[0]
a.scatter(*Et[y == 0].T, s=26, color=NOR, alpha=.55, lw=0, label="train 정상"); a.scatter(*Et[y == 1].T, s=42, color=ABN, alpha=.9, lw=0, label="train 고장")
a.scatter(*Ev.T, s=75, color=PUR, edgecolor="white", lw=0.7, zorder=5, label="val 40 (라벨 없음)")
for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Et[i], fontsize=8, color="k", xytext=(3, 3), textcoords="offset points")
a.set_title(f"t-SNE: {bw} 가중 (열 {int(keep.sum())}개, perplexity {PERP})\n이웃 순도 in-sample {pin:.2f} · 폴드 밖(OOF) {phon:.2f} (무작위 {y.mean():.2f})", fontsize=11); a.set_xticks([]); a.set_yticks([]); a.legend(fontsize=9)
a = ax[1]; top = W.head(15)[::-1]; a.barh(top["피처"], top["가중치"], color="#8E24AA"); a.set_title("가중치 상위 15 (전체 150개 기준)"); a.set_xlabel("가중치")
fig.savefig(os.path.join(FIG, "tsne_best.png"), dpi=125, bbox_inches="tight"); plt.close(fig)

# %% val이 train 어디에 놓이나: 이웃 train 파일의 고장 비율 (라벨 없는 val을 train 라벨로 이웃 조회)
nb = NearestNeighbors(n_neighbors=5).fit(Et).kneighbors(Ev, return_distance=False); vp = y[nb].mean(1)
print("val 40개의 train 이웃 5개 중 고장 비율 분포:", np.bincount((vp * 5).astype(int), minlength=6).tolist(), "(0/5 … 5/5개)")
pd.DataFrame({"file": nv, "train 이웃 고장 비율": vp}).to_csv(os.path.join(HERE, "val_neighbor_fault_ratio.csv"), index=False, encoding="utf-8-sig")
