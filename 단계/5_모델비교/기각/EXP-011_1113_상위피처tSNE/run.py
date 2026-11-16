# %% [markdown]
# EXP-011 상위 피처 몇 개로 t-SNE
# - 가설: H1·H2·H3 (도메인 피처 217개 중 단변량 판별력 상위가 어느 가설 그룹에서 나오는가)
# - 바꾼 것: EXP-010의 217개 전체 대신 단변량 AUC 상위 k개(5·10·20)만. 중복(순위상관 |r|>0.9)은 건너뜀
# - 주의: 상위 피처는 라벨로 고른다 → 같은 150개에 t-SNE를 그리면 '고르는 과정' 때문에 갈라져 보인다.
#   그래서 ① 라벨을 섞어 같은 절차를 돌린 기준선(순열 20회)과 비교하고
#   ② 점수는 피처 선택을 폴드 안에서 하는 중첩 CV(TopAUC)로만 낸다
# 실행: conda run -n qe-phm python 단계/5_모델비교/기각/EXP-011_1113_상위피처tSNE/run.py

# %% 설정
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__) if "__file__" in dir() else ".", "..", "..", "..", "..")))
from src.exp import *
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import QuantileTransformer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

C_NOR, C_ABN = "#2a78d6", "#eb6834"
KS = (5, 10, 20)
K_NB = 5                      # 이웃 순도의 이웃 수
N_PERM = 20                   # 라벨 섞기 횟수
R_MAX = 0.9                   # 이미 고른 피처와 순위상관이 이보다 크면 건너뜀
MEAN_NEG = ["nor_097", "nor_101", "nor_120", "abn_016"]

ex = Experiment(__file__, "EXP-011", "상위 피처 몇 개로 t-SNE", ["H1", "H2", "H3"],
                member="이동훈", model="ExtraTrees (폴드 안 상위 k 선택)",
                changes="도메인 피처 217개 중 단변량 AUC 상위 k개(5·10·20, 중복 제거)만으로 t-SNE·중첩 CV")
X, y, names = load_train()
names = list(names)
F = pd.concat([cached_table("ext", feature_table_ext, X), cached_table("phys", feature_table_phys, X)], axis=1)
F = F.loc[:, F.std() > 0]
cols = np.array(F.columns)
V = F.values
R_all = np.abs(np.corrcoef(np.apply_along_axis(rankdata, 0, V), rowvar=False))   # 순위상관 (라벨 안 씀)


def top_auc(V, y, k, R=None):
    """단변량 AUC(방향 무관) 상위부터, 이미 고른 것과 |순위상관| > R_MAX면 건너뛰며 k개. 열 번호 반환"""
    a = np.array([roc_auc_score(y, V[:, j]) for j in range(V.shape[1])])
    s = np.abs(a - 0.5)
    if R is None:
        R = np.abs(np.corrcoef(np.apply_along_axis(rankdata, 0, V), rowvar=False))
    sel = []
    for j in np.argsort(-s):
        if all(R[j, i] <= R_MAX for i in sel):
            sel.append(j)
        if len(sel) == k:
            break
    return np.array(sel), a


class TopAUC(BaseEstimator):
    """fit한 데이터(=폴드 학습 파일)만으로 상위 k 피처를 골라 ET 학습 → 선택 누수 없음"""
    def __init__(self, k): self.k = k
    def fit(self, X, y):
        self.cols_, _ = top_auc(X, y, self.k)
        self.m_ = make_et()().fit(X[:, self.cols_], y)
        return self
    def predict_proba(self, X): return self.m_.predict_proba(X[:, self.cols_])


def scale(Vs):
    return QuantileTransformer(n_quantiles=len(Vs), output_distribution="normal").fit_transform(Vs)


def purity(Z, yy):
    idx = NearestNeighbors(n_neighbors=K_NB + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    p = yy[idx].mean(1)
    return p[yy == 1].mean(), p


def scatter(ax, Y, title):
    ax.scatter(*Y[y == 0].T, s=22, c=C_NOR, edgecolor="white", lw=0.7, label="정상 (120)")
    ax.scatter(*Y[y == 1].T, s=30, c=C_ABN, edgecolor="white", lw=0.7, label="고장 (30)")
    for n in ex.focus:
        i = names.index(n)
        ax.scatter(*Y[i], s=90, facecolor="none", edgecolor="#3b3a36", lw=1.2)
        ax.annotate(n.replace("abn_", "a"), Y[i], xytext=(5, 4), textcoords="offset points", fontsize=8, color="#3b3a36")
    for n in MEAN_NEG:
        i = names.index(n)
        ax.annotate(n.replace("abn_", "a").replace("nor_", "n"), Y[i], xytext=(5, -9),
                    textcoords="offset points", fontsize=7, color="#7a7974")
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.spines[["top", "right"]].set_visible(False)


# %% 그림 1: 놓친 파일 vs 정상 5–95% (PSD, 매 실험 기본)
fpsd, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(fpsd, L, y, names, ex.focus), "psd_focus",
            "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")

# %% 상위 피처 (전체 150개로 고름 → 시각화·설명용. 점수는 아래 중첩 CV에서 폴드마다 다시 고름)
sel, auc = top_auc(V, y, max(KS), R_all)
top = pd.DataFrame({"순위": range(1, len(sel) + 1), "피처": cols[sel], "AUC": auc[sel].round(3),
                    "방향": np.where(auc[sel] > 0.5, "고장↑", "고장↓")})
ex.note("상위 20 피처 (단변량 AUC, 순위상관 0.9 초과 중복 제거)", top.to_string(index=False))
print(top.to_string(index=False))

# %% 그림 2: 상위 k개 t-SNE + 라벨 섞기 기준선
rng = np.random.default_rng(0)
rows = []
fig, axs = plt.subplots(1, len(KS), figsize=(16, 5.4))
for ax, k in zip(axs, KS):
    Z = scale(V[:, sel[:k]])
    p_real, p_file = purity(Z, y)
    null = []
    for _ in range(N_PERM):                        # 라벨을 섞어 같은 절차(선택 → 순도)를 반복
        yp = rng.permutation(y)
        sp, _ = top_auc(V, yp, k, R_all)
        null.append(purity(scale(V[:, sp]), yp)[0])
    null = np.array(null)
    rows.append(dict(k=k, 순도=round(p_real, 3), 섞은라벨_평균=round(null.mean(), 3), 섞은라벨_최대=round(null.max(), 3),
                     **{n.replace("abn_", "a"): round(p_file[names.index(n)], 2) for n in ex.focus}))
    Y = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
    scatter(ax, Y, f"상위 {k}개\n고장 이웃 순도 {p_real:.2f} (라벨 섞으면 {null.mean():.2f}, 무작위 0.19)")
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle("단변량 AUC 상위 k 피처 t-SNE (라벨로 고른 피처 → 순도는 '섞은 라벨' 값과 비교)", fontsize=12)
fig.tight_layout()
ex.save_fig(fig, "tsne_topk", "상위 피처 몇 개만 쓰면 고장이 모이는가. 그 정도가 라벨을 섞어도 생기는 정도보다 큰가")
T = pd.DataFrame(rows)
ex.note(f"이웃 순도 (고장 파일의 {K_NB} 이웃 중 고장 비율, 라벨 섞기 {N_PERM}회)", T.to_string(index=False))
print(T.to_string(index=False))

# %% 그림 3: 상위 10개에서 놓친 고장의 위치 (정상 5–95% 띠)
k = 10
Zs = scale(V[:, sel[:k]])
fig, ax = plt.subplots(figsize=(11, 4.6))
jit = np.random.default_rng(1)
for j in range(k):
    zn = Zs[y == 0, j]
    lo, hi = np.percentile(zn, [5, 95])
    ax.fill_between([j - 0.4, j + 0.4], lo, hi, color=C_NOR, alpha=0.15, lw=0)
    ax.scatter(j - 0.15 + jit.uniform(-0.08, 0.08, (y == 0).sum()), zn, s=8, c=C_NOR, alpha=0.5, lw=0)
    ax.scatter(j + 0.15 + jit.uniform(-0.08, 0.08, (y == 1).sum()), Zs[y == 1, j], s=10, c=C_ABN, alpha=0.7, lw=0)
for c, n in zip(["#3b3a36", "#7a7974", "#a3a29c"], ex.focus):
    ax.plot(range(k), Zs[names.index(n)], marker="o", ms=5, lw=1, color=c, label=n.replace("abn_", "고장 "))
ax.set_xticks(range(k), [f"{c}\n({'↑' if a > .5 else '↓'} {max(a, 1 - a):.2f})" for c, a in zip(cols[sel[:k]], auc[sel[:k]])],
              rotation=35, ha="right", fontsize=8)
ax.set_ylabel("분위수 정규화 값")
ax.set_title("상위 10 피처: 정상(파랑, 띠 = 5–95%) · 고장(주황) · 놓친 고장 3개(회색 선). 괄호 = 고장 방향·AUC")
ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, ncol=3, loc="upper right")
fig.tight_layout()
ex.save_fig(fig, "top10_focus", "상위 피처에서 놓친 고장 002·022·029가 고장 쪽으로 가는가, 정상 띠 안에 남는가")

# %% 중첩 CV: 상위 k 선택을 폴드 학습 파일 안에서 (정직한 점수)
g = np.arange(len(y))
for k in KS:
    ex.record(f"상위 {k}개 (폴드 안 선택) + ET", nested_cv(V, y, g, lambda k=k: TopAUC(k)), names, main=(k == 10))
