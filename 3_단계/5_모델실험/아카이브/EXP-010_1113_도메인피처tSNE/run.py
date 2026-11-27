# %% [markdown]
# EXP-010 도메인 피처 전체 t-SNE
# - 가설: H1 (대역 충격·포락선·시간 변동 = ext150), H2 (공진 날카로움), H3 (스펙트럼 모양), H4 (하중 반전 충격)
# - 바꾼 것: EXP-009의 기본 14개 대신 지금까지 만든 도메인 피처 전체(ext150 + phys)로 t-SNE.
#   전체 + 가설 그룹별로 따로 펼쳐, 어느 그룹에서 고장이 정상과 갈라지는지, 놓친 고장 002·022·029가 어디 있는지 본다
# - 라벨은 색칠과 사후 집계(이웃 순도)에만 쓴다. 스케일링은 라벨 없이 피처별 분위수 → 정규분포 변환
# 실행: conda run -n qe-phm python 3_단계/5_모델비교/아카이브/EXP-010_1113_도메인피처tSNE/run.py

# %% 설정
import os, sys
sys.path.insert(0, next(_p for _p in (os.path.abspath(os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."), *[".."] * _k)) for _k in range(8)) if os.path.isdir(os.path.join(_p, "src"))))
from src.exp import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import QuantileTransformer
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

C_NOR, C_ABN = "#2a78d6", "#eb6834"
K = 5                                                     # 이웃 순도에 쓰는 이웃 수
MEAN_NEG = ["nor_097", "nor_101", "nor_120", "abn_016"]   # 평균 −1 g 파일 (2-1 조각)

ex = Experiment(__file__, "EXP-010", "도메인 피처 전체 t-SNE", ["H1", "H2", "H3", "H4"],
                member="이동훈", model="— (비지도 시각화)",
                changes="도메인 피처 전체(ext150 + phys H2·H3·H4)와 가설 그룹별 t-SNE")
X, y, names = load_train()
names = list(names)
E = cached_table("ext", feature_table_ext, X)
P = cached_table("phys", feature_table_phys, X)
F = pd.concat([E, P], axis=1)
F = F.loc[:, F.std() > 0]

groups = {
    f"전체 ({F.shape[1]})": list(F.columns),
    "ext150: 시간·주파수·대역 충격 (H1 중심)": list(E.columns),
    "H2 공진 날카로움": [c for c in F if c.startswith("H2_")],
    "H3 스펙트럼 모양": [c for c in F if c.startswith("H3_")],
    "H4 반전 충격 + 충격 횟수": [c for c in F if c.startswith(("H4_", "H1_"))],
}
groups = {(k if k.startswith("전체") else f"{k} ({len(v)})"): v for k, v in groups.items()}
print({k: len(v) for k, v in groups.items()})


def scale(cols):
    """피처별 분위수 → 정규분포 (치우침·이상값·단위 차이를 라벨 없이 정리)"""
    return QuantileTransformer(n_quantiles=len(y), output_distribution="normal").fit_transform(F[cols].values)


def purity(Z):
    """고장 파일의 K 이웃 중 고장 비율 (자기 자신 제외). 무작위 기준 = 29/149 ≈ 0.19"""
    idx = NearestNeighbors(n_neighbors=K + 1).fit(Z).kneighbors(Z)[1][:, 1:]
    p = y[idx].mean(1)
    return p[y == 1].mean(), {n: p[names.index(n)] for n in ex.focus}


def tsne(Z, perp=30, seed=0):
    return TSNE(2, perplexity=perp, init="pca", random_state=seed).fit_transform(Z)


def scatter(ax, Y, title, label_files=True):
    ax.scatter(*Y[y == 0].T, s=22, c=C_NOR, edgecolor="white", lw=0.7, label="정상 (120)")
    ax.scatter(*Y[y == 1].T, s=30, c=C_ABN, edgecolor="white", lw=0.7, label="고장 (30)")
    if label_files:
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

# %% 그림 2: 전체 + 가설 그룹별 t-SNE (작은 여러 개)
rows, emb = [], {}
fig, axs = plt.subplots(2, 3, figsize=(15, 9.6))
for ax, (g, cols) in zip(axs.flat, groups.items()):
    Z = scale(cols)
    Y = tsne(Z)
    emb[g] = Y
    pz, pf = purity(Z)
    py, _ = purity(Y)
    rows.append(dict(그룹=g, 피처수=len(cols), 순도_원공간=round(pz, 3), 순도_tSNE=round(py, 3),
                     **{n.replace("abn_", "a"): round(v, 2) for n, v in pf.items()}))
    scatter(ax, Y, f"{g}\n고장 이웃 순도 {pz:.2f} (무작위 0.19)")
axs[0, 0].legend(frameon=False, fontsize=8, loc="best")
axs.flat[-1].axis("off")
axs.flat[-1].text(0, 0.5, "검은 원 = 자주 놓친 고장 002·022·029\n회색 글자 = 평균 -1 g 파일\n\n"
                  "t-SNE는 가까운 이웃만 보존한다.\n덩어리 사이 거리·크기는 해석하지 않는다.\n\n"
                  f"이웃 순도 = 고장 파일의 {K} 이웃 중 고장 비율\n(t-SNE 전 원래 피처 공간에서 계산)",
                  fontsize=10, color="#3b3a36", va="center")
fig.suptitle("도메인 피처 t-SNE: 전체와 가설 그룹별 (perplexity 30, 라벨은 색칠에만)", fontsize=12)
fig.tight_layout()
ex.save_fig(fig, "tsne_groups",
            "어느 피처 그룹에서 고장이 정상과 따로 모이는가. 놓친 고장 3개는 그룹마다 정상 속에 있는가")
T = pd.DataFrame(rows)
ex.note("이웃 순도 (고장 파일의 5 이웃 중 고장 비율, 무작위 0.19)", T.to_string(index=False))
print(T.to_string(index=False))

# %% 그림 3: 전체 피처 t-SNE의 안정성 (perplexity · seed)
Zall = scale(groups[list(groups)[0]])
fig, axs = plt.subplots(1, 4, figsize=(18, 4.8))
for ax, (perp, seed) in zip(axs, [(10, 0), (30, 0), (50, 0), (30, 1)]):
    scatter(ax, tsne(Zall, perp, seed), f"perplexity {perp} · seed {seed}")
axs[0].legend(frameon=False, fontsize=8)
fig.suptitle(f"전체 {Zall.shape[1]}개 t-SNE 설정 바꿔 보기: 설정과 무관하게 남는 구조만 믿는다", fontsize=12)
fig.tight_layout()
ex.save_fig(fig, "tsne_all_stability", "perplexity·seed를 바꿔도 고장 무리·놓친 고장 위치가 그대로인가")

# %% 그림 4: 놓친 고장의 그룹별 극단 피처 (정상 대비 백분위 상·하위 5개씩)
nor = F[y == 0]
pct = pd.DataFrame({n: [(nor[c] < F[c].iloc[names.index(n)]).mean() * 100 for c in F] for n in ex.focus}, index=F.columns)
ext_cols = list(dict.fromkeys(sum([list(pct[n].nsmallest(4).index) + list(pct[n].nlargest(4).index) for n in ex.focus], [])))
ex.save_fig(viz.feature_percentile(F, y, names, ex.focus, ext_cols), "focus_extreme",
            "놓친 고장 3개가 정상 분포 끝(0·100 근처)에 걸리는 피처가 무엇인가 (파일별 상·하위 4개)")
