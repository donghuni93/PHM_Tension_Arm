"""공용 그림 함수. 모두 Figure를 돌려준다 → Experiment.save_fig(fig, 이름)으로 figs/에 저장

- psd_band: 정상 5–95% 범위 + 정상·고장 중앙값 + 지정 파일 (놓친 파일 점검의 기본 그림)
- psd_auc: PSD 주파수 빈별 AUC (어느 주파수에서 정상·고장이 갈리나)
- waves: 파형 비교
- feature_percentile: 지정 파일의 피처값이 정상 분포에서 몇 백분위인가
- feature_auc_bar: 단변량 AUC 상위 피처
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import welch

FS = 25000
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


def log_psd(X, nperseg=8192, fs=FS):
    """평균 제거 후 Welch log10 PSD. 반환 f, L (파일 수 × 빈 수)"""
    f, P = welch(X - X.mean(1, keepdims=True), fs=fs, nperseg=nperseg)
    return f, np.log10(P)


def psd_band(f, L, y, names, focus, ranges=((0, 12500), (0, 300), (5000, 12500)), title=""):
    """정상 5–95% 범위 위에 지정 파일 PSD를 겹친다. ranges마다 한 행"""
    nor, ab = L[y == 0], L[y == 1]
    fig, axs = plt.subplots(len(ranges), 1, figsize=(13, 4.2 * len(ranges)), squeeze=False)
    for a, (lo, hi) in zip(axs[:, 0], ranges):
        m = (f >= lo) & (f <= hi)
        a.fill_between(f[m], np.percentile(nor, 5, 0)[m], np.percentile(nor, 95, 0)[m],
                       color="C0", alpha=.25, label="정상 5–95%")
        a.plot(f[m], np.median(nor, 0)[m], "C0", lw=1.5, label="정상 중앙값")
        a.plot(f[m], np.median(ab, 0)[m], "C3", lw=1.5, label="고장 중앙값")
        for i, n in enumerate(focus):
            a.plot(f[m], L[names.index(n)][m], lw=.8, color=f"C{i + 4}", label=n)
        a.set_xlim(lo, hi); a.set_xlabel("Hz"); a.set_ylabel("log10 PSD")
        a.legend(fontsize=8, ncol=4); a.grid(alpha=.3)
    axs[0, 0].set_title(title or "Welch PSD: 정상 범위 vs 지정 파일")
    fig.tight_layout()
    return fig


def psd_auc(f, L, y):
    """주파수 빈별 AUC. 반환 (fig, auc 배열)"""
    from sklearn.metrics import roc_auc_score
    auc = np.array([roc_auc_score(y, L[:, k]) for k in range(L.shape[1])])
    fig, a = plt.subplots(figsize=(13, 3.5))
    a.plot(f, auc, lw=.8); a.axhline(.5, color="k", lw=.5)
    a.set_title("PSD 빈별 AUC (>0.5 고장에서 큼, <0.5 고장에서 작음)"); a.set_xlabel("Hz"); a.grid(alpha=.3)
    fig.tight_layout()
    return fig, auc


def waves(X, names, show, fs=FS, t_range=None):
    """파형 비교. t_range=(시작초, 끝초)로 확대"""
    Xc = X - X.mean(1, keepdims=True)
    t = np.arange(X.shape[1]) / fs
    m = slice(None) if t_range is None else (t >= t_range[0]) & (t <= t_range[1])
    fig, axs = plt.subplots(len(show), 1, figsize=(13, 1.8 * len(show)), sharex=True, sharey=True, squeeze=False)
    for a, n in zip(axs[:, 0], show):
        a.plot(t[m], Xc[names.index(n)][m], lw=.4, color="C3" if n.startswith("abn") else "C0")
        a.set_title(n, fontsize=9, loc="left"); a.grid(alpha=.3)
    axs[-1, 0].set_xlabel("초")
    fig.tight_layout()
    return fig


def feature_percentile(F, y, names, focus, cols):
    """지정 파일의 피처값이 정상 분포에서 몇 백분위인지 (0·100이면 정상 밖)"""
    nor = F[y == 0]
    P = np.array([[(nor[c] < F[c].iloc[names.index(n)]).mean() * 100 for c in cols] for n in focus])
    fig, a = plt.subplots(figsize=(max(6, .4 * len(cols)), 3 + .6 * len(focus)))
    im = a.imshow(P, cmap="RdBu_r", vmin=0, vmax=100, aspect="auto")
    a.set_xticks(range(len(cols))); a.set_xticklabels(cols, rotation=90, fontsize=7)
    a.set_yticks(range(len(focus))); a.set_yticklabels(focus)
    for i in range(len(focus)):
        for j in range(len(cols)):
            a.text(j, i, f"{P[i, j]:.0f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=a, label="정상 대비 백분위")
    a.set_title("지정 파일의 정상 대비 백분위 (0·100 = 정상 범위 밖)")
    fig.tight_layout()
    return fig


def feature_auc_bar(A, top=25):
    """A: feat·auc·dir 열을 가진 DataFrame (auc 내림차순)"""
    A = A.head(top).iloc[::-1]
    fig, a = plt.subplots(figsize=(8, .28 * len(A) + 1))
    a.barh(A.feat, A.auc, color=["C3" if d == "+" else "C0" for d in A.dir])
    a.axvline(.5, color="k", lw=.5); a.set_xlim(.4, 1); a.set_xlabel("단변량 AUC")
    a.set_title("피처 AUC 상위 (빨강: 고장에서 큼, 파랑: 고장에서 작음)", fontsize=10)
    a.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def tsne(views, y, names, mark=(), score=None, perplexity=30, seed=0, k=5, scale=True):
    """t-SNE 기본 그림 (매 실험). views = {"패널 제목": (파일 수, 차원) 행렬}. scale=True면 열마다 표준화한 뒤 임베딩
    (블록 가중 등 직접 맞춘 행렬은 scale=False)
    - 1행: 정상 파랑 · 고장 주황. mark에 든 파일은 번호 표시 (예: 놓친 고장·오경보 정상)
    - score를 주면 2행: 같은 배치를 이상 점수로 칠함 (고장은 주황 테두리)
    - 제목의 '이웃 순도' = 2차원 배치에서 고장 점의 가까운 이웃 k개 중 고장 비율 (무작위면 고장 비율과 같음)
    반환 (fig, {패널 제목: 2차원 좌표})"""
    from sklearn.manifold import TSNE
    from sklearn.neighbors import NearestNeighbors
    NOR, ABN = "#2a78d6", "#eb6834"
    y = np.asarray(y)
    rows = 1 if score is None else 2
    fig, axs = plt.subplots(rows, len(views), figsize=(6 * len(views), 5.6 * rows), squeeze=False)
    emb = {}
    for j, (title, A) in enumerate(views.items()):
        A = np.asarray(A, dtype=float)
        if scale:
            A = (A - A.mean(0)) / (A.std(0) + 1e-12)
        E = TSNE(2, perplexity=perplexity, init="pca", random_state=seed).fit_transform(A)
        emb[title] = E
        nb = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E[y == 1], return_distance=False)[:, 1:]
        purity = y[nb].mean()
        for r in range(rows):
            a = axs[r, j]
            if r == 0:
                a.scatter(*E[y == 0].T, s=18, color=NOR, alpha=.55, lw=0, label="정상")
                a.scatter(*E[y == 1].T, s=28, color=ABN, lw=0, label="고장")
                a.set_title(f"{title}\n고장 이웃 순도 {purity:.2f} (무작위 {y.mean():.2f})", fontsize=10)
            else:
                sc = a.scatter(*E.T, c=score, cmap="viridis", s=20, lw=0)
                a.scatter(*E[y == 1].T, s=55, facecolor="none", edgecolor=ABN, lw=1.1, label="고장")
                fig.colorbar(sc, ax=a, shrink=.8, label="이상 점수")
                a.set_title(f"{title} · 색 = 이상 점수", fontsize=10)
            for n in mark:
                i = names.index(n)
                a.annotate(n.replace("abn_", "A").replace("nor_", "N"), E[i], xytext=(3, 3), textcoords="offset points",
                           fontsize=7, color=ABN if y[i] else NOR)
            a.set_xticks([]); a.set_yticks([])
        axs[0, j].legend(fontsize=8, loc="best")
    fig.suptitle(f"t-SNE (perplexity {perplexity}, seed {seed}). A/N 번호 = 표시한 파일", fontsize=11)
    fig.tight_layout()
    return fig, emb
