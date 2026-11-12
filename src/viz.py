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
