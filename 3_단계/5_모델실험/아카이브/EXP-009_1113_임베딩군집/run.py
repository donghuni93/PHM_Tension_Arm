# %% [markdown]
# EXP-009 기본 피처 임베딩·군집 (t-SNE · UMAP · DBSCAN)
# - 가설: H1 (충격 → 첨도·고역 에너지↑), H3 (게인 차이 → 절대 크기보다 비율이 안정)
# - 바꾼 것: ext150 대신 해석 쉬운 피처 14개(RMS·첨도·고역통과 후 RMS·첨도·대역 에너지 비율 10개)만으로
#   라벨 없이 2차원에 펼치고(t-SNE·UMAP) 밀도 군집(DBSCAN)을 한다. 고장이 따로 모이는가, 놓친 고장은 어디 있는가
# - 라벨은 색칠·사후 집계에만 쓴다. 대역·차단 주파수·eps는 라벨을 보지 않고 미리 정한 규칙으로
#   (대역 = 1.25 kHz 균등 10개, 고역통과 = 1 kHz, eps = k-거리 곡선의 무릎)
# 실행: conda run -n qe-phm python 3_단계/5_모델비교/아카이브/EXP-009_1113_임베딩군집/run.py

# %% 설정
import os, sys
sys.path.insert(0, next(_p for _p in (os.path.abspath(os.path.join((os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else "."), *[".."] * _k)) for _k in range(8)) if os.path.isdir(os.path.join(_p, "src"))))
from src.exp import *
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from scipy.signal import butter, sosfiltfilt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

FS = 25000
HP_HZ = 1000           # 고역통과 차단: 구조 저주파 운동·드리프트를 빼고 충격·공진 대역만 남김 (라벨 보기 전 고정)
BAND_HZ = 1250         # 대역 폭: src/features.py의 필터 대역과 같은 1.25 kHz 균등 10개
MIN_SAMPLES = 5        # DBSCAN 최소 이웃 수 (파일 150개 기준 관례값)
C_NOR, C_ABN, C_GRAY = "#2a78d6", "#eb6834", "#b4b2a9"
C_CLU = ["#1baf7a", "#eda100", "#e87ba4", "#008300", "#2a78d6", "#eb6834"]   # 군집 색 (팔레트 3번부터)

ex = Experiment(__file__, "EXP-009", "기본 피처 임베딩·군집 (t-SNE·UMAP·DBSCAN)", ["H1", "H3"],
                member="이동훈", model="ExtraTrees (참고) · DBSCAN",
                changes="피처 14개(RMS·첨도·HP 1 kHz RMS·첨도·대역 에너지 비율 10)로 t-SNE·UMAP·DBSCAN")
X, y, names = load_train()
names = list(names)
g = np.arange(len(y))
MEAN_NEG = ["nor_097", "nor_101", "nor_120", "abn_016"]   # 평균 −1 g 파일 (2-1 조각)


# %% 피처 14개
def basic_features(x, fs=FS):
    x = np.asarray(x, float); x = x - x.mean()
    sos = butter(4, HP_HZ, btype="highpass", fs=fs, output="sos")
    xh = sosfiltfilt(sos, x)
    P = np.abs(np.fft.rfft(x)) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    d = {"rms": np.sqrt(np.mean(x ** 2)),
         "kurtosis": kurtosis(x, fisher=False),
         "hp_rms": np.sqrt(np.mean(xh ** 2)),
         "hp_kurtosis": kurtosis(xh, fisher=False)}
    for lo in np.arange(0, fs / 2, BAND_HZ):
        d[f"band_{lo / 1000:04.1f}k"] = P[(f >= lo) & (f < lo + BAND_HZ)].sum() / P.sum()
    return d


F = cached_table("exp009_basic", lambda X: pd.DataFrame([basic_features(x) for x in X]), X)
# 크기·비율은 치우침이 커서 log10, 첨도는 3 이상이라 log10. 그 뒤 표준화
Z = StandardScaler().fit_transform(np.log10(F.values.clip(1e-12)))
cols = list(F.columns)
print(F.describe().T.round(4).to_string())

# %% 그림 1: 놓친 파일 vs 정상 5–95% (PSD, 매 실험 기본)
fpsd, L = viz.log_psd(X)
ex.save_fig(viz.psd_band(fpsd, L, y, names, ex.focus), "psd_focus",
            "놓친 고장의 PSD가 정상 범위(파란 띠) 밖으로 나가는 주파수가 있는가")


# %% 그림 2: 피처 14개에서 놓친 고장의 위치 (정상 5–95% 띠)
def feature_strip(Z, cols):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rng = np.random.default_rng(0)
    for j in range(len(cols)):
        zn = Z[y == 0, j]
        lo, hi = np.percentile(zn, [5, 95])
        ax.fill_between([j - 0.4, j + 0.4], lo, hi, color=C_NOR, alpha=0.15, lw=0)
        ax.scatter(j - 0.15 + rng.uniform(-0.08, 0.08, (y == 0).sum()), zn, s=8, c=C_NOR, alpha=0.5, lw=0)
        ax.scatter(j + 0.15 + rng.uniform(-0.08, 0.08, (y == 1).sum()), Z[y == 1, j], s=10, c=C_ABN, alpha=0.7, lw=0)
    for k, nm in enumerate(ex.focus):
        i = names.index(nm)
        ax.plot(range(len(cols)), Z[i], marker="o", ms=5, lw=1, color=["#3b3a36", "#7a7974", "#a3a29c"][k],
                label=nm.replace("abn_", "고장 "))
    ax.set_xticks(range(len(cols)), cols, rotation=40, ha="right")
    ax.set_ylabel("표준화 값 (log10 후)")
    ax.set_title("피처 14개: 정상(파랑, 띠 = 5–95%) · 고장(주황) · 놓친 고장 3개(회색 선)")
    ax.grid(axis="y", color="#e6e5e0", lw=0.6); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right", ncol=3)
    fig.tight_layout()
    return fig


ex.save_fig(feature_strip(Z, cols), "feature_focus",
            "피처 14개에서 정상 띠 밖으로 나가는 고장이 어느 피처인가. 놓친 고장 002·022·029는 띠 안에 있는가")

# %% DBSCAN: eps = k-거리 곡선의 무릎 (라벨 안 씀)
kd = np.sort(NearestNeighbors(n_neighbors=MIN_SAMPLES).fit(Z).kneighbors(Z)[0][:, -1])
xs = np.arange(len(kd)); line = kd[0] + (kd[-1] - kd[0]) * xs / xs[-1]
knee = int(np.argmax(line - kd))          # 처음·끝을 잇는 직선에서 가장 아래로 떨어진 점
eps = float(kd[knee])
lab = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit_predict(Z)

fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(xs, kd, color=C_NOR, lw=2)
ax.axhline(eps, color="#7a7974", lw=1, ls="--"); ax.plot(knee, eps, "o", color="#3b3a36", ms=7)
ax.text(2, eps, f" eps = {eps:.2f} (무릎)", va="bottom", color="#3b3a36")
ax.set_xlabel("파일 (k-거리 오름차순)"); ax.set_ylabel(f"{MIN_SAMPLES}번째 이웃까지 거리")
ax.set_title("DBSCAN eps 고르기: k-거리 곡선"); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
ex.save_fig(fig, "dbscan_kdist", "k-거리가 급히 커지기 시작하는 무릎 = 밀집 영역의 경계. 그 거리를 eps로 쓴다")

# 군집 × 라벨 표, 노이즈(−1)를 '이상'으로 본 사후 집계 (중첩 CV 아님, 비교용 점수 아님)
tab = pd.crosstab(pd.Series(lab, name="군집(−1=노이즈)"), pd.Series(np.where(y == 1, "고장", "정상"), name="라벨"))
noise = lab == -1
tp, fp = int((noise & (y == 1)).sum()), int((noise & (y == 0)).sum())
fn = int((~noise & (y == 1)).sum())
f2_noise = 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0
focus_lab = {nm: int(lab[names.index(nm)]) for nm in ex.focus + MEAN_NEG}
txt = (f"eps {eps:.3f} · min_samples {MIN_SAMPLES} · 군집 {len(set(lab) - {-1})}개 · 노이즈 {noise.sum()}개\n\n"
       f"{tab.to_string()}\n\n"
       f"노이즈 = 고장으로 볼 때 (사후 집계, 전체 150개 그대로 · 중첩 CV 아님): "
       f"재현율 {tp / 30:.3f} · 오경보율 {fp / 120:.3f} · F2 {f2_noise:.3f}\n"
       f"노이즈 고장: {' '.join(n for n, l in zip(names, lab) if l == -1 and n.startswith('abn'))}\n"
       f"노이즈 정상: {' '.join(n for n, l in zip(names, lab) if l == -1 and n.startswith('nor'))}\n"
       f"관심 파일 군집: {focus_lab}")
ex.note("DBSCAN 결과", txt)
print(txt)


# %% 임베딩 그림 (왼쪽 = 라벨, 오른쪽 = DBSCAN 군집)
def embed_fig(E, title):
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax in axs:
        ax.set_xticks([]); ax.set_yticks([]); ax.spines[["top", "right"]].set_visible(False)
    a = axs[0]
    a.scatter(*E[y == 0].T, s=28, c=C_NOR, edgecolor="white", lw=0.8, label="정상 (120)")
    a.scatter(*E[y == 1].T, s=36, c=C_ABN, edgecolor="white", lw=0.8, label="고장 (30)")
    a.set_title(f"{title} — 라벨로 색칠 (임베딩에는 라벨 안 씀)")
    b = axs[1]
    for k in sorted(set(lab)):
        m = lab == k
        col = C_GRAY if k == -1 else C_CLU[k % len(C_CLU)]
        b.scatter(*E[m & (y == 0)].T, s=28, c=col, edgecolor="white", lw=0.8,
                  label="노이즈" if k == -1 else f"군집 {k} ({m.sum()})")
        b.scatter(*E[m & (y == 1)].T, s=46, c=col, marker="D", edgecolor="#3b3a36", lw=0.9)
    b.scatter([], [], s=46, marker="D", facecolor="white", edgecolor="#3b3a36", label="◆ = 고장")
    b.set_title(f"{title} — DBSCAN 군집 (14차원 공간에서 군집)")
    for ax in axs:
        for nm in ex.focus + MEAN_NEG:
            i = names.index(nm)
            ax.annotate(nm.replace("abn_", "a").replace("nor_", "n"), E[i], xytext=(4, 4),
                        textcoords="offset points", fontsize=8, color="#3b3a36")
        ax.legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    return fig


# %% t-SNE
E_tsne = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(Z)
ex.save_fig(embed_fig(E_tsne, "t-SNE (perplexity 30)"), "tsne",
            "고장이 정상과 다른 곳에 모이는가, 놓친 고장(a002·a022·a029)과 평균 −1 g 파일은 어디 있는가. "
            "t-SNE는 가까운 이웃만 보존 → 덩어리 사이 거리·크기는 해석하지 않는다")

# %% UMAP (umap-learn 설치 시)
try:
    import umap
    E_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(Z)
    ex.save_fig(embed_fig(E_umap, "UMAP (n_neighbors 15)"), "umap",
                "t-SNE와 같은 질문. UMAP은 전역 구조를 조금 더 보존 → 두 방법에서 공통으로 보이는 구조만 믿는다")
except ImportError:
    print("umap-learn 없음 → UMAP 건너뜀")

# %% 참고: 같은 14개 피처로 지도학습 판별력 (중첩 CV) — 임베딩 분리도와 비교용
ex.record("기본 14개 + ET", nested_cv(F.values, y, g, make_et()), names, main=True)
