# %% [markdown]
# 15-2. 지도 임베딩 + 새 물리 피처로 더 잘 가르기 (train 150개만 사용, val 안 씀)
# 후보 피처: C = 대표 62열 + 울림 4열(66열) · D = C + 오늘 만든 파일 단위 물리 피처 10개(정렬 평균 울림 f·ζ·피크/RMS, 스펙트럼 첨도 4대역, PSD 기울기 2, log RMS; 모두 라벨 없이 계산)
# 임베딩 방식(모두 폴드 안에서 학습, 시험 점은 변환만):
#   S0 균등 t-SNE · S1 AUC 소프트 가중 t-SNE(15의 최고) · S2 NCA(지도 거리 학습, 6차원) → t-SNE · S3 ExtraTrees 근접도(1−proximity) → t-SNE
# 평가: 시험 고장의 2차원 이웃 5개(학습 파일만) 중 고장 비율(OOF 순도, 무작위 0.20), 3회 × 5겹. 같은 폴드로 방식을 고르므로 약간 낙관적.
# 실행: python "3_단계/4_피처/4-4_1119_휴대폰_피처·모델탐색/15_피처정리_tSNE/run2.py"

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
from matplotlib import font_manager as _fm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, stft, welch
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors, NeighborhoodComponentsAnalysis
from sklearn.ensemble import ExtraTreesClassifier
from src.exp import load_train, cached_table, feature_table_ext
from src.final import ring_table
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); FS = 25000
F = cached_table("ext", feature_table_ext, X); F = F.loc[:, F.nunique() > 3]
r, res, n = ring_table(X); RG = pd.DataFrame({"ring24": r[:, 0], "ring_body": r[:, 1], "res83": res, "nimp": n})
MARK = [("nor", 116), ("nor", 59), ("nor", 55), ("nor", 106), ("nor", 25), ("abn", 22), ("abn", 14), ("abn", 17), ("abn", 2)]
mark = [i for i, nm in enumerate(names) if any(nm.startswith(p) and int(nm.split("_")[-1]) == k for p, k in MARK)]

# %% 파일 단위 물리 피처 10개 (라벨 안 씀; 9번 · 13번 · 14번 분석에서 가져옴)
SOS200 = butter(4, 200, btype="highpass", fs=FS, output="sos"); SOS1K = butter(4, 1000, btype="highpass", fs=FS, output="sos"); SOSLP = butter(4, 500, btype="lowpass", fs=FS, output="sos")
PRE, POST, K, MARGIN = 50, 500, 10, int(0.02 * FS)
def pick(x):
    env = sosfiltfilt(SOSLP, np.abs(hilbert(sosfiltfilt(SOS1K, x)))); pk, _ = find_peaks(env, distance=125)
    pk = pk[(pk >= MARGIN + PRE) & (pk < len(x) - MARGIN - POST)]; top = pk[np.argsort(env[pk])[::-1][:K]]
    return np.array(sorted({p - 25 + int(np.argmax(np.abs(x[p - 25:p + 26]))) for p in top}))
def ring_mean(x):
    tr = pick(x); segs = [x[q - PRE:q + POST] * np.sign(x[q]) for q in tr if q - PRE >= 0 and q + POST <= len(x)]
    if not segs: return np.nan, np.nan, np.nan
    m = np.mean(segs, 0); seg = m[PRE:PRE + 375] * np.hanning(750)[375:]; Fq = np.abs(np.fft.rfft(seg, 8192)); fr = np.fft.rfftfreq(8192, 1 / FS)
    mk = (fr > 800) & (fr < 12000); f0 = fr[mk][np.argmax(Fq[mk])]; a = np.abs(hilbert(m))[PRE + 25:PRE + 200]; sl = np.polyfit(np.arange(len(a)) / FS, np.log(a + 1e-12), 1)[0]
    return f0, (-sl / (2 * np.pi * f0) if sl < 0 else np.nan), abs(m[PRE]) / x.std()
Xc = X - X.mean(1, keepdims=True); HP = sosfiltfilt(SOS200, Xc, axis=-1)
RM = np.array([ring_mean(HP[i]) for i in range(len(y))])
fz, _, Zs = stft(Xc, fs=FS, nperseg=512, noverlap=256, boundary=None, padded=False, axis=-1); P2 = np.abs(Zs) ** 2; SK = (P2 ** 2).mean(-1) / (P2.mean(-1) ** 2 + 1e-30) - 2
fw, Pw = welch(Xc, fs=FS, nperseg=2048, axis=-1)
def slope(lo, hi):
    m = (fw >= lo) & (fw < hi); lf = np.log10(fw[m]); return np.array([np.polyfit(lf, np.log10(Pw[i, m] + 1e-30), 1)[0] for i in range(len(y))])
EX = pd.DataFrame({"ring_f": RM[:, 0], "ring_zeta": np.log10(RM[:, 1] + 1e-4), "ring_peak_rms": RM[:, 2],
                   **{f"SK_{a}-{b}k": np.log10(SK[:, (fz >= a * 1000) & (fz <= b * 1000)].mean(1) + 1) for a, b in ((2, 4), (5.5, 7.75), (8, 8.7), (9, 10))},
                   "slope_lo": slope(200, 2000), "slope_hi": slope(2000, 12000), "log_rms": np.log10(Xc.std(1))}).fillna(0)

# %% 변환 · 상관 정리
qt = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0)
Z = qt.fit_transform(F.values); Cm = np.abs(pd.DataFrame(Z).corr(method="spearman").fillna(0).values)
cl = fcluster(linkage(squareform(1 - Cm, checks=False), "average"), 0.15, "distance")
reps = [int(np.where(cl == c)[0][np.argmax(Cm[np.ix_(cl == c, cl == c)].mean(1))]) for c in np.unique(cl)]
colsB = [F.columns[i] for i in reps]; ZB = Z[:, reps]
ZR = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit_transform(RG.values)
ZE = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit_transform(EX.values)
SETS = {"C 대표+울림4 (66)": (np.hstack([ZB, ZR]), colsB + list(RG.columns)), "D C+물리10 (76)": (np.hstack([ZB, ZR, ZE]), colsB + list(RG.columns) + list(EX.columns))}

# %% 임베딩 방식
def rank_auc(Ztr, ytr): return np.array([abs(roc_auc_score(ytr, Ztr[:, j]) - 0.5) for j in range(Ztr.shape[1])])
def tsne(A, perp=15, seed=0, pre=False):
    return TSNE(2, perplexity=perp, init="random" if pre else "pca", random_state=seed, learning_rate="auto", metric="precomputed" if pre else "euclidean").fit_transform(A)
def emb_S0(Ztr, ytr, Zall): return tsne(Zall)
def emb_S1(Ztr, ytr, Zall):
    a = rank_auc(Ztr, ytr); w = np.clip(a - 0.10, 0, None); w /= w.max() + 1e-12; return tsne((Zall * w)[:, w > 0])
def emb_S2(Ztr, ytr, Zall):
    sc = StandardScaler().fit(Ztr); nca = NeighborhoodComponentsAnalysis(n_components=6, random_state=0, max_iter=60).fit(sc.transform(Ztr), ytr); return tsne(nca.transform(sc.transform(Zall)))
def emb_S3(Ztr, ytr, Zall):
    et = ExtraTreesClassifier(500, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=-1).fit(Ztr, ytr)
    L = et.apply(Zall); Dm = 1 - np.mean([(L[:, t][:, None] == L[:, t][None, :]) for t in range(L.shape[1])], axis=0); np.fill_diagonal(Dm, 0); return tsne(Dm, pre=True)
METHODS = {"S0 균등 t-SNE": emb_S0, "S1 AUC 가중": emb_S1, "S2 NCA→t-SNE": emb_S2, "S3 ET 근접도": emb_S3}
def purity(E, test_idx, ref_idx, k=5):
    nb = NearestNeighbors(n_neighbors=k).fit(E[ref_idx]).kneighbors(E[test_idx], return_distance=False); return y[ref_idx][nb].mean()
def oof(setZ, fn, seeds=(0, 1, 2)):
    P = []
    for sd in seeds:
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=sd).split(setZ, y):
            E = fn(setZ[tr], y[tr], setZ); P.append(purity(E, te[y[te] == 1], tr))
    return float(np.mean(P)), float(np.std(P) / np.sqrt(len(P)))
rows = []
for sn, (Zs_, cn) in SETS.items():
    for mn, fn in METHODS.items():
        m_, se = oof(Zs_, fn); rows.append((sn, mn, m_, se)); print(f"{sn:20s} {mn:16s} OOF 순도 {m_:.3f} ± {se:.3f}", flush=True)
R = pd.DataFrame(rows, columns=["피처 집합", "방식", "OOF 이웃 순도", "표준오차"]).sort_values("OOF 이웃 순도", ascending=False)
R.round(3).to_csv(os.path.join(HERE, "supervised_oof_purity.csv"), index=False, encoding="utf-8-sig"); print(R.round(3).to_string(index=False))

# %% 최종: 최고 조합으로 train 150개 전체에서 임베딩 (지도 방식은 in-sample이라 제목에 OOF 순도 병기)
b = R.iloc[0]; Zs_, cn = SETS[b["피처 집합"]]; fn = METHODS[b["방식"]]
E = fn(Zs_, y, Zs_); nb = NearestNeighbors(n_neighbors=6).fit(E).kneighbors(E[y == 1], return_distance=False)[:, 1:]; pin = y[nb].mean()
fig, ax = plt.subplots(1, 2, figsize=(17, 7.2)); a = ax[0]; NOR, ABN = "#2a78d6", "#eb6834"
a.scatter(*E[y == 0].T, s=26, color=NOR, alpha=.6, lw=0, label="정상"); a.scatter(*E[y == 1].T, s=46, color=ABN, alpha=.9, lw=0, label="고장")
for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), E[i], fontsize=8, xytext=(3, 3), textcoords="offset points")
a.set_title(f"{b['피처 집합']} · {b['방식']}\n이웃 순도 in-sample {pin:.2f} · 폴드 밖(OOF) {b['OOF 이웃 순도']:.2f} (무작위 0.20)", fontsize=11); a.set_xticks([]); a.set_yticks([]); a.legend()
a = ax[1]; Rp = R.copy(); Rp["라벨"] = Rp["피처 집합"].str[0] + " " + Rp["방식"].str[:2]; Rp = Rp.sort_values("OOF 이웃 순도"); a.barh(Rp["라벨"], Rp["OOF 이웃 순도"], xerr=Rp["표준오차"], color="#8E24AA"); a.axvline(0.2, color="k", ls=":"); a.set_xlabel("OOF 이웃 순도"); a.set_title("방식 비교 (3회 × 5겹)")
fig.savefig(os.path.join(FIG, "tsne_supervised.png"), dpi=125, bbox_inches="tight"); plt.close(fig); print("최고:", dict(b), "| 저장", FIG)
