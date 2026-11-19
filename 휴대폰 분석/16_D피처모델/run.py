# %% [markdown]
# 16. D 피처(76열)만으로 만든 모델: 중첩 CV 평가 + 경계 plot + val 찍기
# D 피처 = ext150을 상관 군집(|ρ| ≥ 0.85, 라벨 없음)으로 줄인 대표 62열 + 울림 4열(ring24, ring_body, res83, nimp) + 파일 단위 물리 10개
#   (정렬 평균 울림 f·ζ·피크/RMS, 스펙트럼 첨도 4대역, PSD 기울기 2, log RMS). 변환·군집은 train 150개로만 fit(라벨 안 씀), val은 변환만.
# 모델: M1 LR(표준화, C=0.3) · M2 ExtraTrees · M3 스택 = 두 전문가의 안쪽 OOF 점수를 로지스틱으로 결합 (기존 AE+ET 로지스틱과 같은 구조)
# 평가: src.evaluate.nested_cv (seed 5 × 바깥 5 × 안쪽 4, 임계값은 안쪽 OOF의 F₂ 최대) — 기존 EXP 점수(EXP-014 0.857, EXP-025 0.942)와 같은 프로토콜
# 실행: python "휴대폰 분석/16_D피처모델/run.py"

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
from sklearn.base import BaseEstimator
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import fbeta_score
from src.exp import load_train, cached_table, feature_table_ext
from src.data import load_folder
from src.final import ring_table
from src.evaluate import nested_cv, best_f2_threshold
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic", "WenQuanYi Zen Hei"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
X, y, names = load_train(); Xv, nv = load_folder(os.path.join(_d, "data", "TeamProject_val")); FS = 25000
MARK = [("nor", 116), ("nor", 59), ("nor", 55), ("nor", 106), ("nor", 25), ("abn", 22), ("abn", 14), ("abn", 17), ("abn", 2)]
mark = [i for i, nm in enumerate(names) if any(nm.startswith(p) and int(nm.split("_")[-1]) == k for p, k in MARK)]

# %% 파일 단위 물리 10개 (9 · 13 · 14번 분석에서 가져옴, 라벨 안 씀)
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
def extras(Xa):
    Xc = Xa - Xa.mean(1, keepdims=True); HP = sosfiltfilt(SOS200, Xc, axis=-1); RM = np.array([ring_mean(HP[i]) for i in range(len(Xa))])
    fz, _, Zs = stft(Xc, fs=FS, nperseg=512, noverlap=256, boundary=None, padded=False, axis=-1); P2 = np.abs(Zs) ** 2; SK = (P2 ** 2).mean(-1) / (P2.mean(-1) ** 2 + 1e-30) - 2
    fw, Pw = welch(Xc, fs=FS, nperseg=2048, axis=-1)
    def slope(lo, hi):
        m = (fw >= lo) & (fw < hi); lf = np.log10(fw[m]); return np.array([np.polyfit(lf, np.log10(Pw[i, m] + 1e-30), 1)[0] for i in range(len(Xa))])
    return pd.DataFrame({"ring_f": RM[:, 0], "ring_zeta": np.log10(RM[:, 1] + 1e-4), "ring_peak_rms": RM[:, 2],
                         **{f"SK_{a}-{b}k": np.log10(SK[:, (fz >= a * 1000) & (fz <= b * 1000)].mean(1) + 1) for a, b in ((2, 4), (5.5, 7.75), (8, 8.7), (9, 10))},
                         "slope_lo": slope(200, 2000), "slope_hi": slope(2000, 12000), "log_rms": np.log10(Xc.std(1))}).fillna(0)
def ringdf(Xa):
    r, res, n = ring_table(Xa); return pd.DataFrame({"ring24": r[:, 0], "ring_body": r[:, 1], "res83": res, "nimp": n})

# %% D 행렬 만들기 (train fit, val은 변환만)
Ftr = cached_table("ext", feature_table_ext, X); Ftr = Ftr.loc[:, Ftr.nunique() > 3]; Fva = feature_table_ext(Xv)[Ftr.columns]
qt = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(Ftr.values); Z, Zv = qt.transform(Ftr.values), qt.transform(Fva.values)
Cm = np.abs(pd.DataFrame(Z).corr(method="spearman").fillna(0).values); cl = fcluster(linkage(squareform(1 - Cm, checks=False), "average"), 0.15, "distance")
reps = [int(np.where(cl == c)[0][np.argmax(Cm[np.ix_(cl == c, cl == c)].mean(1))]) for c in np.unique(cl)]
RG, RGv = ringdf(X), ringdf(Xv); EX, EXv = extras(X), extras(Xv)
qr = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(RG.values); qe = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(EX.values)
D = np.hstack([Z[:, reps], qr.transform(RG.values), qe.transform(EX.values)]); Dv = np.hstack([Zv[:, reps], qr.transform(RGv.values), qe.transform(EXv.values)])
cols = [Ftr.columns[i] for i in reps] + list(RG.columns) + list(EX.columns); print("D", D.shape, "| val", Dv.shape)

# %% 모델
def logit(p): p = np.clip(p, 0.01, 0.99); return np.log(p / (1 - p))
class LRm(BaseEstimator):
    def fit(self, X_, y_): self.sc_ = StandardScaler().fit(X_); self.m_ = LogisticRegression(C=0.3, class_weight="balanced", max_iter=3000).fit(self.sc_.transform(X_), y_); return self
    def decision_function(self, X_): return self.m_.decision_function(self.sc_.transform(X_))
class ETm(BaseEstimator):
    def fit(self, X_, y_): self.m_ = ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=1).fit(X_, y_); return self
    def decision_function(self, X_): return logit(self.m_.predict_proba(X_)[:, 1])
class Stack(BaseEstimator):
    """LR · ET 전문가 → 안쪽 4겹 OOF 점수 → 로지스틱 결합"""
    def fit(self, X_, y_):
        oof = np.zeros((len(y_), 2))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1).split(X_, y_):
            oof[te, 0] = LRm().fit(X_[tr], y_[tr]).decision_function(X_[te]); oof[te, 1] = ETm().fit(X_[tr], y_[tr]).decision_function(X_[te])
        self.sc_ = StandardScaler().fit(oof); self.st_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(self.sc_.transform(oof), y_)
        self.lr_, self.et_ = LRm().fit(X_, y_), ETm().fit(X_, y_); return self
    def experts(self, X_): return np.c_[self.lr_.decision_function(X_), self.et_.decision_function(X_)]
    def decision_function(self, X_): return self.st_.decision_function(self.sc_.transform(self.experts(X_)))
Fext = Ftr.values; g = np.arange(len(y))
# %% 중첩 CV (기존과 같은 프로토콜)
R = {}
for nm, (F_, mk_) in {"기준: ET (ext150 전체)": (Fext, ETm), "M2 ET (D 76열)": (D, ETm), "M1 LR (D 76열)": (D, LRm), "M3 스택 LR+ET (D 76열)": (D, Stack)}.items():
    r = nested_cv(F_, y, g, mk_, n_jobs=-1); R[nm] = r
    print(f"{nm:26s} F₂ {r['f2'].mean():.3f} ± {r['f2'].std():.3f} | 재현율 {r['recall'].mean():.3f} | 오경보율 {r['fpr'].mean():.3f} | 자주 놓침 {[names[i][-3:] for i in np.where(r['miss_counts'] >= 3)[0]]} | 오경보 3회+ {[names[i][-3:] for i in np.where(r['fp_counts'] >= 3)[0]]}", flush=True)
pd.DataFrame([(n, r["f2"].mean(), r["f2"].std(), r["recall"].mean(), r["precision"].mean(), r["fpr"].mean()) for n, r in R.items()], columns=["모델", "F2", "sd", "재현율", "정밀도", "오경보율"]).round(3).to_csv(os.path.join(HERE, "nested_cv.csv"), index=False, encoding="utf-8-sig")

# %% 경계 plot: 최종 스택 (train OOF 점 + val)
best = "M3 스택 LR+ET (D 76열)"
oofE = np.zeros((len(y), 2)); sc = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(D, y):
    m = Stack().fit(D[tr], y[tr]); oofE[te] = m.experts(D[te]); sc[te] = m.decision_function(D[te])
thr = best_f2_threshold(y, sc); fin = Stack().fit(D, y); Ev = fin.experts(Dv); sv = fin.decision_function(Dv)
mu, sd = oofE[y == 0].mean(0), oofE[y == 0].std(0); Zo, Zvv = (oofE - mu) / sd, (Ev - mu) / sd
# 경계: 스택 결정함수 = w·scale(expert) + b = thr (전문가 점수 평면에서 직선)
w = fin.st_.coef_[0] / fin.sc_.scale_; b = fin.st_.intercept_[0] - (fin.st_.coef_[0] * fin.sc_.mean_ / fin.sc_.scale_).sum()
gx = np.linspace(-3, 6, 80); gy = np.linspace(-3, 6, 80); GX, GY = np.meshgrid(gx, gy); Zg = w[0] * (GX * sd[0] + mu[0]) + w[1] * (GY * sd[1] + mu[1]) + b
pred_tr = (sc >= thr).astype(int); tp = ((pred_tr == 1) & (y == 1)).sum(); fp_ = ((pred_tr == 1) & (y == 0)).sum()
print(f"train OOF(5겹, seed 0) 임계값 {thr:.3f}: 재현율 {tp}/{y.sum()}, 오경보 {fp_}/{(y == 0).sum()} | 놓친 {[names[i] for i in np.where((pred_tr == 0) & (y == 1))[0]]} | 오경보 {[names[i] for i in np.where((pred_tr == 1) & (y == 0))[0]]}")
PUR, LIGHT, DARK = "#7B1FA2", "#CE93D8", "#4A148C"
fig, ax = plt.subplots(1, 2, figsize=(18, 7.2)); a = ax[0]
a.contourf(GX, GY, Zg, levels=[thr, 1e9], colors=[LIGHT], alpha=0.2); a.contour(GX, GY, Zg, levels=[thr], colors=[DARK], linewidths=1.8)
a.scatter(Zo[y == 0, 0], Zo[y == 0, 1], s=24, color="#9E9E9E", alpha=.65, label="train 정상 (OOF)"); a.scatter(Zo[y == 1, 0], Zo[y == 1, 1], s=40, color="#EF6C00", alpha=.9, label="train 고장 (OOF)")
a.scatter(Zvv[:, 0], Zvv[:, 1], c=np.where(sv >= thr, DARK, LIGHT), s=95, edgecolor=DARK, lw=1.4, zorder=5, label="val: 고장 판정 (진한 보라)"); a.scatter([], [], c=LIGHT, s=95, edgecolor=DARK, label="val: 정상 판정 (연보라)")
for i in mark: a.annotate(names[i].replace("nor_", "n").replace("abn_", "a"), Zo[i], fontsize=8, xytext=(4, 4), textcoords="offset points")
for i in np.where(sv >= thr)[0]: a.annotate(nv[i][-3:], Zvv[i], fontsize=8, color=DARK, weight="bold", xytext=(5, 5), textcoords="offset points")
a.set_xlim(-3, 6); a.set_ylim(-3, 6); a.set_xlabel("LR z (D 76열 선형 전문가)"); a.set_ylabel("ET z (D 76열 트리 전문가)"); a.legend(fontsize=8, loc="lower right")
a.set_title(f"D 피처 스택의 경계 (선 = 임계값 {thr:.2f}, 연보라 = 고장 영역)\nval 고장 판정 {int((sv >= thr).sum())}/40 · train OOF 재현율 {tp}/{y.sum()} 오경보 {fp_}/{(y == 0).sum()}", fontsize=10)
a = ax[1]; bins = np.linspace(min(sc.min(), sv.min()) - .3, max(sc.max(), sv.max()) + .3, 45)
a.hist(sc[y == 0], bins=bins, color="#9E9E9E", alpha=.65, label="train 정상 (OOF)"); a.hist(sc[y == 1], bins=bins, color="#EF6C00", alpha=.75, label="train 고장 (OOF)"); a.axvline(thr, color=DARK, ls="--", lw=1.6, label=f"임계값 {thr:.2f}")
a.scatter(sv, -2.5 + np.random.default_rng(0).uniform(-.9, .9, len(sv)), c=np.where(sv >= thr, DARK, LIGHT), s=65, edgecolor=DARK, zorder=5, label="val 40"); a.set_xlabel("스택 결합 점수"); a.set_ylabel("파일 수"); a.legend(fontsize=8); a.set_title("결합 점수 분포: train OOF 대비 val")
fig.savefig(os.path.join(FIG, "boundary_val.png"), dpi=130, bbox_inches="tight"); plt.close(fig)
V = pd.DataFrame({"file": nv, "lr_z": Zvv[:, 0], "et_z": Zvv[:, 1], "score": sv, "pred": (sv >= thr).astype(int)}).sort_values("score", ascending=False); V.round(3).to_csv(os.path.join(HERE, "val_scores.csv"), index=False, encoding="utf-8-sig")
print("val 고장 판정:", list(V[V.pred == 1].file.str[-3:]), "| 경계(±0.5):", list(V[(V.score - thr).abs() < 0.5].file.str[-3:]))
print("val z 평균(LR/ET):", Zvv.mean(0).round(2), "| train 정상 0, 고장 평균:", Zo[y == 1].mean(0).round(2))
