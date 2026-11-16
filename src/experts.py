"""전문가 점수 + 폴드 안 결합 (EXP-014에서 시작, EXP-015부터 공용)

전문가 = (학습 파일 tr, 채점 파일 te) → te 파일별 점수 행렬을 돌려주는 모듈 함수.
같은 분할은 디스크 캐시(outputs/cache/joblib_experts)로 재사용한다 → 구성끼리, 실험끼리 공유.
결합 모델 ScoreStack은 F 1열 = 파일 번호를 받는다 (nested_cv·quick_cv에 g[:, None]으로 넘김).

    from src.experts import ScoreStack, ae_mean, et_logit, warm_cache
    make = lambda: ScoreStack((ae_mean, et_logit), combine="lr")
    quick_cv(g[:, None], y, g, make)

- AE: EXP-012 구조(200 Hz 고역통과 → 1초 구간 8개 → log Welch PSD 1025 bin → 1025-128-16-128-1025, 정상만 학습)
- ET: ext150 ExtraTrees (make_et 기본)
피처 함수나 AE 설정을 바꾸면 outputs/cache/joblib_experts 를 지운다
"""
import os
import copy
import numpy as np
from joblib import Memory, Parallel, delayed
from scipy.signal import butter, sosfiltfilt, welch
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC

from src.exp import CACHE, load_train, cached_table, feature_table_ext, make_et

FS = 25000
mem = Memory(os.path.join(CACHE, "joblib_experts"), verbose=0)

# 데이터 (모듈을 불러올 때 캐시에서 읽음. 병렬 작업자도 각자 읽는다)
X, y, names = load_train()
E = cached_table("ext", feature_table_ext, X).values


def segment_logpsd(X, seg=FS, nseg=8, nperseg=2048):
    """평균 제거 → 200 Hz 고역통과(Butterworth 4차, 양방향) → 1초 구간 nseg개 → log10 Welch PSD. 라벨 안 씀"""
    Xc = X - X.mean(1, keepdims=True)
    H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
    starts = np.linspace(0, X.shape[1] - seg, nseg).astype(int)
    S = np.stack([H[:, s:s + seg] for s in starts], 1)
    f, P = welch(S, fs=FS, nperseg=nperseg, axis=-1)
    return f, np.log10(P + 1e-20)


_p = os.path.join(CACHE, "seg_logpsd_hp200_8x1s.npz")
if os.path.exists(_p):
    _z = np.load(_p); F_PSD, LP = _z["f"], _z["LP"]
else:
    F_PSD, LP = segment_logpsd(X); np.savez(_p, f=F_PSD, LP=LP)


# ---------- AE ----------
@mem.cache
def ae_segerr(tr, te, shape=False, seed=0):
    """정상(tr 중 y=0)의 구간 스펙트럼으로 AE 학습 → te의 구간·bin별 복원 제곱오차 (te, 8, 1025) float32.
    shape=True: 구간마다 평균 log PSD를 빼서 전체 높이(게인)를 지운 '모양'만 학습 (H3)"""
    tr, te = np.array(tr), np.array(te)
    L = LP - LP.mean(-1, keepdims=True) if shape else LP
    S = L[tr[y[tr] == 0]].reshape(-1, L.shape[-1])
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Zs = (S - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3,
                     max_iter=60, alpha=0.0, random_state=seed).fit(Zs, Zs)
    Z = (L[te] - mu) / sd
    R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return ((R - Z) ** 2).astype(np.float32)


def ae_mean(tr, te):
    """EXP-012·014 AE 점수: log10 (구간·bin 평균 복원 오차) → (te, 1)"""
    return np.log10(ae_segerr(tr, te).mean((1, 2)))[:, None]


def ae_bins(tr, te):
    """AE 출력 끝: bin별 log10 복원 오차 (구간 평균) → (te, 1025)"""
    return np.log10(ae_segerr(tr, te).mean(1) + 1e-12)


def ae_max(tr, te):
    """A: 구간 8개 중 오차가 가장 큰 구간 (드물게 나는 국소 고장이 평균에 묻히지 않게) → (te, 1)"""
    return np.log10(ae_segerr(tr, te).mean(2).max(1))[:, None]


def ae_shape_mean(tr, te):
    """C: 모양만 보는 AE (구간마다 평균 log PSD 제거, H3) 오차 평균 → (te, 1)"""
    return np.log10(ae_segerr(tr, te, shape=True).mean((1, 2)))[:, None]


def ae_shape_max(tr, te):
    """A+C: 모양 AE, 구간 최댓값 → (te, 1)"""
    return np.log10(ae_segerr(tr, te, shape=True).mean(2).max(1))[:, None]


# ---------- 충격 피처 (학습 없음, src/features.py feature_table_impact) ----------
_IMP = None


def _imp():
    global _IMP
    if _IMP is None:
        from src.features import feature_table_impact
        _IMP = cached_table("impact", feature_table_impact, X)
    return _IMP


def imp_period(tr, te):
    """B (H4·H1): 포락선 주기성 · 충격 간격 변동계수 · 저주파 동기 · 저주파 비율 → (te, 4)"""
    return _imp()[["B1_env_acf_peak", "B3_interval_cv", "B4_low_sync", "B5_low_frac"]].values[list(te)]


def imp_single(tr, te):
    """E (H5): 사건 수 · 두 번째/첫 번째 높이 · 큰 사건 잔향 비 · 큰 사건 높이 → (te, 4)"""
    return _imp()[["E1_n_impact_log", "E2_top2_ratio", "E3_big_ringing_log", "E4_big_height_log"]].values[list(te)]


# ---------- 대역 비율 축 (2-10) ----------
_BAND = None


def _band():
    global _BAND
    if _BAND is None:
        from src.features import feature_table_band200
        _BAND = cached_table("band200", feature_table_band200, X)
    return _BAND


def band_grid_top(tr, te):
    """200 Hz 격자 61칸 중 학습 파일(tr)에서 단변량 AUC가 가장 큰 칸 하나 → (te, 1). 칸 선택은 폴드 안 (원칙 3)"""
    from sklearn.metrics import roc_auc_score
    T = _band()[[c for c in _band().columns if c.startswith("G")]].values
    tr = list(tr)
    j = int(np.argmax([roc_auc_score(y[tr], T[tr, k]) for k in range(T.shape[1])]))
    return T[list(te), j][:, None]


def band_grid_choice(tr):
    """band_grid_top이 tr에서 고르는 칸 이름 (기록용)"""
    from sklearn.metrics import roc_auc_score
    cols = [c for c in _band().columns if c.startswith("G")]
    T = _band()[cols].values
    tr = list(tr)
    return cols[int(np.argmax([roc_auc_score(y[tr], T[tr, k]) for k in range(T.shape[1])]))]


def band_8k_post(tr, te):
    """8.2–8.4 kHz / 200 Hz 이상 파워 (2-10 사후 정의, 비교용) → (te, 1)"""
    return _band()[["P08200_08400"]].values[list(te)]


# ---------- ET ----------
@mem.cache
def et_trees(tr, te, seed=0):
    """ext150 ET 학습 → te의 나무별 고장 확률 (te, 300)"""
    tr, te = np.array(tr), np.array(te)
    m = make_et(seed=seed)().fit(E[tr], y[tr])
    return np.column_stack([t.predict_proba(E[te])[:, 1] for t in m.estimators_]).astype(np.float32)


def et_logit(tr, te):
    """EXP-014 ET 점수: logit(나무 평균 확률) → (te, 1). predict_proba와 같은 값"""
    p = np.clip(et_trees(tr, te).mean(1), 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p))[:, None]


# ---------- 설정을 바꿀 수 있는 전문가 (EXP-018 역할별 최적화) ----------
@mem.cache
def ae_segerr_robust(tr, te, seed=0):
    """ae_segerr와 같은 AE, 표준화만 로버스트: bin별 중앙값·MAD×1.4826, 척도 하한 = bin 척도들의 중앙값.
    정상 산포가 아주 작은 bin이 조금만 달라도 오차가 커지는(비특이적 오차) 것을 막으려는 것 (2-10)"""
    tr, te = np.array(tr), np.array(te)
    S = LP[tr[y[tr] == 0]].reshape(-1, LP.shape[-1])
    mu = np.median(S, 0)
    sd = 1.4826 * np.median(np.abs(S - mu), 0)
    sd = np.maximum(sd, np.median(sd)) + 1e-8
    Zs = (S - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3,
                     max_iter=60, alpha=0.0, random_state=seed).fit(Zs, Zs)
    Z = (LP[te] - mu) / sd
    R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return ((R - Z) ** 2).astype(np.float32)


class AEScore:
    """AE 점수 (설정 가능). scale: 'std'(EXP-012 그대로) | 'robust'. agg: 'mean' | 'top0.05' | 'top0.01'
    (구간 평균한 bin 오차 중 상위 비율 bin의 평균 — 오차가 한곳에 몰린 파일과 흩어진 파일을 구분). → (te, 1) log10"""
    def __init__(self, scale="std", agg="mean"): self.scale, self.agg = scale, agg
    def __repr__(self): return f"AE({self.scale},{self.agg})"
    def bins(self, tr, te):
        e = ae_segerr(tr, te) if self.scale == "std" else ae_segerr_robust(tr, te)
        return e.mean(1)                                       # (te, 1025)
    def __call__(self, tr, te):
        b = self.bins(tr, te)
        if self.agg == "mean":
            s = b.mean(1)
        else:
            k = max(1, int(round(float(self.agg[3:]) * b.shape[1])))
            s = np.sort(b, 1)[:, -k:].mean(1)
        return np.log10(s + 1e-12)[:, None]


@mem.cache
def et_trees_p(tr, te, leaf=2, mf="sqrt", seed=0):
    """ExtraTrees 설정 가능판 → 나무별 고장 확률 (te, 300). leaf=2, mf='sqrt'면 et_trees와 같은 모델"""
    from sklearn.ensemble import ExtraTreesClassifier
    tr, te = np.array(tr), np.array(te)
    m = ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=leaf, max_features=mf,
                             random_state=seed, n_jobs=1).fit(E[tr], y[tr])
    return np.column_stack([t.predict_proba(E[te])[:, 1] for t in m.estimators_]).astype(np.float32)


class ETScore:
    """ET 점수 (설정 가능): logit(나무 평균 확률) → (te, 1)"""
    def __init__(self, leaf=2, mf="sqrt"): self.leaf, self.mf = leaf, mf
    def __repr__(self): return f"ET(leaf{self.leaf},{self.mf})"
    def trees(self, tr, te):
        return et_trees(tr, te) if (self.leaf, self.mf) == (2, "sqrt") else et_trees_p(tr, te, self.leaf, self.mf)
    def __call__(self, tr, te):
        p = np.clip(self.trees(tr, te).mean(1), 1 / 600, 1 - 1 / 600)
        return np.log(p / (1 - p))[:, None]


# ---------- 결합 ----------
class BlockScaler(BaseEstimator, TransformerMixin):
    """열마다 표준화 후 블록마다 1/√(열 수)를 곱해 블록끼리 전체 분산을 같게 (차원이 큰 블록이 덮지 않게)"""
    def __init__(self, blocks=None): self.blocks = blocks
    def fit(self, Z, yy=None):
        self.mu_, self.sd_ = Z.mean(0), Z.std(0) + 1e-12
        b = self.blocks or [Z.shape[1]]
        self.w_ = np.concatenate([np.full(n, 1 / np.sqrt(n)) for n in b])
        return self
    def transform(self, Z): return (Z - self.mu_) / self.sd_ * self.w_


def _combiner(kind, blocks):
    pre = [] if blocks is None else [BlockScaler(blocks)]
    if kind == "lr":
        return make_pipeline(*pre, LogisticRegression(C=1.0, class_weight="balanced"))
    if kind == "svm_lin":
        return make_pipeline(*pre, SVC(kernel="linear", C=1.0, class_weight="balanced"))
    if kind == "svm_rbf":
        return make_pipeline(*pre, SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))
    return None


class ScoreStack(BaseEstimator):
    """전문가 점수를 폴드 안 OOF로 모아 결합

    fit(F, y): 학습 파일 안 k-fold OOF로 전문가 점수 → (norm=True) 열마다 학습 정상 OOF 기준 z
               = (s − 중앙값) / |90% − 중앙값| → 결합기 학습. 전문가는 학습 파일 전체로 다시 학습해 채점
    combine: 'lr' 'svm_lin' 'svm_rbf' (학습), 'mean' 'max' (학습 0개), 'col0' (첫 열 그대로)
    고차원 전문가(ae_bins·et_trees)는 norm=False, blocks=[열 수 …]로 블록 같은 비중 표준화"""
    def __init__(self, experts=(), combine="lr", norm=True, blocks=None, k=4, seed=0, add_max=False,
                 lr_cols=None, or_cols=(), q=95):
        self.experts, self.combine, self.norm, self.blocks, self.k, self.seed = experts, combine, norm, blocks, k, seed
        self.add_max, self.lr_cols, self.or_cols, self.q = add_max, lr_cols, or_cols, q

    # combine="strong" (EXP-017): 강한 증거 묶음. 로지스틱(lr_cols) 점수와 or_cols 축마다
    #   e = (s − 정상 중앙값) / (정상 q% − 정상 중앙값)  (학습 정상 OOF 기준 → 1 = 정상 q%선)
    #   최종 점수 = max(e). 평균으로 희석하지 않고 가장 강한 증거로 판단. or_cols가 비면 로지스틱과 같은 순위
    def _strong_fit(self, Z, yy):
        lc = list(range(Z.shape[1])) if self.lr_cols is None else list(self.lr_cols)
        self.lc_ = lc
        self.m_ = LogisticRegression(C=1.0, class_weight="balanced").fit(Z[:, lc], yy)
        S = np.column_stack([self.m_.decision_function(Z[:, lc])] + [Z[:, j] for j in self.or_cols])
        n = S[yy == 0]
        self.smed_ = np.median(n, 0)
        d = np.percentile(n, self.q, 0) - self.smed_
        self.ssc_ = np.where(d > 1e-9, d, n.std(0)) + 1e-6

    def strong_parts(self, F):
        """강한 증거 성분 (파일, 1 + len(or_cols)): 0열 = 로지스틱, 나머지 = or_cols 축. 1 = 정상 q%선"""
        Z = self.zscores(F)
        S = np.column_stack([self.m_.decision_function(Z[:, self.lc_])] + [Z[:, j] for j in self.or_cols])
        return (S - self.smed_) / self.ssc_

    def _raw(self, tr, te):
        return np.hstack([fn(tr, te) for fn in self.experts]).astype(float)

    def _z(self, S):
        Z = (S - self.med_) / self.sc_ if self.norm else S
        if self.add_max:                                      # D: 처음 두 전문가 z 중 큰 값을 한 열 더 (강한 한쪽 증거)
            Z = np.column_stack([Z, Z[:, :2].max(1)])
        return Z

    def fit(self, F, yy):
        idx, yy = np.asarray(F)[:, 0].astype(int), np.asarray(yy)
        self.tr_ = tuple(idx.tolist())
        oof = None
        for a, b in StratifiedKFold(self.k, shuffle=True, random_state=self.seed).split(idx, yy):
            s = self._raw(tuple(idx[a].tolist()), tuple(idx[b].tolist()))
            oof = np.empty((len(idx), s.shape[1])) if oof is None else oof
            oof[b] = s
        n = oof[yy == 0]
        self.med_ = np.median(n, 0)
        d = np.abs(np.percentile(n, 90, 0) - self.med_)
        self.sc_ = np.where(d > 1e-9, d, n.std(0)) + 1e-6      # 90%선 = 중앙값(값이 몰린 열)이면 표준편차로
        self.oof_z_ = self._z(oof)
        if self.combine == "strong":
            self._strong_fit(self.oof_z_, yy)
            return self
        self.m_ = _combiner(self.combine, self.blocks)
        if self.m_ is not None:
            self.m_.fit(self.oof_z_, yy)
        return self

    def zscores(self, F):
        return self._z(self._raw(self.tr_, tuple(np.asarray(F)[:, 0].astype(int).tolist())))

    def decision_function(self, F):
        if self.combine == "strong":
            return self.strong_parts(F).max(1)
        Z = self.zscores(F)
        if self.m_ is not None:
            return self.m_.decision_function(Z)
        return {"mean": lambda: Z.mean(1), "max": lambda: Z.max(1), "col0": lambda: Z[:, 0]}[self.combine]()


def warm_cache(fns, seeds=range(3), n_outer=5, k=4, nested=False, n_jobs=-1):
    """quick_cv(seeds)가 부를 (학습, 채점) 분할을 미리 병렬로 캐시에 채운다. fns = 캐시되는 함수들 (ae_segerr, et_trees …).
    nested=True면 nested_cv(seed 5, 안쪽 4)의 분할까지"""
    g = np.arange(len(y))
    outer = []
    for s in seeds:
        for tr, te in StratifiedGroupKFold(n_outer, shuffle=True, random_state=s).split(g, y, g):
            outer.append((tr, te))
            if nested:
                for itr, ite in StratifiedGroupKFold(4, shuffle=True, random_state=s).split(tr, y[tr], tr):
                    outer.append((tr[itr], tr[ite]))
    jobs = []
    for tr, te in outer:
        jobs.append((tuple(tr.tolist()), tuple(te.tolist())))
        for a, b in StratifiedKFold(k, shuffle=True, random_state=0).split(tr, y[tr]):
            jobs.append((tuple(tr[a].tolist()), tuple(tr[b].tolist())))
    Parallel(n_jobs=n_jobs)(delayed(fn)(a, b) for a, b in jobs for fn in fns)
    return len(jobs)


# ---------- 8–9 kHz 봉우리 축 (EXP-020, 2-9·2-11 README2) ----------
# 돌출 = log10 Welch PSD(평균 제거, 8192) 55 Hz 평활 − 1.2 kHz 이동 중앙값 (2-9 정의, 라벨 안 씀)
_PR = None


def prominence():
    """(f, PR): 파일 × 주파수 돌출 (dec). 한 번 계산해 캐시 파일로 저장"""
    global _PR
    if _PR is None:
        p = os.path.join(CACHE, "prominence_w8192_s55_m1200.npz")
        if os.path.exists(p):
            z = np.load(p); _PR = (z["f"], z["PR"])
        else:
            from scipy.ndimage import median_filter, uniform_filter1d
            from src.viz import log_psd
            f, L = log_psd(X)
            df = f[1] - f[0]
            S = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
            PR = S - median_filter(S, size=(1, int(round(1200 / df))), mode="nearest")
            np.savez(p, f=f, PR=PR); _PR = (f, PR)
    return _PR


class PeakExceed:
    """정상 봉우리 덮개 초과: 학습 파일(tr)의 정상만으로 주파수별 돌출 q% 선을 만들고,
    채점 파일의 (돌출 − 덮개) 최댓값을 [lo, hi] Hz 안에서 → (te, 1). 정상에 흔한 위치의 봉우리는 낮게, 드문 위치는 높게"""
    def __init__(self, lo=8000, hi=9000, q=90): self.lo, self.hi, self.q = lo, hi, q
    def __repr__(self): return f"PeakExceed({self.lo},{self.hi},q{self.q})"
    def __call__(self, tr, te):
        f, PR = prominence()
        b = (f >= self.lo) & (f <= self.hi)
        tr = np.array(tr)
        cover = np.percentile(PR[tr[y[tr] == 0]][:, b], self.q, axis=0)
        return (PR[list(te)][:, b] - cover).max(1)[:, None]


def mode_contrast(tr, te):
    """[사후] 두 공진 모드의 상대 크기: 8.20–8.45 kHz 최대 돌출 − 8.45–8.75 kHz 최대 돌출 (2-9에서 라벨 보고 정한 경계) → (te, 1)"""
    f, PR = prominence()
    a, b = (f >= 8200) & (f < 8450), (f >= 8450) & (f < 8750)
    P = PR[list(te)]
    return (P[:, a].max(1) - P[:, b].max(1))[:, None]


# ---------- 8.3 kHz 모드 규칙 (EXP-021) ----------
def peak83_table():
    """파일별 (P83, W83, P86): 8.20–8.45 kHz 최대 돌출(dec), 그 봉우리의 반높이 폭(Hz, 돌출 기준), 8.45–8.75 kHz 최대 돌출.
    대역 경계는 2-9에서 라벨을 보고 정한 값 (결정.md '대역 예외')"""
    f, PR = prominence()
    a, b = (f >= 8200) & (f < 8450), (f >= 8450) & (f < 8750)
    ia = np.flatnonzero(a)
    P83, P86, W = PR[:, a].max(1), PR[:, b].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, a].argmax()]; h = PR[i, k] / 2; l = r = k
        while l > 0 and PR[i, l - 1] >= h: l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h: r += 1
        W[i] = f[r] - f[l]
    return P83, W, P86


class Rule83:
    """규칙: 8.20–8.45 kHz 돌출 ≥ p_min (그리고 폭 ≥ w_min Hz면) → 무조건 고장 (te,) bool"""
    def __init__(self, p_min=0.75, w_min=0): self.p_min, self.w_min = p_min, w_min
    def __repr__(self): return f"Rule83(p≥{self.p_min}, w≥{self.w_min})"
    def __call__(self, te):
        P83, W, _ = peak83_table()
        te = list(te)
        return (P83[te] >= self.p_min) & (W[te] >= self.w_min)


class KnockRule:
    """유격 반복 규칙 (2-12, EXP-025): 정상 중심 거리 r ≥ r_min 이고 2위/1위 사건 높이 ≥ ratio_min이면 무조건 고장 (te,) bool.
    r = √((z1² + z2²)/2), z = K1(유격 반복)·K2(하중 대비 고주파)를 학습 정상의 중앙값·MAD로 맞춘 값 (fit(tr), 폴드 안).
    문턱 2·0.7은 2-12에서 라벨을 본 뒤 정한 사후 값 (결정.md '사후 문턱 예외')"""
    def __init__(self, r_min=2.0, ratio_min=0.7): self.r_min, self.ratio_min = r_min, ratio_min
    def __repr__(self): return f"Knock(r≥{self.r_min}, R≥{self.ratio_min})"
    def fit(self, tr):
        tr = np.asarray(tr)
        K = _knock()[tr[y[tr] == 0]][:, :2]
        self.med_ = np.median(K, 0)
        self.mad_ = 1.4826 * np.median(np.abs(K - self.med_), 0) + 1e-12
        return self
    def r(self, te):
        Z = (_knock()[list(te)][:, :2] - self.med_) / self.mad_
        return np.sqrt((Z ** 2).mean(1))
    def __call__(self, te):
        return (self.r(te) >= self.r_min) & (_knock()[list(te)][:, 2] >= self.ratio_min)


def _knock():
    from src.exp import cached_table as _ct
    from src.features import feature_table_knock
    return _ct("knock", feature_table_knock, X).values


class RuleOverride(BaseEstimator):
    """기본 결합 모델 + 규칙: 규칙이 켜진 파일은 점수에 +100 (어떤 임계값보다도 위 = 무조건 고장)
    rule: 규칙 하나 또는 리스트(하나라도 켜지면 고장). fit(tr)이 있는 규칙은 학습 파일로 먼저 fit (폴드 안)"""
    def __init__(self, base_kw=None, rule=None): self.base_kw, self.rule = base_kw, rule
    def _rules(self):
        return [] if self.rule is None else (list(self.rule) if isinstance(self.rule, (list, tuple)) else [self.rule])
    def fit(self, F, yy):
        idx = np.asarray(F)[:, 0].astype(int)
        self.m_ = ScoreStack(**(self.base_kw or {})).fit(F, yy)
        self.rules_ = [copy.deepcopy(r).fit(idx) if hasattr(r, "fit") else r for r in self._rules()]
        return self
    def decision_function(self, F):
        s = self.m_.decision_function(F)
        te = np.asarray(F)[:, 0].astype(int)
        for r in self.rules_:
            s = s + 100.0 * r(te)
        return s
