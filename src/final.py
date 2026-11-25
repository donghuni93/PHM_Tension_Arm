"""최종 파이프라인 (7단계): 학습 파일 전체로 한 번 학습 → 새 파일(val·test) 판정 + 판정 근거

왜 이 파일이 따로 있나
- src/experts.py는 실험용이다. 학습 150개를 모듈 안에 들고 파일 번호로 주고받는다(분할마다 캐시를 나눠 쓰려고).
  그래서 번호가 없는 새 파일(val·test)은 채점할 수 없다.
- 이 파일은 제출용이다. 신호 배열만 받으면 동작한다. 계산식은 experts.py와 같고,
  같다는 것은 단계/7_최종·제출/검증_동등성.py 가 확인한다 (같은 분할에서 점수 차이 0).

흐름 (위에서 아래로 읽으면 파이프라인 순서다)
  1. signal_tables : 신호 → 숫자 표 (라벨을 쓰지 않는다. 파일 하나씩 따로 계산한다)
  2. Core.fit      : 전문가 2개(AE·ET) 학습 → 정상 기준 z → 로지스틱 결합 + 규칙의 기준값
  3. Core.parts    : 파일별 점수와 규칙 판정
  4. FinalModel    : 임계값(판정 기준선)까지 정해서 0/1과 근거를 낸다

** 지금 제출 구성 = J-축소 (EXP-050, 11-22) → 파일 맨 아래 FinalJ · JShrink · j_tables. 단계/7_최종·제출 스크립트는 FinalJ를 쓴다. **
아래 Core · FinalModel은 예전 최종(EXP-034 c)이다. 비교 · 기록용으로 남긴다.

예전 최종 구성 = EXP-034 c (11-17 사용자 결정으로 EXP-025에서 교체):
  AE + ET(ext150 + 울림 2열) → 로지스틱 + 유격 반복 규칙 + 8.3 kHz 규칙(스펙트럼 봉우리 OR 충격 공진) + 본체 모드 내림.
  ring=False면 예전 최종 EXP-025(울림 열·공진 규칙·본체 내림 없음)와 같다.
주기 잠금·단일 타격은 판정에 쓰지 않고 '관찰' 열로만 남긴다 (use_lock·use_cap 기본값 False).
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, welch
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.neural_network import MLPRegressor

from src.evaluate import best_f2_threshold
from src.features import extract_features_ext, extract_features_hits, extract_features_knock

FS = 25000
N_SAMPLES = 50000
# 규칙 문턱 (모두 학습 라벨을 본 뒤 정한 '사후' 값. 단계/1_기준·기록/결정.md. val·test를 보고 고치지 않는다)
P83_MIN, W83_MIN = 0.75, 150.0      # 8.3 kHz 모드 규칙: 돌출(dec), 반높이 폭(Hz)
# 8.3 kHz 규칙의 세 번째 조건 (11-23, 2-16): 8.20–8.45 kHz 포락 첨도 K83 > K83_MIN → "충격으로 울린 것"만 인정(지속음 제외).
# 이 문턱은 데이터에서 고르지 않았다: 가우스 잡음 K = 2.00 ± 0.08 (99 % 2.2, 시뮬레이션) 바로 위 + 여유 → 3. 순수 지속음 K = 1
K83_MIN = 3.0
R_MIN, RATIO_MIN = 2.0, 0.7         # 유격 반복 규칙: 정상 중심 거리, 2위/1위 타격 높이
LOCK_MARGIN = 0.1                   # 주기 잠금(관찰용): 학습 정상 최대 + 여유
RES_FIT_MIN = 0.7                   # 8.3 kHz 공진 규칙: 8.0–8.7 kHz · ζ ≤ 0.02 충격의 감쇠 사인파 맞춤 fit (2-16, EXP-034)
BODY_MIN, BODY_NMIN = 0.9, 3        # 본체 모드 내림: 5.5–7.75 kHz로 울린 충격 비율, 최소 충격 수
FIXED_THR = 1.0                     # 제출용 고정 임계값 (11-17 사용자 결정, EXP-035: 중첩 CV 바깥 점수의 고정 최적값 1.094에서 왼쪽으로 여유를 둔 값, F2 0.957). None이면 seed 중앙값
KEYS = ("LP", "E", "K", "P83", "W83", "K83", "n20", "S", "RING", "RES", "NIMP")
_SOS83 = butter(4, [8200, 8450], btype="bandpass", fs=FS, output="sos")


def kurt83(Xc):
    """평균 제거 신호 (n, 50000) → 8.20–8.45 kHz 대역 포락 첨도 K83 = mean(p²)/mean(p)², p = 포락² (양 끝 0.05 s 제외).
    순수 지속음 1 · 가우스 잡음 2 · 충격 울림 ≫ 2 (2-12 · 2-16). 규칙 대역은 P83·W83과 같다"""
    p = np.abs(hilbert(sosfiltfilt(_SOS83, np.atleast_2d(Xc), axis=-1), axis=-1))[:, 1250:-1250] ** 2
    return (p ** 2).mean(1) / p.mean(1) ** 2


# ---------------------------------------------------------------------------
# 0. 충격 울림 (2-16, 신효섭 정의): 충격을 찾아 충격 뒤 8 ms가 어떤 감쇠 사인파와 가장 닮았는지
# ---------------------------------------------------------------------------
_SEG = int(0.008 * FS)


def _ring_atoms():
    """감쇠 사인파 사전: 주파수 1–12 kHz(250 Hz 간격) × 감쇠비 0.01·0.02·0.04·0.08 = 180개. 각각 sin·cos 2열의 직교 기저"""
    t = np.arange(_SEG) / FS
    atoms, par = [], []
    for f in np.arange(1000, 12001, 250):
        for z in (0.01, 0.02, 0.04, 0.08):
            w = 2 * np.pi * f
            wd, a = w * np.sqrt(1 - z ** 2), np.exp(-z * w * t)
            Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1))
            atoms.append(Q)
            par.append((f, z))
    return np.stack(atoms), np.array(par)


def impact_rings(X):
    """신호 → 충격 하나하나의 (파일 번호, 울림 주파수 f, 감쇠비 ζ, fit). 2-16_run.py와 같은 계산

    평균 제거 → 1 kHz 고역통과 → Hilbert 포락선 → 500 Hz 저역 → 포락선 > 6 × 중앙값 봉우리(5 ms 간격) = 충격
    fit = 충격 뒤 8 ms(단위 길이로 맞춤)를 사전의 2차원 기저에 투영한 길이 (1 = 완전히 같은 모양)
    """
    X = np.asarray(X, dtype=float)
    X1k = sosfiltfilt(butter(4, 1000, btype="highpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=-1)
    lp500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")
    atoms, par = _ring_atoms()
    rows = []
    for i, x in enumerate(X1k):
        env = sosfiltfilt(lp500, np.abs(hilbert(x)))
        pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
        pk = pk[pk + _SEG <= len(x)]
        if not len(pk):
            continue
        S = np.stack([x[p:p + _SEG] for p in pk])
        S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        fit = np.linalg.norm(np.einsum("ks,ast->kat", S, atoms), axis=2)
        b = fit.argmax(1)
        rows += [(i, par[bb, 0], par[bb, 1], ft) for bb, ft in zip(b, fit.max(1))]
    return np.array(rows, dtype=float).reshape(-1, 4)


def ring_table(X):
    """파일별 울림 요약: RING (n,2) = [2.125–4.125 kHz로 울린 비율, 5.5–7.75 kHz(본체 모드)로 울린 비율] (충격 0개 = −1),
    RES (n,) = 8.0–8.7 kHz · ζ ≤ 0.02 충격의 최대 fit (없으면 0), NIMP (n,) = 충격 수"""
    n = len(X)
    R = impact_rings(X)
    fi, f, z, ft = (R[:, k] for k in range(4))
    fi = fi.astype(int)
    nimp = np.bincount(fi, minlength=n).astype(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = lambda m: np.where(nimp > 0, np.bincount(fi, weights=m.astype(float), minlength=n) / nimp, -1.0)
        ring = np.column_stack([frac((f >= 2125) & (f < 4125)), frac((f >= 5500) & (f < 7750))])
    res = np.zeros(n)
    m = (f >= 8000) & (f < 8700) & (z <= 0.02)
    np.maximum.at(res, fi[m], ft[m])
    return ring, res, nimp


# ---------------------------------------------------------------------------
# 1. 신호 → 표
# ---------------------------------------------------------------------------
def signal_tables(X, cols=None):
    """신호 배열 (파일 수, 50000) → 표 묶음 dict. 라벨을 쓰지 않고 파일마다 독립이라 학습·val·test에 똑같이 쓴다

    LP  (n, 8, 1025) AE 입력: 평균 제거 → 200 Hz 고역통과 → 1초 구간 8개 → log10 Welch PSD
    E   (n, 150)     ET 입력: 확장 피처 (src/features.py extract_features_ext)
    K   (n, 3)       유격 반복: K1 두 번째 타격/σ, K2 저주파 RMS/σ, K3 2위/1위
    P83·W83 (n,)     8.20–8.45 kHz 봉우리의 돌출(dec)과 반높이 폭(Hz)
    K83 (n,)         8.20–8.45 kHz 대역 포락 첨도 (지속음 1 · 잡음 2 · 충격 ≫ 2). 8.3k 규칙의 세 번째 조건
    RING (n, 2)      ET 추가 입력: 충격 중 2.125–4.125 kHz · 5.5–7.75 kHz로 울린 비율 (충격 0개 = −1)
    RES·NIMP (n,)    8.3 kHz 공진 충격의 최대 fit · 충격 수 (ring_table)
    n20 (n,)         큰 타격(바탕의 20배 이상) 수   — 관찰용
    S   (n,)         주기 잠금 점수                 — 관찰용
    cols: 학습 때의 E 열 이름. 주면 같은 순서로 맞춘다 (없는 열은 0)
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_SAMPLES:
        raise ValueError(f"신호 길이는 {N_SAMPLES}점이어야 한다. 받은 모양 {X.shape}")
    Xc = X - X.mean(1, keepdims=True)

    # AE 입력
    H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
    starts = np.linspace(0, X.shape[1] - FS, 8).astype(int)
    seg = np.stack([H[:, s:s + FS] for s in starts], 1)
    _, P = welch(seg, fs=FS, nperseg=2048, axis=-1)
    LP = np.log10(P + 1e-20)

    # ET 입력, 유격 반복, 큰 타격·주기 잠금
    clean = lambda T: T.replace([np.inf, -np.inf], np.nan).fillna(0)
    E = clean(pd.DataFrame([extract_features_ext(x) for x in X]))
    if cols is not None:
        # 피처 함수가 학습 때와 같으면 열 집합이 같다. 일부 파일에서만 생기는 열(F09·F10 봉우리)이 빠진 경우만 0으로 채운다
        extra = set(E.columns) - set(cols)
        if extra:
            raise ValueError(f"학습 때 없던 피처 열이 생겼다 (피처 함수가 바뀌었는가): {sorted(extra)[:5]}")
        E = E.reindex(columns=cols, fill_value=0.0)
    K = clean(pd.DataFrame([extract_features_knock(x) for x in X]))
    Hit = clean(pd.DataFrame([extract_features_hits(x) for x in X])[["HIT_n20", "LOCK_S"]])

    # 8.3 kHz 봉우리: 돌출 = 55 Hz로 고른 log PSD − 1.2 kHz 이동 중앙값 (주변보다 얼마나 솟았나)
    f, P8 = welch(Xc, fs=FS, nperseg=8192)
    L = np.log10(P8)
    df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
    PR = Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest")
    band = (f >= 8200) & (f < 8450)
    ia = np.flatnonzero(band)
    P83, W83 = PR[:, band].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, band].argmax()]
        h = PR[i, k] / 2
        l = r = k
        while l > 0 and PR[i, l - 1] >= h:
            l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h:
            r += 1
        W83[i] = f[r] - f[l]

    RING, RES, NIMP = ring_table(X)
    T = dict(LP=LP, E=E.values.astype(float), K=K[["K1_second_snr", "K2_load_to_hf", "K3_top2_ratio"]].values,
             P83=P83, W83=W83, K83=kurt83(Xc), n20=Hit["HIT_n20"].values.astype(int), S=Hit["LOCK_S"].values,
             RING=RING, RES=RES, NIMP=NIMP)
    return T, list(E.columns)


def take(T, idx):
    """표 묶음에서 파일 일부만"""
    return {k: T[k][idx] for k in KEYS}


# ---------------------------------------------------------------------------
# 2–3. 전문가 2개 + 결합 + 규칙
# ---------------------------------------------------------------------------
def _fit_ae(LP_normal, seed=0):
    """정상 구간 스펙트럼만으로 오토인코더(1025-128-16-128-1025) 학습. 반환 (평균, 표준편차, 모델)"""
    S = LP_normal.reshape(-1, LP_normal.shape[-1])
    mu, sd = S.mean(0), S.std(0) + 1e-8
    Z = (S - mu) / sd
    m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3,
                     max_iter=60, alpha=0.0, random_state=seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        m.fit(Z, Z)
    return mu, sd, m


def _ae_score(ae, LP):
    """AE 점수 = log10(복원 오차 평균). 정상 스펙트럼 모양에서 벗어날수록 크다"""
    mu, sd, m = ae
    Z = (LP - mu) / sd
    R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return np.log10(((R - Z) ** 2).astype(np.float32).mean((1, 2))).astype(float)


def _fit_et(E, y, seed=0):
    return ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt",
                                random_state=seed, n_jobs=1).fit(E, y)


def _et_score(et, E):
    """ET 점수 = logit(나무 300그루의 고장 확률 평균)"""
    p = np.column_stack([t.predict_proba(E)[:, 1] for t in et.estimators_]).astype(np.float32).mean(1)
    p = np.clip(p, 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p)).astype(float)


class Core:
    """전문가 2개(AE·ET) → 정상 기준 z → 로지스틱 결합 → 규칙. 임계값은 여기서 정하지 않는다 (FinalModel)

    ring=True  : EXP-034 c 와 같은 계산 (ET 입력 + 울림 2열, 8.3k = 스펙트럼 OR 공진, 본체 모드 내림)
    ring=False : src/experts.py의 RuleOverride(ScoreStack((ae_mean, et_logit), 'lr'), [Rule83, KnockRule]) = EXP-025
    """

    def __init__(self, use_cap=False, use_lock=False, ring=True):
        self.use_cap, self.use_lock, self.ring = use_cap, use_lock, ring

    def _E(self, T):
        return np.hstack([T["E"], T["RING"]]) if self.ring else T["E"]

    def _experts(self, T, y):
        return _fit_ae(T["LP"][y == 0]), _fit_et(self._E(T), y)

    def _raw(self, experts, T):
        ae, et = experts
        return np.column_stack([_ae_score(ae, T["LP"]), _et_score(et, self._E(T))])

    def fit(self, T, y):
        y = np.asarray(y)
        n = len(y)
        # (1) 전문가 점수를 '자기가 학습에 쓰이지 않은' 상태로 얻는다: 4등분해서 3으로 학습, 1을 채점 (OOF)
        #     이렇게 하지 않으면 ET가 학습 파일을 외워 점수가 부풀고, 그 위의 로지스틱이 잘못 배운다
        raw = np.empty((n, 2))
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(np.arange(n), y):
            raw[b] = self._raw(self._experts(take(T, a), y[a]), take(T, b))
        # (2) 정상 기준 z: (점수 − 정상 중앙값) / (정상 90%선 − 정상 중앙값). 1 = 정상의 90%선
        nz = raw[y == 0]
        self.med_ = np.median(nz, 0)
        d = np.abs(np.percentile(nz, 90, 0) - self.med_)
        self.sc_ = np.where(d > 1e-9, d, nz.std(0)) + 1e-6
        self.oof_z_ = (raw - self.med_) / self.sc_
        # (3) 결합: z 두 개 → 로지스틱 회귀 (계수 2개 + 절편)
        self.lr_ = LogisticRegression(C=1.0, class_weight="balanced").fit(self.oof_z_, y)
        self.cap_ = float(np.median(self.lr_.decision_function(self.oof_z_)[y == 0]))
        # (4) 전문가를 학습 파일 전체로 다시 학습 (새 파일 채점용)
        self.experts_ = self._experts(T, y)
        # (5) 규칙의 기준값: 학습 정상만으로
        Kn = T["K"][y == 0][:, :2]
        self.kmed_ = np.median(Kn, 0)
        self.kmad_ = 1.4826 * np.median(np.abs(Kn - self.kmed_), 0) + 1e-12
        self.lock_th_ = float(T["S"][y == 0].max() + LOCK_MARGIN)
        return self

    def parts(self, T):
        """파일별 점수와 규칙 판정 표"""
        raw = self._raw(self.experts_, T)
        Z = (raw - self.med_) / self.sc_
        lr = self.lr_.decision_function(Z)
        r = np.sqrt((((T["K"][:, :2] - self.kmed_) / self.kmad_) ** 2).mean(1))
        spec83 = (T["P83"] >= P83_MIN) & (T["W83"] >= W83_MIN) & (T["K83"] > K83_MIN)
        res83 = (T["RES"] >= RES_FIT_MIN) & self.ring
        rule83 = spec83 | res83
        knock = (r >= R_MIN) & (T["K"][:, 2] >= RATIO_MIN)
        body = (T["RING"][:, 1] >= BODY_MIN) & (T["NIMP"] >= BODY_NMIN) & self.ring
        lock = T["S"] >= self.lock_th_
        score = lr + 100.0 * rule83 + 100.0 * knock
        if self.use_lock:
            score = score + 100.0 * lock
        if self.use_cap:
            score = np.where((T["n20"] == 1) & (score < 50), np.minimum(score, self.cap_), score)
        # 본체 모드 내림: 충격이 거의 모두 본체 모드(약 7 kHz)로 울리면 정상 OOF 중앙값 이하로. 규칙으로 켜진 파일(+100)은 그대로
        body_cap = body & (score < 50)
        score = np.where(body_cap, np.minimum(score, self.cap_), score)
        return pd.DataFrame(dict(ae_z=Z[:, 0], et_z=Z[:, 1], lr=lr, P83=T["P83"], W83=T["W83"], K83=T["K83"], spec83=spec83,
                                 RES=T["RES"], res83=res83, rule83=rule83, r=r, K3=T["K"][:, 2], knock=knock,
                                 n_imp=T["NIMP"], ring24=T["RING"][:, 0], ring_body=T["RING"][:, 1], body=body_cap,
                                 n20=T["n20"], S=T["S"], lock_obs=lock, score=score))


# ---------------------------------------------------------------------------
# 4. 최종 모델: 임계값까지
# ---------------------------------------------------------------------------
def _inner_job(T, y, tr, te, kw):
    return te, Core(**kw).fit(take(T, tr), y[tr]).parts(take(T, te))["score"].values


class FinalModel:
    """학습 파일 전체로 학습하고, 새 신호를 0(정상)/1(고장)로 판정한다

    임계값: thr_fixed(기본 FIXED_THR = 1.0, 11-17)로 고정한다. thr_fixed=None이면 예전 정책 = 학습 파일을 4등분한 OOF 점수에서
    F2가 최대인 값(같으면 낮은 쪽)을 분할 seed마다 구해 중앙값. 고정이어도 seed별 값은 구해서 그 최소–최대 사이를 '경계'로 표시한다
    """

    def __init__(self, use_cap=False, use_lock=False, ring=True, thr_seeds=(0, 1, 2), n_jobs=-1, thr_fixed=FIXED_THR):
        self.use_cap, self.use_lock, self.ring, self.thr_seeds, self.n_jobs = use_cap, use_lock, ring, tuple(thr_seeds), n_jobs
        self.thr_fixed = thr_fixed

    def fit(self, X, y, names=None):
        y = np.asarray(y)
        self.names_ = list(names) if names is not None else [f"train_{i:03d}" for i in range(len(y))]
        T, self.cols_ = signal_tables(X)
        kw = dict(use_cap=self.use_cap, use_lock=self.use_lock, ring=self.ring)
        idx = np.arange(len(y))
        jobs = [(s, tr, te) for s in self.thr_seeds
                for tr, te in StratifiedGroupKFold(4, shuffle=True, random_state=s).split(idx, y, idx)]
        out = Parallel(n_jobs=self.n_jobs)(delayed(_inner_job)(T, y, tr, te, kw) for _, tr, te in jobs)
        self.oof_ = np.empty((len(self.thr_seeds), len(y)))
        for (s, _, _), (te, sc) in zip(jobs, out):
            self.oof_[self.thr_seeds.index(s), te] = sc
        self.thr_all_ = np.array([best_f2_threshold(y, o) for o in self.oof_])
        self.thr_ = float(np.median(self.thr_all_)) if self.thr_fixed is None else float(self.thr_fixed)
        self.core_ = Core(**kw).fit(T, y)
        self.y_ = y
        return self

    def predict_table(self, X, names):
        """새 신호 → 파일별 판정표 (pred = 0/1, 근거 열 포함)"""
        T, _ = signal_tables(X, self.cols_)
        D = self.core_.parts(T)
        D.insert(0, "file", list(names))
        D["threshold"] = self.thr_
        D["pred"] = (D["score"] >= self.thr_).astype(int)
        lo, hi = self.thr_all_.min(), self.thr_all_.max()
        D["borderline"] = (D["score"] >= lo) & (D["score"] < hi) if hi > lo else False
        D["single_hit_obs"] = D["n20"] == 1
        # 규칙 문턱까지의 거리: 두 조건 중 덜 채운 쪽의 비율 (1 이상 = 규칙 켜짐). 0.8–1.0이면 문턱 20% 이내 = 확인 대상
        D["rule83_ratio"] = np.minimum(np.minimum(D["P83"] / P83_MIN, D["W83"] / W83_MIN), D["K83"] / K83_MIN)
        if self.ring:                                   # 공진 규칙도 8.3 kHz 규칙의 한 갈래 → 둘 중 문턱에 가까운 쪽
            D["rule83_ratio"] = np.maximum(D["rule83_ratio"], D["RES"] / RES_FIT_MIN)
        D["knock_ratio"] = np.minimum(D["r"] / R_MIN, D["K3"] / RATIO_MIN)
        D["rule_near"] = (np.maximum(D["rule83_ratio"], D["knock_ratio"]) >= 0.8) & ~D["rule83"] & ~D["knock"]
        # 본체 모드 내림 문턱 근처(비율 0.72–0.9, 충격 3개 이상)에서 고장 쪽으로 판정된 파일 = 내림이 걸릴 뻔함
        D["body_near"] = (self.ring & (D["n_imp"] >= BODY_NMIN) & (D["ring_body"] >= 0.8 * BODY_MIN) & ~D["body"]
                          & (D["ring_body"] < BODY_MIN))
        D["reason"] = [self._reason(r) for r in D.itertuples()]
        return D

    def _reason(self, r):
        why = []
        if r.spec83:
            why.append(f"8.3 kHz 모드 규칙 (돌출 {r.P83:.2f} dec, 폭 {r.W83:.0f} Hz, 충격성 K {r.K83:.1f})")
        if r.res83:
            why.append(f"8.3 kHz 공진 규칙 (8.0–8.7 kHz 깨끗한 공진 충격 fit {r.RES:.2f})")
        if r.body:
            why.append(f"본체 모드 내림 (충격 {r.n_imp:.0f}개 중 {r.ring_body:.0%}가 5.5–7.75 kHz)")
        if r.knock:
            why.append(f"유격 반복 규칙 (정상 중심 거리 {r.r:.1f}, 2위/1위 {r.K3:.2f})")
        if not (r.rule83 or r.knock):
            side = "≥" if r.pred else "<"
            why.append(f"결합 점수 {r.lr:+.2f} {side} 임계값 {self.thr_:.2f} (스펙트럼 z {r.ae_z:+.1f}, 충격 z {r.et_z:+.1f})")
        if r.borderline:
            why.append("경계: 임계값 seed에 따라 판정이 바뀜")
        if r.rule_near:
            why.append(f"[확인] 규칙 문턱 20% 이내 (8.3 kHz {r.rule83_ratio:.2f}, 유격 {r.knock_ratio:.2f}; 1 이상이면 켜짐)")
        if r.body_near and r.pred:
            why.append(f"[확인] 본체 모드 내림 문턱 20% 이내 (본체 울림 {r.ring_body:.2f}, 0.9 이상이면 내림)")
        if r.lock_obs:
            why.append(f"[관찰] 주기 잠금 점수 {r.S:.2f} ≥ {self.core_.lock_th_:.2f}")
        if r.single_hit_obs:
            why.append("[관찰] 큰 타격 정확히 1개")
        return " · ".join(why)


# ---------------------------------------------------------------------------
# 제출 파일
# ---------------------------------------------------------------------------
def _name_key(v):
    """파일 이름 표기 통일: 앞뒤 공백 · 폴더 경로 · 확장자 · 대소문자 차이를 없앤다 ('val/VAL_001.CSV' → 'val_001')"""
    t = str(v).strip().replace(chr(92), "/").rsplit("/", 1)[-1].lower()
    return t[:-4] if t.endswith(".csv") else t


def _trailing_int(k):
    """이름 끝의 숫자 ('val_001' → 1, '001' → 1). 없으면 None"""
    d = ""
    for ch in reversed(k):
        if not ch.isdigit():
            break
        d = ch + d
    return int(d) if d else None


LABEL_NAMES = ("label", "labels", "target", "pred", "prediction", "class", "y", "라벨", "예측", "정답")


def write_submission(D, out_path, template=None, name_col="file", label_col="label"):
    """판정표 D(file·pred 열) → 제출 CSV

    template: 과제에서 준 제출 양식 CSV 경로. 주면 그 파일의 열 이름·행 순서·파일 이름 표기를 그대로 두고 라벨 열만 채운다.
      - 이름 열 = 값이 우리 파일 이름과 가장 많이 겹치는 열 (확장자·대소문자·경로 차이는 무시, 번호만 있는 양식은 번호로 맞춘다)
      - 라벨 열 = 나머지가 한 열이면 그 열, 여러 열이면 이름이 label·target 등인 열 또는 label_col로 지정한 열
      - 양식의 이름이 예측에 하나라도 없으면 멈춘다 (틀린 파일을 조용히 만들지 않는다)
    template이 없으면 name_col·label_col 두 열로 쓴다 — 11/19 양식 공개 전의 임시 형식
    """
    pred = {_name_key(f): int(v) for f, v in zip(D["file"], D["pred"])}
    if len(pred) != len(D):
        raise ValueError("판정표의 파일 이름이 겹친다")
    if template is None:
        sub = pd.DataFrame({name_col: list(D["file"].astype(str)), label_col: list(D["pred"].astype(int))})
        sub.to_csv(out_path, index=False, encoding="utf-8")
        return sub

    sub, used = None, None
    for enc in ("utf-8-sig", "cp949"):
        try:
            sub = pd.read_csv(template, dtype=str, keep_default_na=False, encoding=enc)   # 문자열로 읽어 '001' 같은 표기를 지킨다
            used = enc
            break
        except UnicodeDecodeError:
            continue
    if sub is None:
        raise ValueError(f"제출 양식을 읽을 수 없다 (utf-8 · cp949 모두 실패): {template}")

    hits = {c: sub[c].map(_name_key).isin(pred).mean() for c in sub.columns}
    ncol = max(hits, key=hits.get)
    keys = sub[ncol].map(_name_key)
    if hits[ncol] < 1.0:
        # 번호만 적힌 양식 ('001', '1'): 이름 끝 번호가 양쪽에서 겹치지 않고 모두 대응될 때만 번호로 맞춘다
        by_num = {_trailing_int(k): v for k, v in pred.items()}
        nums = [{c: sub[c].map(_name_key).map(_trailing_int)} for c in sub.columns]
        cand = [(c, n) for d in nums for c, n in d.items() if n.notna().all() and n.is_unique and set(n) <= set(by_num)]
        if None in by_num or len(by_num) != len(pred) or not cand:
            missing = sorted(set(keys) - set(pred))[:5]
            raise ValueError(f"양식의 파일 이름 중 예측에 없는 것이 있다 (열 {ncol}): {missing}")
        ncol, n = cand[0]
        values = n.map(by_num)
    else:
        if not keys.is_unique:
            raise ValueError(f"양식의 파일 이름이 겹친다 (열 {ncol})")
        values = keys.map(pred)
    others = [c for c in sub.columns if c != ncol]
    if len(others) == 1:
        lcol = others[0]
    else:
        named = [c for c in others if c.strip().lower() in LABEL_NAMES or c == label_col]
        if len(named) != 1:
            raise ValueError(f"라벨 열을 정할 수 없다. 양식의 열: {list(sub.columns)} → write_submission(label_col=...)로 지정한다")
        lcol = named[0]
    sub[lcol] = values.astype(int).values
    sub.to_csv(out_path, index=False, encoding="utf-8-sig" if used == "utf-8-sig" and _has_bom(template) else ("cp949" if used == "cp949" else "utf-8"))
    return sub


def _has_bom(path):
    with open(path, "rb") as f:
        return f.read(3) == bytes([0xEF, 0xBB, 0xBF])



# ===========================================================================
# J-축소 (11-22 기준 구성, EXP-050): 제출용 — 신호 배열만으로 동작
# ===========================================================================
# 계산식은 실험 코드(src.models2 TwoStack + RingShrink, src.exp2util.j_shrink)와 한 줄씩 같다.
# 다른 점은 AE 점수를 파일 번호(experts.ae_mean) 대신 신호에서 바로 계산하는 것뿐이다 (같은 식 · 같은 seed).
# 같다는 것은 단계/7_최종·제출/검증_동등성.py 가 확인한다 (EXP-050 J-축소 전체 학습 모델과 결합 입력 · 계수 · 문턱 · val 판정 비교).
#
#   판정 = [ET 안쪽 OOF logit, 박자 z, AE z] → 로지스틱 ≥ 문턱  OR  L2 위 꼬리 규칙(E2, 하중 대비 고주파)  OR  8.3k 규칙(스펙트럼 OR 공진)
#   ET 입력 = H 피처 13열(울림 비율 H7–H10은 충격 수로 축소) + 가장 솟은 좁은 봉우리 2열(학습 정상 기준)
#   11-25 이름 변경(값 같음): H18_hfl → L2_hfl, L1_S → L1_lock (j_tables 배열 순서는 그대로라 옛 캐시 npz도 그대로 읽힌다)
J_FIXED = ["H1_nimp", "H2_ratio21", "H3_kurt_1-2k", "H5_kurt_4-8k", "H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k",
           "H10_ring_8-12.5k", "H12_share_1-2k", "H13_share_2-4k", "H14_share_4-8k", "H15_share_8-12.5k", "L2_hfl"]   # 4-7 최종피처.json "fixed" (옛 H18_hfl)
J_PEAK = ["H16_peak_h", "H17_peak_w"]                                    # 4-7 "hfold" (학습 정상 중앙 log PSD 기준)
J_RING = ["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]
J_EXTRA = {"D_K1": "DK1",                                                # J_FIXED 밖의 열 → j_tables 키 (11-25 EXP-064, JShrink cols로 고른다)
           "max_2_4k": "MAX24", "max_8_11k": "MAX811"}                   # 대역별 최대 충격 배수 (11-25 4-19 · EXP-065)
J_EXTRA.update({c: "B15_" + c for c in ["C1_kenv_1.6-4.5k", "C1_kc_1.6-4.5k", "C1_kc_2-4k", "H4_kurt_2-4k", "C2_tk_1.6-4.5k", "C2_tk_2-4k",
                                         "C3_w1_all", "C3_w2_all", "C3_w1_imp", "C3_w2_imp", "C4_ib_1.6-4.5k", "C4_ib_2-4k"]})   # 1–5 kHz 후보 C1–C4 (11-25 4-20 · EXP-067, features2.band15_table)
J_ALPHA = 3.0                                                            # 울림 비율 축소 세기 (EXP-050, 실행 전 고정)
J_ET_N = 300                                                             # ET 나무 수 (full 모드와 같음)
J_Q_E2 = 99.0                                                            # E2 문턱 = 학습 정상 L2_hfl의 99 % 분위


def _peak83(Xc):
    """평균 제거 신호 → 8.20–8.45 kHz 봉우리 돌출(dec) · 반높이 폭(Hz). signal_tables · exp2util.rule83_flags와 같은 식"""
    f, P8 = welch(Xc, fs=FS, nperseg=8192)
    L = np.log10(P8)
    df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
    PR = Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest")
    band = (f >= 8200) & (f < 8450)
    ia = np.flatnonzero(band)
    P83, W83 = PR[:, band].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, band].argmax()]
        h = PR[i, k] / 2
        l = r = k
        while l > 0 and PR[i, l - 1] >= h:
            l -= 1
        while r < PR.shape[1] - 1 and PR[i, r + 1] >= h:
            r += 1
        W83[i] = f[r] - f[l]
    return P83, W83


def j_tables(X, fmax=None, events=False, atoms="250", res_new=False, lock_tol=0.03):
    """신호 (n, 50000) → J 입력 묶음 dict. 라벨을 쓰지 않고 파일마다 독립 (학습 · val · test에 똑같이)
    H (n,13) 고정 H 피처(J_FIXED 순서) · S (n,) 박자(L1_lock) · L (n,m) 1 kHz 위 log PSD(봉우리용) · df · LP (n,8,1025) AE 입력 · 8.3k 규칙 재료
    DK1 (n,) D_K1 (features2.d_k1, 11-25 EXP-064. JShrink cols에 "D_K1"이 있을 때만 쓴다)
    MAX24 · MAX811 (n,) 2–4k · 8–11k로 운 충격 중 최대 배수 log10(1 + x) (features2.max_ring_x, 11-25 EXP-065. cols에 "max_2_4k" · "max_8_11k"가 있을 때만)
    B15_{열} (n,) 1–5 kHz 후보 C1–C4 (features2.band15_table, 11-25 4-20 · EXP-067. cols에 그 열 이름이 있을 때만 쓴다)
    fmax: None = 기본. 숫자(11-25, 11000) = 상한 변형: H · S · L은 feat2_table(fmax), AE 입력은 fmax 이하 bin만. 8.3k 규칙 재료는 그대로
    검출기 다듬기 (11-25, 2-36 · EXP-063, 기본값 = 예전과 같음)
      events: True = H1 · H2 · H7–H10(축소 n 포함) · L1 hit를 25 ms 이벤트 단위로 (R1)
      atoms: "250"(기본) | "50" = H7–H10 울림 주파수를 50 Hz 사전으로 (R2)
      res_new: True = RES를 새 검출기(fmax 입력 · events 설정 그대로) + 50 Hz 사전으로 (R3). False = 예전 검출기 ring_table
    lock_tol: L1_lock(S) hit 허용 비율 (기본 0.03 = ±3 %, 예전과 같음. 11-25 EXP-066 ±7 % 비교)"""
    from src.features2 import feat2_table, d_k1, max_ring_x, band15_table
    from src.impacts import all_impacts, res_new as _res_new
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != N_SAMPLES:
        raise ValueError(f"신호 길이는 {N_SAMPLES}점이어야 한다. 받은 모양 {X.shape}")
    imps = all_impacts(X, fmax=fmax, events=events, atoms=atoms)
    T, L, fr = feat2_table(X, imps=imps, fmax=fmax, lock_tol=lock_tol)
    Xc = X - X.mean(1, keepdims=True)
    Hp = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)        # AE 입력 (signal_tables의 LP와 같음)
    starts = np.linspace(0, X.shape[1] - FS, 8).astype(int)
    fa, P = welch(np.stack([Hp[:, s:s + FS] for s in starts], 1), fs=FS, nperseg=2048, axis=-1)
    if fmax is not None:
        P = P[..., fa <= fmax]
    P83, W83 = _peak83(Xc)
    RES = _res_new(imps) if res_new else ring_table(X)[1]
    return dict(H=T[J_FIXED].values.astype(float), S=T["L1_lock"].values.astype(float), L=np.asarray(L, float), df=float(fr[1] - fr[0]),
                LP=np.log10(P + 1e-20), P83=P83, W83=W83, K83=kurt83(Xc), RES=RES, DK1=d_k1(imps),
                MAX24=max_ring_x(imps, 2000, 4000), MAX811=max_ring_x(imps, 8000, 11000),
                **{"B15_" + c: v.values for c, v in band15_table(X, [im["idx"] for im in imps]).items()})


def _lg(p):
    p = np.clip(p, 1 / 600, 1 - 1 / 600)
    return np.log(p / (1 - p))


class _RZ:
    """학습 정상 기준 강건 z (src.features2.robust_z와 같은 식, 저장 가능한 클래스)"""
    def __init__(self, v_normal):
        v = np.asarray(v_normal, float)
        self.med = float(np.median(v))
        sc = 1.4826 * np.median(np.abs(v - self.med))
        if sc < 1e-9:
            sc = (np.percentile(v, 90) - self.med) / 1.2816
            if sc < 1e-9:
                sc = float(np.std(v))
        self.sc = float(sc + 1e-9)

    def __call__(self, v):
        return (np.asarray(v, float) - self.med) / self.sc


class _PeakRef:
    """가장 솟은 좁은 봉우리 (src.features2.PeakRef와 같은 식). fit = 학습 정상 중앙 log PSD"""
    def __init__(self, df):
        self.df = df

    def fit(self, L_normal):
        self.ref_ = np.median(L_normal, axis=0)
        return self

    def transform(self, L):
        out = []
        for d in np.asarray(L) - self.ref_:
            k = int(d.argmax()); h = d[k]; lo = hi = k
            while lo > 0 and d[lo - 1] >= h / 2:
                lo -= 1
            while hi < len(d) - 1 and d[hi + 1] >= h / 2:
                hi += 1
            out.append((h, (hi - lo + 1) * self.df))
        return np.array(out)


def _j_et(seed=0):
    return ExtraTreesClassifier(J_ET_N, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt",
                                random_state=seed, n_jobs=1)


class JShrink:
    """J-축소 본체. fit(Tb, y) · parts(Tb) — Tb = j_tables(X)
    cols: ET에 넣을 H 열 (기본 J_FIXED). J_FIXED의 일부 + J_EXTRA(예 "D_K1"). H1_nimp · L2_hfl은 축소 · E2에 쓰므로 꼭 넣는다.
          울림 비율 축소는 cols에 남은 J_RING 열에만. peak: 봉우리 열 (기본 J_PEAK, [] = 봉우리 안 씀). 11-25 EXP-064, 기본 = 예전과 같음"""

    def __init__(self, alpha=J_ALPHA, seed=0, cols=None, peak=None):
        self.alpha, self.seed = alpha, seed
        self.cols = list(J_FIXED if cols is None else cols)
        self.peak = list(J_PEAK if peak is None else peak)
        self.ri_ = [self.cols.index(c) for c in J_RING if c in self.cols]
        self.i1_, self.i18_ = self.cols.index("H1_nimp"), self.cols.index("L2_hfl")

    def _H(self, Tb):
        """Tb → cols 순서의 H 행렬 (getattr: 옵션 추가 전에 저장한 모델도 읽힘)"""
        cols = getattr(self, "cols", J_FIXED); H = np.asarray(Tb["H"], float)
        if list(cols) == J_FIXED:
            return H
        return np.column_stack([H[:, J_FIXED.index(c)] if c in J_FIXED else np.asarray(Tb[J_EXTRA[c]], float) for c in cols])

    def _shrink(self, H):
        """울림 비율 H7–H10 → (k + α·p0) / (n + α). n = 충격 수, k = 그 대역으로 울린 충격 수. 충격 0개 파일은 그대로(−1)"""
        H = np.array(H, float); n = np.rint(np.expm1(H[:, self.i1_])); has = n > 0
        k = H[np.ix_(has, self.ri_)] * n[has, None]
        H[np.ix_(has, self.ri_)] = (k + self.alpha * self.p0_) / (n[has, None] + self.alpha)
        return H

    def _et_input(self, Tb):
        Hs = self._shrink(self._H(Tb))
        peak = getattr(self, "peak", J_PEAK)
        if not peak:
            return Hs, Hs
        pk = self.pr_.transform(Tb["L"])
        return np.hstack([Hs, pk[:, [["H16_peak_h", "H17_peak_w"].index(p) for p in peak]]]), Hs

    def fit(self, Tb, y):
        y = np.asarray(y); H = self._H(Tb); nn = np.rint(np.expm1(H[:, self.i1_])); m = (y == 0) & (nn > 0)
        # (1) 울림 비율 축소의 정상 기준 p0: 학습 정상의 모든 충격에서 대역별 비율
        self.p0_ = (H[np.ix_(m, self.ri_)] * nn[m, None]).sum(0) / nn[m].sum()
        # (2) 봉우리 기준(학습 정상 중앙 log PSD) → ET 입력
        self.pr_ = _PeakRef(Tb["df"]).fit(Tb["L"][y == 0]) if self.peak else None
        Xe, Hs = self._et_input(Tb)
        # (3) 박자 z
        self.zf_ = _RZ(Tb["S"][y == 0]); z = self.zf_(Tb["S"])
        # (4) ET · AE 점수를 '자기가 학습에 안 쓰인' 상태로: 4등분 OOF
        n = len(y); oof, ae = np.empty(n), np.empty(n); LP = Tb["LP"]
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(Xe, y):
            oof[b] = _lg(_j_et(self.seed).fit(Xe[a], y[a]).predict_proba(Xe[b])[:, 1])
            ae[b] = _ae_score(_fit_ae(LP[a][y[a] == 0], self.seed), LP[b])
        self.aef_ = _RZ(ae[y == 0])
        self.Z_ = np.column_stack([oof, z, self.aef_(ae)]); self.y_ = y
        # (5) 결합: 로지스틱 + 문턱(학습 점수에서 F2 최대)
        self.c_ = LogisticRegression(C=1.0, class_weight="balanced").fit(self.Z_, y)
        self.tc_ = float(best_f2_threshold(y, self.c_.decision_function(self.Z_)))
        # (6) E2 문턱: 학습 정상 L2_hfl의 99 % 분위
        v = Hs[y == 0, self.i18_]; q, med = np.percentile(v, J_Q_E2), np.median(v)
        self.t18_ = float(med + (q - med))
        # (7) 새 파일 채점용: ET · AE를 학습 전체(AE는 정상 전체)로 다시
        self.et_ = _j_et(self.seed).fit(Xe, y)
        self.ae_ = _fit_ae(LP[y == 0], self.seed)
        return self

    def parts(self, Tb):
        Xe, Hs = self._et_input(Tb)
        a = _lg(self.et_.predict_proba(Xe)[:, 1]); z = self.zf_(Tb["S"]); aez = self.aef_(_ae_score(self.ae_, Tb["LP"]))
        s = self.c_.decision_function(np.column_stack([a, z, aez]))
        e2 = (Hs[:, self.i18_] > self.t18_) & (Hs[:, self.i1_] >= np.log(2) - 1e-9)
        spec83 = (Tb["P83"] >= P83_MIN) & (Tb["W83"] >= W83_MIN) & (Tb["K83"] > K83_MIN); res83 = Tb["RES"] >= RES_FIT_MIN
        main = s >= self.tc_
        return pd.DataFrame(dict(et_logit=a, beat_z=z, ae_z=aez, score=s, margin=s - self.tc_, comb=main, L2_hfl=Hs[:, self.i18_],
                                 n_imp=np.rint(np.expm1(Hs[:, self.i1_])).astype(int), E2=e2, P83=Tb["P83"], W83=Tb["W83"], K83=Tb["K83"],
                                 RES=Tb["RES"], spec83=spec83, res83=res83, rule83=spec83 | res83, pred=(main | e2 | spec83 | res83).astype(int)))


class FinalJ:
    """제출용 J-축소: fit(X, y, names) → predict_table(X, names) (pred = 0/1 + 근거). 임계값 = 결합 문턱 tc_(학습 점수에서 F2 최대, J와 같음)"""

    def __init__(self, alpha=J_ALPHA, seed=0, fmax=None, events=False, atoms="250", res_new=False, cols=None, peak=None, lock_tol=0.03):
        self.alpha, self.seed, self.fmax = alpha, seed, fmax                 # fmax: j_tables 상한 변형 (None = 기본)
        self.events, self.atoms, self.res_new = events, atoms, res_new       # 검출기 다듬기 R1–R3 (11-25, 기본 = 예전)
        self.cols, self.peak = cols, peak                                    # JShrink 열 구성 (11-25 EXP-064, None = 기본)
        self.lock_tol = lock_tol                                             # L1_lock hit 허용 비율 (11-25 EXP-066, 기본 0.03 = 예전)

    def _tables(self, X):
        g = lambda k, d: getattr(self, k, d)                                 # getattr: 옵션 추가 전에 저장한 모델도 읽힘
        return j_tables(X, g("fmax", None), g("events", False), g("atoms", "250"), g("res_new", False), g("lock_tol", 0.03))

    def fit(self, X, y, names=None):
        y = np.asarray(y)
        self.names_ = list(names) if names is not None else [f"train_{i:03d}" for i in range(len(y))]
        self.core_ = JShrink(self.alpha, self.seed, getattr(self, "cols", None), getattr(self, "peak", None)).fit(self._tables(X), y)
        self.y_ = y
        self.thr_ = self.core_.tc_
        return self

    def predict_table(self, X, names):
        D = self.core_.parts(self._tables(X))
        D.insert(0, "file", list(names))
        D["reason"] = [self._reason(r) for r in D.itertuples()]
        return D

    def _reason(self, r):
        why = []
        if r.spec83:
            why.append(f"8.3 kHz 모드 규칙 (돌출 {r.P83:.2f} dec, 폭 {r.W83:.0f} Hz, 충격성 K {r.K83:.1f})")
        if r.res83:
            why.append(f"8.3 kHz 공진 규칙 (깨끗한 공진 충격 fit {r.RES:.2f})")
        if r.E2:
            why.append(f"L2 위 꼬리 규칙(E2) (하중 대비 고주파 L2_hfl {r.L2_hfl:.2f} > 정상 99 % {self.core_.t18_:.2f}, 충격 {r.n_imp}개)")
        side = "≥" if r.comb else "<"
        why.append(f"결합 {r.score:+.2f} {side} 문턱 {self.core_.tc_:.2f} (여유 {r.margin:+.2f}; ET logit {r.et_logit:+.1f} · 박자 z {r.beat_z:+.1f} · AE z {r.ae_z:+.1f})")
        return " · ".join(why)
