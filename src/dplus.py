"""D+ 피처와 AE+ET 결합 모델 (EXP-036, 휴대폰 분석 15·16·17·18번에서 정리)

D+ (81열) = ext150을 상관 군집(|ρ| ≥ 0.85)으로 줄인 대표 62열 + 울림 4열(ring24·ring_body·res83·nimp)
            + 파일 단위 물리 10열(정렬 평균 울림 f·ζ·피크/RMS, 스펙트럼 첨도 4대역, PSD 기울기 2, log RMS)
            + 규칙 기반 5열(P83·W83·K1·K2·K3: 8.3 kHz 봉우리와 유격 반복 규칙이 쓰는 연속 수치)
- 변환(분위수)·군집은 학습 파일로만 fit하고 라벨은 쓰지 않는다. 새 파일은 변환만 한다.
- AEET: F = [파일 번호, 피처…]. 전문가 AE(src.experts.ae_mean, 파일 번호로 동작) + ET(피처) → 안쪽 4겹 OOF 점수 → 로지스틱 결합.
        rules=True면 8.3 kHz · 유격 규칙이 켜진 파일 점수 +100 (EXP-025와 같은 규칙).
"""
import copy

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, stft, welch
from scipy.spatial.distance import squareform
from sklearn.base import BaseEstimator
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer, StandardScaler

from src.features import feature_table_ext
from src.final import (P83_MIN, R_MIN, RATIO_MIN, W83_MIN, signal_tables)

FS = 25000
_SOS200 = butter(4, 200, btype="highpass", fs=FS, output="sos")
_SOS1K = butter(4, 1000, btype="highpass", fs=FS, output="sos")
_SOSLP = butter(4, 500, btype="lowpass", fs=FS, output="sos")
_PRE, _POST, _K, _MARGIN = 50, 500, 10, int(0.02 * FS)
PHYS_COLS = ["ring_f", "ring_zeta", "ring_peak_rms", "SK_2-4k", "SK_5.5-7.75k", "SK_8-8.7k", "SK_9-10k", "slope_lo", "slope_hi", "log_rms"]
RULE_COLS = ["P83", "W83", "K1", "K2", "K3"]
RING_COLS = ["ring24", "ring_body", "res83", "nimp"]


def _pick(x):
    env = sosfiltfilt(_SOSLP, np.abs(hilbert(sosfiltfilt(_SOS1K, x))))
    pk, _ = find_peaks(env, distance=125)
    pk = pk[(pk >= _MARGIN + _PRE) & (pk < len(x) - _MARGIN - _POST)]
    top = pk[np.argsort(env[pk])[::-1][:_K]]
    return np.array(sorted({p - 25 + int(np.argmax(np.abs(x[p - 25:p + 26]))) for p in top}))


def _ring_mean(x):
    """충격 정렬 평균(상위 10개)의 울림 주파수·감쇠비·피크/RMS"""
    segs = [x[q - _PRE:q + _POST] * np.sign(x[q]) for q in _pick(x) if q - _PRE >= 0 and q + _POST <= len(x)]
    if not segs:
        return np.nan, np.nan, np.nan
    m = np.mean(segs, 0)
    seg = m[_PRE:_PRE + 375] * np.hanning(750)[375:]
    Fq, fr = np.abs(np.fft.rfft(seg, 8192)), np.fft.rfftfreq(8192, 1 / FS)
    mk = (fr > 800) & (fr < 12000)
    f0 = fr[mk][np.argmax(Fq[mk])]
    a = np.abs(hilbert(m))[_PRE + 25:_PRE + 200]
    sl = np.polyfit(np.arange(len(a)) / FS, np.log(a + 1e-12), 1)[0]
    return f0, (-sl / (2 * np.pi * f0) if sl < 0 else np.nan), abs(m[_PRE]) / x.std()


def physical10(X):
    """파일 단위 물리 10열 (라벨 안 씀)"""
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(1, keepdims=True)
    HP = sosfiltfilt(_SOS200, Xc, axis=-1)
    RM = np.array([_ring_mean(HP[i]) for i in range(len(X))])
    fz, _, Zs = stft(Xc, fs=FS, nperseg=512, noverlap=256, boundary=None, padded=False, axis=-1)
    P2 = np.abs(Zs) ** 2
    SK = (P2 ** 2).mean(-1) / (P2.mean(-1) ** 2 + 1e-30) - 2
    fw, Pw = welch(Xc, fs=FS, nperseg=2048, axis=-1)

    def slope(lo, hi):
        m = (fw >= lo) & (fw < hi)
        lf = np.log10(fw[m])
        return np.array([np.polyfit(lf, np.log10(Pw[i, m] + 1e-30), 1)[0] for i in range(len(X))])

    sk = lambda a, b: np.log10(SK[:, (fz >= a * 1000) & (fz <= b * 1000)].mean(1) + 1)
    return pd.DataFrame({"ring_f": RM[:, 0], "ring_zeta": np.log10(RM[:, 1] + 1e-4), "ring_peak_rms": RM[:, 2],
                         "SK_2-4k": sk(2, 4), "SK_5.5-7.75k": sk(5.5, 7.75), "SK_8-8.7k": sk(8, 8.7), "SK_9-10k": sk(9, 10),
                         "slope_lo": slope(200, 2000), "slope_hi": slope(2000, 12000), "log_rms": np.log10(Xc.std(1))}).fillna(0)


def _blocks(T, X):
    """신호 표 묶음 T → (울림 4열, 규칙 기반 5열, 물리 10열) DataFrame 셋"""
    ring = pd.DataFrame({"ring24": T["RING"][:, 0], "ring_body": T["RING"][:, 1], "res83": T["RES"], "nimp": T["NIMP"]})
    rule = pd.DataFrame({"P83": T["P83"], "W83": T["W83"], "K1": T["K"][:, 0], "K2": T["K"][:, 1], "K3": T["K"][:, 2]})
    return ring, rule, physical10(X)


class DPlus:
    """D+ 피처 변환기. fit(학습 신호) → D_ (n, 81). transform(새 신호) → (D, T) (T = 신호 표 묶음, AE 입력 LP 포함)"""

    def fit(self, X, F_ext=None):
        self.T_, self.cols_ = signal_tables(X)
        F = F_ext if F_ext is not None else feature_table_ext(X)
        F = F.loc[:, F.nunique() > 3]
        self.keep_ = list(F.columns)
        self.qt_ = QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(F.values)
        Z = self.qt_.transform(F.values)
        C = np.abs(pd.DataFrame(Z).corr(method="spearman").fillna(0).values)
        cl = fcluster(linkage(squareform(1 - C, checks=False), "average"), 0.15, "distance")
        self.reps_ = [int(np.where(cl == c)[0][np.argmax(C[np.ix_(cl == c, cl == c)].mean(1))]) for c in np.unique(cl)]
        ring, rule, phys = _blocks(self.T_, X)
        self.q_ = {k: QuantileTransformer(n_quantiles=150, output_distribution="normal", random_state=0).fit(d.values)
                   for k, d in (("ring", ring), ("phys", phys), ("rule", rule))}
        self.names_ = [self.keep_[i] for i in self.reps_] + RING_COLS + PHYS_COLS + RULE_COLS
        self.D_ = np.hstack([Z[:, self.reps_], self.q_["ring"].transform(ring.values), self.q_["phys"].transform(phys.values),
                             self.q_["rule"].transform(rule.values)])
        return self

    def transform(self, X):
        T, _ = signal_tables(X, self.cols_)
        F = pd.DataFrame(T["E"], columns=self.cols_)[self.keep_]
        Z = self.qt_.transform(F.values)
        ring, rule, phys = _blocks(T, X)
        return np.hstack([Z[:, self.reps_], self.q_["ring"].transform(ring.values), self.q_["phys"].transform(phys.values),
                          self.q_["rule"].transform(rule.values)]), T


def rule_flags(T_new, T_tr, y):
    """새 신호 표 → (8.3 kHz 스펙트럼 규칙, 유격 반복 규칙) 불리언. 유격의 정상 중심은 학습 정상의 K1·K2 중앙값·MAD (src.final.Core와 같은 정의)"""
    Kn = T_tr["K"][np.asarray(y) == 0][:, :2]
    med = np.median(Kn, 0)
    mad = 1.4826 * np.median(np.abs(Kn - med), 0) + 1e-12
    r = np.sqrt((((T_new["K"][:, :2] - med) / mad) ** 2).mean(1))
    return (T_new["P83"] >= P83_MIN) & (T_new["W83"] >= W83_MIN), (r >= R_MIN) & (T_new["K"][:, 2] >= RATIO_MIN)


def _logit(p):
    p = np.clip(p, 0.01, 0.99)
    return np.log(p / (1 - p))


class AEET(BaseEstimator):
    """F = [파일 번호, 피처…] → 안쪽 4겹 OOF 점수 [AE, ET] → 로지스틱. rules=True면 8.3k·유격 규칙 켜진 파일 +100"""

    def __init__(self, rules=False):
        self.rules = rules

    @staticmethod
    def _et(Xa, ya, Xb):
        et = ExtraTreesClassifier(300, class_weight="balanced_subsample", min_samples_leaf=2, max_features="sqrt", random_state=0, n_jobs=1)
        return _logit(et.fit(Xa, ya).predict_proba(Xb)[:, 1])

    def fit(self, F, yy):
        from src.experts import KnockRule, ae_mean
        F = np.asarray(F)
        idx, Xf = F[:, 0].astype(int), F[:, 1:]
        oof = np.zeros((len(yy), 2))
        for tr, te in StratifiedKFold(4, shuffle=True, random_state=1).split(Xf, yy):
            oof[te, 0] = ae_mean(idx[tr], idx[te])[:, 0]
            oof[te, 1] = self._et(Xf[tr], yy[tr], Xf[te])
        self.sc_ = StandardScaler().fit(oof)
        self.st_ = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(self.sc_.transform(oof), yy)
        self.idx_, self.X_, self.y_ = idx, Xf, yy
        self.rules_ = [copy.deepcopy(KnockRule()).fit(idx), _rule83()] if self.rules else []
        return self

    def expert_scores(self, F):
        """시험 파일의 전문가 점수 (n, 2) = [AE, ET]와 규칙 켜짐 (n,)"""
        from src.experts import ae_mean
        F = np.asarray(F)
        te = F[:, 0].astype(int)
        s = np.c_[ae_mean(self.idx_, te)[:, 0], self._et(self.X_, self.y_, F[:, 1:])]
        on = np.any([r(te) for r in self.rules_], axis=0) if self.rules_ else np.zeros(len(te), bool)
        return s, on

    def decision_function(self, F):
        s, on = self.expert_scores(F)
        return self.st_.decision_function(self.sc_.transform(s)) + 100.0 * on


def _rule83():
    from src.experts import Rule83
    return Rule83()
