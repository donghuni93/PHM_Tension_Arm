"""피처 추출. 정의는 vault/10_스펙/features.md 의 core 피처와 같다.

키 이름 앞의 id(T01, F07 …)가 스펙 표의 id다.
train·val·test 모두 이 함수로 계산해야 같은 피처가 된다.
"""
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

N_WINDOWS = 10                  # W01 서브윈도우 수 (0.2초씩)
PEAK_MIN_HZ = 20                # F07·F08 탐색 하한 (드리프트 제외)
BANDS = {                       # B01–B03 물리 대역 (Hz), decisions D4
    "B01_band_low_ratio": (0, 2000),
    "B02_band_mid_ratio": (2000, 5000),
    "B03_band_high_ratio": (5000, None),  # None = 나이퀴스트까지
}


def _hjorth_mobility(x):
    """Mobility = sqrt(var(x') / var(x)). x'는 인접 샘플 차분"""
    return np.sqrt(np.var(np.diff(x)) / np.var(x))


def extract_features(x, fs=25000):
    """신호 1개(1차원 배열, 단위 g) → core 피처 13개 dict"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()                      # 평균 제거 (MA는 미적용, D3)
    n = len(x)

    # --- 시간 영역 ---
    rms = np.sqrt(np.mean(x ** 2))
    dx = np.diff(x)
    tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]  # Teager-Kaiser 에너지 연산자
    win_rms = np.array([np.sqrt(np.mean(w ** 2)) for w in np.array_split(x, N_WINDOWS)])

    # --- 주파수 영역 (전체 FFT, 분해능 fs/n = 0.5 Hz) ---
    spec = np.abs(np.fft.rfft(x))         # 진폭 스펙트럼 |X(f)|
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    power = spec ** 2                     # 에너지 = 진폭 제곱
    total_power = power.sum()
    m = freqs >= PEAK_MIN_HZ
    k_peak = np.argmax(spec[m])

    feats = {
        "T01_rms": rms,
        "T02_peak": np.max(np.abs(x)),
        "T07_kurtosis": kurtosis(x, fisher=False),      # 가우시안 = 3
        "T08_skewness": skew(x),
        "T15_hjorth_complexity": _hjorth_mobility(dx) / _hjorth_mobility(x),
        "T16_tkeo_max": tkeo.max(),
        "W01_win_rms_max_to_mean": win_rms.max() / win_rms.mean(),
        "F01_spectral_centroid": (freqs * spec).sum() / spec.sum(),
        "F07_peak1_freq": freqs[m][k_peak],
        "F08_peak1_amp": 2 * spec[m][k_peak] / n,        # 단측 진폭 (g)
    }
    for name, (lo, hi) in BANDS.items():
        band = (freqs >= lo) & ((freqs < hi) if hi is not None else True)
        feats[name] = power[band].sum() / total_power
    return feats


def feature_table(X, fs=25000):
    """신호 배열 (파일 수, 샘플 수) → 피처 표 (파일 수 × 13). 행 순서는 X와 같다"""
    return pd.DataFrame([extract_features(x, fs) for x in X])
