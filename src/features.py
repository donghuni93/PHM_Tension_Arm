"""피처 추출. 정의는 vault/10_스펙/features.md 의 core 피처와 같다.

키 이름 앞의 id(T01, F07 …)가 스펙 표의 id다.
train·val·test 모두 이 함수로 계산해야 같은 피처가 된다.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt, spectrogram
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


# ---------------------------------------------------------------------------
# 확장 피처 (EXP-002 천장 확인용). core 13개 + features.md 후보
# 대역은 데이터를 보기 전에 정한 균등 분할만 쓴다 (EDA 대역을 쓰면 누수, decisions D4)
# ---------------------------------------------------------------------------
UNIFORM_BAND_HZ = 625           # B04: 0–12,500 Hz를 20칸
FILTER_BAND_HZ = 1250           # P02·E01·S01: 0–12,500 Hz를 10칸
FILTER_LOW_HZ = 20              # 첫 칸 하한 (6.1 Hz 드리프트 제외, F07과 같은 이유)
STFT_NPERSEG = 1024             # S01 STFT 창 (약 41 ms)


def _zcr(x):
    """영교차율: 이웃한 두 샘플의 부호가 바뀌는 비율"""
    return np.mean(np.signbit(x[1:]) != np.signbit(x[:-1]))


def _filter_bands(fs):
    """(이름, 하한, 상한) 10개. 상한이 나이퀴스트면 None"""
    nyq = fs / 2
    edges = np.arange(0, nyq + 1, FILTER_BAND_HZ)
    bands = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        name = f"{int(lo / 1000 * 10):03d}"          # 예: 000, 012, 025 … (단위 100 Hz)
        bands.append((name, max(lo, FILTER_LOW_HZ), hi if hi < nyq else None))
    return bands


def extract_features_ext(x, fs=25000):
    """신호 1개 → core 13개 + 확장 후보 피처 dict (약 150개)

    core 13개는 extract_features를 그대로 호출해 EXP-001과 같은 값을 보장한다.
    """
    x_raw = np.asarray(x, dtype=float)
    feats = extract_features(x_raw, fs)                # core 13개
    x = x_raw - x_raw.mean()
    n = len(x)

    # --- 시간 영역 (T) ---
    rms = feats["T01_rms"]
    peak = feats["T02_peak"]
    mav = np.mean(np.abs(x))
    tkeo = x[1:-1] ** 2 - x[:-2] * x[2:]
    feats.update({
        "T03_peak_to_peak": x.max() - x.min(),
        "T04_mav": mav,
        "T06_raw_mean": x_raw.mean(),                 # T05 Std는 평균 제거 후 RMS와 같아 생략 (D7)
        "T09_crest": peak / rms,
        "T10_impulse": peak / mav,
        "T11_shape": rms / mav,
        "T12_clearance": peak / np.mean(np.sqrt(np.abs(x))) ** 2,
        "T13_hjorth_activity": np.var(x),
        "T14_hjorth_mobility": _hjorth_mobility(x),
        "T17_tkeo_mean": tkeo.mean(),
        "T18_tkeo_kurtosis": kurtosis(tkeo, fisher=False),
        "T19_tkeo_energy": np.mean(tkeo ** 2),
        "T20_zcr": _zcr(x),
    })

    # --- 서브윈도우 (W) ---
    wins = np.array_split(x, N_WINDOWS)
    win_rms = np.array([np.sqrt(np.mean(w ** 2)) for w in wins])
    feats.update({
        "W02_win_rms_std": win_rms.std(),
        "W03_win_kurtosis_max": max(kurtosis(w, fisher=False) for w in wins),
        "W04_win_zcr_std": np.std([_zcr(w) for w in wins]),
    })

    # --- 주파수 영역 (F) ---
    spec = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(n, d=1 / fs)
    power = spec ** 2
    p = power / power.sum()                           # 정규화 파워 (합 = 1)
    fc = feats["F01_spectral_centroid"]
    cum = np.cumsum(p)
    feats.update({
        "F02_spectral_bandwidth": np.sqrt(np.sum((freqs - fc) ** 2 * p)),
        "F03_rolloff85": freqs[np.searchsorted(cum, 0.85)],
        "F04_rolloff95": freqs[np.searchsorted(cum, 0.95)],
        "F05_spectral_entropy": -np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(p)),
    })
    # F09·F10: 20 Hz 이상에서 서로 10 Hz 넘게 떨어진 피크 중 2·3번째로 큰 것
    m = freqs >= PEAK_MIN_HZ
    pk, _ = find_peaks(spec[m], distance=int(10 / (fs / n)))
    pk = pk[np.argsort(spec[m][pk])[::-1]]
    for rank, k in zip((2, 3), pk[1:3]):
        feats[f"F09_peak{rank}_freq"] = freqs[m][k]
        feats[f"F10_peak{rank}_amp"] = 2 * spec[m][k] / n

    # --- 대역 에너지 (B04 비율, B05 절대) ---
    for lo in np.arange(0, fs / 2, UNIFORM_BAND_HZ):
        band = (freqs >= lo) & (freqs < lo + UNIFORM_BAND_HZ)
        feats[f"B04_ratio_{int(lo):05d}"] = p[band].sum()
    for name, (lo, hi) in BANDS.items():
        band = (freqs >= lo) & ((freqs < hi) if hi is not None else True)
        # 2·|X|²/n² 합 ≈ 그 대역의 평균제곱 (g²)
        feats["B05_abs_" + name.split("_")[2]] = 2 * power[band].sum() / n ** 2

    # --- 대역 필터 후 국소 충격 (P02), 포락선 (E01) ---
    for name, lo, hi in _filter_bands(fs):
        if hi is None:
            sos = butter(4, lo, btype="highpass", fs=fs, output="sos")
        else:
            sos = butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos")
        xf = sosfiltfilt(sos, x)                      # 앞뒤로 필터 → 위상 지연 없음
        f_rms = np.sqrt(np.mean(xf ** 2))
        f_peak = np.max(np.abs(xf))
        env = np.abs(hilbert(xf))                     # 힐베르트 포락선
        feats.update({
            f"P02_rms_{name}": f_rms,
            f"P02_peak_{name}": f_peak,
            f"P02_kurtosis_{name}": kurtosis(xf, fisher=False),
            f"P02_crest_{name}": f_peak / f_rms,
            f"E01_env_rms_{name}": np.sqrt(np.mean(env ** 2)),
            f"E01_env_kurtosis_{name}": kurtosis(env, fisher=False),
        })

    # --- 스펙트로그램 시간축 요약 (S01) ---
    f_s, _, S = spectrogram(x, fs=fs, nperseg=STFT_NPERSEG)
    for name, lo, hi in _filter_bands(fs):
        band = (f_s >= lo) & ((f_s < hi) if hi is not None else True)
        e_t = S[band].sum(axis=0)                     # 시간에 따른 대역 에너지
        feats.update({
            f"S01_max_to_mean_{name}": e_t.max() / e_t.mean(),
            f"S01_cv_{name}": e_t.std() / e_t.mean(),
            f"S01_kurtosis_{name}": kurtosis(e_t, fisher=False),
        })
    return feats


def feature_table_ext(X, fs=25000):
    """신호 배열 (파일 수, 샘플 수) → 확장 피처 표. 행 순서는 X와 같다"""
    return pd.DataFrame([extract_features_ext(x, fs) for x in X])
