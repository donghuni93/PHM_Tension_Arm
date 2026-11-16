"""피처 추출. 피처의 물리 의미는 단계/4_피처/README.md, 가설 연결은 notes/물리가설.md.

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


# ---------------------------------------------------------------------------
# 물리 가설 피처 (EXP-006). 가설 id는 notes/물리가설.md
# 대역·격자는 데이터를 보기 전에 정한 고정값만 쓴다 (라벨 보고 고르지 않음)
# ---------------------------------------------------------------------------
from scipy.signal import welch

WELCH_NPERSEG = 4096            # Δf ≈ 6.1 Hz
SHAPE_GRID_HZ = 250             # H3 스펙트럼 모양 격자
PEAKY_BAND_HZ = 1250            # H2 공진 날카로움 대역 (P02와 같은 10칸)
SWAY_LP_HZ = 20                 # H4 저주파 흔들림 상한
REV_LP_HZ = 5                   # H4 반전점 검출용 저역통과
IMPACT_HP_HZ = 1000             # H4 충격 포락선용 고역통과
REV_WIN_S = 0.03                # H4 반전점 ±30 ms


def extract_features_phys(x, fs=25000):
    """신호 1개 → 물리 가설 피처 dict (H2 공진 날카로움, H3 스펙트럼 모양, H4 반전 충격)"""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    feats = {}

    # --- H2·H3: Welch 로그 PSD ---
    f, P = welch(x, fs=fs, nperseg=WELCH_NPERSEG)
    logP = np.log10(P + 1e-30)
    # H3 모양: 격자별 평균 파워의 로그 − 파일 전체 평균 (게인 제거)
    edges = np.arange(0, fs / 2 + 1e-9, SHAPE_GRID_HZ)
    shape = np.array([np.log10(P[(f >= a) & (f < b)].mean()) for a, b in zip(edges[:-1], edges[1:])])
    shape -= shape.mean()
    for a, v in zip(edges[:-1], shape):
        feats[f"H3_shape_{int(a):05d}"] = v
    # H2 공진 날카로움: 대역 안 최대 로그 PSD − 대역 중앙값 (좁은 공진 피크가 클수록 큼, 게인 무관)
    for a in np.arange(0, fs / 2, PEAKY_BAND_HZ):
        m = (f >= max(a, FILTER_LOW_HZ)) & (f < a + PEAKY_BAND_HZ)
        feats[f"H2_peaky_{int(a / 100):03d}"] = logP[m].max() - np.median(logP[m])

    # --- H4: 하중 반전점의 충격 ---
    sway = sosfiltfilt(butter(4, SWAY_LP_HZ, btype="lowpass", fs=fs, output="sos"), x)
    rev = sosfiltfilt(butter(2, REV_LP_HZ, btype="lowpass", fs=fs, output="sos"), x)
    xh = sosfiltfilt(butter(4, IMPACT_HP_HZ, btype="highpass", fs=fs, output="sos"), x)
    env = np.abs(hilbert(xh))
    k = int(0.0025 * fs)                                  # 2.5 ms 이동평균으로 포락선 평활
    env = np.convolve(env, np.ones(k) / k, mode="same")
    rms = np.sqrt(np.mean(x ** 2))
    feats["H4_sway_rms_ratio"] = np.sqrt(np.mean(sway ** 2)) / rms
    feats["H4_sway_p2p"] = (rev.max() - rev.min()) / rms
    prom = 0.5 * rev.std()
    pk, _ = find_peaks(rev, prominence=prom, distance=int(0.1 * fs))
    tr, _ = find_peaks(-rev, prominence=prom, distance=int(0.1 * fs))
    ext = np.sort(np.concatenate([pk, tr]))
    w = int(REV_WIN_S * fs)
    env_mean = env.mean()
    if len(ext):
        near = np.array([env[max(e - w, 0):e + w].max() for e in ext]) / env_mean
        feats["H4_rev_count"] = len(ext)
        feats["H4_rev_impact_mean"] = near.mean()          # 반전점 근처 충격 / 평균 포락선
        feats["H4_rev_impact_max"] = near.max()
    else:
        feats["H4_rev_count"] = 0
        feats["H4_rev_impact_mean"] = 1.0
        feats["H4_rev_impact_max"] = 1.0
    # 반전점 집중도: 반전점 근처 포락선 최댓값 / 신호 전체 포락선 최댓값 (1이면 가장 큰 충격이 반전점에서)
    feats["H4_rev_impact_share"] = (feats["H4_rev_impact_max"] * env_mean) / env.max()
    # H1 보조: 충격 횟수 (포락선이 중앙값의 8배를 넘는 봉우리, 10 ms 간격)
    imp, _ = find_peaks(env, height=8 * np.median(env), distance=int(0.01 * fs))
    feats["H1_impact_rate"] = len(imp) / (len(x) / fs)
    return feats


def feature_table_phys(X, fs=25000):
    """신호 배열 → 물리 가설 피처 표. 행 순서는 X와 같다"""
    return pd.DataFrame([extract_features_phys(x, fs) for x in X])


# ---------------------------------------------------------------------------
# 충격 반복·단발 피처 (EXP-016). 정의는 라벨을 보기 전에 고정 (11-14)
# 충격 = 평균 제거 → 200 Hz 고역통과 h에서 |h| > 8σ (σ = MAD × 1.4826), 10 ms 안은 한 사건 (2-6·EDA 뷰어와 같음)
# ---------------------------------------------------------------------------
def extract_features_impact(x, fs=25000):
    """B (H4·H1): 충격이 일정한 주기로 반복되는가, 저주파 흔들림의 정점(반전점)에 맞춰 나는가
    E (H5): 큰 사건 한 번인가 (두 번째 사건 높이 비), 큰 사건 뒤 얼마나 오래 울리나 (잔향 비)"""
    xc = x - x.mean()
    h = sosfiltfilt(butter(4, 200, btype="highpass", fs=fs, output="sos"), xc)
    xl = sosfiltfilt(butter(4, 20, btype="lowpass", fs=fs, output="sos"), xc)
    sig = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if len(idx):
        groups = np.split(idx, np.flatnonzero(np.diff(idx) > int(0.01 * fs)) + 1)
        ev = np.array([g[np.argmax(np.abs(h[g]))] for g in groups])
    else:
        ev = np.array([int(np.abs(h).argmax())])
    hgt = np.sort(np.abs(h[ev]) / sig)[::-1]
    f = {}
    # B1 포락선 주기성: 포락선(1 kHz로 평균) 자기상관의 최댓값, 지연 50 ms – 1 s
    env = np.abs(hilbert(h))
    m = len(env) // 25 * 25
    e = env[:m].reshape(-1, 25).mean(1); e = e - e.mean()
    ac = np.correlate(e, e, "full")[len(e) - 1:]; ac = ac / (ac[0] + 1e-12)
    lo, hi = 50, 1000                                     # 1 kHz 표본 → ms
    k = lo + int(np.argmax(ac[lo:hi]))
    f["B1_env_acf_peak"] = float(ac[k])
    f["B2_env_acf_lag_s"] = k / 1000
    # B3 충격 간격 규칙성: 간격의 변동계수 (사건 3개 미만이면 1 = 불규칙으로 둠)
    t = np.sort(ev) / fs
    d = np.diff(t)
    f["B3_interval_cv"] = float(d.std() / d.mean()) if len(d) >= 2 else 1.0
    # B4 저주파 동기 (H4): 사건 시점의 |저주파| 가 파일 안 |저주파| 분포에서 몇 % 위치인가의 평균 (0.5 = 무관, 1 = 정점에서)
    al = np.abs(xl)
    f["B4_low_sync"] = float(np.mean([(al < al[i]).mean() for i in ev])) if len(idx) else 0.5
    f["B5_low_frac"] = float(xl.var() / (xc.var() + 1e-20))
    # E1 사건 수, E2 두 번째 / 첫 번째 사건 높이 (단발이면 작음)
    f["E1_n_impact_log"] = float(np.log1p(len(groups) if len(idx) else 0))
    f["E2_top2_ratio"] = float(hgt[1] / hgt[0]) if len(hgt) >= 2 else 0.0
    # E3 가장 큰 사건의 잔향 비 (2-6 정의: 최고점 뒤 5–20 ms 포락선 중앙값 / 최고값). 2-6에서 그림을 본 뒤 정의 → 탐색용
    p = int(ev[np.argmax(np.abs(h[ev]))])
    tail = env[p + int(0.005 * fs):p + int(0.02 * fs)]
    # 사건이 신호 끝(마지막 5 ms 안)이라 뒤를 볼 수 없으면 −1 (= 잔향 비 0.1, 2-6 정상 중앙값 근처)
    f["E3_big_ringing_log"] = float(np.log10(np.median(tail) / (env[p] + 1e-12) + 1e-4)) if len(tail) >= 25 else -1.0
    f["E4_big_height_log"] = float(np.log10(hgt[0]))
    return f


def feature_table_impact(X, fs=25000):
    """신호 배열 → 충격 반복·단발 피처 표 (EXP-016). 행 순서는 X와 같다"""
    return pd.DataFrame([extract_features_impact(x, fs) for x in X])


# ---------------------------------------------------------------------------
# 200 Hz 고정 격자 대역 비율 (EXP-017, 2-10과 같은 정의). 라벨 없이 계산, 칸 선택은 폴드 안에서
# ---------------------------------------------------------------------------
GRID200 = np.arange(200, 12600, 200)            # 200–400, …, 12200–12400 Hz → 61칸


def feature_table_band200(X, fs=25000):
    """평균 제거 → FFT 파워 → 200 Hz 칸 파워 / 200 Hz 이상 전체 파워, log10. 열 이름 G{시작 Hz:05d}
    + 사후 정의 열 P08200_08400 (8.2–8.4 kHz, 2-10에서 그림을 본 뒤 정한 대역 — 비교용)"""
    Xc = X - X.mean(1, keepdims=True)
    pw = np.abs(np.fft.rfft(Xc, axis=1)) ** 2
    fq = np.fft.rfftfreq(X.shape[1], 1 / fs)
    tot = pw[:, fq >= 200].sum(1)
    out = {f"G{lo:05d}": pw[:, (fq >= lo) & (fq < lo + 200)].sum(1) / tot for lo in GRID200[:-1]}
    out["P08200_08400"] = pw[:, (fq >= 8200) & (fq < 8400)].sum(1) / tot
    return pd.DataFrame({k: np.log10(v + 1e-20) for k, v in out.items()})


# ---------------------------------------------------------------------------
# 유격 반복 · 하중 대비 고주파 (2-12, EXP-025). 라벨 없이 계산
# ---------------------------------------------------------------------------
def extract_features_knock(x, fs=25000):
    """K1 유격 반복 = log10(두 번째로 큰 사건 / σ), K2 하중 대비 고주파 = log10(저주파(<20 Hz) RMS / σ),
    K3 2위/1위 사건 높이. σ = 200 Hz 고역통과 신호의 1.4826·MAD, 사건 = |h| > 8σ, 10 ms 안은 한 사건.
    2-12 run.py의 D1·D6·R과 같은 정의 (두 번째 사건이 없으면 K1 = log10(8), K3 = 0)"""
    xc = x - x.mean()
    h = sosfiltfilt(butter(4, 200, btype="highpass", fs=fs, output="sos"), xc)
    lo = sosfiltfilt(butter(4, 20, btype="lowpass", fs=fs, output="sos"), xc)
    sig = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    if len(idx):
        gs = np.split(idx, np.flatnonzero(np.diff(idx) > int(0.01 * fs)) + 1)
        hg = np.sort([np.abs(h[g]).max() / sig for g in gs])[::-1]
    else:
        hg = np.array([])
    return {"K1_second_snr": float(np.log10(hg[1])) if len(hg) > 1 else float(np.log10(8)),
            "K2_load_to_hf": float(np.log10(lo.std() / sig)),
            "K3_top2_ratio": float(hg[1] / hg[0]) if len(hg) > 1 else 0.0}


def feature_table_knock(X, fs=25000):
    """신호 배열 → 유격 반복 표 (K1–K3). 행 순서는 X와 같다"""
    return pd.DataFrame([extract_features_knock(x, fs) for x in X])
