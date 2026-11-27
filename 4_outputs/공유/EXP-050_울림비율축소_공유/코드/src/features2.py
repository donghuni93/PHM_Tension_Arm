"""2차 설계 피처 (11-21): H 블록(1 kHz 고역·새 충격 정의) + L1 주기 잠금. 근거 = 단계/5_모델비교/검토문서/모델구성_계획_1121.md 4절

입력은 신호 배열 (n, 50000), 라벨을 쓰지 않고 파일마다 독립이다 (제출 파이프라인에서도 같은 함수).
학습 정상에 의존하는 것(H16–H17 봉우리, 박자 z)은 여기서 계산하지 않는다. 파일별 log PSD 배열만 돌려주고
추정기(fit 안, 학습 행의 정상으로)가 PeakRef·robust_z로 계산한다 → 중첩 CV 폴드 안에서 다시 계산됨.

열 (feat2_table)
  H1_nimp        log1p(충격 수)
  H2_ratio21     2위/1위 충격 높이(바탕 대비 배수). 충격 < 2개면 0
  H3–H6_kurt     옥타브(1–2·2–4·4–8·8–12.5 kHz) 대역통과 신호의 log10(Pearson 첨도) = log10(scipy 초과첨도 + 3), 항상 양수(≥ 0)
  H7–H10_ring    충격 울림 주파수(impact_rings_v2의 f)의 옥타브별 비율. 충격 0개면 −1
  H12–H15_share  Welch(2048점·Hann·50 % 겹침·평균 제거) PSD의 옥타브 파워 ÷ 1 kHz 위 파워 합, log10
  H18_hfl        log10(1 kHz 고역 |Hilbert| 포락선 중앙값 ÷ 평균 제거 신호 20 Hz 저역(4차) RMS)
  L1_S           주기 잠금 S = r_LP × hit (2-13 정의, 충격 시점은 새 정의). 진단용 L1_r · L1_T · L1_hit도 함께
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis

from src.impacts import _sos, all_impacts

FS = 25000
OCT = ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500))
OCT_TAG = ("1-2k", "2-4k", "4-8k", "8-12.5k")
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "cache")

H_FIXED = (["H1_nimp", "H2_ratio21"] + [f"H{3 + i}_kurt_{t}" for i, t in enumerate(OCT_TAG)]
           + [f"H{7 + i}_ring_{t}" for i, t in enumerate(OCT_TAG)]
           + [f"H{12 + i}_share_{t}" for i, t in enumerate(OCT_TAG)] + ["H18_hfl"])
H_FOLD = ["H16_peak_h", "H17_peak_w"]
L_COLS = ["L1_S"]
DIAG = ["L1_r", "L1_T", "L1_hit"]


# ---------------------------------------------------------------------------
def _band_sos(lo, hi):
    if hi >= FS / 2:
        return _sos("highpass", lo, FS)
    return butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")


def octave_kurtosis(x):
    """옥타브 4칸 대역통과 신호의 log10(Pearson 첨도). 마지막 칸(8–12.5 kHz)은 8 kHz 고역 + 12.4 kHz 저역"""
    x = np.asarray(x, dtype=float) - np.mean(x)
    out = []
    for lo, hi in OCT:
        s = sosfiltfilt(_band_sos(lo, 12400 if hi >= 12500 else hi), x)
        out.append(np.log10(kurtosis(s) + 3.0))
    return out


def octave_of(f):
    """주파수 → 옥타브 칸 번호 0–3 (12 kHz까지)"""
    return np.digitize(f, [2000, 4000, 8000])


def welch_log(x):
    """Welch(2048·Hann·50 %·평균 제거) → (f, PSD)"""
    return welch(x, fs=FS, window="hann", nperseg=2048, noverlap=1024, detrend="constant")


def lock_S(x, t, fs=FS):
    """주기 잠금 (2-13): 평균 제거 → 30 Hz 저역(4차) → 50개 중 1개로 솎기(500 Hz) → 지연 0.3–1.2 s 자기상관 최대 r_LP, 그때의 T
    hit = 충격 시점 중 'T ± 3 % 떨어진 다른 충격이 있는' 비율, S = r_LP × hit. 반환 (S, r_LP, T, hit)"""
    d = 50
    s = sosfiltfilt(_sos("lowpass", 30, fs), np.asarray(x, float) - np.mean(x))[::d]
    f2 = fs / d
    ks = np.arange(int(0.3 * f2), int(1.2 * f2) + 1, 2)
    r = np.array([np.corrcoef(s[:len(s) - k], s[k:])[0, 1] for k in ks])
    rl, T = float(r.max()), float(ks[r.argmax()] / f2)
    t = np.asarray(t, float)
    if len(t) < 2:
        return 0.0, rl, T, 0.0
    dd = np.abs(t[None, :] - t[:, None])
    hit = float((np.abs(dd - T) <= 0.03 * T).any(1).mean())
    return rl * hit, rl, T, hit


def feat2_table(X, imps=None):
    """신호 (n, 50000) → (표 DataFrame: H_FIXED + L_COLS + DIAG, 1 kHz 위 log10 PSD (n, m), 그 주파수 (m,))
    imps: all_impacts(X) 결과를 이미 가졌으면 넘긴다"""
    X = np.asarray(X, dtype=float)
    imps = all_impacts(X) if imps is None else imps
    rows, LP = [], []
    for x, im in zip(X, imps):
        h = []
        n = len(im["t"])
        hx = np.sort(im["height_x"])[::-1]
        h += [np.log1p(n), float(hx[1] / hx[0]) if n >= 2 else 0.0]
        h += octave_kurtosis(x)
        if n:
            c = np.bincount(octave_of(im["rings"][:, 0]).astype(int), minlength=4)[:4] / len(im["rings"])
            h += list(c)
        else:
            h += [-1.0] * 4
        f, P = welch_log(x)
        m = f >= 1000
        pw = np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in OCT])
        h += list(np.log10(pw / P[m].sum() + 1e-20))
        env = np.abs(hilbert(im["hp"]))
        rms20 = np.sqrt(np.mean(sosfiltfilt(_sos("lowpass", 20, FS), x - x.mean()) ** 2))
        h.append(float(np.log10(np.median(env) / rms20)))
        S, rl, T, hit = lock_S(x, im["t"])
        rows.append(h + [S, rl, T, hit])
        LP.append(np.log10(P[m] + 1e-20))
    return pd.DataFrame(rows, columns=H_FIXED + L_COLS + DIAG), np.array(LP), f[f >= 1000]


def cached_feat2(tag, X, names=None):
    """캐시 outputs/cache/feat2_{tag}.csv + feat2_psd_{tag}.npz. 피처 함수를 고치면 지우고 다시"""
    p, q = os.path.join(_CACHE, f"feat2_{tag}.csv"), os.path.join(_CACHE, f"feat2_psd_{tag}.npz")
    if os.path.exists(p) and os.path.exists(q):
        z = np.load(q)
        return pd.read_csv(p), z["L"], z["f"]
    T, L, f = feat2_table(X)
    T.to_csv(p, index=False)
    np.savez(q, L=L, f=f)
    return T, L, f


# ---------------------------------------------------------------------------
# 학습 정상에 의존하는 부분 (추정기 fit 안에서 쓴다)
# ---------------------------------------------------------------------------
class PeakRef:
    """H16–H17 가장 솟은 좁은 봉우리. fit(L_normal): 학습 정상의 중앙 log PSD. transform(L): d = L − 중앙
    높이 = max d (dec), 폭 = 그 최대를 포함하는 연속 구간 중 d ≥ 최대/2 인 폭(Hz)"""

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


def robust_z(v_normal):
    """학습 정상의 (중앙값, 1.4826·MAD) → 함수 z(v) = (v − 중앙값)/(1.4826·MAD)"""
    med = np.median(v_normal)
    sc = 1.4826 * np.median(np.abs(v_normal - med))
    if sc < 1e-9:      # S는 정상의 약 79 %가 0이라 MAD = 0 (z가 터짐). 계획에 없는 대체: 정상 90 % 분위수 규모 → 그것도 0이면 표준편차
        sc = (np.percentile(v_normal, 90) - med) / 1.2816
        if sc < 1e-9:
            sc = float(np.std(v_normal))
    sc = sc + 1e-9
    return lambda v: (np.asarray(v, float) - med) / sc


# ---------------------------------------------------------------------------
# 11-22 EXP-047: 8–12.5 kHz 옥타브를 8–10 · 10–12.5 kHz로 쪼갬 (H10 · H15의 대체 후보)
# 근거: 섞인 고장 009 · 018 · 013은 옆 정상보다 8–9 kHz가 높고 11–12.5 kHz가 낮아(4-10 진단) 한 칸으로 합치면 상쇄된다
# 경계 10 kHz는 실행 전 고정(8.3 · 9.5 kHz 금속 모드는 아래 칸, H2′). 라벨 안 씀, 파일마다 독립
# ---------------------------------------------------------------------------
HI_SPLIT = ["H10a_ring_8-10k", "H10b_ring_10-12.5k", "H15a_share_8-10k", "H15b_share_10-12.5k"]


def hi_split_table(X, imps=None):
    """신호 (n, 50000) → DataFrame(HI_SPLIT). 울림 비율 = 충격 울림 주파수가 그 칸인 충격의 비율(충격 0개면 −1),
    파워 몫 = Welch(feat2_table과 같은 설정) 칸 파워 ÷ 1 kHz 위 파워 합, log10"""
    X = np.asarray(X, dtype=float)
    imps = all_impacts(X) if imps is None else imps
    rows = []
    for x, im in zip(X, imps):
        if len(im["t"]):
            fr = im["rings"][:, 0]
            ring = [float(np.mean((fr >= 8000) & (fr < 10000))), float(np.mean(fr >= 10000))]
        else:
            ring = [-1.0, -1.0]
        f, P = welch_log(x)
        tot = P[f >= 1000].sum()
        sh = [float(np.log10(P[(f >= lo) & (f < hi)].sum() / tot + 1e-20)) for lo, hi in ((8000, 10000), (10000, 12500))]
        rows.append(ring + sh)
    return pd.DataFrame(rows, columns=HI_SPLIT)


def cached_hi_split(tag, X):
    """캐시 outputs/cache/feat2split_{tag}.csv. hi_split_table을 고치면 지우고 다시"""
    p = os.path.join(_CACHE, f"feat2split_{tag}.csv")
    if os.path.exists(p):
        return pd.read_csv(p)
    T = hi_split_table(X)
    T.to_csv(p, index=False)
    return T
