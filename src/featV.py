"""V 피처 (11-27, 5_VAL_단계 V-5): 우리 모델(H 열)에 더할 후보. 전처리는 우리 것 그대로 — 1–11 kHz 대역통과 신호(src.impacts.highpass(x, fmax=11000))
라벨을 쓰지 않고 파일마다 독립으로 계산한다 (train · val · test에 똑같이).

- seg_kurt: 0.1초 구간마다 첨도(가우스 잡음 = 3)를 재서 평균 → 시간에 흩어진 뾰족한 사건 (효섭 A_segk · D_segk의 우리 전처리 판)
- ar_whiten: AR(p)로 앞 p점에서 예측되는 성분(지속음 · 공진 웅웅거림)을 빼고 남은 잔차 (효섭 ar_whiten과 같은 식, 자기상관만 FFT)
- spike_counts: 짧게 튀는 사건(검출 신호 |w| > k·σ, σ = MAD/0.6745, 간격 10 ms) 중
    작은 것(우리 충격 기준 포락선/바탕 < 4.5배) · 그중 뒤에 울림이 없는 것(뒤 1–7 ms 에너지 / 앞 −7–−1 ms 에너지 < 2)의 수
"""
import numpy as np
from scipy import linalg, signal, stats

from src.impacts import _baseline, _sos

FS = 25000
SEG = 2500                 # 0.1초
TRIM = 500                 # 양끝 20 ms 자름 (필터 끝단)
P_AR = 40
SMOOTH_HZ = 1000           # 우리 검출기 포락선 다듬기와 같음


def seg_kurt(x, L=SEG):
    s = x[:len(x) // L * L].reshape(-1, L)
    return float(stats.kurtosis(s, axis=1, fisher=False).mean())


def ar_whiten(x, p=P_AR):
    s = x[:20000]; F = np.fft.rfft(s, 65536); r = np.fft.irfft(F * np.conj(F))[:p + 1]
    a = linalg.solve_toeplitz(r[:p], r[1:p + 1])
    out = signal.lfilter(np.r_[1, -a], [1], x); out[:p] = 0.0
    return out


def spike_counts(xh, det, k=8.0, small_hx=4.5, ring_ratio=2.0):
    """xh: 1–11 kHz 신호(우리 전처리), det: 검출에 쓸 신호(xh 자체 또는 백색화). 반환 (사건 수, 작은 사건 수, 울림 없는 작은 사건 수)"""
    d = det[TRIM:-TRIM]; sig = np.median(np.abs(d)) / 0.6745
    pk, _ = signal.find_peaks(np.abs(d), height=k * sig, distance=250); pk = pk + TRIM
    e = signal.sosfiltfilt(_sos("lowpass", SMOOTH_HZ, FS), np.abs(signal.hilbert(xh))); hx = e / _baseline(e)
    small = noring = 0
    for p in pk:
        if p < 300 or p + 300 > len(xh):
            continue
        if hx[p] < small_hx:
            small += 1
            post = np.mean(xh[p + 25:p + 175] ** 2); pre = np.mean(xh[p - 175:p - 25] ** 2) + 1e-20
            noring += int(post / pre < ring_ratio)
    return len(pk), small, noring


def v_row(xh):
    """1–11 kHz 신호 하나 → V 후보 dict (백색화 있음 · 없음 둘 다)"""
    w = ar_whiten(xh)
    n_r, s_r, nr_r = spike_counts(xh, xh)
    n_w, s_w, nr_w = spike_counts(xh, w)
    c = xh[TRIM:-TRIM]; cw = w[TRIM:-TRIM]
    return dict(segk=np.log(seg_kurt(c)), segk_w=np.log(seg_kurt(cw)),
                spike_noring=np.log1p(nr_r), spike_noring_w=np.log1p(nr_w),
                spike_all=np.log1p(n_r), spike_all_w=np.log1p(n_w))


def w_segk_mean(X, fs=FS, p=P_AR, seg=SEG, trim=TRIM):
    """백색화 구간 첨도 (11-27, 5_VAL_단계 V-3 · 3-3 정의 그대로, EXP-068 ET 열 V1). 신호 (n, 50000) → (n,)
    평균 제거 → 1–11 kHz 대역통과(버터워스 4차, 앞뒤) → AR(p) 백색화(계수 = 앞 20000점 자기상관, 앞 p점 버림) → 양끝 trim점 자름
    → 0.1초(seg점) 구간 첨도(가우스 = 3) 평균 → log10. 위 ar_whiten(앞 p점을 0으로 채움 · 자르지 않음)과 다르다"""
    X = np.asarray(X, float)
    X0 = X - X.mean(1, keepdims=True)
    S = signal.sosfiltfilt(signal.butter(4, [1000, 11000], "bandpass", fs=fs, output="sos"), X0, axis=1)
    out = []
    for s in S:
        r = signal.fftconvolve(s[:20000], s[:20000][::-1], "full")[19999:19999 + p + 1]
        a = linalg.solve_toeplitz(r[:p], r[1:p + 1])
        w = signal.lfilter(np.r_[1, -a], [1], s)[p:][trim:-trim]
        k = stats.kurtosis(w[:len(w) // seg * seg].reshape(-1, seg), axis=1, fisher=False)
        out.append(np.log10(k.mean()))
    return np.array(out)
