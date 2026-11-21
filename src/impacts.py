"""충격 정의 (2차 설계, 11-21 확정): 단계/4_피처/4-6_1121_충격재정의/사령탑_7b-7e 의 detect_impacts + 파일 앞뒤 10 ms 제외

평균 제거 → 1 kHz 고역(4차 Butterworth 양방향) → |Hilbert| 포락선 → 1000 Hz 저역 다듬기
→ 바탕 = 100 ms 이동 중앙값(1 kHz로 솎아 계산 후 선형 보간) → 포락선/바탕 ≥ 4.5 인 봉우리, 간격 ≥ 6.75 ms
→ (이 파일에서 추가) 파일 앞뒤 10 ms 안의 봉우리는 제외 (필터 끝단 효과)
라벨을 쓰지 않는다. 기존 src/final.py의 impact_rings(예전 정의 B)는 그대로 두고, 여기는 새 정의 전용이다.
"""
from functools import lru_cache

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, find_peaks, hilbert, sosfiltfilt

from src.final import _SEG, _ring_atoms

FS = 25000
SMOOTH_HZ, K_MIN, DIST_MS, BASE_WIN_MS, HP_HZ = 1000, 4.5, 6.75, 100, 1000
EDGE_MS = 10


@lru_cache(None)
def _sos(kind, f, fs):
    return butter(4, f, btype=kind, fs=fs, output="sos")


def highpass(x, fs=FS, f=HP_HZ):
    """평균 제거 → f Hz 고역 (1차원)"""
    x = np.asarray(x, dtype=float)
    return sosfiltfilt(_sos("highpass", f, fs), x - x.mean())


def _baseline(e, fs=FS, win_ms=BASE_WIN_MS):
    dec = fs // 1000
    m = median_filter(e[::dec], size=int(round(win_ms)) | 1, mode="reflect")
    return np.interp(np.arange(len(e)), np.arange(0, len(e), dec), m)


def detect_impacts(x, fs=FS, edge_ms=EDGE_MS, _hp=None):
    """한 신호(1차원) → 충격 dict. 끝단 제외 후 결과

    t (s, 충격 시점) · height_x (포락선/바탕 배수) · height_g (포락선 절대 높이, g) · idx (표본 번호)
    edge_ms = 0 이면 끝단 제외 전 (사령탑_7b-7e 충격표와 같은 결과)
    """
    xh = highpass(x, fs) if _hp is None else _hp
    e = sosfiltfilt(_sos("lowpass", SMOOTH_HZ, fs), np.abs(hilbert(xh)))
    b = _baseline(e, fs)
    pk = find_peaks(e / b, height=K_MIN, distance=max(1, int(round(DIST_MS * fs / 1000))))[0]
    edge = int(round(edge_ms * fs / 1000))
    if edge:
        pk = pk[(pk >= edge) & (pk < len(x) - edge)]
    return dict(t=pk / fs, height_x=e[pk] / b[pk], height_g=e[pk], idx=pk)


@lru_cache(None)
def _atoms():
    return _ring_atoms()


def impact_rings_v2(x, fs=FS, imp=None, _hp=None):
    """충격마다 충격 뒤 8 ms를 final._ring_atoms 사전(주파수 1–12 kHz × ζ 4개)에 투영 → (f, ζ, fit) 배열 (충격 수, 3)
    끝단 10 ms 제외 덕에 8 ms가 항상 들어간다. 충격 0개면 (0, 3)"""
    xh = highpass(x, fs) if _hp is None else _hp
    imp = detect_impacts(x, fs, _hp=xh) if imp is None else imp
    pk = [p for p in imp["idx"] if p + _SEG <= len(xh)]
    if not pk:
        return np.zeros((0, 3))
    atoms, par = _atoms()
    S = np.stack([xh[p:p + _SEG] for p in pk])
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, atoms), axis=2)
    b = fit.argmax(1)
    return np.column_stack([par[b, 0], par[b, 1], fit.max(1)])


def all_impacts(X, fs=FS):
    """신호 배열 (n, N) → 파일마다 dict(t, height_x, height_g, idx, rings (k,3), hp 신호)"""
    out = []
    for x in np.asarray(X, dtype=float):
        xh = highpass(x, fs)
        imp = detect_impacts(x, fs, _hp=xh)
        imp["rings"] = impact_rings_v2(x, fs, imp=imp, _hp=xh)
        imp["hp"] = xh
        out.append(imp)
    return out
