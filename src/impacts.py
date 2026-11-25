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


def highpass(x, fs=FS, f=HP_HZ, fmax=None):
    """평균 제거 → f Hz 고역 (1차원). fmax를 주면 f–fmax Hz 대역통과(4차, 11-25 11 kHz 상한 변형)"""
    x = np.asarray(x, dtype=float)
    sos = _sos("highpass", f, fs) if fmax is None else _sos("bandpass", (f, fmax), fs)
    return sosfiltfilt(sos, x - x.mean())


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


def impact_rings_v2(x, fs=FS, imp=None, _hp=None, atoms="250"):
    """충격마다 충격 뒤 8 ms를 감쇠 사인파 사전(주파수 1–12 kHz × ζ 4개)에 투영 → (f, ζ, fit) 배열 (충격 수, 3)
    atoms: "250" = final._ring_atoms(250 Hz 간격, 기본) · "50" = _atoms50(50 Hz 간격, 11-25 R2: 1.9 kHz ζ도 맞음, 2-36)
    끝단 10 ms 제외 덕에 8 ms가 항상 들어간다. 충격 0개면 (0, 3)"""
    xh = highpass(x, fs) if _hp is None else _hp
    imp = detect_impacts(x, fs, _hp=xh) if imp is None else imp
    pk = [p for p in imp["idx"] if p + _SEG <= len(xh)]
    if not pk:
        return np.zeros((0, 3))
    A, par = _atoms() if atoms == "250" else _atoms50z()
    S = np.stack([xh[p:p + _SEG] for p in pk])
    S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, A), axis=2)
    b = fit.argmax(1)
    return np.column_stack([par[b, 0], par[b, 1], fit.max(1)])


EVENT_MS = 25                # 25 ms 묶음 (11-25 R1, 2-36 ④-b): 길게 우는 충격 하나가 여러 번 검출되는 것을 하나로


def merge_events(imp, fs=FS, gap_ms=EVENT_MS):
    """검출 dict → 이벤트 dict (11-25 R1). 이웃 검출 간격 ≤ gap_ms면 같은 이벤트(사슬로 이어 묶음).
    이벤트의 t · idx · height_x · height_g · rings = 이벤트 안에서 height_x가 가장 큰 검출의 값. n_det = 묶기 전 검출 수"""
    idx = np.asarray(imp["idx"]); out = dict(imp); out["n_det"] = len(idx)
    if len(idx) == 0:
        return out
    o = np.argsort(idx, kind="stable"); cl = np.r_[0, np.cumsum(np.diff(idx[o]) > gap_ms * fs / 1000)]
    hx = np.asarray(imp["height_x"])[o]
    pick = np.array([o[np.flatnonzero(cl == c)[hx[cl == c].argmax()]] for c in np.unique(cl)])
    for k in ("t", "height_x", "height_g", "idx"):
        out[k] = np.asarray(imp[k])[pick]
    if "rings" in imp:
        assert len(imp["rings"]) == len(idx), "울림 행 수 ≠ 검출 수"
        out["rings"] = np.asarray(imp["rings"])[pick]
    return out


def all_impacts(X, fs=FS, fmax=None, events=False, atoms="250"):
    """신호 배열 (n, N) → 파일마다 dict(t, height_x, height_g, idx, rings (k,3), hp 신호)
    fmax: None = 1 kHz 고역(기본) · 숫자 = 1 kHz–fmax 대역통과를 검출 · 울림 · hp 입력으로 (검출 상수는 그대로)
    events: True = 25 ms 이벤트 단위(merge_events, 11-25 R1) · atoms: 울림 사전 "250"(기본) | "50"(11-25 R2)"""
    out = []
    for x in np.asarray(X, dtype=float):
        xh = highpass(x, fs, fmax=fmax)
        imp = detect_impacts(x, fs, _hp=xh)
        imp["rings"] = impact_rings_v2(x, fs, imp=imp, _hp=xh, atoms=atoms)
        imp["hp"] = xh
        out.append(merge_events(imp, fs) if events else imp)
    return out


def res_new(imps, fs=FS):
    """8.3 kHz 공진 규칙 입력 RES를 새 검출기로 (11-25 R3): 파일마다 imps의 충격(또는 이벤트) 뒤 8 ms를 50 Hz 사전에 투영,
    울림 8.0–8.7 kHz · ζ ≤ 0.02인 것의 최대 fit (없으면 0). final.ring_table의 RES와 같은 조건, 검출기 · 사전만 다르다"""
    res = np.zeros(len(imps))
    for i, im in enumerate(imps):
        r = impact_rings_v2(None, fs, imp=im, _hp=im["hp"], atoms="50")
        m = (r[:, 0] >= 8000) & (r[:, 0] < 8700) & (r[:, 1] <= 0.02)
        if m.any():
            res[i] = r[m, 2].max()
    return res


# ---- 울림 모드 피처 (2-14, 11-23): 라벨 없이 찾은 울림 모드 골짜기로 나눈 대역 ----
# 연결부 1.6–4.5k · 본체 5.5–8k · 고주파 8–12.5k. 공유 대역(1.0–1.6k · 4.5–5.5k)은 쓰지 않는다
# 에너지 비의 연결부 아래 경계 2.2k = 1.9k 모드의 위쪽 끝(정상도 우는 모드를 뺀다)
RM_NO_IMPACT = -3.0          # 충격 없는 파일의 에너지 비 (라벨 없는 상수, H7–H10의 −1과 같은 역할)


@lru_cache(None)
def _atoms50():
    """감쇠 사인파 사전 1–12 kHz 50 Hz 간격 × ζ 4개 (2-14 · 2-29와 같음) → (atoms, 주파수)"""
    t = np.arange(_SEG) / FS; A, P = [], []
    for f in np.arange(1000, 12001, 50):
        for z in (0.01, 0.02, 0.04, 0.08):
            w = 2 * np.pi * f; wd, a = w * np.sqrt(1 - z ** 2), np.exp(-z * w * t)
            Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1)); A.append(Q); P.append(f)
    return np.stack(A), np.array(P, float)


@lru_cache(None)
def _atoms50z():
    """_atoms50과 같은 사전, 매개변수 (f, ζ) 2열 (impact_rings_v2의 atoms="50"용). ζ 순서는 _atoms50의 안쪽 고리와 같다"""
    A, P = _atoms50()
    return A, np.column_stack([P, np.tile((0.01, 0.02, 0.04, 0.08), len(P) // 4)])


def ring_mode_feats(x, fs=FS, top=20):
    """신호 1개 → dict
    RM_n_joint   연결부 모드(1.6–4.5 kHz)로 운 충격 수 (울림 주파수 = 충격 뒤 8 ms의 50 Hz 사전 맞춤)
    RM_e_joint   log10 중앙값 √(E 2.2–4.5k ÷ E 5.5–8k), 큰 충격 top개의 직후 12 ms (소린 비율의 우리 경계판)
    RM_e_high    log10 중앙값 √(E 8–12.5k ÷ E 5.5–8k), 같은 방식
    RM_n_joint22 연결부를 1.9k 공유 모드 끝(2.2k)부터 센 충격 수 2.2–4.5 kHz (EXP-059, 11-23: RM_e_joint와 같은 아래 경계)"""
    xh = highpass(x, fs); imp = detect_impacts(x, fs, _hp=xh); w12 = int(.012 * fs)
    pk = np.array([p for p in imp["idx"] if p + w12 <= len(xh)], int)
    if not len(pk):
        return dict(RM_n_joint=0.0, RM_e_joint=RM_NO_IMPACT, RM_e_high=RM_NO_IMPACT, RM_n_joint22=0.0)
    A, P = _atoms50()
    S = np.stack([xh[p:p + _SEG] for p in pk]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    f = P[np.linalg.norm(np.einsum("ks,ast->kat", S, A), axis=2).argmax(1)]
    h = np.interp(pk, imp["idx"], imp["height_g"]); sel = pk[np.argsort(h)[::-1][:top]]
    fr = np.fft.rfftfreq(w12, 1 / fs)
    Pw = np.abs(np.fft.rfft(np.stack([xh[p:p + w12] for p in sel]) * np.hanning(w12), axis=1)) ** 2
    e = lambda lo, hi: Pw[:, (fr >= lo) & (fr < hi)].sum(1) + 1e-30
    return dict(RM_n_joint=float(((f >= 1600) & (f < 4500)).sum()),
                RM_e_joint=float(np.log10(np.median(np.sqrt(e(2200, 4500) / e(5500, 8000))))),
                RM_e_high=float(np.log10(np.median(np.sqrt(e(8000, 12500) / e(5500, 8000))))),
                RM_n_joint22=float(((f >= 2200) & (f < 4500)).sum()))


def ring_mode_table(X, fs=FS):
    import pandas as pd
    return pd.DataFrame([ring_mode_feats(x, fs) for x in np.asarray(X, float)])
