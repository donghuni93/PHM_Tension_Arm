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
  L2_hfl         (옛 H18_hfl) log10(1 kHz 고역 |Hilbert| 포락선 중앙값 ÷ 평균 제거 신호 20 Hz 저역(4차) RMS)
  L1_lock        (옛 L1_S) 주기 잠금 S = r_LP × hit (2-13 정의, 충격 시점은 새 정의). 진단용 L1_r · L1_T · L1_hit도 함께
  11-25 이름만 바꿈(값 같음). 옛 이름의 캐시 · json · 목록은 new_name / renamed로 읽는다
D_K1 (d_k1, 11-25 EXP-064): 두 번째로 큰 충격 높이 ÷ 검출기 입력 σ (H1′ 반복 타격). 열 표에는 넣지 않고 j_tables가 따로 낸다
"""
import os

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis

from src.impacts import K_MIN, _sos, all_impacts

FS = 25000
OCT = ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500))
OCT_TAG = ("1-2k", "2-4k", "4-8k", "8-12.5k")
_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "cache")

H_FIXED = (["H1_nimp", "H2_ratio21"] + [f"H{3 + i}_kurt_{t}" for i, t in enumerate(OCT_TAG)]
           + [f"H{7 + i}_ring_{t}" for i, t in enumerate(OCT_TAG)]
           + [f"H{12 + i}_share_{t}" for i, t in enumerate(OCT_TAG)] + ["L2_hfl"])
H_FOLD = ["H16_peak_h", "H17_peak_w"]
L_COLS = ["L1_lock"]
DIAG = ["L1_r", "L1_T", "L1_hit"]
RENAMED = {"H18_hfl": "L2_hfl", "L1_S": "L1_lock"}     # 11-25 옛 이름 → 새 이름 (값 같음)


def new_name(c):
    """옛 열 이름 → 새 이름 (없으면 그대로)"""
    return RENAMED.get(c, c)


def renamed(df):
    """옛 이름 열이 있는 표(캐시 csv 등) → 새 이름"""
    return df.rename(columns=RENAMED)


# ---------------------------------------------------------------------------
def _band_sos(lo, hi):
    if hi >= FS / 2:
        return _sos("highpass", lo, FS)
    return butter(4, [lo, hi], btype="bandpass", fs=FS, output="sos")


def octaves(fmax=None):
    """옥타브 4칸. fmax(11-25 상한 변형)를 주면 마지막 칸 = 8 kHz–fmax"""
    return OCT if fmax is None else OCT[:3] + ((OCT[3][0], fmax),)


def octave_kurtosis(x, fmax=None):
    """옥타브 4칸 대역통과 신호의 log10(Pearson 첨도). 마지막 칸(8–12.5 kHz)은 8 kHz 고역 + 12.4 kHz 저역 (fmax면 8 kHz–fmax 대역통과)"""
    x = np.asarray(x, dtype=float) - np.mean(x)
    out = []
    for lo, hi in octaves(fmax):
        s = sosfiltfilt(_band_sos(lo, 12400 if hi >= 12500 else hi), x)
        out.append(np.log10(kurtosis(s) + 3.0))
    return out


def octave_of(f):
    """주파수 → 옥타브 칸 번호 0–3 (12 kHz까지)"""
    return np.digitize(f, [2000, 4000, 8000])


def welch_log(x):
    """Welch(2048·Hann·50 %·평균 제거) → (f, PSD)"""
    return welch(x, fs=FS, window="hann", nperseg=2048, noverlap=1024, detrend="constant")


def lock_S(x, t, fs=FS, tol=0.03):
    """주기 잠금 (2-13): 평균 제거 → 30 Hz 저역(4차) → 50개 중 1개로 솎기(500 Hz) → 지연 0.3–1.2 s 자기상관 최대 r_LP, 그때의 T
    hit = 충격 시점 중 'T ± tol 떨어진 다른 충격이 있는' 비율, S = r_LP × hit. 반환 (S, r_LP, T, hit)
    tol: 허용 비율 (기본 0.03 = ±3 %, 예전과 같음. 11-25 EXP-066 ±7 % 비교 [사후 값])"""
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
    hit = float((np.abs(dd - T) <= tol * T).any(1).mean())
    return rl * hit, rl, T, hit


def feat2_table(X, imps=None, fmax=None, events=False, atoms="250", lock_tol=0.03):
    """신호 (n, 50000) → (표 DataFrame: H_FIXED + L_COLS + DIAG, 1 kHz 위 log10 PSD (n, m), 그 주파수 (m,))
    imps: all_impacts(X) 결과를 이미 가졌으면 넘긴다 (fmax · events · atoms와 같은 설정으로)
    fmax: None = 기본. 숫자(11-25, 11000) = 충격 검출 · L2_hfl 포락선 입력을 1 kHz–fmax 대역통과로, 옥타브 마지막 칸 8 kHz–fmax,
          파워 몫 분모 = 1 kHz–fmax 파워, 봉우리용 log PSD = 1 kHz–fmax. 열 이름은 그대로 둔다 (예: H10_ring_8-12.5k = 8–fmax)
    events: True = H1 수 · H2 높이 비 · H7–H10 울림 비율 · L1 hit 시점을 25 ms 이벤트 단위로 (11-25 R1)
    atoms: 울림 사전 "250"(기본) | "50"(11-25 R2, H7–H10 울림 주파수)
    lock_tol: L1_lock hit 허용 비율 (기본 0.03, lock_S tol)"""
    X = np.asarray(X, dtype=float)
    imps = all_impacts(X, fmax=fmax, events=events, atoms=atoms) if imps is None else imps
    rows, LP = [], []
    for x, im in zip(X, imps):
        h = []
        n = len(im["t"])
        hx = np.sort(im["height_x"])[::-1]
        h += [np.log1p(n), float(hx[1] / hx[0]) if n >= 2 else 0.0]
        h += octave_kurtosis(x, fmax)
        if n:
            c = np.bincount(octave_of(im["rings"][:, 0]).astype(int), minlength=4)[:4] / len(im["rings"])
            h += list(c)
        else:
            h += [-1.0] * 4
        f, P = welch_log(x)
        m = (f >= 1000) if fmax is None else (f >= 1000) & (f <= fmax)
        pw = np.array([P[(f >= lo) & (f < hi)].sum() for lo, hi in octaves(fmax)])
        h += list(np.log10(pw / P[m].sum() + 1e-20))
        env = np.abs(hilbert(im["hp"]))
        rms20 = np.sqrt(np.mean(sosfiltfilt(_sos("lowpass", 20, FS), x - x.mean()) ** 2))
        h.append(float(np.log10(np.median(env) / rms20)))
        S, rl, T, hit = lock_S(x, im["t"], tol=lock_tol)
        rows.append(h + [S, rl, T, hit])
        LP.append(np.log10(P[m] + 1e-20))
    return pd.DataFrame(rows, columns=H_FIXED + L_COLS + DIAG), np.array(LP), f[m]


DK1_FLOOR = float(np.log10(K_MIN * np.sqrt(2 * np.log(2))))   # 충격 < 2개일 때 = 가우스 바탕(포락선 중앙 ≈ 1.18σ)에서 검출 문턱의 σ 배수 (≈ 0.724)


def d_k1(imps):
    """D_K1 (11-25 EXP-064, H1′ 반복 타격): log10(두 번째로 큰 충격의 포락선 높이 height_g ÷ σ), σ = 1.4826·MAD(검출기 입력 hp).
    hp = all_impacts의 검출기 입력(fmax 쓰면 1 kHz–fmax 대역통과) → 진폭 단위가 없다(scale-free). 충격 < 2개면 DK1_FLOOR.
    예전 정의(features.extract_features_knock K1: 200 Hz 고역, |h| > 8σ 사건 · 10 ms 묶음, 바닥 log10 8)와 다른 점:
    대역 = 검출기 입력, 충격 = 지금 검출기(포락선/바탕 ≥ 4.5, 6.75 ms 간격), 높이 = 포락선, 바닥 = log10(4.5·√(2 ln 2))"""
    out = []
    for im in imps:
        hp = np.asarray(im["hp"], float)
        sig = 1.4826 * np.median(np.abs(hp - np.median(hp))) + 1e-12
        hg = np.sort(np.asarray(im["height_g"], float))[::-1]
        out.append(float(np.log10(hg[1] / sig)) if len(hg) >= 2 else DK1_FLOOR)
    return np.array(out)


def max_ring_x(imps, lo, hi):
    """대역별 최대 충격 배수 (11-25 4-19 · EXP-065): 울림 주파수(rings[:, 0])가 [lo, hi) Hz인 충격 중 최대 height_x(포락선 ÷ 바탕),
    log10(1 + x). 그 대역으로 운 충격이 없으면 0. 4-19 run.py와 같은 식"""
    out = []
    for im in imps:
        if not len(im["t"]):
            out.append(0.0); continue
        f, hx = np.asarray(im["rings"])[:, 0], np.asarray(im["height_x"], float)
        m = (f >= lo) & (f < hi)
        out.append(float(np.log10(1 + hx[m].max())) if m.any() else 0.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# 1–5 kHz(연결부 대역) 후보 C1–C4 (11-25 4-20 · EXP-067). 정의는 라벨을 보기 전에 고정. 모두 무차원
#   C1 대역 포락선 첨도(Pearson, |Hilbert|, 끝 10 ms 제외) — 연결부 1.6–4.5k, 대비 = log10(연결부 ÷ 본체 5.5–8k), 옥타브 2–4k 판
#   H4 옥타브 2–4k 대역 신호 log10(Pearson 첨도) (feat2_table H4와 같은 식, J 미사용 열)
#   C2 TKEO Ψ = x(n)² − x(n−1)x(n+1) (대역통과 신호)의 log10(P99 ÷ P50)
#   C3 FFT 대역 분할(WPD 3단계와 같은 1.5625 kHz 경계, PyWavelets 없음): 칸1 1.56–3.12 · 칸2 3.12–4.69 kHz 에너지 ÷ 칸1–6(1.56–10.94 kHz)
#      전체 파일 · 충격 구간만(충격 [−2, +12] ms 합집합, 4096점 조각 FFT 합). 충격 없으면 충격 구간 판 = −1
#   C4 같은 대역의 충격 구간 평균 에너지 ÷ 구간 밖 평균 에너지, log10. 충격 없으면 0(대비 없음)
B15_CONN, B15_BODY, B15_OCT = (1600, 4500), (5500, 8000), (2000, 4000)
B15_WPD = 25000 / 16                                     # 1562.5 Hz
B15_COLS = ["C1_kenv_1.6-4.5k", "C1_kc_1.6-4.5k", "C1_kc_2-4k", "H4_kurt_2-4k", "C2_tk_1.6-4.5k", "C2_tk_2-4k",
            "C3_w1_all", "C3_w2_all", "C3_w1_imp", "C3_w2_imp", "C4_ib_1.6-4.5k", "C4_ib_2-4k"]
B15_PRE_MS, B15_POST_MS, B15_EDGE_MS = 2, 12, 10


def impact_mask(n, idx, fs=FS, pre_ms=B15_PRE_MS, post_ms=B15_POST_MS):
    """충격 표본 번호 → 길이 n의 bool 마스크 ([−pre, +post] ms 합집합)"""
    m = np.zeros(n, bool); a, b = int(round(pre_ms * fs / 1000)), int(round(post_ms * fs / 1000))
    for p in np.asarray(idx, int):
        m[max(0, p - a):min(n, p + b)] = True
    return m


def _wpd_energy(seg_list, nfft=4096):
    """조각들의 칸 0–7(1.5625 kHz 폭) 에너지 합 (조각별 nfft점 rfft, 긴 조각은 nfft씩 나눔)"""
    f = np.fft.rfftfreq(nfft, 1 / FS); k = np.minimum((f // B15_WPD).astype(int), 7); E = np.zeros(8)
    for s in seg_list:
        for j in range(0, len(s), nfft):
            E += np.bincount(k, np.abs(np.fft.rfft(s[j:j + nfft], nfft)) ** 2, minlength=8)[:8]
    return E


def band15_table(X, idx_list, fs=FS):
    """신호 (n, N) + 파일별 충격 표본 번호(all_impacts의 idx) → DataFrame(B15_COLS). 길이 N은 자유(반쪽 안정성 점검용)"""
    rows = []
    sos = {k: butter(4, list(b), btype="bandpass", fs=fs, output="sos") for k, b in (("c", B15_CONN), ("b", B15_BODY), ("o", B15_OCT))}
    pk = lambda v: kurtosis(v, fisher=False)
    for x, idx in zip(np.asarray(X, float), idx_list):
        x = x - x.mean(); N = len(x); e = int(B15_EDGE_MS * fs / 1000); sl = slice(e, N - e)
        s = {k: sosfiltfilt(v, x) for k, v in sos.items()}
        env = {k: np.abs(hilbert(v))[sl] for k, v in s.items()}
        kc, kb, ko = np.log10(pk(env["c"])), np.log10(pk(env["b"])), np.log10(pk(env["o"]))
        h4 = float(np.log10(pk(s["o"])))
        tk = lambda v: (lambda p: float(np.log10(np.percentile(p, 99) / max(np.percentile(p, 50), 1e-30))))(v[1:-1] ** 2 - v[:-2] * v[2:])
        Ea = _wpd_energy([x]); m = impact_mask(N, idx, fs)
        r = [kc, kc - kb, ko - kb, h4, tk(s["c"][sl]), tk(s["o"][sl]), Ea[1] / Ea[1:7].sum(), Ea[2] / Ea[1:7].sum()]
        if m.any():
            d = np.diff(np.r_[0, m.astype(int), 0]); st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
            Ei = _wpd_energy([x[a:b] for a, b in zip(st, en)]); r += [Ei[1] / Ei[1:7].sum(), Ei[2] / Ei[1:7].sum()]
            ib = lambda v: float(np.log10(np.mean(v[m] ** 2) / max(np.mean(v[~m] ** 2), 1e-30))) if (~m).any() else 0.0
            r += [ib(s["c"]), ib(s["o"])]
        else:
            r += [-1.0, -1.0, 0.0, 0.0]
        rows.append(r)
    return pd.DataFrame(rows, columns=B15_COLS)


def cached_feat2(tag, X, names=None, fmax=None):
    """캐시 outputs/cache/feat2_{tag}.csv + feat2_psd_{tag}.npz. 피처 함수를 고치면 지우고 다시. fmax를 쓰면 tag도 따로 (예: train_11k)"""
    p, q = os.path.join(_CACHE, f"feat2_{tag}.csv"), os.path.join(_CACHE, f"feat2_psd_{tag}.npz")
    if os.path.exists(p) and os.path.exists(q):
        z = np.load(q)
        return renamed(pd.read_csv(p)), z["L"], z["f"]                     # 옛 캐시(H18_hfl · L1_S)도 새 이름으로
    T, L, f = feat2_table(X, fmax=fmax)
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


# ---------------------------------------------------------------------------
# 울림 모드 경계 피처 (EXP-057 · 058, 11-23). 경계 = 2-14에서 라벨 없이 찾은 울림 모드 골짜기
# 같은 계산식(첨도 · 울림 비율 · 에너지 몫)을 옥타브 대신 모드 칸으로 + 소린 비율 2열
# ---------------------------------------------------------------------------
MODE = ((1000, 1600), (1600, 4500), (4500, 5500), (5500, 8000), (8000, 12500))
MODE_TAG = ("공유1.0-1.6k", "연결부1.6-4.5k", "전이4.5-5.5k", "본체5.5-8k", "고주파8-12.5k")
MODE_COLS = ([f"MK_{t}" for t in MODE_TAG] + [f"MR_{t}" for t in MODE_TAG] + [f"MS_{t}" for t in MODE_TAG]
             + ["SO_연결부÷본체", "SO_고주파÷본체"])


def mode_band_feats(x, im):
    """신호 1개 + all_impacts 결과 1개 → MODE_COLS 값 list
    MK = 칸 대역통과 log10(Pearson 첨도) (H3–H6과 같은 식) · MR = 충격 울림 주파수(50 Hz 사전)의 칸별 비율, 충격 0개면 −1 (H7–H10과 같은 식)
    MS = log10(칸 Welch 파워 ÷ 1 kHz 위 파워) (H12–H15와 같은 식) · SO = 큰 충격 20개 직후 12 ms log10 중앙값 √(E 분자 ÷ E 본체), 충격 0개면 −3"""
    from src.impacts import _atoms50, RM_NO_IMPACT
    from src.final import _SEG
    x = np.asarray(x, float); xc = x - x.mean(); out = []
    for lo, hi in MODE:
        s = sosfiltfilt(_band_sos(lo, 12400 if hi >= 12500 else hi), xc); out.append(np.log10(kurtosis(s) + 3.0))
    xh = im["hp"]; pk = np.array([p for p in im["idx"] if p + _SEG <= len(xh)], int)
    if len(pk):
        A, P = _atoms50()
        S = np.stack([xh[p:p + _SEG] for p in pk]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
        fr_ = P[np.linalg.norm(np.einsum("ks,ast->kat", S, A), axis=2).argmax(1)]
        out += [float(((fr_ >= lo) & (fr_ < hi)).mean()) for lo, hi in MODE]
    else:
        out += [-1.0] * len(MODE)
    f, Pw = welch_log(x); m = f >= 1000
    out += list(np.log10(np.array([Pw[(f >= lo) & (f < hi)].sum() for lo, hi in MODE]) / Pw[m].sum() + 1e-20))
    w12 = int(.012 * FS); pk12 = np.array([p for p in im["idx"] if p + w12 <= len(xh)], int)
    if len(pk12):
        h = np.interp(pk12, im["idx"], im["height_g"]); sel = pk12[np.argsort(h)[::-1][:20]]
        fq = np.fft.rfftfreq(w12, 1 / FS)
        E = np.abs(np.fft.rfft(np.stack([xh[p:p + w12] for p in sel]) * np.hanning(w12), axis=1)) ** 2
        e = lambda lo, hi: E[:, (fq >= lo) & (fq < hi)].sum(1) + 1e-30
        out += [float(np.log10(np.median(np.sqrt(e(1600, 4500) / e(5500, 8000))))), float(np.log10(np.median(np.sqrt(e(8000, 12500) / e(5500, 8000)))))]
    else:
        out += [RM_NO_IMPACT] * 2
    return out


def mode_band_table(X):
    X = np.asarray(X, dtype=float)
    return pd.DataFrame([mode_band_feats(x, im) for x, im in zip(X, all_impacts(X))], columns=MODE_COLS)
