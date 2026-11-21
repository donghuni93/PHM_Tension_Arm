# %% [markdown]
# 2-4 EDA 뷰어 만들기 — 150개 파일의 시간·주파수 영역 특징을 HTML 한 장으로
# - 지표: 개념 사전(단계/개념사전.md)의 방법 중 계산 가능한 것. 묶음 8개 (기본 크기 · 형상 · 충격·비정상성 · 저주파 ·
#   스펙트럼 모양 · 물리 대역 · 포락선 · 1 kHz 대역). 정상 대비 백분위, 설명용 태그
# - 그림 데이터 (11-21 탭 개편): 고주파 = 신호 5개별 파형 포락·최대 충격 ±20 ms·PSD·스펙트로그램(1024점 + 256점) / 저주파 = 0–200 Hz PSD(8192점)·200·20 Hz 저역 파형 /
#   충격·박자 = 1 kHz 고역 포락선·포락선 스펙트럼·충격 목록(src.final.impact_rings)·주기 잠금(30 Hz 저역)
# - 모델 결과: 유의미 실험 전부(단계/5_모델비교/base/)의 seed 5번 중 미탐·오경보 횟수 + 규칙 켜짐. 뷰어에서 기준 모델을 고른다
# - 라벨은 백분위 기준(정상 분포)과 색에만 쓴다. EDA용 기술 통계이고 모델 선택에 쓰지 않는다
# 실행: conda run -n qe-phm python "단계/2_EDA/2-4_EDA뷰어/build_viewer.py"  → 같은 폴더 EDA뷰어.html

# %% 설정
import os, re, sys, json, base64
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train, cached_table
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram, hilbert
from scipy.ndimage import median_filter
from scipy.stats import kurtosis, skew

FS, N_SAMP = 25000, 50000
# 데이터셋 목록 (설정 한 곳): train = 라벨 있음(정상 0 · 고장 1), val·test = 라벨 없음(코드 2 · 3). 폴더가 없으면 건너뛴다
# 기준값(정상 5–95% 띠·백분위·태그 문턱·정상/고장 중앙)은 train으로만 계산하고 val·test는 그 기준 위에 놓일 뿐이다
from src.data import load_folder
DATASETS = [("val", os.path.join(ROOT, "data", "TeamProject_val", "val"), 2),
            ("test", os.path.join(ROOT, "data", "TeamProject_test", "test"), 3)]
X_tr, y_tr, names_tr = load_train(); names_tr = [str(n) for n in names_tr]; NTR = len(y_tr)
_px, _py, _pn, DS_LOADED = [X_tr], [np.asarray(y_tr)], [names_tr], [("train", NTR)]
for _tag, _dir, _code in DATASETS:
    if not os.path.isdir(_dir): DS_LOADED.append((_tag, 0)); continue
    _x, _n = load_folder(_dir)
    assert _x.shape[1] == N_SAMP, f"{_tag}: 신호 길이 {_x.shape[1]}"
    _px.append(_x); _py.append(np.full(len(_x), _code)); _pn.append([str(n) for n in _n]); DS_LOADED.append((_tag, len(_x)))
X, y, names = np.vstack(_px), np.concatenate(_py), sum(_pn, [])
N = len(y); nor = y == 0; abn = y == 1; trm = y <= 1            # 정상 · 고장 · train 전체 마스크 (val·test = y ≥ 2)
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
IMP_K, IMP_GAP = 8.0, int(0.010 * FS)       # 충격 = |신호| > 8σ(MAD 기준), 10 ms 안은 한 사건
WIN = int(0.050 * FS)                       # 50 ms 창 → 40개
NW = N_SAMP // WIN
PB = [("lo", 200, 2000), ("mid", 2000, 5000), ("hi", 5000, 12501)]    # 노션 물리 분할 예시 (저역은 200 Hz부터)
BANDS = [(k * 1000, min((k + 1) * 1000, 12500)) for k in range(13)]


# %% 분석 한 벌: 신호 Z(평균 제거 Xc 또는 200 Hz 고역통과 H)에 모든 지표·그림 데이터를 같은 정의로 계산
#    → 뷰어의 "신호" 전환이 두 벌 중 하나를 고른다. 스펙트럼 지표는 전 대역(0–12.5 kHz) 기준이라
#      고역통과 쪽에서는 200 Hz 아래가 사라진 효과가 그대로 드러난다
def analyze(Z):
    V, A = {}, {}                                   # 지표 값 (key → (N,)), 그림용 배열
    # 시간 영역: 기본 크기·형상
    ax = np.abs(Z)
    V["mean"] = X.mean(1); V["std"] = Z.std(1); V["maxabs"] = ax.max(1)
    V["p2p"] = Z.max(1) - Z.min(1); V["mav"] = ax.mean(1)
    V["kurt"] = kurtosis(Z, axis=1); V["skew"] = skew(Z, axis=1)
    V["crest"] = V["maxabs"] / V["std"]; V["impulse_f"] = V["maxabs"] / V["mav"]
    V["shape_f"] = V["std"] / V["mav"]; V["clear_f"] = V["maxabs"] / np.sqrt(ax).mean(1) ** 2
    d1 = np.diff(Z, axis=1); d2 = np.diff(d1, axis=1)
    mob = lambda a, da: np.sqrt(da.var(1) / a.var(1))
    V["hjorth_mob"] = mob(Z, d1) * FS / (2 * np.pi)                 # Hz 단위 (평균 주파수 근사)
    V["hjorth_cpx"] = mob(d1, d2) / mob(Z, d1)

    # 충격·비정상성
    for k in ["n_imp", "prot", "imp_gap", "tkeo_kurt", "tkeo_mm", "win_cv", "win_rms_mm", "win_kurt_max", "floor", "zcr", "zcr_std"]:
        V[k] = np.full(N, np.nan)
    events, WRMS, WKURT = [], np.zeros((N, NW)), np.zeros((N, NW))
    for i in range(N):
        h = Z[i]
        sig = 1.4826 * np.median(np.abs(h - np.median(h))) + 1e-12
        idx = np.flatnonzero(np.abs(h) > IMP_K * sig)
        ev = [] if len(idx) == 0 else [g[np.argmax(np.abs(h[g]))] for g in np.split(idx, np.flatnonzero(np.diff(idx) > IMP_GAP) + 1)]
        events.append([round(e / FS, 4) for e in ev])
        V["n_imp"][i] = len(ev); V["prot"][i] = np.abs(h).max() / sig
        if len(ev) >= 2:
            V["imp_gap"][i] = np.median(np.diff(ev)) / FS
        tk = h[1:-1] ** 2 - h[:-2] * h[2:]
        V["tkeo_kurt"][i] = kurtosis(tk); V["tkeo_mm"][i] = tk.max() / (np.abs(tk).mean() + 1e-30)
        hw = h.reshape(NW, WIN)
        w = hw.std(1); WRMS[i] = w; WKURT[i] = kurtosis(hw, axis=1)
        V["win_cv"][i] = w.std() / w.mean(); V["win_rms_mm"][i] = w.max() / w.mean()
        V["win_kurt_max"][i] = WKURT[i].max(); V["floor"][i] = w.min()
        zc = (np.diff(np.signbit(hw), axis=1) != 0).sum(1) / (WIN / FS)
        V["zcr"][i] = (np.diff(np.signbit(h)) != 0).sum() / (N_SAMP / FS); V["zcr_std"][i] = zc.std() / zc.mean()
    A.update(events=events, WRMS=WRMS, WKURT=WKURT)

    # 주파수 영역 (Welch 4096 → 6.1 Hz 간격). detrend=False: 구간마다 평균을 빼면 6 Hz 아래 드리프트가 사라져
    # 저주파 몫이 작게 나온다 (abn_001: 기본값 61% vs 시간 영역 93%)
    f, P = welch(Z, fs=FS, nperseg=4096, detrend=False)
    df = f[1] - f[0]
    band_e = lambda lo, hi: np.log10(P[:, (f >= lo) & (f < hi)].sum(1) * df + 1e-30)
    tot = P.sum(1)
    V["s20"] = P[:, f < 20].sum(1) / tot; V["s200"] = P[:, f < 200].sum(1) / tot
    V["e0_20"] = band_e(0, 20); V["e20_200"] = band_e(20, 200)
    V["e200"] = band_e(200, 12501)
    a1 = f > 0                                                    # DC 칸 제외 전 대역
    Pa, fa = P[:, a1], f[a1]; Pn = Pa / Pa.sum(1, keepdims=True)
    V["centroid"] = (Pn * fa).sum(1)
    V["bandwidth"] = np.sqrt((Pn * (fa[None] - V["centroid"][:, None]) ** 2).sum(1))
    cum = np.cumsum(Pn, 1)
    V["rolloff85"] = fa[(cum < 0.85).sum(1)]; V["rolloff95"] = fa[(cum < 0.95).sum(1)]
    V["hi_ratio"] = np.log10(P[:, f >= 8000].sum(1) / Pa.sum(1))
    V["flat"] = np.exp(np.log(Pa + 1e-30).mean(1)) / Pa.mean(1)
    V["entropy"] = -(Pn * np.log(Pn + 1e-30)).sum(1) / np.log(Pn.shape[1])
    L = np.log10(P + 1e-30)
    base = median_filter(L, size=(1, 31), mode="nearest")      # 약 190 Hz 이동 중앙값 = 매끈한 바탕
    prom = (L - base)[:, a1]
    V["peak_f"] = fa[prom.argmax(1)]; V["peak_prom"] = prom.max(1)
    for k, lo, hi in PB:
        V[f"e_{k}"] = band_e(lo, hi)
        V[f"r_{k}"] = P[:, (f >= lo) & (f < hi)].sum(1) / Pa.sum(1)
    for k, (lo, hi) in enumerate(BANDS):
        V[f"b{k}"] = band_e(lo, hi)
    A.update(f=f, df=df, L=L)

    # 스펙트로그램 · 스펙트럼 첨도 (STFT 1024, 50% 겹침)
    fs_, ts_, S = spectrogram(Z, fs=FS, nperseg=1024, noverlap=512)       # (N,513,96) 파워
    SK = (S ** 2).mean(2) / (S.mean(2) ** 2 + 1e-60) - 2                      # 주파수별 시간 첨도
    k1 = fs_ > 0
    V["sk_max"] = SK[:, k1].max(1); V["sk_f"] = fs_[k1][SK[:, k1].argmax(1)]
    A.update(fs_=fs_, S=S, SK=SK)

    # 포락선 (Hilbert 포락) → 2.5 kHz로 줄여 스펙트럼 0–300 Hz
    ENVS = np.abs(hilbert(Z, axis=1))
    V["env_kurt"] = kurtosis(ENVS, axis=1)
    Ed = ENVS.reshape(N, -1, 10).mean(2); Ed = Ed - Ed.mean(1, keepdims=True)
    fe, PE = welch(Ed, fs=FS / 10, nperseg=2500)                              # 1 Hz 간격
    ke = (fe >= 0.5) & (fe <= 300)
    fe, LE = fe[ke], np.log10(PE[:, ke] + 1e-30)
    pe = LE - median_filter(LE, size=(1, 21), mode="nearest")
    ke1 = fe >= 1
    V["env_pf"] = fe[ke1][pe[:, ke1].argmax(1)]; V["env_prom"] = pe[:, ke1].max(1)
    A.update(fe=fe, LE=LE)
    return V, A


SIGS = {"raw": Xc, "hp": H}                       # 지표·태그·백분위를 계산하는 두 벌 (그림용 신호 5개는 아래 SIGS_PLOT)
RES = {s: analyze(Z) for s, Z in SIGS.items()}

# %% 11-21 2차: 충격·울림 · 저주파·박자 지표 (train·val 모두 같은 함수로, 라벨 안 씀). 아래 SETS 두 벌의 V에 같은 값을 넣어 정렬·표·산점도에서 쓴다
#   (기준값은 stats()가 train 정상으로만 계산한다 — val은 그 위에 놓일 뿐)
from scipy.signal import find_peaks
from src.final import impact_rings, ring_table, _ring_atoms, _SEG
H1K = sosfiltfilt(butter(4, 1000, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)    # impact_rings와 같은 신호
_lowpass = lambda fc: sosfiltfilt(butter(4, fc, btype="lowpass", fs=FS, output="sos"), Xc, axis=-1)
L20_full = _lowpass(20)

# --- 충격 (새 정의 11-21: src/impacts.py = 사령탑 4-6 7b-7e 확정 정의 + 파일 앞뒤 10 ms 제외). 울림 = 같은 충격마다 final._ring_atoms 사전 8 ms 투영
import pandas as pd
from src.impacts import all_impacts, _baseline
from src.features2 import lock_S, octave_of
ENVS1 = np.abs(hilbert(H1K, axis=1))                                # 1 kHz 고역 신호의 Hilbert 포락선 (포락선 스펙트럼용)
_lp1k = butter(4, 1000, btype="lowpass", fs=FS, output="sos")        # 새 정의의 다듬기 1000 Hz
_IMPS = all_impacts(X)
IMP_ALL, IMP_TOP, IMP_NV = [], [], []
ENV_P = np.zeros((N, N_SAMP // 25), np.uint16); ENV_MAX = np.zeros(N); ENV_MED = np.zeros(N)
IMP_ROWS = []                                                       # 파일별 전체 충격 (t, f, ζ, fit)
RING_O = np.full((N, 4), np.nan); RES_T = np.zeros(N); FAST_DECAY = np.full(N, np.nan)
for i in range(N):
    im = _IMPS[i]; env = sosfiltfilt(_lp1k, ENVS1[i]); med = float(np.median(_baseline(env)))   # 바탕(100 ms 이동 중앙값)의 중앙값
    IMP_ALL.append([round(float(t), 4) for t in im["t"]]); IMP_NV.append(len(im["t"]))
    em = env.reshape(-1, 25).max(1); ENV_MAX[i] = em.max(); ENV_MED[i] = med
    ENV_P[i] = np.round(em / em.max() * 65535).astype(np.uint16)
    rg = im["rings"]; rows = [(float(t), float(r[0]), float(r[1]), float(r[2])) for t, r in zip(im["t"], rg)]
    IMP_ROWS.append(rows)
    if len(rg):
        RING_O[i] = np.bincount(octave_of(rg[:, 0]).astype(int), minlength=4)[:4] / len(rg)
        m_ = (rg[:, 0] >= 8000) & (rg[:, 0] < 8700) & (rg[:, 1] <= 0.02); RES_T[i] = rg[m_, 2].max() if m_.any() else 0.0
        FAST_DECAY[i] = float((rg[:, 1] >= 0.04).mean())
    top = sorted(rows, key=lambda r: -r[3])[:30]
    IMP_TOP.append([[round(r[0], 4), round(r[1] / 1000, 3), round(float(r[2]), 2), round(float(r[3]), 3)] for r in top])
_tab = pd.read_csv(os.path.join(ROOT, "단계", "4_피처", "4-6_1121_충격재정의", "사령탑_7b-7e", "충격표_전체.csv"), encoding="utf-8-sig")
_pre = [__import__("src.impacts", fromlist=["x"]).detect_impacts(X[i], edge_ms=0) for i in range(N)]
_ok = all(len(_tab[_tab.file == nm]) == len(_pre[i]["t"]) and np.allclose(_tab[_tab.file == nm].t_s.values, _pre[i]["t"], atol=1e-5) for i, nm in enumerate(names) if nm.startswith(("nor_", "abn_", "val_")))
assert _ok, "끝단 제외 전 충격이 사령탑 충격표와 다름"
IMP_CHECK = f"새 충격 정의(11-21) 확인: 끝단 제외 전 충격 {sum(len(p['t']) for p in _pre)}개 = 사령탑 7b-7e 충격표와 같음 · 끝단 제외 후 {sum(IMP_NV)}개"
del _pre
_nan1 = lambda a: np.where(a < 0, np.nan, a)

# 포락선 스펙트럼 (1 kHz 고역 신호의 포락선): analyze()와 같은 절차 — 2.5 kHz로 줄여 0.5–300 Hz
_Ed = ENVS1.reshape(N, -1, 10).mean(2); _Ed = _Ed - _Ed.mean(1, keepdims=True)
fe1, PE1 = welch(_Ed, fs=FS / 10, nperseg=2500); _ke = (fe1 >= 0.5) & (fe1 <= 300)
fe1, LE1 = fe1[_ke], np.log10(PE1[:, _ke] + 1e-30)
del ENVS1

# --- 주기 잠금 (2-13 run.py의 lock_score와 같은 정의 그대로: 30 Hz 저역 → 500 Hz, T = 0.3–1.2 s, ±3%). train 값은 csv와 일치하는지 assert
_pear = lambda s_, k: np.corrcoef(s_[:len(s_) - k], s_[k:])[0, 1]


def _events8(h, merge_s=0.010):
    sig = 1.4826 * np.median(np.abs(h - np.median(h)))
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    ev = []
    for j in idx:
        if not ev or j - ev[-1][-1] > int(merge_s * FS): ev.append([j])
        else: ev[-1].append(j)
    return np.array([e[np.argmax(np.abs(h[e]))] for e in ev]) / FS if ev else np.array([])


L30 = sosfiltfilt(butter(4, 30, btype="lowpass", fs=FS, output="sos"), Xc, axis=-1)[:, ::50]
_ks = np.arange(int(0.3 * 500), int(1.2 * 500) + 1, 2)
_lock = []
for i in range(N):   # 새 충격 시점(src/features2.lock_S, 2-13 정의 그대로 + 충격은 새 정의)
    S_, r_, T_, hit_ = lock_S(X[i], _IMPS[i]["t"])
    _lock.append((r_, T_, hit_, len(IMP_ALL[i])))
_lock = np.array(_lock)
LOCKC = dict(rlp=_lock[:, 0], T=_lock[:, 1], hit=_lock[:, 2], nev=_lock[:, 3].astype(int), S=_lock[:, 0] * _lock[:, 2])
SC_30 = np.abs(L30).max(1)

# --- 맥락 값 (저주파 탭과 같은 값): 0–20 · 20–200 Hz 에너지(원신호), 하중 대비 고주파, 고주파 RMS
_rms = lambda a: np.sqrt((a ** 2).mean(1))
LOAD_RATIO = np.log10(_rms(H) / _rms(L20_full))                     # 200 Hz 이상 RMS / 20 Hz 저역 RMS
KURT_1K = kurtosis(H1K, axis=1)
EXTRA_V = dict(n_ring=np.array(IMP_NV, float), ring_o1=RING_O[:, 0], ring_o2=RING_O[:, 1], ring_o3=RING_O[:, 2], ring_o4=RING_O[:, 3], res83=RES_T.copy(),
               fast_decay=FAST_DECAY, kurt_1k=KURT_1K,
               lb_S=LOCKC["S"], lb_rlp=LOCKC["rlp"], lb_hit=LOCKC["hit"], lb_e0_20=RES["raw"][0]["e0_20"], lb_e20_200=RES["raw"][0]["e20_200"],
               load_hf=LOAD_RATIO, rms_hf=_rms(H))
EXTRA_FEAT = [
    ("n_ring", "충격 수 (새 정의 11-21)", "개", "ring", "새 정의(11-21): 1 kHz 고역 → |hilbert| → 1000 Hz 다듬기, 100 ms 이동 중앙값 바탕의 4.5배 이상 봉우리(간격 ≥ 6.75 ms), 파일 앞뒤 10 ms 제외 (src/impacts.py)", "impulse-count"),
    ("ring_o1", "울림 몫 1–2 kHz", "", "ring", "새 정의(11-21) 충격 중 울림 주파수가 1–2 kHz인 비율 (옥타브 4칸 H7). 충격 0개 = 없음", "resonance-shift"),
    ("ring_o2", "울림 몫 2–4 kHz", "", "ring", "새 정의(11-21) 충격 중 울림 주파수가 2–4 kHz인 비율 (H8). 충격 0개 = 없음", "resonance-shift"),
    ("ring_o3", "울림 몫 4–8 kHz", "", "ring", "새 정의(11-21) 충격 중 울림 주파수가 4–8 kHz인 비율 (H9, 본체 모드 포함). 충격 0개 = 없음", "resonance-shift"),
    ("ring_o4", "울림 몫 8–12.5 kHz", "", "ring", "새 정의(11-21) 충격 중 울림 주파수가 8–12.5 kHz인 비율 (H10). 충격 0개 = 없음", "resonance-shift"),
    ("res83", "8.3 kHz 공진 fit", "", "ring", "새 정의(11-21) 충격 중 8.0–8.7 kHz · ζ ≤ 0.02로 울린 충격의 최대 fit (없으면 0)", "resonance-shift"),
    ("fast_decay", "빨리 꺼지는 충격 비율", "", "ring", "ζ ≥ 0.04인 충격 / 전체 충격 (새 정의 11-21, 참고용: H11은 2차 설계에서 제외). 충격 0개 = 없음", "impulse-count"),
    ("kurt_1k", "고주파 첨도 (1 kHz 고역)", "", "ring", "1 kHz 고역통과 신호의 첨도 (Fisher)", "kurtosis"),
    ("lb_S", "주기 잠금 S", "", "lowbeat", "S = r_LP × hit (2-13 정의, 충격 시점은 새 정의 11-21)", "acf"),
    ("lb_rlp", "r_LP (느린 파형 반복)", "", "lowbeat", "30 Hz 저역 신호를 T만큼 민 자기 자신과의 상관 최댓값 (2-13)", "acf"),
    ("lb_hit", "hit (충격 주기 잠금)", "", "lowbeat", "새 정의(11-21) 충격 중 T ± 3%에 짝이 있는 비율 (2-13 정의, 충격 시점만 새 정의)", "impulse-count"),
    ("lb_e0_20", "0–20 Hz 에너지 (원신호)", "log g²", "lowbeat", "저주파 탭 맥락 숫자 ①. 평균 제거 신호 기준", "band-energy"),
    ("lb_e20_200", "20–200 Hz 에너지 (원신호)", "log g²", "lowbeat", "저주파 탭 맥락 숫자 ②. 평균 제거 신호 기준", "band-energy"),
    ("load_hf", "하중 대비 고주파", "log10", "lowbeat", "log10(200 Hz 이상 RMS / 20 Hz 저역 RMS). 저주파 탭 맥락 숫자 ③", "rms"),
    ("rms_hf", "고주파 RMS (200 Hz 고역)", "g", "lowbeat", "200 Hz 고역통과 신호의 RMS", "rms"),
]
EXTRA_GROUPS = [("ring", "충격·울림", "impulse-count"), ("lowbeat", "저주파·박자", "acf")]

# %% 지표 설명 (화면 표시용): key, 이름, 단위, 묶음, 설명, 개념 키
GROUPS = [("basic", "기본 크기", "rms"), ("shape", "형상 지표", "shape-factors"), ("impulse", "충격·비정상성", "window-stats"),
          ("low", "저주파", "bandpass"), ("spec", "스펙트럼 모양", "spectral-shape"), ("pband", "물리 대역", "band-split"),
          ("env", "포락선·스펙트럼 첨도", "envelope"), ("band", "1 kHz 대역 에너지", "band-energy")]
FEAT = [
    ("mean", "평균 (원신호)", "g", "basic", "설치 자세·중력 성분. 진동이 아님", "mean-removal"),
    ("std", "RMS", "g", "basic", "진동 크기. 평균 제거 신호에서는 200 Hz 아래 흔들림이 대부분", "rms"),
    ("maxabs", "Peak (최대 |진폭|)", "g", "basic", "가장 큰 순간값", "rms"),
    ("p2p", "Peak-to-Peak", "g", "basic", "최대 − 최소", "rms"),
    ("mav", "MAV (평균 |x|)", "g", "basic", "절대값 평균", "rms"),
    ("kurt", "첨도 (Fisher)", "", "shape", "바탕 대비 튀는 정도. 정규 0, 사인 −1.5 (Pearson으로는 +3)", "kurtosis"),
    ("skew", "왜도", "", "shape", "비대칭", "skewness"),
    ("crest", "파고율", "", "shape", "Peak / RMS", "shape-factors"),
    ("impulse_f", "충격 계수", "", "shape", "Peak / MAV", "shape-factors"),
    ("shape_f", "형상 계수", "", "shape", "RMS / MAV (정규 잡음 ≈ 1.25)", "shape-factors"),
    ("clear_f", "여유 계수", "", "shape", "Peak / (평균 √|x|)²", "shape-factors"),
    ("hjorth_mob", "Hjorth 이동도", "Hz", "shape", "평균 주파수 근사 (평균 제거 신호에서는 저주파 흔들림이 지배)", "hjorth"),
    ("hjorth_cpx", "Hjorth 복잡도", "", "shape", "정현파에서 벗어난 정도 (정현파 = 1)", "hjorth"),
    ("n_imp", "충격 수", "개", "impulse", "|신호| > 8σ(MAD) 사건 수, 10 ms 안은 하나", "impulse-count"),
    ("prot", "충격 돌출도", "σ", "impulse", "가장 큰 충격 / 바탕 σ", "impulse-count"),
    ("imp_gap", "충격 간격 중앙값", "s", "impulse", "충격 2개 이상일 때. 반복 주기 단서 (H4)", "impulse-count"),
    ("tkeo_kurt", "TKEO 첨도", "", "impulse", "Teager 에너지의 첨도", "tkeo"),
    ("tkeo_mm", "TKEO 최대/평균", "", "impulse", "Teager 에너지 최대 / 평균 |Ψ|", "tkeo"),
    ("win_cv", "창 RMS 변동계수", "", "impulse", "50 ms 창 RMS의 표준편차 / 평균", "window-stats"),
    ("win_rms_mm", "창 RMS 최대/평균", "", "impulse", "가장 센 창이 평균의 몇 배인가", "window-stats"),
    ("win_kurt_max", "창 첨도 최대", "", "impulse", "40개 창 중 가장 큰 첨도", "window-stats"),
    ("floor", "조용한 바탕", "g", "impulse", "가장 조용한 50 ms 창의 RMS (센서 잡음 바닥 근처)", "window-stats"),
    ("zcr", "영교차율", "/s", "impulse", "신호가 0을 지나는 횟수 (≈ 주된 주파수 × 2)", "zcr"),
    ("zcr_std", "창 영교차율 변동계수", "", "impulse", "창마다 주된 주파수가 바뀌는 정도", "zcr"),
    ("s20", "0–20 Hz 비율", "", "low", "분산 중 느린 흔들림 몫", "bandpass"),
    ("s200", "0–200 Hz 비율", "", "low", "분산 중 200 Hz 아래 몫 = 고역통과가 지우는 몫", "bandpass"),
    ("e0_20", "0–20 Hz 에너지", "log g²", "low", "느린 흔들림 에너지 (log10)", "band-energy"),
    ("e20_200", "20–200 Hz 에너지", "log g²", "low", "저주파 진동 에너지 (log10)", "band-energy"),
    ("e200", "200 Hz 이상 에너지", "log g²", "spec", "고역 전체 에너지 (log10)", "band-energy"),
    ("centroid", "중심주파수", "Hz", "spec", "전 대역 에너지의 무게중심", "spectral-shape"),
    ("bandwidth", "스펙트럼 대역폭", "Hz", "spec", "중심 주변 퍼짐", "spectral-shape"),
    ("rolloff85", "85% 롤오프", "Hz", "spec", "에너지의 85%가 이 아래", "spectral-shape"),
    ("rolloff95", "95% 롤오프", "Hz", "spec", "에너지의 95%가 이 아래", "spectral-shape"),
    ("hi_ratio", "고역 비 (8 k+ / 전체)", "log", "spec", "8–12.5 kHz 몫 (log10)", "spectral-shape"),
    ("entropy", "스펙트럼 엔트로피", "", "spec", "1 = 고르게 퍼짐, 0 = 한 칸에 몰림", "spectral-entropy"),
    ("flat", "스펙트럼 평탄도", "", "spec", "기하평균 / 산술평균. 1 = 백색 잡음", "spectral-entropy"),
    ("peak_f", "최대 돌출 피크 주파수", "Hz", "spec", "매끈한 바탕보다 가장 튀어나온 칸", "resonance-shift"),
    ("peak_prom", "피크 돌출도", "dec", "spec", "그 피크가 바탕보다 몇 자리(log10) 높은가", "resonance-shift"),
    ("e_lo", "저 200–2k 에너지", "log g²", "pband", "노션 물리 분할 '저주파' (200 Hz부터)", "band-split"),
    ("e_mid", "중 2–5k 에너지", "log g²", "pband", "노션 물리 분할 '중주파'", "band-split"),
    ("e_hi", "고 5–12.5k 에너지", "log g²", "pband", "노션 물리 분할 '고주파'", "band-split"),
    ("r_lo", "저 200–2k 비율", "", "pband", "전체 에너지 중 몫 (게인 무관)", "payne"),
    ("r_mid", "중 2–5k 비율", "", "pband", "전체 에너지 중 몫", "payne"),
    ("r_hi", "고 5–12.5k 비율", "", "pband", "전체 에너지 중 몫", "payne"),
    ("env_kurt", "포락선 첨도", "", "env", "Hilbert 포락의 첨도", "envelope"),
    ("env_pf", "포락선 반복 주파수", "Hz", "env", "포락선 스펙트럼(1–300 Hz)에서 가장 튀어나온 피크 = 충격 반복률 후보", "envelope-spectrum"),
    ("env_prom", "포락선 피크 돌출도", "dec", "env", "그 반복 피크가 얼마나 뚜렷한가", "envelope-spectrum"),
    ("sk_max", "스펙트럼 첨도 최대", "", "env", "시간적으로 가장 충격적인 주파수 칸의 첨도", "spectral-kurtosis"),
    ("sk_f", "스펙트럼 첨도 최대 주파수", "Hz", "env", "그 칸의 주파수", "spectral-kurtosis"),
] + [(f"b{k}", f"{lo/1000:g}–{hi/1000:g} kHz", "log g²", "band", "고정 1 kHz 격자 대역 에너지 (log10)", "band-energy")
     for k, (lo, hi) in enumerate(BANDS)]
FEAT += EXTRA_FEAT; GROUPS += EXTRA_GROUPS                      # 11-21 2차: 새 지표 그룹 2개 (n_out·태그에는 안 씀)

# %% 정상 대비 백분위·기준값
def pct_of(v):
    vn = v[nor & ~np.isnan(v)]
    return np.array([np.nan if np.isnan(a) else 100 * ((vn < a).mean() + 0.5 * (vn == a).mean()) for a in v])


MAIN = [k for k, _, _, g, *_ in FEAT if g not in ("band", "ring", "lowbeat")]
FEAT.append(("n_out", "정상 범위 밖 지표 수", "개", "summary", f"1 kHz 대역을 뺀 지표 {len(MAIN)}개 중 정상 5–95% 밖인 개수. 정상도 우연히 몇 개는 밖으로 나간다", "collinearity"))


def stats(V):
    pct, ref = {}, {}
    for k, *_ in FEAT[:-1]:
        v = V[k]; vn = v[nor & ~np.isnan(v)]; vf = v[abn & ~np.isnan(v)]
        pct[k] = pct_of(v)
        ref[k] = dict(p5=np.percentile(vn, 5), p50=np.median(vn), p95=np.percentile(vn, 95), f50=np.median(vf) if len(vf) else np.nan)
    n_out = np.array([sum(1 for k in MAIN if not np.isnan(pct[k][i]) and (pct[k][i] < 5 or pct[k][i] > 95)) for i in range(N)])
    V["n_out"] = n_out.astype(float); pct["n_out"] = pct_of(V["n_out"])
    ref["n_out"] = dict(p5=np.percentile(n_out[nor], 5), p50=np.median(n_out[nor]), p95=np.percentile(n_out[nor], 95), f50=np.median(n_out[abn]))
    return pct, ref


# %% 설명용 태그 (규칙은 물리·정상 분포 기준. 한 파일에 여러 개)
def tags_of(V, ref):
    R = lambda k, q: ref[k][q]
    return [
    ("offset", "평균 오프셋", "|평균| > 0.1 g (설치 자세가 다른 측정)", np.abs(V["mean"]) > 0.1),
    ("quiet", "조용함", "RMS < 정상 5%", V["std"] < R("std", "p5")),
    ("loud", "큰 진동", "RMS > 정상 95%", V["std"] > R("std", "p95")),
    ("imp0", "충격 없음", "충격 수 0", V["n_imp"] == 0),
    ("imp1", "단발 충격", "충격 1–2개", (V["n_imp"] >= 1) & (V["n_imp"] <= 2)),
    ("impS", "몇 번 충격", "충격 3–9개", (V["n_imp"] >= 3) & (V["n_imp"] <= 9)),
    ("impN", "잦은 충격", "충격 10개 이상", V["n_imp"] >= 10),
    ("lowf", "저주파 흔들림 우세", "0–20 Hz가 분산의 60% 이상", V["s20"] >= 0.6),
    ("noise", "잡음·연속형", "첨도 < 1이고 충격 없음", (V["kurt"] < 1) & (V["n_imp"] == 0)),
    ("tonal", "협대역 피크", "피크 돌출도 > 정상 95%", V["peak_prom"] > R("peak_prom", "p95")),
    ("hif", "고주파 우세", "고역 비 > 정상 95%", V["hi_ratio"] > R("hi_ratio", "p95")),
    ("hie", "고역 에너지 큼", "200 Hz 이상 에너지 > 정상 95%", V["e200"] > R("e200", "p95")),
    ("skb", "충격 대역 뚜렷", "스펙트럼 첨도 최대 > 정상 95%", V["sk_max"] > R("sk_max", "p95")),
    ("envp", "포락선 주기성", "포락선 피크 돌출도 > 정상 95%", V["env_prom"] > R("env_prom", "p95")),
    ("multi", "다수 지표 이탈", f"정상 범위 밖 지표 수 > 정상의 95% 분위({R('n_out', 'p95'):.0f}개)", V["n_out"] > R("n_out", "p95")),
    ]


SETS = {}
for s, (V, A) in RES.items():
    V.update(EXTRA_V)                                              # 새 지표는 신호와 무관한 값이라 두 벌에 같은 값
    pct, ref = stats(V)
    TAGS = tags_of(V, ref)
    SETS[s] = dict(V=V, A=A, pct=pct, ref=ref, TAGS=TAGS, ftags=[[t[0] for t in TAGS if t[3][i]] for i in range(N)])

# %% 개념 사전 읽기 (## 분류 / ### 이름 {#키} / 본문)
md = open(os.path.join(ROOT, "단계", "개념사전.md"), encoding="utf-8").read()
intro = md.split("\n## ", 1)[0]
CONCEPTS, sec = [], None
for blk in re.split(r"\n(?=##+ )", md):
    m2 = re.match(r"## (.+)", blk); m3 = re.match(r"### (.+?)\s*\{#([\w-]+)\}\s*\n?(.*)", blk, re.S)
    if m3:
        CONCEPTS.append(dict(key=m3.group(2), title=m3.group(1).strip(), section=sec, body=m3.group(3).strip()))
    elif m2:
        sec = m2.group(1).strip()
ckeys = {c["key"] for c in CONCEPTS}
missing = sorted({c for *_, c in FEAT if c not in ckeys} | {c for *_, c in GROUPS if c not in ckeys})
assert not missing, f"개념사전에 없는 키: {missing}"

# %% 모델 결과 (seed 5번 중 횟수) — 단계/5_모델비교/base/의 실험을 모두 읽는다
#   뷰어 머리줄 "모델"에서 하나를 고르면 필터·묶기·배지·표·✕·경계·분석 조각이 모두 그 모델 기준 (11-16)
from src.experts import peak83_table, Rule83, KnockRule, LockRule, ETLogitX, HIT_COLS, EV_COLS, _hits
from src.features import feature_table_knock
from sklearn.model_selection import StratifiedGroupKFold
_SIG = os.path.join(ROOT, "단계", "5_모델비교")                      # 11-22: 최신 실험은 base/가 아니라 바로 아래
# 11-22: 뷰어 모델은 EXP-039 · EXP-044 둘만 (나머지는 뷰어에서 뺌. 실험 폴더는 그대로)
# result.json에 없는 정보: 짧은 이름 · 판정에 쓰는 규칙(RULES 키) · 메모. 평면(점 좌표 · 경계 · 축)은 plane_models.py가 figs/plane_models.json에 저장
EXP_SPEC = {
    "EXP-039": dict(short="D+ 규칙 없음", rules=[],
                    note="대표 구성 = AE + ET(D+ 81열) 규칙 없음. 규칙을 모델 안으로 넣은 실험이라 규칙 표시 없음"),
    "EXP-044": dict(short="AE 추가 (J)", rules=["r83", "res83", "e2"],
                    note="대표 구성 J = ET logit + 박자 z + AE z 결합 OR E2 OR 8.3k. 결합은 3입력이라 박자 z = 0 단면으로 그림 · E2 · 8.3k는 경계와 무관하게 고장 판정"),
}
# %% 2-16 충격 울림 (EXP-034): 충격마다 울림 주파수 f · 감쇠비 ζ · 맞춤 fit → 파일별 울림 비율 · 공진 fit
import pandas as pd
_IM = pd.read_csv(os.path.join(ROOT, "단계", "2_EDA", "8kHz공진", "2-16_1116_울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
_key = [os.path.splitext(os.path.basename(str(n)))[0] for n in names_tr]
_cnt = _IM.groupby("file").size().reindex(_key).fillna(0).values
_frac = lambda lo, hi: np.nan_to_num(_IM.assign(b=(_IM.f >= lo) & (_IM.f < hi)).groupby("file").b.mean().reindex(_key).values.astype(float), nan=-1.0)
RING = dict(RING_24=_frac(2125, 4125), RING_BODY=_frac(5500, 7750))
_RES = _IM[(_IM.f >= 8000) & (_IM.f < 8700) & (_IM.zeta <= 0.02)].groupby("file").fit.max().reindex(_key).fillna(0).values
_BODY = (RING["RING_BODY"] >= 0.9) & (_cnt >= 3)
WARN = []


def model_entry(folder):
    """그 실험의 대표 구성(main), main이 없으면 첫 구성"""
    r = json.load(open(os.path.join(_SIG, folder, "result.json"), encoding="utf-8"))
    cfg = r.get("main") or next(iter(r["configs"])); c = r["configs"][cfg]
    sp = EXP_SPEC.get(r["id"])
    if sp is None: WARN.append(f"EXP_SPEC에 {r['id']} 없음 → 규칙 없음 · 기본 평면으로 표시"); sp = {}
    return dict(id=r["id"], title=r["title"], model=r.get("model", ""), cfg=cfg, short=sp.get("short", r["id"]),
                rules=sp.get("rules", []), note=sp.get("note", ""),
                f2=c["f2"], f2_sd=c["f2_sd"], recall=c["recall"], fpr=c.get("fpr"),
                miss=[int(c["miss_counts"].get(n, 0)) for n in names], fp=[int(c["fp_counts"].get(n, 0)) for n in names])


# 최신 번호 먼저 (폴더 이름이 EXP-nnn으로 시작). 기본 선택 = 맨 앞
def _has_configs(d):
    p = os.path.join(_SIG, d, "result.json")
    if not os.path.isfile(p): return False
    if json.load(open(p, encoding="utf-8")).get("configs"): return True
    WARN.append(f"{d}: result.json에 configs 없음 → 뷰어에서 뺌"); return False


MODELS = [model_entry(d) for d in sorted(os.listdir(_SIG), reverse=True) if d[:7] in EXP_SPEC and _has_configs(d)]
assert [m["id"] for m in MODELS] == ["EXP-044", "EXP-039"], [m["id"] for m in MODELS]

# 규칙이 켜지는 파일 (라벨 안 씀). n = seed 5번 중 켜진 횟수. kind: up = 무조건 고장, cap = 점수 감점 대상
_g = np.arange(NTR)
_P83, _W83, _ = peak83_table()
_r83 = Rule83(0.75, 150)(_g)
_kn = KnockRule(2.0, 0.7).fit(_g)                                  # 설명용: 정상 150개 중심 (평가에서는 fold 안, 켜지는 파일은 25번 모두 같음)
_K = cached_table("knock", feature_table_knock, X_tr)
_HT = _hits()
_lockn = np.zeros(NTR, int)                                  # 주기 잠금 문턱(학습 정상 최대 + 0.1)은 fold마다 다름 → EXP-029와 같은 분할로 센다
for _s in range(5):
    for _tr, _te in StratifiedGroupKFold(5, shuffle=True, random_state=_s).split(_g, y_tr, _g):
        _lockn[_te[LockRule(0.1).fit(_tr)(_te)]] += 1
_n20 = _HT["HIT_n20"].values.astype(int)
_f = lambda v, nd=2: round(float(v), nd)
RULES = [
    dict(key="r83", label="8.3k 규칙", kind="up", since="EXP-021", n=(5 * _r83.astype(int)).tolist(),
         desc="8.20–8.45 kHz 최대 돌출 ≥ 0.75 dec 이고 봉우리 폭 ≥ 150 Hz면 무조건 고장",
         info=[f"돌출 {_f(p)} · 폭 {int(round(float(w)))} Hz" for p, w in zip(_P83, _W83)]),
    dict(key="knock", label="유격 규칙", kind="up", since="EXP-025", n=(5 * _kn(_g).astype(int)).tolist(),
         desc="정상 중심 거리 r(유격 반복·하중 대비 고주파) ≥ 2 이고 2위/1위 사건 ≥ 0.7이면 무조건 고장 (2-12)",
         info=[f"r {_f(a)} · 2위/1위 {_f(b)}" for a, b in zip(_kn.r(_g), _K["K3_top2_ratio"])]),
    dict(key="lock", label="주기 잠금 규칙", kind="up", since="EXP-029", n=_lockn.tolist(),
         desc="주기 잠금 점수 LOCK_S ≥ 학습 정상 최대 + 0.1이면 무조건 고장 (2-13). 문턱이 fold마다 달라 켜짐 횟수/5로 표시",
         info=[f"S {_f(v)}" for v in _HT["LOCK_S"].values]),
    dict(key="cap", label="단일 타격 감점", kind="cap", since="EXP-029", n=(5 * (_n20 == 1)).tolist(),
         desc="큰 타격(≥ 20σ)이 정확히 1개면 점수를 학습 정상 OOF 중앙값 이하로 누름 (2-14). 규칙으로 켜진 파일은 제외",
         info=[f"큰 타격 {v}개" for v in _n20]),
    dict(key="res83", label="8.3k 공진 규칙", kind="up", since="EXP-034", n=(5 * (_RES >= 0.7)).tolist(),
         desc="8.0–8.7 kHz로 울린 충격 중 감쇠비 ζ ≤ 0.02 이고 감쇠 사인파 맞춤 fit ≥ 0.7인 충격이 하나라도 있으면 무조건 고장 (2-16). 8.3k 규칙과 OR",
         info=[f"공진 fit {_f(v)}" if v > 0 else "8 kHz 깨끗한 공진 없음" for v in _RES]),
    dict(key="body", label="본체 모드 내림", kind="cap", since="EXP-034", n=(5 * _BODY).tolist(),
         desc="충격 3개 이상이고 그중 5.5–7.75 kHz(본체 모드)로 울린 비율 ≥ 0.9면 점수를 학습 정상 OOF 중앙값 이하로 누름 (2-16). 규칙으로 켜진 파일은 제외",
         info=[f"본체 울림 {_f(r) if r >= 0 else '-'} · 충격 {int(c)}개" for r, c in zip(RING["RING_BODY"], _cnt)]),
]
_E2N = _PM_E2 = json.load(open(os.path.join(ROOT, "단계", "2_EDA", "2-4_EDA뷰어", "figs", "plane_models.json"), encoding="utf-8"))["EXP-044"]["e2n"]
RULES.append(dict(key="e2", label="E2 전문가", kind="up", since="EXP-043", n=list(_E2N),
                  desc="H18_hfl(고주파 충격 에너지)이 학습 정상 99% 문턱을 넘고 H1_nimp ≥ ln 2이면 무조건 고장 (EXP-043 E2). 문턱이 fold마다 달라 켜짐 횟수/5로 표시",
                  info=[f"E2 켜짐 {v}/5" for v in _E2N]))
RULES = [r for r in RULES if r["key"] in {k for m in MODELS for k in m["rules"]}]      # 뷰어 모델이 쓰는 규칙만
for _ru in RULES:                                                  # 규칙 켜짐은 OOF 판정이라 train만. val·test(라벨 없음)는 0 · "—" (채점 안 함)
    _ru["n"] = list(_ru["n"]) + [0] * (N - NTR); _ru["info"] = list(_ru["info"]) + ["—"] * (N - NTR)

# %% 분석 조각 화면 (11-14): 2-12 유격 반복 지도 · 2-13 주기 잠금 점수. 각 조각 run.py가 남긴 파일별 값 표를 그대로 읽는다
import pandas as pd
_EDA = os.path.join(ROOT, "단계", "2_EDA")
_F12 = pd.read_csv(os.path.join(_EDA, "유격·반복충격/2-12_1114_저에너지고장", "figs", "features.csv"), index_col=0, encoding="utf-8-sig").loc[list(names_tr)]
_F13 = pd.read_csv(os.path.join(_EDA, "유격·반복충격/2-13_1114_주기잠금충격_022", "figs", "lock_score.csv"), index_col=0, encoding="utf-8-sig").loc[list(names_tr)]
assert (_F12.y.values == y_tr).all() and (_F13.y.values == y_tr).all()
# 2-12 값(D1·D6·R·사건 수·틱·σ·하중)을 run.py의 feats()와 같은 식으로 train·val 모두 계산하고, z1·z6은 train 정상(120)의 중앙값·MAD로 맞춘다.
# train 150개 값이 2-12 csv와 같은지 assert (같으면 csv 대신 계산값을 쓴다 → val도 같은 정의)
_SIG = 1.4826 * np.median(np.abs(H - np.median(H, 1, keepdims=True)), 1) + 1e-12


def _feats12(h, lo, sig):
    idx = np.flatnonzero(np.abs(h) > 8 * sig)
    gs = np.split(idx, np.flatnonzero(np.diff(idx) > 250) + 1) if len(idx) else []
    ev = np.array([g[np.argmax(np.abs(h[g]))] for g in gs], int)
    ev = ev[np.argsort(-np.abs(h[ev]))] if len(ev) else ev
    hg = np.abs(h[ev]) / sig
    i5 = np.flatnonzero(np.abs(h) > 5 * sig)
    return dict(n_events=len(ev), D1=float(np.log10(hg[1])) if len(hg) > 1 else float(np.log10(8)), D6=float(np.log10(lo.std() / sig)),
                D2=(len(np.split(i5, np.flatnonzero(np.diff(i5) > 250) + 1)) if len(i5) else 0) / 2.0,
                R=float(hg[1] / hg[0]) if len(hg) > 1 else 0.0, sigma_mg=sig * 1000, load_mg=lo.std() * 1000)


_F12c = pd.DataFrame([_feats12(H[i], L20_full[i], _SIG[i]) for i in range(N)], index=list(names))
_rz = lambda v: (v - np.median(v[nor])) / (1.4826 * np.median(np.abs(v[nor] - np.median(v[nor]))) + 1e-12)
_F12c["z1"], _F12c["z6"] = _rz(_F12c.D1.values), _rz(_F12c.D6.values)
_F12c["r"] = np.sqrt((_F12c.z1 ** 2 + _F12c.z6 ** 2) / 2)
for _c in ("n_events", "D1", "D6", "D2", "R", "z1", "z6", "r"):
    assert np.allclose(_F12c[_c].values[:NTR], _F12[_c].values, atol=2e-3), f"2-12 {_c}가 csv와 다름"
_S0 = pd.read_csv(os.path.join(ROOT, "단계", "4_피처", "4-6_1121_충격재정의", "사령탑_7b-7e", "S_새충격.csv"), encoding="utf-8-sig", index_col=0)
assert np.allclose(LOCKC["rlp"][:NTR], _S0.loc[list(names)[:NTR], "r_LP"].values, atol=1e-6) and np.allclose(LOCKC["T"][:NTR], _S0.loc[list(names)[:NTR], "T"].values, atol=1e-6)   # r_LP·T는 충격과 무관 → 사령탑 값과 같아야 한다
assert abs(LOCKC["S"][list(names).index("abn_022")] - 0.963) < 2e-3, "022 S가 기록 0.963과 다름"
_r = lambda s, nd=3: [round(float(v), nd) for v in s]
KNOCK2 = dict(z1=_r(_F12c.z1), z6=_r(_F12c.z6), r=_r(_F12c.r), R=_r(_F12c.R), d1=_r(_F12c.D1), d6=_r(_F12c.D6),
              nev=_F12c.n_events.astype(int).tolist(), d2=_r(_F12c.D2, 2), sig=_r(_F12c.sigma_mg, 2), load=_r(_F12c.load_mg, 2))
LOCK = dict(rlp=_r(LOCKC["rlp"]), T=_r(LOCKC["T"]), hit=_r(LOCKC["hit"]), nev=LOCKC["nev"].tolist(), S=_r(LOCKC["S"]))
_png = lambda p: "data:image/png;base64," + base64.b64encode(open(os.path.join(_EDA, p), "rb").read()).decode()
DOCFIGS = dict(
    knock=[dict(src=_png("유격·반복충격/2-12_1114_저에너지고장/figs/2_left_waves.png"), cap="왼쪽 무리(하중은 작은데 타격 반복) 파형 — 색 = 200 Hz 고역통과, 검은 선 = 20 Hz 저역(하중)")],
    lock=[dict(src=_png("유격·반복충격/2-13_1114_주기잠금충격_022/figs/1_022_wave.png"), cap="abn_022 — 느린 파형이 0.94 s마다 같은 모양으로 반복되고 충격이 그 주기에 잠김"),
          dict(src=_png("유격·반복충격/2-13_1114_주기잠금충격_022/figs/5_022_014_001.png"), cap="같은 유형 비교 — 022 · 014(약함) · 001(부분)")])

# %% 로지스틱 경계 (11-22): 모델마다 평면이 달라 plane_models.py(EXP-039 · EXP-044 run.py와 같은 정의, OOF seed 0)가 저장한 점 좌표 · 경계 w,b · 축 범위를 읽는다
#   결정값 0 = 그 실험의 최종 임계값(039) · 결합 문턱(044). 점선 ±1 = 로지스틱 결정값 ±1
_PM_PATH = os.path.join(ROOT, "단계", "2_EDA", "2-4_EDA뷰어", "figs", "plane_models.json")
assert os.path.isfile(_PM_PATH), "plane_models.py를 먼저 실행하세요 (figs/plane_models.json)"
_PM = json.load(open(_PM_PATH, encoding="utf-8"))
for m in MODELS:
    pm = _PM[m["id"]]; assert len(pm["pts"]) == NTR
    m.update(plane=m["id"], w=[round(v, 5) for v in pm["w"]], b=round(pm["b"], 5), lim=pm["lim"], xl=pm["xl"], yl=pm["yl"])
LR = dict(planes={m["id"]: _PM[m["id"]]["pts"] for m in MODELS}, plane_label={m["id"]: _PM[m["id"]]["label"] for m in MODELS})

# %% 그림 데이터 (int16·uint8로 줄임)
NB, NZ = 1000, 1000                         # 파형 포락 50 샘플(2 ms)씩 최소·최대 / 최대 충격 ±20 ms
q = lambda a, s: np.clip(np.round(a / s[:, None] * 32767), -32767, 32767).astype("<i2")
fix = lambda a, s: np.round(np.clip(a, -32 / s * 1000, 32 / s * 1000) * s).astype("<i2")    # 고정 배율
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def clean(a, nd=5):
    return [None if (isinstance(v, (float, np.floating)) and np.isnan(v)) else (round(float(v), nd) if isinstance(v, (float, np.floating)) else v) for v in a]


band_ref = lambda A: dict(p5=clean(np.percentile(A[nor], 5, 0), 3), p50=clean(np.median(A[nor], 0), 3),
                          p95=clean(np.percentile(A[nor], 95, 0), 3), f50=clean(np.median(A[abn], 0), 3))

# %% 11-21 개편: "신호 전환" 5개. 그림용 배열(파형·확대·PSD·스펙트로그램 두 종)은 5개 모두 만들고,
#    지표·태그·백분위(SETS)는 평균 제거(raw)·200 Hz 고역통과(hp) 두 벌만 유지한다 (빌드 시간·크기 절약).
#    새 신호(20 Hz·1 kHz·1.5 kHz)는 지표가 없어 뷰어가 가장 가까운 벌(20 Hz → raw, 1 kHz·1.5 kHz → hp)의 지표·태그·충격 눈금을 쓴다
H20 = sosfiltfilt(butter(4, 20, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
# H1K(1 kHz 고역통과)는 위 "11-21 2차" 블록에서 정의
H15 = sosfiltfilt(butter(4, 1500, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
SIGS_PLOT = {"raw": Xc, "hp20": H20, "hp": H, "hp1k": H1K, "hp15": H15}
SIG_NAMES = dict(raw="평균 제거", hp20="20 Hz 고역통과", hp="200 Hz 고역통과", hp1k="1 kHz 고역통과", hp15="1.5 kHz 고역통과")
SIG_FALLBACK = dict(hp20="raw", hp1k="hp", hp15="hp")


def plot_arrays(Z, A=None):
    """그림용: PSD(log10, Welch 4096, detrend=False) · 스펙트로그램(1024점) 64줄 로그 파워. A(analyze 결과)가 있으면 재사용"""
    if A is None:
        _, P_ = welch(Z, fs=FS, nperseg=4096, detrend=False); L = np.log10(P_ + 1e-30)
        _, _, S_ = spectrogram(Z, fs=FS, nperseg=1024, noverlap=512)
    else:
        L, S_ = A["L"], A["S"]
    return L, np.log10(S_[:, :512].reshape(N, 64, 8, -1).mean(2) + 1e-30)


PL = {s: plot_arrays(Z, RES[s][1] if s in RES else None) for s, Z in SIGS_PLOT.items()}
# 스펙트로그램 색 범위는 5개 신호 공통 (평균 제거 기준) → 고역통과에서 저주파 줄이 비는 것이 그대로 보인다
SLO, SHI = np.percentile(PL["raw"][1][:NTR], 1), np.percentile(PL["raw"][1][:NTR], 99.9)     # 색 범위도 train으로만


# 짧은 창 스펙트로그램 (11-21): 256점 = 10 ms 창, 50% 겹침 → 5.1 ms 간격, 98 Hz 칸 2개씩 묶어 195 Hz × 64줄. 충격 하나(울림 ~8 ms)를 보려는 용도.
# 신호 전환을 따르게 하되 크기를 줄이려고: 200 Hz 고역통과 신호는 64줄 전부, 나머지 신호는 맨 아래 16줄(0–3.1 kHz)만 따로 싣고
# 그 위(3.1 kHz 이상)는 200 Hz 고역통과 줄을 쓴다. 4차 버터워스 filtfilt는 차단의 2배(1.5 kHz → 3 kHz) 위에서 파워 차이가 1% 안쪽이다. 색 범위는 따로
def spec2_of(Z):
    fq, _, S = spectrogram(Z, fs=FS, nperseg=256, noverlap=128)                    # (N,129,389)
    return fq, np.log10(S[:, :128].reshape(N, 64, 2, -1).mean(2) + 1e-30)


fq2, S2h = spec2_of(H)
LO2, HI2 = np.percentile(S2h[:NTR], 1), np.percentile(S2h[:NTR], 99.9)
nrm2 = lambda S: np.clip(np.round((S - LO2) / (HI2 - LO2) * 255), 0, 255).astype(np.uint8)
SPEC2 = nrm2(S2h); del S2h
SPEC2_NLO = 16
SPEC2LO = {s: nrm2(spec2_of(Z)[1][:, :SPEC2_NLO]) for s, Z in SIGS_PLOT.items() if s != "hp"}


def pack_plot(s, Z):
    L, sl = PL[s]
    sc = np.abs(Z).max(1)
    st = np.clip(np.abs(Z).argmax(1) - NZ // 2, 0, N_SAMP - NZ)
    ZOOM = q(np.stack([Z[i, s0:s0 + NZ] for i, s0 in enumerate(st)]), sc)
    SPEC = np.clip(np.round((sl - SLO) / (SHI - SLO) * 255), 0, 255).astype(np.uint8)
    zr = Z.reshape(N, NB, -1)
    WAVE = np.stack([q(zr.min(2), sc), q(zr.max(2), sc)], 1)                          # (N,2,NB) 최소·최대
    b = dict(zoom=b64(ZOOM), psd=b64(fix(L, 1000)), spec=b64(SPEC), env=b64(WAVE))
    if s in SPEC2LO: b["spec2lo"] = b64(SPEC2LO[s])
    return dict(psdref=band_ref(L), zoom_start=st.tolist(), bin=b)


def pack_metrics(s):
    R = SETS[s]; V, A, pct, ref = R["V"], R["A"], R["pct"], R["ref"]
    return dict(
        val={k: clean(V[k]) for k, *_ in FEAT}, pct={k: clean(pct[k], 1) for k, *_ in FEAT},
        ref={k: {a: (None if np.isnan(b) else round(float(b), 5)) for a, b in v.items()} for k, v in ref.items()},
        tags=[dict(key=k, label=l, desc=d) for k, l, d, _ in R["TAGS"]], ftags=R["ftags"], events=A["events"])


# %% 11-21 저주파 탭 (신호 = 평균만 뺀 원신호 고정)
#   0–200 Hz PSD: Welch 8192점(3.05 Hz 칸) · 겹침 4096 · Hann · detrend=False (위 analyze의 73행 주석과 같은 이유: 느린 드리프트를 지우지 않으려고)
#   저역 파형: 200 Hz · 20 Hz 저역 (4차 버터워스 sosfiltfilt) → 1250 Hz로 줄여 2500점
from scipy.signal import find_peaks
fl, PL8 = welch(Xc, fs=FS, nperseg=8192, noverlap=4096, window="hann", detrend=False)
kl = fl <= 200
LOW_L = np.log10(PL8[:, kl] + 1e-30)                                                  # (N,66)
LOW_REF = band_ref(LOW_L)
lp_ = lambda fc: sosfiltfilt(butter(4, fc, btype="lowpass", fs=FS, output="sos"), Xc, axis=-1)
L20_full = lp_(20)
LP20, LP200 = L20_full[:, ::20], lp_(200)[:, ::20]
SC_LOW = np.maximum(np.abs(LP200).max(1), np.abs(LP20).max(1))


def top_peak_f(Lrow, f):
    """PSD에서 가장 높은 봉우리(양옆보다 높은 칸) 주파수. 가장자리 칸(0 Hz 포함)은 봉우리로 치지 않는다. 뷰어 JS도 같은 정의"""
    pk, _ = find_peaks(Lrow)
    return float(f[pk[Lrow[pk].argmax()]]) if len(pk) else float("nan")


LOW_PEAK = np.array([top_peak_f(LOW_L[i], fl[kl]) for i in range(N)])


def ctx_row(key, label, concept, v, unit="", desc=""):
    """맥락 숫자 한 줄: 값 + 정상 120개 중 백분위 + 정상 5–95%·중앙·고장 중앙"""
    v = np.asarray(v, float)
    return dict(key=key, label=label, concept=concept, unit=unit, desc=desc, val=clean(v, 4), pct=clean(pct_of(v), 1),
                ref={"p5": round(float(np.percentile(v[nor], 5)), 4), "p50": round(float(np.median(v[nor])), 4),
                     "p95": round(float(np.percentile(v[nor], 95)), 4), "f50": round(float(np.median(v[abn])), 4)})


_Vr = SETS["raw"]["V"]                                              # 평균 제거(raw) 기준 값
_rms = lambda a: np.sqrt((a ** 2).mean(1))
LOAD_RATIO = np.log10(_rms(H) / _rms(L20_full))                     # 200 Hz 이상 RMS / 20 Hz 저역 RMS
LOW_CTX = [
    ctx_row("e0_20", "0–20 Hz 에너지", "band-energy", _Vr["e0_20"], "log g²", "느린 출렁임(하중) 에너지, 평균 제거 신호의 Welch 4096 합"),
    ctx_row("e20_200", "20–200 Hz 에너지", "band-energy", _Vr["e20_200"], "log g²", "저주파 진동 에너지, 평균 제거 신호"),
    ctx_row("load_hf", "하중 대비 고주파", "rms", LOAD_RATIO, "log10", "200 Hz 이상 RMS / 20 Hz 저역 RMS. 하중이 작은데 고주파가 크면 높다"),
]

# %% 11-21 충격·박자 탭 (신호 = 1 kHz 고역통과 고정, src.final.impact_rings와 같은 신호)
#   충격 검출·울림 맞춤·포락선 스펙트럼·30 Hz 저역 파형은 위 "11-21 2차" 블록에서 train·val 모두 계산했다 (impact_rings 일치 assert 포함)
RHY_CTX = [
    ctx_row("n_ring", "충격 수 (새 정의 11-21)", "impulse-count", IMP_NV, "개", "1 kHz 고역 포락선이 100 ms 이동 중앙값 바탕의 4.5배 이상인 봉우리, 앞뒤 10 ms 제외 (src/impacts.py)"),
    ctx_row("rlp", "r_LP (느린 파형 반복)", "acf", LOCK["rlp"], "", "30 Hz 저역 신호를 T만큼 민 자기 자신과의 상관 최댓값 (2-13)"),
    ctx_row("hit", "hit (충격 주기 잠금)", "impulse-count", LOCK["hit"], "", "새 정의(11-21) 충격 중 T ± 3%에 짝이 있는 비율 (2-13 정의, 충격 시점만 새 정의)"),
    ctx_row("S", "S = r_LP × hit", "acf", LOCK["S"], "", "주기 잠금 점수 (2-13)"),
]

A0 = SETS["raw"]["A"]
D = dict(
    names=list(names), y=(y == 1).astype(int).tolist(), grp=y.tolist(), ntr=NTR, fs=FS, df=float(A0["df"]), nf=int(len(A0["f"])),
    groups=[dict(key=k, label=l, concept=c) for k, l, c in GROUPS],
    feat=[dict(key=k, label=l, unit=u, group=g, desc=d, concept=c) for k, l, u, g, d, c in FEAT],
    models=MODELS, rules=RULES, lr=LR, knock2=KNOCK2, lock=LOCK, docfigs=DOCFIGS, concepts=CONCEPTS, concept_intro=intro,
    nb=NB, nz=NZ,
    sc={s: clean(np.abs(Z).max(1), 6) for s, Z in SIGS_PLOT.items()},
    spec_shape=[64, int(A0["S"].shape[2])], spec_fmax=64 * 8 * float(A0["fs_"][1] - A0["fs_"][0]), imp_k=IMP_K,
    spec2=b64(SPEC2), spec2_shape=[64, int(SPEC2.shape[2])], spec2_fmax=64 * 2 * float(fq2[1] - fq2[0]), spec2_nlo=SPEC2_NLO,
    pbands=[dict(label="저 (200–2k)", lo=200, hi=2000), dict(label="중 (2–5k)", lo=2000, hi=5000), dict(label="고 (5–12.5k)", lo=5000, hi=12500)],
    sig_keys=list(SIGS_PLOT), sig_names=SIG_NAMES, sig_fallback=SIG_FALLBACK,
    sets={s: pack_metrics(s) for s in SIGS}, plots={s: pack_plot(s, Z) for s, Z in SIGS_PLOT.items()},
    low=dict(nf=int(kl.sum()), df=float(fl[1] - fl[0]), psd=b64(fix(LOW_L, 1000)), ref=LOW_REF, peak=clean(LOW_PEAK, 2),
             lp200=b64(q(LP200, SC_LOW)), lp20=b64(q(LP20, SC_LOW)), sc=clean(SC_LOW, 6), nlp=int(LP20.shape[1]), fs_lp=FS / 20, ctx=LOW_CTX),
    rhy=dict(env_f0=float(fe1[0]), env_df=float(fe1[1] - fe1[0]), env_nf=int(len(fe1)), envs=b64(fix(LE1, 1000)), envref=band_ref(LE1),
             wave=b64(ENV_P), nwave=int(ENV_P.shape[1]), wave_max=clean(ENV_MAX, 6), wave_med=clean(ENV_MED, 6),
             imp_t=IMP_ALL, imp_top=IMP_TOP, imp_n=IMP_NV,
             lock30=b64(q(L30, SC_30)), lock30_sc=clean(SC_30, 6), nlock=int(L30.shape[1]), lock_fs=FS / 50, ctx=RHY_CTX),
)

tpl = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
out = tpl.replace("__DATA_JSON__", json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
p = os.path.join(HERE, "EDA뷰어.html")
open(p, "w", encoding="utf-8").write(out)

# 확인용 요약 (UTF-8 파일)
from sklearn.metrics import roc_auc_score
lines = [f"HTML {os.path.getsize(p)/1e6:.1f} MB · 지표 {len(FEAT)}개 · 개념 {len(CONCEPTS)}개",
         "유의미 모델: " + " · ".join(f"{m['id']}({m['short']}, F2 {m['f2']}, 평면 {m['plane']})" for m in MODELS),
         "규칙 켜짐 파일 수: " + " · ".join(f"{r['label']} {sum(v > 0 for v in r['n'])}" for r in RULES)] + [f"경고: {w}" for w in WARN]
for s, R in SETS.items():
    V, ref = R["V"], R["ref"]; n_out = V["n_out"]
    lines += ["", f"=== 신호: {D['sig_names'][s]} ===", "태그별 파일 수 (정상/고장):"]
    for k, l, d, m in R["TAGS"]:
        lines.append(f"  {l}: {int((m & nor).sum())}/{int((m & abn).sum())}  ({d})")
    lines.append(f"정상 범위 밖 지표 수: 정상 중앙 {np.median(n_out[nor]):.0f} (95% {ref['n_out']['p95']:.0f}), 고장 중앙 {np.median(n_out[abn]):.0f}")
    aucs = sorted(((abs(roc_auc_score(y[trm], np.nan_to_num(V[k], nan=np.nanmedian(V[k][trm]))[trm]) - .5) + .5, k) for k in MAIN), reverse=True)
    lines.append("설명용 단변량 AUC 상위 (|AUC−0.5|+0.5): " + ", ".join(f"{k} {a:.2f}" for a, k in aucs[:15]))
# 11-21 개편 확인 숫자 (기록과 대조용)
from collections import Counter
_nm = [str(n) for n in names]
_i = lambda n: _nm.index(n)
# --- 11-21 2차 확인: 데이터셋 · 기준값 불변 · val 값 · 추천 쌍
from scipy.stats import spearmanr
lines += ["", "=== 11-21 2차 확인 ===", "데이터셋: " + " · ".join(f"{t} {n}개" for t, n in DS_LOADED) + (" (없는 폴더는 건너뜀)" if any(n == 0 for _, n in DS_LOADED) else "")]
_vi = np.flatnonzero(y == 2)
if len(_vi):
    lines.append(f"val 로드: 이름 {names[_vi[0]]}…{names[_vi[-1]]} ({len(_vi)}개) · 신호 길이 {X.shape[1]} (전부 50000)")
    _mn = X[_vi].mean(1)
    lines.append("val 중 평균 < −0.5 g: " + (", ".join(f"{names[i]} ({_mn[j]:.2f} g)" for j, i in enumerate(_vi) if _mn[j] < -0.5) or "없음") + f" · val 평균 최소 {_mn.min():.2f} g · train 정상 평균 최소 {X_tr[y_tr == 0].mean(1).min():.2f} g")
    _top = _vi[np.argsort(-LOCKC["S"][_vi])[:5]]
    lines.append("val 주기 잠금 S 상위 5: " + ", ".join(f"{names[i]} {LOCKC['S'][i]:.3f}(T {LOCKC['T'][i]:.2f})" for i in _top) + f" · train 정상 최대 {LOCKC['S'][nor].max():.3f}({names[int(np.flatnonzero(nor)[LOCKC['S'][nor].argmax()])]}) · val 중 정상 최대 초과 {int((LOCKC['S'][_vi] > LOCKC['S'][nor].max()).sum())}개")
# 기준값이 train 정상만으로 정해지는지: ① 모든 지표의 ref가 train 행만으로 다시 계산한 값과 같음 ② 신호 일부 지표는 train 150개만으로 따로 계산해도 같음
for _s_, _R in SETS.items():
    _V = _R["V"]
    for _k, *_ in FEAT[:-1]:
        _v = _V[_k][:NTR]; _vn = _v[(y_tr == 0) & ~np.isnan(_v)]
        assert np.allclose([np.percentile(_vn, 5), np.median(_vn), np.percentile(_vn, 95)], [_R["ref"][_k]["p5"], _R["ref"][_k]["p50"], _R["ref"][_k]["p95"]], rtol=1e-9, atol=1e-12), f"{_s_}/{_k} 기준값이 train 정상만의 값이 아님"
_Zt = Xc[:NTR]; _Ht = H[:NTR]
_chk = dict(std=(Xc.std(1), _Zt.std(1)), kurt=(SETS["raw"]["V"]["kurt"], kurtosis(_Zt, axis=1)), e200_hp=(SETS["hp"]["V"]["std"], _Ht.std(1)))
_cs = []
for _k, (_a, _b) in _chk.items():
    _p1 = np.percentile(_a[:NTR][y_tr == 0], [5, 95]); _p2 = np.percentile(_b[y_tr == 0], [5, 95]); assert np.allclose(_p1, _p2); _cs.append(f"{_k} p5/p95 {_p1[0]:.4g}/{_p1[1]:.4g}")
lines.append("val 추가 전후 기준값 동일 (assert 통과): 전 지표 ref = train 정상 행만의 값 · train 150개만으로 따로 계산한 " + " · ".join(_cs))
_PAIRS = [("1 고주파 첨도(1 kHz) × 4–8 kHz 울림 몫 (새 정의)", "kurt_1k", "ring_o3"), ("2 2–4 kHz 울림 몫 × 8.3 kHz 공진 fit (새 정의)", "ring_o2", "res83"),
          ("3 빨리 꺼지는 충격 비율 × 충격 수", "fast_decay", "n_ring"), ("4 주기 잠금 S × 하중 대비 고주파", "lb_S", "load_hf"),
          ("5 0–20 Hz 에너지 × 고주파 RMS", "lb_e0_20", "rms_hf")]
_Vx = SETS["raw"]["V"]
def _auc(v, m):
    ok = m & ~np.isnan(v); a = roc_auc_score(y[ok], v[ok]); return a
for _lab, _kx, _ky in _PAIRS:
    _o = []
    for _nm_, _m in (("전체", trm), ("정상", nor), ("고장", abn)):
        _ok = _m & ~np.isnan(_Vx[_kx]) & ~np.isnan(_Vx[_ky]); _o.append(f"{_nm_} ρ {spearmanr(_Vx[_kx][_ok], _Vx[_ky][_ok])[0]:.2f}(n={_ok.sum()})")
    lines.append(f"추천 쌍 {_lab}: " + " · ".join(_o) + f" | AUC x({_kx}) {_auc(_Vx[_kx], trm):.2f} · y({_ky}) {_auc(_Vx[_ky], trm):.2f}")
lines += ["", "=== 11-21 개편 확인 ===", IMP_CHECK, f"신호 {len(SIGS_PLOT)}개: " + " · ".join(SIG_NAMES.values()) + " (지표·태그는 raw·hp 두 벌만, 나머지는 가까운 벌을 씀)"]
_a = _i("abn_015"); _pk = find_peaks(LOW_L[_a])[0]; _top = _pk[np.argsort(-LOW_L[_a][_pk])[:3]]
lines.append(f"abn_015 0–200 Hz PSD: 최대 봉우리 {LOW_PEAK[_a]:.2f} Hz (기록 35–44 Hz) · 상위 봉우리 " + ", ".join(f"{fl[kl][k]:.1f} Hz" for k in _top)
             + f" · 0 Hz 칸 제외 단순 최댓값 {fl[kl][1 + LOW_L[_a][1:].argmax()]:.2f} Hz")
_b = _i("abn_022")
lines.append(f"abn_014 S {LOCK['S'][_i('abn_014')]} · 정상 최대 S {max(LOCK['S'][k] for k in np.flatnonzero(nor))} · val 최대 S {max(LOCK['S'][k] for k in np.flatnonzero(y == 2))}")
lines.append(f"abn_022 주기 잠금: S {LOCK['S'][_b]} (기록 0.963) · T {LOCK['T'][_b]} s (기록 약 0.94) · r_LP {LOCK['rlp'][_b]} · hit {LOCK['hit'][_b]} · 새 정의 충격 {LOCK['nev'][_b]}개")
for _n in ("abn_022", "abn_002"):
    _r = IMP_ROWS[_i(_n)]
    lines.append(f"{_n} 충격 {len(_r)}개 (전체 검출 {len(IMP_ALL[_i(_n)])}개): ζ 분포 " + ", ".join(f"{z:g}: {c}" for z, c in sorted(Counter(round(float(r[2]), 2) for r in _r).items()))
                 + " · 울림 f(kHz) " + ", ".join(f"{r[1] / 1000:g}" for r in _r[:12]) + (" …" if len(_r) > 12 else ""))
lines.append(f"abn_024 충격 {IMP_NV[_i('abn_024')]}개 · abn_015 충격 {IMP_NV[_i('abn_015')]}개 · 충격 총합 정상 {int(np.array(IMP_NV)[nor].sum())} · 고장 {int(np.array(IMP_NV)[abn].sum())} · val {int(np.array(IMP_NV)[y == 2].sum())}")
lines.append(f"충격 수 분포: 정상 중앙 {np.median(np.array(IMP_NV)[nor]):.0f} · 고장 중앙 {np.median(np.array(IMP_NV)[abn]):.0f} · 최대 {max(IMP_NV)}개 (표는 fit 상위 30개)")
lines.append(f"HTML 크기: 개편 전 13.58 MB(13,578,280 B) → 개편 후 {os.path.getsize(p):,} B ({os.path.getsize(p)/1e6:.1f} MB)")
open(os.path.join(HERE, "build_log.txt"), "w", encoding="utf-8").write("\n".join(lines))

# 다른 조각(2-0 유형 산포도 등)이 같은 값을 쓰도록 파일별 지표 표를 남긴다 (신호별 한 장)
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
for s, R in SETS.items():
    df = pd.DataFrame({k: R["V"][k][:NTR] for k, *_ in FEAT}, index=list(names_tr)); df.insert(0, "y", y_tr)   # 다른 조각이 읽는 표라 train만
    df.to_csv(os.path.join(HERE, "figs", f"features_{s}.csv"), encoding="utf-8-sig", float_format="%.6g")
print("done")
