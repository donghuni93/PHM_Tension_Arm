# %% [markdown]
# 2-4 EDA 뷰어 만들기 — 150개 파일의 시간·주파수 영역 특징을 HTML 한 장으로
# - 지표: 개념 사전(단계/개념사전.md)의 방법 중 계산 가능한 것. 묶음 8개 (기본 크기 · 형상 · 충격·비정상성 · 저주파 ·
#   스펙트럼 모양 · 물리 대역 · 포락선 · 1 kHz 대역). 정상 대비 백분위, 설명용 태그
# - 그림 데이터: 파형 포락(2초), 최대 충격 ±20 ms, 창별 RMS·첨도, PSD, 스펙트럼 첨도, 포락선 스펙트럼, 스펙트로그램
# - 모델 결과: 최종 EXP-025(EXP-014 결합 + 8.3 kHz 규칙 + 유격 반복 규칙)의 seed 5번 중 미탐·오경보 횟수 + 두 규칙 켜짐
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
X, y, names = load_train()
N = len(y); nor = y == 0
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


SIGS = {"raw": Xc, "hp": H}
RES = {s: analyze(Z) for s, Z in SIGS.items()}

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

# %% 정상 대비 백분위·기준값
def pct_of(v):
    vn = v[nor & ~np.isnan(v)]
    return np.array([np.nan if np.isnan(a) else 100 * ((vn < a).mean() + 0.5 * (vn == a).mean()) for a in v])


MAIN = [k for k, _, _, g, *_ in FEAT if g != "band"]
FEAT.append(("n_out", "정상 범위 밖 지표 수", "개", "summary", f"1 kHz 대역을 뺀 지표 {len(MAIN)}개 중 정상 5–95% 밖인 개수. 정상도 우연히 몇 개는 밖으로 나간다", "collinearity"))


def stats(V):
    pct, ref = {}, {}
    for k, *_ in FEAT[:-1]:
        v = V[k]; vn = v[nor & ~np.isnan(v)]; vf = v[~nor & ~np.isnan(v)]
        pct[k] = pct_of(v)
        ref[k] = dict(p5=np.percentile(vn, 5), p50=np.median(vn), p95=np.percentile(vn, 95), f50=np.median(vf) if len(vf) else np.nan)
    n_out = np.array([sum(1 for k in MAIN if not np.isnan(pct[k][i]) and (pct[k][i] < 5 or pct[k][i] > 95)) for i in range(N)])
    V["n_out"] = n_out.astype(float); pct["n_out"] = pct_of(V["n_out"])
    ref["n_out"] = dict(p5=np.percentile(n_out[nor], 5), p50=np.median(n_out[nor]), p95=np.percentile(n_out[nor], 95), f50=np.median(n_out[~nor]))
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

# %% 모델 결과 (seed 5번 중 횟수)
def model_counts(exp_dir, short, cfg=None):
    """cfg가 없으면 그 실험의 대표 구성(main), main도 없으면 첫 구성"""
    r = json.load(open(os.path.join(ROOT, "단계", "5_모델비교", exp_dir, "result.json"), encoding="utf-8"))
    name = cfg or r.get("main") or next(iter(r["configs"]))
    c = r["configs"][name]
    return dict(name=f"{r['id']} {name}", short=short, f2=c["f2"], f2_sd=c["f2_sd"], recall=c["recall"], fpr=c.get("fpr"),
                miss=[int(c["miss_counts"].get(n, 0)) for n in names], fp=[int(c["fp_counts"].get(n, 0)) for n in names])


# 11-14: 최종 모델만 = EXP-025 (EXP-014 결합 + 8.3 kHz 모드 규칙 + 유격 반복 규칙). AE·ET·결합 비교는 뺌 (사용자 결정)
MODELS = {"fin": model_counts("유의미/EXP-025_1114_유격반복규칙", "최종")}
MODEL_ORDER = ["fin"]
from src.experts import peak83_table, Rule83, KnockRule             # 최종 모델의 두 규칙이 켜지는 파일 (라벨 안 씀)
from src.features import feature_table_knock
_P83, _W83, _ = peak83_table()
RULE83 = dict(on=Rule83(0.75, 150)(np.arange(len(names))).astype(int).tolist(),
              p83=[round(float(v), 2) for v in _P83], w83=[int(round(float(v))) for v in _W83])
_kn = KnockRule(2.0, 0.7).fit(np.arange(len(names)))              # 설명용: 정상 150개 중심 (평가에서는 fold 안, 켜지는 파일은 25번 모두 같음)
_K = cached_table("knock", feature_table_knock, X)
KNOCK = dict(on=_kn(np.arange(len(names))).astype(int).tolist(),
             r=[round(float(v), 2) for v in _kn.r(np.arange(len(names)))], k3=[round(float(v), 2) for v in _K["K3_top2_ratio"]])

# %% 분석 조각 화면 (11-14): 2-12 유격 반복 지도 · 2-13 주기 잠금 점수. 각 조각 run.py가 남긴 파일별 값 표를 그대로 읽는다
import pandas as pd
_EDA = os.path.join(ROOT, "단계", "2_EDA")
_F12 = pd.read_csv(os.path.join(_EDA, "2-12_1114_저에너지고장", "figs", "features.csv"), index_col=0, encoding="utf-8-sig").loc[list(names)]
_F13 = pd.read_csv(os.path.join(_EDA, "2-13_1114_주기잠금충격_022", "figs", "lock_score.csv"), index_col=0, encoding="utf-8-sig").loc[list(names)]
assert (_F12.y.values == y).all() and (_F13.y.values == y).all()
_r = lambda s, nd=3: [round(float(v), nd) for v in s]
KNOCK2 = dict(z1=_r(_F12.z1), z6=_r(_F12.z6), r=_r(_F12.r), R=_r(_F12.R), d1=_r(_F12.D1), d6=_r(_F12.D6),
              nev=_F12.n_events.astype(int).tolist(), d2=_r(_F12.D2, 2), sig=_r(_F12.sigma_mg, 2), load=_r(_F12.load_mg, 2))
LOCK = dict(rlp=_r(_F13.r_LP), T=_r(_F13["T"]), hit=_r(_F13.hit), nev=_F13.n_ev.astype(int).tolist(), S=_r(_F13.S))
_png = lambda p: "data:image/png;base64," + base64.b64encode(open(os.path.join(_EDA, p), "rb").read()).decode()
DOCFIGS = dict(
    knock=[dict(src=_png("2-12_1114_저에너지고장/figs/2_left_waves.png"), cap="왼쪽 무리(하중은 작은데 타격 반복) 파형 — 색 = 200 Hz 고역통과, 검은 선 = 20 Hz 저역(하중)")],
    lock=[dict(src=_png("2-13_1114_주기잠금충격_022/figs/1_022_wave.png"), cap="abn_022 — 느린 파형이 0.94 s마다 같은 모양으로 반복되고 충격이 그 주기에 잠김"),
          dict(src=_png("2-13_1114_주기잠금충격_022/figs/5_022_014_001.png"), cap="같은 유형 비교 — 022 · 014(약함) · 001(부분)")])

# %% 로지스틱 경계 화면 (11-14): 두 전문가 z 평면(AE z, ET z, OOF seed 0) + 유의미 실험별 경계·규칙·틀린 횟수
#   단계/5_모델비교/fig_logistic_plane.py와 같은 정의. 경계 = 그 평면에 맞춘 로지스틱 결정값 0 (설명용, 실제 임계값은 fold마다)
from src.experts import ScoreStack, ae_mean, et_logit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
_g = np.arange(len(names)); _F = _g[:, None]
_Z = np.empty((len(names), 2))
for _tr, _te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(_g, y, _g):
    _Z[_te] = ScoreStack(experts=(ae_mean, et_logit), combine="lr").fit(_F[_tr], y[_tr]).zscores(_F[_te])[:, :2]
_Zc = np.clip(_Z, -3, 4.5)
LR_EXPS = []
for folder, label, kind, rules in (
        ("유의미/EXP-025_1114_유격반복규칙", "EXP-025 최종 = 결합 + 8.3k 규칙 + 유격 규칙", "both", ["r83", "knock"]),
        ("유의미/EXP-021_1114_8k모드규칙", "EXP-021 = 결합 + 8.3k 규칙", "both", ["r83"]),
        ("유의미/EXP-020_1114_8k봉우리축", "EXP-020 (8–9 kHz 축은 평면 밖, EXP-014 경계)", "both", []),
        ("유의미/EXP-014_1114_ET·AE결합", "EXP-014 = AE + ET 로지스틱 결합", "both", []),
        ("유의미/EXP-012_1113_스펙트럼AE_tSNE", "EXP-012 = AE 단독 (AE z 한 축)", "ae", [])):
    _m = LogisticRegression(C=1.0, class_weight="balanced").fit(_Zc[:, [0] if kind == "ae" else [0, 1]], y)
    w = _m.coef_[0].tolist() + ([0.0] if kind == "ae" else [])
    mc = model_counts(folder, label)
    LR_EXPS.append(dict(id=mc["name"].split()[0], label=label, w=[round(v, 5) for v in w], b=round(float(_m.intercept_[0]), 5),
                        rules=rules, f2=mc["f2"], f2_sd=mc["f2_sd"], recall=mc["recall"], fpr=mc["fpr"], miss=mc["miss"], fp=mc["fp"]))
LR = dict(z=[[round(float(a), 3), round(float(b), 3)] for a, b in _Z], exps=LR_EXPS)

# %% 그림 데이터 (int16·uint8로 줄임)
NB, NZ = 1000, 1000                         # 파형 포락 50 샘플(2 ms)씩 최소·최대 / 최대 충격 ±20 ms
sc_x = np.abs(Xc).max(1); sc_h = np.abs(H).max(1)
q = lambda a, s: np.clip(np.round(a / s[:, None] * 32767), -32767, 32767).astype("<i2")
xr, hr = Xc.reshape(N, NB, -1), H.reshape(N, NB, -1)
ENV = np.stack([q(xr.min(2), sc_x), q(xr.max(2), sc_x), q(hr.min(2), sc_h), q(hr.max(2), sc_h)], 1)
fix = lambda a, s: np.round(np.clip(a, -32 / s * 1000, 32 / s * 1000) * s).astype("<i2")    # 고정 배율
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def clean(a, nd=5):
    return [None if (isinstance(v, (float, np.floating)) and np.isnan(v)) else (round(float(v), nd) if isinstance(v, (float, np.floating)) else v) for v in a]


band_ref = lambda A: dict(p5=clean(np.percentile(A[nor], 5, 0), 3), p50=clean(np.median(A[nor], 0), 3),
                          p95=clean(np.percentile(A[nor], 95, 0), 3), f50=clean(np.median(A[~nor], 0), 3))
# 스펙트로그램 색 범위는 두 신호 공통 (평균 제거 기준) → 고역통과에서 저주파 줄이 비는 것이 그대로 보인다
spec_log = {s: np.log10(R["A"]["S"][:, :512].reshape(N, 64, 8, -1).mean(2) + 1e-30) for s, R in SETS.items()}
SLO, SHI = np.percentile(spec_log["raw"], 1), np.percentile(spec_log["raw"], 99.9)


def pack(s, Z):
    R = SETS[s]; V, A, pct, ref = R["V"], R["A"], R["pct"], R["ref"]
    sc = np.abs(Z).max(1)
    st = np.clip(np.abs(Z).argmax(1) - NZ // 2, 0, N_SAMP - NZ)
    ZOOM = q(np.stack([Z[i, s0:s0 + NZ] for i, s0 in enumerate(st)]), sc)
    LSK = np.log10(1 + np.clip(A["SK"], 0, None))                                    # 스펙트럼 첨도 → log10(1+SK)
    WR, WK = np.log10(A["WRMS"] + 1e-12), np.log10(1 + np.clip(A["WKURT"], 0, None))
    WQ = np.stack([fix(WR, 1000), fix(WK, 1000)], 1)                                  # (N,2,40)
    SPEC = np.clip(np.round((spec_log[s] - SLO) / (SHI - SLO) * 255), 0, 255).astype(np.uint8)
    return dict(
        val={k: clean(V[k]) for k, *_ in FEAT}, pct={k: clean(pct[k], 1) for k, *_ in FEAT},
        ref={k: {a: (None if np.isnan(b) else round(float(b), 5)) for a, b in v.items()} for k, v in ref.items()},
        tags=[dict(key=k, label=l, desc=d) for k, l, d, _ in R["TAGS"]], ftags=R["ftags"], events=A["events"],
        psdref=band_ref(A["L"]), skref=band_ref(LSK), envref=band_ref(A["LE"]),
        wref=dict(rms_p95=float(np.percentile(WR[nor], 95)), kurt_p95=float(np.percentile(WK[nor], 95))),
        zoom_start=st.tolist(),
        bin=dict(zoom=b64(ZOOM), psd=b64(fix(A["L"], 1000)), spec=b64(SPEC), sk=b64(fix(LSK, 1000)), envs=b64(fix(A["LE"], 1000)), win=b64(WQ)),
    )


A0 = SETS["raw"]["A"]
D = dict(
    names=list(names), y=y.tolist(), fs=FS, df=float(A0["df"]), nf=int(len(A0["f"])),
    groups=[dict(key=k, label=l, concept=c) for k, l, c in GROUPS],
    feat=[dict(key=k, label=l, unit=u, group=g, desc=d, concept=c) for k, l, u, g, d, c in FEAT],
    models=MODELS, model_order=MODEL_ORDER, rule83=RULE83, knock=KNOCK, lr=LR, knock2=KNOCK2, lock=LOCK, docfigs=DOCFIGS, concepts=CONCEPTS, concept_intro=intro,
    sk_df=float(A0["fs_"][1] - A0["fs_"][0]), sk_nf=int(len(A0["fs_"])), env_f0=float(A0["fe"][0]), env_df=float(A0["fe"][1] - A0["fe"][0]), env_nf=int(len(A0["fe"])),
    nw=NW, win_s=WIN / FS, nb=NB, nz=NZ,
    sc={s: clean(np.abs(Z).max(1), 6) for s, Z in SIGS.items()},
    spec_shape=[64, int(A0["S"].shape[2])], spec_fmax=64 * 8 * float(A0["fs_"][1] - A0["fs_"][0]), imp_k=IMP_K,
    pbands=[dict(label="저 (200–2k)", lo=200, hi=2000), dict(label="중 (2–5k)", lo=2000, hi=5000), dict(label="고 (5–12.5k)", lo=5000, hi=12500)],
    gbands=[dict(label="1차 공진? 100–300", lo=100, hi=300), dict(label="고차 모드? 1–3k", lo=1000, hi=3000), dict(label="링잉? 5–10k", lo=5000, hi=10000)],
    sig_names=dict(raw="평균 제거", hp="200 Hz 고역통과"),
    env=b64(ENV),
    sets={s: pack(s, Z) for s, Z in SIGS.items()},
)

tpl = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
out = tpl.replace("__DATA_JSON__", json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
p = os.path.join(HERE, "EDA뷰어.html")
open(p, "w", encoding="utf-8").write(out)

# 확인용 요약 (UTF-8 파일)
from sklearn.metrics import roc_auc_score
lines = [f"HTML {os.path.getsize(p)/1e6:.1f} MB · 지표 {len(FEAT)}개 · 개념 {len(CONCEPTS)}개"]
for s, R in SETS.items():
    V, ref = R["V"], R["ref"]; n_out = V["n_out"]
    lines += ["", f"=== 신호: {D['sig_names'][s]} ===", "태그별 파일 수 (정상/고장):"]
    for k, l, d, m in R["TAGS"]:
        lines.append(f"  {l}: {int((m & nor).sum())}/{int((m & ~nor).sum())}  ({d})")
    lines.append(f"정상 범위 밖 지표 수: 정상 중앙 {np.median(n_out[nor]):.0f} (95% {ref['n_out']['p95']:.0f}), 고장 중앙 {np.median(n_out[~nor]):.0f}")
    aucs = sorted(((abs(roc_auc_score(y, np.nan_to_num(V[k], nan=np.nanmedian(V[k]))) - .5) + .5, k) for k in MAIN), reverse=True)
    lines.append("설명용 단변량 AUC 상위 (|AUC−0.5|+0.5): " + ", ".join(f"{k} {a:.2f}" for a, k in aucs[:15]))
open(os.path.join(HERE, "build_log.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("done")
