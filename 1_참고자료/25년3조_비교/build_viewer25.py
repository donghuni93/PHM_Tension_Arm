# %% [markdown]
# 25년 3조 뷰어 만들기 — 25년 데이터 190개(train 150 + val 40)를 2-4 EDA 뷰어와 같은 화면으로 + 3조 노트북 + 25↔26 비교
# - 지표·태그 정의는 3_단계/2_EDA/EDA뷰어/build_viewer.py와 같다. 5 kHz × 10초에 맞춰 바꾼 것:
#   창 50 ms × 200개, 물리 대역 200–1k / 1–2k / 2–2.5k, 고정 격자 250 Hz × 10칸, 고역 비 = 2 kHz 이상,
#   STFT 256 (51 ms), PSD 바탕 이동 중앙값 155칸(≈190 Hz)
# - 정상 백분위·태그 기준은 train 정상 120개로만. val은 라벨이 없어 값만 놓는다
# - 모델 결과 = pipeline_3조.py의 3조 SVM_C2 (중첩 CV 미탐/오경보 횟수, OOF 확률, val 확률)
# 실행: conda run -n qe-phm python 1_참고자료/25년3조_비교/build_viewer25.py  → 같은 폴더 뷰어25.html (먼저 pipeline_3조.py)

# %% 설정
import os, re, sys, json, base64, csv, io
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT); sys.path.insert(0, HERE)
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch, spectrogram, hilbert, decimate
from scipy.ndimage import median_filter
from scipy.stats import kurtosis, skew, spearmanr
from load25 import load25, SRC
from src.exp import load_train

FS, N_SAMP = 5000, 50000
DUR = N_SAMP / FS
X, y, names = load25()
N = len(y); nor, abn, trn = y == 0, y == 1, y >= 0
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
IMP_K, IMP_GAP = 8.0, int(0.010 * FS)
WIN = int(0.050 * FS)                       # 50 ms 창 → 200개
NW = N_SAMP // WIN
PB = [("lo", 200, 1000), ("mid", 1000, 2000), ("hi", 2000, 2501)]
BANDS = [(k * 250, (k + 1) * 250) for k in range(10)]
RES = json.load(open(os.path.join(HERE, "4_outputs", "results_3조.json"), encoding="utf-8"))


# %% 분석 한 벌 (build_viewer.py analyze와 같은 정의, 숫자만 5 kHz용)
def analyze(Z):
    V, A = {}, {}
    ax = np.abs(Z)
    V["mean"] = X.mean(1); V["std"] = Z.std(1); V["maxabs"] = ax.max(1)
    V["p2p"] = Z.max(1) - Z.min(1); V["mav"] = ax.mean(1)
    V["kurt"] = kurtosis(Z, axis=1); V["skew"] = skew(Z, axis=1)
    V["crest"] = V["maxabs"] / V["std"]; V["impulse_f"] = V["maxabs"] / V["mav"]
    V["shape_f"] = V["std"] / V["mav"]; V["clear_f"] = V["maxabs"] / np.sqrt(ax).mean(1) ** 2
    d1 = np.diff(Z, axis=1); d2 = np.diff(d1, axis=1)
    mob = lambda a, da: np.sqrt(da.var(1) / a.var(1))
    V["hjorth_mob"] = mob(Z, d1) * FS / (2 * np.pi)
    V["hjorth_cpx"] = mob(d1, d2) / mob(Z, d1)

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

    f, P = welch(Z, fs=FS, nperseg=4096, detrend=False)            # 1.22 Hz 간격
    df = f[1] - f[0]
    band_e = lambda lo, hi: np.log10(P[:, (f >= lo) & (f < hi)].sum(1) * df + 1e-30)
    tot = P.sum(1)
    V["s20"] = P[:, f < 20].sum(1) / tot; V["s200"] = P[:, f < 200].sum(1) / tot
    V["e0_20"] = band_e(0, 20); V["e20_200"] = band_e(20, 200)
    V["e200"] = band_e(200, 2501)
    a1 = f > 0
    Pa, fa = P[:, a1], f[a1]; Pn = Pa / Pa.sum(1, keepdims=True)
    V["centroid"] = (Pn * fa).sum(1)
    V["bandwidth"] = np.sqrt((Pn * (fa[None] - V["centroid"][:, None]) ** 2).sum(1))
    cum = np.cumsum(Pn, 1)
    V["rolloff85"] = fa[(cum < 0.85).sum(1)]; V["rolloff95"] = fa[(cum < 0.95).sum(1)]
    V["hi_ratio"] = np.log10(P[:, f >= 2000].sum(1) / Pa.sum(1))
    V["flat"] = np.exp(np.log(Pa + 1e-30).mean(1)) / Pa.mean(1)
    V["entropy"] = -(Pn * np.log(Pn + 1e-30)).sum(1) / np.log(Pn.shape[1])
    L = np.log10(P + 1e-30)
    base = median_filter(L, size=(1, 155), mode="nearest")      # ≈190 Hz 이동 중앙값
    prom = (L - base)[:, a1]
    V["peak_f"] = fa[prom.argmax(1)]; V["peak_prom"] = prom.max(1)
    for k, lo, hi in PB:
        V[f"e_{k}"] = band_e(lo, hi)
        V[f"r_{k}"] = P[:, (f >= lo) & (f < hi)].sum(1) / Pa.sum(1)
    for k, (lo, hi) in enumerate(BANDS):
        V[f"b{k}"] = band_e(lo, hi)
    A.update(f=f, df=df, L=L)

    fs_, ts_, S = spectrogram(Z, fs=FS, nperseg=256, noverlap=128)         # (N,129,389)
    SK = (S ** 2).mean(2) / (S.mean(2) ** 2 + 1e-60) - 2
    k1 = fs_ > 0
    V["sk_max"] = SK[:, k1].max(1); V["sk_f"] = fs_[k1][SK[:, k1].argmax(1)]
    A.update(fs_=fs_, S=S, SK=SK)

    ENVS = np.abs(hilbert(Z, axis=1))
    V["env_kurt"] = kurtosis(ENVS, axis=1)
    Ed = ENVS.reshape(N, -1, 2).mean(2); Ed = Ed - Ed.mean(1, keepdims=True)  # 2.5 kHz
    fe, PE = welch(Ed, fs=FS / 2, nperseg=2500)                               # 1 Hz 간격
    ke = (fe >= 0.5) & (fe <= 300)
    fe, LE = fe[ke], np.log10(PE[:, ke] + 1e-30)
    pe = LE - median_filter(LE, size=(1, 21), mode="nearest")
    ke1 = fe >= 1
    V["env_pf"] = fe[ke1][pe[:, ke1].argmax(1)]; V["env_prom"] = pe[:, ke1].max(1)
    A.update(fe=fe, LE=LE)
    return V, A


SIGS = {"raw": Xc, "hp": H}
RUN = {s: analyze(Z) for s, Z in SIGS.items()}

# %% 지표 설명 (build_viewer.py와 같은 키·개념, 대역 이름만 5 kHz용)
GROUPS = [("basic", "기본 크기", "rms"), ("shape", "형상 지표", "shape-factors"), ("impulse", "충격·비정상성", "window-stats"),
          ("low", "저주파", "bandpass"), ("spec", "스펙트럼 모양", "spectral-shape"), ("pband", "대역 3분할", "band-split"),
          ("env", "포락선·스펙트럼 첨도", "envelope"), ("band", "250 Hz 대역 에너지", "band-energy")]
FEAT = [
    ("mean", "평균 (원신호)", "g", "basic", "설치 자세·중력 성분. 25년 데이터는 이미 거의 0 (평균이 빠진 채 배포된 것으로 보임)", "mean-removal"),
    ("std", "RMS", "g", "basic", "진동 크기", "rms"),
    ("maxabs", "Peak (최대 |진폭|)", "g", "basic", "가장 큰 순간값", "rms"),
    ("p2p", "Peak-to-Peak", "g", "basic", "최대 − 최소", "rms"),
    ("mav", "MAV (평균 |x|)", "g", "basic", "절대값 평균", "rms"),
    ("kurt", "첨도 (Fisher)", "", "shape", "바탕 대비 튀는 정도. 정규 0, 사인 −1.5 (Pearson으로는 +3)", "kurtosis"),
    ("skew", "왜도", "", "shape", "비대칭", "skewness"),
    ("crest", "파고율", "", "shape", "Peak / RMS", "shape-factors"),
    ("impulse_f", "충격 계수", "", "shape", "Peak / MAV", "shape-factors"),
    ("shape_f", "형상 계수", "", "shape", "RMS / MAV (정규 잡음 ≈ 1.25)", "shape-factors"),
    ("clear_f", "여유 계수", "", "shape", "Peak / (평균 √|x|)²", "shape-factors"),
    ("hjorth_mob", "Hjorth 이동도", "Hz", "shape", "평균 주파수 근사", "hjorth"),
    ("hjorth_cpx", "Hjorth 복잡도", "", "shape", "정현파에서 벗어난 정도 (정현파 = 1)", "hjorth"),
    ("n_imp", "충격 수", "개", "impulse", "|신호| > 8σ(MAD) 사건 수, 10 ms 안은 하나 (10초 기준 — 26년 2초와 직접 비교 금지)", "impulse-count"),
    ("prot", "충격 돌출도", "σ", "impulse", "가장 큰 충격 / 바탕 σ", "impulse-count"),
    ("imp_gap", "충격 간격 중앙값", "s", "impulse", "충격 2개 이상일 때. 반복 주기 단서", "impulse-count"),
    ("tkeo_kurt", "TKEO 첨도", "", "impulse", "Teager 에너지의 첨도", "tkeo"),
    ("tkeo_mm", "TKEO 최대/평균", "", "impulse", "Teager 에너지 최대 / 평균 |Ψ|", "tkeo"),
    ("win_cv", "창 RMS 변동계수", "", "impulse", "50 ms 창 RMS의 표준편차 / 평균", "window-stats"),
    ("win_rms_mm", "창 RMS 최대/평균", "", "impulse", "가장 센 창이 평균의 몇 배인가", "window-stats"),
    ("win_kurt_max", "창 첨도 최대", "", "impulse", "200개 창 중 가장 큰 첨도", "window-stats"),
    ("floor", "조용한 바탕", "g", "impulse", "가장 조용한 50 ms 창의 RMS", "window-stats"),
    ("zcr", "영교차율", "/s", "impulse", "신호가 0을 지나는 횟수 (≈ 주된 주파수 × 2)", "zcr"),
    ("zcr_std", "창 영교차율 변동계수", "", "impulse", "창마다 주된 주파수가 바뀌는 정도", "zcr"),
    ("s20", "0–20 Hz 비율", "", "low", "분산 중 느린 흔들림 몫", "bandpass"),
    ("s200", "0–200 Hz 비율", "", "low", "분산 중 200 Hz 아래 몫 = 고역통과가 지우는 몫", "bandpass"),
    ("e0_20", "0–20 Hz 에너지", "log g²", "low", "느린 흔들림 에너지 (log10)", "band-energy"),
    ("e20_200", "20–200 Hz 에너지", "log g²", "low", "저주파 진동 에너지 (log10)", "band-energy"),
    ("e200", "200 Hz 이상 에너지", "log g²", "spec", "200–2500 Hz 에너지 (log10)", "band-energy"),
    ("centroid", "중심주파수", "Hz", "spec", "전 대역(0–2.5 kHz) 에너지의 무게중심", "spectral-shape"),
    ("bandwidth", "스펙트럼 대역폭", "Hz", "spec", "중심 주변 퍼짐", "spectral-shape"),
    ("rolloff85", "85% 롤오프", "Hz", "spec", "에너지의 85%가 이 아래", "spectral-shape"),
    ("rolloff95", "95% 롤오프", "Hz", "spec", "에너지의 95%가 이 아래", "spectral-shape"),
    ("hi_ratio", "고역 비 (2 k+ / 전체)", "log", "spec", "2–2.5 kHz 몫 (log10)", "spectral-shape"),
    ("entropy", "스펙트럼 엔트로피", "", "spec", "1 = 고르게 퍼짐, 0 = 한 칸에 몰림", "spectral-entropy"),
    ("flat", "스펙트럼 평탄도", "", "spec", "기하평균 / 산술평균. 1 = 백색 잡음", "spectral-entropy"),
    ("peak_f", "최대 돌출 피크 주파수", "Hz", "spec", "매끈한 바탕보다 가장 튀어나온 칸", "resonance-shift"),
    ("peak_prom", "피크 돌출도", "dec", "spec", "그 피크가 바탕보다 몇 자리(log10) 높은가", "resonance-shift"),
    ("e_lo", "200–1k 에너지", "log g²", "pband", "3분할 저역 (Nyquist 2.5 kHz에 맞춘 분할)", "band-split"),
    ("e_mid", "1–2k 에너지", "log g²", "pband", "3분할 중역", "band-split"),
    ("e_hi", "2–2.5k 에너지", "log g²", "pband", "3분할 고역", "band-split"),
    ("r_lo", "200–1k 비율", "", "pband", "전체 에너지 중 몫 (게인 무관)", "payne"),
    ("r_mid", "1–2k 비율", "", "pband", "전체 에너지 중 몫", "payne"),
    ("r_hi", "2–2.5k 비율", "", "pband", "전체 에너지 중 몫", "payne"),
    ("env_kurt", "포락선 첨도", "", "env", "Hilbert 포락의 첨도", "envelope"),
    ("env_pf", "포락선 반복 주파수", "Hz", "env", "포락선 스펙트럼(1–300 Hz)에서 가장 튀어나온 피크", "envelope-spectrum"),
    ("env_prom", "포락선 피크 돌출도", "dec", "env", "그 반복 피크가 얼마나 뚜렷한가", "envelope-spectrum"),
    ("sk_max", "스펙트럼 첨도 최대", "", "env", "시간적으로 가장 충격적인 주파수 칸의 첨도", "spectral-kurtosis"),
    ("sk_f", "스펙트럼 첨도 최대 주파수", "Hz", "env", "그 칸의 주파수", "spectral-kurtosis"),
] + [(f"b{k}", f"{lo}–{hi} Hz", "log g²", "band", "고정 250 Hz 격자 대역 에너지 (log10)", "band-energy")
     for k, (lo, hi) in enumerate(BANDS)]


# %% 정상(train 정상 120개) 대비 백분위·기준값
def pct_of(v):
    vn = v[nor & ~np.isnan(v)]
    return np.array([np.nan if np.isnan(a) else 100 * ((vn < a).mean() + 0.5 * (vn == a).mean()) for a in v])


MAIN = [k for k, _, _, g, *_ in FEAT if g != "band"]
FEAT.append(("n_out", "정상 범위 밖 지표 수", "개", "summary", f"250 Hz 대역을 뺀 지표 {len(MAIN)}개 중 정상 5–95% 밖인 개수", "collinearity"))


def refq(v):
    vn = v[nor & ~np.isnan(v)]; vf = v[abn & ~np.isnan(v)]
    return dict(p5=np.percentile(vn, 5), p50=np.median(vn), p95=np.percentile(vn, 95), f50=np.median(vf) if len(vf) else np.nan)


def stats(V):
    pct, ref = {}, {}
    for k, *_ in FEAT[:-1]:
        pct[k] = pct_of(V[k]); ref[k] = refq(V[k])
    n_out = np.array([sum(1 for k in MAIN if not np.isnan(pct[k][i]) and (pct[k][i] < 5 or pct[k][i] > 95)) for i in range(N)])
    V["n_out"] = n_out.astype(float); pct["n_out"] = pct_of(V["n_out"]); ref["n_out"] = refq(V["n_out"])
    return pct, ref


def tags_of(V, ref):
    R = lambda k, q: ref[k][q]
    return [
    ("offset", "평균 오프셋", "|평균| > 0.1 g", np.abs(V["mean"]) > 0.1),
    ("quiet", "조용함", "RMS < 정상 5%", V["std"] < R("std", "p5")),
    ("loud", "큰 진동", "RMS > 정상 95%", V["std"] > R("std", "p95")),
    ("imp0", "충격 없음", "충격 수 0", V["n_imp"] == 0),
    ("imp1", "단발 충격", "충격 1–2개", (V["n_imp"] >= 1) & (V["n_imp"] <= 2)),
    ("impS", "몇 번 충격", "충격 3–9개", (V["n_imp"] >= 3) & (V["n_imp"] <= 9)),
    ("impN", "잦은 충격", "충격 10개 이상", V["n_imp"] >= 10),
    ("lowf", "저주파 흔들림 우세", "0–20 Hz가 분산의 60% 이상", V["s20"] >= 0.6),
    ("noise", "잡음·연속형", "첨도 < 1이고 충격 없음", (V["kurt"] < 1) & (V["n_imp"] == 0)),
    ("tonal", "협대역 피크", "피크 돌출도 > 정상 95%", V["peak_prom"] > R("peak_prom", "p95")),
    ("hif", "고주파 우세", "고역 비(2 k+) > 정상 95%", V["hi_ratio"] > R("hi_ratio", "p95")),
    ("hie", "고역 에너지 큼", "200 Hz 이상 에너지 > 정상 95%", V["e200"] > R("e200", "p95")),
    ("skb", "충격 대역 뚜렷", "스펙트럼 첨도 최대 > 정상 95%", V["sk_max"] > R("sk_max", "p95")),
    ("envp", "포락선 주기성", "포락선 피크 돌출도 > 정상 95%", V["env_prom"] > R("env_prom", "p95")),
    ("multi", "다수 지표 이탈", f"정상 범위 밖 지표 수 > 정상의 95% 분위({R('n_out', 'p95'):.0f}개)", V["n_out"] > R("n_out", "p95")),
    ]


SETS = {}
for s, (V, A) in RUN.items():
    pct, ref = stats(V)
    TAGS = tags_of(V, ref)
    SETS[s] = dict(V=V, A=A, pct=pct, ref=ref, TAGS=TAGS, ftags=[[t[0] for t in TAGS if t[3][i]] for i in range(N)])

# %% 개념 사전 (2-4 뷰어와 같은 파일)
md = open(os.path.join(ROOT, "3_단계", "개념사전.md"), encoding="utf-8").read()
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

# %% 3조 모델 결과 (pipeline_3조.py)
ntr = int(trn.sum())
ns = RES["nested_25"]
pad = lambda a: list(a) + [0] * (N - ntr)
prob = RES["oof_25"]["mean"] + RES["val"]["proba"]
MODELS = {"svm": dict(name=f"3조 SVM_C2 · 우리 중첩 CV (바깥 5 × 안쪽 4, seed 5)", short="3조 SVM", f2=round(ns["f2"], 3), f2_sd=round(ns["f2_sd"], 3),
                      miss=pad(ns["miss"]), fp=pad(ns["fp"]))}
F3 = RES["features"]
f3val = {k: RES["feat25"][k] for k in F3}
f3ref = {k: {q: (None if np.isnan(v) else round(float(v), 6)) for q, v in refq(np.array([np.nan if a is None else a for a in f3val[k]], float)).items()} for k in F3}

# %% 3조 노트북 셀 (코드 + 출력 텍스트·그림) + 셀별 주석
nb = json.load(open(os.path.join(SRC, "Final_Submission-1.ipynb"), encoding="utf-8"))
NOTES = [
    ("준비", "Colab 드라이브에서 정상 120 · 고장 30을 읽는다. 데이터 구성은 우리와 같다(4:1)."),
    ("전체 파형 겹침", "sample_rate=5000 — 25년 데이터는 실제로 5 kHz × 10초라 시간축이 맞다. 150개를 겹쳐 그려서 파일별 차이는 안 보인다."),
    ("확대 25,000–35,000", "정상 1개(nor_001)와 고장 1개(abn_001)만 겹쳐 본 그림. 여기서 본 구간이 아래 '5–7초 창' 피처가 된다 → <b>라벨을 보고 고른 구간</b>."),
    ("정규화 후 확대", "최대 |x|로 나눈 뒤 같은 구간. 크기 정보가 사라진 신호로 비교한다."),
    ("피처 28개", "<b>load_signal이 최대 |x|로 나눈다</b> → RMS·MAV·분산은 'Peak 대비 비율'(= 1/파고율 계열)로 바뀌고, abs_max는 1에 고정돼야 하지만 분모의 <code>+1e-8</code> 때문에 1 − 1e-8/Peak로 원래 Peak 순서가 그대로 남는다(아래 빌드 로그의 순위상관 참고). FFT 대역 2000–5000 Hz는 Nyquist 2.5 kHz를 넘어 실제로는 2–2.5 kHz. 4구간 RMS·첨도 8개 추가."),
    ("모델 10종 5-fold", "<b>폴드마다 그 검증 폴드의 정답으로 F2 최대 임계값을 고른다</b>(tune_threshold) → 점수가 낙관적. 모델 선택도 같은 폴드 점수로 → 선택 편향. 같은 피처·SVM_C2를 우리 중첩 CV로 돌리면 아래 비교 탭의 '중첩 CV' 값."),
    ("val 예측·제출", "임계값을 CV 중앙값(0.37) 대신 <b>손으로 0.20</b>으로 낮춰 제출(재현율 쪽으로). 이 노트북의 출력 파일명은 submission_safe.csv."),
    ("val 채점 (실행 안 됨)", "정답 파일이 없어 건너뜀. 코드의 <code>sub</code>는 정의된 적 없는 변수라 정답이 있었어도 에러였다."),
    ("val로 임계값 튜닝 (실행 안 됨)", "val 정답으로 임계값을 고르는 코드 — 정답이 있었다면 val 점수에 누수."),
    ("val 확률 관리도", "점선은 CV 임계값 0.37이지만 실제 제출은 0.20."),
    ("끝", ""),
    ("val 확률 출력", "이 40개 값을 pipeline_3조.py가 다시 계산한 값과 대조했다(최대 차이는 비교 탭)."),
]
CELLS = []
for k, c in enumerate([c for c in nb["cells"] if c["cell_type"] == "code"]):
    outs = []
    for o in c.get("4_outputs", []):
        if o.get("output_type") == "stream":
            outs.append(dict(kind="text", data="".join(o["text"])[:6000]))
        elif o.get("output_type") in ("execute_result", "display_data"):
            d = o.get("data", {})
            if "image/png" in d:
                outs.append(dict(kind="img", data="".join(d["image/png"]).replace("\n", "")))
            elif "text/plain" in d:
                outs.append(dict(kind="text", data="".join(d["text/plain"])[:6000]))
        elif o.get("output_type") == "error":
            outs.append(dict(kind="text", data=f"{o.get('ename')}: {o.get('evalue')}"))
    t, note = NOTES[k] if k < len(NOTES) else ("", "")
    CELLS.append(dict(title=t, note=note, src="".join(c["source"]), outs=outs))

# %% 25↔26 비교 데이터: 같은 코드로 기본 지표 · 0–2.5 kHz PSD
X26, y26, n26 = load_train()
X26c = X26 - X26.mean(1, keepdims=True)
X26d = decimate(X26c, 5, ftype="fir", axis=1, zero_phase=True)     # 25 kHz → 5 kHz (안티에일리어싱 FIR)


def basic(Xraw, Z, fs):
    h = sosfiltfilt(butter(4, 200, btype="highpass", fs=fs, output="sos"), Z, axis=-1)
    n_imp = []
    for r in h:
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-12
        idx = np.flatnonzero(np.abs(r) > IMP_K * s)
        n_imp.append(0 if len(idx) == 0 else len(np.split(idx, np.flatnonzero(np.diff(idx) > int(0.010 * fs)) + 1)))
    dur = Z.shape[1] / fs
    f, P = welch(Z, fs=fs, nperseg=int(1024 * fs / 5000), detrend=False)
    a1 = f > 0; Pn = P[:, a1] / P[:, a1].sum(1, keepdims=True)
    return dict(mean=Xraw.mean(1), std=Z.std(1), maxabs=np.abs(Z).max(1), kurt=kurtosis(Z, axis=1),
                crest=np.abs(Z).max(1) / Z.std(1), imp_rate=np.array(n_imp) / dur,
                s200=P[:, f < 200].sum(1) / P.sum(1), centroid=(Pn * f[a1]).sum(1), rms_hp=h.std(1))


CMP_SETS = [("25", "25년 (5 kHz · 10초)", basic(X[trn], Xc[trn], FS), y[trn]),
            ("26d", "26년 → 5 kHz로 낮춤 (0–2.5 kHz)", basic(X26, X26d, 5000), y26),
            ("26", "26년 원본 (25 kHz · 2초)", basic(X26, X26c, 25000), y26)]
CMP_METRICS = [("mean", "평균 (원신호)", "g", False), ("std", "RMS", "g", True), ("rms_hp", "200 Hz 고역통과 RMS", "g", True),
               ("maxabs", "Peak", "g", True), ("kurt", "첨도 (Fisher)", "", True), ("crest", "파고율", "", True),
               ("imp_rate", "초당 충격 수 (8σ)", "/s", False), ("s200", "0–200 Hz 비율", "", False), ("centroid", "중심주파수", "Hz", False)]


def psd_ref(Z, yy, fs):
    f, P = welch(Z, fs=fs, nperseg=1024, detrend=False)
    L = np.log10(P + 1e-30)
    q = lambda a: [round(float(v), 3) for v in a]
    return dict(f0=float(f[0]), df=float(f[1] - f[0]), n=len(f), p5=q(np.percentile(L[yy == 0], 5, 0)), p50=q(np.median(L[yy == 0], 0)),
                p95=q(np.percentile(L[yy == 0], 95, 0)), f50=q(np.median(L[yy == 1], 0)))


notice = list(csv.reader(io.StringIO(open(os.path.join(SRC, "@25-2 품질공학 성능 공지-1.csv"), encoding="cp949").read())))
CMP = dict(
    sets=[dict(key=k, label=l, y=yy.tolist(), val={m: [round(float(v), 6) for v in d[m]] for m, *_ in CMP_METRICS}) for k, l, d, yy in CMP_SETS],
    metrics=[dict(key=k, label=l, unit=u, log=g) for k, l, u, g in CMP_METRICS],
    psd={"25": psd_ref(Xc[trn], y[trn], FS), "26d": psd_ref(X26d, y26, 5000)},
    spec=[["샘플링", "5 kHz", "25 kHz"], ["길이", "10초 (50,000)", "2초 (50,000)"], ["Nyquist (볼 수 있는 최고 주파수)", "2.5 kHz", "12.5 kHz"],
          ["파일 (정상 / 고장)", f"{int(nor.sum())} / {int(abn.sum())} + val {int((y < 0).sum())}", f"{int((y26 == 0).sum())} / {int((y26 == 1).sum())}"],
          ["원신호 |평균| 중앙", f"{np.median(np.abs(X[trn].mean(1))):.2g} g", f"{np.median(np.abs(X26.mean(1))):.2g} g"],
          ["RMS 중앙 (평균 제거)", f"{np.median(Xc[trn].std(1)):.3g} g", f"{np.median(X26c.std(1)):.3g} g"]],
    auc25=RES["auc_25"], auc26=RES["auc_26"], feats=F3,
    cv3_25=RES["cv3_25"], cv3_26=RES["cv3_26"], nested25=RES["nested_25"], nested26=RES["nested_26"],
    notice=notice, val_diff=RES["val"]["max_diff"], val_n1=sum(RES["val"]["pred_02"]),
    miss26=[n for n, c in zip(n26, RES["nested_26"]["miss"]) if c >= 3], fp26=sum(1 for c in RES["nested_26"]["fp"] if c >= 3),
    fp25=[n for n, c in zip(names, RES["nested_25"]["fp"]) if c >= 3], names26=list(n26),
)

# %% 그림 데이터 (2-4와 같은 압축)
NB, NZ = 1000, 1000                         # 파형 포락 50 샘플(10 ms)씩 / 최대 충격 ±100 ms
sc_x = np.abs(Xc).max(1); sc_h = np.abs(H).max(1)
q = lambda a, s: np.clip(np.round(a / s[:, None] * 32767), -32767, 32767).astype("<i2")
xr, hr = Xc.reshape(N, NB, -1), H.reshape(N, NB, -1)
ENV = np.stack([q(xr.min(2), sc_x), q(xr.max(2), sc_x), q(hr.min(2), sc_h), q(hr.max(2), sc_h)], 1)
fix = lambda a, s: np.round(np.clip(a, -32 / s * 1000, 32 / s * 1000) * s).astype("<i2")
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()


def clean(a, nd=5):
    return [None if (isinstance(v, (float, np.floating)) and np.isnan(v)) else (round(float(v), nd) if isinstance(v, (float, np.floating)) else v) for v in a]


band_ref = lambda A: dict(p5=clean(np.percentile(A[nor], 5, 0), 3), p50=clean(np.median(A[nor], 0), 3),
                          p95=clean(np.percentile(A[nor], 95, 0), 3), f50=clean(np.median(A[abn], 0), 3))
ST_KEEP = 387 // 3                                                # 시간 389칸 → 3칸씩 129칸
spec_log = {s: np.log10(R["A"]["S"][:, :128, :ST_KEEP * 3].reshape(N, 64, 2, ST_KEEP, 3).mean((2, 4)) + 1e-30) for s, R in SETS.items()}
SLO, SHI = np.percentile(spec_log["raw"], 1), np.percentile(spec_log["raw"], 99.9)
Lr = SETS["raw"]["A"]["L"][trn]
PSD_YLIM = [float(np.floor(np.percentile(Lr, 0.5))), float(np.ceil(np.percentile(Lr, 99.9)))]


def pack(s, Z):
    R = SETS[s]; V, A, pct, ref = R["V"], R["A"], R["pct"], R["ref"]
    sc = np.abs(Z).max(1)
    st = np.clip(np.abs(Z).argmax(1) - NZ // 2, 0, N_SAMP - NZ)
    ZOOM = q(np.stack([Z[i, s0:s0 + NZ] for i, s0 in enumerate(st)]), sc)
    LSK = np.log10(1 + np.clip(A["SK"], 0, None))
    WR, WK = np.log10(A["WRMS"] + 1e-12), np.log10(1 + np.clip(A["WKURT"], 0, None))
    WQ = np.stack([fix(WR, 1000), fix(WK, 1000)], 1)
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
    names=list(names), y=y.tolist(), fs=FS, dur=DUR, df=float(A0["df"]), nf=int(len(A0["f"])),
    groups=[dict(key=k, label=l, concept=c) for k, l, c in GROUPS],
    feat=[dict(key=k, label=l, unit=u, group=g, desc=d, concept=c) for k, l, u, g, d, c in FEAT],
    models=MODELS, prob=[round(p, 4) for p in prob], thr_cv=RES["thr_cv"], thr_submit=RES["thr_submit"],
    val_pred=RES["val"]["pred_02"], f3=dict(keys=F3, val=f3val, ref=f3ref, auc=RES["auc_25"]),
    concepts=CONCEPTS, concept_intro=intro, cells=CELLS, cmp=CMP,
    sk_df=float(A0["fs_"][1] - A0["fs_"][0]), sk_nf=int(len(A0["fs_"])), env_f0=float(A0["fe"][0]), env_df=float(A0["fe"][1] - A0["fe"][0]), env_nf=int(len(A0["fe"])),
    nw=NW, win_s=WIN / FS, nb=NB, nz=NZ, nbands=len(BANDS), psd_ylim=PSD_YLIM,
    sc={s: clean(np.abs(Z).max(1), 6) for s, Z in SIGS.items()},
    spec_shape=[64, ST_KEEP], spec_fmax=64 * 2 * float(A0["fs_"][1] - A0["fs_"][0]), imp_k=IMP_K,
    pbands=[dict(label="200–1k", lo=200, hi=1000), dict(label="1–2k", lo=1000, hi=2000), dict(label="2–2.5k", lo=2000, hi=2500)],
    gbands=[dict(label="1차 공진? 100–300", lo=100, hi=300), dict(label="고차 모드? 1–3k", lo=1000, hi=3000)],
    sig_names=dict(raw="평균 제거", hp="200 Hz 고역통과"),
    env=b64(ENV),
    sets={s: pack(s, Z) for s, Z in SIGS.items()},
)

tpl = open(os.path.join(HERE, "viewer_template.html"), encoding="utf-8").read()
out = tpl.replace("__DATA_JSON__", json.dumps(D, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/"))
p = os.path.join(HERE, "뷰어25.html")
open(p, "w", encoding="utf-8").write(out)

# %% 확인용 요약
from sklearn.metrics import roc_auc_score
raw_peak = np.abs(Xc[trn]).max(1)
am = np.array(f3val["abs_max"], float)[trn]
lines = [f"HTML {os.path.getsize(p)/1e6:.1f} MB · 파일 {N} (train {ntr}, val {N - ntr}) · 지표 {len(FEAT)}개 · 개념 {len(CONCEPTS)}개 · 노트북 셀 {len(CELLS)}",
         f"3조 abs_max: 범위 {am.min():.10f} ~ {am.max():.10f}, 원래 Peak와 순위상관 {spearmanr(am, raw_peak)[0]:.3f}"]
for s, R in SETS.items():
    V, ref = R["V"], R["ref"]; n_out = V["n_out"]
    lines += ["", f"=== 신호: {D['sig_names'][s]} ===", "태그별 파일 수 (정상/고장/val):"]
    for k, l, d, m in R["TAGS"]:
        lines.append(f"  {l}: {int((m & nor).sum())}/{int((m & abn).sum())}/{int((m & (y < 0)).sum())}  ({d})")
    lines.append(f"정상 범위 밖 지표 수: 정상 중앙 {np.median(n_out[nor]):.0f} (95% {ref['n_out']['p95']:.0f}), 고장 중앙 {np.median(n_out[abn]):.0f}")
    aucs = sorted(((abs(roc_auc_score(y[trn], np.nan_to_num(V[k][trn], nan=np.nanmedian(V[k]))) - .5) + .5, k) for k in MAIN), reverse=True)
    lines.append("설명용 단변량 AUC 상위 (|AUC−0.5|+0.5): " + ", ".join(f"{k} {a:.2f}" for a, k in aucs[:15]))
open(os.path.join(HERE, "4_outputs", "build_log.txt"), "w", encoding="utf-8").write("\n".join(lines))
print("done")
