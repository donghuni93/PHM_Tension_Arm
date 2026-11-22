"""신효섭 11-19 최종 모델(RF 배깅 + 8.3k 스펙트럼 규칙) — 원본 참고자료/팀원 공유 정보/신효섭/1119_최종/7_코드/최종모델.py 를 그대로 복사 (EXP-048, 11-22).
계산은 고치지 않는다. 원본 설명은 아래 문자열."""
"""9조 텐션암 이상감지 — 최종 모델: RF 배깅 + 8.3k 규칙 (단독 실행 파일)

이 파일 하나로 원본 데이터에서 학습 · 예측 · 검증 재현을 한다. 캐시나 다른 스크립트가 필요 없다.
필요 라이브러리: numpy · scipy · pandas · scikit-learn (qe-phm 환경 그대로)

사용법
  python 최종모델.py 예측 <train 폴더> <val 폴더> [출력 csv]   # 제출 파일 + _상세.csv
  python 최종모델.py 검증 <train 폴더>                         # 같은 분할 20회 중첩 교차검증 → F2 0.956 ± 0.020 재현 (약 15분)
  <train 폴더> = normal/ · abnormal/ 하위 폴더가 있는 학습 데이터 (정상 120 · 고장 30)

판정 순서
  1) 신호 → 피처 32개 (라벨 안 씀, 파일마다 독립): 평균 제거 · 10 Hz / 200 Hz 고역통과 · 버터워스 대역 5개 · AR(40) 백색화 · 충격 검출 · 울림 주파수
  2) 학습 파일에서 정상 · 고장을 각각 80%씩 10번 뽑아(층화 · 비복원), 뽑을 때마다
     피처 10개 선택(AUC 순 → |r| > 0.85 제거 → VIF > 10 제거 → 상위 10개) → 랜덤 포레스트 300그루 → 고장 확률
  3) 10개 확률 평균 ≥ 임계값(학습 파일 안쪽 5-fold OOF 의 F2 평탄 구간 가운데)  또는  8.3k 규칙(8.20–8.45 kHz 봉우리 돌출 ≥ 0.75 dec · 반높이 폭 ≥ 150 Hz) → 고장
  AR 잔차 계수 · 결측 대체 중앙값 · 피처 선택 · 임계값은 모두 학습 파일로만 정한다 (새 파일은 판정에만)

출처: 피처 계산 = 효섭(품공)/코드/rf_검증.py (팀 노트북 텐션암_9조_최종모델.ipynb 셀 3 · 4) · 8.3k 계산 = 동훈 님 src/final.py signal_tables
     배깅 시드 · 분할 = 효섭(품공)/코드/배깅_최종.py 와 같음 → 같은 결과
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, signal, stats
from scipy.ndimage import median_filter, uniform_filter1d
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
FS, TR, P = 25000, 500, 40                      # 표본 주파수 · 양끝 자르기(필터 경계) · AR 차수
HP_HZ, BAND = 10, (2500, 5000)
K_FEATURES, R_MAX, VIF_MAX, THR_TOL = 10, 0.85, 10, 0.02
N_BAGS, BAG_FRAC = 10, 0.8
P83_MIN, W83_MIN = 0.75, 150.0
TS = np.round(np.arange(0.05, 0.951, 0.01), 2)


# ---------------------------------------------------------------------------
# 데이터 읽기
# ---------------------------------------------------------------------------
def read_signal(path):
    """신호 CSV 1개 (1열). BOM · 머리글 한 줄 허용"""
    try:
        return np.loadtxt(path, encoding="utf-8-sig")
    except ValueError:
        return np.loadtxt(path, encoding="utf-8-sig", skiprows=1)


def load_train(folder):
    """folder/normal (0) · folder/abnormal (1), 이름순"""
    X, y, names = [], [], []
    for sub, lab in (("normal", 0), ("abnormal", 1)):
        for p in sorted((Path(folder) / sub).glob("*.csv")):
            X.append(read_signal(p)); y.append(lab); names.append(p.stem)
    return np.stack(X), np.array(y), names


def load_folder(folder):
    """라벨 없는 폴더(val · test). 하위 폴더 포함, 숨김 파일 · __MACOSX 제외, 이름순"""
    paths = sorted((p for p in Path(folder).rglob("*.csv") if not p.name.startswith(".") and "__MACOSX" not in p.parts), key=lambda p: p.stem)
    if not paths:
        raise FileNotFoundError(f"{folder} 아래에 csv 가 없다")
    return np.stack([read_signal(p) for p in paths]), [p.stem for p in paths]


# ---------------------------------------------------------------------------
# 1) 전처리 + 피처 32개 (파일마다 독립, 라벨 안 씀)
# ---------------------------------------------------------------------------
def _sos(lo, hi=None):
    return signal.butter(4, lo if hi is None else [lo, hi], "highpass" if hi is None else "bandpass", fs=FS, output="sos")


SOS = {"hp10": _sos(HP_HZ), "hp200": _sos(200)}
BANDS = {"b1_25": (1000, 2500), "b25_5": BAND, "b5_8": (5000, 8000), "b8_11": (8000, 11000), "b11_12": (11000, 12400)}
for _k, (_a, _b) in BANDS.items():
    SOS[_k] = _sos(_a, _b)


def ar_whiten(x, p=P):
    """AR(p)로 예측 가능한 진동(공진 · 흔들림)을 빼고 남은 '갑작스러운' 성분"""
    r = np.correlate(x[:20000], x[:20000], "full")[19999:19999 + p + 1]
    a = linalg.solve_toeplitz(r[:p], r[1:p + 1])
    return signal.lfilter(np.r_[1, -a], [1], x)[p:]


def seg_kurt(x, L=2500):
    """0.1초 구간마다 첨도 → 평균 (충격이 여러 구간에서 반복되는가)"""
    s = x[:len(x) // L * L].reshape(-1, L)
    return stats.kurtosis(s, axis=1, fisher=False).mean()


def basic(x, p):
    r = np.sqrt((x ** 2).mean())
    return {p + "rms": r, p + "kurt": stats.kurtosis(x, fisher=False), p + "crest": np.abs(x).max() / r, p + "segk": seg_kurt(x)}


def shape(x, p):
    d = basic(x, p)
    dx = np.diff(x)
    ddx = np.diff(dx)
    m = np.sqrt(dx.var() / x.var())
    d[p + "hj_mob"] = m
    d[p + "hj_cpx"] = np.sqrt(ddx.var() / dx.var()) / m
    f, Pw = signal.welch(x, FS, nperseg=4096)
    d[p + "centroid"] = (f * Pw).sum() / Pw.sum()
    for lo, hi in [(10, 1000), (1000, 2500), (2500, 5000), (5000, 8000), (8000, 12500)]:
        d[p + f"er_{lo // 1000}_{hi // 1000}k"] = Pw[(f >= lo) & (f < hi)].sum() / Pw.sum()
    return d


def features(name, raw):
    """파일 1개 → (피처 dict, 자기상관 — AR 잔차 계산용)"""
    x = raw - raw.mean()
    d = {"file": name}
    v = {k: signal.sosfiltfilt(s, x)[TR:-TR] for k, s in SOS.items()}
    d.update(shape(v["hp10"], "A_"))
    d.update(shape(v["hp200"], "H2_"))
    for k in BANDS:
        d.update(basic(v[k], k + "_"))
    xh = signal.sosfiltfilt(SOS["hp10"], x)
    w = ar_whiten(xh)[TR:-TR]
    d.update({k: v_ for k, v_ in basic(w, "D_").items() if k != "D_rms"})
    sig = np.median(np.abs(w)) / 0.6745
    pk, pr = signal.find_peaks(np.abs(w), height=8 * sig, distance=250)          # 충격 = 백색화 신호가 바탕의 8배를 넘는 곳
    amp = pr["peak_heights"] / sig
    d["D_nimp"] = len(pk)
    xs = xh[TR:-TR]
    env = np.abs(signal.hilbert(xs))
    d["I_density"] = (np.convolve(env > np.median(env) * 6, np.ones(250), "same") > 0).mean()
    for k in ["I_top5_ratio", "I_count_per_max", "I_interval_cv", "R_fpeak_med", "R_frac_2_5k", "R_ring_25_58", "R_ring_811_58"]:
        d[k] = np.nan
    if len(amp):
        s_ = np.sort(amp)[::-1]
        d["I_top5_ratio"] = s_[:5].mean() / s_[0]
        d["I_count_per_max"] = len(amp) / s_[0]
    if len(pk) >= 3:
        iv = np.diff(pk)
        d["I_interval_cv"] = iv.std() / iv.mean()
    fr = np.fft.rfftfreq(1024, 1 / FS)
    fp, r25, r811 = [], [], []
    for p0 in pk[np.argsort(amp)[::-1][:20]]:                                   # 큰 충격 20개 뒤 12 ms 울림의 주파수
        seg = xs[p0:p0 + 300]
        if len(seg) < 300:
            continue
        Sp = np.abs(np.fft.rfft(seg * np.hanning(300), 1024)) ** 2
        m = (fr >= 1000) & (fr < 12000)
        fp.append(fr[m][Sp[m].argmax()] / 1000)
        e = lambda a, b: Sp[(fr >= a) & (fr < b)].sum() + 1e-20  # noqa: E731
        r25.append(e(2000, 5000) / e(5000, 8000))
        r811.append(e(8000, 11000) / e(5000, 8000))
    if fp:
        d["R_fpeak_med"] = np.median(fp)
        d["R_frac_2_5k"] = np.mean((np.array(fp) >= 2) & (np.array(fp) < 5))
        d["R_ring_25_58"] = np.median(r25)
        d["R_ring_811_58"] = np.median(r811)
    f, t, Sx = signal.stft(xs, FS, nperseg=512, noverlap=384)
    A2 = np.abs(Sx) ** 2
    sk = (A2 ** 2).mean(1) / (A2.mean(1) ** 2 + 1e-30) - 2
    d["SK_200_3k"] = sk[(f >= 200) & (f < 3000)].mean()
    r = np.correlate(xs, xs, "full")[len(xs) - 1:len(xs) + P + 1]
    return d, r / r[0]


def build_X(df):
    """피처 dict 표 → 모델 입력 (대부분 로그 눈금)"""
    X = pd.DataFrame(index=df.index)
    no_log = {"R_frac_2_5k", "SK_200_3k"}
    for c in df.columns:
        if c == "file":
            continue
        X[c] = df[c].values if c in no_log else np.log(df[c].astype(float).clip(lower=1e-6 if c.startswith(("I_", "R_")) else 1e-9)).values
    X["R_811_over_58"] = np.log(df.b8_11_rms / df.b5_8_rms).values
    X["R_811_over_25"] = np.log(df.b8_11_rms / df.b25_5_rms).values
    X["AR_resid"] = np.nan
    return X


def candidates(X):
    """후보 32개"""
    return ([c for c in X if c.startswith("A_")] + [c for c in X if c.startswith("b25_5_")] + [c for c in X if c.startswith("D_")] +
            ["R_811_over_58", "R_811_over_25", "I_density", "I_top5_ratio", "I_count_per_max", "I_interval_cv", "R_fpeak_med",
             "R_frac_2_5k", "R_ring_25_58", "R_ring_811_58", "SK_200_3k", "AR_resid"])


def p83_w83(X):
    """8.3 kHz 봉우리: 돌출(dec) = 55 Hz로 고른 log PSD − 1.2 kHz 이동 중앙값, 반높이 폭(Hz) — 동훈 님 src/final.py 와 같은 계산"""
    Xc = X - X.mean(1, keepdims=True)
    f, P8 = signal.welch(Xc, fs=FS, nperseg=8192)
    L = np.log10(P8)
    df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))), axis=1)
    PR = Sm - median_filter(Sm, size=(1, int(round(1200 / df))), mode="nearest")
    band = (f >= 8200) & (f < 8450)
    ia = np.flatnonzero(band)
    P83, W83 = PR[:, band].max(1), np.zeros(len(PR))
    for i in range(len(PR)):
        k = ia[PR[i, band].argmax()]
        h = PR[i, k] / 2
        lo = hi = k
        while lo > 0 and PR[i, lo - 1] >= h:
            lo -= 1
        while hi < PR.shape[1] - 1 and PR[i, hi + 1] >= h:
            hi += 1
        W83[i] = f[hi] - f[lo]
    return P83, W83


def tables(X, names):
    """신호 → (모델 입력 표, 자기상관, 8.3k 규칙 켜짐, P83, W83)"""
    from joblib import Parallel, delayed
    out = Parallel(n_jobs=-1)(delayed(features)(n, x) for n, x in zip(names, X))   # 파일마다 독립이라 병렬 (계산 결과는 같음)
    rows, acs = [d for d, _ in out], [ac for _, ac in out]
    P83, W83 = p83_w83(X)
    return build_X(pd.DataFrame(rows)), np.array(acs), (P83 >= P83_MIN) & (W83 >= W83_MIN), P83, W83


# ---------------------------------------------------------------------------
# 2) 학습 파일로만 정하는 것들: AR 잔차 · 결측 대체 · 피처 선택 · 임계값
# ---------------------------------------------------------------------------
def prep(X, AC, y, tr):
    """AR 잔차(학습 정상의 평균 자기상관으로 AR 계수) · 결측 = 학습 파일 중앙값"""
    Xc = X.copy()
    r = AC[[i for i in tr if y[i] == 0]].mean(0)
    a = linalg.solve_toeplitz(r[:P], r[1:P + 1])
    Xc["AR_resid"] = np.log(np.clip([1 - 2 * a @ q[1:P + 1] + a @ linalg.toeplitz(q[:P]) @ a for q in AC], 1e-6, None))
    return Xc.fillna(Xc.iloc[tr].median())


def select(Xtr, ytr, k=K_FEATURES):
    """AUC 순 → 이미 고른 것과 |r| > 0.85면 건너뜀 → VIF > 10 이면 가장 큰 것부터 뺌 → 상위 k개"""
    au = {c: abs(roc_auc_score(ytr, Xtr[c]) - .5) for c in Xtr}
    order = sorted(au, key=au.get, reverse=True)
    C = Xtr.corr().abs()
    sel = []
    for c in order:
        if all(C.loc[c, s] <= R_MAX for s in sel):
            sel.append(c)
    while len(sel) > 2:
        v = np.diag(np.linalg.pinv(np.corrcoef(StandardScaler().fit_transform(Xtr[sel]).T)))
        if v.max() > VIF_MAX:
            sel.pop(int(v.argmax()))
        else:
            break
    return sel[:k]


def threshold(y, p):
    """F2 가 최대 −0.02 이내인 임계값 구간의 가운데 (한 점에 맞추지 않음)"""
    f = np.array([fbeta_score(y, p >= t, beta=2) for t in TS])
    ok = TS[f >= f.max() - THR_TOL]
    return float((ok.min() + ok.max()) / 2)


def bag(Xa, ya, Xb, seed, cand, log=None):
    """80% 층화 뽑기 10번 × (피처 10개 선택 + RF 300그루) → 확률 평균"""
    out = np.zeros(len(Xb))
    for b, (s, _) in enumerate(StratifiedShuffleSplit(N_BAGS, train_size=BAG_FRAC, random_state=seed).split(Xa, ya)):
        Xs, ys = Xa.iloc[s], ya[s]
        fe = select(Xs[cand], ys)
        m = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=b, n_jobs=1).fit(Xs[fe], ys)
        out += m.predict_proba(Xb[fe])[:, 1]
        if log is not None:
            log.append(fe)
    return out / N_BAGS


def fit_predict(X, AC, y, r83, tr, inner_pairs, te, seed, log=None):
    """학습 tr 로 임계값(안쪽 OOF) · 배깅 → te 판정. 반환 (평균 확률, 임계값, 판정)"""
    cand = candidates(X)
    ytr = y[tr]
    inner = np.zeros(len(tr))
    for j, (a, b) in enumerate(inner_pairs):
        Xi = prep(X.iloc[tr].reset_index(drop=True), AC[tr], ytr, a)
        inner[b] = bag(Xi.iloc[a], ytr[a], Xi.iloc[b], seed + j, cand)
    thr = threshold(ytr, inner + 100 * r83[tr])
    Xf = prep(X, AC, y, tr)
    p = bag(Xf.iloc[tr], ytr, Xf.iloc[te], seed + 9, cand, log)
    return p, thr, ((p >= thr) | r83[te]).astype(int)


# ---------------------------------------------------------------------------
# 실행
# ---------------------------------------------------------------------------
def run_predict(train_dir, new_dir, out):
    t0 = time.time()
    Xtr_sig, y, _ = load_train(train_dir)
    Xnew_sig, names = load_folder(new_dir)
    print(f"학습 {len(y)}개 (고장 {y.sum()}) · 새 파일 {len(names)}개 — 피처 계산 중", flush=True)
    A, ACa, r83a, _, _ = tables(Xtr_sig, [f"tr{i}" for i in range(len(y))])
    B, ACb, r83b, P83, W83 = tables(Xnew_sig, names)
    X = pd.concat([A, B], ignore_index=True)
    AC, r83 = np.vstack([ACa, ACb]), np.r_[r83a, r83b]
    yall = np.r_[y, np.zeros(len(names), int)]                 # 새 파일 라벨 없음 — prep · 선택은 학습 파일 라벨만 봄
    tr, te = np.arange(len(y)), np.arange(len(y), len(y) + len(names))
    pairs = list(StratifiedKFold(5, shuffle=True, random_state=0).split(tr, y))
    log = []
    p, thr, pred = fit_predict(X, AC, yall, r83, tr, pairs, te, 777, log)
    D = pd.DataFrame(dict(file=names, label=pred))
    D.to_csv(out, index=False, encoding="utf-8-sig")
    det = pd.DataFrame(dict(file=names, label=pred, prob=np.round(p, 4), threshold=round(thr, 4), rule83=r83b.astype(int),
                            P83=np.round(P83, 3), W83=np.round(W83, 1), near_threshold=(np.abs(p - thr) < 0.05).astype(int)))
    det.to_csv(Path(out).with_name(Path(out).stem + "_상세.csv"), index=False, encoding="utf-8-sig")
    from collections import Counter
    sel = Counter(f for fe in log for f in fe)
    print(f"임계값 {thr:.3f} · 고장 판정 {pred.sum()} / {len(pred)} (8.3k 규칙 {int(r83b.sum())}) · 임계값 ±0.05 경계 {int(det.near_threshold.sum())}개")
    print("뽑기 10번에서 고른 피처:", ", ".join(f"{k} {v}" for k, v in sel.most_common()))
    print(f"저장: {out} ({time.time() - t0:.0f}초)")


def run_cv(train_dir, reps=20):
    from joblib import Parallel, delayed
    t0 = time.time()
    Xs, y, _ = load_train(train_dir)
    X, AC, r83, _, _ = tables(Xs, [f"tr{i}" for i in range(len(y))])
    print(f"피처 계산 끝 {time.time() - t0:.0f}초 — 같은 분할 {reps}회 × 바깥 5-fold × 안쪽 5-fold", flush=True)
    tasks = []
    for r in range(reps):
        for k, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=r).split(np.arange(len(y)), y)):
            pairs = list(StratifiedKFold(5, shuffle=True, random_state=r).split(tr, y[tr]))
            tasks.append((r, tr, te, pairs, 1000 * r + 10 * k))
    res = Parallel(n_jobs=-1, verbose=5)(delayed(lambda r, tr, te, pr, s: (r, te, fit_predict(X, AC, y, r83, tr, pr, te, s)[2]))(*t) for t in tasks)
    pred = {r: np.zeros(len(y), int) for r in range(reps)}
    for r, te, pr in res:
        pred[r][te] = pr
    f2 = np.array([fbeta_score(y, pred[r], beta=2) for r in range(reps)])
    wrong = np.array([pred[r] != y for r in range(reps)]).sum(0)
    print(f"F2 {f2.mean():.3f} ± {f2.std(ddof=1):.3f} · 놓친 고장 {wrong[y == 1].sum() / reps:.2f} / 회 · 오경보 {wrong[y == 0].sum() / reps:.2f} / 회 ({time.time() - t0:.0f}초)")
    print("반복별 F2:", " ".join(f"{v:.3f}" for v in f2))


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) >= 3 and a[0] == "예측":
        run_predict(a[1], a[2], a[3] if len(a) > 3 else "submission_9조.csv")
    elif len(a) >= 2 and a[0] == "검증":
        run_cv(a[1])
    else:
        sys.exit(__doc__)
