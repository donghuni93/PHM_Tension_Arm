# %% [markdown]
# # 4-12 팀원 피처 비교 — ① 권소린 M1 피처 재현 (11-23)
# 피처 계산 · 선택 코드는 권소린 1116 노트북(`1_참고자료/팀원 공유 정보/권소린/1116_1차/권소린_텐션암_9조_최종모델.ipynb`
# cell 6 · 8)을 **그대로** 옮겼다. 11-23 M1 html의 최종 10개와 이름 · 순서가 같은지로 재현을 확인한다.
# 출력: figs/소린_파일표.csv (train 150 + val 40, 후보 32열 + AR_resid는 학습 정상 120개 AR로 계산),
#       figs/소린_결과.txt (150개 전체 선택 결과 · 단독 AUC · val 이동)

# %% 0. 준비
import os, sys, glob, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
ROOT = _d; HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import signal, stats, linalg
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

CFG = dict(FS=25000, TRIM=500, HP_HZ=10, BAND=(2500, 5000), AR_P=40, R_MAX=0.85, VIF_MAX=10, K_FEATURES=10)

def read_signal(path):
    return pd.read_csv(path, header=None, encoding="utf-8-sig")[0].values.astype(float)

D = os.path.join(ROOT, "2_data", "TeamProject_train_변경")
RAW, LABEL = {}, {}
for sub, lab in [("normal", 0), ("abnormal", 1)]:
    for f in sorted(glob.glob(os.path.join(D, sub, "*.csv"))):
        n = os.path.basename(f)[:-4]; RAW[n] = read_signal(f); LABEL[n] = lab
NAMES = sorted(RAW, key=lambda n: (LABEL[n], n)); y = np.array([LABEL[n] for n in NAMES])
VRAW = {os.path.basename(f)[:-4]: read_signal(f) for f in sorted(glob.glob(os.path.join(ROOT, "2_data", "TeamProject_val", "val", "*.csv")))}
VNAMES = sorted(VRAW)

# %% 1. 피처 계산 (권소린 cell 6 그대로)
FS, TR, P = CFG["FS"], CFG["TRIM"], CFG["AR_P"]
def _sos(lo, hi=None):
    return signal.butter(4, lo if hi is None else [lo, hi], "highpass" if hi is None else "bandpass", fs=FS, output="sos")
SOS = {"hp10": _sos(CFG["HP_HZ"]), "hp200": _sos(200)}
BANDS = {"b1_25": (1000, 2500), "b25_5": CFG["BAND"], "b5_8": (5000, 8000), "b8_11": (8000, 11000), "b11_12": (11000, 12400)}
for k, (a, b) in BANDS.items(): SOS[k] = _sos(a, b)

def acf(x, p):   # = np.correlate(x, x, "full")[len(x)-1 : len(x)+p] (원본 코드), 필요한 지연 0..p만 직접 계산 (같은 값, 빠름)
    return np.array([np.dot(x[:len(x) - k], x[k:]) for k in range(p + 1)])
def ar_whiten(x, p=P):
    r = acf(x[:20000], p)
    a = linalg.solve_toeplitz(r[:p], r[1:p + 1]); return signal.lfilter(np.r_[1, -a], [1], x)[p:]
def seg_kurt(x, L=2500):
    s = x[:len(x) // L * L].reshape(-1, L); return stats.kurtosis(s, axis=1, fisher=False).mean()
def basic(x, p):
    r = np.sqrt((x ** 2).mean())
    return {p + "rms": r, p + "kurt": stats.kurtosis(x, fisher=False), p + "crest": np.abs(x).max() / r, p + "segk": seg_kurt(x)}
def shape(x, p):
    d = basic(x, p); dx = np.diff(x); ddx = np.diff(dx)
    m = np.sqrt(dx.var() / x.var()); d[p + "hj_mob"] = m; d[p + "hj_cpx"] = np.sqrt(ddx.var() / dx.var()) / m
    f, Pw = signal.welch(x, FS, nperseg=4096); d[p + "centroid"] = (f * Pw).sum() / Pw.sum()
    for lo, hi in [(10, 1000), (1000, 2500), (2500, 5000), (5000, 8000), (8000, 12500)]:
        d[p + f"er_{lo // 1000}_{hi // 1000}k"] = Pw[(f >= lo) & (f < hi)].sum() / Pw.sum()
    return d

def features(name, raw):
    x = raw - raw.mean(); d = {"file": name}
    v = {k: signal.sosfiltfilt(s, x)[TR:-TR] for k, s in SOS.items()}
    d.update(shape(v["hp10"], "A_")); d.update(shape(v["hp200"], "H2_"))
    for k in BANDS: d.update(basic(v[k], k + "_"))
    xh = signal.sosfiltfilt(SOS["hp10"], x); w = ar_whiten(xh)[TR:-TR]
    d.update({k: v_ for k, v_ in basic(w, "D_").items() if k != "D_rms"})
    sig = np.median(np.abs(w)) / 0.6745
    pk, pr = signal.find_peaks(np.abs(w), height=8 * sig, distance=250); amp = pr["peak_heights"] / sig
    d["D_nimp"] = len(pk)
    xs = xh[TR:-TR]; env = np.abs(signal.hilbert(xs))
    d["I_density"] = (np.convolve(env > np.median(env) * 6, np.ones(250), "same") > 0).mean()
    for k in ["I_top5_ratio", "I_count_per_max", "I_interval_cv", "R_fpeak_med", "R_frac_2_5k", "R_ring_25_58", "R_ring_811_58"]: d[k] = np.nan
    if len(amp):
        s_ = np.sort(amp)[::-1]; d["I_top5_ratio"] = s_[:5].mean() / s_[0]; d["I_count_per_max"] = len(amp) / s_[0]
    if len(pk) >= 3: iv = np.diff(pk); d["I_interval_cv"] = iv.std() / iv.mean()
    fr = np.fft.rfftfreq(1024, 1 / FS); fp, r25, r811 = [], [], []
    for p0 in pk[np.argsort(amp)[::-1][:20]]:
        seg = xs[p0:p0 + 300]
        if len(seg) < 300: continue
        Sp = np.abs(np.fft.rfft(seg * np.hanning(300), 1024)) ** 2; m = (fr >= 1000) & (fr < 12000)
        fp.append(fr[m][Sp[m].argmax()] / 1000)
        e = lambda a, b: Sp[(fr >= a) & (fr < b)].sum() + 1e-20
        r25.append(e(2000, 5000) / e(5000, 8000)); r811.append(e(8000, 11000) / e(5000, 8000))
    if fp:
        d["R_fpeak_med"] = np.median(fp); d["R_frac_2_5k"] = np.mean((np.array(fp) >= 2) & (np.array(fp) < 5))
        d["R_ring_25_58"] = np.median(r25); d["R_ring_811_58"] = np.median(r811)
    f, t, Sx = signal.stft(xs, FS, nperseg=512, noverlap=384); A2 = np.abs(Sx) ** 2
    sk = (A2 ** 2).mean(1) / (A2.mean(1) ** 2 + 1e-30) - 2; d["SK_200_3k"] = sk[(f >= 200) & (f < 3000)].mean()
    r = acf(xs, P)
    return d, r / r[0]

def extract(raw, names):
    rows, acs = [], []
    for n in names:
        d, ac = features(n, raw[n]); rows.append(d); acs.append(ac)
    return pd.DataFrame(rows), np.array(acs)

def build_X(df):
    X = pd.DataFrame(index=df.index); no_log = {"R_frac_2_5k", "SK_200_3k"}
    for c in df.columns:
        if c == "file": continue
        X[c] = df[c].values if c in no_log else np.log(df[c].clip(lower=1e-6 if c.startswith(("I_", "R_")) else 1e-9)).values
    X["R_811_over_58"] = np.log(df.b8_11_rms / df.b5_8_rms).values
    X["R_811_over_25"] = np.log(df.b8_11_rms / df.b25_5_rms).values
    X["AR_resid"] = np.nan
    return X

def ar_coef(ac_normals):
    r = ac_normals.mean(0); return linalg.solve_toeplitz(r[:P], r[1:P + 1])
def ar_resid(a, ac_rows):
    return np.log(np.clip([1 - 2 * a @ q[1:P + 1] + a @ linalg.toeplitz(q[:P]) @ a for q in ac_rows], 1e-6, None))

def select(Xtr, ytr, k):
    au = {c: abs(roc_auc_score(ytr, Xtr[c]) - .5) for c in Xtr}; order = sorted(au, key=au.get, reverse=True)
    C = Xtr.corr().abs(); sel = []
    for c in order:
        if all(C.loc[c, s] <= CFG["R_MAX"] for s in sel): sel.append(c)
    while len(sel) > 2:
        Zs = StandardScaler().fit_transform(Xtr[sel]); v = np.diag(np.linalg.pinv(np.corrcoef(Zs.T)))
        if v.max() > CFG["VIF_MAX"]: sel.pop(int(v.argmax()))
        else: break
    return sel[:k]

df_t, AC = extract(RAW, NAMES); df_v, ACv = extract(VRAW, VNAMES)
X = build_X(df_t); Xv = build_X(df_v)
CAND = ([c for c in X if c.startswith("A_")] + [c for c in X if c.startswith("b25_5_")] + [c for c in X if c.startswith("D_")]
        + ["R_811_over_58", "R_811_over_25"]
        + ["I_density", "I_top5_ratio", "I_count_per_max", "I_interval_cv", "R_fpeak_med", "R_frac_2_5k", "R_ring_25_58", "R_ring_811_58", "SK_200_3k", "AR_resid"])
A = ar_coef(AC[y == 0]); X["AR_resid"] = ar_resid(A, AC); Xv["AR_resid"] = ar_resid(A, ACv)
nan_t = X[CAND].isna().sum(); nan_v = Xv[CAND].isna().sum()
MED = X[CAND].median(); Xf = X[CAND].fillna(MED); Xvf = Xv[CAND].fillna(MED)

# %% 2. 150개 전체로 선택 → 소린 F10과 비교
F10 = ["b25_5_segk", "D_segk", "A_segk", "R_frac_2_5k", "R_ring_25_58", "AR_resid", "R_ring_811_58", "SK_200_3k", "b25_5_kurt", "I_density"]
sel = select(Xf, y, CFG["K_FEATURES"])
auc = {c: roc_auc_score(y, Xf[c]) for c in CAND}
nor = Xf[y == 0]
pct = {c: (nor[c].values[:, None] < Xvf[c].values[None, :]).mean(0) for c in CAND}   # val 파일별 train 정상 내 백분위
q95 = {c: (Xvf[c] > nor[c].quantile(.95)).mean() for c in CAND}

out = pd.concat([pd.DataFrame({"file": NAMES, "split": "train", "label": y}), Xf.reset_index(drop=True)], axis=1)
outv = pd.concat([pd.DataFrame({"file": VNAMES, "split": "val", "label": -1}), Xvf.reset_index(drop=True)], axis=1)
pd.concat([out, outv]).to_csv(os.path.join(FIG, "소린_파일표.csv"), index=False, encoding="utf-8-sig")

L = [f"후보 {len(CAND)}개 · train {len(NAMES)} · val {len(VNAMES)}",
     f"결측(충격 0 · 3개 미만 파일) train: {dict(nan_t[nan_t > 0])} / val: {dict(nan_v[nan_v > 0])}",
     f"150개 전체 선택 = {sel}", f"소린 F10      = {F10}", f"이름 · 순서 일치 = {sel == F10} / 집합 일치 = {set(sel) == set(F10)}", "",
     "피처 | 단독 AUC(고장 높음 > 0.5) | 선택 순위 | val 중앙값의 train 정상 백분위 | val 중 정상 95% 초과 비율"]
for c in sorted(CAND, key=lambda c: -abs(auc[c] - .5)):
    L.append(f"{c} | {auc[c]:.3f} | {sel.index(c) + 1 if c in sel else '-'} | {np.median(pct[c]) * 100:.0f}% | {q95[c] * 100:.0f}%")
open(os.path.join(FIG, "소린_결과.txt"), "w", encoding="utf-8").write("\n".join(L))
print("\n".join(L[:5]))
