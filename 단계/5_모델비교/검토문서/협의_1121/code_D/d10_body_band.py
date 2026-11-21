from common import *
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold
P = log_to("d10_body_band_log.txt")
X, y, names, Xv, nv = load_all(); nor = y == 0; n = len(y)
H = pd.read_csv(os.path.join(HERE, "D_file_feats.csv"), index_col=0).join(pd.read_csv(os.path.join(HERE, "D_repeat_feats.csv"), index_col=0)); Ht = H.loc[names]
# 1. 본체 울림 몫 정상 증거: 컷 폴드 안 재선택 (n_imp>=3 인 파일 중 학습 고장의 최대 body_frac 이상을 정상 증거로)
m3 = (Ht.n_imp >= 3).values; bf = Ht.body_frac.values.astype(float)
P("n_imp>=3 고장의 body_frac 상위:", sorted([(round(bf[i], 2), names[i]) for i in np.flatnonzero(m3 & (y == 1))], reverse=True)[:5], "정상 body_frac>=0.7:", int((m3 & nor & (bf >= 0.7)).sum()), "/ 정상 n_imp>=3:", int((m3 & nor).sum()))
cs = []; hit = {}; off = {}
for s in range(20):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
        mf = a[(y[a] == 1) & m3[a]]; c = np.nanmax(bf[mf]) + 1e-9; cs.append(round(c, 2))
        for i in b[m3[b] & (bf[b] > c)]: (hit if y[i] == 1 else off)[names[i]] = (hit if y[i] == 1 else off).get(names[i], 0) + 1
P("폴드 안 컷(학습 고장 최대 body_frac) 분포:", dict(zip(*[x.tolist() for x in np.unique(cs, return_counts=True)])), "| 테스트 켜진 고장:", hit, "| 켜진 정상 종류", len(off), "A오경보 포함:", {k: v for k, v in off.items() if k in ("nor_025", "nor_116", "nor_055", "nor_065", "nor_059")})
# 2. val AE 이동이 어느 대역인가: 200Hz 고역 이후 log PSD 를 1 kHz 칸으로 묶어 정상 vs val 중앙값 차이
def bandpsd(M):
    f, p = welch(M - M.mean(1, keepdims=True), fs=25000, nperseg=4096, axis=1)
    edges = np.arange(0, 12501, 1000); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        k = (f >= max(lo, 200)) & (f < hi); out.append(np.log10(p[:, k].mean(1) + 1e-20))
    return np.array(out).T
Bt, Bv = bandpsd(X), bandpsd(Xv)
P("== 2. 1 kHz 칸별 log10 PSD 중앙값: 정상(120) / val(40) / 차이 / 정상 IQR (절대 게인 포함) ==")
for j in range(Bt.shape[1]):
    nm = Bt[nor, j]; q1, q3 = np.percentile(nm, [25, 75]); d = np.median(Bv[:, j]) - np.median(nm)
    P(f"  {j}-{j+1}k: 정상 {np.median(nm):.2f} val {np.median(Bv[:, j]):.2f} 차 {d:+.2f} (정상 IQR {q3-q1:.2f}, 차/IQR {d/(q3-q1):+.2f})")
# 모양만(파일 평균 빼기)
St = Bt - Bt.mean(1, keepdims=True); Sv = Bv - Bv.mean(1, keepdims=True)
P("== 모양만(파일 평균 log PSD 뺀 값) ==")
for j in range(Bt.shape[1]):
    nm = St[nor, j]; q1, q3 = np.percentile(nm, [25, 75]); d = np.median(Sv[:, j]) - np.median(nm)
    P(f"  {j}-{j+1}k: 정상 {np.median(nm):+.2f} val {np.median(Sv[:, j]):+.2f} 차 {d:+.2f} (차/IQR {d/(q3-q1):+.2f})")
