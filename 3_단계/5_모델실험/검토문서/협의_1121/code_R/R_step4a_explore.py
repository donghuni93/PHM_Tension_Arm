"""R 4a: 라벨을 쓴 탐색 (11-22 원칙). 8.3k 군과 유격 군을 라벨로 분해한다. 출력 out/R4a.txt"""
import os, sys, pickle, io
here = os.path.dirname(os.path.abspath(__file__)); root = here
while not os.path.isdir(os.path.join(root, "src")): root = os.path.dirname(root)
sys.path.insert(0, root); os.chdir(root)
import numpy as np, pandas as pd
from scipy.stats import fisher_exact
from sklearn.metrics import roc_auc_score
R = pickle.load(open(os.path.join(here, "out", "raw.pkl"), "rb"))
names, split, y, imps, T2, f, PR, K = R["names"], R["split"], R["y"], R["imps"], R["T2"], R["f"], R["PR"], R["K"]
n = len(names); TR = np.flatnonzero(split == "train"); VAL = np.flatnonzero(split == "val")
NOR = np.flatnonzero((split == "train") & (y == 0)); FAU = np.flatnonzero((split == "train") & (y == 1))
sn = [s.replace("abn_", "A").replace("nor_", "N").replace("val_", "V") for s in names]
buf = io.StringIO()
def p(*a): print(*a, file=buf)
auc = lambda v: roc_auc_score(y[TR], v[TR])

# --- 1. 주파수별 AUC 지도 (PR(f): 고장 > 정상)
df = f[1] - f[0]; band = (f >= 2000) & (f <= 12000)
Au = np.array([roc_auc_score(y[TR], PR[TR, k]) for k in np.flatnonzero(band)])
fb = f[band]
p("[1] PR(f) 주파수별 단변량 AUC, 상위 구간 (>=0.70, 라벨 사용)")
hi = Au >= 0.70
# 연속 구간으로 묶기
segs = []; k = 0
while k < len(fb):
    if hi[k]:
        j = k
        while j + 1 < len(fb) and hi[j + 1]: j += 1
        segs.append((fb[k], fb[j], Au[k:j + 1].max(), fb[k + Au[k:j + 1].argmax()])); k = j + 1
    else: k += 1
for a, b, m, fm in segs: p(f"   {a:7.0f}-{b:7.0f} Hz  최대 AUC {m:.3f} @ {fm:.0f}")
p("   8.0-8.7k 최대 AUC: %.3f @ %.0f" % (Au[(fb >= 8000) & (fb < 8700)].max(), fb[(fb >= 8000) & (fb < 8700)][Au[(fb >= 8000) & (fb < 8700)].argmax()]))

# --- 2. 울림 격자 셀: 깨끗한 울림(zeta<=0.02, fit>=0.6) 있는 파일 수
rf = [im["rings"][:, 0] if len(im["rings"]) else np.zeros(0) for im in imps]
rz = [im["rings"][:, 1] if len(im["rings"]) else np.zeros(0) for im in imps]
rt = [im["rings"][:, 2] if len(im["rings"]) else np.zeros(0) for im in imps]
p("\n[2] 250 Hz 셀별: 깨끗한 울림(zeta<=0.02, fit>=0.6)이 있는 파일 수 (고장30/정상120/val40) · Fisher p")
for c0 in range(7000, 11001, 250):
    has = np.array([bool(((rf[i] >= c0) & (rf[i] < c0 + 250) & (rz[i] <= 0.02) & (rt[i] >= 0.6)).any()) for i in range(n)])
    a, b_, v = has[FAU].sum(), has[NOR].sum(), has[VAL].sum()
    if a + b_ + v:
        pv = fisher_exact([[a, 30 - a], [b_, 120 - b_]])[1]
        p(f"   {c0}-{c0+250}: 고장 {a:2d} 정상 {b_:2d} val {v:2d}  p={pv:.1e}  고장: {' '.join(sn[i] for i in FAU if has[i])}  정상: {' '.join(sn[i] for i in NOR if has[i])}")

# --- 3. 8.3k 군 후보 연속 피처
def ringfeat(lo, hi, zmax=0.02):
    return np.array([ (rt[i][(rf[i] >= lo) & (rf[i] < hi) & (rz[i] <= zmax)].max() if ((rf[i] >= lo) & (rf[i] < hi) & (rz[i] <= zmax)).any() else 0.0) for i in range(n)])
def ringcnt(lo, hi, zmax=0.02, fmin=0.6):
    return np.array([ int(((rf[i] >= lo) & (rf[i] < hi) & (rz[i] <= zmax) & (rt[i] >= fmin)).sum()) for i in range(n)])
RES2 = ringfeat(8000, 8700)
P83, W83, P86 = R["P83"], R["W83"], R["P86"]
area = P83 * np.minimum(W83, 400)
cand = {"P83": P83, "W83": W83, "P83*W83": area, "P83-P86": P83 - P86, "RES2(8.0-8.7 새충격)": RES2, "RES(예전충격)": R["RES"],
        "ring8.0-8.5 fit": ringfeat(8000, 8500), "ring8.0-8.5 개수": ringcnt(8000, 8500).astype(float)}
p("\n[3] 8.3k 군 후보 단변량 AUC(150) · 고장30 중 / 정상120 중 켜진 값 분포")
for k_, v in cand.items():
    p(f"   {k_:>22}: AUC {auc(v):.3f}  고장 중앙 {np.median(v[FAU]):.2f}  정상 95% {np.percentile(v[NOR],95):.2f} 정상 max {v[NOR].max():.2f} ({sn[NOR[v[NOR].argmax()]]})  val max {v[VAL].max():.2f}")
grp83 = [sn.index(s) for s in ["A002", "A003", "A006", "A010", "A029", "A030"]]
p("\n   8.3k 군(002 003 006 010 029 030) 값 vs 정상 상위 5:")
for k_, v in cand.items():
    top = NOR[np.argsort(-v[NOR])][:5]
    p(f"   {k_:>22}: " + " ".join(f"{sn[i]}={v[i]:.2f}" for i in grp83) + " | 정상 " + " ".join(f"{sn[i]}={v[i]:.2f}" for i in top))
# 라벨 없는 기준(예전 사후 규칙)과의 일치
p("   old83s(P83>=.75&W83>=150) 고장 %s / 정상 %s / val %s" % (R['P83'][FAU][(R['P83'][FAU] >= .75) & (R['W83'][FAU] >= 150)].size, int(((P83 >= .75) & (W83 >= 150))[NOR].sum()), [sn[i] for i in VAL if P83[i] >= .75 and W83[i] >= 150]))
p("   RES2>=0.7: 고장 %s / 정상 %s / val %s" % ([sn[i] for i in FAU if RES2[i] >= .7], [sn[i] for i in NOR if RES2[i] >= .7], [sn[i] for i in VAL if RES2[i] >= .7]))
p("   RES(예전)>=0.7: 고장 %s / 정상 %s / val %s" % ([sn[i] for i in FAU if R['RES'][i] >= .7], [sn[i] for i in NOR if R['RES'][i] >= .7], [sn[i] for i in VAL if R['RES'][i] >= .7]))

# --- 4. 유격 군
hx = [np.sort(im["height_x"])[::-1] for im in imps]
K1n = np.array([np.log10(h[1]) if len(h) > 1 else np.log10(4.5) for h in hx])
K3n = T2["H2_ratio21"].values.astype(float); K2n = -T2["H18_hfl"].values.astype(float)
nimp = np.expm1(T2["H1_nimp"].values).round().astype(int)
def m_sim(h, a):  # 최대 높이의 a배 이상인 충격 수
    return float((h >= a * h[0]).sum()) if len(h) else 0.0
M70 = np.array([m_sim(h, .7) for h in hx]); M50 = np.array([m_sim(h, .5) for h in hx])
frac70 = np.array([ (h >= .7 * h[0]).sum() / len(h) if len(h) else 0 for h in hx])
cv_top = np.array([ (np.std(h[:min(4,len(h))]) / np.mean(h[:min(4,len(h))])) if len(h) >= 3 else 1.0 for h in hx])  # 상위 4개 높이 변동계수
H18 = T2["H18_hfl"].values.astype(float)
cand2 = {"K1n(2위/바탕)": K1n, "K2n(하중/고주파)": K2n, "K3n(2위/1위)": K3n, "충격수": nimp.astype(float), "M70(>=0.7*max 개수)": M70, "M50": M50, "frac70": frac70, "상위4 변동계수(-)": -cv_top, "H18": H18, "-H18": -H18}
p("\n[4] 유격 군 후보 단변량 AUC(150)")
for k_, v in cand2.items(): p(f"   {k_:>22}: AUC {auc(v):.3f}  고장 중앙 {np.median(v[FAU]):.2f}  정상 중앙 {np.median(v[NOR]):.2f}  정상 95% {np.percentile(v[NOR],95):.2f}")
grpK = [sn.index(s) for s in ["A003", "A004", "A008", "A010", "A011", "A014", "A017", "A019", "A028", "A030"]]
p("\n   유격 군 10개 값:")
for k_, v in cand2.items(): p(f"   {k_:>22}: " + " ".join(f"{sn[i]}={v[i]:.2f}" for i in grpK))
# 하중 계열: 저주파 RMS (K2n)에서 고장 군이 갈리는 모양
p("\n   유격 군 (K2n 부호: 하중 작음 = K2n 음수):  하중 작은 무리 A004 A011 A019 A017: " + " ".join(f"{sn[sn.index(s)]}:K2n={K2n[sn.index(s)]:.2f} nimp={nimp[sn.index(s)]}" for s in ["A004","A011","A019","A017"]))
p("   정상 중 K2n<=-0.8 개수: %d (%s)" % ((K2n[NOR] <= -0.8).sum(), " ".join(sn[i] for i in NOR if K2n[i] <= -0.8)))
p("   고장 중 K2n<=-0.8: " + " ".join(sn[i] for i in FAU if K2n[i] <= -0.8))
p("   E2 규칙(H18>정상q99 and 충격>=1) 정상 q99 H18=%.2f: 고장 %s" % (np.percentile(H18[NOR], 99), [sn[i] for i in FAU if H18[i] > np.percentile(H18[NOR], 99) and nimp[i] >= 1]))
np.savez(os.path.join(here, "out", "R4a.npz"), RES2=RES2, K1n=K1n, K2n=K2n, K3n=K3n, M70=M70, M50=M50, frac70=frac70, cv_top=cv_top, nimp=nimp, area=area)
open(os.path.join(here, "out", "R4a.txt"), "w", encoding="utf-8").write(buf.getvalue())
