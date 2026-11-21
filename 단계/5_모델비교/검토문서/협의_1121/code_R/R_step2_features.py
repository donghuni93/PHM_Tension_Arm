"""R 역할 2단계: 규칙 -> 라벨 없는 연속 피처 + 정상 99% 분위 전문가 + 세 번째 축.
정의는 이 파일 위쪽에서 고정(라벨을 보기 전)하고, 라벨은 맨 아래 '보고'에서만 쓴다.
정상 기준값(중앙값 · 분위)은 정상 파일 자신을 뺀 나머지 정상(LOO)으로 계산한다 -> 정상 오경보가 낙관적이지 않게.
출력: out/R_features.csv, out/R_report.txt
"""
import os, sys, pickle, io
here = os.path.dirname(os.path.abspath(__file__))
root = here
while not os.path.isdir(os.path.join(root, "src")):
    root = os.path.dirname(root)
sys.path.insert(0, root)
os.chdir(root)

import numpy as np
import pandas as pd

R = pickle.load(open(os.path.join(here, "out", "raw.pkl"), "rb"))
names, split, y, imps, T2 = R["names"], R["split"], R["y"], R["imps"], R["T2"]
f, PR = R["f"], R["PR"]
n = len(names)
NOR = np.flatnonzero((split == "train") & (y == 0))
FAU = np.flatnonzero((split == "train") & (y == 1))
VAL = np.flatnonzero(split == "val")
short = lambda s: s.replace("abn_", "A").replace("nor_", "N").replace("val_", "V")
sn = [short(s) for s in names]

# ============================================================================
# 정의 (사전 고정)
# ============================================================================
# ---- S: 스펙트럼 돌출 z (8.3k 스펙트럼 규칙의 라벨 없는 대체) -------------------
#  PR(f) = 55 Hz 평활 log PSD - 1.2 kHz 이동 중앙값 (4-8과 final과 같은 식, 주변 대비 돌출 dec)
#  z(f)  = (PR(f) - 정상 중앙 PR(f)) / scale(f),  scale = 정상 1.4826*MAD를 500 Hz로 고른 값, 하한 0.08 dec
#  mode_z = 2-12 kHz에서 max z, mode_f = 그 위치, mode_w = 그 봉우리 반높이 폭(Hz, PR 기준)
band = (f >= 2000) & (f <= 12000)
fb = f[band]
df = f[1] - f[0]
from scipy.ndimage import uniform_filter1d


def spec_ref(rows):
    P = PR[rows][:, band]
    med = np.median(P, 0)
    sc = uniform_filter1d(1.4826 * np.median(np.abs(P - med), 0), int(round(500 / df)))
    return med, np.maximum(sc, 0.08)


def spec_feats(i, med, sc):
    d = (PR[i, band] - med) / sc
    k = int(d.argmax()); z = d[k]
    p = PR[i, band]; h = p[k] / 2; l = r = k
    while l > 0 and p[l - 1] >= h: l -= 1
    while r < len(p) - 1 and p[r + 1] >= h: r += 1
    return z, fb[k], (r - l) * df


med_all, sc_all = spec_ref(NOR)
mode_z, mode_f, mode_w = np.zeros(n), np.zeros(n), np.zeros(n)
for i in range(n):
    if y[i] == 0 and split[i] == "train":
        med, sc = spec_ref(NOR[NOR != i])
    else:
        med, sc = med_all, sc_all
    mode_z[i], mode_f[i], mode_w[i] = spec_feats(i, med, sc)

# ---- Q: 충격 울림 놀람 (8.3k 공진 규칙의 라벨 없는 대체) ------------------------
#  충격마다 (f, zeta, fit)는 impact_rings_v2 (새 충격 정의). 정상 충격의 f 분포를 가우시안 KDE(Scott 폭)로 만들어
#  s_j = -log10 밀도(f_j) 를 정상 충격 s의 중앙값으로 맞춘 값(양수 = 정상에 드문 주파수).
#  q_j = fit_j * max(s_j, 0) ; 파일 값 ring_q = max_j q_j (충격 없으면 0). zeta는 쓰지 않는다(4-8: 반쪽 신뢰도 0.43).
#  정상 파일 자신의 충격은 KDE에서 뺀다 (LOO).
rf = [im["rings"][:, 0] if len(im["rings"]) else np.zeros(0) for im in imps]
rz = [im["rings"][:, 1] if len(im["rings"]) else np.zeros(0) for im in imps]
rt = [im["rings"][:, 2] if len(im["rings"]) else np.zeros(0) for im in imps]


def kde_logdens(ref, x):
    h = 1.06 * ref.std() * len(ref) ** (-0.2)
    return np.log10(np.mean(np.exp(-0.5 * ((x[:, None] - ref[None, :]) / h) ** 2), 1) / (h * np.sqrt(2 * np.pi)) + 1e-12)


def ring_scores(i, ref_files):
    ref = np.concatenate([rf[j] for j in ref_files])
    s_ref = -kde_logdens(ref, ref)
    s0 = np.median(s_ref)
    if len(rf[i]) == 0:
        return 0.0, 0.0
    s = -kde_logdens(ref, rf[i]) - s0
    q = rt[i] * np.maximum(s, 0)
    # 보조: zeta <= 0.02 (깨끗한 울림)만
    qc = np.where(rz[i] <= 0.02, q, 0)
    return float(q.max()), float(qc.max())


ring_q, ring_qc = np.zeros(n), np.zeros(n)
for i in range(n):
    ref = NOR[NOR != i] if (y[i] == 0 and split[i] == "train") else NOR
    ring_q[i], ring_qc[i] = ring_scores(i, ref)

# ---- J: 유격 반복 (K1-K3의 새 충격 정의 판) ---------------------------------
#  K1n = log10(2위 충격 높이 / 바탕)  (충격 < 2개면 log10(4.5)=문턱값)   [H2 높이 = 포락선/100 ms 중앙값 바탕]
#  K2n = -H18_hfl = log10(20 Hz 저역 RMS / 1 kHz 고역 포락선 중앙값)   (하중 대비 고주파의 거울, 크면 하중이 큼)
#  K3n = H2_ratio21 (2위/1위 높이, 충격 < 2개면 0)
#  r_n = sqrt((z(K1n)^2 + z(K2n)^2)/2)  z = 정상 중앙값 / 강건 규모(MAD, 0이면 P90 규모) -- final.robust_z와 같은 방식
#  J = min( r_n / q99(r_n), K3n / q99(K3n) )  : 정상 99% 선을 둘 다 넘어야 1 이상 ("그리고"를 비율의 최솟값으로)
K_MIN = 4.5
hx = [np.sort(im["height_x"])[::-1] for im in imps]
K1n = np.array([np.log10(h[1]) if len(h) > 1 else np.log10(K_MIN) for h in hx])
K3n = T2["H2_ratio21"].values.astype(float)
K2n = -T2["H18_hfl"].values.astype(float)
nimp = T2["H1_nimp"].values.astype(float)  # log1p(충격 수)
nimp_n = np.expm1(nimp).round().astype(int)


def rscale(v):
    med = np.median(v)
    sc = 1.4826 * np.median(np.abs(v - med))
    if sc < 1e-9:
        sc = (np.percentile(v, 90) - med) / 1.2816
        if sc < 1e-9:
            sc = v.std()
    return med, sc + 1e-9


def r_of(rows_ref, idx):
    m1, s1 = rscale(K1n[rows_ref]); m2, s2 = rscale(K2n[rows_ref])
    return np.sqrt((((K1n[idx] - m1) / s1) ** 2 + ((K2n[idx] - m2) / s2) ** 2) / 2)


r_n = np.zeros(n)
for i in range(n):
    ref = NOR[NOR != i] if (y[i] == 0 and split[i] == "train") else NOR
    r_n[i] = r_of(ref, np.array([i]))[0]
# r_n 정상 분위는 LOO 값으로 계산한 정상 120개의 q99
q_r, q_k3 = np.percentile(r_n[NOR], 99), np.percentile(K3n[NOR], 99)
J = np.minimum(r_n / q_r, K3n / q_k3)

# ---- E2: 하중 대비 덜컥 (EXP-042 그대로): H18 > 정상 99% 그리고 충격 >= 1 -------------
H18 = T2["H18_hfl"].values.astype(float)
H18g = np.where(nimp_n >= 1, H18, np.median(H18[NOR]))

# ============================================================================
# 전문가 점수 정규화와 세 번째 축 A3
#  u_k = (v_k - 정상 중앙) / (정상 q99 - 정상 중앙)  : 정상 99% 선이 u = 1 (정상 중 LOO)
#  A3 = max_k u_k  (k = S 스펙트럼 mode_z, Q 울림 놀람 ring_q, J 유격 반복, L 하중 대비 H18)
# ============================================================================
EXP = dict(S=mode_z, Q=ring_q, J=J, L=H18g)


def norm_u(v):
    u = np.zeros(n)
    for i in range(n):
        ref = NOR[NOR != i] if (y[i] == 0 and split[i] == "train") else NOR
        m, q = np.median(v[ref]), np.percentile(v[ref], 99)
        u[i] = (v[i] - m) / max(q - m, 1e-9)
    return u


U = {k: norm_u(v) for k, v in EXP.items()}
A3 = np.max(np.column_stack([U[k] for k in "SQJL"]), 1)
A3_noL = np.max(np.column_stack([U[k] for k in "SQJ"]), 1)
# A3 자신의 정상 99% 선 (LOO 정상 값의 분위)
thrA3 = np.percentile(A3[NOR], 99)
fire = {k: U[k] >= 1.0 for k in "SQJL"}
fire["A3"] = A3 >= thrA3

# ---- 원래 규칙 (비교용, 예전 정의 + 사후 문턱) -------------------------------
from src.final import R_MIN, RATIO_MIN
K = R["K"]
medK = np.median(K[NOR][:, :2], 0); madK = 1.4826 * np.median(np.abs(K[NOR][:, :2] - medK), 0) + 1e-12
r_old = np.sqrt((((K[:, :2] - medK) / madK) ** 2).mean(1))
old83s = (R["P83"] >= 0.75) & (R["W83"] >= 150)
old83r = R["RES"] >= 0.7
oldK = (r_old >= R_MIN) & (K[:, 2] >= RATIO_MIN)
old_any = old83s | old83r | oldK

df_out = pd.DataFrame(dict(file=names, split=split, y=y, nimp=nimp_n, mode_z=mode_z, mode_f=mode_f, mode_w=mode_w,
                           ring_q=ring_q, ring_qc=ring_qc, K1n=K1n, K2n=K2n, K3n=K3n, r_n=r_n, J=J, H18=H18,
                           uS=U["S"], uQ=U["Q"], uJ=U["J"], uL=U["L"], A3=A3, A3_noL=A3_noL,
                           old83s=old83s, old83r=old83r, oldK=oldK, r_old=r_old, K3_old=K[:, 2],
                           P83=R["P83"], W83=R["W83"], P86=R["P86"], RES=R["RES"]))
df_out.to_csv(os.path.join(here, "out", "R_features.csv"), index=False, encoding="utf-8-sig")

# ============================================================================
# 보고 (여기서만 라벨을 쓴다)
# ============================================================================
buf = io.StringIO()
def p(*a):
    print(*a, file=buf)

p("정상 99%% 선: q99(r_n)=%.3f q99(K3n)=%.3f thrA3=%.3f" % (q_r, q_k3, thrA3))
p("\n[전문가별 켜짐 수]  정상(LOO) / 고장 30 / val 40")
for k in ["S", "Q", "J", "L", "A3"]:
    fa = fire[k]
    p(f"  {k:>2}: 정상 {fa[NOR].sum():>2}  고장 {fa[FAU].sum():>2}  val {fa[VAL].sum():>2}")
p("  원래 규칙: 8.3k 스펙트럼 정상 %d 고장 %d val %d | 공진 정상 %d 고장 %d val %d | 유격 정상 %d 고장 %d val %d | 합집합 정상 %d 고장 %d val %d" % (
    old83s[NOR].sum(), old83s[FAU].sum(), old83s[VAL].sum(), old83r[NOR].sum(), old83r[FAU].sum(), old83r[VAL].sum(),
    oldK[NOR].sum(), oldK[FAU].sum(), oldK[VAL].sum(), old_any[NOR].sum(), old_any[FAU].sum(), old_any[VAL].sum()))

p("\n[켜진 정상 파일 (LOO)]")
for k in ["S", "Q", "J", "L", "A3"]:
    p(f"  {k}: " + " ".join(sn[i] for i in NOR if fire[k][i]))
p("\n[켜진 val 파일]")
for k in ["S", "Q", "J", "L", "A3"]:
    p(f"  {k}: " + " ".join(sn[i] for i in VAL if fire[k][i]))
p("  원래 규칙(합집합): " + " ".join(sn[i] for i in VAL if old_any[i]))
p("  원래 규칙(8.3k 스펙): " + " ".join(sn[i] for i in VAL if old83s[i]) + " | 공진: " + " ".join(sn[i] for i in VAL if old83r[i]) + " | 유격: " + " ".join(sn[i] for i in VAL if oldK[i]))

p("\n[val 이동: 정상 대비]  u 중앙값 정상 / val, val 중 u>=0.5 비율, u>=1 비율")
for k in "SQJL":
    p(f"  {k}: 중앙 {np.median(U[k][NOR]):.2f} / {np.median(U[k][VAL]):.2f}  u>=0.5 정상 {np.mean(U[k][NOR] >= .5):.3f} val {np.mean(U[k][VAL] >= .5):.3f}  u>=1 정상 {np.mean(U[k][NOR] >= 1):.3f} val {np.mean(U[k][VAL] >= 1):.3f}")
p("  A3 중앙 정상 %.2f / val %.2f ; A3 >= 0.5thr 정상 %.3f val %.3f" % (np.median(A3[NOR]), np.median(A3[VAL]), np.mean(A3[NOR] >= .5 * thrA3), np.mean(A3[VAL] >= .5 * thrA3)))
p("  val 충격 수 중앙 %.1f (정상 %.1f); val 충격 0개 %d, 1개 %d" % (np.median(nimp_n[VAL]), np.median(nimp_n[NOR]), (nimp_n[VAL] == 0).sum(), (nimp_n[VAL] == 1).sum()))

p("\n[원래 규칙이 잡던 고장 30개 중 라벨 보고 (옮긴 전문가가 다시 켜는가)]  u값 (>=1 켜짐), A3 켜짐")
groups = {"8.3k": ["002", "003", "006", "010", "029", "030"], "유격": ["003", "004", "008", "010", "011", "014", "017", "019", "028", "030"]}
hdr = "  파일  원래(8.3s 공진 유격) | uS   uQ   uJ   uL   A3(thr %.2f) | 켜짐" % thrA3
p(hdr)
for i in FAU:
    s = short(names[i]);
    p(f"  {s:>4}  {int(old83s[i])}   {int(old83r[i])}   {int(oldK[i])}       | {U['S'][i]:5.2f} {U['Q'][i]:5.2f} {U['J'][i]:5.2f} {U['L'][i]:5.2f}  {A3[i]:6.2f} | " + ("A3 " if fire['A3'][i] else "-- ") + "".join(k for k in "SQJL" if fire[k][i]))
miss_old = [i for i in FAU if old_any[i]]
p("  원래 규칙 합집합 고장 %d개 중 A3 켜짐 %d, S %d Q %d J %d L %d, 어느 하나라도(OR) %d" % (
    len(miss_old), sum(fire['A3'][i] for i in miss_old), sum(fire['S'][i] for i in miss_old), sum(fire['Q'][i] for i in miss_old),
    sum(fire['J'][i] for i in miss_old), sum(fire['L'][i] for i in miss_old), sum(any(fire[k][i] for k in 'SQJL') for i in miss_old)))
p("  원래 규칙이 안 잡던 고장 중 A3 켜짐: " + " ".join(short(names[i]) for i in FAU if (not old_any[i]) and fire['A3'][i]))
p("  고장 전체 30 중 A3 켜짐: " + " ".join(short(names[i]) for i in FAU if fire['A3'][i]))

p("\n[정상 상위 u 값 (경계 정상)]")
for k in "SQJL":
    o = NOR[np.argsort(-U[k][NOR])][:6]
    p(f"  {k}: " + " ".join(f"{sn[i]}:{U[k][i]:.2f}" for i in o))
o = NOR[np.argsort(-A3[NOR])][:8]
p("  A3: " + " ".join(f"{sn[i]}:{A3[i]:.2f}" for i in o))

p("\n[분해: 8.3k 규칙이 재는 것 — 파일별 P83 W83 P86 RES / mode_z mode_f mode_w ring_q]")
for s in ["A002", "A003", "A006", "A010", "A029", "A030", "N019", "N092", "N004", "N070", "N045", "N009", "N066"]:
    i = sn.index(s)
    p(f"  {s}: P83 {R['P83'][i]:.2f} W83 {R['W83'][i]:.0f} P86 {R['P86'][i]:.2f} RES {R['RES'][i]:.2f} | mode_z {mode_z[i]:.1f} f {mode_f[i]:.0f} w {mode_w[i]:.0f} ring_q {ring_q[i]:.2f} qc {ring_qc[i]:.2f} nimp {nimp_n[i]}")
p("\n[분해: 유격 규칙 — K1 K2 K3 r (예전) / K1n K2n K3n r_n J (새)]")
for s in ["A014", "A017", "A004", "A011", "A019", "A003", "A008", "A010", "A028", "A030", "A005", "A001", "A009", "N009", "N054", "N086", "N001", "N093", "N106"]:
    i = sn.index(s)
    p(f"  {s}: K1 {K[i,0]:.2f} K2 {K[i,1]:.2f} K3 {K[i,2]:.2f} r {r_old[i]:.2f} | K1n {K1n[i]:.2f} K2n {K2n[i]:.2f} K3n {K3n[i]:.2f} r_n {r_n[i]:.2f} J {J[i]:.2f} nimp {nimp_n[i]}")

# 단변량 AUC (150 train)
from sklearn.metrics import roc_auc_score
tr = np.flatnonzero(split == "train")
p("\n[단변량 AUC (train 150, 라벨 보고용)]")
for nm_, v in [("P83", R["P83"]), ("W83", R["W83"]), ("mode_z", mode_z), ("ring_q", ring_q), ("ring_qc", ring_qc), ("RES", R["RES"]),
               ("K1(old)", K[:, 0]), ("K2(old)", -K[:, 1]), ("K3(old)", K[:, 2]), ("r_old", r_old),
               ("K1n", K1n), ("K2n(=-H18)", K2n), ("K3n", K3n), ("r_n", r_n), ("J", J), ("H18", H18), ("A3", A3), ("A3_noL", A3_noL)]:
    p(f"  {nm_:>10}: {roc_auc_score(y[tr], v[tr]):.3f}")

# 새/예전 충격 정의 차이: K1과 K1n, K3과 K3n 순위상관
from scipy.stats import spearmanr
p("\n[새 충격 정의 vs 예전 K 상관 (정상 120)]  K1~K1n %.2f  K3~K3n %.2f  r_old~r_n %.2f" % (
    spearmanr(K[NOR, 0], K1n[NOR])[0], spearmanr(K[NOR, 2], K3n[NOR])[0], spearmanr(r_old[NOR], r_n[NOR])[0]))
p("  정상 중 K1 바닥값 (충격<2) 비율: 예전 %.2f 새 %.2f" % (np.mean(K[NOR, 0] <= np.log10(8) + 1e-9), np.mean(K1n[NOR] <= np.log10(K_MIN) + 1e-9)))
p("  정상 K3n 분위: " + " ".join(f"{q}%={np.percentile(K3n[NOR], q):.2f}" for q in (50, 90, 95, 99)) + " | 예전 K3: " + " ".join(f"{q}%={np.percentile(K[NOR,2], q):.2f}" for q in (50, 90, 95, 99)))

open(os.path.join(here, "out", "R_report.txt"), "w", encoding="utf-8").write(buf.getvalue())
print(buf.getvalue())
