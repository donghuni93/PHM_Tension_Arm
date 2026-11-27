"""R 역할 3단계 (2차 정의): 1차(R_step2) 결과를 보고 세 정의를 고쳤다 -> 이 사실을 R_1.md에 적는다 (라벨을 본 뒤 수정 = 탐색 낙관 가능).
고친 이유(물리/구조 이유, 라벨 아님):
  S2: 1차 S(전 대역 최대 z)는 8.7k 정상 봉우리(정상 8개)가 기준선을 올려 8.3k를 못 봄 -> '주파수별 정상 99% 덮개를 넘은 높이'로 (위치마다 정상이 얼마나 흔한지 반영)
  Q2: 1차 Q의 KDE 폭(약 560 Hz)이 울림 격자(250 Hz)보다 넓어 8.25k의 드묾이 7.25k에 번짐 -> 250 Hz 격자 계수로
  J2: 1차 J는 '정상 99%' 문턱을 K3(2위/1위)에 따로 걸어 원 규칙의 '둘 다' 구조(r가 먼저 정상 일부를 걸러냄)를 잃음 -> 두 축의 정상 백분위 순위의 최솟값
출력: out/R_report2.txt, out/R_features2.csv
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
from scipy.ndimage import maximum_filter1d
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

R = pickle.load(open(os.path.join(here, "out", "raw.pkl"), "rb"))
F1 = pd.read_csv(os.path.join(here, "out", "R_features.csv"), encoding="utf-8-sig")
names, split, y, imps, T2 = R["names"], R["split"], R["y"], R["imps"], R["T2"]
f, PR = R["f"], R["PR"]
n = len(names)
NOR = np.flatnonzero((split == "train") & (y == 0))
FAU = np.flatnonzero((split == "train") & (y == 1))
VAL = np.flatnonzero(split == "val")
TR = np.flatnonzero(split == "train")
short = lambda s: s.replace("abn_", "A").replace("nor_", "N").replace("val_", "V")
sn = [short(s) for s in names]
isnor = lambda i: (y[i] == 0 and split[i] == "train")
ref_of = lambda i: NOR[NOR != i] if isnor(i) else NOR
df = f[1] - f[0]
band = (f >= 2000) & (f <= 12000)
fb = f[band]

# ---- S2: 주파수별 정상 99% 덮개 초과 -----------------------------------------
#  env(f) = 정상 PR(f)의 99% 분위수를 +-100 Hz 최대값 필터로 번지게 한 것(해상도 오차 흡수). S2 = max_f (PR(f) - env(f)), 2-12 kHz
w = int(round(200 / df)) | 1


def env_of(rows):
    return maximum_filter1d(np.percentile(PR[rows][:, band], 99, axis=0), w)


env_all = env_of(NOR)
S2, S2f = np.zeros(n), np.zeros(n)
for i in range(n):
    env = env_of(ref_of(i)) if isnor(i) else env_all
    d = PR[i, band] - env
    k = int(d.argmax()); S2[i], S2f[i] = d[k], fb[k]

# ---- Q2: 250 Hz 격자 울림 드묾 -----------------------------------------------
#  정상 충격(LOO)의 울림 주파수를 250 Hz 격자로 세어 s(f) = -log10((c(f) + 0.5)/(N + 0.5*격자수)) , s0 = 정상 충격 s의 중앙값.
#  충격 j 점수 q_j = fit_j * max(s_j - s0, 0).  파일 점수: Qmax = max_j q_j, Qsum = sum_j q_j (한 사건에 몰린 것보다 여러 충격에 퍼진 것을 더 크게)
rf = [im["rings"][:, 0] if len(im["rings"]) else np.zeros(0) for im in imps]
rt = [im["rings"][:, 2] if len(im["rings"]) else np.zeros(0) for im in imps]
grid = np.arange(1000, 12001, 250)
gi = lambda a: np.clip(np.round((a - 1000) / 250).astype(int), 0, len(grid) - 1)


def ring_pair(i, rows):
    ref = np.concatenate([rf[j] for j in rows])
    c = np.bincount(gi(ref), minlength=len(grid)).astype(float)
    s = -np.log10((c + 0.5) / (len(ref) + 0.5 * len(grid)))
    s0 = np.median(s[gi(ref)])
    if len(rf[i]) == 0:
        return 0.0, 0.0
    q = rt[i] * np.maximum(s[gi(rf[i])] - s0, 0)
    return float(q.max()), float(q.sum())


Qmax, Qsum = np.zeros(n), np.zeros(n)
for i in range(n):
    Qmax[i], Qsum[i] = ring_pair(i, ref_of(i))

# ---- J2: 유격 반복 = 두 축의 정상 백분위의 최솟값 ---------------------------------
#  r_n, K3n은 1차와 같은 정의(새 충격 정의). 각 값의 정상(LOO) 백분위 순위 in [0,1] -> J2 = min(순위_r, 순위_K3).  문턱 = 정상 J2의 99%
r_n, K3n = F1["r_n"].values, F1["K3n"].values


def prank(v, i):
    ref = v[ref_of(i)]
    return (np.sum(ref < v[i]) + 0.5 * np.sum(ref == v[i])) / len(ref)


J2 = np.array([min(prank(r_n, i), prank(K3n, i)) for i in range(n)])


# ---- 정규화 u_k, 세 번째 축 ------------------------------------------------
def norm_u(v):
    u = np.zeros(n)
    for i in range(n):
        ref = ref_of(i)
        m, q = np.median(v[ref]), np.percentile(v[ref], 99)
        u[i] = (v[i] - m) / max(q - m, 1e-9)
    return u


H18g = np.where(np.expm1(T2["H1_nimp"].values).round() >= 1, T2["H18_hfl"].values, np.median(T2["H18_hfl"].values[NOR]))
V = dict(S2=S2, Qmax=Qmax, Qsum=Qsum, J2=J2, L=H18g)
U = {k: norm_u(v) for k, v in V.items()}


def axis(keys):
    A = np.max(np.column_stack([U[k] for k in keys]), 1)
    return A, np.percentile(A[NOR], 99)


combos = {"A3a (S2 Qmax J2 L)": ["S2", "Qmax", "J2", "L"], "A3b (S2 Qsum J2 L)": ["S2", "Qsum", "J2", "L"],
          "A3c (S2 Qmax J2)": ["S2", "Qmax", "J2"]}
AX = {k: axis(v) for k, v in combos.items()}

old_any = F1["old83s"].values | F1["old83r"].values | F1["oldK"].values
old83 = F1["old83s"].values | F1["old83r"].values

buf = io.StringIO()
def p(*a):
    print(*a, file=buf)

p("정상 99% 선: " + " ".join(f"{k}={np.percentile(U[k][NOR],99):.2f}(u)" for k in U))
p("\n[전문가 단독 켜짐: u>=1 (정상 LOO) / 고장 / val]")
for k in U:
    fa = U[k] >= 1
    p(f"  {k:>5}: 정상 {fa[NOR].sum():>2}  고장 {fa[FAU].sum():>2}  val {fa[VAL].sum():>2}   AUC(150) {roc_auc_score(y[TR], V[k][TR]):.3f}   정상: {' '.join(sn[i] for i in NOR if fa[i])}   val: {' '.join(sn[i] for i in VAL if fa[i])}")

p("\n[세 번째 축]")
for k, (A, th) in AX.items():
    fa = A >= th
    p(f"  {k}: thr {th:.2f}  정상 {fa[NOR].sum()} 고장 {fa[FAU].sum()} val {fa[VAL].sum()}  AUC {roc_auc_score(y[TR], A[TR]):.3f}")
    p(f"     켜진 고장: {' '.join(sn[i] for i in FAU if fa[i])}")
    p(f"     켜진 정상: {' '.join(sn[i] for i in NOR if fa[i])}   켜진 val: {' '.join(sn[i] for i in VAL if fa[i])}")
    oo = [i for i in FAU if old_any[i]]
    p(f"     원래 규칙 합집합 고장 {len(oo)}개 중 켜짐 {sum(fa[i] for i in oo)} / 8.3k군(002 003 006 010 029 030) {sum(fa[sn.index(s)] for s in ['A002','A003','A006','A010','A029','A030'])} / 014 017 {int(fa[sn.index('A014')])}{int(fa[sn.index('A017')])}")

p("\n[고장 파일별 u]  (>=1 켜짐)")
p("  파일  원래(8.3s 공진 유격) |  uS2  uQmax uQsum  uJ2   uL  | A3a")
A, th = AX["A3a (S2 Qmax J2 L)"]
for i in FAU:
    p(f"  {sn[i]:>4}  {int(F1.old83s[i])}   {int(F1.old83r[i])}   {int(F1.oldK[i])}       | {U['S2'][i]:5.2f} {U['Qmax'][i]:5.2f} {U['Qsum'][i]:5.2f} {U['J2'][i]:5.2f} {U['L'][i]:5.2f} | {A[i]:5.2f} {'ON' if A[i]>=th else ''}")

p("\n[정상 상위 u]")
for k in U:
    o = NOR[np.argsort(-U[k][NOR])][:6]
    p(f"  {k}: " + " ".join(f"{sn[i]}:{U[k][i]:.2f}" for i in o))

p("\n[8.3k 군과 경계 정상: S2 (초과 높이 dec, 위치) / Qmax Qsum / 격자 계수]")
for s in ["A002", "A003", "A006", "A010", "A029", "A030", "N019", "N092", "N004", "N070", "N045", "N009", "N066", "N065", "N116", "N059", "N055"]:
    i = sn.index(s)
    p(f"  {s}: S2 {S2[i]:.2f} @{S2f[i]:.0f} | Qmax {Qmax[i]:.2f} Qsum {Qsum[i]:.2f} | 충격 {len(rf[i])}, 울림 f {sorted(set(rf[i].astype(int).tolist()))}")

p("\n[val 이동 u: 정상 / val (중앙, u>=0.5 비율, u>=1)]")
for k in U:
    p(f"  {k:>5}: 중앙 {np.median(U[k][NOR]):.2f}/{np.median(U[k][VAL]):.2f}  u>=0.5 {np.mean(U[k][NOR]>=.5):.3f}/{np.mean(U[k][VAL]>=.5):.3f}  u>=1 {np.mean(U[k][NOR]>=1):.3f}/{np.mean(U[k][VAL]>=1):.3f}")
p("  val 상위 u (켜짐 직전 파일):")
for k in U:
    o = VAL[np.argsort(-U[k][VAL])][:4]
    p(f"    {k}: " + " ".join(f"{sn[i]}:{U[k][i]:.2f}" for i in o))
A3 = AX["A3a (S2 Qmax J2 L)"][0]
o = VAL[np.argsort(-A3[VAL])][:8]
p(f"    A3a 상위 val: " + " ".join(f"{sn[i]}:{A3[i]:.2f}" for i in o) + f"  (문턱 {AX['A3a (S2 Qmax J2 L)'][1]:.2f})")
p("    원래 규칙이 켠 val: " + " ".join(sn[i] for i in VAL if old_any[i]))

# 겹침: 원래 규칙과 A3a가 val에서 겹치는가
fa = A3 >= AX["A3a (S2 Qmax J2 L)"][1]
p("  val: A3a 켜짐 %d, 원래 규칙 켜짐 %d, 겹침 %d" % (fa[VAL].sum(), old_any[VAL].sum(), (fa & old_any)[VAL].sum()))

pd.DataFrame(dict(file=names, split=split, y=y, S2=S2, S2f=S2f, Qmax=Qmax, Qsum=Qsum, J2=J2, uS2=U["S2"], uQmax=U["Qmax"], uQsum=U["Qsum"],
                  uJ2=U["J2"], uL=U["L"], A3a=A3, A3b=AX["A3b (S2 Qsum J2 L)"][0], A3c=AX["A3c (S2 Qmax J2)"][0])).to_csv(
    os.path.join(here, "out", "R_features2.csv"), index=False, encoding="utf-8-sig")

# ---- quick_cv (탐색용, PR-AUC): ET 입력에 연속 피처를 더하면 오르나 -------------------
from src.evaluate import quick_cv
from src.exp import make_et
import json
FJ = json.load(open(os.path.join(root, "3_단계", "4_피처", "4-2_H피처목록", "자료", "최종피처.json"), encoding="utf-8"))
base = T2[list(FJ["fixed"]) + ["L1_S"]].values.astype(float)
cont = np.column_stack([S2, Qmax, Qsum, F1["K1n"].values, K3n, r_n, J2])
cont_names = ["S2", "Qmax", "Qsum", "K1n", "K3n", "r_n", "J2"]
g = np.arange(len(TR))
res = {}
for nm_, F in [("H(+박자)만", base), ("H + 연속 7열", np.hstack([base, cont])), ("H + S2 Qmax", np.hstack([base, np.column_stack([S2, Qmax])])),
               ("H + K1n K3n r_n J2", np.hstack([base, cont[:, 3:]])), ("A3a 단독 점수", A.reshape(-1, 1))]:
    if nm_ == "A3a 단독 점수":
        from sklearn.metrics import average_precision_score
        res[nm_] = (average_precision_score(y[TR], A[TR]), roc_auc_score(y[TR], A[TR]), None)
        continue
    q = quick_cv(F[TR], y[TR], g, make_et(300, 0), seeds=range(3))
    res[nm_] = (q["pr_auc"].mean(), q["roc_auc"].mean(), q["miss_rank"])
p("\n[quick_cv (탐색용, seed 3 x 5-fold): ET 입력 비교]  PR-AUC  ROC-AUC  | 핵심 고장 순위백분위(0 = 가장 잘 잡힘)")
key = [sn.index(s) for s in ["A002", "A014", "A017", "A022", "A029", "A003", "A006", "A010"]]
for k, (pr, ra, mr) in res.items():
    ex = "" if mr is None else "  ".join(f"{sn[i]}:{mr[i]:.2f}" for i in key)
    p(f"  {k:>16}: {pr:.3f}  {ra:.3f} | {ex}")

open(os.path.join(here, "out", "R_report2.txt"), "w", encoding="utf-8").write(buf.getvalue())
print(buf.getvalue())
