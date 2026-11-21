# F 피처 담당 step3 (11-22 사용자 원칙: 라벨을 적극 사용, 정직성 = 기전 + 겨냥 3개 이상 + 폴드 안 재선택 안정)
# 1) 라벨로 본 울림 주파수 칸 (250 Hz 격자): 고장 쪽/정상 쪽 파일 수, 정확검정
# 2) 경계(고주파 b_hi, 저/본체 b_lo)를 라벨로 고르고, CV 학습 폴드(5겹 x seed 20)마다 같은 방법으로 다시 골라 안정성 확인
# 3) 후보 피처 공간(ext150 / A 입력 / 묶음 대표 / 폴드 안 AUC로 거른 안정 열 / new16 / ring 피처 조합)의 고장 5-NN 순도(폴드 밖 선택) + t-SNE(라벨은 색칠만)
# 4) 피처별 단변량 AUC · val 이동 · 상관, quick_cv(ET 축만, 탐색용)
# 결과 result_F5.txt (UTF-8), 그림 figs_F5/
import os, sys, json, warnings, collections
warnings.filterwarnings("ignore")
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(_f == x.name for x in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
from scipy.stats import spearmanr, fisher_exact
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE
from sklearn.base import BaseEstimator
from src.exp import make_et
from src.evaluate import quick_cv
from src.features2 import robust_z

HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs_F5"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(HERE, "result_F5.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); OUT.write(s + "\n"); OUT.flush()

# ---------------- 데이터 ----------------
IMP = pd.read_csv(os.path.join(HERE, "impacts_new.csv"), encoding="utf-8-sig")
NI = pd.read_csv(os.path.join(HERE, "nimp.csv"), encoding="utf-8-sig")
files = NI.file.tolist(); N = len(files); ntr = 150
yall = NI.y.values.astype(int); y = yall[:ntr]; nor = y == 0; fault = y == 1
MM = pd.read_csv(os.path.join(_d, "단계", "4_피처", "4-8_1121_모드지도", "features_modemap.csv"), encoding="utf-8-sig").set_index("file").loc[files]
EX = pd.read_csv(os.path.join(_d, "outputs", "cache", "feat_ext.csv")); EXv = pd.read_csv(os.path.join(_d, "outputs", "cache", "feat_ext_val.csv"))
assert list(EX.columns) == list(EXv.columns)
E_all = np.nan_to_num(np.vstack([EX.values, EXv.values])); ecols = list(EX.columns)
F2 = pd.concat([pd.read_csv(os.path.join(_d, "outputs", "cache", "feat2_train.csv")), pd.read_csv(os.path.join(_d, "outputs", "cache", "feat2_val.csv"))], ignore_index=True)
FINAL16 = json.load(open(os.path.join(_d, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))["final"]
FINAL16 = [c for c in FINAL16 if c in F2.columns] + [c for c in ("L1_S",) if c not in FINAL16]   # H16-H17(PeakRef, 폴드 안 정상 기준)은 표에 없어 뺌 -> 14열
NEW16 = np.nan_to_num(F2[FINAL16].values.astype(float))
OLD = pd.read_csv(os.path.join(_d, "단계", "2_EDA", "8kHz공진", "2-16_1116_울림주파수", "figs", "impacts.csv"), encoding="utf-8-sig")
def oldfrac(lo, hi):
    return np.nan_to_num(OLD.assign(b=(OLD.f >= lo) & (OLD.f < hi)).groupby("file").b.mean().reindex(files).values.astype(float), nan=-1.0)
AOLD = np.column_stack([oldfrac(2125, 4125), oldfrac(5500, 7750)])      # A의 울림 2열(예전 정의, 라벨 보고 고른 대역)
ground = {k: v for k, v in IMP.groupby("file")}
FILE_F = [ground[fl].f.values if fl in ground else np.zeros(0) for fl in files]
FILE_FIT = [ground[fl].fit.values if fl in ground else np.zeros(0) for fl in files]
BINS = np.arange(1000, 12001, 250); NB = len(BINS)
C_any = np.zeros((N, NB)); C_fit = np.zeros((N, NB))
for i in range(N):
    k = np.clip(np.round((FILE_F[i] - 1000) / 250).astype(int), 0, NB - 1)
    for kk, ft in zip(k, FILE_FIT[i]):
        C_any[i, kk] += 1; C_fit[i, kk] += ft >= 0.5
nimp = C_any.sum(1)
BODY = 7000.0

# ---------------- 1) 칸별 고장/정상 파일 수 (라벨 사용) ----------------
P("== 1) 250 Hz 칸별: 이 칸에 충격이 하나라도 있는 파일 수 (train 고장 30 / 정상 120 / val 40) · 정확검정 p(고장 쪽) ==")
tab = []
for k in range(NB):
    has = C_any[:, k] > 0
    a, b, c = int(has[:ntr][fault].sum()), int(has[:ntr][nor].sum()), int(has[ntr:].sum())
    if a + b + c == 0: continue
    p = fisher_exact([[a, 30 - a], [b, 120 - b]], alternative="greater")[1]
    tab.append((BINS[k], a, b, c, p))
for f0, a, b, c, p in tab:
    flag = "  <-- 고장 쪽" if (p < 0.01 and a >= 3) else ""
    P(f"  {f0 / 1000:5.2f}k | 고장 {a:2d} | 정상 {b:3d} | val {c:2d} | p={p:.4f}{flag}")
# 큰 틀: 3 영역 (저 <3.8k / 본체 / 고 >=7.9k) 의 파일 수
def has_ge(b): return (C_any[:, int((b - 1000) // 250):].sum(1) > 0)
for b in (7750, 8000, 8250, 8500, 9000, 9250):
    h = has_ge(b); P(f"  [단일 문턱 f>={b / 1000:.2f}k 충격 >=1개] 고장 {h[:ntr][fault].sum()}/30 · 정상 {h[:ntr][nor].sum()}/120 · val {h[ntr:].sum()}/40 | 고장 파일: {[files[i][-3:] for i in np.flatnonzero(h[:ntr] & fault)]} | 정상 파일: {[files[i][-3:] for i in np.flatnonzero(h[:ntr] & nor)]} | val: {[files[ntr + i][-3:] for i in np.flatnonzero(h[ntr:])]}")

# ---------------- 2) 경계 선택 + 폴드 안 재선택 안정성 ----------------
HI_GRID = np.arange(7500, 10501, 250); LO_GRID = np.arange(2500, 5501, 250)
def kidx(b): return int((b - 1000) // 250)
def sel_hi(idx, use_fit=False):
    """고주파 경계 b_hi: 학습 파일에서 '충격 >= b에 하나라도' 지표의 TPR - FPR(Youden) 최대, 같은 값이 이어지면 구간 가운데"""
    C = C_fit if use_fit else C_any
    yy = yall[idx]; best = []; Js = []
    for b in HI_GRID:
        h = C[idx][:, kidx(b):].sum(1) > 0
        Js.append(h[yy == 1].mean() - h[yy == 0].mean())
    Js = np.array(Js); m = np.flatnonzero(Js >= Js.max() - 1e-12)
    return float(HI_GRID[m][len(m) // 2]), float(Js.max())
def sel_lo(idx):
    """저/본체 경계 b_lo: 학습 파일에서 '충격 중 f<b 몫'의 단변량 AUC 최대(0.003 안 동률은 구간 가운데)"""
    yy = yall[idx]; A = []
    for b in LO_GRID:
        sh = np.where(nimp[idx] > 0, C_any[idx][:, :kidx(b)].sum(1) / np.maximum(nimp[idx], 1), 0.0)
        A.append(roc_auc_score(yy, sh))
    A = np.array(A); m = np.flatnonzero(A >= A.max() - 0.003)
    return float(LO_GRID[m][len(m) // 2]), float(A.max())
WA_GRID = np.arange(2000, 3251, 250); WB_GRID = np.arange(3250, 5251, 250)
def sel_win(idx):
    """중간 창 [a,b): 학습 파일에서 '창 안에 충격 >=1' 지표의 TPR-FPR 최대. 같은 값 모음의 가운데 a, 가운데 b"""
    yy = yall[idx]; best = (-9, []); rows = []
    for a in WA_GRID:
        for b in WB_GRID:
            h = C_any[idx][:, kidx(a):kidx(b)].sum(1) > 0
            J = h[yy == 1].mean() - h[yy == 0].mean(); rows.append((J, a, b))
    J0 = max(r[0] for r in rows); top = [r for r in rows if r[0] >= J0 - 0.01]
    return float(np.median([r[1] for r in top])), float(np.median([r[2] for r in top])), float(J0)
idx_all = np.arange(ntr)
vals = []
for s_ in range(20):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s_).split(idx_all, y, idx_all):
        vals.append(sel_win(tr)[:2])
vals = np.array(vals); full_w = sel_win(idx_all)
P(f"\n== 2) 중간 창 [a,b): 전체 150 선택 {full_w[0]:.0f}-{full_w[1]:.0f} Hz (J {full_w[2]:.3f}) | 폴드 안 100번: a 중앙 {np.median(vals[:,0]):.0f} 범위 {vals[:,0].min():.0f}-{vals[:,0].max():.0f} {dict(sorted(collections.Counter(vals[:,0].astype(int)).items()))} | b 중앙 {np.median(vals[:,1]):.0f} 범위 {vals[:,1].min():.0f}-{vals[:,1].max():.0f} {dict(sorted(collections.Counter(vals[:,1].astype(int)).items()))}")
for wa, wb in ((2250, 4000), (2250, 3750), (2500, 4000), (3000, 5000)):
    h = C_any[:ntr][:, kidx(wa):kidx(wb)].sum(1) > 0; hv = C_any[ntr:][:, kidx(wa):kidx(wb)].sum(1) > 0
    P(f"   창 [{wa},{wb}) 충격>=1: 고장 {h[fault].sum()}/30 · 정상 {h[nor].sum()}/120 · val {hv.sum()}/40 | 고장 {[files[i][-3:] for i in np.flatnonzero(h & fault)]} | 정상 {[files[i][-3:] for i in np.flatnonzero(h & nor)]} | val {[files[ntr+i][-3:] for i in np.flatnonzero(hv)]}")
hg = [(a_, int((C_any[:ntr][:, kidx(a_):].sum(1) > 0)[fault].sum())) for a_ in (2000,)]
for nm, fn in (("b_hi(any)", lambda ix: sel_hi(ix, False)), ("b_hi(fit>=0.5)", lambda ix: sel_hi(ix, True)), ("b_lo(참고)", sel_lo)):
    full = fn(idx_all); vals = []
    for s in range(20):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx_all, y, idx_all):
            vals.append(fn(tr)[0])
    vals = np.array(vals); cnt = collections.Counter(vals.astype(int)); mode = cnt.most_common(1)[0][0]
    P(f"\n== 2) {nm}: 전체 150에서 선택 {full[0]:.0f} Hz (기준값 {full[1]:.3f}) | 폴드 안 재선택 100번: 중앙 {np.median(vals):.0f}, 최빈 {mode}({cnt[mode]}번), 범위 {vals.min():.0f}-{vals.max():.0f}, 최빈 ±250 Hz 안 {np.mean(np.abs(vals - mode) <= 250):.2f}, 분포 {dict(sorted(cnt.items()))}")
BHI_FULL, _ = sel_hi(idx_all, False); BLO_FULL, _ = sel_lo(idx_all); WA_FULL, WB_FULL = full_w[0], full_w[1]

# 고주파 칸의 AUC는 문턱 위치에 얼마나 민감한가(평탄도): b_hi를 옮길 때 고장 수/정상 수
P("\n   b_hi 이동에 따른 (고장,정상) 파일 수, train 150:")
for b in HI_GRID:
    h = C_any[:ntr][:, kidx(b):].sum(1) > 0
    P(f"     b={b / 1000:.2f}k: ({h[fault].sum()},{h[nor].sum()})", end="") if False else None
P("    " + " | ".join(f"{b / 1000:.2f}k ({int((C_any[:ntr][:, kidx(b):].sum(1) > 0)[fault].sum())},{int((C_any[:ntr][:, kidx(b):].sum(1) > 0)[nor].sum())})" for b in HI_GRID))

# ---------------- 3) ring 피처 정의 ----------------
RING_COLS = ["ring_lo_share", "ring_mid_share", "ring_body_share", "ring_hi_share", "ring_mid_n", "ring_hi_n", "ring_mid_fit", "ring_hi_fit", "dev_tsa", "ev_hi_top1", "ev_tone_prom", "ev_tone_w"]
TONE = pd.read_csv(os.path.join(HERE, "tone.csv"), encoding="utf-8-sig").set_index("file").loc[files]
TONE_ARR = np.column_stack([TONE.tone_prom.values, np.log1p(TONE.tone_w.values)])
FILE_H = [ground[fl].height_g.values if fl in ground else np.zeros(0) for fl in files]
FT = MM.f_tsa.values.astype(float)
DEV_TSA = np.where(np.isnan(FT), 0.0, np.abs(np.log2(np.where(np.isnan(FT), BODY, FT) / BODY)))
def ring_feats(wa, wb, b_hi):
    """낮은 모드(<a) / 중간 창[a,b) / 본체[b,hi) / 고주파(>=hi) 4칸의 충격 몫, 중간·고주파의 충격 수(log1p)와 최대 fit, 대표 울림 이탈 dev_tsa"""
    M = np.zeros((N, 10))
    for i in range(N):
        f, ft = FILE_F[i], FILE_FIT[i]; n = len(f)
        if n:
            lo, mid, hi = f < wa, (f >= wa) & (f < wb), f >= b_hi; bd = ~(lo | mid | hi)
            hh = FILE_H[i][hi]; top1 = (hh.max() / hh.sum()) if hi.any() else 0.0
            M[i] = [lo.mean(), mid.mean(), bd.mean(), hi.mean(), np.log1p(mid.sum()), np.log1p(hi.sum()), (ft[mid].max() if mid.any() else 0.0), (ft[hi].max() if hi.any() else 0.0), DEV_TSA[i], top1]
        else:
            M[i, -1] = DEV_TSA[i]
    return np.hstack([M, TONE_ARR])
RING_FULL = ring_feats(WA_FULL, WB_FULL, BHI_FULL)
P(f"\n전체 150으로 고른 경계: 중간창=[{WA_FULL:.0f},{WB_FULL:.0f}) Hz, b_hi={BHI_FULL:.0f} Hz, (참고) b_lo={BLO_FULL:.0f} (라벨 사용)")

# ---------------- ext150 묶음 대표(라벨 없음) + 폴드 안 AUC 거름 ----------------
R = np.abs(np.nan_to_num(spearmanr(E_all).correlation, nan=0.0)); np.fill_diagonal(R, 1.0)
L = linkage(squareform(1 - R, checks=False), "complete"); cl = fcluster(L, t=0.2, criterion="distance")
groups = {}
for j, c in enumerate(cl): groups.setdefault(c, []).append(j)
REP = sorted(js[0] if len(js) == 1 else js[int(np.argmax([np.mean([R[a, b] for b in js if b != a]) for a in js]))] for js in groups.values())
P(f"\next150 -> |rho|>=0.8 완전연결 묶음(train+val 190, 라벨 없음) 대표 {len(REP)}열")
AUC_KEEP = 0.15      # 사전 고정: 학습 폴드에서 |AUC-0.5| >= 0.15인 대표 열만 남김
def uni_auc(M, idx):
    yy = yall[idx]; out = np.empty(M.shape[1])
    for j in range(M.shape[1]):
        v = M[idx, j]
        out[j] = 0.5 if np.ptp(v) == 0 else roc_auc_score(yy, v)
    return out
def sel_cols(M, idx, thr=AUC_KEEP):
    a = uni_auc(M, idx); return np.flatnonzero(np.abs(a - 0.5) >= thr)
Mrep = E_all[:, REP]
cnt_sel = np.zeros(len(REP)); nsel = 0
for s in range(20):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx_all, y, idx_all):
        cnt_sel[sel_cols(Mrep, tr)] += 1; nsel += 1
freq = cnt_sel / nsel
STABLE = np.flatnonzero(freq >= 0.8); UNSTABLE = np.flatnonzero((freq > 0.2) & (freq < 0.8)); NEVER = np.flatnonzero(freq <= 0.2)
P(f"== 대표 {len(REP)}열 중 폴드 안 AUC 거름(|AUC-.5|>=0.15) 결과(100번): 80% 이상 선택 {len(STABLE)}열(안정), 20-80% {len(UNSTABLE)}열(흔들림), 20% 이하 {len(NEVER)}열(기여 없음)")
fam = lambda c: c.split("_")[0]
P("   안정 열: " + ", ".join(f"{ecols[REP[j]]}({freq[j]:.2f})" for j in STABLE))
P("   흔들림 열: " + ", ".join(f"{ecols[REP[j]]}({freq[j]:.2f})" for j in UNSTABLE))
P("   기여 없음 가족 수: " + str(dict(collections.Counter(fam(ecols[REP[j]]) for j in NEVER))))

cnt8 = np.zeros(len(REP)); 
for s_ in range(20):
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s_).split(idx_all, y, idx_all):
        cnt8[sel_cols(Mrep, tr, 0.08)] += 1
f8 = cnt8 / 100
P(f"== 약한 열 제거(|AUC-.5|>=0.08): 80% 이상 남음 {int((f8>=.8).sum())}열 / 20-80% {int(((f8>.2)&(f8<.8)).sum())}열 / 20% 이하 {int((f8<=.2).sum())}열(제거). 제거 열: " + ", ".join(f"{ecols[REP[j]]}({f8[j]:.2f})" for j in np.flatnonzero(f8 <= .2)))
# ---------------- 공간 정의 (열 선택·경계를 학습 폴드에서 다시 고름) ----------------
DCABS = np.abs(E_all[:, ecols.index("T06_raw_mean")])
rho_dc = np.array([abs(np.nan_to_num(spearmanr(Mrep[:, j], DCABS).correlation)) for j in range(Mrep.shape[1])])
NODC = set(np.flatnonzero(rho_dc < 0.5).tolist())
P(f"\n== 방향(평균 DC) 의존 열: 대표 {len(REP)}열 중 |평균 DC|와 |rho|>=0.5 인 열 {int((rho_dc>=0.5).sum())}개: " + ", ".join(f"{ecols[REP[j]]}({rho_dc[j]:.2f})" for j in np.flatnonzero(rho_dc >= 0.5)))
def build(space, tr, thr=0.08):
    tr = idx_all if tr is None else tr
    if space == "ext150": return E_all
    if space == "A_in(ext150+예전울림2열)": return np.hstack([E_all, AOLD])
    if space == "ext_rep60": return Mrep
    if space == "ext_rep60_약한열제거": return Mrep[:, sel_cols(Mrep, tr, thr)]
    if space == "new14": return NEW16
    wa, wb, _ = sel_win(tr); bh = sel_hi(tr)[0]; Rg = ring_feats(wa, wb, bh)
    base = {"ring10": Rg[:, :10], "ring12": Rg, "ev4": Rg[:, [2, 9, 10, 11]]}
    if space in base: return base[space]
    if space == "ext_rep60_약한열제거+ring12": return np.hstack([Mrep[:, sel_cols(Mrep, tr, thr)], Rg])
    if space == "ext_rep60_약한열제거_noDC+ring12":
        ks = [j for j in sel_cols(Mrep, tr, thr) if j in NODC]; return np.hstack([Mrep[:, ks], Rg])
    if space == "new14+ring12": return np.hstack([NEW16, Rg])
    if space == "rep약한제거+new14+ring12": return np.hstack([Mrep[:, sel_cols(Mrep, tr, thr)], NEW16, Rg])
    raise KeyError(space)
SPACES = ["ext150", "A_in(ext150+예전울림2열)", "ext_rep60", "ext_rep60_약한열제거", "new14", "ring10", "ev4", "ring12", "ext_rep60_약한열제거+ring12", "ext_rep60_약한열제거_noDC+ring12", "new14+ring12", "rep약한제거+new14+ring12"]
FAN = ["nor_" + k for k in ("065", "116", "037", "110", "095", "059", "055", "025", "097", "101", "120")]
isFA = np.array([f in FAN for f in files[:ntr]]); fa_idx = np.flatnonzero(isFA)

def standardize(M, tr):
    """학습 폴드 정상의 로버스트 z (4-7과 같은 방식), 열마다"""
    trn = np.array([i for i in tr if yall[i] == 0]); Z = np.empty_like(M, dtype=float)
    for j in range(M.shape[1]):
        Z[:, j] = robust_z(M[trn, j])(M[:, j])
    return Z
def purity(Z, who):
    """who(train 번호)의 고장 점: 자기 제외 train 150 중 5-NN 이웃의 고장 비율"""
    nb = NearestNeighbors(n_neighbors=6).fit(Z[:ntr])
    idx = nb.kneighbors(Z[who], return_distance=False)[:, 1:]
    return float(y[idx].mean())
def npurity(Z, who):
    nb = NearestNeighbors(n_neighbors=6).fit(Z[:ntr])
    idx = nb.kneighbors(Z[who], return_distance=False)[:, 1:]
    return float(1 - y[idx].mean())
RANDOM = (fault.sum() - 1) / (ntr - 1)
P(f"\n== 3) 고장 5-NN 순도(자기 제외 train 150, 무작위 {RANDOM:.3f}). '폴드 밖' = 학습 폴드로 열·경계를 다시 골라 시험 폴드 고장에서 잰 값(5겹 x seed 5, 정직), '전체선택' = 150 전체로 고름(낙관적). clip = z를 ±10으로 자름 ==")
res_pur = {}; nor_idx = np.flatnonzero(nor)
fault_idx = np.flatnonzero(fault)
for sp in SPACES:
    # 전체 선택
    M = build(sp, None); Z = standardize(M, idx_all); full = purity(Z, fault_idx); fullc = purity(np.clip(Z, -10, 10), fault_idx); fa_full = npurity(np.clip(Z, -10, 10), fa_idx); nn_full = npurity(np.clip(Z, -10, 10), nor_idx)
    # 폴드 밖
    vals, valsc, vfa = [], [], []
    for s in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx_all, y, idx_all):
            Mf = build(sp, tr); Zf = standardize(Mf, tr); who = te[fault[te]]
            if len(who): vals.append(purity(Zf, who)); valsc.append(purity(np.clip(Zf, -10, 10), who))
            wn = np.array([i for i in te if i < ntr and isFA[i]])
            if len(wn): vfa.append(npurity(np.clip(Zf, -10, 10), wn))
    res_pur[sp] = dict(dim=M.shape[1], full=full, full_clip=fullc, oof=float(np.mean(vals)), oof_clip=float(np.mean(valsc)), oof_sd=float(np.std(vals)), fa_full=fa_full, nn_full=nn_full, fa_oof=float(np.mean(vfa)))
    r = res_pur[sp]
    P(f"  {sp:28s} d={r['dim']:3d} | 전체선택 {r['full']:.3f} (clip {r['full_clip']:.3f}) | 폴드 밖 {r['oof']:.3f} +-{r['oof_sd']:.3f} (clip {r['oof_clip']:.3f}) | 오경보정상11개 정상5-NN순도 전체 {r['fa_full']:.3f} 폴드밖 {r['fa_oof']:.3f} (전체정상 {r['nn_full']:.3f})")

# ---------------- t-SNE (라벨은 색칠만) ----------------
def tsne_best(A, perp, nseed=3):
    best = None
    for s in range(nseed):
        t = TSNE(2, perplexity=perp, init="pca", random_state=s, learning_rate="auto"); Em = t.fit_transform(A)
        if best is None or t.kl_divergence_ < best[0]: best = (t.kl_divergence_, Em)
    return best[1]
def pur2d(Em):
    nb = NearestNeighbors(n_neighbors=6).fit(Em[:ntr]).kneighbors(Em[fault_idx], return_distance=False)[:, 1:]
    return float(y[nb].mean())
def pur2d_fa(Em):
    nb = NearestNeighbors(n_neighbors=6).fit(Em[:ntr]).kneighbors(Em[fa_idx], return_distance=False)[:, 1:]
    return float(1 - y[nb].mean())
NOR, ABN, PUR = "#2a78d6", "#eb6834", "#7B1FA2"
PERPS = (5, 15, 30)
P("\n== 3b) t-SNE 2차원 배치의 고장 5-NN 순도 (train 150 + val 40 함께 임베딩, 입력 = 로버스트 z를 ±10으로 자름, perplexity별) ==")
tsne_pur = {}; tsne_fa = {}
fig_all, axs_all = plt.subplots(len(SPACES), len(PERPS), figsize=(4.2 * len(PERPS), 3.8 * len(SPACES)))
for si, sp in enumerate(SPACES):
    M = build(sp, None); Z = np.clip(standardize(M, idx_all), -10, 10); tsne_pur[sp] = {}; tsne_fa[sp] = {}
    for pj, perp in enumerate(PERPS):
        Em = tsne_best(Z, perp); pr = pur2d(Em); tsne_pur[sp][perp] = pr; tsne_fa[sp][perp] = pur2d_fa(Em)
        a = axs_all[si, pj]
        a.scatter(*Em[:ntr][nor].T, s=14, color=NOR, alpha=.5, lw=0); a.scatter(*Em[:ntr][fault].T, s=26, color=ABN, lw=0); a.scatter(*Em[ntr:].T, s=26, color=PUR, marker="^", lw=0, alpha=.8)
        a.set_xticks([]); a.set_yticks([]); a.set_title(f"{sp} | perp {perp} | 5-NN 순도 {pr:.2f}", fontsize=8)
    P(f"  {sp:28s} " + " | ".join(f"perp{p}: {tsne_pur[sp][p]:.3f}" for p in PERPS) + f" | 평균 {np.mean(list(tsne_pur[sp].values())):.3f} | 오경보정상 정상5-NN순도 평균 {np.mean(list(tsne_fa[sp].values())):.3f}")
fig_all.suptitle("후보 피처 공간 t-SNE (파랑 정상 · 주황 고장 · 보라 val, 라벨은 색칠에만; ring 경계는 150 전체로 고름 = 낙관적). 숫자는 폴드 밖 표 참고", x=.01, ha="left", fontsize=10)
fig_all.tight_layout(rect=(0, 0, 1, .985)); fig_all.savefig(os.path.join(FIG, "tsne_후보공간_F5.png"), dpi=70); plt.close(fig_all)

# ---------------- 4) 피처별 진단 (ring7: 경계는 전체 150 선택값) ----------------
P("\n== 4) ring 피처 진단 (중간 창 [%.0f,%.0f), b_hi=%.0f). val_out = val 중 정상 5-95%% 밖 비율(기대 .10), shift = (val 중앙-정상 중앙)/정상 sd ==" % (WA_FULL, WB_FULL, BHI_FULL))
P("feature | 고장 AUC(150) | 정상 p5/med/p95 | val_med | val_out | shift_sd | ext150 최대|rho|(열) | A울림2열 최대|rho| | new14 최대|rho|")
diag = []
for j, c in enumerate(RING_COLS):
    v = RING_FULL[:, j]; vn = v[:ntr][nor]; vv = v[ntr:]
    p5, med, p95 = np.percentile(vn, [5, 50, 95]); sd = vn.std(ddof=1) + 1e-12
    auc = roc_auc_score(y, v[:ntr]) if np.ptp(v[:ntr]) > 0 else 0.5
    rc = max(((abs(spearmanr(v[:ntr], E_all[:ntr, k]).correlation), ecols[k]) for k in range(E_all.shape[1]) if np.ptp(E_all[:ntr, k]) > 0 and np.ptp(v[:ntr]) > 0), default=(0, ""))
    ra = max((abs(spearmanr(v[:ntr], AOLD[:ntr, k]).correlation) for k in range(2)), default=0) if np.ptp(v[:ntr]) > 0 else 0
    rn = max(((abs(spearmanr(v[:ntr], NEW16[:ntr, k]).correlation), FINAL16[k]) for k in range(NEW16.shape[1]) if np.ptp(NEW16[:ntr, k]) > 0 and np.ptp(v[:ntr]) > 0), default=(0, ""))
    vout = float(np.mean((vv < p5) | (vv > p95))); sh = (np.median(vv) - med) / sd
    P(f"{c} | {auc:.3f} | {p5:.3g}/{med:.3g}/{p95:.3g} | {np.median(vv):.3g} | {vout:.3f} | {sh:+.2f} | {rc[0]:.2f} ({rc[1]}) | {ra:.2f} | {rn[0]:.2f} ({rn[1]})")
    diag.append(dict(feature=c, auc=auc, nor_p5=p5, nor_med=med, nor_p95=p95, val_med=float(np.median(vv)), val_out=vout, shift_sd=sh, ext_rho=rc[0], ext_col=rc[1], A2_rho=ra))
pd.DataFrame(diag).to_csv(os.path.join(HERE, "F5_diag.csv"), index=False, encoding="utf-8-sig")

P("\n== 겨냥 파일별 ring 값 (경계는 전체 선택, 라벨은 보고용) ==")
tg = ["abn_002", "abn_003", "abn_006", "abn_010", "abn_029", "abn_013", "abn_025", "abn_030", "abn_004", "abn_007", "abn_011", "abn_019", "abn_020", "abn_026", "abn_014", "abn_017", "abn_022", "abn_005", "abn_015", "nor_019", "nor_092", "nor_043", "nor_116", "nor_055", "nor_059", "nor_025"]
P("file | n_imp | " + " | ".join(RING_COLS))
for t in tg:
    i = files.index(t); P(f"{t} | {int(nimp[i])} | " + " | ".join(f"{RING_FULL[i, j]:.2f}" for j in range(len(RING_COLS))))
hi_flag = RING_FULL[:, RING_COLS.index("ring_hi_n")] > 0
P(f"\nring_hi_n>0 (경계 {BHI_FULL:.0f} Hz 이상 충격 있음): 고장 {int(hi_flag[:ntr][fault].sum())}/30 {[files[i][-3:] for i in np.flatnonzero(hi_flag[:ntr] & fault)]} | 정상 {int(hi_flag[:ntr][nor].sum())}/120 {[files[i][-3:] for i in np.flatnonzero(hi_flag[:ntr] & nor)]} | val {int(hi_flag[ntr:].sum())}/40 {[files[ntr + i][-3:] for i in np.flatnonzero(hi_flag[ntr:])]}")
for lab_, mask in (("고장", fault), ("정상", nor)):
    v = RING_FULL[:ntr][mask][:, RING_COLS.index("ring_hi_n")]
    P(f"  {lab_} ring_hi_n(=log1p 개수) 분포: 0: {int((v == 0).sum())}, >0: {int((v > 0).sum())}, 개수>=2: {int((v > np.log1p(1.5)).sum())}")

# ---------------- 5) quick_cv (ET 축만, 탐색용). 열·경계는 폴드 안에서 다시 고름 ----------------
class SpaceET(BaseEstimator):
    def __init__(self, space="ext150"): self.space = space
    def fit(self, Fi, yy):
        tr = Fi[:, 0].astype(int); self.M_ = build(self.space, tr)
        self.m_ = make_et(seed=0)().fit(self.M_[tr], yy); return self
    def predict_proba(self, Fi): return self.m_.predict_proba(self.M_[Fi[:, 0].astype(int)])
Fidx = np.arange(ntr)[:, None].astype(float)
P("\n== 5) quick_cv: ET(300) 입력별, 5-fold x seed 5, PR-AUC / ROC-AUC (OOF, 임계값 없음). 열 선택·경계는 폴드 안 재선택. 기준 = A_in ==")
qres = {}
for sp in SPACES:
    q = quick_cv(Fidx, y, np.arange(ntr), lambda sp=sp: SpaceET(sp), seeds=range(5))
    qres[sp] = dict(pr=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std(ddof=1)), roc=float(q["roc_auc"].mean()), roc_sd=float(q["roc_auc"].std(ddof=1)))
    worst = sorted(q["miss_rank"].items(), key=lambda kv: -kv[1])[:6]
    P(f"  {sp:28s} PR-AUC {qres[sp]['pr']:.3f} +-{qres[sp]['pr_sd']:.3f} | ROC-AUC {qres[sp]['roc']:.3f} +-{qres[sp]['roc_sd']:.3f} | 놓친 고장(순위 나쁨): {[(files[k][-3:], v) for k, v in worst]}")
json.dump(dict(pur=res_pur, tsne=tsne_pur, tsne_fa=tsne_fa, quick=qres, wa=WA_FULL, wb=WB_FULL, b_hi=BHI_FULL), open(os.path.join(HERE, "result_F5.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
P("완료")
