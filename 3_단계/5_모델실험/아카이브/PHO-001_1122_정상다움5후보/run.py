# %% [markdown]
# PHO-001 (휴대폰 세션, 11-22) "정상을 더 정상답게" 후보 5개를 하나씩 바꿔 quick_cv(PR-AUC)로 비교 — 탐색 단계, F2 없음
# - 기준 W = Two('lin'): ET(H 15열 = 최종피처 fixed 13 + H16·H17 폴드 안) 안쪽 OOF logit + 박자 z → 로지스틱 (J에서 AE · E2 · 8.3k를 뺀 것)
# - 후보 (한 번에 하나만 바꿈, 값은 실행 전 고정)
#   1 IMP_BG  : 옥타브(1–2·2–4·4–8·8–12.5 kHz)별 log10(전체 PSD / 배경 PSD) 4열 추가. 배경 = 충격(새 정의) −5~+30 ms에 걸친 2048점 프레임 제외
#               (깨끗한 프레임 < 5 → 1 kHz 위 에너지 하위 25 % 프레임). 충격 0개 = 0
#   2 NG_k4   : 정상 하위군 이상도 1열 추가. 1 kHz 위 log PSD를 250 Hz 칸 평균 → 정상만으로 강건 표준화 → KMeans(k=4) → 가장 가까운 중심까지 거리 / 그 군 중앙 거리.
#               학습 행은 안쪽 4겹 OOF(정상), 시험 행은 학습 정상 전체로. 대조: NG_k1 (하위군 없이 중심 하나)
#   3 RING_FIT: H7–H10(울림 옥타브 비율)을 fit ≥ 0.7인 충격만으로 다시 계산해 **교체** (fit 0.7 = 기존 공진 규칙 값). 해당 충격 없음 = −1
#   4 RING_IQR: 파일 안 울림 주파수 log2(f)의 사분위 범위(옥타브) 1열 추가. 충격 < 3개 = 0
#   5 AE_BG   : (별도 축) 스펙트럼 AE 단독 점수의 입력을 200 Hz 고역 8×1초 log PSD → 같은 것의 배경 프레임만 평균으로 교체. AE 단독 PR-AUC와 val AE z 이동을 비교
# - 가설: H1(유격 충격) · H2(8.3k 모드) · H3(게인 무관) — 1·4는 "충격이 에너지를 얹는 정도·울림 일관성", 2·5는 "정상 상시 진동의 다양성"
# 실행: conda run -n qe-phm python "3_단계/5_모델실험/PHO-001_1122_정상다움5후보/run.py"

# %% 설정
import os, sys, json, time, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from scipy.signal import spectrogram, butter, sosfiltfilt
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.model_selection import StratifiedKFold
from src.exp import load_train
from src.data import load_folder
from src.features2 import feat2_table, OCT, robust_z
from src.impacts import all_impacts
from src.models2 import Two
from src.evaluate import quick_cv, separation
from src.final import _fit_ae, _ae_score
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000
CACHE = os.path.join(_d, "4_outputs", "cache"); os.makedirs(CACHE, exist_ok=True)
LOG = open(os.path.join(HERE, "log.txt"), "w", encoding="utf-8")
def say(*a):
    s = " ".join(str(v) for v in a); print(s); LOG.write(s + "\n"); LOG.flush()

FJ = json.load(open(os.path.join(_d, "3_단계", "4_피처", "4-2_H피처목록", "자료", "최종피처.json"), encoding="utf-8"))
FIXED, PEAK = list(FJ["fixed"]), list(FJ["hfold"])
X, y, names = load_train(); names = [str(n) for n in names]; n = len(y); g = np.arange(n)
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val"))

# %% 충격(새 정의) · 2차 피처 (train · val), 캐시
def feats(tag, XX):
    p = os.path.join(CACHE, f"pho001_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p, allow_pickle=True); return z["T"].item(), z["L"], z["f"], list(z["imps"])
    imps = all_impacts(XX); T, L, f = feat2_table(XX, imps)
    keep = [{k: im[k] for k in ("t", "height_x", "rings")} for im in imps]
    np.savez(p, T=np.array(T.to_dict("list"), dtype=object), L=L, f=f, imps=np.array(keep, dtype=object))
    return T.to_dict("list"), L, f, keep
t0 = time.time()
Td, L, fL, imps = feats("train", X); Tvd, Lv, _, impsv = feats("val", Xv)
col = lambda D, c: np.asarray(D[c], float)
H = np.column_stack([col(Td, c) for c in FIXED]); S1 = col(Td, "L1_S"); df = float(fL[1] - fL[0])
say(f"피처 준비 {time.time() - t0:.0f}s · 충격 수 중앙 정상 {np.median([len(i['t']) for i, yy in zip(imps, y) if yy == 0]):.0f} 고장 {np.median([len(i['t']) for i, yy in zip(imps, y) if yy == 1]):.0f}")

# %% 후보 피처 (라벨 없이, 파일마다 독립)
def mask_ok(tc, tim, pre=0.005, post=0.030, half=1024 / FS):
    ok = np.ones(len(tc), bool)
    for ti in tim:
        ok &= ~((tc + half > ti - pre) & (tc - half < ti + post))
    return ok

def imp_bg(x, tim):
    f, t, P = spectrogram(x - x.mean(), fs=FS, nperseg=2048, noverlap=1024, mode="psd")
    if len(tim) == 0:
        return np.zeros(4)
    ok = mask_ok(t, tim)
    if ok.sum() < 5:
        e = P[f >= 1000].sum(0); ok = e <= np.percentile(e, 25)
    out = []
    for lo, hi in OCT:
        m = (f >= lo) & (f < hi)
        out.append(np.log10(P[m].sum(0).mean() / (P[m][:, ok].sum(0).mean() + 1e-30)))
    return np.array(out)

def ring_fit(r, thr=0.7):
    r = r[r[:, 2] >= thr] if len(r) else r
    if not len(r):
        return np.full(4, -1.0)
    b = np.digitize(r[:, 0], [2000, 4000, 8000])
    return np.bincount(b, minlength=4)[:4] / len(r)

def ring_iqr(r):
    if len(r) < 3:
        return 0.0
    q = np.percentile(np.log2(r[:, 0]), [25, 75]); return float(q[1] - q[0])

F1 = np.array([imp_bg(x, im["t"]) for x, im in zip(X, imps)])
F3 = np.array([ring_fit(im["rings"]) for im in imps])
F4 = np.array([[ring_iqr(im["rings"])] for im in imps])
say(f"후보 피처 계산 {time.time() - t0:.0f}s")

# %% 모델: 기준 W와 열 추가 · 교체 · 하위군 이상도
def mat(Hb):
    return np.column_stack([Hb, S1, L])

def mk(names_fixed, nf):
    return lambda: Two("lin", (), fixed=tuple(names_fixed), nf=nf, df=df, peak=tuple(PEAK))

def coarse(Lb):      # 1 kHz 위 log PSD → 250 Hz 칸 평균
    edges = np.arange(1000, 12500 + 1, 250); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (fL >= lo) & (fL < hi); out.append(Lb[:, m].mean(1))
    return np.column_stack(out)

class NormGroup:
    """정상만으로: 강건 표준화 → KMeans(k) → score = min_k ||b − c_k|| / (그 군 중앙 거리)"""
    def __init__(self, k): self.k = k
    def fit(self, B):
        self.med_ = np.median(B, 0); self.sc_ = 1.4826 * np.median(np.abs(B - self.med_), 0) + 1e-6
        Z = (B - self.med_) / self.sc_
        self.km_ = KMeans(self.k, n_init=10, random_state=0).fit(Z)
        d = np.linalg.norm(Z - self.km_.cluster_centers_[self.km_.labels_], axis=1)
        self.s_ = np.array([np.median(d[self.km_.labels_ == j]) if (self.km_.labels_ == j).any() else 1.0 for j in range(self.k)]) + 1e-6
        return self
    def score(self, B):
        Z = (B - self.med_) / self.sc_
        D = np.linalg.norm(Z[:, None, :] - self.km_.cluster_centers_[None], axis=2) / self.s_[None]
        return np.log(D.min(1) + 1e-9)

class TwoNG(Two):
    """Two + 정상 하위군 이상도를 H 마지막 열로. 학습 행 값은 안쪽 4겹 OOF(정상), 고장은 학습 정상 전체 모델"""
    def __init__(self, k=4, comb="lin", experts=(), fixed=(), nf=0, df=1.0, peak=(), q_exp=99.0, seed=0, scale_thr=1.0):
        super().__init__(comb, experts, fixed, nf, df, peak, q_exp, seed, scale_thr); self.k = k
    def _parts(self, F):
        Hh, pk, S = super()._parts(F)
        gcol = self._g_fit if getattr(self, "_g_fit", None) is not None else self.ng_.score(coarse(np.asarray(F, float)[:, self.nf + 1:]))
        return np.column_stack([Hh, gcol]), pk, S
    def fit(self, F, y):
        F, y = np.asarray(F, float), np.asarray(y); B = coarse(F[:, self.nf + 1:])
        nor = np.flatnonzero(y == 0); self.ng_ = NormGroup(self.k).fit(B[nor])
        gtr = self.ng_.score(B)
        for a, b in StratifiedKFold(4, shuffle=True, random_state=0).split(nor, np.zeros(len(nor))):
            gtr[nor[b]] = NormGroup(self.k).fit(B[nor[a]]).score(B[nor[b]])
        self._g_fit = gtr
        super().fit(F, y)
        self._g_fit = None
        return self

def mkNG(k):
    return lambda: TwoNG(k, "lin", (), fixed=tuple(FIXED), nf=len(FIXED), df=df, peak=tuple(PEAK))

# %% quick_cv: 기준 + 후보 1–4 (하나씩)
FOCUS_A = ["abn_002", "abn_014", "abn_017", "abn_022"]; FOCUS_N = ["nor_116", "nor_065", "nor_059", "nor_025"]
R = {}
def run(tag, F, make, note):
    t1 = time.time(); q = quick_cv(F, y, g, make)
    oof = q["oof"]; sp = separation(oof, y, names)
    ab = oof[:, y == 1]
    nrank = {k: float(np.mean([(ab[s] <= oof[s, names.index(k)]).mean() for s in range(len(oof))])) for k in FOCUS_N}
    arank = {k: q["miss_rank"][names.index(k)] for k in FOCUS_A}
    R[tag] = dict(note=note, pr_auc=float(q["pr_auc"].mean()), pr_sd=float(q["pr_auc"].std()), roc_auc=float(q["roc_auc"].mean()),
                  risk_n=sp["risk_n"], sure_a=sp["sure_a"], sure_n=sp["sure_n"], margin=sp["margin"],
                  fault_rank=arank, normal_faultlike=nrank)
    say(f"{tag:10s} PR-AUC {R[tag]['pr_auc']:.3f} ± {R[tag]['pr_sd']:.3f} · ROC {R[tag]['roc_auc']:.3f} · 위험 정상 {sp['risk_n']:.1f} · "
        f"확실 고장 {sp['sure_a']} · 확실 정상 {sp['sure_n']} | 고장 순위(0=잘 잡힘) " + " ".join(f"{k[-3:]} {v:.2f}" for k, v in arank.items())
        + " | 정상의 고장다움(1=고장보다 높음) " + " ".join(f"{k[-3:]} {v:.2f}" for k, v in nrank.items()) + f" ({time.time() - t1:.0f}s)")

nf = len(FIXED)
run("기준 W", mat(H), mk(FIXED, nf), "ET(H15) + 박자 → 로지스틱")
n1 = [f"P1_impbg_{a}" for a in ("1-2k", "2-4k", "4-8k", "8-12.5k")]
run("1 IMP_BG", mat(np.column_stack([H, F1])), mk(FIXED + n1, nf + 4), "+ 충격/배경 PSD 비 4열")
run("2 NG_k4", mat(H), mkNG(4), "+ 정상 하위군(k=4) 이상도")
run("2c NG_k1", mat(H), mkNG(1), "대조: 하위군 없이 중심 하나")
H3 = H.copy()
for j, c in enumerate(["H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k"]):
    H3[:, FIXED.index(c)] = F3[:, j]
run("3 RING_FIT", mat(H3), mk(FIXED, nf), "H7–H10을 fit ≥ 0.7 충격만으로 교체")
run("4 RING_IQR", mat(np.column_stack([H, F4])), mk(FIXED + ["P4_ring_iqr"], nf + 1), "+ 울림 주파수 사분위 범위(옥타브)")

# %% 5: AE 입력 교체 (AE 단독)
def seg_psd(XX, impsX, bg):
    hp = butter(4, 200, btype="highpass", fs=FS, output="sos")
    Hh = sosfiltfilt(hp, XX - XX.mean(1, keepdims=True), axis=-1)
    starts = np.linspace(0, XX.shape[1] - FS, 8).astype(int); out = []
    for x, im in zip(Hh, impsX):
        segs = []
        for s0 in starts:
            f, t, P = spectrogram(x[s0:s0 + FS], fs=FS, nperseg=2048, noverlap=1024, mode="psd")
            if bg:
                ok = mask_ok(t + s0 / FS, im["t"])
                p = P[:, ok].mean(1) if ok.sum() >= 5 else np.median(P, 1)
            else:
                p = P.mean(1)
            segs.append(np.log10(p + 1e-20))
        out.append(segs)
    return np.array(out)

ARR = {}
for tag, bg in (("std", False), ("bg", True)):
    p = os.path.join(CACHE, f"pho001_ae_{tag}.npz")
    if os.path.exists(p):
        z = np.load(p); ARR[tag], ARR[tag + "_v"] = z["tr"], z["v"]
    else:
        ARR[tag], ARR[tag + "_v"] = seg_psd(X, imps, bg), seg_psd(Xv, impsv, bg)
        np.savez(p, tr=ARR[tag], v=ARR[tag + "_v"])

class AEM(BaseEstimator):
    def __init__(self, which="std"): self.which = which
    def fit(self, F, yy):
        idx = F[:, 0].astype(int); self.ae_ = _fit_ae(ARR[self.which][idx[np.asarray(yy) == 0]]); return self
    def decision_function(self, F):
        return _ae_score(self.ae_, ARR[self.which][F[:, 0].astype(int)])

for tag in ("std", "bg"):
    run(f"5 AE_{tag}", g[:, None].astype(float), (lambda w=tag: AEM(w)), "AE 단독 (기준 입력)" if tag == "std" else "AE 단독 (배경 프레임만)")
    oofn = R[f"5 AE_{tag}"]
    ae = AEM(tag).fit(g[:, None].astype(float), y)
    q = quick_cv(g[:, None].astype(float), y, g, (lambda w=tag: AEM(w)), seeds=[0])["oof"][0]
    zf = robust_z(q[y == 0]); zv = zf(_ae_score(ae.ae_, ARR[tag + "_v"])); za = zf(q[y == 1])
    oofn["val_ae_z_median"] = float(np.median(zv)); oofn["val_ae_z_ge1"] = float((zv >= 1).mean()); oofn["train_fault_ae_z_median"] = float(np.median(za))
    say(f"   val AE z 중앙 {np.median(zv):+.2f} · z ≥ 1 비율 {(zv >= 1).mean():.2f} · (참고) 학습 고장 AE z 중앙 {np.median(za):+.2f}")

json.dump(dict(rule="quick_cv 5-fold × seed 3, PR-AUC (임계값 없음). 하나씩만 바꿈", results=R), open(os.path.join(HERE, "result.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
say(f"완료 {time.time() - t0:.0f}s")
