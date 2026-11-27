# %% NEXP-02 ① 검증 정합성 점검(코드 레벨) ② 기준선 quick_cv ③ 구성 A·B·C quick_cv
# 평가는 src.evaluate.quick_cv 그대로(5-fold StratifiedGroupKFold, seed 5, PR-AUC). ExcludeTrain 미사용 = Train 150 전체.
# 그룹 = 097·101·120 한 그룹, 나머지 파일당 1그룹. AE는 폴드 학습 정상으로만, 학습 행의 AE 점수는 안쪽 4-fold OOF.
import glob, hashlib, os, sys, warnings
import numpy as np, pandas as pd
from scipy.signal import welch, butter, sosfiltfilt
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedGroupKFold
warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")): ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
from src.evaluate import quick_cv
FS = 25000; FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
SEEDS = list(range(5)); L = []

# %% 데이터: Train 150 (정상 120 → 고장 30 순서), 이벤트 표(NEXP-01 캐시)
fl = []
for g in ("normal", "abnormal"): fl += sorted(glob.glob(os.path.join(ROOT, "2_data", "TeamProject_train_변경", g, "*.csv")))
names = [os.path.basename(f)[:-4] for f in fl]; y = np.array([0] * 120 + [1] * 30)
X = np.stack([np.loadtxt(f) for f in fl]); assert len(names) == 150 and names[0] == "nor_001" and names[-1] == "abn_030"
groups = np.array([0 if n in ("nor_097", "nor_101", "nor_120") else i + 1 for i, n in enumerate(names)])
groups[[names.index(n) for n in ("nor_097", "nor_101", "nor_120")]] = 0
E = pd.read_csv(os.path.join(FIG, "events_190.csv")); EV = {f: g for f, g in E.groupby("file")}
def ev_feats(f):
    e = EV.get(f)
    if e is None: return dict(low=0, trans=0, body=0, high=0, total=0, off_max=0.0)
    c = lambda a, b: e[(e.f >= a) & (e.f < b)]
    low, tr, body, hi = c(1600, 4500), c(4500, 5500), c(5500, 8000), c(8000, 11000)
    off = pd.concat([low, hi]); amp = float((off.h / (off.h / off.x)).max()) if len(off) else 0.0
    return dict(low=len(low), trans=len(tr), body=len(body), high=len(hi), total=len(e), off_max=amp)
EF = pd.DataFrame([ev_feats(n) for n in names]); EF["ratio_off"] = (EF.low + EF.high) / (EF.total + 1)
# 스펙트럼: AE 입력 LP (src/final.py signal_tables와 같은 전처리) + 상대 대역 로그 에너지 2열
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
starts = np.linspace(0, X.shape[1] - FS, 8).astype(int)
LP = np.log10(welch(np.stack([H[:, s:s + FS] for s in starts], 1), fs=FS, nperseg=2048, axis=-1)[1] + 1e-20)
f, P = welch(Xc, fs=FS, nperseg=4096); en = lambda a, b: P[:, (f >= a) & (f < b)].sum(1)
BAND = np.column_stack([np.log10(en(1600, 5500) / en(1000, 11000)), np.log10(en(8000, 11000) / en(1000, 11000))])

# %% 모델: 열 = [이벤트 열…, (대역 열…), 파일 번호]. AE 점수는 fit 안에서만 학습
AE_CACHE = {}
def fit_ae(rows, seed=0):
    key = ("m", tuple(rows), seed)
    if key not in AE_CACHE:
        S = LP[list(rows)].reshape(-1, LP.shape[-1]); mu, sd = S.mean(0), S.std(0) + 1e-8; Z = (S - mu) / sd
        m = MLPRegressor(hidden_layer_sizes=(128, 16, 128), activation="relu", solver="adam", learning_rate_init=1e-3, max_iter=60, alpha=0.0, random_state=seed).fit(Z, Z)
        AE_CACHE[key] = (mu, sd, m)
    return AE_CACHE[key]
def ae_score(ae, rows):
    mu, sd, m = ae; Z = (LP[rows] - mu) / sd; R = m.predict(Z.reshape(-1, Z.shape[-1])).reshape(Z.shape)
    return np.log10(((R - Z) ** 2).astype(np.float32).mean((1, 2)))
FIT_LOG = {}   # 구성 이름 → fit 호출 때 학습 행 해시 · 개수 · 이상치 정상 포함 수
class Model:
    def __init__(self, tag, use_ae, depth=3, leaf=5): self.tag, self.use_ae, self.depth, self.leaf = tag, use_ae, depth, leaf
    def fit(self, F, yy):
        F = np.asarray(F); rows = F[:, -1].astype(int); self.rows_ = rows
        FIT_LOG.setdefault(self.tag, []).append((hashlib.md5(np.sort(rows).tobytes()).hexdigest()[:8], len(rows),
                                                 int(np.isin(rows, [names.index(n) for n in ("nor_059", "nor_065", "nor_116")]).sum()),
                                                 len(set(groups[rows]) & {0}) and int((groups[rows] == 0).sum())))
        Xf = F[:, :-1]
        if self.use_ae:
            oof = np.zeros(len(rows))
            for a, b in StratifiedGroupKFold(4, shuffle=True, random_state=0).split(rows, yy, groups[rows]):
                oof[b] = ae_score(fit_ae(tuple(sorted(rows[a][yy[a] == 0]))), rows[b])
            self.ae_ = fit_ae(tuple(sorted(rows[yy == 0]))); Xf = np.column_stack([Xf, oof])
        self.m_ = ExtraTreesClassifier(300, max_depth=self.depth, min_samples_leaf=self.leaf, class_weight="balanced", random_state=0, n_jobs=-1).fit(Xf, yy); return self
    def decision_function(self, F):
        F = np.asarray(F); Xf = F[:, :-1]
        if self.use_ae: Xf = np.column_stack([Xf, ae_score(self.ae_, F[:, -1].astype(int))])
        return self.m_.predict_proba(Xf)[:, 1]
idx = np.arange(150)
CONF = {
    "기준선(옛 5개: low·high·ratio·total·off_max)": (EF[["low", "high", "ratio_off", "total", "off_max"]].values, False),
    "A (low·trans·high·ratio·off_max)": (EF[["low", "trans", "high", "ratio_off", "off_max"]].values, False),
    "A+body (A + 5.5–8 kHz 이벤트 수)": (EF[["low", "trans", "high", "ratio_off", "off_max", "body"]].values, False),
    "B (A + AE 점수)": (EF[["low", "trans", "high", "ratio_off", "off_max"]].values, True),
    "C (B + 골짜기 상대 대역 2열)": (np.column_stack([EF[["low", "trans", "high", "ratio_off", "off_max"]].values, BAND]), True),
}
# %% ① 검증 정합성 점검
L.append("① 검증 파이프라인 정합성 (src.evaluate.quick_cv, 5-fold StratifiedGroupKFold, seed 0–4)")
dc = [names.index(n) for n in ("nor_097", "nor_101", "nor_120")]
split_ok, dc_ok, cover_ok = True, True, True
for s in SEEDS:
    sp = list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx, y, groups))
    for tr, te in sp:
        k = len(set(dc) & set(te)); dc_ok &= (k in (0, 3)); split_ok &= (len(set(tr) & set(te)) == 0)
    cover_ok &= (sorted(np.concatenate([te for _, te in sp])) == list(range(150)))
    sp2 = list(StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx, y, groups))
    split_ok &= all(np.array_equal(a[1], b[1]) for a, b in zip(sp, sp2))
L.append(f"(b) DC 그룹(097·101·120)이 한 폴드에 함께 들어감(모든 seed·폴드): {dc_ok} | 학습·시험 겹침 없음 + 같은 seed 재호출 시 동일 분할: {split_ok} | 모든 파일이 정확히 한 번 시험: {cover_ok}")
# 그룹을 빼고 분할하면 DC 파일이 갈라지는지(대조)
sep = 0
for s in SEEDS:
    for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx, y, idx):
        sep += (len(set(dc) & set(te)) not in (0, 3))
L.append(f"   대조: 파일마다 1그룹(옛 방식)으로 나누면 DC 3개가 갈라진 폴드 수 = {sep}/25")

# %% ② ③ quick_cv
res = {}
for nm, (Fe, ae) in CONF.items():
    F = np.column_stack([Fe, idx]); tag = nm
    r = quick_cv(F, y, groups, lambda t=tag, a=ae: Model(t, a), n_splits=5, seeds=SEEDS); res[nm] = r
tags = list(CONF)
h0 = [(h, n) for h, n, _, _ in FIT_LOG[tags[0]]]
same = all([(h, n) for h, n, _, _ in FIT_LOG[t]] == h0 for t in tags)
L.append(f"(c) 5개 구성이 같은 폴드 분할 · 같은 seed로 비교됨(모든 fit 호출의 학습 행 해시 일치, fit 호출 수 {len(h0)}): {same}")
L.append(f"(a) ExcludeTrain 미사용: 폴드별 학습 행 수 = {sorted(set(n for _, n, _, _ in FIT_LOG[tags[0]]))}, 학습 폴드에 이상치 정상(059·065·116) 포함된 폴드 수(합) = {sum(o for _, _, o, _ in FIT_LOG[tags[0]])} (폴드당 3개 중 시험으로 빠진 것 제외), OOF 점수가 있는 파일 = {res[tags[0]]['oof'].shape[1]}/150")
L.append(f"    DC 그룹 학습 행 수 분포(폴드): {sorted(set(d for _, _, _, d in FIT_LOG[tags[0]]))} (0 = 시험 폴드, 3 = 학습)")
L.append("\n② ③ quick_cv (5-fold OOF, seed 5회 평균 ± sd). ET 300, max_depth 3, leaf 5, balanced (NEXP-01에서 가장 높았던 설정, 고정)")
def risk(o, yy):
    thr = np.percentile(o[yy == 1], 10); return int((o[yy == 0] > thr).sum())
for nm in tags:
    r = res[nm]; rk = np.mean([risk(o, y) for o in r["oof"]])
    L.append(f"{nm:46s} PR-AUC {r['pr_auc'].mean():.3f} ± {r['pr_auc'].std():.3f} | ROC-AUC {r['roc_auc'].mean():.3f} ± {r['roc_auc'].std():.3f} | 위험 정상(고장 하위 10 % 선보다 높은 정상) {rk:.1f}개")
# 오류 분석: 파일별 OOF 점수 순위 백분위(150개 중, seed 평균). 높을수록 고장 쪽
def prank(o): return (o[None, :] <= o[:, None]).mean(1)
focus_a = ["abn_014", "abn_023", "abn_028", "abn_017"]
focus_n = ["nor_054", "nor_059", "nor_065", "nor_116", "nor_002", "nor_003", "nor_007", "nor_030", "nor_037", "nor_066", "nor_074", "nor_082", "nor_098", "nor_104", "nor_106"]
L.append("\n오류 분석: 파일별 OOF 점수의 순위 백분위(150개 중, 5 seed 평균, 1.00 = 가장 고장스러움). 고장은 높을수록, 정상은 낮을수록 좋다")
L.append(f"{'파일':8s} " + " | ".join(f"{t.split(' ')[0]:>8s}" for t in tags))
for f in focus_a + focus_n:
    i = names.index(f); L.append(f"{f:8s} " + " | ".join(f"{np.mean([prank(o)[i] for o in res[t]['oof']]):8.2f}" for t in tags))
L.append("고장 30개의 순위 백분위 중앙값 / 정상 120개의 중앙값: " + " | ".join(f"{t.split(' ')[0]} {np.median(np.mean([prank(o)[y==1] for o in res[t]['oof']],0)):.2f}/{np.median(np.mean([prank(o)[y==0] for o in res[t]['oof']],0)):.2f}" for t in tags))
L.append("\n 각 구성에서 정상 중 점수가 '고장 하위 25 %선(seed 평균)'보다 높은 파일 (자주 틀리는 오경보 후보, 5 seed 중 3번 이상)")
for t in tags:
    cnt = np.zeros(150)
    for o in res[t]["oof"]: cnt += (y == 0) & (o > np.percentile(o[y == 1], 25))
    L.append(f"{t.split(' ')[0]:8s} " + ", ".join(names[i] for i in np.where(cnt >= 3)[0]))
cnt = np.zeros(150)
open(os.path.join(FIG, "수치.txt"), "w", encoding="utf-8").write("\n".join(L)); print("done")
