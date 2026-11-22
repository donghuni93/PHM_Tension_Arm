# %% [markdown]
# 4-9 정상 증거 피처 (11-22) — "정상을 뽑는 피처". 정의는 src/normal_feats.py (실행 전 고정), 평가는 라벨 사용(탐색, 중첩 CV 아님)
# 실행: conda run -n qe-phm python run.py  (출력 = log.txt, claims.md, eval_*.csv, figs/)

# %% 설정
import os, sys, json, pickle, warnings
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
HERE = _d
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
ROOT = _d; sys.path.insert(0, ROOT)
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from src.exp import load_train, make_et
from src.data import load_folder
from src.features2 import cached_feat2, PeakRef
from src.normal_feats import file_work
from src.evaluate import quick_cv
warnings.filterwarnings("ignore")
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)
LOGP = os.path.join(HERE, "log.txt"); open(LOGP, "w", encoding="utf-8").close()
def log(*a):
    s = " ".join(str(v) for v in a); open(LOGP, "a", encoding="utf-8").write(s + "\n")
X, y, names = load_train(); names = [str(n) for n in names]; y = np.asarray(y)
Xv, nv = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val")); nv = [str(n) for n in nv]
n, nvl = len(y), len(nv); nor = y == 0
FJ = json.load(open(os.path.join(ROOT, "단계", "4_피처", "4-7_1121_2차피처점검", "최종피처.json"), encoding="utf-8"))
FIXED = list(FJ["fixed"])
T, L, fr = cached_feat2("train", X); Tv, Lv, _ = cached_feat2("val", Xv)
FP_J = ["nor_116", "nor_065", "nor_059", "nor_037", "nor_025", "nor_095", "nor_106"]
FP_PASTE = ["nor_027", "nor_039", "nor_110"]; FP_ALL = FP_J + FP_PASTE
ix = {k: i for i, k in enumerate(names)}

# %% 1. 파일별 원값 (캐시)
CP = os.path.join(HERE, "_raw.pkl")
if os.path.exists(CP):
    rows, arrs = pickle.load(open(CP, "rb"))
else:
    res = Parallel(n_jobs=4)(delayed(file_work)(x) for x in np.vstack([X, Xv]))
    rows, arrs = [r[0] for r in res], [r[1] for r in res]
    pickle.dump((rows, arrs), open(CP, "wb"))
R = pd.DataFrame(rows, index=names + nv)
nimp = R.n_imp.values
tot = dict(normal=int(nimp[:n][nor].sum()), val=int(nimp[n:].sum()), fault=int(nimp[:n][~nor].sum()))
log("충격 합계", tot); assert tot == dict(normal=854, val=247, fault=402), tot
Rt, Rv = R.iloc[:n], R.iloc[n:]

# %% 2. N4: 15열 중 정상 5–95 % 밖 개수 (정상 기준은 학습 정상에서만)
N4COLS = FIXED + ["log_hp_rms", "log_raw200"]        # 사전 선언 15열 (뷰어 피처는 train 150만 있어 val 불가 → 제외)
CT = np.hstack([T[FIXED].values, R[["log_hp_rms", "log_raw200"]].values[:n]])
CV_ = np.hstack([Tv[FIXED].values, R[["log_hp_rms", "log_raw200"]].values[n:]])
def n4_fn(Cref, Ctgt, loo=None):
    """Cref (정상 m, 15) 의 5/95 % 밖 개수. loo[i] = Ctgt[i] 가 Cref의 몇 번째 행이면 그 행을 빼고 계산 (없으면 -1)"""
    lo, hi = np.percentile(Cref, [5, 95], axis=0); out = ((Ctgt < lo) | (Ctgt > hi)).sum(1).astype(float)
    if loo is not None:
        for i, j in enumerate(loo):
            if j >= 0:
                r = np.delete(Cref, j, 0); l_, h_ = np.percentile(r, [5, 95], axis=0); out[i] = ((Ctgt[i] < l_) | (Ctgt[i] > h_)).sum()
    return out
def n4_split(tr, te):
    """학습 인덱스 tr (train 150 중) 의 정상으로 기준 → (tr 점수, te 점수). tr 안 정상은 LOO"""
    trn = tr[nor[tr]]; Cref = CT[trn]; pos = {int(j): k for k, j in enumerate(trn)}
    s_tr = n4_fn(Cref, CT[tr], [pos.get(int(j), -1) for j in tr])
    return s_tr, n4_fn(Cref, CT[te])
all_idx = np.arange(n)
N4_tr = np.empty(n); N4_tr[:] = n4_fn(CT[nor], CT, [({int(j): k for k, j in enumerate(np.flatnonzero(nor))}).get(i, -1) for i in range(n)])
N4_va = n4_fn(CT[nor], CV_)

# %% 3. 피처 표 (방향: s = 정상일수록 큼)
F = pd.DataFrame(index=R.index)
F["N1_reb"] = R.N1_reb; F["N2_ret_ms"] = R.N2_ret_ms; F["N3_ifrac"] = R.N3_ifrac; F["N3_crest"] = R.N3_crest
F["N4_nout"] = np.r_[N4_tr, N4_va]; F["N5_edge"] = R.N5_edge; F["N6_calm"] = R.N6_calm
F["ref_n_ev"] = R.n_ev; F["ref_H1_nimp"] = R.n_imp; F["ref_H14_share"] = np.r_[T["H14_share_4-8k"].values, Tv["H14_share_4-8k"].values]
F["dom"] = R.dom
F["y"] = np.r_[y, np.full(nvl, np.nan)]
F.to_csv(os.path.join(HERE, "normal_features.csv"), encoding="utf-8-sig")
SIGN = {"N1_reb": -1, "N2_ret_ms": -1, "N3_ifrac": -1, "N3_crest": -1, "N4_nout": -1, "N5_edge": 1, "N6_calm": 1,
        "ref_n_ev": -1, "ref_H1_nimp": -1, "ref_H14_share": 1}
NEWF = ["N1_reb", "N2_ret_ms", "N3_ifrac", "N3_crest", "N4_nout", "N5_edge", "N6_calm"]
S_all = {k: SIGN[k] * F[k].values for k in SIGN}       # (150+40)

# %% A. 붙여 넣은 주장 검증 (실측 수치 덤프)
def pct_nor(col, f):     # 정상 120개 중 f 파일보다 값이 작은(또는 같은) 비율 %
    v = Rt[col].values[nor]; return 100 * (v <= Rt.loc[f, col]).mean()
lines = []
def claim(s): lines.append(s); log(s)
for f in ["nor_116", "nor_095", "nor_039", "nor_110", "nor_027", "nor_037", "nor_059", "nor_065"]:
    i = ix[f]; a = arrs[i]; r = Rt.loc[f]
    claim(f"[{f}] n_imp={int(r.n_imp)} n_ev={int(r.n_ev)} 충격시점(s)={np.round(a['t'], 3).tolist()} 높이g={np.round(a['hg'], 2).tolist()} 높이x={np.round(a['hx'], 1).tolist()} 울림f(Hz)={np.round(a['f'], 0).tolist()}")
    claim(f"      hp최대|x|={r.hp_peak:.2f}g raw최대|x-평균|={r.raw_peak:.2f}g 최대충격 포락선높이={r.max_hg:.2f}g | N2 바탕2배복귀={r.N2_ret_ms:.1f}ms 반감={r.N2_half_ms:.1f}ms | N1 후속비={r.N1_reb:.2f} 후속개수={int(r.N1_nreb)} | 나머지 구간 N6 calm={r.N6_calm:.2f}")
    claim(f"      L1_S={T.loc[i, 'L1_S']:.3f} L1_r={T.loc[i, 'L1_r']:.3f} | 8–11 kHz 울림 충격 수={int(r.n_ring811)} H15_share(8–12.5k)={T.loc[i, 'H15_share_8-12.5k']:.2f} (정상 중앙 {np.median(T['H15_share_8-12.5k'].values[nor]):.2f}) | env중앙={r.env_med:.4f} (정상 백분위 {pct_nor('env_med', f):.0f}%) H18_hfl={T.loc[i, 'H18_hfl']:.2f} (정상 백분위 {100 * (T['H18_hfl'].values[nor] <= T.loc[i, 'H18_hfl']).mean():.0f}%) crest={r.N3_crest:.1f} (정상 백분위 {pct_nor('N3_crest', f):.0f}%) hp_rms 정상 백분위 {pct_nor('hp_rms', f):.0f}%")
    claim(f"      N4(15열 중 정상5–95% 밖)={N4_tr[i]:.0f} (정상 중앙 {np.median(N4_tr[nor]):.0f}, 0개인 정상 {int((N4_tr[nor] == 0).sum())}/120) | 첫충격 {r.first_t:.3f}s 마지막 {r.last_t:.3f}s 끝 100ms 몫 N5={r.N5_edge:.2f}")
# 116 상세: 최대 충격 이후 hp 포락선
from scipy.signal import hilbert
from src.impacts import highpass
x116 = highpass(X[ix["nor_116"]]); p = int(arrs[ix["nor_116"]]["t"][np.argmax(arrs[ix["nor_116"]]["hg"])] * 25000)
rr = lambda a, b: float(np.sqrt(np.mean(x116[max(0, a):b] ** 2)))
claim(f"[nor_116 상세] 최대 충격 시점 {p / 25000:.3f}s, hp RMS: 충격 전 0.5s={rr(p - 12500, p - 250):.4f} 충격 후 0–20ms={rr(p, p + 500):.4f} 20–100ms={rr(p + 500, p + 2500):.4f} 100ms 이후={rr(p + 2500, p + 47500):.4f} g | 정상 hp_rms 중앙 {Rt.hp_rms.values[nor].mean():.4f}")
# 095 상세: 앞부분 조용 구간
x95 = highpass(X[ix["nor_095"]]); t95 = arrs[ix["nor_095"]]["t"]
r95 = lambda a, b: float(np.sqrt(np.mean(x95[int(a * 25000):int(b * 25000)] ** 2)))
claim(f"[nor_095 상세] hp RMS: 0–1.5s={r95(0.01, 1.5):.4f} 1.5–1.94s={r95(1.5, 1.94):.4f} 1.94–1.99s={r95(1.94, 1.99):.4f} | 충격 시점 {np.round(t95, 3).tolist()}")
# 110: 뷰어 피처(raw+hp, train 150) 정상 5–95 % 밖 개수 (정상 LOO)
for nm_ in ("features_raw.csv", "features_hp.csv"):
    V_ = pd.read_csv(os.path.join(ROOT, "단계", "2_EDA", "EDA뷰어", "figs", nm_), index_col=0).loc[names].drop(columns="y").astype(float)
    Vn = V_.values[nor]; cnt = np.zeros(n); ni = np.flatnonzero(nor)
    for i in range(n):
        ref = np.delete(Vn, int(np.where(ni == i)[0][0]), 0) if nor[i] else Vn
        lo, hi = np.percentile(ref, [5, 95], axis=0); cnt[i] = ((V_.values[i] < lo) | (V_.values[i] > hi)).sum()
    claim(f"[viewer {nm_} {V_.shape[1]}열] nor_110 밖 개수={cnt[ix['nor_110']]:.0f} (정상 중앙 {np.median(cnt[nor]):.0f}, 정상 최소 {cnt[nor].min():.0f}, 0개인 정상 {int((cnt[nor] == 0).sum())}/120) | 참고 nor_116={cnt[ix['nor_116']]:.0f} nor_065={cnt[ix['nor_065']]:.0f}")
claim(f"[기존] n_ev<=1 정상 {int((R.n_ev.values[:n][nor] <= 1).sum())} 고장 {int((R.n_ev.values[:n][~nor] <= 1).sum())} | H1_nimp==0 정상 {int((R.n_imp.values[:n][nor] == 0).sum())} 고장 {int((R.n_imp.values[:n][~nor] == 0).sum())} val {int((R.n_imp.values[n:] == 0).sum())}/40 | n_ev<=1 val {int((R.n_ev.values[n:] <= 1).sum())}/40")
claim(f"[N1==0] 정상 {int((R.N1_reb.values[:n][nor] == 0).sum())} 고장 {[names[i] for i in np.flatnonzero((~nor) & (R.N1_reb.values[:n] == 0))]} val {int((R.N1_reb.values[n:] == 0).sum())}/40; N1==0 & n_ev>=2: 정상 {int(((R.N1_reb.values[:n] == 0) & (R.n_ev.values[:n] >= 2) & nor).sum())} {[names[i][-3:] for i in np.flatnonzero((R.N1_reb.values[:n] == 0) & (R.n_ev.values[:n] >= 2) & nor)]} 고장 {int(((R.N1_reb.values[:n] == 0) & (R.n_ev.values[:n] >= 2) & ~nor).sum())}")
claim(f"[N5] 끝 100ms 몫>=0.5 정상 {int((R.N5_edge.values[:n][nor] >= .5).sum())} {[names[i][-3:] for i in np.flatnonzero(nor & (R.N5_edge.values[:n] >= .5))]} 고장 {[names[i][-3:] for i in np.flatnonzero(~nor & (R.N5_edge.values[:n] >= .5))]} val {int((R.N5_edge.values[n:] >= .5).sum())}/40")
open(os.path.join(HERE, "claims.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")

# %% C-1. 단변량: 정상 쪽 판정 (피처 s > 고장 중 (k+1)번째로 큰 값 → 고장 k개까지 허용)
def side_info(s, k):
    sf = np.sort(s[:n][~nor])[::-1]; thr = sf[k]
    on = s[:n] > thr
    return thr, on
def uni_table():
    out = []
    for f in list(SIGN):
        s = S_all[f]; st = s[:n]
        auc = roc_auc_score(nor, st)
        d = dict(feature=f, AUC=round(auc, 3))
        for k in (0, 1, 2):
            thr, on = side_info(s, k)
            d[f"cov@{k}"] = round(100 * on[nor].mean(), 1)
            d[f"leak@{k}"] = " ".join(names[i][-3:] for i in np.flatnonzero(on & ~nor))
            if k == 0:
                d["thr0"] = round(float(thr), 4)
                d["FP_rescued@0"] = " ".join(m[-3:] for m in FP_ALL if on[ix[m]])
                d["val_side@0"] = round(100 * (s[n:] > thr).mean(), 1)
                d["val_N"] = int((s[n:] > thr).sum())
                d["n_ties_at_thr"] = int((st == thr).sum())
            if k == 1: d["FP_rescued@1"] = " ".join(m[-3:] for m in FP_ALL if on[ix[m]])
            if k == 2: d["FP_rescued@2"] = " ".join(m[-3:] for m in FP_ALL if on[ix[m]])
        # val 이동: 학습 정상 분위수 안에서 val 중앙값
        d["val_median_pct_in_train_normals"] = round(100 * (st[nor] <= np.median(s[n:])).mean(), 0)
        d["val_vs_trainNormal_samples_on_normal_side@0"] = f"{int((s[n:] > thr).sum())}/{nvl} vs {int(on[nor].sum())}/120"
        out.append(d)
    return pd.DataFrame(out)
U = uni_table(); U.to_csv(os.path.join(HERE, "eval_univariate.csv"), index=False, encoding="utf-8-sig")
log(U.drop(columns=["val_vs_trainNormal_samples_on_normal_side@0"]).to_string())

# %% C-2. 폴드 안 임계값 안정성 (5-fold × 20 seeds): 임계 = 학습 고장 최댓값(k=0) 또는 Youden(정상 선별)
def stab(f):
    s = S_all[f]; rec = []; rec1 = []
    for seed in range(20):
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(all_idx, y):
            if f == "N4_nout": st_tr, st_te = n4_split(tr, te); st_tr, st_te = -st_tr, -st_te
            else: st_tr, st_te = s[tr], s[te]
            ytr, yte = y[tr], y[te]
            t0 = st_tr[ytr == 1].max()
            t1 = np.sort(st_tr[ytr == 1])[::-1][1]
            rec1.append(((st_te[yte == 0] > t1).mean(), int((st_te[yte == 1] > t1).sum()), t1))
            cand = np.unique(st_tr); J = [((st_tr[ytr == 0] > c).mean() - (st_tr[ytr == 1] > c).mean()) for c in cand]
            ty = cand[int(np.argmax(J))]
            rec.append((seed, t0, ty, (st_te[yte == 0] > t0).mean(), int(((st_te[yte == 1]) > t0).sum()), (st_te[yte == 0] > ty).mean(), int(((st_te[yte == 1]) > ty).sum())))
    r = pd.DataFrame(rec, columns=["seed", "t0", "ty", "cov0", "leak0", "covY", "leakY"])
    iqr = np.subtract(*np.percentile(s[:n][nor], [75, 25])) or 1.0
    mode_share = lambda v: pd.Series(np.round(v, 3)).value_counts(normalize=True).iloc[0]
    r1 = pd.DataFrame(rec1, columns=["cov1", "leak1", "t1"])
    return dict(feature=f, **{"cov(k=1) test %": round(100 * r1.cov1.mean(), 1), "leak(k=1) 폴드당": round(r1.leak1.mean(), 3), "t1 최빈값 비율": round(mode_share(r1.t1), 2)},**{"cov(k=0) test %": round(100 * r.cov0.mean(), 1), "leak(k=0) 폴드당 고장": round(r.leak0.mean(), 3),
                "t0 IQR/정상IQR": round(float(np.subtract(*np.percentile(r.t0, [75, 25])) / iqr), 3), "t0 최빈값 비율": round(mode_share(r.t0), 2),
                "Youden cov test %": round(100 * r.covY.mean(), 1), "Youden leak 폴드당": round(r.leakY.mean(), 3), "Youden 최빈값 비율": round(mode_share(r.ty), 2),
                "leak>0인 폴드 %": round(100 * (r.leak0 > 0).mean(), 1)})
ST = pd.DataFrame([stab(f) for f in list(SIGN)]); ST.to_csv(os.path.join(HERE, "eval_stability.csv"), index=False, encoding="utf-8-sig")
log(ST.to_string())

# %% C-3. 기존 열과의 중복 (스피어만, train 150)
EXIST = FIXED + ["L1_S"]
EM = pd.concat([T[EXIST].reset_index(drop=True), pd.DataFrame(dict(n_ev=R.n_ev.values[:n], dom=R.dom.values[:n]))], axis=1)
Dr = pd.read_csv(os.path.join(ROOT, "단계", "5_모델비교", "검토문서", "협의_1121", "code_D", "D_repeat_feats.csv"), index_col=0)
log("D_repeat dom 일치 확인 (max |차|):", float(np.abs(Dr.loc[names, "dom"].values - R.dom.values[:n]).max()))
EM["rattle"] = Dr.loc[names, "rattle"].values; EM["r21"] = Dr.loc[names, "r21"].values
red = []
for f in NEWF + ["ref_n_ev", "ref_H1_nimp", "ref_H14_share"]:
    v = F[f].values[:n]; rho = {c: spearmanr(v, EM[c].values)[0] for c in EM.columns}
    top = sorted(rho.items(), key=lambda kv: -abs(np.nan_to_num(kv[1])))[:2]
    red.append(dict(feature=f, max_abs_rho=round(abs(top[0][1]), 2), with_=top[0][0], second=f"{top[1][0]} {top[1][1]:+.2f}"))
RD = pd.DataFrame(red); RD.to_csv(os.path.join(HERE, "eval_redundancy.csv"), index=False, encoding="utf-8-sig"); log(RD.to_string())
# N2와 울림 주파수 의존
m_ = (R.N2_ret_ms.values[:n] > 0) & np.isfinite(R.N2_f.values[:n])
rho_f = spearmanr(R.N2_ret_ms.values[:n][m_], R.N2_f.values[:n][m_])[0]
med_n = np.median(R.N2_ret_ms.values[:n][nor & (R.N2_ret_ms.values[:n] > 0)])
log(f"N2: 울림 f와 스피어만 {rho_f:+.2f} (충격 있는 {m_.sum()}개). 정상 중앙 {med_n:.1f} ms → 정규화(÷중앙)는 순위 불변, AUC 같음")
for nm_, mk in (("f<4kHz", R.N2_f.values[:n] < 4000), ("f>=4kHz", R.N2_f.values[:n] >= 4000)):
    mm = m_ & mk
    if (y[mm] == 1).sum() > 2 and (y[mm] == 0).sum() > 2: log(f"  N2 AUC (정상 낮음) [{nm_}] n={mm.sum()} 고장{int((y[mm] == 1).sum())}: {roc_auc_score(y[mm] == 0, -R.N2_ret_ms.values[:n][mm]):.3f}")
    else: log(f"  N2 [{nm_}] n={mm.sum()} 고장{int((y[mm] == 1).sum())} 부족")

# %% C-4. 간이 증분 (quick_cv 5-fold × 5 seeds, 중첩 CV 아님). ET 입력 = J의 ET 열 (FIXED 13 + H16/H17 폴드 안) + 정상 피처
class ETq:
    def __init__(self, nf=13, nex=0, seed=0): self.nf, self.nex, self.seed = nf, nex, seed
    def get_params(self, deep=False): return dict(nf=self.nf, nex=self.nex, seed=self.seed)
    def _H(self, F_):
        H = F_[:, :self.nf]; pk = self.pr_.transform(F_[:, self.nf:self.nf + L.shape[1]]); ex = F_[:, self.nf + L.shape[1]:]
        return np.hstack([H, pk, ex])
    def fit(self, F_, y_):
        self.pr_ = PeakRef(float(fr[1] - fr[0])).fit(F_[y_ == 0, self.nf:self.nf + L.shape[1]])
        self.m_ = make_et(seed=self.seed)().fit(self._H(F_), y_); return self
    def decision_function(self, F_):
        p = np.clip(self.m_.predict_proba(self._H(F_))[:, 1], 1 / 600, 1 - 1 / 600); return np.log(p / (1 - p))
base = np.hstack([T[FIXED].values, L])
rawv = {k: F[k].values[:n] for k in NEWF}
# 사전 선언 생존 규칙: 단변량 AUC >= 0.70 AND cov@0 >= 10 % AND 폴드 안 leak(k=0) 폴드당 <= 0.2
surv = [f for f in NEWF if U.set_index("feature").loc[f, "AUC"] >= 0.70 and U.set_index("feature").loc[f, "cov@0"] >= 10 and ST.set_index("feature").loc[f, "leak(k=0) 폴드당 고장"] <= 0.2]
log("생존 피처:", surv)
cfgs = {"기준(J의 ET 열)": []}
for f in NEWF: cfgs["+" + f] = [f]
if surv: cfgs["+생존 전부 (" + ",".join(surv) + ")"] = surv
cfgs["+7개 전부"] = NEWF
oofs = {}; qrows = []
grp = np.arange(n)
for nm_, fs in cfgs.items():
    Fm = np.hstack([base] + [rawv[f][:, None] for f in fs]); q = quick_cv(Fm, y, grp, lambda fs=fs: ETq(13, len(fs)), seeds=range(5))
    o = q["oof"].mean(0); oofs[nm_] = (q["oof"], q["miss_rank"])
    pr = q["pr_auc"]
    pn = lambda f: 100 * (o[nor] <= o[ix[f]]).mean()
    qrows.append(dict(config=nm_, PR_AUC=round(pr.mean(), 3), PR_sd=round(pr.std(), 3), ROC=round(q["roc_auc"].mean(), 3),
                      **{f"pct_{f[-3:]}": round(pn(f)) for f in FP_ALL}, FP7_mean_pct=round(np.mean([pn(f) for f in FP_J]), 1),
                      fault_miss_rank_mean=round(float(np.mean(list(q["miss_rank"].values()))), 4)))
QR = pd.DataFrame(qrows); QR.to_csv(os.path.join(HERE, "eval_quick_incremental.csv"), index=False, encoding="utf-8-sig"); log(QR.to_string())
b_mr = oofs["기준(J의 ET 열)"][1]
for nm_ in cfgs:
    if nm_.startswith("기준"): continue
    mr = oofs[nm_][1]; dd = {names[k]: round(mr[k] - b_mr[k], 3) for k in mr}
    wors = {k: v for k, v in dd.items() if v > 0.03}; imp_ = {k: v for k, v in dd.items() if v < -0.03}
    log(f"{nm_}: 고장 miss_rank 악화(>0.03) {wors} | 개선(<-0.03) {imp_}")

# %% C-5. t-SNE 5-NN 순도 (perplexity 15, random_state 0–4)
def purity(Z, files):
    out = []
    for rs in range(5):
        E = TSNE(2, perplexity=15, random_state=rs, init="pca").fit_transform(Z)
        nn = NearestNeighbors(n_neighbors=6).fit(E); idx = nn.kneighbors(E)[1][:, 1:]
        out.append([np.mean(y[idx[ix[f]]] == 0) for f in files] + [np.mean(y[idx[~nor]] == 1)])
    return np.mean(out, 0), E
tfiles = FP_ALL
Zb = StandardScaler().fit_transform(T[FIXED].values)
spaces = {"기준 (FIXED 13열)": Zb}
if surv: spaces["+생존 피처"] = np.hstack([Zb, StandardScaler().fit_transform(np.column_stack([F[f].values[:n] for f in surv]))])
spaces["+7개 전부"] = np.hstack([Zb, StandardScaler().fit_transform(np.column_stack([F[f].values[:n] for f in NEWF]))])
TS = []; embs = {}
for k_, Z in spaces.items():
    pu, E = purity(Z, tfiles); embs[k_] = E
    TS.append(dict(space=k_, **{f[-3:]: round(v, 2) for f, v in zip(tfiles, pu[:-1])}, FP7_mean=round(float(np.mean(pu[:len(FP_J)])), 3), 고장_5NN_고장비율=round(float(pu[-1]), 3)))
TS = pd.DataFrame(TS); TS.to_csv(os.path.join(HERE, "eval_tsne_purity.csv"), index=False, encoding="utf-8-sig"); log(TS.to_string())

# %% 그림
cm = {"normal": "#2a78d6", "fault": "#eb6834", "val": "#8a8a8a"}
fig, axs = plt.subplots(2, 4, figsize=(20, 9)); axs = axs.ravel()
for ax, f in zip(axs, NEWF + ["ref_n_ev"]):
    v = F[f].values; rng = np.random.RandomState(0)
    for g_, msk, c in (("정상", np.r_[nor, np.zeros(nvl, bool)], cm["normal"]), ("고장", np.r_[~nor, np.zeros(nvl, bool)], cm["fault"]), ("val", np.r_[np.zeros(n, bool), np.ones(nvl, bool)], cm["val"])):
        yy = {"정상": 0, "고장": 1, "val": 2}[g_] + rng.uniform(-.25, .25, msk.sum())
        ax.scatter(v[msk], yy, s=14, color=c, alpha=.7, label=g_)
    for m in FP_ALL:
        ax.scatter(v[ix[m]], 0, s=60, facecolor="none", edgecolor="k", lw=1.1)
        ax.annotate(m[-3:], (v[ix[m]], 0), xytext=(0, 9 + 6 * (FP_ALL.index(m) % 3)), textcoords="offset points", fontsize=6.5, ha="center")
    thr = U.set_index("feature").loc[f, "thr0"] * SIGN[f] if f in U.feature.values else None
    ax.axvline(thr, color="k", ls="--", lw=.8)
    ax.set_yticks([0, 1, 2]); ax.set_yticklabels(["정상", "고장", "val"]); ax.set_title(f + ("  (정상=낮음)" if SIGN[f] < 0 else "  (정상=높음)") + "  점선=고장 0개 경계", fontsize=9)
    if f in ("N1_reb", "N2_ret_ms", "N3_ifrac", "N3_crest", "N4_nout"): ax.set_xscale("symlog", linthresh=max(1e-3, np.percentile(np.abs(v[v != 0]), 20))) if f in ("N1_reb", "N2_ret_ms") else None
axs[-1].axis("off"); axs[-1].text(0, .5, "원 = J 오경보 7 + 붙여 넣은 027·039·110\n(번호 3자리)", fontsize=10)
fig.suptitle("정상 증거 피처의 분포 — 정상 / 고장 / val (점선 = 고장을 모두 제외하는 경계, 그 바깥쪽이 '정상 쪽')"); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "distributions.png"), dpi=110); plt.close(fig)

fig, axs = plt.subplots(1, len(embs), figsize=(7 * len(embs), 6.5))
axs = np.atleast_1d(axs)
for ax, (k_, E) in zip(axs, embs.items()):
    ax.scatter(E[nor, 0], E[nor, 1], s=18, color=cm["normal"], marker="s", alpha=.7, label="정상"); ax.scatter(E[~nor, 0], E[~nor, 1], s=24, color=cm["fault"], label="고장")
    for m in FP_ALL: ax.scatter(*E[ix[m]], s=90, facecolor="none", edgecolor="k", lw=1.2); ax.annotate(m[-3:], E[ix[m]], xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.set_title(k_ + " (perplexity 15, random_state 4)", fontsize=10); ax.legend(fontsize=8)
fig.suptitle("t-SNE (라벨은 색에만) — 원 = 오경보 후보 10개"); fig.tight_layout(); fig.savefig(os.path.join(FIGS, "tsne.png"), dpi=110); plt.close(fig)

fig, axs = plt.subplots(2, 2, figsize=(16, 8))
for r_, f in enumerate(["nor_116", "nor_095"]):
    i = ix[f]; xh = highpass(X[i]); tt = np.arange(len(xh)) / 25000
    axs[r_, 0].plot(tt, xh, lw=.4, color="#2a78d6"); axs[r_, 0].set_title(f"{f} 1 kHz 고역 신호 (g)"); axs[r_, 0].set_xlabel("s")
    for t_ in arrs[i]["t"]: axs[r_, 0].axvline(t_, color="r", lw=.5, alpha=.5)
    tp = arrs[i]["t"][np.argmax(arrs[i]["hg"])]; sl = slice(int((tp - .01) * 25000), int((tp + .05) * 25000))
    axs[r_, 1].plot(tt[sl] - tp, xh[sl], lw=.6); axs[r_, 1].set_title(f"{f} 최대 충격 전후 60 ms 확대 (0 = 최대 충격)"); axs[r_, 1].set_xlabel("s")
fig.tight_layout(); fig.savefig(os.path.join(FIGS, "claim_116_095.png"), dpi=110); plt.close(fig)
log("done")
