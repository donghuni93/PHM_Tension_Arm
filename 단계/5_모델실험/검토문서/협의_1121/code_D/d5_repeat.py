"""D 제안 검증 3: '한 번의 큰 타격이 지배하나, 비슷한 크기가 반복되나' (H1' 반복 기전)을 새 충격 정의로 연속 피처화. 라벨 사용(탐색)."""
from common import *
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
P = log_to("d5_repeat_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; THR = FM.thr_
Ttr, Tva, cols = tables(FM, X, Xv); n = len(y); nor = y == 0
Dt = core.parts(Ttr); Dv = core.parts(Tva); oofz = core.oof_z_; lr_oof = core.lr_.decision_function(oofz)
rule_tr = (Dt.rule83 | Dt.knock).values; rule_va = (Dv.rule83 | Dv.knock).values
NA, IMP = pickle.load(open(os.path.join(CACHE, "impacts.pkl"), "rb"))
def fe(im):
    t, h = im["t"], im["hx"]; k = len(t)
    if k == 0: return dict(r21=0., dom=1., n_big=0, rattle=0, h1=0.)
    ev = np.r_[0, np.cumsum(np.diff(t) > 0.05)]; ne = ev.max() + 1
    eh = np.array([h[ev == e].max() for e in range(ne)]); ec = np.array([(ev == e).sum() for e in range(ne)])
    s = np.sort(eh)[::-1]
    return dict(r21=float(s[1] / s[0]) if ne > 1 else 0., dom=float(s[0] / s.sum()), n_big=int((eh >= 0.5 * s[0]).sum()), rattle=int(ec.max()), h1=float(s[0]))
G = pd.DataFrame([fe(im) for im in IMP], index=NA); G.to_csv(os.path.join(HERE, "D_repeat_feats.csv"), encoding="utf-8-sig")
Gtr, Gva = G.loc[names], G.loc[nv]
P("== 1. 반복 기전 피처 (새 충격 정의: 사건 = 50 ms 안 충격 묶음, 높이 = 사건 최대 height_x) ==")
for c in ("r21", "dom", "n_big", "rattle", "h1"):
    v = Gtr[c].values.astype(float); P(f"{c}: AUC {roc_auc_score(y, v):.3f} | 정상 중앙 {np.median(v[nor]):.2f} p5/p95 {np.percentile(v[nor],5):.2f}/{np.percentile(v[nor],95):.2f} | 고장 중앙 {np.median(v[y==1]):.2f} p5/p95 {np.percentile(v[y==1],5):.2f}/{np.percentile(v[y==1],95):.2f} | val 중앙 {np.median(Gva[c].values.astype(float)):.2f}")
KEY = ["abn_022", "abn_002", "abn_017", "abn_009", "nor_116", "nor_065", "nor_059", "nor_055", "nor_025", "nor_037", "nor_095"]
P(Gtr.loc[KEY].round(2).to_string())
# 정상 전용: dom >= c (한 사건이 높이 합의 c 이상) → 고장 몇 개?
for c in (0.5, 0.6, 0.7, 0.8, 0.9):
    P(f"dom>={c}: 정상 {int((nor&(Gtr.dom>=c)).sum())} 고장 {int(((y==1)&(Gtr.dom>=c)).sum())} {[names[i] for i in np.flatnonzero((y==1)&(Gtr.dom>=c).values)]} val {int((Gva.dom>=c).sum())}")
for c in (1, 2):
    P(f"n_big<={c}: 정상 {int((nor&(Gtr.n_big<=c)).sum())} 고장 {int(((y==1)&(Gtr.n_big<=c)).sum())} val {int((Gva.n_big<=c).sum())}")
# 폴드 안 재선택: '정상 전용 컷' = dom >= c 중 학습 고장 0이 되는 최소 c(후보 격자), 테스트 폴드에서 켜진 고장
grid = np.round(np.arange(0.5, 1.001, 0.05), 2); dom = Gtr.dom.values; cs = []; hit = {}; offn = {}
for s in range(20):
    for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
        c = next((g for g in grid if ((dom[a][y[a] == 1]) >= g).sum() == 0), 1.01); cs.append(c)
        for i in b[dom[b] >= c]:
            (hit if y[i] == 1 else offn)[names[i]] = (hit if y[i] == 1 else offn).get(names[i], 0) + 1
P("폴드 안 컷 분포(dom>=c, 학습 고장 0 되는 최소 c):", dict(zip(*[x.tolist() for x in np.unique(cs, return_counts=True)])), "| 테스트 폴드에서 켜진 고장:", hit, "| 켜진 정상 종류", len(offn), "/ 켜진 정상 중 OOF 오경보:", {k: v for k, v in offn.items() if k in ("nor_055", "nor_065", "nor_116", "nor_059", "nor_025")})
P("고장 dom 큰 상위 5:", [(names[i], round(dom[i], 2)) for i in np.argsort(-np.where(y == 1, dom, -1))[:5]])

P("== 2. 결합 점수 위에서 효과 (로지스틱: AE z, ET z + log1p 축 / 정상 기준 z; ET스택) ==")
def cv(extra, model="lr", seeds=range(5)):
    M = np.hstack([oofz] + extra) if extra else oofz; S = np.zeros((len(seeds), n))
    for si, s in enumerate(seeds):
        for a, b in StratifiedKFold(5, shuffle=True, random_state=s).split(np.arange(n), y):
            Mz = M.copy()
            if extra:
                nm = a[y[a] == 0]; med = np.median(M[nm, 2:], 0); sp = np.maximum(np.percentile(M[nm, 2:], 90, 0) - med, 1e-6); Mz[:, 2:] = (M[:, 2:] - med) / sp
            if model == "lr": m = LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000).fit(Mz[a], y[a]); S[si, b] = m.decision_function(Mz[b])
            else: m = ExtraTreesClassifier(300, min_samples_leaf=2, max_features=None, class_weight="balanced", random_state=s, n_jobs=1).fit(Mz[a], y[a]); S[si, b] = m.predict_proba(Mz[b])[:, 1]
    return S
Fa = pd.read_csv(os.path.join(HERE, "D_file_feats.csv"), index_col=0).loc[names]
ex = {"기준": [], "+dom": [Gtr[["dom"]].values], "+r21": [Gtr[["r21"]].values], "+n_big": [np.log1p(Gtr[["n_big"]].values)], "+rattle": [np.log1p(Gtr[["rattle"]].values)],
      "+dom,n_imp,res": [Gtr[["dom"]].values, np.log1p(Fa[["n_imp"]].values), Fa[["res_fit"]].values],
      "+dom,n_big,n_imp,res,n_hi": [Gtr[["dom"]].values, np.log1p(Gtr[["n_big"]].values), np.log1p(Fa[["n_imp"]].values), Fa[["res_fit"]].values, np.log1p(Fa[["n_hi"]].values)]}
for model in ("lr", "et"):
    for k, v in ex.items():
        S = cv(v, model); sm = S.mean(0)
        P(f"  [{model}] {k}: ROC {np.mean([roc_auc_score(y,s) for s in S]):.3f} PR {np.mean([average_precision_score(y,s) for s in S]):.3f} | " + " ".join(f"{f[-7:]}:{100*(sm[nor]<=sm[names.index(f)]).mean():.0f}" for f in KEY))
# 5-NN 순도 (z 공간)
from sklearn.neighbors import NearestNeighbors
def purity(M):
    Mz = (M - np.median(M[nor], 0)) / np.maximum(np.percentile(M[nor], 90, 0) - np.median(M[nor], 0), 1e-6); Mz = np.clip(Mz, -6, 6)
    idx = NearestNeighbors(n_neighbors=6).fit(Mz).kneighbors(Mz, return_distance=False)[:, 1:]; return float(y[idx[y == 1]].mean())
for k, v in ex.items(): P(f"  5-NN 고장 순도 {k}: {purity(np.hstack([oofz]+v) if v else oofz):.3f}")
P("== 3. val: 지배 사건 dom 높은 파일, val 경계 파일의 위치 ==")
bd = [i for i in range(40) if (not rule_va[i]) and abs(Dv.lr.values[i] - THR) < 1.0]
T = Gva.iloc[bd].round(2).copy(); T["lr"] = Dv.lr.values[bd].round(2); P(T.to_string())
P("val dom>=0.8:", [(nv[i], round(Gva.dom.values[i], 2)) for i in np.flatnonzero(Gva.dom.values >= 0.8)])
P("완료")
