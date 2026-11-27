# F R2: R의 u/A3/E, D의 dom/n_ev 를 F19 위에 얹었을 때 ET 축 quick_cv · 5-NN 순도 · FA 정상 순도 (탐색용)
import os, re
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "step5_final.py"), encoding="utf-8").read()
src = src.split("RANDOM = (fault.sum()")[0].replace("result_F5.txt", "result_F2check_tmp.txt")
exec(compile(src, "step5_part", "exec"))
OUT.close(); OUT = open(os.path.join(HERE, "result_F2.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); OUT.write(s + "\n"); OUT.flush()
RU = pd.read_csv(os.path.join(os.path.dirname(HERE), "code_R", "out", "R4b_u.csv"), encoding="utf-8-sig").set_index("file").loc[files]
DR = pd.read_csv(os.path.join(os.path.dirname(HERE), "code_D", "D_repeat_feats.csv"), encoding="utf-8-sig", index_col=0).loc[files]
DF = pd.read_csv(os.path.join(os.path.dirname(HERE), "code_D", "D_file_feats.csv"), encoding="utf-8-sig", index_col=0).loc[files]
Ucols = RU[["u_ring", "u_spec", "u_L"]].values.astype(float); A3 = RU[["A3"]].values.astype(float)
BF = DF[["body_frac"]].values.astype(float); BF = np.where(np.isnan(BF), np.nanmedian(BF[:ntr][nor]), BF)
Dd = np.column_stack([DR.dom.values, np.log1p(DF.n_ev.values)]).astype(float)
for nm, M in (("A3", A3), ("body_frac", BF), ("dom", Dd[:, :1]), ("log n_ev", Dd[:, 1:])):
    v = M[:, 0]; P(f"{nm}: AUC {roc_auc_score(y, v[:ntr]):.3f} | 정상 med {np.median(v[:ntr][nor]):.3f} val med {np.median(v[ntr:]):.3f}")
def mk(extra):
    def b(tr):
        wa, wb, _ = sel_win(tr); bh = sel_hi(tr)[0]; Rg = ring_feats(wa, wb, bh)
        rm, rh, rb = Rg[:, 4:5], Rg[:, 5:6], Rg[:, 2:3]
        F19 = np.hstack([NEW16, rh, rm, rb, Rg[:, 10:12]])
        return np.hstack([F19] + extra) if extra else F19
    return b
CONF = {"new14": lambda tr: NEW16, "F19": mk([]), "F19+dom+nev(D)": mk([Dd]), "F19+A3(R)": mk([A3]), "F19+A3+bodyfrac(R)": mk([A3, BF]),
        "F19+A3+bf+dom+nev": mk([A3, BF, Dd]), "new14+dom+nev": lambda tr: np.hstack([NEW16, Dd]), "F19+bodyfrac": mk([BF])}
def pur(Z, who, lab):
    nb = NearestNeighbors(n_neighbors=6).fit(Z[:ntr]).kneighbors(Z[who], return_distance=False)[:, 1:]
    return float(y[nb].mean()) if lab == 1 else float(1 - y[nb].mean())
class ET_(BaseEstimator):
    def __init__(self, nm="F19"): self.nm = nm
    def fit(self, Fi, yy):
        tr = Fi[:, 0].astype(int); self.M_ = CONF[self.nm](tr); self.m_ = make_et(seed=0)().fit(self.M_[tr], yy); return self
    def predict_proba(self, Fi): return self.m_.predict_proba(self.M_[Fi[:, 0].astype(int)])
Fidx = np.arange(ntr)[:, None].astype(float); fault_idx = np.flatnonzero(fault)
for nm, fn in CONF.items():
    pf, pn = [], []
    for s in range(5):
        for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=s).split(idx_all, y, idx_all):
            Z = np.clip(standardize(fn(tr), tr), -10, 10); w = te[fault[te]]
            if len(w): pf.append(pur(Z, w, 1))
            wn = np.array([i for i in te if isFA[i]]) if False else np.array([i for i in te if np.isin(files[i], FAN)])
            if len(wn): pn.append(pur(Z, wn, 0))
    q = quick_cv(Fidx, y, np.arange(ntr), lambda nm=nm: ET_(nm), seeds=range(5))
    worst = sorted(q["miss_rank"].items(), key=lambda kv: -kv[1])[:5]
    P(f"{nm:22s} PR-AUC {q['pr_auc'].mean():.3f}+-{q['pr_auc'].std(ddof=1):.3f} | 고장5NN {np.mean(pf):.3f}+-{np.std(pf):.3f} | FA정상5NN {np.mean(pn):.3f} | 놓침 {[(files[k][-3:], v) for k, v in worst]}")
