from common import *
from scipy.stats import spearmanr
P = log_to("d9_shift_cause_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; nor = y == 0
Ttr, Tva, cols = tables(FM, X, Xv); Dv = core.parts(Tva); oofz = core.oof_z_
H = pd.read_csv(os.path.join(HERE, "D_file_feats.csv"), index_col=0).join(pd.read_csv(os.path.join(HERE, "D_repeat_feats.csv"), index_col=0))
Ht, Hv = H.loc[names], H.loc[nv]
for c in ("lo_frac", "body_frac", "n_imp", "n_ev", "h1"):
    for lab, d, z in (("정상", Ht[nor], oofz[nor]), ("val", Hv, np.c_[Dv.ae_z.values, Dv.et_z.values])):
        m = ~np.isnan(d[c].values.astype(float))
        P(f"{c} vs AEz/ETz Spearman [{lab}]: {spearmanr(d[c].values[m], z[m,0])[0]:.2f} / {spearmanr(d[c].values[m], z[m,1])[0]:.2f} (n={int(m.sum())})")
# val 중 lo_frac 큰(>=0.5) / 작은 파일의 AE z 중앙
for lab, d, z in (("정상", Ht[nor], oofz[nor]), ("val", Hv, np.c_[Dv.ae_z.values, Dv.et_z.values])):
    hi = (d.lo_frac >= 0.5).values; lo = (d.lo_frac < 0.5).values
    P(f"[{lab}] lo_frac>=0.5: n={hi.sum()} AEz 중앙 {np.median(z[hi,0]):.2f} ETz {np.median(z[hi,1]):.2f} | lo_frac<0.5: n={lo.sum()} AEz {np.median(z[lo,0]):.2f} ETz {np.median(z[lo,1]):.2f}")
# 충격 없는 파일(n_imp==0)에서의 AE z : 충격과 무관한 스펙트럼 이동인가
for lab, d, z in (("정상", Ht[nor], oofz[nor]), ("val", Hv, np.c_[Dv.ae_z.values, Dv.et_z.values])):
    k = (d.n_imp <= 1).values; P(f"[{lab}] 충격<=1: n={k.sum()} AEz 중앙 {np.median(z[k,0]):.2f} ETz {np.median(z[k,1]):.2f} | 충격>=2: n={(~k).sum()} AEz {np.median(z[~k,0]):.2f} ETz {np.median(z[~k,1]):.2f}")
