from common import *
P = log_to("d8_oldclaims_log.txt")
X, y, names, Xv, nv = load_all(); nor = y == 0
H = pd.read_csv(os.path.join(HERE, "D_file_feats.csv"), index_col=0).join(pd.read_csv(os.path.join(HERE, "D_repeat_feats.csv"), index_col=0))
Ht, Hv = H.loc[names], H.loc[nv]
for lab, f in [("n_imp>=3 & body_frac>=0.9", lambda d: (d.n_imp >= 3) & (d.body_frac >= 0.9)), ("n_imp>=3 & body_frac>=0.8", lambda d: (d.n_imp >= 3) & (d.body_frac >= 0.8)), ("n_imp>=3 & body_frac>=0.7", lambda d: (d.n_imp >= 3) & (d.body_frac >= 0.7)),
               ("n_big==1 (새 정의: 최대의 50% 이상 사건 수)", lambda d: d.n_big == 1), ("n_big==1 & n_imp>=2", lambda d: (d.n_big == 1) & (d.n_imp >= 2)), ("n_ev==1", lambda d: d.n_ev == 1), ("n_ev<=1", lambda d: d.n_ev <= 1), ("n_imp==0", lambda d: d.n_imp == 0)]:
    a = f(Ht); P(f"{lab}: 정상 {int((a&nor).sum())} 고장 {int((a&~nor).sum())} {list(np.array(names)[(a&~nor).values])[:8]} val {int(f(Hv).sum())}")
