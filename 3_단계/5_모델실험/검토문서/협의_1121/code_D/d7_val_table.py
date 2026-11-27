from common import *
P = log_to("d7_val_table_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM(); core = FM.core_; THR = FM.thr_
Ttr, Tva, cols = tables(FM, X, Xv); Dv = core.parts(Tva); rule_va = (Dv.rule83 | Dv.knock).values
H = pd.read_csv(os.path.join(HERE, "D_file_feats.csv"), index_col=0).join(pd.read_csv(os.path.join(HERE, "D_repeat_feats.csv"), index_col=0))
Hv = H.loc[nv].copy(); Hv["ae_z"] = Dv.ae_z.values; Hv["et_z"] = Dv.et_z.values; Hv["lr"] = Dv.lr.values; Hv["rule"] = rule_va.astype(int); Hv["A판정"] = ((Dv.lr.values >= THR) | rule_va).astype(int)
Hv["d_thr"] = Hv.lr - THR
sel = Hv[(Hv.A판정 == 1) | (np.abs(Hv.d_thr) < 1.0)]
P(sel[["A판정", "rule", "lr", "d_thr", "ae_z", "et_z", "n_imp", "n_ev", "dom", "n_hi", "lo_frac", "body_frac", "h1"]].round(2).sort_values("lr", ascending=False).to_string())
P("val lr 분포: lr>=1 %d, 0<=lr<1 %d, -1<=lr<0 %d, <-1 %d" % ((Hv.lr >= 1).sum(), ((Hv.lr >= 0) & (Hv.lr < 1)).sum(), ((Hv.lr >= -1) & (Hv.lr < 0)).sum(), (Hv.lr < -1).sum()))
P("A 판정 고장 중 n_ev<=1:", list(Hv.index[(Hv.A판정 == 1) & (Hv.n_ev <= 1)]), "/ n_ev<=2:", list(Hv.index[(Hv.A판정 == 1) & (Hv.n_ev <= 2)]))
P("val 경계(|lr-1|<1) 중 n_ev>=4 (고장쪽 증거):", list(Hv.index[(np.abs(Hv.d_thr) < 1) & (Hv.A판정 == 0) & (Hv.n_ev >= 4)]))
P("val n_imp==0:", list(Hv.index[Hv.n_imp == 0]))
P("val dc 최소", float(pd.Series(Xv.mean(1)).min()))
