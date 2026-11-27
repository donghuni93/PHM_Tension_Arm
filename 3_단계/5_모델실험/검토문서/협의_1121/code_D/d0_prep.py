from common import *
P = log_to("d0_prep_log.txt")
X, y, names, Xv, nv = load_all(); FM = load_FM()
P("train", X.shape, "val", Xv.shape, "thr", FM.thr_, "thr_all", FM.thr_all_, "cols", len(FM.cols_))
Ttr, Tva, cols = tables(FM, X, Xv)
Dv = FM.core_.parts(Tva); Dv.insert(0, "file", nv)
Dt = FM.core_.parts(Ttr); Dt.insert(0, "file", names)
Dv.to_csv(os.path.join(HERE, "val_A_parts.csv"), index=False, encoding="utf-8-sig")
c = FM.core_
oofz = c.oof_z_; lr_oof = c.lr_.decision_function(oofz)
P("lr coef", c.lr_.coef_, c.lr_.intercept_)
P("OOF lr: normal sd %.3f median %.3f p95 %.3f ; fault median %.3f" % (lr_oof[y==0].std(), np.median(lr_oof[y==0]), np.percentile(lr_oof[y==0],95), np.median(lr_oof[y==1])))
P("OOF z normal med/sd AE %.2f/%.2f ET %.2f/%.2f" % (np.median(oofz[y==0,0]), oofz[y==0,0].std(), np.median(oofz[y==0,1]), oofz[y==0,1].std()))
P("val z median AE %.2f ET %.2f ; train-normal-p95 AE %.2f ET %.2f" % (Dv.ae_z.median(), Dv.et_z.median(), np.percentile(oofz[y==0,0],95), np.percentile(oofz[y==0,1],95)))
P("val rule on (rule83|knock):", int((Dv.rule83|Dv.knock).sum()), list(Dv.file[Dv.rule83|Dv.knock]))
pre = Dv.lr + 100*Dv.rule83 + 100*Dv.knock
P("val A(b) pred (pre-body, thr 1.0):", int((pre>=FM.thr_).sum()), list(Dv.file[pre>=FM.thr_]))
P("val lr within +-0.5 of thr (no rule):", list(Dv.file[(~Dv.rule83&~Dv.knock)&((Dv.lr-FM.thr_).abs()<0.5)]))
P("val lr within +-1.0 of thr (no rule):", int(((~Dv.rule83&~Dv.knock)&((Dv.lr-FM.thr_).abs()<1.0)).sum()))
