import os
import numpy as np, glob, os
from scipy.signal import welch
D="c:/Users/User/WorkSpace/Donghun/3학년2학기/품질공학/data"
groups={'normal':sorted(glob.glob(D+"/TeamProject_train_변경/normal/*.csv")),
        'abnormal':sorted(glob.glob(D+"/TeamProject_train_변경/abnormal/*.csv")),
        'val':sorted(glob.glob(D+"/TeamProject_val/val/*.csv"))}
fs=25000
out=[]
P={}
steps={}
for g,fl in groups.items():
    ps=[]; st=[]
    for f in fl:
        x=np.loadtxt(f); x=x-x.mean()
        fr,p=welch(x,fs=fs,window='hann',nperseg=5000,noverlap=2500)
        ps.append(p)
        d=np.abs(np.diff(np.loadtxt(f))); d=d[d>0]; st.append(d.min())
    P[g]=np.array(ps); steps[g]=np.median(st)
    out.append(f"{g}: n={len(fl)} median min-nonzero-step(g)={steps[g]:.3e}")
edges=np.arange(0,12501,500)
def binm(p):  # p: files x freq -> files x bins (mean power in bin)
    return np.array([p[:,(fr>=a)&(fr<b+(1 if b==12500 else 0))].mean(1) for a,b in zip(edges[:-1],edges[1:])]).T
B={g:binm(P[g]) for g in P}
db=lambda v:10*np.log10(v)
out.append("\nbin(kHz) | median dB: normal abnormal val | abn-nor(dB) | frac abn files > nor median | nor 5-95% spread(dB)")
nm=np.median(B['normal'],0)
for i,(a,b) in enumerate(zip(edges[:-1],edges[1:])):
    n=db(np.median(B['normal'][:,i])); ab=db(np.median(B['abnormal'][:,i])); v=db(np.median(B['val'][:,i]))
    frac=(B['abnormal'][:,i]>nm[i]).mean()
    sp=db(np.percentile(B['normal'][:,i],95))-db(np.percentile(B['normal'][:,i],5))
    out.append(f"{a/1000:5.1f}-{b/1000:4.1f} | {n:7.1f} {ab:7.1f} {v:7.1f} | {ab-n:6.1f} | {frac:4.2f} | {sp:5.1f}")
# fine near top
out.append("\nfine tail (250 Hz bins above 10 kHz), median dB normal/abnormal/val")
for a in np.arange(10000,12500,250):
    m=(fr>=a)&(fr<a+250)
    out.append(f"{a/1000:6.2f} | "+" ".join(f"{db(np.median(P[g][:,m].mean(1))):7.1f}" for g in ['normal','abnormal','val']))
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figs","floor_median.txt"),"w",encoding="utf-8").write("\n".join(out))
