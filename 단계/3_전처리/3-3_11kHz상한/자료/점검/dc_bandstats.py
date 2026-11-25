import os
import numpy as np, glob, os
from scipy.signal import butter, sosfiltfilt
from scipy.stats import kurtosis, mannwhitneyu
R="c:/Users/User/WorkSpace/Donghun/3학년2학기/품질공학/data/"
G={'normal':sorted(glob.glob(R+"TeamProject_train_변경/normal/*.csv")),
   'abnormal':sorted(glob.glob(R+"TeamProject_train_변경/abnormal/*.csv")),
   'val':sorted(glob.glob(R+"TeamProject_val/val/*.csv"))}
X={g:{os.path.basename(f)[:-4]:np.loadtxt(f) for f in fl} for g,fl in G.items()}
out=[]
# 1) DC means
out.append("1) DC mean per group: count with mean < -0.5 g, and distinct clusters (rounded 0.05)")
for g in X:
    m={k:v.mean() for k,v in X[g].items()}
    low={k:round(v,3) for k,v in m.items() if v<-0.5}
    out.append(f"{g}: n={len(m)} min={min(m.values()):+.3f} max={max(m.values()):+.3f} median={np.median(list(m.values())):+.3f} | mean<-0.5: {low}")
    hi={k:round(v,3) for k,v in m.items() if abs(v)>0.1 and v>=-0.5}
    out.append(f"   |mean|>0.1 (not in the -0.87 cluster): {hi}")
# 2) 8-10.5k bandpass stats
sos=butter(4,[8000,10500],btype='band',fs=25000,output='sos')
def stats(x):
    x=x-x.mean(); y=sosfiltfilt(sos,x)
    w=y[:50000].reshape(20,2500); r=np.sqrt((w**2).mean(1))   # 0.1 s windows
    return dict(kurt=kurtosis(y), crest=np.abs(y).max()/np.sqrt((y**2).mean()), wr=r.max()/np.median(r))
S={g:{k:stats(v) for k,v in X[g].items()} for g in ['normal','abnormal']}
out.append("\n2) band 8-10.5k (4th-order Butterworth, zero-phase): excess kurtosis / crest / max-over-median 0.1s RMS")
for name in ['kurt','crest','wr']:
    n=np.array([S['normal'][k][name] for k in S['normal']]); a=np.array([S['abnormal'][k][name] for k in S['abnormal']])
    auc=mannwhitneyu(a,n).statistic/(len(a)*len(n))
    out.append(f"{name}: normal median={np.median(n):.2f} (p5 {np.percentile(n,5):.2f}, p95 {np.percentile(n,95):.2f}) | abnormal median={np.median(a):.2f} (p5 {np.percentile(a,5):.2f}, p95 {np.percentile(a,95):.2f}) | AUC={auc:.2f}")
out.append("\n3) tone normals vs rest (rank percentile within normals, 100=highest)")
keys=list(S['normal'])
for k in ['nor_055','nor_116','nor_097','nor_101','nor_120']:
    row=[]
    for name in ['kurt','crest','wr']:
        n=np.array([S['normal'][q][name] for q in keys]); v=S['normal'][k][name]
        row.append(f"{name}={v:.2f} (pct {100*(n<v).mean():.0f})")
    out.append(f"{k}: "+" | ".join(row))
# 4) how many normals are above abnormal median kurt
n=np.array([S['normal'][q]['kurt'] for q in keys]); a=np.array([S['abnormal'][q]['kurt'] for q in S['abnormal']])
out.append(f"\nnormals with kurt below -0.5 (tone-like): {(n<-0.5).sum()} of {len(n)}; abnormals: {(a<-0.5).sum()} of {len(a)}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figs","dc_bandstats.txt"),"w",encoding="utf-8").write("\n".join(out))
