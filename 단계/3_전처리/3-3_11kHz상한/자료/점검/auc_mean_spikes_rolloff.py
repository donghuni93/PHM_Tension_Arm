import os
import numpy as np, glob, os
from scipy.signal import welch
from scipy.stats import mannwhitneyu
D="c:/Users/User/WorkSpace/Donghun/3학년2학기/품질공학/data/TeamProject_train_변경"
def load(g):
    fl=sorted(glob.glob(D+"/"+g+"/*.csv")); P=[]
    for f in fl:
        x=np.loadtxt(f); x-=x.mean()
        fr,p=welch(x,fs=25000,window='hann',nperseg=5000,noverlap=2500); P.append(p)
    return [os.path.basename(f)[:-4] for f in fl],fr,np.array(P)
nn,fr,Pn=load("normal"); na,_,Pa=load("abnormal")
db=lambda v:10*np.log10(v)
def bm(P,a,b): return P[:,(fr>=a)&(fr<b)].mean(1)
out=[]
out.append("A) per 500Hz bin: AUC(abn>nor) | abn p25 vs nor p75 (dB) | overlap?")
for a in np.arange(0,12500,500):
    n=bm(Pn,a,a+500+(1 if a==12000 else 0)); b=bm(Pa,a,a+500+(1 if a==12000 else 0))
    u=mannwhitneyu(b,n).statistic/(len(b)*len(n))
    p25=db(np.percentile(b,25)); p75=db(np.percentile(n,75))
    out.append(f"{a/1000:5.2f}-{(a+500)/1000:5.2f} AUC={u:.2f} abn_p25={p25:6.1f} nor_p75={p75:6.1f} {'OVERLAP' if p25<p75 else 'sep'}")
out.append("\nB) normal files driving mean spikes (share of linear mean in band)")
for a,b in [(9200,9600),(12200,12450),(8000,10500)]:
    v=bm(Pn,a,b); s=v/v.sum(); idx=np.argsort(-s)[:4]
    out.append(f"{a/1000:.2f}-{b/1000:.2f} kHz: top normals "+", ".join(f"{nn[i]} {s[i]*100:.0f}%" for i in idx)+f"  | median-file share would be {100/len(v):.1f}%")
out.append("\nC) near-Nyquist rolloff: L(12.25-12.5k) - L(10.5-11k) in dB, files with L(10.5-11k) >= floor+15 dB (floor=-107)")
for name,P in [("normal",Pn),("abnormal",Pa)]:
    l1=db(bm(P,10500,11000)); l2=db(bm(P,12250,12501)); l3=db(bm(P,11500,12000))
    sel=l1>=-107+15
    d=(l2-l1)[sel]
    out.append(f"{name}: n_sel={sel.sum()} median diff={np.median(d):.1f} dB (p10 {np.percentile(d,10):.1f}, p90 {np.percentile(d,90):.1f}); L(11.5-12)-L(10.5-11) median={np.median((l3-l1)[sel]):.1f}")
out.append("\nD) all train: fraction of files whose last 250Hz (12.25-12.5k) is LOWER than 11.5-12k by >6 dB, among files >= floor+10 dB at 11.5-12k")
for name,P in [("normal",Pn),("abnormal",Pa)]:
    l3=db(bm(P,11500,12000)); l2=db(bm(P,12250,12501)); sel=l3>=-107+10
    out.append(f"{name}: n_sel={sel.sum()} frac drop>6dB = {(l3-l2)[sel].__gt__(6).mean():.2f}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figs","auc_mean_spikes_rolloff.txt"),"w",encoding="utf-8").write("\n".join(out))
