import numpy as np, glob, os
from scipy.signal import welch
R="c:/Users/User/WorkSpace/Donghun/3학년2학기/품질공학/data/"
G={'normal':sorted(glob.glob(R+"TeamProject_train_변경/normal/*.csv")),
   'abnormal':sorted(glob.glob(R+"TeamProject_train_변경/abnormal/*.csv")),
   'val':sorted(glob.glob(R+"TeamProject_val/val/*.csv"))}
db=lambda v:10*np.log10(v)
F={}; 
for g,fl in G.items():
    for f in fl:
        x=np.loadtxt(f); x-=x.mean()
        fr,p=welch(x,fs=25000,window='hann',nperseg=5000,noverlap=2500)
        F[(g,os.path.basename(f)[:-4])]=p
out=[]
def peak_ratio(p,a,b):
    mk=(fr>=a)&(fr<b); i=np.argmax(p[mk]); f0=fr[mk][i]
    nb=(fr>=f0-300)&(fr<f0+300)&(np.abs(fr-f0)>50)
    return db(p[mk][i]/np.median(p[nb])), f0
for name,(a,b) in {'9.3-9.6k':(9300,9600),'6.0-6.3k':(6000,6300)}.items():
    out.append(f"narrowband peak/neighbor >= 10 dB in {name}:")
    for g in ['normal','abnormal','val']:
        hit=[(k[1],round(peak_ratio(p,a,b)[0],1),round(peak_ratio(p,a,b)[1])) for k,p in F.items() if k[0]==g and peak_ratio(p,a,b)[0]>=10]
        n=sum(1 for k in F if k[0]==g)
        out.append(f"  {g}: {len(hit)}/{n}  {hit[:12]}")
out.append("\nlow band: 0.005-0.7k mean dB minus 0.7-1.5k mean dB (more negative = low-frequency cut)")
rows=[]
for k,p in F.items():
    lo=db(p[(fr>5)&(fr<700)].mean()); mid=db(p[(fr>=700)&(fr<1500)].mean()); rows.append((lo-mid,k))
rows.sort()
for d,k in rows[:8]: out.append(f"  lowest: {k[0]} {k[1]} {d:.1f} dB")
for k in [('normal','nor_055'),('normal','nor_116'),('normal','nor_001')]:
    out.append(f"  {k[1]}: {[round(d,1) for d,kk in rows if kk==k][0]} dB")
import collections
for g in ['normal','abnormal','val']:
    v=np.array([d for d,k in rows if k[0]==g]); out.append(f"  {g}: median {np.median(v):.1f}, p5 {np.percentile(v,5):.1f}, p95 {np.percentile(v,95):.1f}")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figs","peak_prevalence_lowband.txt"),"w",encoding="utf-8").write("\n".join(out))
