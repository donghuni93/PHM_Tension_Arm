import os
import numpy as np, glob
from scipy.signal import welch
D="c:/Users/User/WorkSpace/Donghun/3학년2학기/품질공학/data"
fl=sorted(glob.glob(D+"/TeamProject_train_변경/*/*.csv"))+sorted(glob.glob(D+"/TeamProject_val/val/*.csv"))
ps=[]
for f in fl:
    x=np.loadtxt(f); x-=x.mean()
    fr,p=welch(x,fs=25000,window='hann',nperseg=5000,noverlap=2500); ps.append(p)
P=np.array(ps); db=lambda v:10*np.log10(v)
edges=np.arange(0,12501,1000); out=["all 190 files, per 1 kHz bin: min / p5 / p50 dB"]
for a,b in zip(edges[:-1],edges[1:]):
    m=(fr>=a)&(fr<b+(1 if b==12000 else 0)); v=P[:,m].mean(1)
    out.append(f"{a/1000:4.1f}-{b/1000:4.1f}  {db(v.min()):7.1f} {db(np.percentile(v,5)):7.1f} {db(np.median(v)):7.1f}")
q=(0.0000141**2)/12/12500; out.append(f"quantization floor approx {db(q):.1f} dB")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figs","floor_percentile.txt"),"w").write("\n".join(out))
