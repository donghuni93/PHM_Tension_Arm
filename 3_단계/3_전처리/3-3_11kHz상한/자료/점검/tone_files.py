import os
import numpy as np
from scipy.signal import welch
D="c:/Users/User/WorkSpace/Donghun/3학년2학기/품질공학/data/TeamProject_train_변경/normal/"
out=[]
for n in ["nor_097","nor_101","nor_120","nor_116","nor_055","nor_001"]:
    x=np.loadtxt(D+n+".csv"); m=x.mean(); x=x-m
    fr,p=welch(x,fs=25000,window='hann',nperseg=25000//1,noverlap=0)  # 1 Hz res on 2 s (single segment x2)
    out.append(f"{n}: mean={m:+.3f} g")
    for a,b in [(12000,12500),(9000,9700)]:
        mk=(fr>=a)&(fr<b); i=np.argmax(p[mk]); f0=fr[mk][i]
        nb=(fr>=f0-300)&(fr<f0+300)&(np.abs(fr-f0)>30)
        out.append(f"   peak in {a/1000:.1f}-{b/1000:.1f}k: {f0:.0f} Hz, {10*np.log10(p[mk][i]):.1f} dB, peak/neighbor median = {10*np.log10(p[mk][i]/np.median(p[nb])):.1f} dB")
open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","figs","tone_files.txt"),"w").write("\n".join(out))
