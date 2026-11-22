import sys, numpy as np
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic","AppleGothic","Noto Sans CJK KR","WenQuanYi Zen Hei"):
    if any(f == x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from src.data import load_dataset
S="/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad"
FS=25000
X,y,names = load_dataset("/home/user/PHM_Tension_Arm/data/TeamProject_train_변경")
num=lambda n:int("".join(c for c in n if c.isdigit()))
find=lambda k,l: next(i for i,(n,t) in enumerate(zip(names,y)) if t==l and num(n)==k)
lp=butter(4,500,fs=FS,output="sos"); hp=butter(4,1000,btype="high",fs=FS,output="sos")
def prep(i):
    x=X[i]-X[i].mean(); h=sosfiltfilt(hp,x); e=sosfiltfilt(lp,np.abs(hilbert(h))); return x,h,e
def impacts(e):
    thr=6*np.median(e); p,_=find_peaks(e,height=thr,distance=125)
    p=[k for k in p if 250<k<len(e)-250]
    # 더 큰 봉우리에서 30 ms 이내인 것(꼬리)은 제외
    out=[k for k in p if not any(abs(k-j)<750 and e[j]>e[k] for j in p)]
    return sorted(out,key=lambda k:-e[k])
def ring(h,k,n=200):  # 충격 직후 8 ms
    seg=h[k:k+n]; F=np.abs(np.fft.rfft(seg*np.hanning(n),4096)); fr=np.fft.rfftfreq(4096,1/FS)
    m=(fr>1000)&(fr<12000); f0=fr[m][np.argmax(F[m])]
    bp=butter(4,[f0*.85,f0*1.15],btype="band",fs=FS,output="sos")
    b=sosfiltfilt(bp,h[k-100:k+n+100]); env=np.abs(hilbert(b))[100:100+n]
    pk=np.argmax(env[:60]); t=np.arange(n-pk)/FS; ee=env[pk:]
    ok=ee>0.05*ee.max()
    a=np.polyfit(t[ok][:int(.006*FS)],np.log(ee[ok][:int(.006*FS)]),1)[0]   # 약 6 ms 이내 기울기
    zeta=-a/(2*np.pi*f0)
    return f0,zeta
rows=[]
for k_,lab in ((3,1),(22,1)):
    i=find(k_,lab); x,h,e=prep(i); im=impacts(e)
    print(names[i],"충격(큰 순)",[(round(k/FS,3),round(e[k],4)) for k in im])
    rows.append((i,im[0],"큰 충격")); rows.append((i,im[-1],"작은 충격"))
fig,ax=plt.subplots(4,3,figsize=(15,12))
for r,(i,k,tag) in enumerate(rows):
    x,h,e=prep(i); a0=ax[r]
    w=np.arange(k-50,k+250); tt=(w-k)/FS*1000
    f0,z=ring(h,k)
    a0[0].plot(tt,x[w],color="0.55",lw=.9,label="원본"); a0[0].plot(tt,h[w],color="C3",lw=1,label="1 kHz 고역통과 후")
    a0[0].axvline(0,color="k",lw=.5,ls=":"); a0[0].legend(fontsize=8,loc="upper right")
    a0[0].set_title(f"{names[i]} {tag} (t={k/FS:.3f} s, 포락선 {e[k]:.3f} g)",fontsize=10)
    a0[0].set_ylabel("가속도 (g)"); a0[0].set_xlabel("충격 기준 시간 (ms)")
    a0[1].plot(tt,h[w],color="C3",lw=.8); env=np.abs(hilbert(h))[w]; a0[1].plot(tt,env,color="k",lw=1)
    a0[1].plot(tt,-env,color="k",lw=1); a0[1].set_xlim(-2,10)
    tau=1/(2*np.pi*f0*z) if z>0 else np.nan
    a0[1].set_title(f"필터 후 확대: 울림 {f0/1000:.2f} kHz, ζ≈{z:.3f}, τ≈{tau*1000:.2f} ms",fontsize=10); a0[1].set_xlabel("ms")
    seg=h[k:k+200]*np.hanning(200); F=np.abs(np.fft.rfft(seg,4096)); fr=np.fft.rfftfreq(4096,1/FS)
    F0=np.abs(np.fft.rfft(x[k:k+200]*np.hanning(200),4096))
    a0[2].semilogy(fr/1000,F0,color="0.55",label="원본 8 ms"); a0[2].semilogy(fr/1000,F,color="C3",label="필터 후 8 ms"); a0[2].set_xlim(0,12.5)
    a0[2].set_xlabel("주파수 (kHz)"); a0[2].set_title("충격 직후 8 ms 스펙트럼",fontsize=10); a0[2].legend(fontsize=8)
    print(names[i],tag,"t=%.3f"%(k/FS),"env=%.4f"%e[k],"f=%.0f"%f0,"zeta=%.3f"%z,"peak|h|=%.3f"%np.abs(h[k-20:k+20]).max())
fig.suptitle("큰 충격 vs 작은 충격 (같은 파일 안에서), 충격 직후 8 ms 확대")
fig.tight_layout(); fig.savefig(f"{S}/zoom.png",dpi=115)
