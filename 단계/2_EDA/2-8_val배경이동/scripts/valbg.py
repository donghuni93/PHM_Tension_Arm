S='/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad'
import sys, numpy as np
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic","Noto Sans CJK KR","WenQuanYi Zen Hei"):
    if any(f==x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False
from scipy.signal import spectrogram, butter, sosfiltfilt, hilbert, find_peaks
from src.data import load_dataset, load_folder
FS=25000
X,y,names=load_dataset("/home/user/PHM_Tension_Arm/data/TeamProject_train_변경")
Xv,nv=load_folder("/home/user/PHM_Tension_Arm/data/TeamProject_val")
hp=butter(4,1000,btype="high",fs=FS,output="sos"); lp=butter(4,500,fs=FS,output="sos")
def imp_times(x):   # final.impact_rings와 같은 검출 (1 kHz HP → 포락선 → 500 Hz LP → 중앙값×6, 5 ms)
    h=sosfiltfilt(hp,x-x.mean()); e=sosfiltfilt(lp,np.abs(hilbert(h)))
    p,_=find_peaks(e,height=6*np.median(e),distance=125); return p/FS
def run(XX, post):
    A,B,N=[],[],[]
    for x in XX:
        x=x-x.mean(); f,t,Sg=spectrogram(x,fs=FS,nperseg=2048,noverlap=1024,mode="psd")
        imp=imp_times(x); ok=np.ones(len(t),bool)
        for ti in imp: ok&=~((t+0.041>ti-0.005)&(t-0.041<ti+post))
        A.append(Sg.mean(1)); N.append(len(imp))
        B.append(Sg[:,ok].mean(1) if ok.sum()>=5 else np.full(len(f),np.nan))
    return f,np.array(A),np.array(B),np.array(N)
f,Atr,Btr,Ntr=run(X,0.030); _,Av,Bv,Nv=run(Xv,0.030)
okv=np.isfinite(Bv[:,0]); okt=np.isfinite(Btr[:,0])
q=lambda M,p:np.nanpercentile(M,p,axis=0)
nor=(y==0)&okt; flt=(y==1)&okt
print("충격 수 중앙값: 정상 %.0f 고장 %.0f val %.0f | 배경 프레임 부족 val %d"%(np.median(Ntr[y==0]),np.median(Ntr[y==1]),np.median(Nv),(~okv).sum()))
bands=[(100,1000),(1000,2000),(2000,3000),(3000,5000),(5000,8000),(8000,10000),(10000,12500)]
db=lambda a,b: 10*np.log10(a/b)
rows=[]
print("대역 | 전체 val/정상 | 배경 val/정상 | 배경 고장/정상 | (전체) 고장/정상")
for lo,hi in bands:
    m=(f>=lo)&(f<hi); g=lambda M,s: np.nanmedian(M[s][:,m].mean(1))
    r=[db(g(Av,okv|True),g(Atr,y==0)), db(g(Bv,okv),g(Btr,nor)), db(g(Btr,flt),g(Btr,nor)), db(g(Atr,y==1),g(Atr,y==0))]
    print(f"{lo}-{hi}: "+" | ".join(f"{v:+.1f}" for v in r)); rows.append(r)
# 파일별: 정상 배경 95% 선을 넘는 파일 수
print("\n배경 PSD가 정상 배경 95% 선을 넘는 파일 비율(대역 평균 기준)")
for lo,hi in ((3000,5000),(8000,10000)):
    m=(f>=lo)&(f<hi); lv=lambda M:M[:,m].mean(1)
    th=np.nanpercentile(lv(Btr[nor]),95)
    print(f"{lo}-{hi}: 정상 {np.mean(lv(Btr[nor])>th)*100:.0f}% · 고장 {np.mean(lv(Btr[flt])>th)*100:.0f}% · val {np.nanmean(lv(Bv[okv])>th)*100:.0f}% (n_val={okv.sum()})")
fig,ax=plt.subplots(3,1,figsize=(12,11),sharex=True)
for M,c,l in ((Btr[nor],"C0","정상 배경(훈련)"),(Btr[flt],"C1","고장 배경(훈련)"),(Bv[okv],"C2","val 배경(라벨 없음)")):
    ax[0].fill_between(f/1000,q(M,25),q(M,75),color=c,alpha=.15); ax[0].plot(f/1000,q(M,50),color=c,label=l)
ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_ylabel("PSD (g²/Hz)"); ax[0].set_title("충격 구간을 뺀 배경 PSD (중앙값, 25–75%)")
rv=db(q(Bv[okv],50),q(Btr[nor],50)); rf=db(q(Btr[flt],50),q(Btr[nor],50))
ax[1].plot(f/1000,rf,color="C1",label="훈련 고장/정상"); ax[1].plot(f/1000,rv,color="C2",label="val/훈련 정상"); ax[1].axhline(0,color="r",lw=.8); ax[1].legend(); ax[1].set_ylabel("dB"); ax[1].set_title("배경만: 정상 대비")
ra=db(q(Av,50),q(Atr[y==0],50)); raf=db(q(Atr[y==1],50),q(Atr[y==0],50))
ax[2].plot(f/1000,raf,color="C1",label="훈련 고장/정상 (충격 포함)"); ax[2].plot(f/1000,ra,color="C2",label="val/훈련 정상 (충격 포함)"); ax[2].axhline(0,color="r",lw=.8); ax[2].legend()
ax[2].set_ylabel("dB"); ax[2].set_xlabel("주파수 (kHz)"); ax[2].set_xlim(0,12.5); ax[2].set_title("전체(충격 포함): 정상 대비")
fig.tight_layout(); fig.savefig(S+"/valbg.png",dpi=115)

print("\n=== 분포 점검 (대역 평균 dB, g²/Hz) ===")
for lo,hi in ((3000,5000),(8000,10000)):
    m=(f>=lo)&(f<hi)
    for lab,M in (("정상",Btr[nor]),("고장",Btr[flt]),("val",Bv[okv])):
        v=10*np.log10(M[:,m].mean(1)); print(f"{lo}-{hi} {lab}: p5 {np.percentile(v,5):.1f} p25 {np.percentile(v,25):.1f} 중앙 {np.median(v):.1f} p75 {np.percentile(v,75):.1f} p95 {np.percentile(v,95):.1f} max {v.max():.1f}")
