S='/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad'
import sys, numpy as np, pandas as pd
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic","Noto Sans CJK KR","WenQuanYi Zen Hei"):
    if any(f==x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False
from scipy.signal import spectrogram
from src.data import load_dataset
FS=25000
X,y,names=load_dataset("/home/user/PHM_Tension_Arm/data/TeamProject_train_변경")
I=pd.read_csv("/home/user/PHM_Tension_Arm/3_단계/2_EDA/유의미/2-16_1116_울림주파수/figs/impacts.csv",encoding="utf-8-sig")
byfile={k:v.t.values for k,v in I.groupby("file")}
Pall,Pbg,keep=[],[],[]
for i,n in enumerate(names):
    x=X[i]-X[i].mean()
    f,t,Sg=spectrogram(x,fs=FS,nperseg=2048,noverlap=1024,mode="psd")
    imp=byfile.get(n,np.array([]))
    ok=np.ones(len(t),bool)
    for ti in imp: ok&=~((t+0.041>ti-0.005)&(t-0.041<ti+0.100))
    Pall.append(Sg.mean(1))
    if ok.sum()>=5: Pbg.append(Sg[:,ok].mean(1)); keep.append(i)
    else: Pbg.append(np.full(len(f),np.nan))
Pall=np.array(Pall); Pbg=np.array(Pbg); ok=np.isfinite(Pbg[:,0])
print("배경 프레임 5개 이상 파일: 정상 %d/120 · 고장 %d/30"%((ok&(y==0)).sum(),(ok&(y==1)).sum()))
q=lambda A,p:np.nanpercentile(A,p,axis=0)
fig,ax=plt.subplots(3,1,figsize=(12,11),sharex=True)
for A,c,l in ((Pbg[ok&(y==0)],"C0","정상"),(Pbg[ok&(y==1)],"C1","고장")):
    ax[0].fill_between(f/1000,q(A,25),q(A,75),color=c,alpha=.2); ax[0].plot(f/1000,q(A,50),color=c,label=f"{l} 배경 구간 ({len(A)}파일)")
ax[0].set_yscale("log"); ax[0].legend(); ax[0].set_ylabel("PSD (g²/Hz)"); ax[0].set_title("충격 구간을 뺀 배경 PSD (중앙값, 25–75%)")
rb=10*np.log10(q(Pbg[ok&(y==1)],50)/q(Pbg[ok&(y==0)],50))
ra=10*np.log10(q(Pall[y==1],50)/q(Pall[y==0],50))
ax[1].plot(f/1000,ra,color="0.6",label="전체(충격 포함)"); ax[1].plot(f/1000,rb,color="k",label="배경만"); ax[1].axhline(0,color="r",lw=.8); ax[1].legend()
ax[1].set_ylabel("고장/정상 (dB)"); ax[1].set_title("주파수별 비율: 충격을 빼도 남는가?")
# 같은 파일 안: 충격 포함 전체 / 배경 (충격이 얼마나 더하나)
r_n=10*np.log10(q(Pall[ok&(y==0)]/Pbg[ok&(y==0)],50)); r_f=10*np.log10(q(Pall[ok&(y==1)]/Pbg[ok&(y==1)],50))
ax[2].plot(f/1000,r_n,color="C0",label="정상"); ax[2].plot(f/1000,r_f,color="C1",label="고장"); ax[2].axhline(0,color="r",lw=.8); ax[2].legend()
ax[2].set_ylabel("전체/배경 (dB)"); ax[2].set_xlabel("주파수 (kHz)"); ax[2].set_xlim(0,12.5); ax[2].set_title("같은 파일에서 충격이 PSD를 얼마나 올리나 (파일별 비의 중앙값)")
fig.tight_layout(); fig.savefig(S+"/bg100.png",dpi=115)
for lo,hi in ((100,1000),(1000,2000),(2000,4000),(4000,6000),(6000,8000),(8000,9000),(9000,12500)):
    m=(f>=lo)&(f<hi); print(f"{lo}-{hi}: 배경 고장/정상 {rb[m].mean():+.1f} dB | 전체 {ra[m].mean():+.1f} | 충격 기여 정상 {r_n[m].mean():+.1f} 고장 {r_f[m].mean():+.1f}")
