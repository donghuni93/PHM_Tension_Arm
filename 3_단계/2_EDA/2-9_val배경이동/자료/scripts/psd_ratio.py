S='/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad'
import sys, numpy as np
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic","Noto Sans CJK KR","WenQuanYi Zen Hei"):
    if any(f==x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False
from scipy.signal import welch
from src.data import load_dataset
FS=25000
X,y,names=load_dataset("/home/user/PHM_Tension_Arm/data/TeamProject_train_변경")
f,P=welch(X-X.mean(1,keepdims=True),fs=FS,nperseg=2048,axis=-1)
Pn,Pf=P[y==0],P[y==1]
q=lambda A,p:np.percentile(A,p,axis=0)
mn,mf=q(Pn,50),q(Pf,50)
ratio=10*np.log10(mf/mn)
# 파일별 총 파워(100 Hz 이상) 비교
tot=P[:,f>=100].sum(1)
print("총 파워(100Hz+) 중앙값 정상 %.3g 고장 %.3g 비 %.2f"%(np.median(tot[y==0]),np.median(tot[y==1]),np.median(tot[y==1])/np.median(tot[y==0])))
for lo,hi in ((100,1000),(1000,2000),(2000,4000),(4000,6000),(6000,8000),(8000,9000),(9000,12500)):
    m=(f>=lo)&(f<hi); print(f"{lo}-{hi} Hz: 고장/정상 = {10*np.log10(np.median(Pf[:,m].mean(1))/np.median(Pn[:,m].mean(1))):+.1f} dB")
fig,ax=plt.subplots(2,1,figsize=(12,8),sharex=True)
for A,c,l in ((Pn,"C0","정상 120"),(Pf,"C1","고장 30")):
    ax[0].fill_between(f/1000,q(A,25),q(A,75),color=c,alpha=.2); ax[0].plot(f/1000,q(A,50),color=c,label=l+" (중앙값, 25–75%)")
ax[0].set_yscale("log"); ax[0].set_ylabel("PSD (g²/Hz)"); ax[0].legend(); ax[0].set_title("파일별 Welch PSD의 중앙값")
ax[1].plot(f/1000,ratio,color="k",lw=.9); ax[1].axhline(0,color="r",lw=.8)
for k in (7,8.3): ax[1].axvline(k,color="C2",ls=":",lw=.8)
ax[1].set_ylabel("고장/정상 (dB)"); ax[1].set_xlabel("주파수 (kHz)"); ax[1].set_xlim(0,12.5)
ax[1].set_title("주파수별 비율 (전 대역이 같은 양이면 수평선 = 이득 차이, 굴곡이 있으면 특정 대역 변화)")
fig.tight_layout(); fig.savefig(S+"/psd_ratio.png",dpi=115)
