import sys, numpy as np
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic","AppleGothic","Noto Sans CJK KR","WenQuanYi Zen Hei"):
    if any(f == x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
from scipy.signal import butter, sosfiltfilt, hilbert
from src.data import load_dataset
FS=25000
X,y,names = load_dataset("/home/user/PHM_Tension_Arm/data/TeamProject_train_변경")
num=lambda n:int("".join(c for c in n if c.isdigit()))
find=lambda k,l: next(i for i,(n,t) in enumerate(zip(names,y)) if t==l and num(n)==k)
lp=butter(4,500,fs=FS,output="sos"); hp=butter(4,1000,btype="high",fs=FS,output="sos")
def env(x,hpf):
    x=x-x.mean()
    if hpf: x=sosfiltfilt(hp,x)
    return sosfiltfilt(lp,np.abs(hilbert(x)))
picks=[("정상",find(6,0)),("정상",find(46,0)),("고장",find(3,1)),("고장",find(22,1))]
t=np.arange(50000)/FS
fig,ax=plt.subplots(4,2,figsize=(14,10),sharex=True)
for r,(lab,i) in enumerate(picks):
    for c,h in enumerate((False,True)):
        e=env(X[i],h); med=np.median(e); thr=6*med
        a=ax[r,c]; a.plot(t,e,lw=.7,color="C0" if not h else "C3")
        a.axhline(thr,color="k",ls="--",lw=.8)
        pk=np.where((e[1:-1]>e[:-2])&(e[1:-1]>=e[2:])&(e[1:-1]>thr))[0]+1
        keep=[];
        for p in pk:
            if not keep or p-keep[-1]>125: keep.append(p)
        a.plot(t[keep],e[keep],"v",color="k",ms=5)
        a.set_title(f"{lab} {names[i]} · {'1 kHz 고역통과 후' if h else '원본'} · 문턱(중앙값×6) 넘은 봉우리 {len(keep)}개",fontsize=10)
        a.set_ylabel("포락선 (g)")
for a in ax[-1]: a.set_xlabel("시간 (s)")
fig.suptitle("충격 검출용 포락선: 원본(왼쪽) vs 1 kHz 고역통과 후(오른쪽), 점선 = 중앙값의 6배")
fig.tight_layout()
S="/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad"; out=f"{S}/env_pair.png"; fig.savefig(out,dpi=120); print(out)
for lab,i in picks:
    for h in (False,True):
        e=env(X[i],h); print(names[i],"HPF" if h else "raw ","최대/중앙값 = %.1f"%(e.max()/np.median(e)))
