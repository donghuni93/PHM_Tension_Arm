S='/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad'
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic","Noto Sans CJK KR","WenQuanYi Zen Hei"):
    if any(f==x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False
D=pd.read_csv("/home/user/PHM_Tension_Arm/3_단계/2_EDA/유의미/2-16_1116_울림주파수/figs/impacts.csv",encoding="utf-8-sig")
g=D.groupby("file").agg(y=("y","first"),n=("f","size"),med=("f","median"),iqr=("f",lambda s:s.quantile(.75)-s.quantile(.25)))
g=g[g.n>=3]
print(g.groupby("y").iqr.describe().round(0)); print("파일 수",g.groupby("y").size().to_dict())
fig,ax=plt.subplots(1,2,figsize=(15,7),gridspec_kw={"width_ratios":[2.3,1]})
for lab,col,off in ((0,"C0",0),(1,"C1",1)):
    order=g[g.y==lab].sort_values("med").index
    pos={f:i for i,f in enumerate(order)}; base=0 if lab==0 else 135
    d=D[D.file.isin(order)]
    ax[0].scatter([base+pos[f] for f in d.file],d.f/1000,s=6+np.sqrt(d.height)*4,c=col,alpha=.5,lw=0)
ax[0].axvline(130,color="k",lw=.8); ax[0].set_xticks([])
ax[0].set_ylabel("울림 주파수 (kHz)"); ax[0].set_ylim(0.8,12.3)
ax[0].set_title("파일 하나 = 세로 한 줄, 점 하나 = 충격 하나 (왼쪽 정상, 오른쪽 고장, 각각 중앙값 순)")
ax[0].text(5,12.0,"정상 (충격 3개 이상인 파일)",color="C0"); ax[0].text(137,12.0,"고장",color="C1")
for lab,col in ((0,"C0"),(1,"C1")):
    ax[1].hist(g[g.y==lab].iqr.clip(upper=6000),bins=np.arange(0,6200,250),color=col,alpha=.55,density=True,label=("정상" if lab==0 else "고장"))
ax[1].set_xlabel("파일 안 울림 주파수의 퍼짐 (IQR, Hz)"); ax[1].set_ylabel("파일 비율 밀도"); ax[1].legend(); ax[1].set_title("파일 안에서 얼마나 들쭉날쭉한가")
fig.tight_layout(); fig.savefig(f"{S}/strip.png",dpi=120)
