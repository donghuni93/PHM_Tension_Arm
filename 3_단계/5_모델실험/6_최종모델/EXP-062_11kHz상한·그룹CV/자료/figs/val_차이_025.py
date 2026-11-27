import re, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic"):
    if any(f==x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"]=f; break
plt.rcParams["axes.unicode_minus"]=False
rows=[]
for l in open("val_판정.md",encoding="utf-8").read().split("## ")[0].split("전체")[-1].splitlines():
    c=[x.strip() for x in l.split("|")]
    if len(c)>8 and c[1].isdigit(): rows.append((c[1],float(c[5]),float(c[7]),c[2],c[4]))
assert len(rows)==40
fig,(a,b)=plt.subplots(1,2,figsize=(14,6),gridspec_kw=dict(width_ratios=[1.15,1]))
BL,OR="#2a6fbb","#e0622d"
for n,m1,m3,j1,j3 in rows:
    d=j1!=j3
    a.scatter(m1,m3,s=110 if d else 45,marker="^",c=OR if j3=="고장" else BL,edgecolor="k",lw=1.4 if d else .5,zorder=3)
    if d or abs(m1)<1 or abs(m3)<1 or m3>0: a.annotate("V"+n,(m1,m3),(5,5),textcoords="offset points",fontsize=9,fontweight="bold" if d else "normal")
lo,hi=-10,4.2
a.plot([lo,hi],[lo,hi],c="gray",ls=":",lw=1)
a.axhline(0,c="k",ls="--",lw=1);a.axvline(0,c="k",ls="--",lw=1)
a.fill_betweenx([0,hi],lo,0,color=OR,alpha=.18,zorder=0)
a.text(-9.7,3.4,"EXP-062만 고장",color=OR,fontsize=10,fontweight="bold")
a.set_xlim(lo,hi);a.set_ylim(-5,hi)
a.set_xlabel("현재 최종 모델(제출 파일) 여유 = 결합 점수 - 문턱");a.set_ylabel("EXP-062 (11 kHz + 그룹 CV) 여유")
a.set_title("val 40개: 두 모델의 여유 (점 색 = EXP-062 판정, 0 위 = 고장)",loc="left",fontsize=11)
a.grid(alpha=.25)
# 오른쪽: 경계 근처 파일 막대
near=sorted(rows,key=lambda r:min(abs(r[1]),abs(r[2])))[:10]
near=sorted(near,key=lambda r:r[2])
y=range(len(near))
b.barh([i+.2 for i in y],[r[1] for r in near],.38,color="#9aa5b1",label="현재 최종 모델")
b.barh([i-.2 for i in y],[r[2] for r in near],.38,color=[OR if r[2]>0 else BL for r in near],label="EXP-062")
b.set_yticks(list(y));b.set_yticklabels(["V"+r[0] for r in near]);b.axvline(0,c="k",ls="--")
for i,r in zip(y,near):
    if r[3]!=r[4]: b.annotate("판정 다름",(max(r[1],r[2])+.1,i),fontsize=9,color=OR,fontweight="bold",va="center")
b.set_xlabel("여유 (0 = 문턱)");b.set_title("경계에 가장 가까운 10개",loc="left",fontsize=11);b.legend(loc="lower right");b.grid(alpha=.25,axis="x")
fig.suptitle("val 판정이 다른 파일은 V025 하나 (최종 모델 정상 -0.09 → EXP-062 고장 +1.08). 나머지 39개는 같은 판정, 고장 9개 공통",x=.01,ha="left",fontsize=11)
fig.tight_layout(rect=(0,0,1,.94));fig.savefig("figs/val_차이_025.png",dpi=140)
