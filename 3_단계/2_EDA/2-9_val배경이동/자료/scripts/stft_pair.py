import os, sys, numpy as np
sys.path.insert(0, "/home/user/PHM_Tension_Arm")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
for f in ("Malgun Gothic","NanumGothic","AppleGothic","WenQuanYi Zen Hei","Noto Sans CJK KR"):
    if any(f == x.name for x in fm.fontManager.ttflist): plt.rcParams["font.family"] = f; break
plt.rcParams["axes.unicode_minus"] = False
from scipy.signal import stft
from src.data import load_dataset
FS = 25000
X, y, names = load_dataset("/home/user/PHM_Tension_Arm/data/TeamProject_train_변경")
num = lambda n: int("".join(c for c in n if c.isdigit()))
def find(k, lab): return next(i for i,(n,l) in enumerate(zip(names,y)) if l==lab and num(n)==k)
rms = X.std(1); nor = np.where(y==0)[0]
ref = nor[np.argsort(np.abs(rms[nor]-np.median(rms[nor])))[:2]]
picks = [("정상", ref[0]), ("정상", ref[1]), ("고장 (8.3k 공진형)", find(3,1)), ("고장 (놓치는 022)", find(22,1))]
def spec(i, nper=1024):
    x = X[i]-X[i].mean()
    f,t,Z = stft(x, fs=FS, nperseg=nper, noverlap=nper*3//4, boundary=None, padded=False)
    return f,t,10*np.log10(np.abs(Z)**2+1e-15)
S = [spec(i) for _,i in picks]
allv = np.concatenate([s[2][s[0]<=12500].ravel() for (lab,_),s in zip(picks,S) if lab=="정상"])
vmin, vmax = np.percentile(allv,1), np.percentile(allv,99)
fig, ax = plt.subplots(2,2, figsize=(13,8), sharex=True, sharey=True)
for a,(lab,i),(f,t,P) in zip(ax.ravel(), picks, S):
    m = a.pcolormesh(t, f/1000, P, vmin=vmin, vmax=vmax, shading="auto", cmap="magma")
    a.set_title(f"{lab}: {names[i]} (RMS {rms[i]:.3f} g)"); a.set_ylim(0,12.5)
    for k in (7, 8.3): a.axhline(k, color="cyan", lw=.6, ls=":")
for a in ax[1]: a.set_xlabel("시간 (s)")
for a in ax[:,0]: a.set_ylabel("주파수 (kHz)")
fig.colorbar(m, ax=ax, label="파워 (dB, 같은 색 범위)", shrink=.9)
fig.suptitle("STFT 정상 vs 고장 (창 1024점 41 ms, 점선 = 7 kHz · 8.3 kHz)")
out="/tmp/claude-0/-home-user-PHM-Tension-Arm/094be5bc-e118-5964-bddf-7cc4ac1183fd/scratchpad/stft_pair.png"
fig.savefig(out, dpi=130, bbox_inches="tight"); print(out, [n for _,i in picks for n in [names[i]]])
# 띠 여부: 주파수별 시간 변동(표준편차/평균) 요약
for (lab,i),(f,t,P) in zip(picks,S):
    lin = 10**(P/10); b = lambda lo,hi: lin[(f>=lo)&(f<hi)].mean(0)
    for nm,(lo,hi) in {"7k":(6500,7500),"8.3k":(8000,8700)}.items():
        e=b(lo,hi); print(lab, names[i], nm, "시간변동 CV=%.2f"%(e.std()/e.mean()), "상위5퍼센트 시간창 에너지비 %.2f"%(np.sort(e)[-len(e)//20:].sum()/e.sum()))
