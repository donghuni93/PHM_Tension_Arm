# 2-24 (11-22) 단발 충격 파일: 정상(037 059 065 039 …)과 고장(002 016)의 충격 하나가 어떻게 다른가 — "단발이면 정상" 규칙을 만들 수 있는지 진단
# 질문: 충격이 ≤ 2개인 파일에서 정상 오경보(037 059 065)와 고장(002 016)을 가르는 충격 모양이 있는가 (울림 주파수 · 감쇠 · 스펙트럼 · 크기 · 시점 · 평균 오프셋)
# 라벨을 보고 문턱을 정하지 않는다: 이 진단은 가르는 신호가 있는지만 본다. 새 피처 후보가 나오면 새 EXP에서 사전 정의로 평가
# 실행: conda run -n qe-phm python "단계/2_EDA/아카이브/2-24_1122_단발충격_정상vs고장/run.py"
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, welch, find_peaks
from src.exp import load_train
from src.final import impact_rings
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; nor = y == 0
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 1000, "highpass", fs=FS, output="sos"), Xc, axis=-1)
lp = butter(4, 500, "lowpass", fs=FS, output="sos")
R = impact_rings(X); fi = R[:, 0].astype(int); nimp = np.bincount(fi, minlength=len(y))
mean_g = X.mean(1)

rows, SEG = [], {}
for i in range(len(y)):
    if nimp[i] == 0 or nimp[i] > 2: continue
    env = sosfiltfilt(lp, np.abs(hilbert(H[i]))); base = np.median(env)
    pk, _ = find_peaks(env, height=6 * base, distance=int(0.005 * FS)); pk = pk[pk + int(0.008 * FS) <= len(H[i])]
    for k, p in enumerate(pk):
        a0 = max(0, p - int(0.002 * FS)); seg = H[i, a0: p + int(0.016 * FS)]           # 충격 전 2 ms ~ 후 16 ms
        f, P = welch(H[i, p:p + int(0.016 * FS)], fs=FS, nperseg=256)
        Pn = P / P.sum(); bands = [Pn[(f >= a) & (f < b)].sum() for a, b in ((1000, 2000), (2000, 4000), (4000, 8000), (8000, 12500))]
        cen = float((f * Pn).sum()); rms_post = float(np.sqrt((H[i, p + int(0.008 * FS): p + int(0.016 * FS)] ** 2).mean())); rms_base = float(np.sqrt(np.median(H[i] ** 2)))
        dec = 0
        while p + dec < len(env) and env[p + dec] > 2 * base: dec += 1
        rr = R[fi == i][min(k, (fi == i).sum() - 1)]
        rows.append(dict(file=names[i], label="고장" if y[i] else "정상", n_impacts=int(nimp[i]), t_s=round(p / FS, 3), amp_g=round(float(np.abs(H[i, p:p + 200]).max()), 3),
                         ring_f_Hz=int(rr[1]), ring_zeta=rr[2], ring_fit=round(float(rr[3]), 2), centroid_Hz=int(cen), b1_2k=round(bands[0], 2), b2_4k=round(bands[1], 2), b4_8k=round(bands[2], 2), b8_12k=round(bands[3], 2),
                         decay_ms=round(dec / FS * 1000, 1), post_over_base=round(rms_post / (rms_base + 1e-12), 2), mean_g=round(float(mean_g[i]), 2)))
        SEG[(names[i], k)] = (seg, f, Pn, p - a0)
T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "단발충격_표.csv"), index=False, encoding="utf-8-sig")
one = T[T.n_impacts == 1]
fault = T[T.label == "고장"]; hot = T[T.file.isin(["nor_037", "nor_059", "nor_065", "nor_039"])]
print("충격 ≤ 2개 파일 수: 정상", int(T[T.label == '정상'].file.nunique()), "고장", int(fault.file.nunique()))
print("\n[고장]\n", fault.drop(columns=["label"]).to_string(index=False))
print("\n[우리 오경보 단발 정상]\n", hot.drop(columns=["label"]).to_string(index=False))
cols = ["amp_g", "ring_f_Hz", "ring_fit", "centroid_Hz", "b1_2k", "b2_4k", "b4_8k", "b8_12k", "decay_ms", "post_over_base", "t_s"]
print("\n[정상 단발(충격 수 1) 분포: 최소 / 중앙 / 최대]\n", one[one.label == "정상"][cols].describe().loc[["min", "50%", "max"]].round(2).to_string())
print("\n정상 단발 파일의 울림 주파수 분포:", one[one.label == "정상"].ring_f_Hz.value_counts().sort_index().to_dict())
print("고장(002, 016) 울림:", fault[["file", "ring_f_Hz", "ring_zeta", "ring_fit"]].values.tolist())

# ---- 그림: 충격 파형(충격 전 2 ms ~ 후 16 ms, 1 kHz 고역)과 충격 구간 스펙트럼, 정상 단발(회색 + 우리 오경보 파랑) vs 고장(주황)
fig, axs = plt.subplots(2, 2, figsize=(17, 10))
for (nm, k), (seg, f, Pn, off) in SEG.items():
    i = names.index(nm)
    if nm in ("abn_002", "abn_016"): c, lw, a_, z = "#eb6834", 2.2, 1, 5
    elif nm in ("nor_037", "nor_059", "nor_065", "nor_039"): c, lw, a_, z = "#2a78d6", 1.8, .95, 4
    elif nimp[i] == 1 and y[i] == 0: c, lw, a_, z = "#b0b0b0", .8, .6, 1
    else: continue
    tt = (np.arange(len(seg)) - off) / FS * 1000
    axs[0, 0].plot(tt, seg / (np.abs(seg).max() + 1e-12), color=c, lw=lw, alpha=a_, zorder=z); axs[0, 1].plot(tt, seg, color=c, lw=lw, alpha=a_, zorder=z)
    axs[1, 0].semilogy(f, Pn, color=c, lw=lw, alpha=a_, zorder=z)
    if nm in ("abn_002", "abn_016", "nor_037", "nor_059", "nor_065", "nor_039"): axs[1, 0].text(f[np.argmax(Pn)], Pn.max(), nm[-3:], color=c, fontsize=9, weight="bold")
for nm in ("abn_002", "abn_016", "nor_037", "nor_059", "nor_065", "nor_039"):
    pass
# 크기 vs 울림 주파수 산점
ax = axs[1, 1]
g1 = one[one.label == "정상"]; ax.scatter(g1.ring_f_Hz, g1.amp_g, color="#b0b0b0", s=40, label="정상 단발")
h1 = hot.groupby("file").first().reset_index(); ax.scatter(h1.ring_f_Hz, h1.amp_g, color="#2a78d6", s=80, label="우리 오경보 단발 정상 (037 059 065 039)")
ax.scatter(fault.ring_f_Hz, fault.amp_g, color="#eb6834", s=110, marker="D", label="고장 단발 (002 016)")
for r in pd.concat([h1, fault]).itertuples(): ax.annotate(r.file[-3:], (r.ring_f_Hz, r.amp_g), xytext=(5, 5), textcoords="offset points", fontsize=9)
ax.set_xlabel("충격 울림 주파수 (사전 맞춤, Hz)"); ax.set_ylabel("충격 크기 (1 kHz 고역, g)"); ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=9)
axs[0, 0].set_title("충격 파형 (진폭 정규화, 충격 전 2 ms ~ 후 16 ms)", fontsize=11); axs[0, 1].set_title("충격 파형 (g)", fontsize=11); axs[1, 0].set_title("충격 뒤 16 ms 스펙트럼 (전체 합 1로 정규화)", fontsize=11); axs[1, 1].set_title("단발 파일의 충격 크기 × 울림 주파수", fontsize=11)
axs[0, 0].set_xlabel("ms"); axs[0, 1].set_xlabel("ms"); axs[1, 0].set_xlabel("Hz"); axs[0, 1].set_ylim(-.6, .6)
for a in axs.ravel(): a.grid(alpha=.3)
fig.suptitle("단발 충격 파일 진단 — 회색 = 정상 단발(충격 수 1), 파랑 = 우리 오경보 단발 정상, 주황 = 고장 단발", x=.01, ha="left"); fig.tight_layout(rect=(0, 0, 1, .96))
fig.savefig(os.path.join(FIG, "단발충격_비교.png"), dpi=110); plt.close(fig)
print("done")
