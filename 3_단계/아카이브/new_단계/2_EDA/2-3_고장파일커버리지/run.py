# %% 2-3 고장 30개 파일 커버리지: 이벤트 수 + 울림 대역별 이벤트 수 (2-2의 events.csv 사용)
# 라벨을 본 탐색이다. 규칙 문턱을 고르는 것이 아니라, 미리 적어 둔 몇 가지 단순 규칙이 파일을 얼마나 덮는지 본다.
import glob, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "2_data")):
    ROOT = os.path.dirname(ROOT)
EV = os.path.join(ROOT, "3_단계", "아카이브", "new_단계", "2_EDA", "2-2_충격이벤트", "figs")
D = os.path.join(ROOT, "2_data", "TeamProject_train_변경")
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)

R = pd.read_csv(os.path.join(EV, "files.csv")); E = pd.read_csv(os.path.join(EV, "events.csv"))
BANDS = [("1.0-1.6k", 1000, 1600), ("1.6-4.5k", 1600, 4500), ("4.5-5.5k", 4500, 5500),
         ("5.5-8k", 5500, 8000), ("8-11k", 8000, 11000), (">=11k", 11000, 12501)]
for n, a, b in BANDS:
    cnt = E[(E.f >= a) & (E.f < b)].groupby("file").size()
    R[n] = R.file.map(cnt).fillna(0).astype(int)
R["n_ev"] = R.n_ev25
R["off"] = R["1.6-4.5k"] + R["8-11k"]          # 정상 울림 대역(5.5-8k) 밖 + 연결부 · 고주파 대역
R["max_x"] = R.file.map(E.groupby("file").x.max()).fillna(0)

# 파일별 대역 PSD (3–5 kHz, 8–10.5 kHz), 정상 120 안에서의 백분위
def psd_bands(group):
    out = {}
    for f in sorted(glob.glob(os.path.join(D, group, "*.csv"))):
        x = np.loadtxt(f); x -= x.mean()
        fr, p = welch(x, fs=25000, window="hann", nperseg=5000, noverlap=2500)
        out[os.path.basename(f)[:-4]] = (10 * np.log10(p[(fr >= 3000) & (fr < 5000)].mean()),
                                         10 * np.log10(p[(fr >= 8000) & (fr < 10500)].mean()),
                                         10 * np.log10(p[(fr >= 5500) & (fr < 8000)].mean()))
    return out
P = {**psd_bands("normal"), **psd_bands("abnormal")}
R["psd35"] = R.file.map(lambda k: P[k][0]); R["psd8105"] = R.file.map(lambda k: P[k][1]); R["psd558"] = R.file.map(lambda k: P[k][2])
nor = R[R.group == "normal"]
for c in ("psd35", "psd8105", "psd558"):
    R[c + "_pct"] = R[c].map(lambda v: 100 * (nor[c] < v).mean())
R.to_csv(os.path.join(HERE, "figs", "files_cover.csv"), index=False)

L = []
abn = R[R.group == "abnormal"].sort_values(["n_ev", "off"]).copy()
L.append("1) 고장 30개: 이벤트 수와 대표 울림 대역별 이벤트 수 (이벤트 수 오름차순)")
L.append("file     n_det n_ev | 1.0-1.6 1.6-4.5 4.5-5.5 5.5-8 8-11 >=11 | off(1.6-4.5+8-11) | max배수 | PSD3-5k 정상백분위 | PSD8-10.5k 정상백분위")
for _, r in abn.iterrows():
    L.append(f"{r['file']} {int(r['n_det']):5d} {int(r['n_ev']):4d} | {int(r['1.0-1.6k']):6d} {int(r['1.6-4.5k']):7d} {int(r['4.5-5.5k']):7d} {int(r['5.5-8k']):5d} {int(r['8-11k']):4d} {int(r['>=11k']):4d} | {int(r['off']):5d} | {r['max_x']:6.1f} | {r['psd35_pct']:5.0f} | {r['psd8105_pct']:5.0f}")
L.append("\n2) 단순 규칙이 덮는 파일 (정상 120 · 고장 30)")
rules = {"n_ev >= 3": lambda d: d.n_ev >= 3, "off >= 1": lambda d: d.off >= 1, "off >= 2": lambda d: d.off >= 2,
         "n_ev >= 3 & off >= 1": lambda d: (d.n_ev >= 3) & (d.off >= 1), "n_ev >= 3 & off >= 2": lambda d: (d.n_ev >= 3) & (d.off >= 2),
         "n_ev >= 3 | off >= 1": lambda d: (d.n_ev >= 3) | (d.off >= 1)}
nor, ab = R[R.group == "normal"], R[R.group == "abnormal"]
for k, f in rules.items():
    L.append(f"{k:24s} 고장 {f(ab).sum():2d}/30 | 정상 {f(nor).sum():3d}/120 ({f(nor).mean()*100:4.1f} %)")
L.append("\n3) off >= 1을 놓친 고장 (정상 울림 대역에만 울린 고장)")
miss = ab[ab.off < 1]
for _, r in miss.iterrows():
    L.append(f"{r['file']}: n_ev {int(r['n_ev'])}, 5.5-8k {int(r['5.5-8k'])}개, 1.6-4.5k {int(r['1.6-4.5k'])}개, 8-11k {int(r['8-11k'])}개, max배수 {r['max_x']:.1f}, PSD3-5k pct {r['psd35_pct']:.0f}, PSD8-10.5k pct {r['psd8105_pct']:.0f}")
L.append("\n4) 이벤트 ≤ 2개인 고장")
for _, r in ab[ab.n_ev <= 2].iterrows():
    L.append(f"{r['file']}: 이벤트 {int(r['n_ev'])}, 검출 {int(r['n_det'])}, 시점/울림 " + "; ".join(f"{e.t:.3f}s {int(e.f)}Hz x{e.x:.1f}" for e in E[E.file == r['file']].itertuples())
             + f" | PSD3-5k pct {r['psd35_pct']:.0f}, PSD8-10.5k pct {r['psd8105_pct']:.0f}, PSD5.5-8k pct {r['psd558_pct']:.0f}")
L.append("\n5) 정상 120의 같은 열 분포 (중앙 · p90 · 0개 비율)")
for c in ["n_ev", "off"] + [b[0] for b in BANDS]:
    v = nor[c]; L.append(f"{c:9s} 중앙 {v.median():4.1f} · p90 {v.quantile(.9):4.1f} · 0개 {np.mean(v==0)*100:3.0f} %")
open(os.path.join(HERE, "figs", "커버리지_수치.txt"), "w", encoding="utf-8").write("\n".join(L))

# %% 그림: 고장 30개 파일 × 대역별 이벤트 수 (쌓은 막대)
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COL = ["#e87ba4", "#2a78d6", "#1baf7a", "#eda100", "#eb6834", "#9a9a95"]   # 1-1.6, 1.6-4.5, 4.5-5.5, 5.5-8, 8-11, >=11
fig, ax = plt.subplots(figsize=(11, 9), facecolor=SURF)
ax.set_facecolor(SURF)
d = abn.iloc[::-1]
left = np.zeros(len(d))
for (n, a, b), c in zip(BANDS, COL):
    v = d[n].values
    ax.barh(range(len(d)), v, left=left, color=c, edgecolor=SURF, lw=1.2, label=n.replace("k", " kHz"))
    left += v
ax.set_yticks(range(len(d))); ax.set_yticklabels(d.file.str.replace("abn_", "고장 "), fontsize=9, color=INK2)
ax.set_xlabel("2초 안의 충격 이벤트 수 (대표 울림 대역별 색)", color=INK2)
ax.grid(True, axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
for s in ("top", "right"): ax.spines[s].set_visible(False)
for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
for i, r in enumerate(d.itertuples()):
    ax.annotate(f"{r.n_ev}", (left[i], i), xytext=(4, 0), textcoords="offset points", fontsize=8.5, color=INK, va="center")
ax.set_title("고장 30개: 이벤트 수와 울림 대역 (위 = 이벤트 적음, 아래 = 많음)", loc="left", color=INK, fontsize=12)
ax.legend(frameon=False, ncol=6, loc="lower right", fontsize=9, labelcolor=INK2)
fig.text(0.01, 0.005, "Train 고장 30개, 2-2의 이벤트(25 ms 묶음)와 대표 울림(50 Hz 격자). 정상의 대표 울림은 5.5–8 kHz(노랑)가 73 %.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.02, 1, 1))
fig.savefig(os.path.join(HERE, "figs", "고장30_커버리지.png"), dpi=140, facecolor=SURF)
print("done")
