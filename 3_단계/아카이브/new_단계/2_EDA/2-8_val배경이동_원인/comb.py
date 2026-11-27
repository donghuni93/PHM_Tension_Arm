# %% 빗살(고조파 열) 점검: 2.5–5.5 kHz 로그 PSD의 자기상관에서 일정 간격(80–400 Hz) 봉우리를 찾는다. 라벨 미사용(집단 구분만)
import glob, os
import numpy as np, pandas as pd
from scipy.signal import welch, find_peaks
from scipy.ndimage import median_filter
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "2_data")): ROOT = os.path.dirname(ROOT)
FIG = os.path.join(HERE, "figs"); FS = 25000
fl, grp = [], []
for g, p in (("normal", "TeamProject_train_변경/normal"), ("abnormal", "TeamProject_train_변경/abnormal"), ("val", "TeamProject_val/val")):
    s = sorted(glob.glob(os.path.join(ROOT, "2_data", p, "*.csv"))); fl += s; grp += [g] * len(s)
names = [os.path.basename(f)[:-4] for f in fl]; grp = np.array(grp)
X = np.stack([np.loadtxt(f) for f in fl]); X = X - X.mean(1, keepdims=True)
f, P = welch(X, fs=FS, nperseg=8192, average="median", detrend=False); df = f[1] - f[0]
LP = 10 * np.log10(P)
band = (f >= 2500) & (f < 5500)
R = LP[:, band] - median_filter(LP[:, band], size=(1, int(600 / df)), mode="nearest")   # 완만한 모양 제거 → 좁은 봉우리만
lags = np.arange(1, int(450 / df)); lag_hz = lags * df
ac = np.array([[np.corrcoef(r[:-k], r[k:])[0, 1] for k in lags] for r in R])
sel = (lag_hz >= 80) & (lag_hz <= 400)
best = ac[:, sel].max(1); best_hz = lag_hz[sel][ac[:, sel].argmax(1)]
D = pd.DataFrame(dict(file=names, group=grp, comb_r=best, comb_hz=best_hz))
D.to_csv(os.path.join(FIG, "comb.csv"), index=False)
L = ["빗살 점수 = 2.5–5.5 kHz 로그 PSD(600 Hz 이동 중앙값 제거)의 자기상관 최대값, 간격 80–400 Hz"]
thr = np.percentile(D.comb_r[D.group == "normal"], 95)
L.append(f"기준: Train 정상 95 % 분위 = {thr:.2f}")
for g in ("normal", "abnormal", "val"):
    d = D[D.group == g]; hi = d[d.comb_r > thr]
    L.append(f"{g:8s} 중앙 {d.comb_r.median():.2f} | 기준 초과 {len(hi)}/{len(d)} | 초과 파일 간격 중앙 {hi.comb_hz.median() if len(hi) else float('nan'):.0f} Hz")
    L.append("   " + ", ".join(f"{r.file}({r.comb_r:.2f},{r.comb_hz:.0f}Hz)" for _, r in hi.sort_values('comb_r', ascending=False).iterrows()))
open(os.path.join(FIG, "빗살_수치.txt"), "w", encoding="utf-8").write("\n".join(L))
# 그림: 빗살이 가장 강한 val 2개 + 정상 중앙 PSD
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
top = D[D.group == "val"].sort_values("comb_r", ascending=False).file.values[:2]
fig, ax = plt.subplots(1, 1, figsize=(12, 4.5), facecolor="#fcfcfb"); ax.set_facecolor("#fcfcfb")
m = (f >= 2000) & (f < 6000)
ax.plot(f[m] / 1000, np.median(LP[grp == "normal"], 0)[m], color="#2a78d6", lw=2, label="Train 정상 중앙")
for k, nm in enumerate(top):
    ax.plot(f[m] / 1000, LP[names.index(nm)][m], color=["#1baf7a", "#eb6834"][k], lw=1, label=nm)
ax.set_xlabel("주파수 (kHz)"); ax.set_ylabel("PSD (dB)"); ax.legend(frameon=False); ax.grid(True, color="#e6e5e1")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.set_title("빗살이 가장 강한 val 파일 2개 vs Train 정상 중앙 (2–6 kHz)", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "빗살.png"), dpi=140); print("done")
