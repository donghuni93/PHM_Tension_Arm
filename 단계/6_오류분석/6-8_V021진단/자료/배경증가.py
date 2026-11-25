# V021 진단 보조 — 파일 후반/전반 배경 rms 비(1 kHz 고역, 0.1 s 창 20개). 결과 = 배경증가.txt
import sys, os
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d); os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.signal import butter, sosfiltfilt
from src.exp2util import NS
ns = NS(); sos = butter(4, 1000, btype="highpass", fs=25000, output="sos")
def r(x):
    h = sosfiltfilt(sos, x - x.mean()); w = h.reshape(20, -1).std(1); return w[10:].mean() / w[:10].mean(), w.max() / np.median(w)
R = np.array([r(x) for x in ns.X]); Rv = np.array([r(x) for x in ns.Xv]); y = ns.y; L = []
for nm, m in (("train 정상", y == 0), ("train 고장", y == 1)):
    v = R[m, 0]; L.append(f"{nm}: 후반/전반 rms 중앙 {np.median(v):.2f} · p95 {np.percentile(v, 95):.2f} · 최대 {v.max():.2f} · ≥1.5 {int((v >= 1.5).sum())}개")
k = [i for i, v in enumerate(ns.nv) if v.endswith("021")][0]
L.append(f"V021 후반/전반 {Rv[k, 0]:.2f} → train 정상 백분위 {(R[y == 0, 0] < Rv[k, 0]).mean() * 100:.0f} · 고장 백분위 {(R[y == 1, 0] < Rv[k, 0]).mean() * 100:.0f}")
L.append(f"val 중앙 {np.median(Rv[:, 0]):.2f} · val ≥1.5: " + ", ".join(f"{ns.nv[i][-3:]}({Rv[i, 0]:.2f})" for i in np.where(Rv[:, 0] >= 1.5)[0]))
L.append("train ≥1.5: " + ", ".join(f"{ns.names[i]}({R[i, 0]:.2f})" for i in np.where(R[:, 0] >= 1.5)[0]))
open("배경증가.txt", "w", encoding="utf-8").write("\n".join(L))
