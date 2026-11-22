# 8 kHz 대 에너지가 '지속음'인가 '충격 울림'인가: 0.1 s 프레임 STFT로 7.8–9.0 kHz 봉우리 추적 (V001 vs train 8.3k 규칙 파일 6개 · train 정상 전체)
# 실행: conda run -n qe-phm python V001_지속음.py → V001_지속음.txt (8.3k 규칙 값 비교는 V001_규칙값.txt, 같은 날 scratch 스크립트로 계산)
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); R = HERE
while not os.path.isdir(os.path.join(R, "src")) and os.path.dirname(R) != R:
    R = os.path.dirname(R)
sys.path.insert(0, R)
from scipy.signal import stft
from scipy.ndimage import median_filter
from src.exp2util import NS
FS = 25000
ns = NS(); y = ns.y; names = list(ns.names); nv = [v[-3:] for v in ns.nv]
def track(x, lo, hi):
    f, t, Z = stft(x - x.mean(), fs=FS, nperseg=2500, noverlap=1250)       # 0.1 s 창, 10 Hz 해상도
    L = 10 * np.log10(np.abs(Z) ** 2 + 1e-20); base = median_filter(L, size=(41, 1), mode="nearest")   # 주파수 방향 ±200 Hz 중앙값 대비
    b = (f >= lo) & (f <= hi); P = (L - base)[b]; fb = f[b]
    k = P.argmax(0); prom = P.max(0); fr = fb[k]
    on = prom >= 10                                                         # 10 dB 넘게 솟은 프레임 = 좁은 지속음이 있는 프레임
    slope = np.polyfit(t[on], fr[on], 1)[0] if on.sum() >= 5 else np.nan   # Hz/s
    return on.mean(), float(np.median(fr[on])) if on.any() else np.nan, float(np.median(prom)), slope
out = ["파일 | 8 kHz 대(7.8–9.0k) 좁은 지속음 프레임 비율 · 중앙 주파수 · 중앙 돌출(dB) · 주파수 기울기(Hz/s) | 절반 대(3.9–4.5k) 프레임 비율 · 중앙 주파수"]
def row(lab, x):
    a = track(x, 7800, 9000); b = track(x, 3900, 4500)
    return f"{lab} | {a[0]:.2f} · {a[1]:.0f} Hz · {a[2]:.1f} dB · {a[3]:+.0f} | {b[0]:.2f} · {b[1]:.0f} Hz", a, b
k = nv.index("001"); s, a1, b1 = row("V001", ns.Xv[k]); out.append(s)
for nm in ["abn_002", "abn_003", "abn_006", "abn_010", "abn_029", "abn_030"]:
    out.append(row(nm, ns.X[names.index(nm)])[0])
A = [(names[i], track(ns.X[i], 7800, 9000), track(ns.X[i], 3900, 4500)) for i in range(len(y))]
fracN = np.array([a[1][0] for a, yy in zip(A, y) if yy == 0]); fracA = np.array([a[1][0] for a, yy in zip(A, y) if yy == 1])
out.append(f"\ntrain 정상 120: 8 kHz 대 지속음 프레임 비율 ≥ 0.8인 파일 {int((fracN >= .8).sum())} · 중앙 {np.median(fracN):.2f} | 고장 30: ≥ 0.8 {int((fracA >= .8).sum())} · 중앙 {np.median(fracA):.2f}")
both = [(n_, a[0], a[1], b[0], b[1]) for n_, a, b in A if a[0] >= .8 and b[0] >= .8]
out.append("train에서 8 kHz 대와 절반 대 모두 지속음(≥ 0.8)인 파일: " + (", ".join(f"{n_}({fa:.0f}·{fb:.0f} Hz, 비 {fa / fb:.2f})" for n_, _, fa, _, fb in both) or "없음"))
top = sorted(A, key=lambda r: -r[1][0])[:8]
out.append("train 8 kHz 대 지속음 비율 상위 8: " + ", ".join(f"{n_}({a[0]:.2f}, {a[1]:.0f} Hz, {a[3]:+.0f} Hz/s)" for n_, a, _ in top))
vals = [(f"V{nv[i]}", track(ns.Xv[i], 7800, 9000), track(ns.Xv[i], 3900, 4500)) for i in range(40)]
out.append("val 8 kHz 대 지속음 ≥ 0.8: " + ", ".join(f"{n_}({a[0]:.2f}, {a[1]:.0f} Hz, {a[3]:+.0f} Hz/s · 절반 {b[0]:.2f} {b[1]:.0f} Hz)" for n_, a, b in vals if a[0] >= .8))
open(os.path.join(HERE, "V001_지속음.txt"), "w", encoding="utf-8").write("\n".join(out))
