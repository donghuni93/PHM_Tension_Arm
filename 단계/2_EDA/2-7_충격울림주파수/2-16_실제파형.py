# %% [markdown]
# 2-16 울림 주파수 — 실제 파형으로 보기 (4-3에서 옮김)
# 파일마다 ① 2초 파형(1 kHz 고역통과) + 충격 위치(울림 대역 색) ② 가장 큰 충격 10 ms 확대 + 맞춘 감쇠 사인파
#           ③ 그 충격 8 ms의 스펙트럼 (2–5 / 5–8 / 8–11 kHz 음영)
# 충격 검출·울림 주파수는 2-16_run.py(신효섭 정의)와 같다. 라벨은 파일 선택에만 썼다
# 실행: conda run -n qe-phm python "단계/2_EDA/2-7_충격울림주파수/2-16_실제파형.py"

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks
from src.exp import load_train
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000

X, y, names = load_train()
short = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
idx = {s: i for i, s in enumerate(short)}
SHOW = [("nor_019", "전형적인 정상"), ("nor_025", "EXP-025 오경보"), ("nor_116", "EXP-025 오경보"),
        ("abn_010", "잘 잡히는 고장"), ("abn_022", "EXP-025 미탐"), ("abn_005", "EXP-025 경계")]
SHOW = [(f, d) for f, d in SHOW if f in idx]

X1k = sosfiltfilt(butter(4, 1000, btype="highpass", fs=FS, output="sos"), X - X.mean(1, keepdims=True), axis=-1)
lp500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")
SEG = int(0.008 * FS); t = np.arange(SEG) / FS
fgrid = np.arange(1000, 12001, 250); zgrid = [0.01, 0.02, 0.04, 0.08]
atoms, par = [], []
for f in fgrid:
    for z in zgrid:
        w = 2 * np.pi * f; wd = w * np.sqrt(1 - z ** 2); a = np.exp(-z * w * t)
        Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1))
        atoms.append(Q); par.append((f, z))
atoms = np.stack(atoms)


def band_color(f):
    return "#d62728" if 2000 <= f < 5000 else "#4472c4" if 5000 <= f < 8000 else "#2ca02c" if f >= 8000 else "#7f7f7f"


fig, axs = plt.subplots(len(SHOW), 3, figsize=(18, 2.9 * len(SHOW)), gridspec_kw={"width_ratios": [2.6, 1.3, 1.3]})
for r, (fname, desc) in enumerate(SHOW):
    i = idx[fname]; x = X1k[i]
    env = sosfiltfilt(lp500, np.abs(hilbert(x)))
    pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
    pk = pk[pk + SEG <= len(x)]
    S = np.stack([x[p:p + SEG] for p in pk]) if len(pk) else np.zeros((0, SEG))
    nrm = np.linalg.norm(S, axis=1, keepdims=True) + 1e-12
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S / nrm, atoms), axis=2) if len(pk) else np.zeros((0, len(par)))
    best = fit.argmax(1); fr = np.array([par[b][0] for b in best])
    n25 = np.mean((fr >= 2000) & (fr < 5000)) if len(fr) else np.nan

    a0 = axs[r, 0]; tt = np.arange(len(x)) / FS
    a0.plot(tt, x, lw=0.4, color="#999")
    for p, f in zip(pk, fr):
        a0.plot(p / FS, np.abs(x).max() * 1.05, "v", ms=6, color=band_color(f))
    a0.set_title(f"{fname} ({desc}) — 충격 {len(pk)}개, 2–5 kHz로 울린 비율 ring_2_5k = {n25:.2f}", loc="left", fontsize=10)
    a0.set_xlim(0, 2); a0.set_ylabel("g (1 kHz 고역통과)")

    a1, a2 = axs[r, 1], axs[r, 2]
    if len(pk):
        k = np.argmax(np.abs(S).max(1)); p = pk[k]; f_hat, z_hat = par[best[k]]
        lo = max(0, p - int(0.001 * FS)); seg = x[lo:p + int(0.009 * FS)]
        ts = (np.arange(len(seg)) + lo - p) / FS * 1000
        a1.plot(ts, seg, lw=1, color="#333", label="실제 신호")
        Q = atoms[best[k]]; coef = Q.T @ S[k]
        a1.plot(np.arange(SEG) / FS * 1000, Q @ coef, "--", lw=1.2, color=band_color(f_hat), label=f"감쇠 사인파 {f_hat / 1000:.2f} kHz")
        a1.set_title(f"가장 큰 충격 확대 — {f_hat / 1000:.2f} kHz로 울림", loc="left", fontsize=10)
        a1.set_xlabel("충격 후 시간 (ms)"); a1.legend(fontsize=8, loc="upper right")
        P = np.abs(np.fft.rfft(S[k] * np.hanning(SEG), 4096)) ** 2; ff = np.fft.rfftfreq(4096, 1 / FS)
        for (b0, b1), c in (((2000, 5000), "#d62728"), ((5000, 8000), "#4472c4"), ((8000, 11000), "#2ca02c")):
            a2.axvspan(b0 / 1000, b1 / 1000, color=c, alpha=0.08)
        a2.plot(ff / 1000, P / P.max(), color="#333", lw=1)
        a2.axvline(f_hat / 1000, color=band_color(f_hat), ls="--")
        a2.set_xlim(0, 12.5); a2.set_xlabel("주파수 (kHz)"); a2.set_title("그 충격 8 ms의 스펙트럼 (최대 = 1)", loc="left", fontsize=10)
    else:
        a1.text(0.5, 0.5, "충격 없음", ha="center", transform=a1.transAxes); a2.axis("off")

hand = [plt.Line2D([], [], marker="v", ls="", color=c, label=l) for c, l in
        (("#d62728", "2–5 kHz로 울림 (고장 쪽)"), ("#4472c4", "5–8 kHz로 울림 (정상 쪽)"), ("#2ca02c", "8 kHz 이상"), ("#7f7f7f", "2 kHz 미만"))]
fig.legend(handles=hand, loc="upper right", ncol=4, fontsize=9)
fig.suptitle("울림(ring) = 충격을 받은 구조물이 고유진동수로 떨리다 잦아드는 진동. ▼ 색 = 그 충격이 울린 대역", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig(os.path.join(FIG, "2-16_5_실제파형.png"), dpi=110); plt.close(fig)
print("saved")
