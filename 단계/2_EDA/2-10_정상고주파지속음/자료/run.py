# 2-24 (11-22) 정상 평균 스펙트럼의 8–10 kHz 봉우리는 어디서 오나 (시각화)
# 평균 제거 → Welch PSD 4096점(≈6.1 Hz). 대역 몫 = 정상 120개의 그 대역 선형 파워 합 중 파일 하나의 비율
import os, sys, json, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np
from scipy.signal import welch, spectrogram
from scipy.ndimage import uniform_filter1d
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from src.exp import load_train
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000; INK, MUTED, GRID = "#1d2430", "#5b6575", "#c9ced6"
NOR, ABN = "#2a78d6", "#eb6834"
X, y, names = load_train(); names = [str(n) for n in names]; nor = y == 0; nn = np.array(names)[nor]
fp = json.load(open(os.path.join(__import__("src.exp", fromlist=["find_exp"]).find_exp("EXP-044"), "result.json"), encoding="utf-8"))["configs"]["중첩: J | W + AE + E2 + 8.3k (최종 후보)"]["fp_counts"]
f, P = welch(X - X.mean(1, keepdims=True), fs=FS, nperseg=4096, axis=-1); k = f / 1000
Pn, Pa = P[nor], P[~nor]
win = max(1, int(round(100 / (f[1] - f[0])))); sm = lambda v: uniform_filter1d(v, win, mode="nearest"); db = lambda v: 10 * np.log10(v)
def style(ax):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["left"].set_color(GRID); ax.spines["bottom"].set_color(GRID); ax.tick_params(colors=MUTED, length=0); ax.grid(axis="y", alpha=.3, lw=.6)

# 대역 몫
B = (f >= 8000) & (f < 10500); e = Pn[:, B].sum(1); o = np.argsort(-e); share = e / e.sum()
TOP = [nn[i] for i in o[:3]]                                                          # 116 055 + 다음
DROP = ["nor_116", "nor_055", "nor_065"]
lines = [f"8–10.5 kHz 정상 에너지 몫: " + " · ".join(f"{nn[i][-3:]} {share[i]:.0%}" for i in o[:8]),
         f"평균/중앙값 (8–10.5 kHz): {10*np.log10(Pn[:, B].mean() / np.median(Pn[:, B].mean(1))):.1f} dB"]
for lo, hi in ((8.75, 9.05), (9.3, 9.7), (10.0, 10.4)):
    b = (f >= lo * 1e3) & (f < hi * 1e3); ee = Pn[:, b].sum(1); oo = np.argsort(-ee)
    lines.append(f"{lo}–{hi} kHz: " + " · ".join(f"{nn[i][-3:]} {ee[i] / ee.sum():.0%}" for i in oo[:4]))
open(os.path.join(HERE, "몫.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines))

# 그림 1: 정상 평균 · 중앙값 · 세 파일 뺀 평균 vs 고장 (7–12.4 kHz) + 파일별 몫 막대
m = (k >= 7) & (k <= 12.4)
keep = ~np.isin(nn, DROP)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw=dict(width_ratios=[2.1, 1]))
q25, q75 = np.percentile(Pa, [25, 75], axis=0)
a1.fill_between(k[m], db(sm(q25))[m], db(sm(q75))[m], color=ABN, alpha=.15, lw=0)
S = [("정상 평균 (120)", db(sm(Pn.mean(0))), NOR, 2.4, "-"), ("정상 평균, 116 · 055 · 065 제외 (117)", db(sm(Pn[keep].mean(0))), NOR, 2.0, (0, (5, 2.5))),
     ("정상 중앙값 (120)", db(sm(np.median(Pn, 0))), "#8fb6e8", 2.0, "-"), ("고장 중앙값 (30)", db(sm(np.median(Pa, 0))), ABN, 2.2, "-")]
for nm, s, c, lw, ls in S: a1.plot(k[m], s[m], color=c, lw=lw, ls=ls)
a1.legend(handles=[Line2D([], [], color=c, lw=lw, ls=ls, label=nm) for nm, s, c, lw, ls in S] + [plt.Rectangle((0, 0), 1, 1, color=ABN, alpha=.15, label="고장 25–75%")],
          frameon=False, fontsize=9.5, loc="upper right")
a1.set_xlim(7, 12.4); a1.set_xlabel("주파수 (kHz)", color=MUTED); a1.set_ylabel("PSD (dB re 1 g²/Hz)", color=MUTED); style(a1)
a1.set_title("정상 평균의 8–10 kHz 봉우리", fontsize=13, color=INK, loc="left")
top = o[:10]; lab = [f"{nn[i][-3:]}" + (f"  (J 오경보 {fp.get(nn[i], 0)}/5)" if fp.get(nn[i], 0) else "") for i in top]
a2.barh(range(10)[::-1], share[top] * 100, color=[NOR if fp.get(nn[i], 0) == 0 else "#c0392b" for i in top], height=.62)
a2.set_yticks(range(10)[::-1]); a2.set_yticklabels(lab, fontsize=10, color=INK)
for j, i in enumerate(top): a2.text(share[i] * 100 + 1, 9 - j, f"{share[i]:.0%}", va="center", fontsize=9.5, color=INK)
a2.set_xlim(0, max(share) * 100 + 12); a2.set_xlabel("8–10.5 kHz 정상 에너지 중 몫 (%)", color=MUTED); style(a2); a2.grid(axis="y", alpha=0); a2.grid(axis="x", alpha=.3, lw=.6)
a2.set_title("파일별 몫 (상위 10, 빨강 = J 오경보)", fontsize=13, color=INK, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림1_봉우리_출처.png"), dpi=140); plt.close(fig)

# 그림 2: 세 파일 각각 vs 정상 중앙값 · 고장 25–75%
F3 = ["nor_116", "nor_055", "nor_065"]
fig, axs = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, fn in zip(axs, F3):
    i = names.index(fn)
    ax.fill_between(k[m], db(sm(q25))[m], db(sm(q75))[m], color=ABN, alpha=.15, lw=0, label="고장 25–75%")
    ax.plot(k[m], db(sm(np.median(Pn, 0)))[m], color="#8fb6e8", lw=2, label="정상 중앙값")
    ax.plot(k[m], db(P[i])[m], color=NOR, lw=1.1, label=fn[-3:] + " (원 PSD)")
    ax.set_xlim(7, 12.4); ax.set_xlabel("주파수 (kHz)", color=MUTED); style(ax); ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title(f"정상 {fn[-3:]} · 8–10.5 kHz 몫 {share[list(nn).index(fn)]:.0%} · J 오경보 {fp.get(fn, 0)}/5", fontsize=11.5, color=INK, loc="left")
axs[0].set_ylabel("PSD (dB re 1 g²/Hz)", color=MUTED)
fig.tight_layout(); fig.savefig(os.path.join(HERE, "그림2_세파일_스펙트럼.png"), dpi=140); plt.close(fig)

# 그림 3: 스펙트로그램 (시간 × 6–12.5 kHz) — 세 파일 + 비교용 고장 하나(8–10.5 kHz 에너지 최대 고장)
ea = Pa[:, B].sum(1); fa = np.array(names)[~nor][np.argmax(ea)]
F4 = F3 + [fa]
fig, axs = plt.subplots(len(F4), 1, figsize=(15, 3.1 * len(F4)), sharex=True)
for ax, fn in zip(axs, F4):
    x = X[names.index(fn)]; x = x - x.mean()
    fs_, ts, Sxx = spectrogram(x, fs=FS, nperseg=512, noverlap=384)
    mm_ = fs_ >= 6000; Z = 10 * np.log10(Sxx[mm_] + 1e-20)
    ax.pcolormesh(ts, fs_[mm_] / 1000, Z, shading="auto", cmap="magma", vmin=np.percentile(Z, 5), vmax=np.percentile(Z, 99.8))
    for fr in (8.9, 9.5, 10.2): ax.axhline(fr, color="white", lw=.5, ls=":", alpha=.6)
    tag = "정상" if fn.startswith("nor") else "고장"
    ax.set_title(f"{tag} {fn[-3:]}", fontsize=11.5, color=INK, loc="left"); ax.set_ylabel("kHz", color=MUTED); ax.tick_params(colors=MUTED, length=0)
axs[-1].set_xlabel("시간 (s)", color=MUTED)
fig.suptitle("스펙트로그램 6–12.5 kHz (512점 창, 점선 = 8.9 · 9.5 · 10.2 kHz). 색은 파일마다 자기 범위", fontsize=11, color=MUTED, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(os.path.join(HERE, "그림3_스펙트로그램.png"), dpi=130); plt.close(fig)
print("비교 고장:", fa, "· saved")
