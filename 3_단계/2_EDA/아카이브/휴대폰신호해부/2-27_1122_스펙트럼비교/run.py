# 2-23 (11-22) 정상 · 고장 · val · 전체 스펙트럼 (시각화만)
# 평균 제거 → Welch PSD (nperseg 4096 ≈ 6.1 Hz, Hann, 50% 겹침) → 10·log10 (dB re 1 g²/Hz). 필터 없음
import os, sys
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import numpy as np
from scipy.signal import welch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp import load_train
from src.data import load_folder
plt.rcParams["font.family"] = "WenQuanYi Zen Hei" if os.name != "nt" else "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FS = 25000
X, y, _ = load_train()
vd = os.path.join(_d, "2_data", "TeamProject_val", "val"); vd = vd if os.path.isdir(vd) else os.path.dirname(vd)
Xv, _ = load_folder(vd)
def psd(A):
    A = np.asarray(A, float); f, P = welch(A - A.mean(1, keepdims=True), fs=FS, nperseg=4096, axis=-1)
    return f, 10 * np.log10(P + 1e-20)
f, Pn = psd(X[y == 0]); _, Pa = psd(X[y == 1]); _, Pv = psd(Xv)
Pall = np.vstack([Pn, Pa, Pv]); k = f / 1000
INK, MUTED = "#1d2430", "#5b6575"
G = [("정상 (train 120)", Pn, "#2a78d6"), ("고장 (train 30)", Pa, "#eb6834"), ("val (40)", Pv, "#8a4fc9")]
lo, hi = np.percentile(Pall, 0.5), np.percentile(Pall, 99.9) + 3
fig, axs = plt.subplots(2, 2, figsize=(16, 10), sharex=True, sharey=True)
for ax, (nm, P, c) in zip(axs.ravel()[:3], G):
    for p in P: ax.plot(k, p, color=c, lw=.4, alpha=.18)
    q25, q50, q75 = np.percentile(P, [25, 50, 75], axis=0)
    ax.fill_between(k, q25, q75, color=c, alpha=.25, lw=0, label="25–75%")
    ax.plot(k, q50, color=INK, lw=2, label="중앙값")
    ax.set_title(nm, fontsize=13, color=INK, loc="left"); ax.legend(fontsize=9, loc="upper right", frameon=False)
ax = axs[1, 1]
for p in Pall: ax.plot(k, p, color="#9aa0aa", lw=.3, alpha=.15)
for nm, P, c in G: ax.plot(k, np.median(P, 0), color=c, lw=2, label=f"{nm} 중앙값")
ax.plot(k, np.median(Pall, 0), color=INK, lw=1.4, ls="--", label="전체 중앙값")
ax.set_title(f"전체 ({len(Pall)}개)", fontsize=13, color=INK, loc="left"); ax.legend(fontsize=9, loc="upper right", frameon=False)
for ax in axs.ravel():
    ax.set_xlim(0, 12.5); ax.set_ylim(lo, hi); ax.grid(alpha=.25, lw=.6)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)
for ax in axs[1]: ax.set_xlabel("주파수 (kHz)", color=MUTED)
for ax in axs[:, 0]: ax.set_ylabel("PSD (dB re 1 g²/Hz)", color=MUTED)
fig.suptitle("파워 스펙트럼 — 평균 제거 · Welch 4096점 · 필터 없음 (가는 선 = 파일 하나)", fontsize=12, color=MUTED, x=.01, ha="left")
fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(os.path.join(HERE, "스펙트럼_4패널.png"), dpi=130); plt.close(fig)
print("saved")

# 평균 스펙트럼 한 장 (선형 파워를 평균한 뒤 dB로)
mdb = lambda P: 10 * np.log10(np.mean(10 ** (P / 10), 0))
fig, ax = plt.subplots(figsize=(14, 6.5))
for nm, P, c in G: ax.plot(k, mdb(P), color=c, lw=1.8, label=nm)
ax.plot(k, mdb(Pall), color=INK, lw=1.3, ls="--", label=f"전체 ({len(Pall)})")
ax.set_xlim(0, 12.5); ax.grid(alpha=.25, lw=.6); ax.legend(fontsize=10, frameon=False, loc="upper right")
for s in ("top", "right"): ax.spines[s].set_visible(False)
ax.tick_params(colors=MUTED); ax.set_xlabel("주파수 (kHz)", color=MUTED); ax.set_ylabel("평균 PSD (dB re 1 g²/Hz)", color=MUTED)
ax.set_title("평균 파워 스펙트럼 — 평균 제거 · Welch 4096점 · 필터 없음 · 파일들의 선형 파워 평균 → dB", fontsize=11, color=MUTED, loc="left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "스펙트럼_평균.png"), dpi=130); plt.close(fig)
print("saved mean")

# 평균 스펙트럼 · 깔끔한 판 (선형 파워 평균 → 100 Hz 이동평균으로 다듬기 → dB, 선 끝에 이름)
from scipy.ndimage import uniform_filter1d
win = max(1, int(round(100 / (f[1] - f[0]))))
sm = lambda P: 10 * np.log10(uniform_filter1d(np.mean(10 ** (P / 10), 0), win, mode="nearest"))
S = [(nm.split(" (")[0], sm(P), c, 2.4, "-") for nm, P, c in G] + [("전체", sm(Pall), "#8a8f99", 1.6, "--")]
fig, ax = plt.subplots(figsize=(13, 6))
m = k <= 12.4
for nm, s, c, lw, ls in S:
    ax.plot(k[m], s[m], color=c, lw=lw, ls=ls, solid_capstyle="round")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=c, lw=lw, ls=ls, label=nm) for nm, s, c, lw, ls in S], ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0),
          frameon=False, fontsize=10.5, handlelength=2.2, columnspacing=1.8, borderaxespad=0)
lo_, hi_ = min(s[m].min() for _, s, *_ in S), max(s[m][k[m] > .05].max() for _, s, *_ in S)
ax.set_xlim(0, 12.4); ax.set_ylim(np.floor(lo_ / 5) * 5 - 2, np.ceil(hi_ / 5) * 5 + 2); ax.set_xticks(range(0, 13, 1))
ax.grid(axis="y", alpha=.3, lw=.6); ax.grid(axis="x", alpha=.12, lw=.6)
for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
ax.spines["left"].set_color("#c9ced6"); ax.spines["bottom"].set_color("#c9ced6"); ax.tick_params(colors=MUTED, length=0)
ax.set_xlabel("주파수 (kHz)", color=MUTED); ax.set_ylabel("평균 PSD (dB re 1 g²/Hz)", color=MUTED)
ax.set_title("평균 파워 스펙트럼", fontsize=14, color=INK, loc="left", pad=48)
ax.text(0, 1.085, "정상 120 · 고장 30 · val 40 · 전체 190 — 파일별 Welch PSD(4096점)를 선형 파워로 평균, 100 Hz 이동평균, 필터 없음",
        transform=ax.transAxes, fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=.07, right=.98, top=.82, bottom=.11); fig.savefig(os.path.join(HERE, "스펙트럼_평균_정리.png"), dpi=150); plt.close(fig)
print("saved clean")

# 평균 스펙트럼 · 큰 파일에 덜 끌리는 판 (왼쪽 dB 평균 = 기하평균, 오른쪽 중앙값). dB 값에 100 Hz 이동평균
smd = lambda v: uniform_filter1d(v, win, mode="nearest")
VER = [("dB 평균 (기하평균)", lambda P: P.mean(0)), ("중앙값", lambda P: np.median(P, 0))]
fig, axs = plt.subplots(1, 2, figsize=(17, 6.2), sharey=True)
for ax, (tt, fn) in zip(axs, VER):
    S2 = [(nm.split(" (")[0], smd(fn(P)), c, 2.4, "-") for nm, P, c in G] + [("전체", smd(fn(Pall)), "#8a8f99", 1.6, "--")]
    for nm, s, c, lw, ls in S2: ax.plot(k[m], s[m], color=c, lw=lw, ls=ls)
    ax.set_xlim(0, 12.4); ax.set_xticks(range(0, 13)); ax.grid(axis="y", alpha=.3, lw=.6); ax.grid(axis="x", alpha=.12, lw=.6)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    ax.spines["left"].set_color("#c9ced6"); ax.spines["bottom"].set_color("#c9ced6"); ax.tick_params(colors=MUTED, length=0)
    ax.set_xlabel("주파수 (kHz)", color=MUTED); ax.set_title(tt, fontsize=13, color=INK, loc="left")
axs[0].set_ylabel("PSD (dB re 1 g²/Hz)", color=MUTED)
fig.legend(handles=[Line2D([], [], color=c, lw=lw, ls=ls, label=nm) for nm, s, c, lw, ls in S2], ncol=4, loc="upper left", bbox_to_anchor=(.055, .905),
           frameon=False, fontsize=10.5, handlelength=2.2, columnspacing=1.8)
fig.suptitle("파워 스펙트럼 — 큰 파일 몇 개에 끌리지 않는 두 가지 요약", fontsize=14, color=INK, x=.055, y=.985, ha="left")
fig.text(.055, .925, "정상 120 · 고장 30 · val 40 · 전체 190 — 파일별 Welch PSD(4096점)를 dB로 바꾼 뒤 요약, 100 Hz 이동평균, 필터 없음", fontsize=9.5, color=MUTED)
fig.subplots_adjust(left=.055, right=.99, top=.80, bottom=.1, wspace=.06); fig.savefig(os.path.join(HERE, "스펙트럼_dB평균_중앙값.png"), dpi=150); plt.close(fig)
print("saved robust")
