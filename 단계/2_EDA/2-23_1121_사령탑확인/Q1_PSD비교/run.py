# %% [markdown]
# Q1 정상 vs 고장 PSD 중앙값 비교 (사령탑 확인 요청, 11-21)
# 질문: 전체 중앙값과 고장·정상 PSD가 어떻게 다른가. val은 어느 쪽 모양인가
# 방법 (라벨은 색칠·비교에만, 대역을 고르지 않는다):
#   - 각 파일 평균(중력 오프셋) 제거 → Welch, 50% 겹침, Hann
#   - 전 대역: 2048점 (칸 12.2 Hz) / 0–200 Hz: 8192점 (칸 3 Hz, detrend 끔)
#   - 정상 120개의 5%·중앙·95% 선(로그 PSD 기준), 고장 30 중앙, val 40 중앙
#   - 차이 곡선 = (고장 중앙 - 정상 중앙) dec, dec = log10 비율 (1 dec = 10배)
#   - 히트맵 = 파일 로그 PSD - 정상 95%선, 0 이하는 0 (정상 띠 안), 행 = 고장 30 + val 40
# 실행: conda run -n qe-phm python run.py  (숫자는 result.txt, 그림은 figs/)

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()

def find_root(start):
    p = os.path.abspath(start)
    while not os.path.isdir(os.path.join(p, "src")):
        q = os.path.dirname(p)
        if q == p:
            raise RuntimeError("src/ 폴더를 찾지 못함")
        p = q
    return p

ROOT = find_root(HERE)
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import welch, find_peaks, butter, sosfiltfilt
from src.exp import load_train
from src.data import load_folder

plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False
FS = 25000
NOR, ABN, VAL = "#2a78d6", "#eb6834", "#2a9d5c"
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
lines = []
def log(*a): lines.append(" ".join(str(v) for v in a))

X, y, names = load_train(); names = [str(n) for n in names]; y = np.asarray(y)
XV, vnames = load_folder(os.path.join(ROOT, "data", "TeamProject_val", "val"))
log(f"train {X.shape} (정상 {(y==0).sum()}, 고장 {(y==1).sum()}), val {XV.shape}")
nor, abn = y == 0, y == 1

def logpsd(S, nperseg, detrend):
    S = S - S.mean(1, keepdims=True)
    f, P = welch(S, fs=FS, nperseg=nperseg, noverlap=nperseg // 2, window="hann",
                 detrend=detrend, axis=-1)
    return f, np.log10(P + 1e-20)

def summarize(L):
    return {"lo": np.percentile(L[nor], 5, axis=0), "med": np.median(L[nor], axis=0),
            "hi": np.percentile(L[nor], 95, axis=0)}

def spans(mask, f):
    """True가 이어지는 주파수 구간 목록"""
    out, s = [], None
    for i, m in enumerate(mask):
        if m and s is None: s = i
        if (not m or i == len(mask) - 1) and s is not None:
            e = i if m else i - 1
            out.append((f[s], f[e])); s = None
    return out

def set_xticks(ax, xlog, xlim):
    """x축 눈금을 촘촘하게: 로그 = 1·2·3·5·7 배수, 선형 = 0–200 Hz는 10 Hz, 그 밖은 500 Hz 간격"""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
    if xlog:
        t = [v * 10 ** k for k in range(1, 5) for v in (1, 2, 3, 5, 7) if xlim[0] <= v * 10 ** k <= xlim[1]]
        ax.xaxis.set_major_locator(FixedLocator(t))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v/1000:g}k" if v >= 1000 else f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
    else:
        span = xlim[1] - xlim[0]
        step = 10 if xlim[1] <= 200 else (250 if span <= 6000 else 500)
        ax.set_xticks(np.arange(np.ceil(xlim[0] / step) * step, xlim[1] + 1, step))
        ax.tick_params(axis="x", labelsize=8, labelrotation=90 if step >= 250 else 0)
    ax.grid(alpha=.3, which="major", axis="x")

def curves_fig(f, Lt, Lv, band, xlog, xlim, title, fname):
    """위: 정상 중앙·5–95% 띠 + 고장 중앙 + val 중앙. 아래: 차이 곡선(dec)"""
    m = (f >= xlim[0]) & (f <= xlim[1])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.5), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    a1.fill_between(f[m], band["lo"][m], band["hi"][m], color=NOR, alpha=.18, label="정상 5–95% 띠")
    a1.plot(f[m], band["med"][m], color=NOR, lw=1.6, label="정상 중앙 (120)")
    am, vm = np.median(Lt[abn], 0), np.median(Lv, 0)
    a1.plot(f[m], am[m], color=ABN, lw=1.6, label="고장 중앙 (30)")
    a1.plot(f[m], vm[m], color=VAL, lw=1.6, ls="--", label="val 중앙 (40, 라벨 없음)")
    a1.set_ylabel("log10 PSD (g²/Hz)"); a1.set_title(title); a1.legend(ncol=2); a1.grid(alpha=.3, which="both")
    a2.axhline(0, color="k", lw=.8)
    a2.fill_between(f[m], band["lo"][m] - band["med"][m], band["hi"][m] - band["med"][m],
                    color=NOR, alpha=.18, label="정상 5–95% 띠 (정상 중앙 기준)")
    a2.plot(f[m], (am - band["med"])[m], color=ABN, lw=1.4, label="고장 중앙 - 정상 중앙")
    a2.plot(f[m], (vm - band["med"])[m], color=VAL, lw=1.4, ls="--", label="val 중앙 - 정상 중앙")
    a2.set_ylabel("차이 (dec)"); a2.set_xlabel("주파수 (Hz)"); a2.legend(ncol=1, fontsize=7, loc="lower right"); a2.grid(alpha=.3, which="both")
    if xlog: a1.set_xscale("log")
    a1.set_xlim(*xlim)
    set_xticks(a2, xlog, xlim)
    a1.tick_params(labelbottom=True)
    # 고장-정상 차이 곡선의 봉우리(상위 6)를 Hz로 적는다
    d = (am - band["med"])[m]; ff = f[m]
    ds = np.convolve(d, np.ones(3) / 3, mode="same")
    pk, pr = find_peaks(ds, prominence=0.15, distance=max(2, len(ff) // 60))
    pk = pk[np.argsort(-ds[pk])[:6]]
    items = []
    for k, i in enumerate(sorted(pk), 1):
        for a in (a1, a2):
            a.axvline(ff[i], color=ABN, lw=.7, ls=":", alpha=.8)
        a2.plot(ff[i], d[i], "o", ms=11, mfc="white", mec=ABN, zorder=5)
        a2.text(ff[i], d[i], str(k), ha="center", va="center", fontsize=7, color=ABN, fontweight="bold", zorder=6)
        items.append(f"{k}  {ff[i]:.0f} Hz  {d[i]:+.1f} dec")
        log(f"[{fname}] 차이 곡선 봉우리 {k}: {ff[i]:.0f} Hz, {d[i]:+.2f} dec")
    if items:
        a2.text(0.01, 0.97, "고장-정상 차이 봉우리" + chr(10) + chr(10).join(items), transform=a2.transAxes, va="top", ha="left",
                fontsize=8, color=ABN, bbox=dict(boxstyle="round", fc="white", ec=ABN, alpha=.9), zorder=7)
    a2.set_ylim(top=max(a2.get_ylim()[1], d.max() + 0.5))
    fig.savefig(os.path.join(FIG, fname + ".png"), dpi=90, bbox_inches="tight"); plt.close(fig)

def heat_fig(f, Lt, Lv, band, xlim, title, fname):
    """행 = 고장 30 + val 40, 열 = 주파수. 값 = 정상 95%선을 넘은 dec (0 = 띠 안)"""
    m = (f >= xlim[0]) & (f <= xlim[1])
    Ea = np.clip(Lt[abn] - band["hi"], 0, None)[:, m]
    Ev = np.clip(Lv - band["hi"], 0, None)[:, m]
    E = np.vstack([Ea, Ev])
    rows = [n for n, a in zip(names, abn) if a] + vnames
    fig, ax = plt.subplots(figsize=(12, 14))
    im = ax.imshow(E, aspect="auto", cmap="magma_r", vmin=0, vmax=1.0,
                   extent=[f[m][0], f[m][-1], len(rows), 0], interpolation="nearest")
    ax.axhline(abn.sum(), color="k", lw=1.5)
    ax.set_yticks(np.arange(len(rows)) + .5); ax.set_yticklabels(rows, fontsize=6)
    set_xticks(ax, False, xlim)
    ax.set_xlabel("주파수 (Hz)"); ax.set_title(title + "\n(위 30행 = 고장, 검은 선 아래 40행 = val)")
    fig.colorbar(im, ax=ax, fraction=.025, label="정상 95%선을 넘은 크기 (dec, 1 이상은 같은 색)")
    fig.savefig(os.path.join(FIG, fname + ".png"), dpi=90, bbox_inches="tight"); plt.close(fig)

# %% 전 대역 (2048점)
f2, Lt2 = logpsd(X, 2048, "constant"); _, Lv2 = logpsd(XV, 2048, "constant")
b2 = summarize(Lt2)
curves_fig(f2, Lt2, Lv2, b2, True, (20, 12500), "전 대역 PSD (Welch 2048점, 칸 12.2 Hz, 로그 x)", "q1_fullband")
heat_fig(f2, Lt2, Lv2, b2, (0, 12500), "정상 95%선을 넘는 주파수 — 전 대역 (2048점)", "q1_heat_fullband")

curves_fig(f2, Lt2, Lv2, b2, False, (6000, 11000), "6–11 kHz 확대 (Welch 2048점, 선형 x)", "q1_zoom_6to11k")

# %% 고주파 (1 kHz 고역통과 4차 Butterworth, 앞뒤 두 번 = 위상 이동 없음) — 선형 x
sos = butter(4, 1000, btype="highpass", fs=FS, output="sos")
fh, Lth = logpsd(sosfiltfilt(sos, X, axis=-1), 2048, "constant")
_, Lvh = logpsd(sosfiltfilt(sos, XV, axis=-1), 2048, "constant")
bh = summarize(Lth)
curves_fig(fh, Lth, Lvh, bh, False, (1000, 12500), "고주파 PSD (1 kHz 고역통과 후, Welch 2048점, 선형 x)", "q1_high1k")
heat_fig(fh, Lth, Lvh, bh, (1000, 12500), "정상 95%선을 넘는 주파수 — 1 kHz 고역통과 후 (2048점)", "q1_heat_high1k")

# %% 0–200 Hz (8192점, detrend 끔)
f8, Lt8 = logpsd(X, 8192, False); _, Lv8 = logpsd(XV, 8192, False)
b8 = summarize(Lt8)
curves_fig(f8, Lt8, Lv8, b8, False, (3, 200), "0–200 Hz PSD (Welch 8192점, 칸 3 Hz, 선형 x)", "q1_low200")
heat_fig(f8, Lt8, Lv8, b8, (3, 200), "정상 95%선을 넘는 주파수 — 0–200 Hz (8192점)", "q1_heat_low200")

# %% 숫자
def report(tag, f, Lt, Lv, band, lo_hz):
    m = f >= lo_hz
    am, vm = np.median(Lt[abn], 0), np.median(Lv, 0)
    d_a, d_v = am - band["med"], vm - band["med"]
    log(f"\n=== {tag} ===")
    log("고장 중앙이 정상 95%선 위인 구간 (Hz):", [(round(a), round(b)) for a, b in spans((am > band["hi"]) & m, f)])
    log("고장 중앙이 정상 5%선 아래인 구간 (Hz):", [(round(a), round(b)) for a, b in spans((am < band["lo"]) & m, f)])
    log("val 중앙이 정상 95%선 위인 구간 (Hz):", [(round(a), round(b)) for a, b in spans((vm > band["hi"]) & m, f)])
    log("val 중앙이 정상 5%선 아래인 구간 (Hz):", [(round(a), round(b)) for a, b in spans((vm < band["lo"]) & m, f)])
    k = np.argsort(-d_a * m)[:5]
    log("고장-정상 차이가 큰 칸 상위 5 (Hz, dec):", [(round(float(f[i])), round(float(d_a[i]), 2)) for i in k])
    log("val-정상 차이 절댓값 최대 칸 (Hz, dec):", (round(float(f[m][np.argmax(np.abs(d_v[m]))])), round(float(d_v[m][np.argmax(np.abs(d_v[m]))]), 2)))
    # 파일별: 정상 95%선을 넘는 칸 비율. 정상 자신(자기 포함 기준)도 같이 재서 기준선으로 쓴다
    def frac(L): return ((L > band["hi"]) & m).sum(1) / m.sum()
    fn, fa, fv = frac(Lt[nor]), frac(Lt[abn]), frac(Lv)
    log(f"95%선 초과 칸 비율 — 정상 중앙 {np.median(fn):.3f} (최대 {fn.max():.3f}), 고장 중앙 {np.median(fa):.3f}, val 중앙 {np.median(fv):.3f}")
    thr = fn.max()
    log(f"정상 최대({thr:.3f})보다 비율이 큰 파일: 고장 {(fa > thr).sum()}/30, val {(fv > thr).sum()}/40")
    log("val에서 비율이 큰 상위 8:", [(vnames[i], round(float(fv[i]), 3)) for i in np.argsort(-fv)[:8]])
    log("고장에서 비율이 작은 하위 5 (정상 띠 안에 숨은 고장):",
        [(n, round(float(v), 3)) for n, v in sorted(zip([n for n, a in zip(names, abn) if a], fa), key=lambda t: t[1])[:5]])

report("전 대역 2048점 (20 Hz 이상)", f2, Lt2, Lv2, b2, 20)
report("1 kHz 고역통과 후 2048점 (1.2 kHz 이상)", fh, Lth, Lvh, bh, 1200)
report("0–200 Hz 8192점 (3 Hz 이상)", f8, Lt8, Lv8, b8, 3)
with open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("done")
