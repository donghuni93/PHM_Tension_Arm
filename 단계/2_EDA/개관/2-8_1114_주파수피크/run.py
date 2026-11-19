# %% [markdown]
# 2-8 주파수 피크 위치 — 150개 파일의 PSD 피크가 어디에 있나, 고장 유형(데이터 기반 군집)과 관계는?
# 피크 정의 (그림 보기 전에 고정). PSD = 평균 제거 Welch 4096 (detrend 없음, 6.1 Hz 칸), 200 Hz 이상만
#  P1 최대 에너지 주파수: PSD argmax
#  P2 협대역 피크: L − 이동 중앙값(31칸 ≈ 190 Hz) = 돌출도(dec). 100 Hz 이상 떨어진 상위 3개
#  P3 정상 대비 초과 주파수: (L − 정상 120개 중앙값)을 9칸(≈55 Hz) 이동평균한 뒤 argmax
# 고장 유형: 라벨 없음 (1-3 합의: 유형별 라벨링 안 함). 참고 축으로만 2-3 군집(k=5), 1-3 예시 파일, 모델 미탐 여부를 붙인다
# 실행: conda run -n qe-phm python "단계/2_EDA/개관/2-8_1114_주파수피크/run.py"

# %% 설정
import os, sys, json, re
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train
import src.viz  # noqa: F401
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import butter, sosfiltfilt, welch, find_peaks
from scipy.ndimage import median_filter, uniform_filter1d

FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; N = len(y); nor = y == 0
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
Xc = X - X.mean(1, keepdims=True)
f, P = welch(Xc, fs=FS, nperseg=4096, detrend=False)
L = np.log10(P + 1e-30); df = f[1] - f[0]
m = f >= 200; fm = f[m]; Lm = L[:, m]
PROM = (L - median_filter(L, size=(1, 31), mode="nearest"))[:, m]
EXC = uniform_filter1d(Lm - np.median(Lm[nor], 0), 9, axis=1)

P1 = fm[Lm.argmax(1)]
P3 = fm[EXC.argmax(1)]; P3v = EXC.max(1)
P2 = []
for i in range(N):
    pk, _ = find_peaks(PROM[i], distance=int(100 / df))
    pk = pk[np.argsort(PROM[i][pk])[::-1]][:3]
    P2.append([(float(fm[k]), float(PROM[i][k])) for k in pk])

# %% 참고 축: 2-3 군집(k=5), 1-3 예시, 모델 미탐, 충격 수·저주파 몫
txt = open(os.path.join(ROOT, "단계", "2_EDA", "개관", "2-3_1113_고장모드", "figs", "2-3_fault_modes.txt"), encoding="utf-8").read()
k5 = re.search(r"k=5: (.+)", txt).group(1)
CL = {}
for part in k5.split("|"):
    mm = re.match(r"\s*\[(\d)\]\s*(.+)", part)
    for s in mm.group(2).split(","):
        CL[f"abn_{s.strip()}"] = int(mm.group(1))
EX13 = {**{f"abn_{s}": "①광대역충격" for s in ["004", "008"]}, **{f"abn_{s}": "②출렁임+충격" for s in ["005", "022", "028"]},
        **{f"abn_{s}": "③희소·위장" for s in ["014", "017", "021"]}, "abn_015": "④저주파공진"}
def counts(exp_dir):
    r = json.load(open(os.path.join(ROOT, "단계", "5_모델비교", exp_dir, "result.json"), encoding="utf-8"))
    c = next(iter(r["configs"].values())); return c["miss_counts"]
aem, etm = counts("base/EXP-012_1113_스펙트럼AE_tSNE"), counts("기각/EXP-007_1113_공선성제거")
MISS = {n: "+".join(k for k, d in [("ET", etm), ("AE", aem)] if d.get(n, 0) >= 3) for n in names}
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
nimp = []
for h in H:
    s = 1.4826 * np.median(np.abs(h - np.median(h))); idx = np.flatnonzero(np.abs(h) > 8 * s)
    nimp.append(0 if len(idx) == 0 else len(np.split(idx, np.flatnonzero(np.diff(idx) > 250) + 1)))
s20 = P[:, f < 20].sum(1) / P.sum(1)
hrms = H.std(1)

# %% 대역 집계: 250 Hz 칸마다 "그 칸에 P1·P2·P3가 있는 파일 비율" (정상 / 고장)
edges = np.arange(0, 12750, 250)
def frac_in(freqs_per_file, grp):
    out = np.zeros(len(edges) - 1)
    for i in np.flatnonzero(grp):
        hit = np.zeros(len(edges) - 1, bool)
        for fq in freqs_per_file[i]:
            hit[min(np.searchsorted(edges, fq, side="right") - 1, len(edges) - 2)] = True
        out += hit
    return out / grp.sum()
F1 = [[v] for v in P1]; F2 = [[fq for fq, pr in p if pr >= 0.5] for p in P2]; F3 = [[v] for v in P3]
lines = ["P2는 돌출도 0.5 dec(약 3배) 이상만"]
for nm, FF in [("P1 최대 에너지", F1), ("P2 협대역 피크", F2), ("P3 정상 대비 초과", F3)]:
    a, b = frac_in(FF, nor), frac_in(FF, ~nor)
    top = np.argsort(b - a)[::-1][:6]
    lines.append(f"[{nm}] 고장 − 정상 비율 차가 큰 250 Hz 칸: " + ", ".join(f"{edges[k]/1000:g}–{edges[k+1]/1000:g}k 고장 {b[k]*100:.0f}% / 정상 {a[k]*100:.0f}%" for k in top))
def share(FF, lo, hi, grp):
    return np.mean([any(lo <= v < hi for v in FF[i]) for i in np.flatnonzero(grp)])
for lo, hi in [(7500, 9000), (8000, 8500)]:
    lines.append(f"{lo/1000:g}–{hi/1000:g} kHz에 피크 있는 파일 비율 (고장 / 정상): " + " | ".join(
        f"{nm} {share(FF, lo, hi, ~nor)*100:.0f}% / {share(FF, lo, hi, nor)*100:.0f}%" for nm, FF in [("P1", F1), ("P2", F2), ("P3", F3)]))

# %% 고장 파일 표
rows = []
for i in np.flatnonzero(~nor):
    n = names[i]
    rows.append(dict(file=n, 군집=CL.get(n), 예시유형=EX13.get(n, ""), 미탐=MISS[n], 충격수=nimp[i], 저주파몫=round(s20[i], 2),
                     HP_RMS=round(hrms[i], 4), P1_Hz=round(P1[i]), P3_Hz=round(P3[i]), P3_dec=round(P3v[i], 2),
                     P2=" ".join(f"{fq/1000:.2f}k({pr:.1f})" for fq, pr in P2[i])))
T = pd.DataFrame(rows).sort_values(["군집", "P3_Hz"])
T.to_csv(os.path.join(HERE, "figs", "fault_peaks.csv"), index=False, encoding="utf-8-sig")
lines += ["", "고장 30개 (군집 = 2-3 k=5, P2 = 주파수(돌출도 dec))", T.to_string(index=False)]
# 군집별 요약
lines.append("")
lines.append("군집별: P3 중앙 Hz · 7.5–9 kHz에 P2(≥0.5)나 P3가 있는 비율 · 충격 수 중앙 · HP RMS 중앙")
for c in sorted(set(CL.values())):
    idx = [names.index(n) for n, v in CL.items() if v == c]
    has8 = np.mean([share([F2[i] + F3[i] for i in range(N)], 7500, 9000, np.isin(np.arange(N), [i])) for i in idx])
    lines.append(f"  군집 {c} ({len(idx)}개): P3 중앙 {np.median(P3[idx]):.0f} Hz · 8k대 {has8*100:.0f}% · 충격 {np.median(np.array(nimp)[idx]):.0f} · HP RMS {np.median(hrms[idx]):.4f}")
# 8 kHz대 피크 유무로 나눈 고장 특성
has8 = np.array([any(7500 <= v < 9000 for v in F2[i] + F3[i]) for i in range(N)])
lines.append("")
for g, grp in [("고장", ~nor), ("정상", nor)]:
    for h8 in [True, False]:
        idx = np.flatnonzero(grp & (has8 == h8))
        if len(idx) == 0: continue
        miss = sum(bool(MISS[names[i]]) for i in idx) if g == "고장" else None
        lines.append(f"{g} 8k대 피크 {'있음' if h8 else '없음'} ({len(idx)}개): 충격 수 중앙 {np.median(np.array(nimp)[idx]):.0f}, HP RMS 중앙 {np.median(hrms[idx]):.4f}, 저주파몫 중앙 {np.median(s20[idx]):.2f}"
                     + (f", 모델 미탐 {miss}개" if miss is not None else ""))
open(os.path.join(HERE, "figs", "peak_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))

# %% 그림 1: 250 Hz 칸별 피크 있는 파일 비율 (정상 vs 고장) — P1 · P2 · P3
fig, ax = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
ctr = (edges[:-1] + edges[1:]) / 2 / 1000
for a, (nm, FF) in zip(ax, [("P1 최대 에너지 주파수 (200 Hz 이상)", F1), ("P2 협대역 피크 (돌출도 ≥ 0.5 dec, 파일당 최대 3개)", F2), ("P3 정상 중앙 대비 가장 많이 높은 주파수", F3)]):
    a.bar(ctr - 0.06, frac_in(FF, nor) * 100, width=0.12, color="#2a78d6", label="정상 120")
    a.bar(ctr + 0.06, frac_in(FF, ~nor) * 100, width=0.12, color="#eb6834", label="고장 30")
    a.axvspan(7.5, 9, color="#f0efec", zorder=0)
    a.set_title(nm, loc="left", fontsize=11); a.set_ylabel("파일 %"); a.grid(axis="y", color="#e6e5e0", lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
ax[0].legend(frameon=False); ax[-1].set_xlabel("kHz (250 Hz 칸, 음영 = 7.5–9 kHz)")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "peak_hist.png"), dpi=120)

# %% 그림 2: 파일 × 주파수 정상 대비 초과 (EXC) 히트맵. 고장은 군집별, 정상은 P3 순
order_f = list(T["file"]); order_n = [names[i] for i in np.flatnonzero(nor)[np.argsort(P3[nor])]]
ordr = [names.index(n) for n in order_f + order_n]
fig, ax = plt.subplots(figsize=(14, 13))
im = ax.imshow(EXC[ordr], aspect="auto", cmap="RdBu_r", vmin=-1.5, vmax=1.5, interpolation="nearest",
               extent=[fm[0] / 1000, fm[-1] / 1000, len(ordr), 0])
ax.axhline(30, color="#0b0b0b", lw=1.2)
ax.set_yticks(np.arange(30) + 0.5, [f"{n[4:]} c{CL.get(n)} {MISS[n]}" for n in order_f], fontsize=7)
for c in sorted(set(CL.values())):
    pos = [k for k, n in enumerate(order_f) if CL.get(n) == c]
    if pos: ax.axhline(pos[-1] + 1, color="#85847e", lw=0.5, ls="--")
ax.text(12.6, 15, "고장 30\n(2-3 군집 c1–c5,\nET/AE = 미탐)", fontsize=8, va="center")
ax.text(12.6, 90, "정상 120\n(P3 주파수 순)", fontsize=8, va="center")
ax.set_xlabel("kHz"); ax.set_title("정상 중앙 PSD 대비 log 차이 (빨강 = 정상보다 높음, 파랑 = 낮음)", loc="left", fontsize=11)
fig.colorbar(im, ax=ax, shrink=0.5, label="dec")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "excess_heatmap.png"), dpi=120)
print("done")
