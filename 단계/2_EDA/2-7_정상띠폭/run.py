# %% [markdown]
# 2-7 정상 띠 폭 — 오탐(오경보) 정상이 정상 띠 경계에서 노는가, 띠를 위아래로 좁히면 어떻게 되나
# 가설(사용자): PSD·스펙트럼 첨도·포락선 스펙트럼에서 오탐 정상이 정상 띠 경계 근처에 있어 띠를 넓힌다.
#              이들을 빼거나 띠를 좁히면(5–95 → 15–85 · 20–80 · 25–75%) 폭이 줄어든다
# 확인: ① 오탐 정상 파일별로 칸이 어느 구역에 있나 (<5 · 5–25 · 25–75 · 75–95 · >95%)
#       ② 띠 폭 (위 − 아래, 주파수 평균): 정상 120개 vs 오탐 14개를 뺀 106개
#       ③ 띠마다 위·아래로 벗어나는 칸 비율 (묶음별)
# 신호·계산은 EDA 뷰어와 같다: PSD = 평균 제거 Welch 4096 (detrend 없음, 20 Hz 이상),
#   스펙트럼 첨도 = 평균 제거 STFT 1024 (DC 제외), 포락선 스펙트럼 = 200 Hz 고역통과 포락 (1–300 Hz)
# 묶음: 2-6과 같음 (EXP-007 ET · EXP-012 AE 중 하나라도 5번 중 3번 이상 틀림)
# 실행: conda run -n qe-phm python "단계/2_EDA/2-7_정상띠폭/run.py"

# %% 설정
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, ROOT)
from src.exp import load_train
import src.viz  # noqa: F401
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.signal import butter, sosfiltfilt, welch, spectrogram, hilbert

FS = 25000
X, y, names = load_train(); names = [str(n) for n in names]; N = len(y); nor = y == 0
os.makedirs(os.path.join(HERE, "figs"), exist_ok=True)
Xc = X - X.mean(1, keepdims=True)
H = sosfiltfilt(butter(4, 200, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)

f, P = welch(Xc, fs=FS, nperseg=4096, detrend=False)
mP = f >= 20; SPEC = {"PSD": (f[mP], np.log10(P[:, mP] + 1e-30), "log10 g²/Hz")}
fs_, _, S = spectrogram(Xc, fs=FS, nperseg=1024, noverlap=512)
SK = (S ** 2).mean(2) / (S.mean(2) ** 2 + 1e-60) - 2
mS = fs_ > 0; SPEC["스펙트럼 첨도"] = (fs_[mS], np.log10(1 + np.clip(SK[:, mS], 0, None)), "log10(1+SK)")
ENV = np.abs(hilbert(H, axis=1)); Ed = ENV.reshape(N, -1, 10).mean(2); Ed -= Ed.mean(1, keepdims=True)
fe, PE = welch(Ed, fs=FS / 10, nperseg=2500); mE = (fe >= 1) & (fe <= 300)
SPEC["포락선 스펙트럼"] = (fe[mE], np.log10(PE[:, mE] + 1e-30), "log10")

# %% 묶음
def counts(exp_dir):
    r = json.load(open(os.path.join(ROOT, "단계", "5_모델비교", exp_dir, "result.json"), encoding="utf-8"))
    c = next(iter(r["configs"].values())); return c["miss_counts"], c["fp_counts"]
aem, aef = counts("유의미/EXP-012_스펙트럼AE_tSNE"); etm, etf = counts("기각/EXP-007_공선성제거")
def who(n):
    d = [("ET", etm), ("AE", aem)] if n.startswith("abn") else [("ET", etf), ("AE", aef)]
    return "+".join(m for m, c in d if c.get(n, 0) >= 3)
WHO = [who(n) for n in names]; err = np.array([bool(w) for w in WHO])
GRP = np.where(y == 1, np.where(err, "놓친 고장", "잡힌 고장"), np.where(err, "오탐 정상", "문제없는 정상"))
ORDER = ["오탐 정상", "문제없는 정상", "놓친 고장", "잡힌 고장"]
BANDS = [(5, 95), (15, 85), (20, 80), (25, 75)]
ok = GRP == "문제없는 정상"; fp = np.flatnonzero(GRP == "오탐 정상")
ZONES = [("<5%", None, 5), ("5–25%", 5, 25), ("25–75%", 25, 75), ("75–95%", 75, 95), (">95%", 95, None)]

def zone_frac(A, ref):
    """파일마다 칸이 각 구역에 있는 비율 (ref 집단의 백분위 기준)"""
    q = {p: np.percentile(A[ref], p, 0) for p in (5, 25, 75, 95)}
    out = []
    for _, lo, hi in ZONES:
        m = np.ones_like(A, bool)
        if lo is not None: m &= A >= q[lo]
        if hi is not None: m &= A < q[hi]
        out.append(m.mean(1))
    return np.stack(out, 1)                       # (N, 5)

# %% 계산
lines, ZF = [], {}
for name, (ff, A, unit) in SPEC.items():
    lines.append(f"=== {name} ({len(ff)}칸, {ff[0]:.0f}–{ff[-1]:.0f} Hz, {unit}) ===")
    # ① 구역
    Z = zone_frac(A, nor); ZF[name] = Z
    lines.append("[1] 칸이 있는 구역 (정상 120개 백분위 기준): <5 / 5–25 / 25–75 / 75–95 / >95%   (정상 전체 평균이면 5/20/50/20/5)")
    for g in ORDER:
        m = GRP == g; lines.append(f"    {g} ({m.sum()}): " + " / ".join(f"{v*100:.0f}" for v in Z[m].mean(0)))
    lines.append("    오탐 정상 파일별:")
    for i in fp:
        z = Z[i] * 100
        three = {"가운데": z[2], "경계": z[1] + z[3], "밖": z[0] + z[4]}
        tag = max(three, key=three.get); side = "위" if z[3] + z[4] > z[0] + z[1] else "아래"
        lines.append(f"      {names[i]} ({WHO[i]}): " + " / ".join(f"{v:3.0f}" for v in z) +
                     f"   가운데 {z[2]:.0f} · 경계 {z[1]+z[3]:.0f} · 밖 {z[0]+z[4]:.0f} → {tag}{'' if tag == '가운데' else '(' + side + ')'}")
    tags = []
    for i in fp:
        z = Z[i]; three = {"가운데": z[2], "경계": z[1] + z[3], "밖": z[0] + z[4]}; tags.append(max(three, key=three.get))
    lines.append("    오탐 14개 분류: " + ", ".join(f"{t} {tags.count(t)}" for t in ["가운데", "경계", "밖"]))
    # ② 폭
    lines.append("[2] 띠 폭 (위 − 아래, 주파수 평균) · 5–95% 대비 · 오탐 14개를 뺀 106개로 구한 같은 띠의 폭")
    w0 = (np.percentile(A[nor], 95, 0) - np.percentile(A[nor], 5, 0)).mean()
    for lo, hi in BANDS:
        wa = (np.percentile(A[nor], hi, 0) - np.percentile(A[nor], lo, 0)).mean()
        wo = (np.percentile(A[ok], hi, 0) - np.percentile(A[ok], lo, 0)).mean()
        ua = (np.percentile(A[nor], hi, 0) - np.median(A[nor], 0)).mean(); uo = (np.percentile(A[ok], hi, 0) - np.median(A[ok], 0)).mean()
        la = (np.median(A[nor], 0) - np.percentile(A[nor], lo, 0)).mean(); lo_ = (np.median(A[ok], 0) - np.percentile(A[ok], lo, 0)).mean()
        lines.append(f"    {lo}–{hi}%: 120개 {wa:.3f} ({wa/w0*100:.0f}%) → 106개 {wo:.3f} ({wo/wa*100:.0f}% of 120개) | 위쪽 반폭 {ua:.3f}→{uo:.3f} ({uo/ua*100:.0f}%), 아래쪽 반폭 {la:.3f}→{lo_:.3f} ({lo_/la*100:.0f}%)")
    # ③ 벗어나는 칸
    lines.append("[3] 띠를 벗어나는 칸 비율 (정상 120개 띠 기준): 위 / 아래 — 묶음 중앙값")
    for lo, hi in BANDS:
        up_ = (A > np.percentile(A[nor], hi, 0)).mean(1); dn = (A < np.percentile(A[nor], lo, 0)).mean(1)
        lines.append(f"    {lo}–{hi}%: " + " | ".join(f"{g} {np.median(up_[GRP==g])*100:.0f}/{np.median(dn[GRP==g])*100:.0f}" for g in ORDER))
    lines.append("")
open(os.path.join(HERE, "figs", "band_width_numbers.txt"), "w", encoding="utf-8").write("\n".join(lines))

# %% 그림 1: 스펙트럼 3개 × 대칭 띠 4겹 + 묶음 중앙선
fig, ax = plt.subplots(3, 1, figsize=(14, 13))
for a, (name, (ff, A, unit)) in zip(ax, SPEC.items()):
    for (lo, hi), al in zip(BANDS, [0.12, 0.18, 0.24, 0.32]):
        a.fill_between(ff, np.percentile(A[nor], lo, 0), np.percentile(A[nor], hi, 0), color="#2a78d6", alpha=al, lw=0, label=f"정상 {lo}–{hi}%")
    a.plot(ff, np.median(A[nor], 0), color="#2a78d6", lw=1, ls="--", label="정상 중앙")
    a.plot(ff, np.median(A[GRP == "오탐 정상"], 0), color="#184f95", lw=1.8, label="오탐 정상 중앙 (14)")
    a.plot(ff, np.median(A[GRP == "놓친 고장"], 0), color="#eb6834", lw=1.4, label="놓친 고장 중앙 (6)")
    a.plot(ff, np.median(A[GRP == "잡힌 고장"], 0), color="#eb6834", lw=1, ls=":", label="잡힌 고장 중앙 (24)")
    if name != "스펙트럼 첨도":
        a.set_xscale("log"); a.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_title(f"{name} ({unit})", loc="left", fontsize=11); a.set_xlabel("Hz"); a.grid(color="#e6e5e0", lw=0.6)
    a.spines[["top", "right"]].set_visible(False)
ax[0].legend(ncol=4, fontsize=8, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(os.path.join(HERE, "figs", "band_width.png"), dpi=120)

# %% 그림 2: 오탐 정상 파일별 칸 구역 (정상 120개 백분위 기준) + 묶음 평균
ZC = ["#184f95", "#9ec5f4", "#e6e5e0", "#f4b391", "#c4471c"]
rows = [names[i][4:] + f" ({WHO[i]})" for i in fp] + ["오탐 정상 평균", "문제없는 정상 평균", "놓친 고장 평균"]
fig, ax = plt.subplots(1, 3, figsize=(15, 6.5), sharey=True)
for a, (name, Z) in zip(ax, ZF.items()):
    M = np.vstack([Z[fp], Z[GRP == "오탐 정상"].mean(0), Z[ok].mean(0), Z[GRP == "놓친 고장"].mean(0)]) * 100
    left = np.zeros(len(rows))
    for k, (zl, _, _) in enumerate(ZONES):
        a.barh(range(len(rows)), M[:, k], left=left, color=ZC[k], edgecolor="white", linewidth=1, label=zl, height=0.8)
        left += M[:, k]
    a.axvline(5, color="#85847e", lw=0.6, ls=":"); a.axvline(95, color="#85847e", lw=0.6, ls=":")
    a.set_title(name, loc="left", fontsize=11); a.set_xlim(0, 100); a.set_xlabel("칸 비율 %")
    a.spines[["top", "right"]].set_visible(False)
ax[0].set_yticks(range(len(rows)), rows, fontsize=9); ax[0].invert_yaxis()
ax[0].axhline(len(fp) - 0.5, color="#0b0b0b", lw=0.8)
hd, lb = ax[0].get_legend_handles_labels()
fig.legend(hd, lb, title="정상 120개 기준 구역", ncol=5, fontsize=9, frameon=False, loc="lower center")
fig.suptitle("오탐 정상의 주파수 칸이 정상 분포의 어느 구역에 있나 (가운데 회색 = 25–75%)", x=0.01, ha="left", fontsize=12)
fig.tight_layout(rect=[0, 0.08, 1, 0.96]); fig.savefig(os.path.join(HERE, "figs", "fp_zones.png"), dpi=120)
print("done")
