# %% 2-13 전 대역 스펙트럼 첨도 지도 (STFT 커토그램) — 충격이 가장 잘 보이는 대역을 파일마다 라벨 없이 고르면 무엇이 보이나
# 출발: 2-12는 8.2–8.45 kHz 한 대역의 포락 첨도만 봤다. 지금 모델의 첨도(H3–H6)는 옥타브 4칸 고정 대역이다.
#       PHM의 흔한 방법(스펙트럼 첨도 SK → 커토그램으로 대역 고르기, Antoni 2006)을 전 대역에 적용해 본다.
# 정의 (결과 보기 전에 고정, 11-23)
#   신호 = 평균 제거 → 1 kHz 고역통과(4차, filtfilt). 가장자리 0.05 s 버림
#   STFT 창 N ∈ {64, 128, 256, 512, 1024}점(Hann, 홉 N/4). 칸 폭 ≈ 390 · 195 · 98 · 49 · 24 Hz
#   SK(f) = mean_t |X(t,f)|^4 / (mean_t |X(t,f)|^2)^2 − 2   (2-12의 K − 2와 같은 뜻: 지속음 −1 · 잡음 0 · 충격 > 0)
#   주파수 1–12.4 kHz 칸만. 파일별 SKmax = 모든 (N, f) 중 최대, 그때의 f* · N* = "라벨 없이 고른 대역"
#   비교 대상 = 지금 모델의 옥타브 첨도 H3–H6(feat2 캐시, log10 Pearson 첨도)의 최대 Hmax, 충격 수 H1
# 질문 (미리 적음)
#   Q1 고장 파일의 f*는 어디에 몰리나, 옥타브 경계(2 · 4 · 8 kHz)에 걸치나
#   Q2 SKmax가 Hmax와 다른 정보인가: Spearman ρ(SKmax, Hmax) · ρ(SKmax, H1), AUC(SKmax) vs AUC(Hmax) — 기술 통계만
#      (ρ ≥ 0.9면 같은 정보로 본다)
#   Q3 J-축소가 어려워하는 파일(학습 오경보 116 · 025 · 059 · 065, val 흔들림 001 · 025 · 018)은 어디에 있나
# 이상치 정상 059 · 065 · 116은 모든 수치에 포함, 그림에서 ◆
# 출력: figs/*.png · figs/결과.txt · figs/파일표.csv
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.exp import load_train, OUTLIER_NORMALS
from src.data import load_folder
import src.viz  # noqa: F401  (한글 글꼴)

FS, TR = 25000, 1250
NS = [64, 128, 256, 512, 1024]
F_LO, F_HI = 1000, 12400
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#7c3aed", "#1F2937", "#6B7280", "#E5E7EB"
HARD_N = ["116", "025", "059", "065"]
HARD_V = ["001", "025", "018"]
OUT = {o[-3:] for o in OUTLIER_NORMALS}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
SOS_HP = signal.butter(4, 1000, "highpass", fs=FS, output="sos")


def sk_maps(x):
    """신호 → {N: (f, SK(f))} (1–12.4 kHz)"""
    xh = signal.sosfiltfilt(SOS_HP, x - x.mean())[TR:-TR]
    out = {}
    for N in NS:
        f, _, Z = signal.stft(xh, fs=FS, window="hann", nperseg=N, noverlap=N - N // 4, boundary=None, padded=False)
        P = np.abs(Z) ** 2
        sk = (P ** 2).mean(1) / (P.mean(1) ** 2 + 1e-30) - 2
        m = (f >= F_LO) & (f <= F_HI)
        out[N] = (f[m], sk[m])
    return out


X, y, names = load_train(); names = [str(n) for n in names]
vd = os.path.join(ROOT, "data", "TeamProject_val", "val"); vd = vd if os.path.isdir(vd) else os.path.dirname(vd)
Xv, nv = load_folder(vd); nv = [str(n) for n in nv]
H = pd.concat([pd.read_csv(os.path.join(ROOT, "outputs", "cache", "feat2_train.csv")),
               pd.read_csv(os.path.join(ROOT, "outputs", "cache", "feat2_val.csv"))], ignore_index=True)
assert len(H) == len(X) + len(Xv)
HK = ["H3_kurt_1-2k", "H4_kurt_2-4k", "H5_kurt_4-8k", "H6_kurt_8-12.5k"]

rows, MAPS = [], {}
for grp, XX, NN, YY in [("train", X, names, y), ("val", Xv, nv, [np.nan] * len(nv))]:
    for x, n, yy in zip(XX, NN, YY):
        M = sk_maps(np.asarray(x, float))
        lab = "val" if grp == "val" else ("고장" if yy == 1 else "정상")
        best = max(((N, f[i], s[i]) for N, (f, s) in M.items() for i in [int(np.argmax(s))]), key=lambda t: t[2])
        rows.append(dict(set=grp, file=n, y=yy, lab=lab, num=n[-3:], SKmax=best[2], f_star=best[1], N_star=best[0]))
        MAPS[(lab, n[-3:])] = M
D = pd.DataFrame(rows)
D["Hmax"] = H[HK].max(1).values; D["H_oct"] = H[HK].values.argmax(1); D["H1"] = H["H1_nimp"].values
D["logSK"] = np.log10(D.SKmax + 1)
D["outlier"] = (D.lab == "정상") & D.num.isin(OUT)
D["oct_star"] = np.digitize(D.f_star, [2000, 4000, 8000])
D["edge"] = np.min(np.abs(D.f_star.values[:, None] - np.array([2000, 4000, 8000])[None]), 1) <= (FS / D.N_star.values) / 2
D.to_csv(os.path.join(FIG, "파일표.csv"), index=False, encoding="utf-8-sig")

# %% 결과 표
T = D[D.set == "train"]; yt = T.y.astype(int).values
txt = ["정의: 1 kHz 고역 → STFT(N = 64–1024, 홉 N/4) → SK(f) = E|X|^4/(E|X|^2)^2 − 2. SKmax = 모든 (N, f) 최대, f* · N* = 그 자리",
       "Hmax = 지금 모델 옥타브 첨도 H3–H6 최대 (log10 Pearson 첨도), H1 = log1p(충격 수)", ""]
txt.append("## Q1 f*(라벨 없이 고른 대역) 옥타브 분포 (1–2 · 2–4 · 4–8 · 8–12.4 kHz)")
for lab in ["정상", "고장", "val"]:
    d = D[D.lab == lab]
    c = np.bincount(d.oct_star, minlength=4)
    txt.append(f"  {lab} ({len(d)}): {c.tolist()} · 옥타브 경계에 걸친 칸 {int(d.edge.sum())} · N* 분포 "
               + " ".join(f"{N}:{int((d.N_star == N).sum())}" for N in NS))
    txt.append(f"     f* 옥타브 == Hmax 옥타브: {int((d.oct_star == d.H_oct).sum())}/{len(d)}")
txt.append("")
txt.append("## Q2 다른 정보인가 (학습 150)")
for a, b in [("logSK", "Hmax"), ("logSK", "H1"), ("Hmax", "H1")]:
    txt.append(f"  Spearman ρ({a}, {b}) = {spearmanr(T[a], T[b])[0]:.3f}")
for c in ["logSK", "Hmax", "H1"]:
    txt.append(f"  AUC({c}) = {roc_auc_score(yt, T[c]):.3f}")
txt.append("  그룹 중앙 [5 % – 95 %] SKmax: " + " · ".join(
    f"{lab} {D[D.lab == lab].SKmax.median():.2f} [{D[D.lab == lab].SKmax.quantile(.05):.2f} – {D[D.lab == lab].SKmax.quantile(.95):.2f}]"
    for lab in ["정상", "고장", "val"]))
txt.append("")
txt.append("## Q3 어려운 파일")
fmt = lambda r: (f"SKmax {r.SKmax:.2f} (f* {r.f_star:.0f} Hz, N {r.N_star}) · Hmax {r.Hmax:.2f} · H1 {r.H1:.2f} · "
                 f"정상 중 SKmax 순위 {int((T[T.y == 0].SKmax < r.SKmax).mean() * 100)} 백분위")
for lab, nums in [("정상", HARD_N), ("val", HARD_V)]:
    for n in nums:
        r = D[(D.lab == lab) & (D.num == n)].iloc[0]
        txt.append(f"  {lab} {n}: " + fmt(r))
txt.append("  고장 SKmax 하위 5: " + ", ".join(f"{r.num} {r.SKmax:.2f} (f* {r.f_star:.0f})" for _, r in T[T.y == 1].nsmallest(5, "SKmax").iterrows()))
txt.append("  정상 SKmax 상위 5: " + ", ".join(f"{r.num} {r.SKmax:.2f} (f* {r.f_star:.0f})" for _, r in T[T.y == 0].nlargest(5, "SKmax").iterrows()))
open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8").write("\n".join(txt))

# %% 그림 1 — 그룹별 커토그램 (창 길이 × 주파수, 칸 = 그룹 중앙 log10(1 + SK))
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=True)
vmax = 0
grids = {}
for lab in ["정상", "고장", "val"]:
    keys = [k for k in MAPS if k[0] == lab]
    g = []
    for N in NS:
        S = np.array([MAPS[k][N][1] for k in keys]); f = MAPS[keys[0]][N][0]
        g.append(np.interp(np.linspace(F_LO, F_HI, 400), f, np.median(np.log10(np.clip(S, -0.99, None) + 1), 0)))
    grids[lab] = np.array(g); vmax = max(vmax, grids[lab].max())
for a, lab in zip(axes, ["정상", "고장", "val"]):
    im = a.imshow(grids[lab], aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=vmax,
                  extent=[F_LO / 1000, F_HI / 1000, -.5, len(NS) - .5])
    for e in (2, 4, 8):
        a.axvline(e, color="white", lw=1, ls="--", alpha=.7)
    a.set_title(f"{lab} ({sum(1 for k in MAPS if k[0] == lab)}) 중앙", loc="left")
    a.set_xlabel("주파수 (kHz)")
axes[0].set_yticks(range(len(NS))); axes[0].set_yticklabels([f"{N}점 ({FS / N:.0f} Hz)" for N in NS])
axes[0].set_ylabel("STFT 창 (칸 폭)")
fig.colorbar(im, ax=axes, label="log10(1 + SK)", shrink=.85)
fig.suptitle("그룹별 커토그램: 밝을수록 그 대역이 충격적(뾰족) · 점선 = 옥타브 경계(지금 H3–H6 칸)", x=.06, ha="left", fontsize=13)
fig.savefig(os.path.join(FIG, "1_그룹커토그램.png"), dpi=150, bbox_inches="tight"); plt.close(fig)

# %% 그림 2 — f* 분포 (라벨 없이 고른 대역)
fig, ax = plt.subplots(figsize=(12, 4.2))
bins = np.arange(1000, 12600, 250)
for lab, col in [("정상", NOR), ("고장", ABN), ("val", VAL)]:
    d = D[D.lab == lab]
    ax.hist(d.f_star, bins=bins, histtype="step", lw=2, color=col, density=True, label=f"{lab} ({len(d)})")
for e in (2000, 4000, 8000):
    ax.axvline(e, color=MUTED, lw=1, ls="--")
ax.set_xlabel("f* = 파일에서 SK가 가장 큰 주파수 (Hz)"); ax.set_ylabel("밀도")
ax.set_title("파일마다 라벨 없이 고른 '가장 충격적인 대역' · 점선 = 옥타브 경계", loc="left")
ax.legend(frameon=False); ax.grid(color=GRID, lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_f별분포.png"), dpi=150); plt.close(fig)

# %% 그림 3 — SKmax vs Hmax (어려운 파일 표시)
fig, ax = plt.subplots(figsize=(10, 7.5))
for lab, col, s, a in [("정상", NOR, 20, .45), ("고장", ABN, 28, .75), ("val", VAL, 24, .6)]:
    d = D[(D.lab == lab) & ~D.outlier]
    ax.scatter(d.Hmax, d.logSK, s=s, c=col, alpha=a, lw=0, label=f"{'학습 ' if lab != 'val' else ''}{lab}")
d = D[D.outlier]
ax.scatter(d.Hmax, d.logSK, s=90, marker="D", facecolor=NOR, edgecolor="white", lw=1.5, label="이상치 정상 ◆", zorder=5)
for lab, nums, col in [("정상", HARD_N, NOR), ("val", HARD_V, VAL)]:
    for n in nums:
        r = D[(D.lab == lab) & (D.num == n)].iloc[0]
        if not r.outlier:
            ax.scatter(r.Hmax, r.logSK, s=80, facecolor=col, edgecolor=INK, lw=1.2, zorder=6)
        ax.annotate(("N" if lab == "정상" else "V") + n, (r.Hmax, r.logSK), xytext=(6, 4), textcoords="offset points", fontsize=11, color=INK)
ax.set_xlabel("지금 모델: 옥타브 첨도 최대 Hmax (log10 Pearson 첨도)")
ax.set_ylabel("커토그램 최대 log10(1 + SKmax)")
ax.set_title(f"고정 옥타브 첨도 대 라벨 없이 고른 대역의 첨도 · Spearman ρ = {spearmanr(T.logSK, T.Hmax)[0]:.2f} (학습)\n"
             "N = J-축소 학습 오경보 · V = val 흔들림", loc="left", fontsize=12)
ax.legend(frameon=False, loc="upper left"); ax.grid(color=GRID, lw=.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_SK대Hmax.png"), dpi=150); plt.close(fig)
print("done")
