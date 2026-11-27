# %% 2-16 8.3k 규칙 다시 보기 — val 40개를 얹고, 왜 val_001만 켜졌나
# 출발(11-23): val_001이 8.3k 규칙(P83 ≥ 0.75 AND W83 ≥ 150 Hz)으로 켜져 고장이 됐다. 2-12에서 신호는 "미끄러지는 지속음"으로 확인.
#   질문 ① 같은 지속음 특징이 있는 정상(014 053 019 …)은 왜 안 켜지고 001만 켜지나  ② 경계를 어떻게 고치면 되나
# 정의 (결과 보기 전에 고정)
#   규칙 재료 P83 · W83 = src/final._peak83과 같은 식(2초 전체 Welch 8192, 55 Hz 평활 − 1.2 kHz 이동 중앙값, 8.20–8.45 kHz 최대 돌출 · 반높이 폭)
#   K_raw = 2-12 파일표의 8.2–8.45 kHz 포락 첨도 (지속음 1 · 잡음 2 · 충격 > 2)
#   [후보 B] 구간별 규칙: 2초를 0.5 s 4구간으로 나눠 같은 식(Welch 4096)으로 P · W를 구간마다 계산. 한 구간에서 두 조건이 함께 켜지면 켜짐.
#            전제 = 지속음이 구간 안에서는 덜 미끄러져서 좁고, 충격 울림은 한 구간 안에서 넓다 (결과 보기 전 가설)
#   트랙 = 스펙트로그램(2048, 중첩 50 %) 8.0–8.8 kHz 프레임별 돌출 최대 주파수. 이동폭 = 트랙(5프레임 중앙값 평활)의 최대−최소,
#          지속성 Pf = 프레임별 돌출 최대의 중앙값(시간 내내 이어지는 정도)
# 출력: figs/*.png · figs/표.md · figs/파일표.csv
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import median_filter, uniform_filter1d
from src.exp import load_train
from src.data import load_folder
import src.viz  # noqa: F401
from src.exp import find_exp  # noqa: F401

FS = 25000
B_LO, B_HI = 8200, 8450
P_MIN, W_MIN = 0.75, 150.0
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#7c3aed", "#1F2937", "#6B7280", "#E5E7EB"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def prom(x, nper, hop_avg=None):
    f, P = signal.welch(x - x.mean(), fs=FS, nperseg=nper)
    L = np.log10(P); df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))))
    return f, Sm - median_filter(Sm, size=int(round(1200 / df)), mode="nearest")


def peak_pw(f, PR):
    band = (f >= B_LO) & (f < B_HI); ia = np.flatnonzero(band)
    k = ia[PR[band].argmax()]; h = PR[k] / 2; l = r = k
    while l > 0 and PR[l - 1] >= h:
        l -= 1
    while r < len(PR) - 1 and PR[r + 1] >= h:
        r += 1
    return float(PR[k]), float(f[r] - f[l]), float(f[k])


def measure(x):
    x = np.asarray(x, float)
    f, PR = prom(x, 8192); P, W, F0 = peak_pw(f, PR)
    seg = []
    for s in range(4):
        fs_, PRs = prom(x[s * 12500:(s + 1) * 12500], 4096); seg.append(peak_pw(fs_, PRs))
    on_seg = [(p >= P_MIN and w >= W_MIN) for p, w, _ in seg]
    # 트랙
    xc = x - x.mean()
    f2, t2, S = signal.spectrogram(xc, fs=FS, nperseg=2048, noverlap=1024)
    L = np.log10(S + 1e-20); df2 = f2[1] - f2[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df2))), axis=0)
    PR2 = Sm - median_filter(Sm, size=(int(round(1200 / df2)), 1), mode="nearest")
    band = (f2 >= 8000) & (f2 < 8800); ia = np.flatnonzero(band)
    Pf = PR2[band].max(0); trk = f2[ia[PR2[band].argmax(0)]]
    trk_s = median_filter(trk, size=5, mode="nearest")
    return dict(P83=P, W83=W, F83=F0, spec=(P >= P_MIN) and (W >= W_MIN),
                Pseg_max=max(p for p, _, _ in seg), Wseg_max=max(w for _, w, _ in seg),
                Wseg_at_Pmax=seg[int(np.argmax([p for p, _, _ in seg]))][1], seg_on=int(sum(on_seg)), seg_rule=any(on_seg),
                Pf=float(np.median(Pf)), drift=float(np.ptp(trk_s)), trk_med=float(np.median(trk))), (f, PR, seg, t2, f2, S)


X, y, names = load_train(); names = [str(n) for n in names]
vd = os.path.join(ROOT, "2_data", "TeamProject_val", "val")
Xv, nv = load_folder(vd); nv = [str(n) for n in nv]
K = pd.read_csv(os.path.join(ROOT, "3_단계", "2_EDA", "아카이브", "8kHz공진", "2-12_1123_8.3k지속음대충격", "figs", "파일표.csv"), encoding="utf-8-sig").set_index("file")["K_raw"]
rows, SIG = [], {}
for grp, XX, NN, YY in [("train", X, names, y), ("val", Xv, nv, [np.nan] * len(nv))]:
    for x, n, yy in zip(XX, NN, YY):
        m, _ = measure(x)
        lab = "val" if grp == "val" else ("고장" if yy == 1 else "정상")
        rows.append(dict(set=grp, file=n, lab=lab, num=n[-3:], K=float(K[n]), **m)); SIG[(lab, n[-3:])] = np.asarray(x, float)
D = pd.DataFrame(rows)
D["막은조건"] = np.where(D.spec, "켜짐", np.where((D.P83 < P_MIN) & (D.W83 < W_MIN), "둘 다 부족", np.where(D.P83 < P_MIN, "돌출 부족", "폭 부족")))
D.to_csv(os.path.join(FIG, "파일표.csv"), index=False, encoding="utf-8-sig")
g = lambda lab, num: D[(D.lab == lab) & (D.num == num)].iloc[0]

# %% 표
out = ["# 2-16 수치 (자동 생성)", "", "P83 = 돌출(dec), W83 = 반높이 폭(Hz), K = 대역 포락 첨도, Pf = 프레임별 돌출 중앙값(지속성), 이동폭 = 8.0–8.8k 트랙 최대−최소(Hz)",
       "구간별 = 0.5 s × 4 구간에서 각각 P · W를 재고 한 구간에서 두 조건이 함께 켜지면 켜짐", ""]
fm = lambda r: (f"| {r.lab} {r.num} | {r.P83:.2f} | {r.W83:.0f} | {r.막은조건} | {r.K:.1f} | {r.Pf:.2f} | {r.drift:.0f} | "
                f"{r.Pseg_max:.2f} | {r.Wseg_at_Pmax:.0f} | {'켜짐' if r.seg_rule else '-'} |")
hdr = "| 파일 | P83 | W83 | 지금 규칙 | K | Pf | 이동폭 | 구간 P최대 | 그 구간 W | 구간별 규칙 |\n|---|---|---|---|---|---|---|---|---|---|"
out += ["## 지금 규칙이 켜진 파일", hdr] + [fm(r) for _, r in D[D.spec].iterrows()]
out += ["", "## 구간별 규칙이 켜지는 파일 (지금 규칙과 다른 것)", hdr] + [fm(r) for _, r in D[D.seg_rule != D.spec].iterrows()]
out += ["", "## 지속성 Pf ≥ 1.0 인 파일 (시간 내내 8 kHz대 봉우리가 있음)", hdr] + [fm(r) for _, r in D[D.Pf >= 1.0].sort_values("Pf", ascending=False).iterrows()]
out += ["", "## val 중 P83 ≥ 0.75 또는 W83 ≥ 120 (경계 근처)", hdr] + [fm(r) for _, r in D[(D.lab == "val") & ((D.P83 >= .75) | (D.W83 >= 120))].iterrows()]
out += ["", "## 요약", f"- 지금 규칙 켜짐: 학습 고장 {int(((D.lab=='고장')&D.spec).sum())} · 학습 정상 {int(((D.lab=='정상')&D.spec).sum())} · val {int(((D.lab=='val')&D.spec).sum())}",
        f"- 구간별 규칙 켜짐: 학습 고장 {int(((D.lab=='고장')&D.seg_rule).sum())} · 학습 정상 {int(((D.lab=='정상')&D.seg_rule).sum())} · val {int(((D.lab=='val')&D.seg_rule).sum())}",
        f"- 지금 규칙 AND K > 2: 학습 고장 {int(((D.lab=='고장')&D.spec&(D.K>2)).sum())} · 학습 정상 {int(((D.lab=='정상')&D.spec&(D.K>2)).sum())} · val {int(((D.lab=='val')&D.spec&(D.K>2)).sum())}"]
open(os.path.join(FIG, "표.md"), "w", encoding="utf-8").write("\n".join(out))

# %% 그림 1 — 훈련 산점도에 val을 얹음 (뷰어 03 그림과 같은 도식)
fig, ax = plt.subplots(figsize=(11, 7.5))
ax.axvspan(P_MIN, 2.2, ymin=W_MIN / 700, color="#f1f1f1", lw=0, zorder=0)
for lab, col, s, a, mk in [("정상", NOR, 28, .75, "o"), ("고장", ABN, 36, .85, "o"), ("val", VAL, 52, .9, "D")]:
    d = D[D.lab == lab]
    ax.scatter(d.P83, d.W83, s=s, c=col, alpha=a, marker=mk, lw=0, label=f"{'학습 ' if lab != 'val' else ''}{lab} ({len(d)})", zorder=3)
ax.axvline(P_MIN, color=INK, ls="--", lw=1.2); ax.axhline(W_MIN, color=INK, ls="--", lw=1.2)
for _, r in D[(D.lab == "val") & ((D.P83 >= .75) | (D.W83 >= 140))].iterrows():
    ax.annotate("V" + r.num, (r.P83, r.W83), xytext=(7, 5), textcoords="offset points", fontsize=11, color=VAL, fontweight="bold")
for num in ["019", "092", "119", "088"]:
    r = g("정상", num); ax.annotate("N" + num, (r.P83, r.W83), xytext=(6, 4), textcoords="offset points", fontsize=10, color=NOR)
ax.set_xlim(0, 1.6); ax.set_ylim(0, 700)
ax.set_xlabel("P83 · 돌출 (dec)"); ax.set_ylabel("W83 · 반높이 폭 (Hz)"); ax.grid(color=GRID, lw=.6)
ax.set_title("8.3k 규칙 평면에 val을 얹음 (회색 = 규칙 영역)\nval_001은 훈련 고장보다 오른쪽 위로 크게 벗어남", loc="left", fontsize=13)
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_산점도_val포함.png"), dpi=140); plt.close(fig)

# %% 그림 2 — 같은 파일을 K · 이동폭 축으로 (규칙 영역 점은 크게)
fig, axs = plt.subplots(1, 2, figsize=(16, 7))
for ax, (xc, xl, lg) in zip(axs, [("K", "8.2–8.45 kHz 포락 첨도 K (지속음 1 · 잡음 2 · 충격 ≫ 2)", True), ("drift", "8.0–8.8 kHz 트랙 이동폭 (Hz)", False)]):
    for lab, col, s, a, mk in [("정상", NOR, 20, .5, "o"), ("고장", ABN, 28, .75, "o"), ("val", VAL, 40, .9, "D")]:
        d = D[D.lab == lab]; ax.scatter(d[xc], d.P83, s=s, c=col, alpha=a, marker=mk, lw=0, label=lab)
    d = D[D.spec]
    ax.scatter(d[xc], d.P83, s=170, facecolor="none", edgecolor=INK, lw=1.6, zorder=6, label="지금 규칙 켜짐")
    for _, r in d.iterrows():
        ax.annotate(r.lab[0] + r.num, (r[xc], r.P83), xytext=(8, 6), textcoords="offset points", fontsize=10, color=INK)
    ax.axhline(P_MIN, color=MUTED, ls="--", lw=1)
    if lg: ax.set_xscale("log"); ax.axvline(2, color=MUTED, ls=":", lw=1)
    ax.set_xlabel(xl); ax.set_ylabel("P83 (dec)"); ax.grid(color=GRID, lw=.5)
axs[0].legend(frameon=False, loc="upper center", ncol=4, bbox_to_anchor=(0.5, -0.14))
axs[0].set_title("K: 규칙이 켜진 고장은 모두 K ≥ 16, val_001만 1.7", loc="left", fontsize=12)
axs[1].set_title("이동폭: 켜진 고장 98–391 Hz, val_001 427 Hz로 겹침 → 이동폭으로는 안 갈림", loc="left", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_K_이동폭.png"), dpi=140); plt.close(fig)

# %% 그림 3 — 스펙트로그램 (규칙 켜진 파일 · 지속음 정상 · val 경계)
CASES = [("고장", "002"), ("고장", "006"), ("고장", "010"), ("고장", "029"), ("고장", "030"),
         ("val", "001"), ("val", "037"), ("val", "019"), ("val", "020"), ("val", "028"),
         ("정상", "014"), ("정상", "053"), ("정상", "019"), ("정상", "092"), ("정상", "066")]
fig, axs = plt.subplots(3, 5, figsize=(22, 11.5)); axs = axs.ravel()
for ax, (lab, num) in zip(axs, CASES):
    x = SIG[(lab, num)]; x = x - x.mean()
    f, t, S = signal.spectrogram(x, fs=FS, nperseg=2048, noverlap=1536); m = (f >= 7800) & (f <= 8900)
    ax.pcolormesh(t, f[m] / 1000, 10 * np.log10(S[m] + 1e-20), shading="auto", cmap="viridis", rasterized=True)
    ax.axhline(B_LO / 1000, color="white", lw=.8, ls="--"); ax.axhline(B_HI / 1000, color="white", lw=.8, ls="--")
    r = g(lab, num); col = {"고장": ABN, "val": VAL, "정상": NOR}[lab]
    ax.set_title(f"{lab} {num} · P83 {r.P83:.2f} · W83 {r.W83:.0f} · K {r.K:.1f}" + (" · 켜짐" if r.spec else ""), fontsize=12, color=col, loc="left")
    ax.set_xlabel("시간 (s)", fontsize=9); ax.set_ylabel("kHz", fontsize=9)
fig.suptitle("8.0–8.9 kHz 스펙트로그램 (흰 점선 = 규칙 대역 8.20–8.45 kHz). 위 = 규칙 켜진 학습 고장 · 가운데 = val 경계 · 아래 = 지속음 정상", fontsize=14, y=1.0)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_스펙트로그램.png"), dpi=75, bbox_inches="tight"); plt.close(fig)

# %% 그림 4 — 001이 켜진 이유: 2초 평균은 이동하는 좁은 음을 뭉갠다 (평균 스펙트럼 vs 0.5 s 구간 스펙트럼)
fig, axs = plt.subplots(1, 3, figsize=(18, 5.2))
for ax, (lab, num, col) in zip(axs, [("val", "001", VAL), ("고장", "030", ABN), ("정상", "019", NOR)]):
    x = SIG[(lab, num)]; r = g(lab, num)
    f, PR = prom(x, 8192); m = (f >= 7800) & (f <= 9000)
    ax.axvspan(B_LO / 1000, B_HI / 1000, color=GRID, lw=0)
    ax.plot(f[m] / 1000, PR[m], color=INK, lw=2.4, label=f"2초 평균: P {r.P83:.2f} · W {r.W83:.0f}")
    for s in range(4):
        fs_, PRs = prom(x[s * 12500:(s + 1) * 12500], 4096); p, w, f0 = peak_pw(fs_, PRs); mm = (fs_ >= 7800) & (fs_ <= 9000)
        ax.plot(fs_[mm] / 1000, PRs[mm], color=col, lw=1.1, alpha=.45 + .15 * s, label=f"{s * .5:.1f}–{(s + 1) * .5:.1f} s: P {p:.2f} · W {w:.0f}")
    ax.set_title(f"{lab} {num}", loc="left", color=col); ax.set_xlabel("주파수 (kHz)"); ax.set_ylabel("돌출 (dec)"); ax.grid(color=GRID, lw=.5); ax.legend(frameon=False, fontsize=9)
fig.suptitle("굵은 검정 = 규칙이 쓰는 2초 평균 돌출, 색 선 = 0.5 s 구간별 (회색 띠 = 규칙 대역)", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_평균대구간.png"), dpi=140); plt.close(fig)
print("done")
