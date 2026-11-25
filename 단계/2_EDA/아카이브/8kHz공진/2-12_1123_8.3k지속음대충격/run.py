# %% 2-12 8.3k 대역 에너지는 지속음인가 충격 울림인가 — 스펙트럴 첨도 · 파일별 AR 백색화
# 출발: val_001은 8.3k 스펙트럼 갈래만으로 고장(공진 fit 0, 충격 없음, 미끄러지는 지속음, 6-7 · 2-10).
#       규칙 전제(H2) = "충격이 8.3 kHz 공진을 울린다". 평균 스펙트럼 돌출(P83)은 지속음과 충격 울림을 못 가른다.
# 정의 (결과 보기 전에 고정, 11-23)
#   대역 = 규칙 대역 8200–8450 Hz (src/final.py와 같음). 버터워스 4차 대역통과(filtfilt) → 힐베르트 포락 e(t)
#   M1 대역 첨도 K = mean(e^4) / mean(e^2)^2  (복소 해석신호 기준 이론값: 순수 지속음 1 · 가우스 잡음 2 · 충격 > 2)
#   M2 = 같은 K를 "파일별 AR(32) 백색화" 신호에서 계산. 1 kHz 고역통과 → 그 파일 자신으로 Yule-Walker AR(32)
#        → 예측 오차(잔차). 지속음(예측 가능)은 지워지고 충격은 남는다
#   M3 = 잔존 비율 = (백색화 후 대역 에너지 / 백색화 후 1–12.4 kHz 에너지) ÷ (원신호의 같은 비)  (참고)
#   가장자리 0.05 s 잘라냄. P83 · W83 = src/final.py와 같은 정의(55 Hz 평활 log PSD − 1.2 kHz 이동 중앙값)
# 통과 조건 (미리 적음): 측정 하나가 학습 8.3k 규칙 고장 6개(002 003 006 010 029 030) 모두를 충격 쪽(K > 2),
#   val_001을 지속음 쪽(K ≤ 2)에 놓는다. 문턱은 이론값 2, 001에 맞춰 바꾸지 않는다
# 출력: figs/*.png · figs/결과.txt · figs/파일표.csv
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import signal, linalg
from scipy.ndimage import median_filter, uniform_filter1d
from src.exp import load_train
from src.data import load_folder
import src.viz  # noqa: F401

FS, TR, P_AR = 25000, 1250, 32
B_LO, B_HI = 8200, 8450
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#7c3aed", "#1F2937", "#6B7280", "#E5E7EB"
RULE_ABN = ["002", "003", "006", "010", "029", "030"]
TONE_NOR = ["014", "053", "009", "022", "045", "030", "090", "066", "019"]   # 6-7: 8 kHz 대 지속음 상위 + 019(P83 최대 정상)
VAL_SLIDE = ["001", "017", "024", "028", "018"]
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})

SOS_HP = signal.butter(4, 1000, "highpass", fs=FS, output="sos")
SOS_B = signal.butter(4, [B_LO, B_HI], "bandpass", fs=FS, output="sos")
SOS_W = signal.butter(4, [1000, 12400], "bandpass", fs=FS, output="sos")


def band_env(x):
    e = np.abs(signal.hilbert(signal.sosfiltfilt(SOS_B, x)))[TR:-TR]
    return e


def kurt_env(e):
    p = e ** 2
    return float((p ** 2).mean() / p.mean() ** 2)


def ar_whiten_self(x, p=P_AR):
    r = np.array([x[:len(x) - k] @ x[k:] for k in range(p + 1)])   # 필요한 지연 0..p만 (full 상관은 너무 느림)
    a = linalg.solve_toeplitz(r[:p], r[1:p + 1])
    w = signal.lfilter(np.r_[1, -a], [1], x)
    w[:p] = 0.0
    return w


def band_share(x):
    b = signal.sosfiltfilt(SOS_B, x)[TR:-TR]; w = signal.sosfiltfilt(SOS_W, x)[TR:-TR]
    return (b ** 2).sum() / (w ** 2).sum()


def p83(x):
    f, P = signal.welch(x - x.mean(), fs=FS, nperseg=8192)
    L = np.log10(P); df = f[1] - f[0]
    Sm = uniform_filter1d(L, max(1, int(round(55 / df))))
    PR = Sm - median_filter(Sm, size=int(round(1200 / df)), mode="nearest")
    band = (f >= B_LO) & (f < B_HI); ia = np.flatnonzero(band)
    k = ia[PR[band].argmax()]; h = PR[k] / 2; l = r = k
    while l > 0 and PR[l - 1] >= h:
        l -= 1
    while r < len(PR) - 1 and PR[r + 1] >= h:
        r += 1
    return float(PR[k]), float(f[r] - f[l])


def measure(x):
    xh = signal.sosfiltfilt(SOS_HP, x - x.mean())
    w = ar_whiten_self(xh)
    e0, e1 = band_env(xh), band_env(w)
    P, W = p83(x)
    return dict(K_raw=kurt_env(e0), K_white=kurt_env(e1), keep=band_share(w) / band_share(xh), P83=P, W83=W,
                rule_spec=(P >= 0.75) and (W >= 150.0)), e0, e1


X, y, names = load_train(); names = [str(n) for n in names]
vd = os.path.join(ROOT, "data", "TeamProject_val", "val"); vd = vd if os.path.isdir(vd) else os.path.dirname(vd)
Xv, nv = load_folder(vd); nv = [str(n) for n in nv]
rows, ENV = [], {}
for grp, XX, NN, YY in [("train", X, names, y), ("val", Xv, nv, [np.nan] * len(nv))]:
    for x, n, yy in zip(XX, NN, YY):
        m, e0, e1 = measure(np.asarray(x, float))
        lab = "val" if grp == "val" else ("고장" if yy == 1 else "정상")
        rows.append(dict(set=grp, file=n, y=yy, lab=lab, num=n[-3:], **m))
        ENV[(lab, n[-3:])] = (e0, e1)
D = pd.DataFrame(rows)
D.to_csv(os.path.join(FIG, "파일표.csv"), index=False, encoding="utf-8-sig")


def pick(lab, nums):
    return D[(D.lab == lab) & D.num.isin(nums)].set_index("num").loc[[n for n in nums if n in set(D[D.lab == lab].num)]]


# %% 결과 표
txt = ["정의: 대역 8200–8450 Hz 포락 첨도 K = mean(e^4)/mean(e^2)^2 (지속음 1 · 잡음 2 · 충격 > 2). "
       "K_raw = 1 kHz 고역통과 신호, K_white = 파일별 AR(32) 백색화 후. keep = 백색화 후 대역 몫 ÷ 원래 대역 몫", ""]
cols = ["P83", "W83", "rule_spec", "K_raw", "K_white", "keep"]
fmt = lambda r: f"P83 {r.P83:.2f} · W83 {r.W83:.0f} · 스펙트럼 갈래 {'켜짐' if r.rule_spec else '-'} · K_raw {r.K_raw:.2f} · K_white {r.K_white:.2f} · keep {r.keep:.2f}"
for title, lab, nums in [("학습 8.3k 규칙 고장 6", "고장", RULE_ABN), ("val 미끄러지는 지속음 5", "val", VAL_SLIDE),
                         ("학습 정상 8 kHz 지속음 · P83 상위", "정상", TONE_NOR)]:
    txt.append(f"## {title}")
    for num, r in pick(lab, nums).iterrows():
        txt.append(f"  {lab} {num}: " + fmt(r))
    txt.append("")
txt.append("## 그룹 분포 (중앙 [5 % – 95 %])")
for c in ["K_raw", "K_white", "keep"]:
    s = []
    for lab in ["정상", "고장", "val"]:
        v = D[D.lab == lab][c]
        s.append(f"{lab} {v.median():.2f} [{v.quantile(.05):.2f} – {v.quantile(.95):.2f}]")
    txt.append(f"  {c}: " + " · ".join(s))
txt.append("")
txt.append("## 통과 조건 (K > 2 = 충격 쪽)")
v1 = D[(D.lab == "val") & (D.num == "001")].iloc[0]
for c in ["K_raw", "K_white"]:
    a = pick("고장", RULE_ABN)[c]
    ok = bool((a > 2).all() and v1[c] <= 2)
    txt.append(f"  {c}: 규칙 고장 6 중 K > 2 = {int((a > 2).sum())}/6 (최소 {a.min():.2f}, {a.idxmin()}) · val 001 = {v1[c]:.2f} → {'통과' if ok else '불통과'}")
for c in ["K_raw", "K_white"]:
    txt.append(f"  [참고] {c} ≤ 2 비율: 정상 {(D[D.lab=='정상'][c] <= 2).mean():.0%} · 고장 {(D[D.lab=='고장'][c] <= 2).mean():.0%} · val {(D[D.lab=='val'][c] <= 2).mean():.0%}")
txt.append("")
txt.append("## 스펙트럼 갈래 켜진 파일 전부")
for _, r in D[D.rule_spec].iterrows():
    txt.append(f"  {r.lab} {r.num}: " + fmt(r))
open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8").write("\n".join(txt))

# %% 그림 1 — K_raw × K_white 산점도
fig, ax = plt.subplots(figsize=(10, 7.5))
for lab, col, s, a in [("정상", NOR, 18, .45), ("고장", ABN, 26, .7), ("val", VAL, 26, .7)]:
    d = D[D.lab == lab]
    ax.scatter(d.K_raw, d.K_white, s=s, c=col, alpha=a, lw=0, label=f"{'학습 ' if lab != 'val' else ''}{lab} ({len(d)})")
hl = [("고장", RULE_ABN, ABN, "o"), ("val", VAL_SLIDE, VAL, "D"), ("정상", ["014", "053", "019"], NOR, "s")]
for lab, nums, col, mk in hl:
    for num, r in pick(lab, nums).iterrows():
        ax.scatter(r.K_raw, r.K_white, s=90, facecolor=col, edgecolor="white", lw=2, marker=mk, zorder=5)
        pre = {"고장": "A", "val": "V", "정상": "N"}[lab]
        off = {"V001": (-14, -20), "N053": (8, -14), "N014": (8, 6)}.get(pre + num, (6, 4))
        ax.annotate(pre + num, (r.K_raw, r.K_white), xytext=off, textcoords="offset points", fontsize=11, color=INK)
for v in (1, 2):
    ax.axvline(v, color=MUTED if v == 2 else GRID, lw=1, ls="--"); ax.axhline(v, color=MUTED if v == 2 else GRID, lw=1, ls="--")
ax.set_xscale("log"); ax.set_yscale("log")
ax.text(2.03, 0.02, "잡음 K = 2", color=MUTED, fontsize=10, transform=ax.get_xaxis_transform())
ax.set_xlabel("8.2–8.45 kHz 포락 첨도 K (원신호, 1 kHz 고역)"); ax.set_ylabel("같은 K, 파일별 AR(32) 백색화 후")
ax.set_title("8.3k 대역: 지속음(K = 1) · 잡음(K = 2) · 충격(K > 2)\n큰 점 = 학습 8.3k 규칙 고장 6(A) · val 미끄러지는 지속음(V) · 8 kHz 지속음 정상(N)", fontsize=13, loc="left")
ax.legend(frameon=False, loc="upper left")
ax.grid(color=GRID, lw=.6)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_대역첨도_산점도.png"), dpi=150); plt.close(fig)

# %% 그림 2 — 대역 포락 시간 추적 (원신호 vs 백색화)
show = [("val", "001"), ("고장", "030"), ("고장", "002"), ("고장", "006"), ("정상", "014"), ("val", "017")]
show = [k for k in show if k in ENV]
t = np.arange(len(next(iter(ENV.values()))[0])) / FS + TR / FS
fig, axes = plt.subplots(len(show), 2, figsize=(13, 2.0 * len(show)), sharex=True)
for i, (lab, num) in enumerate(show):
    e0, e1 = ENV[(lab, num)]
    r = D[(D.lab == lab) & (D.num == num)].iloc[0]
    col = {"val": VAL, "고장": ABN, "정상": NOR}[lab]
    for j, (e, k) in enumerate([(e0, r.K_raw), (e1, r.K_white)]):
        a = axes[i, j]
        a.plot(t, e / np.median(e), color=col, lw=.6)
        a.set_title(f"{lab} {num} · K = {k:.2f}" + (f" · P83 {r.P83:.2f}" if j == 0 else ""), fontsize=11, loc="left")
        a.set_ylabel("포락 ÷ 중앙", fontsize=10); a.grid(color=GRID, lw=.5)
axes[0, 0].text(0, 1.35, "원신호 (1 kHz 고역) 8.2–8.45 kHz 포락", transform=axes[0, 0].transAxes, fontsize=12, color=INK)
axes[0, 1].text(0, 1.35, "파일별 AR(32) 백색화 후", transform=axes[0, 1].transAxes, fontsize=12, color=INK)
for a in axes[-1]:
    a.set_xlabel("시간 (s)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_대역포락_시간.png"), dpi=150); plt.close(fig)
print("done")

# %% 그림 3 — 왜 첨도가 가르나 (설명용, 11-23 사용자 요청)
#   ① 평균 스펙트럼: 001(지속음)과 030(충격)이 둘 다 8.3k 봉우리 → 규칙은 구분 못 함
#   ② 같은 대역을 시간으로 펼침: 순간 파워 p(t) = e² ÷ 평균(e²) (평균이 1이 되게 맞춤)
#   ③ 에너지 몰림 곡선: 세기 큰 순간부터 시간 x %가 전체 에너지의 몇 %를 갖나. K = 평균(p²)
SIG = {("val", n[-3:]): x for x, n in zip(Xv, nv)}
SIG.update({("고장" if yy == 1 else "정상", n[-3:]): x for x, n, yy in zip(X, names, y)})
CASES = [("val", "001", "지속음 — val 001", VAL), ("정상", "014", "잡음 — 정상 014", NOR), ("고장", "030", "충격 울림 — 고장 030", ABN)]
fig = plt.figure(figsize=(15, 12.5))
gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.25], hspace=.55, wspace=.25)
curves = []
for j, (lab, num, title, col) in enumerate(CASES):
    x = np.asarray(SIG[(lab, num)], float); r = D[(D.lab == lab) & (D.num == num)].iloc[0]
    xh = signal.sosfiltfilt(SOS_HP, x - x.mean())
    f, P = signal.welch(xh, fs=FS, nperseg=8192); m = (f >= 7000) & (f <= 9500)
    a = fig.add_subplot(gs[0, j])
    a.axvspan(B_LO / 1000, B_HI / 1000, color=GRID, alpha=.9, lw=0)
    a.plot(f[m] / 1000, 10 * np.log10(P[m]), color=col, lw=1.2)
    a.set_title(f"{title}\n봉우리 돌출 P83 = {r.P83:.2f}" + ("  → 규칙 켜짐" if r.rule_spec else ""), loc="left", fontsize=12)
    a.set_xlabel("주파수 (kHz)", fontsize=10); a.grid(color=GRID, lw=.5)
    if j == 0:
        a.set_ylabel("① 평균 스펙트럼 (dB)", fontsize=12)
    e = band_env(xh); p = e ** 2 / (e ** 2).mean(); tt = np.arange(len(p)) / FS + TR / FS
    a = fig.add_subplot(gs[1, j])
    a.plot(tt, p, color=col, lw=.6); a.axhline(1, color=INK, lw=1, ls="--")
    a.set_title(f"가장 센 순간 = 평균의 {p.max():.0f}배", loc="left", fontsize=12)
    a.set_xlabel("시간 (s)", fontsize=10); a.grid(color=GRID, lw=.5)
    if j == 0:
        a.set_ylabel("② 순간 파워 ÷ 평균", fontsize=12)
    ps = np.sort(p)[::-1]; cum = np.cumsum(ps) / ps.sum(); xs = np.arange(1, len(ps) + 1) / len(ps) * 100
    curves.append((title, col, xs, cum, float((p ** 2).mean())))
a = fig.add_subplot(gs[2, :2])
for title, col, xs, cum, K in curves:
    a.plot(xs, cum * 100, color=col, lw=2.2, label=f"{title}   K = {K:.1f}")
    i1 = np.searchsorted(xs, 1.0)
    a.scatter([xs[i1]], [cum[i1] * 100], color=col, s=50, zorder=5, edgecolor="white", lw=1.5)
    a.annotate(f"{cum[i1] * 100:.0f} %", (xs[i1], cum[i1] * 100), xytext={"val 001": (10, -14)}.get(title.split("— ")[-1], (8, 2)), textcoords="offset points", fontsize=11, color=INK)
a.axvline(1, color=MUTED, lw=1, ls=":")
a.set_xscale("log"); a.set_xlim(0.05, 100); a.set_ylim(0, 102)
a.set_xticks([0.1, 1, 10, 100]); a.set_xticklabels(["0.1 %", "1 %", "10 %", "100 %"])
a.set_xlabel("세기가 큰 순간부터 모은 시간 비율 (로그)"); a.set_ylabel("③ 그 시간이 가진 에너지 몫 (%)", fontsize=12)
a.set_title("에너지가 시간 1 % 안에 얼마나 몰렸나 (점 = 시간 1 % 지점)", loc="left", fontsize=12)
a.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.32, 0.78), fontsize=11); a.grid(color=GRID, lw=.5)
a = fig.add_subplot(gs[2, 2]); a.axis("off")
a.text(0, 1, "첨도 K = 평균( p² )\n(p = 순간 파워 ÷ 평균)\n\n"
       "• 늘 1이면 → 1² 평균 = 1\n   (지속음)\n\n"
       "• 무작위로 출렁이면 → 2\n   (잡음, 이론값)\n\n"
       "• 99 %는 0, 1 %만 100이면\n   → 0.01 × 100² = 100\n   (충격)\n\n"
       "같은 평균 에너지라도\n몰릴수록 제곱이 키워 준다", va="top", fontsize=12.5, color=INK, linespacing=1.35)
fig.suptitle("평균 스펙트럼으로는 둘 다 8.3k 봉우리 → 시간으로 펼치면 판판함 대 뾰족함 → 첨도는 그 차이를 숫자 하나로", x=.06, ha="left", fontsize=14)
fig.savefig(os.path.join(FIG, "3_왜갈리나.png"), dpi=150, bbox_inches="tight"); plt.close(fig)
print("fig3 done")
