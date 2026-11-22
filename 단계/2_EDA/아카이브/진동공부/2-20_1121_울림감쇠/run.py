# %% [markdown]
# 2-20 울림이 꺼지는 속도(감쇠) — 고장 충격은 빨리 꺼지나, 8.3 kHz 봉우리가 넓은 이유는 무엇인가
# - 배운 것: 감쇠 사인파(Stage 2), 봉우리 반높이 폭 ≈ 2 × 감쇠비 × 주파수(Stage 3), 짧은 구간 = 듬성한 주파수 칸(Stage 4)
# - 충격 찾기는 src/final.impact_rings와 같다 (1 kHz 고역통과 → 포락선 → 바닥의 6배 넘는 봉우리, 5 ms 간격)
# - 감쇠를 연속값으로 다시 잰다 (사전은 0.01·0.02·0.04·0.08 네 칸뿐):
#   울림 주파수 = 충격 뒤 8 ms를 0을 채워 길게 FFT한 최고점 (보간으로 125 Hz 칸보다 촘촘하게)
#   감쇠 = 그 주파수 ±15% 대역 포락선의 log를 직선으로 맞춘 기울기 (최고점 → 20%로 떨어질 때까지, 최대 8 ms, 다음 충격 전까지)
#   감쇠비 ζ = −기울기 / (2π × 울림 주파수)
# - Q4: 8.3 kHz 봉우리 폭(W83)이 넓은 이유 = (가) 빨리 꺼짐 → 폭 ≈ 2ζf  vs  (나) 충격마다 음이 다름 → 충격 주파수의 흩어짐
# 실행: conda run -n qe-phm python run.py  (결과 숫자는 result.txt, 충격 표는 impacts.csv)

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.dirname(HERE))
from _common import *
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, welch
from scipy.ndimage import uniform_filter1d, median_filter
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from src.final import impact_rings

X, y, names, log, save, flush = setup(HERE)
nor, abn = y == 0, y == 1
SEG = int(0.008 * FS)
Xc = X - X.mean(1, keepdims=True)
X1k = sosfiltfilt(butter(4, 1000, btype="highpass", fs=FS, output="sos"), Xc, axis=-1)
lp500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")
NFFT = 4096
FR = np.fft.rfftfreq(NFFT, 1 / FS)
WIN = np.hanning(SEG)

# %% 충격마다 주파수·감쇠 (연속값)
rows, examples = [], {}
for i, x in enumerate(X1k):
    env = sosfiltfilt(lp500, np.abs(hilbert(x)))
    pk, _ = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
    pk = pk[pk + SEG <= len(x)]
    floor = 3 * np.median(env)
    for k, p in enumerate(pk):
        s = x[p:p + SEG]
        A = np.abs(np.fft.rfft(s * WIN, NFFT))
        j = int(np.argmax(A[1:-1])) + 1
        a, b, c = np.log(A[j - 1:j + 2] + 1e-20)
        f0 = FR[j] + 0.5 * (a - c) / (a - 2 * b + c) * (FR[1] - FR[0])
        if f0 < 1200 or f0 > 12000:
            continue
        lo, hi = max(f0 * .85, 1000), min(f0 * 1.15, 12400)
        seg_end = min(p + SEG, pk[k + 1] if k + 1 < len(pk) else len(x))
        a0 = max(p - 50, 0)
        xb = sosfiltfilt(butter(2, [lo, hi], btype="bandpass", fs=FS, output="sos"), x[a0:seg_end + 50])
        eb = np.abs(hilbert(xb))[p - a0:seg_end - a0]
        m0 = int(np.argmax(eb[:25]))                          # 1 ms 안 최고점
        e = eb[m0:]
        stop = np.flatnonzero((e < .2 * e[0]) | (e < floor))
        n_fit = stop[0] if len(stop) else len(e)
        if n_fit < 12:                                       # 0.5 ms 미만이면 기울기 못 믿음
            continue
        t = np.arange(n_fit) / FS
        slope, icpt = np.polyfit(t, np.log(e[:n_fit]), 1)
        zeta = -slope / (2 * np.pi * f0)
        rows.append((i, p, f0, zeta, n_fit / FS * 1000, e[0]))
        if names[i] in ("nor_001", "abn_002", "abn_022") and names[i] not in examples:
            examples[names[i]] = (x[a0:seg_end + 50], p - a0, m0, n_fit, slope, icpt, f0, zeta, eb)
D = pd.DataFrame(rows, columns=["file", "pos", "f", "zeta", "fit_ms", "amp"])
D["name"] = [names[i] for i in D.file]; D["y"] = y[D.file]
D = D[D.zeta > 0]
D.to_csv(os.path.join(HERE, "impacts.csv"), index=False, encoding="utf-8-sig")
log(f"충격 {len(D)}개 (정상 {int((D.y==0).sum())} · 고장 {int((D.y==1).sum())}), 감쇠 맞춤 길이 중앙 {D.fit_ms.median():.2f} ms")

# %% 사전 방식(4칸)과 비교
R = impact_rings(X)
zd = pd.Series(R[:, 2]).groupby(R[:, 0].astype(int)).median()
zc = D.groupby("file").zeta.median()
both = zd.index.intersection(zc.index)
log(f"파일별 감쇠비 중앙: 사전(4칸) vs 연속값 순위상관 {spearmanr(zd[both], zc[both]).correlation:.2f} (파일 {len(both)}개)")

# %% 그림 1: 예시 충격 3개 — 감쇠 사인파와 포락선 직선
fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
for ax, n_ in zip(axs, ("nor_001", "abn_002", "abn_022")):
    if n_ not in examples:
        ax.set_title(f"{n_}: 맞춤 충격 없음"); continue
    seg, off, m0, n_fit, slope, icpt, f0, zeta, eb = examples[n_]
    tt = (np.arange(len(seg)) - off) / FS * 1000
    ax.plot(tt, seg, color="#999999", lw=.6, label="1 kHz 고역통과 신호")
    te = (np.arange(len(eb))) / FS * 1000
    ax.plot(te, eb, color=NOR if n_.startswith("nor") else ABN, lw=1.2, label="울림 대역 포락선")
    tf = np.arange(n_fit) / FS
    ax.plot((tf * 1000) + m0 / FS * 1000, np.exp(icpt + slope * tf), "k--", lw=1.5, label="맞춘 지수 감쇠")
    ax.set_xlim(-1, 8); ax.set_xlabel("충격 뒤 시간 (ms)")
    ax.set_title(f"{n_}: {f0/1000:.2f} kHz · 감쇠비 {zeta:.3f}")
axs[0].set_ylabel("가속도 (g)"); axs[0].legend(fontsize=8)
fig.suptitle("충격 하나의 울림: 포락선(겉모양)이 얼마나 빨리 내려가나 = 감쇠", y=1.02)
save(fig, "1_예시충격_감쇠맞춤")

# %% 파일별 감쇠 요약
F = D.groupby("file").agg(n=("zeta", "size"), zmed=("zeta", "median"), fmed=("f", "median"),
                          zhi=("zeta", lambda v: (v >= .04).mean()))
F["y"] = y[F.index]
for col, lab in (("zmed", "감쇠비 중앙"), ("zhi", "감쇠비 ≥ 0.04 충격 비율")):
    a, b = F.loc[F.y == 0, col], F.loc[F.y == 1, col]
    log(f"{lab}: 정상 중앙 {a.median():.3f} · 고장 중앙 {b.median():.3f} · AUC {roc_auc_score(F.y, F[col]):.2f} (충격 있는 파일 정상 {len(a)} · 고장 {len(b)})")
log(f"충격 없는 파일: 정상 {int(nor.sum() - (F.y==0).sum())} · 고장 {int(abn.sum() - (F.y==1).sum())}")

# %% 그림 2: 주파수 × 감쇠 (충격 하나 = 점), 오른쪽 = 파일별 감쇠 중앙
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), gridspec_kw=dict(width_ratios=[3, 1]))
for lab, c, m in (("정상 충격", NOR, D.y == 0), ("고장 충격", ABN, D.y == 1)):
    ax.scatter(D.f[m] / 1000, D.zeta[m], s=6, alpha=.35, color=c, label=f"{lab} {int(m.sum())}")
ax.set_yscale("log"); ax.set_ylim(.002, .3); plain_log(ax)
ax.set_xlabel("울림 주파수 (kHz)"); ax.set_ylabel("감쇠비 (로그 눈금, 클수록 빨리 꺼짐)")
ax.set_title("충격 하나하나의 울림 음 높이와 꺼지는 속도")
ax.legend(markerscale=3); ax.grid(alpha=.3, which="both")
rng = np.random.default_rng(0)
for k, (m, c) in enumerate(((F.y == 0, NOR), (F.y == 1, ABN))):
    ax2.scatter(k + rng.uniform(-.15, .15, m.sum()), F.zmed[m], s=12, color=c, alpha=.7)
    ax2.hlines(F.zmed[m].median(), k - .25, k + .25, color="k")
for n_ in ("abn_002", "abn_015", "abn_017", "abn_022", "abn_014"):
    i = names.index(n_)
    if i in F.index:
        ax2.annotate(n_[-3:], (1.2, F.zmed[i]), fontsize=8)
ax2.set_yscale("log"); plain_log(ax2); ax2.set_xticks([0, 1]); ax2.set_xticklabels(["정상", "고장"])
ax2.set_title("파일별 감쇠비 중앙값"); ax2.grid(alpha=.3, which="both")
save(fig, "2_주파수x감쇠_충격과파일")

# %% Q4: 8.3 kHz 봉우리 폭 — 빨리 꺼짐인가, 음이 흩어짐인가
f8, P8 = welch(Xc, fs=FS, nperseg=8192)
L8 = np.log10(P8); d8 = f8[1] - f8[0]
Sm = uniform_filter1d(L8, max(1, int(round(55 / d8))), axis=1)
PR = Sm - median_filter(Sm, size=(1, int(round(1200 / d8))), mode="nearest")
band = (f8 >= 8200) & (f8 < 8450); ia = np.flatnonzero(band)
W83 = np.zeros(len(X)); P83 = PR[:, band].max(1)
for i in range(len(X)):
    k = ia[PR[i, band].argmax()]; h = PR[i, k] / 2; l = r = k
    while l > 0 and PR[i, l - 1] >= h: l -= 1
    while r < PR.shape[1] - 1 and PR[i, r + 1] >= h: r += 1
    W83[i] = f8[r] - f8[l]
E8 = D[(D.f >= 8000) & (D.f < 8700)]
G = E8.groupby("file").agg(n=("f", "size"), fstd=("f", "std"), zmed=("zeta", "median"), fmed=("f", "median"))
G = G[G.n >= 3]
G["W83"] = W83[G.index]; G["P83"] = P83[G.index]; G["y"] = y[G.index]
G["w_damp"] = 2 * G.zmed * G.fmed            # 빨리 꺼짐만으로 생기는 반높이 폭
G["w_spread"] = 2.355 * G.fstd              # 충격마다 음이 다른 것만으로 생기는 폭 (정규분포 반높이 폭)
log(f"\nQ4 — 8.0–8.7 kHz로 울린 충격 ≥ 3개인 파일: 정상 {int((G.y==0).sum())} · 고장 {int((G.y==1).sum())}")
for lab, m in (("정상", G.y == 0), ("고장", G.y == 1)):
    g = G[m]
    if len(g):
        log(f"  {lab}: 측정 폭 중앙 {g.W83.median():.0f} Hz · 감쇠로 설명되는 폭 {g.w_damp.median():.0f} Hz · 흩어짐으로 설명되는 폭 {g.w_spread.median():.0f} Hz")
if len(G) >= 5:
    log(f"  순위상관: 측정 폭 ↔ 감쇠 폭 {spearmanr(G.W83, G.w_damp).correlation:.2f}, 측정 폭 ↔ 흩어짐 폭 {spearmanr(G.W83, G.w_spread).correlation:.2f}")
for n_ in G.index:
    log(f"    {names[n_]}: 충격 {G.n[n_]} · 측정 폭 {G.W83[n_]:.0f} · 감쇠 폭 {G.w_damp[n_]:.0f} · 흩어짐 폭 {G.w_spread[n_]:.0f} · 돌출 {G.P83[n_]:.2f}")

# %% 그림 3: Q4
fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, col, lab in zip(axs, ("w_damp", "w_spread"), ("(가) 빨리 꺼짐: 2 × 감쇠비 × 주파수", "(나) 충격마다 음이 다름: 주파수 흩어짐 × 2.355")):
    for m, c in ((G.y == 0, NOR), (G.y == 1, ABN)):
        ax.scatter(G[col][m], G.W83[m], s=40, color=c, alpha=.8, edgecolor="k", lw=.3)
    for n_ in G.index:
        ax.annotate(names[n_].replace("abn_", "A").replace("nor_", "N"), (G[col][n_], G.W83[n_]), fontsize=7)
    mx = max(G[col].max(), G.W83.max()) * 1.1
    ax.plot([0, mx], [0, mx], ":", color="gray"); ax.set_xlim(0, mx)
    ax.set_xlabel(f"{lab} (Hz)"); ax.set_title(lab, fontsize=10); ax.grid(alpha=.3)
axs[0].set_ylabel("실제 8.3 kHz 봉우리 반높이 폭 (Hz)")
fig.suptitle("8.3 kHz 봉우리 폭은 어느 쪽으로 설명되나 (점선 위 = 그 원인만으로는 폭이 모자람) — 파랑 정상, 주황 고장", y=1.02)
save(fig, "3_Q4_8k봉우리폭_원인")

# %% 특이 파일
log("\n특이 파일 (충격 수 · 감쇠비 중앙 · 정상 백분위)")
zfull = np.full(len(X), np.nan); zfull[F.index] = F.zmed
for n_ in ("nor_001", "abn_002", "abn_014", "abn_015", "abn_017", "abn_022"):
    i = names.index(n_)
    pc = (F.zmed[F.y == 0] < zfull[i]).mean() * 100 if not np.isnan(zfull[i]) else np.nan
    log(f"  {n_}: 충격 {int(F.n.get(i, 0))} · 감쇠비 중앙 {zfull[i]:.3f} · 정상 중 {pc:.0f}%")
flush()
