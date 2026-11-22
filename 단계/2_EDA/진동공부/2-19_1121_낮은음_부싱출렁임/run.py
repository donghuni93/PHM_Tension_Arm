# %% [markdown]
# 2-19 낮은 음 찾기 — 암이 부싱 위에서 출렁이는 낮은 고유진동수(20–200 Hz)는 어디이고, 고장 때 옮겨 가나
# - 배운 것: 질량-스프링(Stage 2) → 봉우리 = 고유진동수(Stage 3) → 촘촘히 보려면 긴 구간(Stage 4, 분해능 ≈ 1/구간 길이)
# - 방법 (라벨 무관 정의): 평균 제거 → Welch 16384점(0.66 s, 칸 간격 1.5 Hz), 반씩 겹쳐 5구간 평균
#   공통 봉우리 = 150개 전체 중앙 PSD에서 찾은 봉우리 (정상·고장 구분 없이)
#   파일별 봉우리 위치 = 공통 봉우리 ±20% 안에서 그 파일 PSD가 가장 높은 곳
#   모양 = 20–200 Hz log PSD에서 그 파일의 평균을 뺀 것 (크기 무관)
# - 라벨은 색칠과 비교에만. AUC 그림은 "사후" 관찰
# 실행: conda run -n qe-phm python run.py  (결과 숫자는 result.txt)

# %% 설정
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
sys.path.insert(0, os.path.dirname(HERE))
from _common import *
from scipy.signal import welch, find_peaks
from scipy.ndimage import uniform_filter1d
from sklearn.metrics import roc_auc_score

X, y, names, log, save, flush = setup(HERE)
nor, abn = y == 0, y == 1
Xc = X - X.mean(1, keepdims=True)
FOCUS = ["abn_015", "abn_017", "abn_022", "nor_097"]

# %% 긴 구간 PSD
f, P = welch(Xc, fs=FS, nperseg=16384, axis=-1)
L = np.log10(P + 1e-20)
df = f[1] - f[0]
log(f"Welch 16384점: 칸 간격 {df:.2f} Hz, 구간 길이 {16384/FS:.2f} s")
lo_m = (f >= 2) & (f <= 300)

# %% 그림 0: 분해능 — 같은 파일을 짧은 구간(AE 입력과 같은 2048점) vs 긴 구간으로
i0 = names.index("nor_001")
fig, ax = plt.subplots(figsize=(10, 4))
for nps, c, lab in ((2048, "#999999", "2048점 (0.08 s, 칸 12.2 Hz) — 지금 AE 입력"), (16384, NOR, "16384점 (0.66 s, 칸 1.5 Hz)")):
    ff, pp = welch(Xc[i0], fs=FS, nperseg=nps)
    m = (ff >= 2) & (ff <= 300)
    ax.semilogx(ff[m], np.log10(pp[m]), "o-" if nps == 2048 else "-", ms=4, color=c, label=lab)
ax.set_xlabel("주파수 (Hz)"); ax.set_ylabel("log10 파워"); ax.set_title("nor_001 낮은 주파수: 구간이 짧으면 칸이 듬성듬성해 봉우리가 뭉개진다")
ax.legend(); ax.grid(alpha=.3, which="both")
save(fig, "0_분해능_짧은구간vs긴구간")

# %% 공통 봉우리 (150개 전체 중앙, 라벨 무관)
med_all = np.median(L, 0)
sm = uniform_filter1d(med_all, 3)
band = (f >= 10) & (f <= 300)
ib = np.flatnonzero(band)
pk, pr = find_peaks(sm[band], prominence=0.08)
peaks = f[ib[pk]]
prom = pr["prominences"]
log("공통 봉우리 (전체 중앙 PSD, 10–300 Hz, 돌출 ≥ 0.08 dec):")
for fr, p_ in zip(peaks, prom):
    log(f"  {fr:6.1f} Hz  돌출 {p_:.2f} dec")

# %% 그림 1: 정상 5–95% 띠 + 중앙 + 공통 봉우리
fig, ax = plt.subplots(figsize=(12, 5))
q5, q50, q95 = np.percentile(L[nor][:, lo_m], [5, 50, 95], 0)
ax.fill_between(f[lo_m], q5, q95, color=NOR, alpha=.18, label="정상 5–95%")
ax.plot(f[lo_m], q50, color=NOR, lw=2, label="정상 중앙")
ax.plot(f[lo_m], np.median(L[abn][:, lo_m], 0), color=ABN, lw=2, label="고장 중앙")
for n_, c in zip(FOCUS, ["#b5179e", "#7b2cbf", "#d00000", "#555555"]):
    ax.plot(f[lo_m], L[names.index(n_)][lo_m], color=c, lw=.9, label=n_)
for fr in peaks:
    ax.axvline(fr, color="k", ls=":", lw=.8)
    ax.text(fr, ax.get_ylim()[1] if False else q95.max() + .3, f"{fr:.0f}", ha="center", fontsize=8)
ax.set_xscale("log"); ax.set_xlim(2, 300)
ax.set_xlabel("주파수 (Hz, 로그 눈금)"); ax.set_ylabel("log10 파워 (g²/Hz)")
ax.set_title("낮은 주파수 PSD — 점선 = 150개 전체 중앙에서 찾은 공통 봉우리(라벨 안 봄)")
ax.legend(fontsize=8, ncol=2); ax.grid(alpha=.3, which="both")
save(fig, "1_낮은주파수_정상띠_공통봉우리")

# %% 파일별 봉우리 위치 (공통 봉우리 ±20%)
Ls = uniform_filter1d(L, 3, axis=1)
pos = np.full((len(X), len(peaks)), np.nan)
for j, fr in enumerate(peaks):
    w = (f >= fr * 0.8) & (f <= fr * 1.2)
    iw = np.flatnonzero(w)
    pos[:, j] = f[iw[Ls[:, w].argmax(1)]]
log("\n파일별 봉우리 위치 (공통 봉우리 ±20% 안 최고점): 정상 중앙 [25–75%] / 고장 중앙 [25–75%] / 위치 AUC(사후)")
for j, fr in enumerate(peaks):
    a, b = pos[nor, j], pos[abn, j]
    auc = roc_auc_score(y, pos[:, j])
    log(f"  {fr:6.1f} Hz: 정상 {np.median(a):6.1f} [{np.percentile(a,25):.1f}–{np.percentile(a,75):.1f}]  "
        f"고장 {np.median(b):6.1f} [{np.percentile(b,25):.1f}–{np.percentile(b,75):.1f}]  AUC {auc:.2f}")

# %% 그림 2: 봉우리 위치 — 고장 때 옮겨 가나
rng = np.random.default_rng(0)
fig, axs = plt.subplots(1, len(peaks), figsize=(2.6 * len(peaks), 4.5), squeeze=False)
for j, (ax, fr) in enumerate(zip(axs[0], peaks)):
    for k, (m, c) in enumerate(((nor, NOR), (abn, ABN))):
        ax.scatter(k + rng.uniform(-.15, .15, m.sum()), pos[m, j], s=10, color=c, alpha=.6)
        ax.hlines(np.median(pos[m, j]), k - .25, k + .25, color="k")
    for n_ in FOCUS[:3]:
        i = names.index(n_); ax.annotate(n_[-3:], (1.2, pos[i, j]), fontsize=7)
    ax.axhline(fr, color="gray", ls=":")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상", "고장"])
    ax.set_title(f"공통 {fr:.0f} Hz 근처", fontsize=10)
axs[0, 0].set_ylabel("그 파일의 봉우리 위치 (Hz)")
fig.suptitle("파일마다 공통 봉우리 근처에서 가장 높은 곳 — 고장이 위·아래로 몰리면 '단단함이 바뀜'")
save(fig, "2_봉우리위치_정상vs고장")

# %% 모양 (크기 무관)
sh_m = (f >= 20) & (f <= 200)
S = L[:, sh_m] - L[:, sh_m].mean(1, keepdims=True)
lev = L[:, sh_m].mean(1)
auc_shape = np.array([roc_auc_score(y, S[:, k]) for k in range(S.shape[1])])
log(f"\n20–200 Hz 크기(평균 log 파워) AUC {roc_auc_score(y, lev):.2f}")
log(f"20–200 Hz 모양 칸별 AUC: 최소 {auc_shape.min():.2f} ({f[sh_m][auc_shape.argmin()]:.0f} Hz), 최대 {auc_shape.max():.2f} ({f[sh_m][auc_shape.argmax()]:.0f} Hz) (사후)")

# %% 그림 3: 모양 열지도 (정상 위, 고장 아래, 각각 20–200 Hz 크기 순)
order = np.r_[np.flatnonzero(nor)[np.argsort(lev[nor])], np.flatnonzero(abn)[np.argsort(lev[abn])]]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(11, 9), gridspec_kw=dict(height_ratios=[3, 1]), sharex=True)
im = ax.imshow(S[order], aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2,
               extent=[f[sh_m][0], f[sh_m][-1], len(order), 0])
ax.axhline(nor.sum(), color="k", lw=1.5)
ax.text(205, nor.sum() / 2, "정상", va="center"); ax.text(205, nor.sum() + abn.sum() / 2, "고장", va="center")
for fr in peaks[(peaks >= 20) & (peaks <= 200)]:
    ax.axvline(fr, color="k", ls=":", lw=.7)
ax.set_ylabel("파일 (그룹 안에서 20–200 Hz 크기 순)")
ax.set_title("20–200 Hz 모양 (각 파일 평균을 뺀 log 파워) — 빨강 = 그 파일 안에서 상대적으로 큰 주파수")
fig.colorbar(im, ax=ax, label="dec (파일 평균 대비)")
ax2.plot(f[sh_m], auc_shape, color="k"); ax2.axhline(.5, color="gray", ls=":")
ax2.fill_between(f[sh_m], .5, auc_shape, where=np.abs(auc_shape - .5) > .2, color=ABN, alpha=.3)
ax2.set_ylim(.2, .8); ax2.set_xlabel("주파수 (Hz)"); ax2.set_ylabel("AUC (사후)")
ax2.set_title("칸별 AUC: 0.5 = 못 가름, 위 = 고장에서 상대적으로 큼")
save(fig, "3_모양_열지도_AUC")

# %% 놓친·특이 파일 숫자
log("\n특이 파일: 20–200 Hz 크기의 정상 백분위")
for n_ in FOCUS + ["abn_014", "abn_002"]:
    i = names.index(n_)
    log(f"  {n_}: 크기 {pct_in_normal(lev, y, i):.0f}%  봉우리 위치 {np.round(pos[i], 1).tolist()}")
flush()
