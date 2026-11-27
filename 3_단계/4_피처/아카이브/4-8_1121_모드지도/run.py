# %% [markdown]
# 4-8 모드지도 (라벨 없이): 옛 대역 경계(2–4k · 5.5–7.75k · 8.0–8.6k · 9.25–10k)가 데이터 구조인가, 라벨에 맞춘 것인가
# 사전 규칙(라벨을 보기 전에 고정): 충격 울림 주파수(1–12 kHz) 가우시안 KDE, 폭 = Scott(h0) · 민감도 0.5·h0, 2·h0.
#   봉우리 = 최대 밀도의 10% 이상 prominence, 경계 = 인접 봉우리 사이 밀도 최저점. 라벨은 대역 확정 뒤 보고에만 쓴다.
# 실행: conda run -n qe-phm python run.py (결과는 result.txt 로 저장)
# %% 설정
import os, sys
for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): os.environ[k] = "1"
import numpy as np, pandas as pd
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d: _d = os.path.dirname(_d)
sys.path.insert(0, _d)
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm
for _f in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if any(_f == x.name for x in _fm.fontManager.ttflist): plt.rcParams["font.family"] = _f; break
plt.rcParams["axes.unicode_minus"] = False
from joblib import Parallel, delayed
from scipy.signal import welch, find_peaks, butter, sosfiltfilt, hilbert
from scipy.stats import gaussian_kde, spearmanr
from src.data import load_dataset, load_folder
from src.impacts import detect_impacts, impact_rings_v2, highpass
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000
OUT = open(os.path.join(HERE, "result.txt"), "w", encoding="utf-8")
def P(*a):
    s = " ".join(str(x) for x in a); print(s); OUT.write(s + "\n"); OUT.flush()
X, y, names = load_dataset(os.path.join(_d, "2_data", "TeamProject_train_변경"))
Xv, nv = load_folder(os.path.join(_d, "2_data", "TeamProject_val", "val"))
XA = np.vstack([X, Xv]); NA = list(names) + list(nv); N = len(XA); ntr = len(X)
yA = np.r_[y, -np.ones(len(Xv), int)]            # -1 = val (라벨 없음)
P("train", X.shape, "val", Xv.shape, "normal", (y == 0).sum(), "fault", (y == 1).sum())

# %% 충격 · 울림 · PSD (파일 단위 병렬)
def work(x):
    xh = highpass(x); imp = detect_impacts(x, _hp=xh)
    r = impact_rings_v2(x, imp=imp, _hp=xh)
    fw, Pw = welch(x - x.mean(), fs=FS, nperseg=2048)
    return imp["idx"], imp["height_x"], r, Pw, fw
if __name__ == "__main__":
    pass
res = Parallel(n_jobs=-1)(delayed(work)(x) for x in XA)
IDX = [r[0] for r in res]; RING = [r[2] for r in res]; PSD = np.array([r[3] for r in res]); FW = res[0][4]
nimp = np.array([len(i) for i in IDX])
tot = dict(normal=int(nimp[:ntr][y == 0].sum()), val=int(nimp[ntr:].sum()), fault=int(nimp[:ntr][y == 1].sum()))
P("충격 수 합계", tot); assert (tot["normal"], tot["val"], tot["fault"]) == (854, 247, 402), tot
F = np.concatenate([r[:, 0] for r in RING]); Z = np.concatenate([r[:, 1] for r in RING]); FIT = np.concatenate([r[:, 2] for r in RING])
FID = np.concatenate([np.full(len(r), i) for i, r in enumerate(RING)])

# %% 모드지도: KDE 봉우리 · 골짜기 (사전 규칙)
grid = np.linspace(1000, 12000, 2201)
def modemap(f, w=None, mult=1.0):
    k = gaussian_kde(f, weights=w); k.set_bandwidth(k.factor * mult)   # factor = Scott
    d = k(grid); d = d / d.max()
    pk, _ = find_peaks(d, prominence=0.10)
    vs = [grid[a + np.argmin(d[a:b])] for a, b in zip(pk[:-1], pk[1:])]
    return d, grid[pk], np.array(vs), k.factor * f.std(ddof=1)
MM = {}
for nm, w in (("균등", None), ("fit가중", FIT)):
    for mult in (1.0, 0.5, 2.0):
        d, pks, vs, bw = modemap(F, w, mult); MM[(nm, mult)] = (d, pks, vs)
        P(f"[KDE {nm} x{mult}] 폭 {bw:.0f} Hz | 봉우리(kHz) {np.round(pks/1000,2).tolist()} | 골짜기(kHz) {np.round(vs/1000,2).tolist()}")
# 탐색용(사전 규칙 밖, 명시): 폭 x0.5 · prominence 0.03 / 0.05 로 약한 봉우리까지
def modemap2(f, mult, prom):
    k = gaussian_kde(f); k.set_bandwidth(k.factor * mult); d = k(grid); d = d / d.max()
    pk, _ = find_peaks(d, prominence=prom); return grid[pk], np.array([grid[a + np.argmin(d[a:b])] for a, b in zip(pk[:-1], pk[1:])])
for mult, prom in ((0.5, 0.05), (0.5, 0.03), (0.3, 0.03), (0.3, 0.015)):
    pk_, vs_ = modemap2(F, mult, prom); P(f"[탐색 x{mult} prom {prom}] 봉우리 {np.round(pk_/1000,2).tolist()} | 골짜기 {np.round(vs_/1000,2).tolist()}")
cnt = np.histogram(F, bins=np.arange(875, 12126, 250))[0]; P("250 Hz 격자별 충격 수 (1.0k부터):", dict(zip((np.arange(1000, 12001, 250) / 1000).round(2).tolist(), cnt.tolist())))
LP = np.log10(PSD + 1e-30); MED = np.median(LP, 0)
sm = np.convolve(MED, np.ones(5) / 5, "same"); m = (FW >= 1000) & (FW <= 12000)
pp, _ = find_peaks(sm[m], prominence=0.05); P("PSD 풀링 중앙값 봉우리(kHz):", np.round(FW[m][pp] / 1000, 2).tolist())
d0, PK, VS = MM[("균등", 1.0)]
edges = np.r_[1000, VS, 12000]
BANDS = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
P("고정 대역(kHz):", [(round(a / 1000, 2), round(b / 1000, 2)) for a, b in BANDS])
OLD = {"2–4k": (2000, 4000), "5.5–7.75k": (5500, 7750), "8.0–8.6k": (8000, 8600), "9.25–10k": (9250, 10000)}
P("옛 경계(kHz):", {k: (a / 1000, b / 1000) for k, (a, b) in OLD.items()})

fig, ax = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
ax[0].hist(F / 1000, bins=np.arange(0.875, 12.2, 0.25), color="lightgray", density=True, label="충격 울림 f 히스토그램 (250 Hz 격자)")
ymax = ax[0].get_ylim()[1]
for (nm, mult), c, ls in zip(MM, ["k", "tab:blue", "tab:green", "tab:red", "tab:orange", "tab:purple"], ["-", "--", ":", "-", "--", ":"]):
    ax[0].plot(grid / 1000, MM[(nm, mult)][0] * ymax * 0.9, c=c, ls=ls, lw=1.2, label=f"KDE {nm} x{mult}")
for v in VS: ax[0].axvline(v / 1000, color="k", lw=1); ax[0].text(v / 1000, ymax * 0.95, f"{v/1000:.2f}", ha="center", fontsize=8)
for a, b in OLD.values(): ax[0].axvspan(a / 1000, b / 1000, color="gold", alpha=0.2)
ax[0].set_ylabel("밀도 (KDE는 최대 1로 정규화)"); ax[0].legend(fontsize=7, ncol=2); ax[0].set_title("라벨 없는 모드지도: 노랑 = 옛 대역, 검정선 = 사전 규칙(KDE Scott) 골짜기")
ax[1].plot(FW / 1000, MED, c="k", lw=1); ax[1].plot(FW / 1000, np.percentile(LP, 10, 0), c="gray", lw=.6); ax[1].plot(FW / 1000, np.percentile(LP, 90, 0), c="gray", lw=.6)
for v in VS: ax[1].axvline(v / 1000, color="k", lw=1)
for a, b in OLD.values(): ax[1].axvspan(a / 1000, b / 1000, color="gold", alpha=0.2)
ax[1].set_xlim(1, 12); ax[1].set_xlabel("주파수 (kHz)"); ax[1].set_ylabel("log10 PSD, 190파일 중앙값 · 10/90%")
fig.savefig(os.path.join(FIG, "mode_map.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 대역 확정 뒤에야 라벨 보고 (정보용)
def band_of(f): return np.clip(np.searchsorted(edges, f, side="right") - 1, 0, len(BANDS) - 1)
B = band_of(F); grp = np.where(yA[FID] < 0, "val", np.where(yA[FID] == 0, "normal", "fault"))
tab = pd.crosstab(B, grp).reindex(columns=["normal", "fault", "val"], fill_value=0)
tab.index = [f"{BANDS[i][0]/1000:.2f}–{BANDS[i][1]/1000:.2f}k" for i in tab.index]; P("대역별 충격 수(라벨은 보고용):\n" + tab.to_string())
def where(name):
    i = NA.index(name); r = RING[i]; return [(round(f / 1000, 2), int(band_of(np.array([f]))[0]), round(z, 2), round(ft, 2)) for f, z, ft in r]
for nm in ("abn_002", "nor_116", "nor_055", "nor_059", "nor_106", "abn_022", "abn_014", "abn_017"):
    if nm in NA: P(nm, "(f kHz, 대역번호, zeta, fit):", where(nm))
b83 = int(band_of(np.array([8280.0]))[0]); b94 = int(band_of(np.array([9400.0]))[0]); P("8.28k 대역번호", b83, "9.4k 대역번호", b94, "분리:", b83 != b94)
for (nm, mult), (d, pks, vs) in MM.items():
    e2 = np.r_[1000, vs, 12000]; P(f"  민감도 {nm} x{mult}: 8.28k와 9.4k 서로 다른 대역?", int(np.searchsorted(e2, 8280, "right")) != int(np.searchsorted(e2, 9400, "right")))

# %% 파일별 피처 (라벨 없음)
rows = []
for i in range(N):
    r = RING[i]; d = dict(file=NA[i], split="train" if i < ntr else "val", label=int(yA[i]), n_imp=int(nimp[i]))
    bi = band_of(r[:, 0]) if len(r) else np.array([], int)
    for b, (a, c) in enumerate(BANDS):
        k = bi == b
        d[f"share_B{b}"] = k.mean() if len(r) else np.nan
        d[f"fit_B{b}"] = r[k, 2].max() if k.any() else 0.0
        mm = (FW >= a) & (FW < c); d[f"psdprom_B{b}"] = float((LP[i, mm] - MED[mm]).mean())
    rows.append(d)
FE = pd.DataFrame(rows)
# 탐색 대역 (x0.3 · prom 0.03 골짜기, 사전 규칙 밖): 파일별 비율·최대 fit도 함께 저장 (접미사 X)
_, VSX = modemap2(F, 0.3, 0.03); eX = np.r_[1000, VSX, 12000]; P("탐색 대역 경계(kHz):", np.round(eX / 1000, 2).tolist())
bandX = lambda f: np.clip(np.searchsorted(eX, f, side="right") - 1, 0, len(eX) - 2)
for b in range(len(eX) - 1):
    FE[f"shareX_B{b}"] = [ (bandX(r[:, 0]) == b).mean() if len(r) else np.nan for r in RING]
    FE[f"fitX_B{b}"] = [ r[bandX(r[:, 0]) == b, 2].max() if (len(r) and (bandX(r[:, 0]) == b).any()) else 0.0 for r in RING]
    FE[f"psdpromX_B{b}"] = [float((LP[i, (FW >= eX[b]) & (FW < eX[b + 1])] - MED[(FW >= eX[b]) & (FW < eX[b + 1])]).mean()) for i in range(N)]
tX = pd.crosstab(bandX(F), grp).reindex(columns=["normal", "fault", "val"], fill_value=0); tX.index = [f"{eX[i]/1000:.2f}–{eX[i+1]/1000:.2f}k" for i in tX.index]
P("탐색 대역별 충격 수(라벨은 보고용):\n" + tX.to_string())
for lo_, hi_ in ((8000, 8500), (9250, 9750)):
    k = (F >= lo_) & (F < hi_); P(f"f in [{lo_},{hi_}) 충격 {k.sum()}개, 파일 {len(set(FID[k]))}개; 라벨별 파일 수(보고용):", {g: len(set(FID[k & (grp == g)])) for g in ("normal", "fault", "val")}, "| 파일:", sorted({NA[i] for i in FID[k]}))

# %% 충격 정렬 평균 (새 충격 정의)
PRE, POST = 50, 500; NSUR = 5
SOS200 = butter(4, 200, btype="highpass", fs=FS, output="sos")
def ring(m):
    seg = m[PRE:PRE + 375] * np.hanning(2 * 375)[375:]
    Fm = np.abs(np.fft.rfft(seg, 8192)); fr = np.fft.rfftfreq(8192, 1 / FS)
    mk = (fr > 800) & (fr < 12000); f0 = fr[mk][np.argmax(Fm[mk])]
    a = np.abs(hilbert(m))[PRE + 25:PRE + 200]; tt = np.arange(len(a)) / FS
    sl = np.polyfit(tt, np.log(a + 1e-12), 1)[0]
    return f0, (-sl / (2 * np.pi * f0) if sl < 0 else np.nan)
def align(x200, idx):
    out = []
    for p in idx:
        q = p - 25 + int(np.argmax(np.abs(x200[max(p - 25, 0):p + 26])))
        if q - PRE >= 0 and q + POST <= len(x200): out.append(x200[q - PRE:q + POST] * np.sign(x200[q]))
    return np.array(out).reshape(-1, PRE + POST)
def p2p(m): return m.max() - m.min()
def work2(i):
    rng = np.random.default_rng(i); x = XA[i]; x200 = sosfiltfilt(SOS200, x - x.mean())
    S = align(x200, IDX[i]); n = len(S)
    if n == 0: return dict(n_aligned=0)
    m = S.mean(0); f0, z = ring(m); o = dict(n_aligned=n, f_tsa=f0, zeta_tsa=z, p2p=p2p(m))
    if n >= 2:
        for s, tag in ((0, "odd"), (1, "even")):
            o[f"f_{tag}"], o[f"z_{tag}"] = ring(S[s::2].mean(0))
    sp = []
    from src.impacts import _sos, _baseline
    for _ in range(NSUR):
        Fx = np.fft.rfft(x200); ph = rng.uniform(0, 2 * np.pi, len(Fx)); ph[0] = ph[-1] = 0
        xs = sosfiltfilt(SOS200, np.fft.irfft(np.abs(Fx) * np.exp(1j * ph), n=len(x200)))
        # 대용 신호는 임계값(4.5배)을 못 넘기 때문에, 같은 개수 n 만큼 포락선/바탕 비 상위 봉우리를 강제로 뽑는다
        e = sosfiltfilt(_sos("lowpass", 1000, FS), np.abs(hilbert(highpass(xs)))); r_ = e / _baseline(e)
        pk, _h = find_peaks(r_, distance=169); pk = pk[(pk >= 250) & (pk < len(xs) - 250)]
        pk = np.sort(pk[np.argsort(r_[pk])[::-1][:n]])
        Ss = align(xs, pk); sp.append(p2p(Ss.mean(0)) if len(Ss) else np.nan)
    sm_ = np.nanmean(sp) if not np.all(np.isnan(sp)) else np.nan
    o["R"] = o["p2p"] / sm_ if np.isfinite(sm_) else np.nan
    return o
T2 = pd.DataFrame(Parallel(n_jobs=-1)(delayed(work2)(i) for i in range(N)))
for c in ("n_aligned", "f_tsa", "zeta_tsa", "R", "f_odd", "f_even", "z_odd", "z_even"):
    if c not in T2: T2[c] = np.nan
FE = pd.concat([FE, T2[["n_aligned", "f_tsa", "zeta_tsa", "R", "f_odd", "f_even", "z_odd", "z_even"]]], axis=1)
FE.round(5).to_csv(os.path.join(HERE, "features_modemap.csv"), index=False, encoding="utf-8-sig")
P("충격 수 0,1,2,3개 파일 수:", [int((nimp == k).sum()) for k in range(4)], "4개 이상:", int((nimp >= 4).sum()))
lab_ = {0: "정상", 1: "고장", -1: "val"}
for lab in (0, 1, -1):
    s = FE[FE.label == lab]; P(f"  {lab_[lab]}: 파일 {len(s)}, 충격 1개 {int((s.n_imp==1).sum())}, 0개 {int((s.n_imp==0).sum())}, 4개 이상 {int((s.n_imp>=4).sum())}")
s = FE[(FE.n_aligned >= 4)].dropna(subset=["f_odd", "f_even", "z_odd", "z_even"])
rf = spearmanr(s.f_odd, s.f_even); rz = spearmanr(s.z_odd, s.z_even)
P(f"분할 신뢰도 (충격 >=4, 파일 {len(s)}): f Spearman {rf.statistic:.3f} (p={rf.pvalue:.1e}) · zeta Spearman {rz.statistic:.3f} (p={rz.pvalue:.1e})")
for lo_ in (6, 8):
    t = FE[(FE.n_aligned >= lo_)].dropna(subset=["f_odd", "f_even", "z_odd", "z_even"])
    if len(t) > 5: P(f"  충격 >={lo_}: 파일 {len(t)}, f {spearmanr(t.f_odd, t.f_even).statistic:.3f}, zeta {spearmanr(t.z_odd, t.z_even).statistic:.3f}")
med_f = np.array([np.median(r[:, 0]) if len(r) else np.nan for r in RING]); ok = (FE.f_tsa.notna() & ~np.isnan(med_f)).values
P("TSA f vs 개별 충격 f 중앙값 Spearman:", round(spearmanr(FE.f_tsa[ok], med_f[ok]).statistic, 3))
P("R 중앙값 (정상/고장/val):", [round(float(FE.R[FE.label == l].median()), 2) for l in (0, 1, -1)])

col = {0: "gray", 1: "tab:red", -1: "tab:purple"}
fig, ax = plt.subplots(1, 3, figsize=(19, 5.2))
for l in (0, 1, -1):
    s = FE[FE.label == l]; one = s.n_aligned <= 1
    ax[0].scatter(s.f_tsa[~one] / 1000, s.zeta_tsa[~one], c=col[l], s=22, alpha=.7, label=f"{lab_[l]} (충격 2개 이상)")
    ax[0].scatter(s.f_tsa[one] / 1000, s.zeta_tsa[one], c=col[l], s=40, marker="x", label=f"{lab_[l]} (충격 1개)")
for v in VS: ax[0].axvline(v / 1000, color="k", lw=.6)
ax[0].set_yscale("log"); ax[0].set_xlabel("정렬 평균 울림 f (kHz)"); ax[0].set_ylabel("zeta"); ax[0].legend(fontsize=7); ax[0].set_title("f x zeta (세로선 = 라벨 없는 모드 경계)")
q = FE[FE.n_aligned >= 4]
ax[1].scatter(q.f_odd / 1000, q.f_even / 1000, c=[col[l] for l in q.label], s=20); ax[1].plot([1, 12], [1, 12], "k", lw=.5)
ax[1].set_xlabel("홀수 충격 평균 f (kHz)"); ax[1].set_ylabel("짝수 충격 평균 f (kHz)"); ax[1].set_title(f"분할 신뢰도 f rho={rf.statistic:.2f}")
ax[2].scatter(q.z_odd, q.z_even, c=[col[l] for l in q.label], s=20); ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].plot([.003, .3], [.003, .3], "k", lw=.5)
ax[2].set_xlabel("홀수 zeta"); ax[2].set_ylabel("짝수 zeta"); ax[2].set_title(f"분할 신뢰도 zeta rho={rz.statistic:.2f}")
fig.savefig(os.path.join(FIG, "aligned_scatter.png"), dpi=120, bbox_inches="tight"); plt.close(fig)

# %% 정상 5–95% 범위와 val 이동
nor = FE[FE.label == 0]; va = FE[FE.label == -1]; fa = FE[FE.label == 1]
cols = [c for c in FE.columns if c.startswith(("share", "fit", "psdprom")) or c in ("n_imp", "f_tsa", "zeta_tsa", "R")]
rr = []
for c in cols:
    lo, hi = nor[c].quantile([.05, .95]); rr.append(dict(feature=c, nor_p5=lo, nor_med=nor[c].median(), nor_p95=hi, val_med=va[c].median(),
        val_out=float(((va[c] < lo) | (va[c] > hi)).mean()), fault_out=float(((fa[c] < lo) | (fa[c] > hi)).mean()),
        shift_sd=(va[c].median() - nor[c].median()) / (nor[c].std() + 1e-12)))
RR = pd.DataFrame(rr); RR.round(4).to_csv(os.path.join(HERE, "normal_range_val_shift.csv"), index=False, encoding="utf-8-sig")
P("정상 5–95% 범위 · val 이동 (val_out = 정상 범위 밖 val 비율, fault_out = 고장 중 밖 비율[보고용])\n" + RR.round(3).to_string(index=False))
nb = len(BANDS)
fig, ax = plt.subplots(2, nb, figsize=(4 * nb, 7))
for b in range(nb):
    for l, c in ((0, "gray"), (1, "tab:red"), (-1, "tab:purple")):
        ax[0][b].hist(FE[FE.label == l][f"share_B{b}"].dropna(), bins=np.linspace(0, 1, 11), alpha=.5, color=c, density=True, label=lab_[l])
    ax[0][b].set_title(f"대역{b} {BANDS[b][0]/1000:.2f}–{BANDS[b][1]/1000:.2f}k 울림 비율", fontsize=9); ax[0][b].legend(fontsize=7)
    ax[1][b].boxplot([FE[FE.label == l][f"psdprom_B{b}"] for l in (0, 1, -1)], tick_labels=["정상", "고장", "val"]); ax[1][b].set_title(f"대역{b} PSD 돌출 (풀 중앙값 대비)", fontsize=9)
fig.savefig(os.path.join(FIG, "band_features.png"), dpi=110, bbox_inches="tight"); plt.close(fig)
P("완료")
