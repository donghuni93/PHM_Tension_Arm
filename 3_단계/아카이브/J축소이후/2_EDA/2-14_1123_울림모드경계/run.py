# %% [markdown]
# 2-14 (11-23) 울림 모드로 대역 경계 정하기 + 소린 비율의 의미 (EDA, 모델 실험 없음)
# 1. 경계: 모든 파일(학습 150 + val 40)의 충격 울림 주파수를 라벨 없이 모아, 파일마다 같은 무게로 분포를 그리고 골짜기를 찾는다
# 2. 확인: 그 경계로 나눈 대역에서 정상 · 고장이 갈리나 (라벨은 확인에만), 2 kHz 아래 공유 대역
# 3. 소린 비율: 충격 직후 12 ms의 대역 에너지 비 (큰 충격 20개 중앙값, log10). 새 경계로도 계산
# 실행: conda run -n qe-phm python run.py (첫 실행은 충격 계산 몇 분, 이후 figs/impacts.csv 캐시)

# %% 설정
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__) if "__file__" in dir() else ".")
while not os.path.isdir(os.path.join(_d, "src")) and os.path.dirname(_d) != _d:
    _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.metrics import roc_auc_score
from src.exp2util import NS
from src.exp import OUTLIER_NORMALS
from src.impacts import all_impacts, highpass
from src.final import _SEG
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575", "#d9dde3"
FS = 25000; W12 = int(.012 * FS); EB = np.arange(1000, 12501, 250)[:-1]   # 12 ms 스펙트럼을 250 Hz 칸으로
ns = NS(); y = ns.y; names = list(ns.names); nvn = [f"val_{v[-3:]}" for v in ns.nv]
short = lambda s: s.replace("nor_", "N").replace("abn_", "A").replace("val_", "V")
G = {"정상": NOR, "고장": ABN, "val": VAL}

# %% 충격별 값 (캐시): 울림 주파수 f50(감쇠 사인파 사전 50 Hz 간격, 충격 뒤 8 ms) · 직후 12 ms 250 Hz 칸 에너지
CSV, NPY = os.path.join(FIG, "impacts.csv"), os.path.join(FIG, "impact_spec250.npy")
if not os.path.exists(CSV):
    def atoms(step):
        t = np.arange(_SEG) / FS; A, P = [], []
        for f in np.arange(1000, 12001, step):
            for z in (0.01, 0.02, 0.04, 0.08):
                w = 2 * np.pi * f; wd, a = w * np.sqrt(1 - z ** 2), np.exp(-z * w * t)
                Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1)); A.append(Q); P.append(f)
        return np.stack(A), np.array(P)
    A50, P50 = atoms(50); fr = np.fft.rfftfreq(W12, 1 / FS); rows, SP = [], []
    def go(ims, grps, fn):
        for im, g, nm in zip(ims, grps, fn):
            xh = im["hp"]
            for k, p in enumerate(im["idx"]):
                if p + W12 > len(xh): continue
                s = xh[p:p + _SEG]; s = s / (np.linalg.norm(s) + 1e-12)
                fit = np.linalg.norm(np.einsum("s,ast->at", s, A50), axis=1)
                P = np.abs(np.fft.rfft(xh[p:p + W12] * np.hanning(W12))) ** 2
                SP.append([P[(fr >= a) & (fr < a + 250)].sum() for a in EB])
                rows.append(dict(file=nm, grp=g, hg=float(im["height_g"][k]), t=float(im["t"][k]), f250=float(im["rings"][k, 0]), f50=float(P50[fit.argmax()])))
    go(all_impacts(ns.X), ["정상" if v == 0 else "고장" for v in y], names); go(all_impacts(ns.Xv), ["val"] * len(nvn), nvn)
    pd.DataFrame(rows).to_csv(CSV, index=False, encoding="utf-8-sig"); np.save(NPY, np.array(SP))
D = pd.read_csv(CSV); SP = np.load(NPY)
F = pd.DataFrame(dict(file=names + nvn, grp=["정상" if v == 0 else "고장" for v in y] + ["val"] * len(nvn)))
NF = F.grp.value_counts(); D["w"] = 1 / D.groupby("file").f50.transform("size")   # 파일마다 같은 무게
TR = F.grp != "val"; YY = (F.grp[TR] == "고장").astype(int).values
sep = lambda v: abs(roc_auc_score(YY, np.asarray(v)[TR.values]) - .5) * 2      # 분리도 = |AUC − 0.5| × 2

# %% 1. 라벨 없는 골짜기 (190개 파일 전체, 50 Hz 칸, 가우스 평활 σ = 100 Hz)
B50 = np.arange(1000, 12301, 50); C50 = B50[:-1] + 25
hp_, _ = np.histogram(D.f50, bins=B50, weights=D.w); hs = gaussian_filter1d(hp_, 2)
vi, _ = find_peaks(-hs, prominence=.3); pi, _ = find_peaks(hs, prominence=.3)
say("봉우리(Hz):", [int(C50[i]) for i in pi]); say("골짜기(Hz):", [int(C50[i]) for i in vi])
# 1.9k 모드의 위쪽 끝 = 봉우리(1.9k)보다 높은 쪽에서 밀도가 봉우리의 25 % 아래로 처음 내려가는 곳 (라벨 없이). 에너지 비의 아래 경계
p19 = [i for i in pi if 1700 < C50[i] < 2200][0]; E19 = int(C50[p19 + np.argmax(hs[p19:] < .25 * hs[p19])] - 25)
say("1.9k 모드 위쪽 끝(Hz):", E19)
# 골짜기 → 대역. 6.3 kHz 골은 본체 모드의 두 봉우리(6.0 · 7.2k, 둘 다 정상) 사이라 합친다. 8 kHz 위는 충격이 드물어 한 대역
NEW = [("공유 저주파", 1000, 1600), ("연결부 모드", 1600, 4500), ("전이", 4500, 5500), ("본체 모드", 5500, 8000), ("고주파", 8000, 12500)]
OCT = [("H7 1–2k", 1000, 2000), ("H8 2–4k", 2000, 4000), ("H9 4–8k", 4000, 8000), ("H10 8–12.5k", 8000, 12500)]
TEAM = [("효섭 2–5k", 2000, 5000), ("효섭 5–8k", 5000, 8000), ("효섭 ≥8k", 8000, 12500)]

# %% 파일별 울림 비율 · 개수
def frac(lo, hi):
    m = (D.f50 >= lo) & (D.f50 < hi); return F.file.map(D.assign(m=m).groupby("file").m.mean()).fillna(-1).values
def cnt(lo, hi):
    m = (D.f50 >= lo) & (D.f50 < hi); return F.file.map(D.assign(m=m).groupby("file").m.sum()).fillna(0).values
SEP = []
for setn, bs in (("지금 옥타브 (H7–H10)", OCT), ("팀원 (효섭)", TEAM), ("새 경계 (골짜기)", NEW)):
    for nm, lo, hi in bs:
        SEP.append(dict(묶음=setn, 대역=nm, lo=lo, hi=hi, 비율=sep(frac(lo, hi)), 개수=sep(cnt(lo, hi))))
SEP = pd.DataFrame(SEP); say("\n대역별 분리도 (|AUC−0.5|×2, 학습 150, 충격 없음 = −1 · 0개)"); say(SEP.round(3).to_string(index=False))

# %% 3. 소린 비율: 큰 충격 20개, 충격마다 sqrt(E_a / E_b) → 중앙값 → log10. 충격 없음 = 정상 중앙값
def eband(lo, hi): return SP[:, (EB >= lo) & (EB + 250 <= hi)].sum(1)
def ratio(a, b):
    d = D.assign(r=np.sqrt(eband(*a) / eband(*b)))
    s = d.sort_values("hg", ascending=False).groupby("file").head(20).groupby("file").r.median()
    v = np.log10(F.file.map(s)); return v.fillna(v[F.grp == "정상"].median()).values
R = {"소린 2–5k ÷ 5–8k": ratio((2000, 5000), (5000, 8000)), "소린 8–11k ÷ 5–8k": ratio((8000, 11000), (5000, 8000)),
     "새 연결부(1.6k부터) ÷ 본체": ratio((1600, 4500), (5500, 8000)), "새 연결부(1.9k 모드 뒤부터) ÷ 본체": ratio((E19, 4500), (5500, 8000)), "새 고주파 ÷ 본체": ratio((8000, 12500), (5500, 8000)),
     "참고: 1.0–1.6k ÷ 본체": ratio((1000, 1600), (5500, 8000))}
TM = pd.read_csv(os.path.join(_d, "3_단계/4_피처/아카이브/4-3_1116_팀원피처·중복지도/figs/teammate_features.csv"), index_col=0)
say("\n소린 비율 재현 (팀원 값과 Spearman ρ): 2–5/5–8 %.2f · 8–11/5–8 %.2f" % (
    pd.Series(R["소린 2–5k ÷ 5–8k"]).corr(F.file.map(TM.so_ring25_over58), method="spearman"),
    pd.Series(R["소린 8–11k ÷ 5–8k"]).corr(F.file.map(TM.so_ring811_over58), method="spearman")))
for k, v in R.items(): say(f"분리도 {k}: {sep(v):.3f}")
OUTL = F.file.isin(OUTLIER_NORMALS).values
for k in ("새 연결부(1.9k 모드 뒤부터) ÷ 본체", "새 고주파 ÷ 본체"):
    v = R[k]; nv = v[(F.grp == "정상").values]
    say(f"{k}: 이상치 정상 백분위 " + " · ".join(f"{short(f)} {(nv < v[F.file == f][0]).mean() * 100:.0f}" for f in OUTLIER_NORMALS))
F["J"], F["H"] = R["새 연결부(1.9k 모드 뒤부터) ÷ 본체"], R["새 고주파 ÷ 본체"]; F["nJ"] = cnt(1600, 4500); F["n"] = cnt(1000, 12500)
F.round(4).to_csv(os.path.join(FIG, "파일표.csv"), index=False, encoding="utf-8-sig")


# %% 그림 1: 울림 모드 지도 — 위 = 라벨 없이 모든 파일, 아래 = 정상 · 고장 (파일당 평균 충격 수)
def per_file(g):   # 파일 안 충격 비율(파일마다 같은 무게)의 그룹 평균, 충격 있는 파일만
    d = D[D.grp == g]; h, _ = np.histogram(d.f50, bins=B50, weights=d.w); return gaussian_filter1d(h / d.file.nunique(), 2)
BAND_C = ["#c9ced6", ABN, "#e7d9a8", NOR, "#c98a5a"]
fig, axs = plt.subplots(2, 1, figsize=(14, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1.3]))
for ax in axs:
    for (nm, lo, hi), c in zip(NEW, BAND_C): ax.axvspan(lo, hi, color=c, alpha=.13, lw=0)
    for e in (2000, 4000): ax.axvline(e, color=MUTED, ls=":", lw=1)
ax = axs[0]; ax.fill_between(C50, hs, color=INK, alpha=.25); ax.plot(C50, hs, color=INK, lw=1.4)
for i in vi: ax.axvline(C50[i], color=INK, lw=1.2); ax.text(C50[i], hs.max() * 1.02, f"{C50[i] / 1000:.1f}k", ha="center", fontsize=8.5)
for nm, lo, hi in NEW: ax.text((lo + hi) / 2, hs.max() * .82, nm, ha="center", fontsize=10, weight="bold", color=INK)
ax.set_title("① 라벨 없이: 190개 파일(학습 + val)의 울림 주파수, 파일마다 같은 무게 — 실선 = 골짜기, 점선 = 지금 옥타브 경계(2k · 4k)", loc="left", fontsize=10.5)
ax.set_ylim(0, hs.max() * 1.12); ax.set_yticks([])
ax = axs[1]
for g in ("정상", "고장"): ax.plot(C50, per_file(g), color=G[g], lw=2, label=f"{g} (파일 {NF[g]})")
ax.set_title("② 라벨로 확인: 파일 안 충격 비율의 평균(충격 수 차이를 지움) — 2–4.5k · 8.3k = 고장, 5.5–8k = 정상, 2k 아래 = 비슷", loc="left", fontsize=10.5)
ax.legend(fontsize=9); ax.grid(alpha=.2); ax.set_xlabel("충격 울림 주파수 (Hz)", color=MUTED); ax.set_ylabel("파일 안 충격 비율 (50 Hz 칸)", color=MUTED)
ax.set_xlim(1000, 12000); fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_울림모드지도.png"), dpi=150); plt.close(fig)

# %% 그림 2: 2 kHz 아래 확대 — 파일 안 충격 비율로 보면 2.0k 아래는 정상 · 고장이 비슷, 1.9k 모드 뒤(E19)부터 고장만
fig, axs = plt.subplots(1, 2, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
ax = axs[0]; bz = np.arange(1000, 2601, 100)
for g in ("정상", "고장"):
    d = D[D.grp == g]; h, _ = np.histogram(d.f50, bins=bz, weights=d.w)
    ax.step(bz, np.r_[h, h[-1]] / d.file.nunique() * 100, where="post", color=G[g], lw=2, label=g)
ax.axvline(1600, color=INK, lw=1.2); ax.axvline(E19, color=ABN, lw=1.2); ax.axvline(2000, color=MUTED, ls=":", lw=1)
yt = ax.get_ylim()[1]
ax.text(1450, yt * .92, "비슷", ha="center", fontsize=11, weight="bold"); ax.text(1800, yt * .92, "1.9k 모드\n(둘 다 울림)", ha="center", fontsize=9.5, weight="bold")
ax.text(2420, yt * .92, "고장만", ha="center", fontsize=11, weight="bold", color=ABN)
ax.set_title(f"파일 안 충격 비율 (100 Hz 칸) — 검정 1.6k = 골짜기, 주황 {E19 / 1000:.2f}k = 1.9k 모드 끝, 점선 2k = 지금 경계", loc="left", fontsize=10)
ax.legend(fontsize=9, loc="center left"); ax.grid(alpha=.2); ax.set_xlabel("울림 주파수 (Hz)", color=MUTED); ax.set_ylabel("%", color=MUTED)
ax = axs[1]; xs = [("1.0–1.6k", 1000, 1600), (f"1.6–{E19 / 1000:.2f}k", 1600, E19), (f"{E19 / 1000:.2f}–4.5k", E19, 4500)]
for j, (lab, lo, hi) in enumerate(xs):
    v = frac(lo, hi)
    for k, g in enumerate(("정상", "고장")):
        m = (F.grp == g).values & (v >= 0); ax.bar(j * 3 + k, v[m].mean() * 100, color=G[g], width=.9)
        ax.text(j * 3 + k, v[m].mean() * 100 + .6, f"{v[m].mean() * 100:.0f}", ha="center", fontsize=8.5)
    ax.text(j * 3 + .5, -5, f"{lab}\n분리도 {sep(v):.2f}", ha="center", fontsize=9)
ax.set_xticks([]); ax.set_ylabel("파일 안 충격 비율 평균 (%)", color=MUTED); ax.grid(axis="y", alpha=.2)
ax.set_title("세 구간의 비율 — 파랑 정상 · 주황 고장", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_2k아래.png"), dpi=150); plt.close(fig)

# %% 그림 3: 대역별 분리도 — 지금 · 팀원 · 새 경계
fig, ax = plt.subplots(figsize=(14, 4.2)); x = 0; ticks = []
for setn, c in (("지금 옥타브 (H7–H10)", MUTED), ("팀원 (효섭)", VAL), ("새 경계 (골짜기)", INK)):
    s = SEP[SEP.묶음 == setn]
    for r in s.itertuples():
        ax.bar(x - .2, r.비율, .4, color=c); ax.bar(x + .2, r.개수, .4, color=c, alpha=.45)
        ax.text(x - .2, r.비율 + .01, f"{r.비율:.2f}", ha="center", fontsize=8); ticks.append((x, r.대역)); x += 1
    ax.text(x - len(s) / 2 - .5, .82, setn, ha="center", fontsize=10, weight="bold", color=c); x += .8
ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks], fontsize=9)
ax.set_ylim(0, .9); ax.set_ylabel("분리도 |AUC-0.5|×2", color=MUTED); ax.grid(axis="y", alpha=.2)
ax.set_title("대역별 분리도 (학습 150) — 진한 막대 = 울림 비율, 옅은 막대 = 충격 개수", loc="left", fontsize=10.5)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_분리도비교.png"), dpi=150); plt.close(fig)

# %% 그림 4: 소린 비율이 재는 것 — 큰 충격 하나의 12 ms와 스펙트럼 (정상 본체 울림 · 고장 연결부 울림 · 고장 고주파 울림)
EX = []
for g, lo, hi in (("정상", 6500, 7500), ("고장", 1800, 4000), ("고장", 8200, 8500)):
    d = D[(D.grp == g) & (D.f50 >= lo) & (D.f50 < hi) & ~D.file.isin(OUTLIER_NORMALS)].sort_values("hg", ascending=False).iloc[0]; EX.append(d)
fr = np.fft.rfftfreq(W12, 1 / FS); t12 = np.arange(W12) / FS * 1000
fig, axs = plt.subplots(2, 3, figsize=(15, 6.2))
for j, d in enumerate(EX):
    i = names.index(d.file); xh = highpass(ns.X[i]); p = int(round(d.t * FS)); s = xh[p:p + W12]
    axs[0, j].plot(t12, s, color=G[d.grp], lw=1); axs[0, j].set_title(f"{short(d.file)} ({d.grp}) 큰 충격 — 울림 {d.f50 / 1000:.1f} kHz", loc="left", fontsize=10)
    axs[0, j].set_xlabel("충격 뒤 시간 (ms)", color=MUTED, fontsize=8.5); axs[0, j].set_yticks([])
    P = np.abs(np.fft.rfft(s * np.hanning(W12))) ** 2; P = P / P.max(); ax = axs[1, j]
    for (nm, lo, hi), c in zip(NEW, BAND_C): ax.axvspan(lo, hi, color=c, alpha=.2, lw=0)
    ax.plot(fr, P, color=G[d.grp], lw=1.4); ax.set_xlim(1000, 12500); ax.set_yticks([])
    eJ, eB, eH = [P[(fr >= lo) & (fr < hi)].sum() for lo, hi in ((E19, 4500), (5500, 8000), (8000, 12500))]
    ax.set_title(f"연결부 ÷ 본체 = {np.log10(np.sqrt(eJ / eB)):+.2f} · 고주파 ÷ 본체 = {np.log10(np.sqrt(eH / eB)):+.2f}  (log10)", loc="left", fontsize=9.5)
    ax.set_xlabel("주파수 (Hz) — 주황 띠 = 연결부, 파랑 띠 = 본체, 갈색 띠 = 고주파", color=MUTED, fontsize=8.5)
fig.suptitle("그림 4 — 비율 = 충격 직후 12 ms에서 '어느 모드가 더 울렸나'. 0 = 같음, + = 분자 대역이 더 울림", x=.01, ha="left", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, .95)); fig.savefig(os.path.join(FIG, "4_비율개념.png"), dpi=150); plt.close(fig)

# %% 그림 5: 파일 지도 — 가로 = 연결부 ÷ 본체, 세로 = 고주파 ÷ 본체 (새 경계 에너지 비)
NAMED = {"abn_022", "abn_005", "val_025", "nor_025"} | set(OUTLIER_NORMALS)
fig, ax = plt.subplots(figsize=(10, 7.4))
for g, mk in (("정상", "o"), ("고장", "o"), ("val", "^")):
    m = (F.grp == g).values & ~OUTL
    ax.scatter(F.J[m], F.H[m], s=30, marker=mk, color=G[g], alpha=.75 if g != "val" else .55, label=f"{g} {m.sum()}", edgecolor="white", lw=.4)
ax.scatter(F.J[OUTL], F.H[OUTL], s=80, marker="D", color=NOR, edgecolor=INK, lw=1, label="이상치 정상 ◆")
for r in F[F.file.isin(NAMED)].itertuples(): ax.annotate(short(r.file), (r.J, r.H), xytext=(5, 4), textcoords="offset points", fontsize=9)
nv = F[(F.grp == "정상")]
ax.axvline(nv.J.quantile(.95), color=MUTED, ls="--", lw=1); ax.axhline(nv.H.quantile(.95), color=MUTED, ls="--", lw=1)
ax.set_xlabel(f"연결부({E19 / 1000:.2f}–4.5k) ÷ 본체(5.5–8k), log10 — 분리도 {sep(F.J):.2f}", color=INK)
ax.set_ylabel(f"고주파(8–12.5k) ÷ 본체(5.5–8k), log10 — 분리도 {sep(F.H):.2f}", color=INK)
ax.set_title("그림 5 — 큰 충격 20개의 에너지 비(중앙값). 점선 = 정상 95 %. 정상은 왼쪽 아래, 고장은 오른쪽 · 위로", loc="left", fontsize=10.5)
ax.legend(fontsize=9, loc="lower right"); ax.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "5_비율지도.png"), dpi=150); plt.close(fig)
for g in ("정상", "고장", "val"):
    s = F[F.grp == g]; say(f"{g}: 연결부 > 정상95 {(s.J > nv.J.quantile(.95)).sum()} · 고주파 > 정상95 {(s.H > nv.H.quantile(.95)).sum()} · 둘 중 하나 {((s.J > nv.J.quantile(.95)) | (s.H > nv.H.quantile(.95))).sum()} / {len(s)}")
OUT.close()
