# %% [markdown]
# 4-16 (11-24) 라벨 없는 지도 · val 이동 진단 (EDA, 모델 실험 없음)
# 목적: 모델 정합성 설명 + 일반화된 피처 찾기. "물리적으로 절대적인 피처" = ① 학습 라벨로 갈리고 ② val에서 측정 조건 때문에 움직이지 않는 피처
# 1. 라벨 없는 지도: 학습 150 + val 40 = 190개 파일의 충격 뺀 배경 PSD · 충격 울림 주파수를 라벨 없이 모아 봉우리 · 골짜기를 찾는다
# 2. val 이동 진단: 피처마다 val 40개가 학습 정상에서 얼마나 움직였나를, "학습 고장이 p만큼 섞였다면 생길 이동"(시뮬레이션)과 비교
#    - 고장 40 %가 섞여도 설명 안 되는 고장 방향 이동 = 조건 이동 또는 새 고장 (애매)
#    - 고장이 하나도 없을 때보다 더 반대로 간 이동 = 조건 이동 (깨끗한 증거)
#    - 적대 검증: ET로 학습 150 vs val 40을 가려 보고(AUC), 열 하나씩 빼며 어느 열이 가르는지
# 라벨은 학습 150에만 쓰고(방향 · 섞임 시뮬레이션), val은 라벨 없이 쓴다 (판단차이 A13, 11-24 사용자 결정)
# 실행: conda run -n qe-phm python run.py  (충격 시점은 2-14 캐시 impacts.csv 재사용)

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
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import StratifiedKFold
from src.exp2util import NS, RING_COLS
from src.exp import OUTLIER_NORMALS, cached_table, ROOT
from src.impacts import ring_mode_table
from src.features2 import PeakRef, H_FOLD, mode_band_table
from src.features import feature_table_ext
plt.rcParams["font.family"] = "Malgun Gothic" if os.name == "nt" else "WenQuanYi Zen Hei"; plt.rcParams["axes.unicode_minus"] = False
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
OUT = open(os.path.join(FIG, "결과.txt"), "w", encoding="utf-8")
def say(*a): print(*a, file=OUT, flush=True)
NOR, ABN, VAL, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#8a63d2", "#1d2430", "#5b6575", "#d9dde3"
FS = 25000; RNG = np.random.default_rng(0)

ns = NS(); y = ns.y; names = list(ns.names); n = len(y); nor = y == 0
nvn = [f"val_{v[-3:]}" for v in ns.nv]; nV = len(nvn)
out_ = np.isin(names, list(OUTLIER_NORMALS))
RM = ["RM_n_joint", "RM_e_joint", "RM_e_high"]

# %% ET 입력 18열 (J-축소 + RM, 학습 전체 기준으로 한 번: 울림 비율 축소 α=3 · 봉우리 PeakRef는 학습 정상(이상치 3개 제외)으로)
T = pd.concat([ns.T.reset_index(drop=True), cached_table("ringmode_train", ring_mode_table, ns.X), cached_table("modeband_train", mode_band_table, ns.X)], axis=1)
Tv = pd.concat([ns.Tv.reset_index(drop=True), cached_table("ringmode_val", ring_mode_table, ns.Xv), cached_table("modeband_val", mode_band_table, ns.Xv)], axis=1)
FX = list(ns.FIXED); ri = [FX.index(c) for c in RING_COLS]; i1 = FX.index("H1_nimp")
ref = nor & ~out_
H, Hv = T[FX].values.astype(float), Tv[FX].values.astype(float)
cnt, cntv = np.rint(np.expm1(H[:, i1])), np.rint(np.expm1(Hv[:, i1]))
m = ref & (cnt > 0); p0 = (H[m][:, ri] * cnt[m][:, None]).sum(0) / cnt[m].sum()
for A, c in ((H, cnt), (Hv, cntv)):
    h = c > 0; A[np.ix_(h, ri)] = (A[np.ix_(h, ri)] * c[h, None] + 3.0 * p0) / (c[h, None] + 3.0)
pr = PeakRef(ns.df).fit(ns.L[ref]); pk_i = [H_FOLD.index(p) for p in ns.PEAK]
ET18 = FX + list(ns.PEAK) + RM
A18 = pd.DataFrame(np.hstack([H, pr.transform(ns.L)[:, pk_i], T[RM].values]), columns=ET18)
A18v = pd.DataFrame(np.hstack([Hv, pr.transform(ns.Lv)[:, pk_i], Tv[RM].values]), columns=ET18)

# 비교용 풀 (4-15와 같은 ext150 · D+ · 우리 모드)
E = cached_table("ext", feature_table_ext, ns.X); Ev = cached_table("ext_val", feature_table_ext, ns.Xv).reindex(columns=E.columns, fill_value=0.0)
D415 = os.path.join(ROOT, "단계", "4_피처", "4-15_1124_피처풀선별", "figs")
Dp, Dpv = pd.read_csv(os.path.join(D415, "dplus_train.csv")), pd.read_csv(os.path.join(D415, "dplus_val.csv"))
OURS = [c for c in T.columns if c.startswith(("MK_", "MR_", "MS_", "SO_")) or c == "RM_n_joint22"]
POOL = pd.concat([E.add_prefix("E:"), Dp, T[OURS].add_prefix("O:")], axis=1)
POOLv = pd.concat([Ev.add_prefix("E:"), Dpv, Tv[OURS].add_prefix("O:")], axis=1)
POOL = POOL.loc[:, POOL.nunique() > 3]; POOLv = POOLv[POOL.columns]

# %% 섞임 시뮬레이션 기준 이동 진단
P_LIST = (0.0, 0.225, 0.4)          # 0 = 고장 없음, 0.225 = 지금 val 판정 9/40, 0.4 = 넉넉한 위
def auc_gt(a, b):
    """P(a > b) + 0.5 P(a = b)"""
    return roc_auc_score(np.r_[np.zeros(len(b)), np.ones(len(a))], np.r_[b, a])

def shift_test(v, vv, B=1000):
    v = np.asarray(v, float); vv = np.asarray(vv, float)
    med = np.nanmedian(v); v = np.where(np.isfinite(v), v, med); vv = np.where(np.isfinite(vv), vv, med)
    a_f = auc_gt(v[y == 1], v[nor]); d = 1 if a_f >= .5 else -1; w, ww = d * v, d * vv    # w: 고장 쪽이 크게
    wn, wf = w[nor], w[y == 1]
    obs = auc_gt(ww, wn)
    null = {}
    for p in P_LIST:
        s = np.empty(B)
        for b in range(B):
            k = RNG.binomial(nV, p); smp = np.r_[RNG.choice(wf, k), RNG.choice(wn, nV - k)]
            s[b] = auc_gt(smp, RNG.choice(wn, len(wn)))
        null[p] = np.percentile(s, [2.5, 50, 97.5])
    if obs > null[0.4][2]: flag = "고장방향 과잉"
    elif obs < null[0.0][0]: flag = "반대방향(조건)"
    else: flag = "설명 가능"
    whole = (np.percentile(ww, 25) > np.percentile(wn, 75)) or (np.percentile(ww, 75) < np.percentile(wn, 25))
    return dict(분리도=abs(a_f - .5) * 2, 방향="고장↑" if d > 0 else "고장↓", val_AUC=obs,
                기대_p0=null[0.0][1], 기대_p225=null[0.225][1], 범위_p0=f"{null[0.0][0]:.2f}–{null[0.0][2]:.2f}",
                범위_p40=f"{null[0.4][0]:.2f}–{null[0.4][2]:.2f}", 초과=obs - null[0.225][1], 판정=flag, 통째이동=whole)

# %% 1a. 배경 PSD (충격 −5 ~ +30 ms를 덮는 조각을 뺀 Welch, 2048점, 원신호 평균 제거)
IMP = pd.read_csv(os.path.join(ROOT, "단계", "2_EDA", "2-14_1123_울림모드경계", "figs", "impacts.csv"))
tmap = IMP.groupby("file").t.apply(lambda s: (s.values * FS).astype(int)).to_dict()
NP_, HOP = 2048, 1024; win = np.hanning(NP_); fr = np.fft.rfftfreq(NP_, 1 / FS)
def bg_psd(x, idx):
    x = np.asarray(x, float) - np.mean(x); st = np.arange(0, len(x) - NP_ + 1, HOP)
    ok = np.ones(len(st), bool)
    for p in idx:
        ok &= ~((st < p + int(.005 * FS)) & (st + NP_ > p - int(.030 * FS)))  # 조각 [s, s+N)이 [p−30ms, p+5ms]와 겹치면 뺀다 (충격 뒤 30 ms 울림)
    ok &= True
    use = st[ok] if ok.sum() >= 4 else st[np.argsort([np.abs(x[s:s + NP_]).max() for s in st])[:4]]   # 남는 조각이 4개 미만이면 가장 조용한 4개
    P = np.mean([np.abs(np.fft.rfft(x[s:s + NP_] * win)) ** 2 for s in use], 0)
    return 10 * np.log10(P + 1e-30), ok.sum()
BGC = os.path.join(FIG, "bg_psd.npz")
if not os.path.exists(BGC):
    rows = [bg_psd(x, tmap.get(nm, [])) for x, nm in zip(list(ns.X) + list(ns.Xv), names + nvn)]
    np.savez(BGC, P=np.array([r[0] for r in rows]), k=np.array([r[1] for r in rows]))
Z = np.load(BGC); P, kseg = Z["P"], Z["k"]; Pt, Pv = P[:n], P[n:]
say(f"배경 조각 수(48개 중): 학습 정상 중앙 {np.median(kseg[:n][nor]):.0f} · 고장 {np.median(kseg[:n][y == 1]):.0f} · val {np.median(kseg[n:]):.0f}, 4개 미만 파일 {(kseg < 4).sum()}")

band = (fr >= 50) & (fr <= 12400)
pooled = gaussian_filter1d(P[:, band].mean(0), 8)                 # 190개 같은 무게, σ ≈ 100 Hz
fb = fr[band]
pi_, _ = find_peaks(pooled, prominence=1.0); vi_, _ = find_peaks(-pooled, prominence=1.0)
say("\n[1a 배경 PSD, 라벨 없음] 봉우리(Hz):", [int(fb[i]) for i in pi_]); say("골짜기(Hz):", [int(fb[i]) for i in vi_])

# 대역 = 라벨 없는 울림 모드 골짜기(2-14) + 1 kHz 아래 둘
BANDS = [("저주파 0.05–0.2k", 50, 200), ("중저주파 0.2–1k", 200, 1000), ("공유 1.0–1.6k", 1000, 1600), ("연결부 1.6–4.5k", 1600, 4500),
         ("전이 4.5–5.5k", 4500, 5500), ("본체 5.5–8k", 5500, 8000), ("고주파 8–12.5k", 8000, 12500)]
def band_db(Pm, lo, hi):
    s = (fr >= lo) & (fr < hi); return 10 * np.log10(np.mean(10 ** (Pm[:, s] / 10), 1))
say("\n[1b 배경 대역 크기 dB: 학습 정상 대비 val 이동, 고장 섞임으로 설명되나]")
BG_rows = []
for nm_, lo, hi in BANDS:
    v, vv = band_db(Pt, lo, hi), band_db(Pv, lo, hi)
    r = shift_test(v, vv); r.update(대역=nm_, 고장_dB=np.median(v[y == 1]) - np.median(v[nor]), val_dB=np.median(vv) - np.median(v[nor]))
    BG_rows.append(r)
BG = pd.DataFrame(BG_rows)[["대역", "고장_dB", "val_dB", "방향", "분리도", "val_AUC", "범위_p0", "범위_p40", "판정", "통째이동"]]
say(BG.round(2).to_string(index=False))
# 게인 무관(모양) 버전: 대역 dB − 파일 전체(1–12.5k) dB
say("\n[1c 배경 모양 = 대역 dB − 1–12.5k 전체 dB (게인 지움)]")
BGS = []
for nm_, lo, hi in BANDS:
    v, vv = band_db(Pt, lo, hi) - band_db(Pt, 1000, 12500), band_db(Pv, lo, hi) - band_db(Pv, 1000, 12500)
    r = shift_test(v, vv); r.update(대역=nm_, 고장_dB=np.median(v[y == 1]) - np.median(v[nor]), val_dB=np.median(vv) - np.median(v[nor])); BGS.append(r)
BGS = pd.DataFrame(BGS)[["대역", "고장_dB", "val_dB", "방향", "분리도", "val_AUC", "범위_p0", "범위_p40", "판정", "통째이동"]]
say(BGS.round(2).to_string(index=False))

# %% 그림 1: 라벨 없는 배경 PSD 지도 + 울림 모드 + 집단 차이
IMP["w"] = 1 / IMP.groupby("file").f50.transform("size"); IMP["g"] = np.where(IMP.file.str.startswith("val_"), "val", np.where(IMP.file.str.startswith("abn_"), "고장", "정상"))
B50 = np.arange(1000, 12301, 50); C50 = B50[:-1] + 25
fig, ax = plt.subplots(3, 1, figsize=(13, 11), sharex=True)
ax[0].plot(fb / 1000, pooled, color=INK, lw=1.4)
for i in pi_: ax[0].annotate(f"{fb[i] / 1000:.2f}", (fb[i] / 1000, pooled[i]), xytext=(0, 5), textcoords="offset points", ha="center", fontsize=7.5, color=INK)
for i in vi_: ax[0].axvline(fb[i] / 1000, color=MUTED, ls=":", lw=.8)
ax[0].set_ylabel("배경 PSD (dB, 190개 평균)"); ax[0].set_title("① 라벨 없음: 학습 150 + val 40의 충격 뺀 배경 PSD (점선 = 골짜기)", loc="left", fontsize=10)
hs = gaussian_filter1d(np.histogram(IMP.f50, bins=B50, weights=IMP.w)[0], 2); ax[1].plot(C50 / 1000, hs / hs.sum(), color=INK, lw=1.4, label="190개 전체 (라벨 없음)")
for g, c in (("정상", NOR), ("고장", ABN), ("val", VAL)):
    s = IMP[IMP.g == g]; h = gaussian_filter1d(np.histogram(s.f50, bins=B50, weights=s.w)[0], 2); ax[1].plot(C50 / 1000, h / h.sum(), color=c, lw=1, alpha=.9, label=g)
for b in (1.6, 4.5, 5.5, 8.0): ax[1].axvline(b, color=MUTED, ls="--", lw=.8)
ax[1].set_ylabel("충격 울림 주파수 밀도\n(파일마다 같은 무게)"); ax[1].legend(fontsize=8, frameon=False); ax[1].set_title("② 라벨 없음: 충격 울림 모드 (파선 = 2-14 골짜기 1.6 · 4.5 · 5.5 · 8.0k)", loc="left", fontsize=10)
mN = np.median(Pt[nor][:, band], 0)
ax[2].axhline(0, color=GRID, lw=1)
ax[2].plot(fb / 1000, gaussian_filter1d(np.median(Pt[y == 1][:, band], 0) - mN, 4), color=ABN, lw=1.2, label="학습 고장 - 학습 정상 (중앙값)")
ax[2].plot(fb / 1000, gaussian_filter1d(np.median(Pv[:, band], 0) - mN, 4), color=VAL, lw=1.6, label="val - 학습 정상 (중앙값)")
for b in (1.6, 4.5, 5.5, 8.0): ax[2].axvline(b, color=MUTED, ls="--", lw=.8)
ax[2].set_ylabel("dB 차이"); ax[2].set_xlabel("주파수 (kHz)"); ax[2].legend(fontsize=8, frameon=False)
ax[2].set_title("③ 라벨은 색칠만: 배경이 고장 때문에 움직이나(주황), val에서 움직이나(보라)", loc="left", fontsize=10)
for a in ax: a.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_라벨없는지도.png"), dpi=130); plt.close(fig)

# %% 2. 18열 이동 진단
say("\n[2 ET 입력 18열: val 이동 vs 고장 섞임 시뮬레이션]  val_AUC = P(val > 학습 정상), 고장 쪽이 크도록 뒤집은 값")
R18 = pd.DataFrame([dict(열=c, **shift_test(A18[c].values, A18v[c].values)) for c in ET18])
say(R18.round(3).to_string(index=False))
R18.to_csv(os.path.join(FIG, "18열_이동.csv"), index=False, encoding="utf-8-sig")

# 적대 검증: 학습 150 vs val 40 (ET 300, 5-fold, seed 3)
def adv_auc(X, Xv, seeds=(0, 1, 2)):
    XX = np.vstack([X, Xv]); g = np.r_[np.zeros(len(X)), np.ones(len(Xv))]; out = []
    for s in seeds:
        p = np.zeros(len(g))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=s).split(XX, g):
            p[te] = ExtraTreesClassifier(300, min_samples_leaf=2, max_features="sqrt", random_state=s, n_jobs=-1).fit(XX[tr], g[tr]).predict_proba(XX[te])[:, 1]
        out.append(roc_auc_score(g, p))
    return float(np.mean(out)), p
a_all, p_adv = adv_auc(A18.values, A18v.values)
say(f"\n[적대 검증] 18열로 학습 vs val 가르기 AUC {a_all:.3f} (0.5 = 못 가름)")
# 학습 고장을 val 비율만큼 섞인 것과 비교하기 위한 기준: 학습 정상만 vs val
a_nor, _ = adv_auc(A18.values[nor], A18v.values); say(f"  학습 정상만 vs val: {a_nor:.3f}")
drop = []
for c in ET18:
    keep = [k for k in ET18 if k != c]; drop.append((c, adv_auc(A18[keep].values, A18v[keep].values, seeds=(0, 1))[0]))
DR = pd.DataFrame(drop, columns=["뺀 열", "AUC"]).assign(감소=lambda d: a_all - d.AUC).sort_values("감소", ascending=False)
say("  열 하나 뺄 때 적대 AUC (많이 떨어질수록 그 열이 val을 가름):"); say(DR.round(3).to_string(index=False))
say("  val 파일별 '학습과 다르다' 점수 상위 8: " + ", ".join(f"{nvn[i][-3:]}({p_adv[n + i]:.2f})" for i in np.argsort(-p_adv[n:])[:8]))

# 그림 2: 18열 small multiples
fig, axs = plt.subplots(3, 6, figsize=(19, 9.5)); axs = axs.ravel()
for a, c in zip(axs, ET18):
    r = R18.set_index("열").loc[c]; v, vv = A18[c].values, A18v[c].values
    for j, (vals, col, lab) in enumerate(((v[nor], NOR, "정상"), (v[y == 1], ABN, "고장"), (vv, VAL, "val"))):
        a.scatter(j + RNG.uniform(-.18, .18, len(vals)), vals, s=9, color=col, alpha=.55, lw=0)
        a.hlines(np.median(vals), j - .3, j + .3, color=INK, lw=1.5)
    for q in np.percentile(v[nor], [25, 75]): a.axhline(q, color=NOR, lw=.6, ls=":")
    fc = {"설명 가능": INK, "고장방향 과잉": "#b35a00", "반대방향(조건)": "#c62828"}[r["판정"]]
    a.set_title(f"{c}\nval AUC {r['val_AUC']:.2f} · 고장40% 범위 {r['범위_p40']}\n{r['판정']}", fontsize=7.8, color=fc)
    a.set_xticks([0, 1, 2]); a.set_xticklabels(["정상", "고장", "val"], fontsize=8); a.tick_params(axis="y", labelsize=7); a.grid(alpha=.2)
fig.suptitle("ET 입력 18열: 학습 정상 · 학습 고장 · val (가로선 = 중앙값, 점선 = 학습 정상 25–75 %). 제목 색: 검정 = 고장 섞임으로 설명됨, 주황 = 고장 40 %로도 넘침, 빨강 = 고장 없을 때보다 반대로 감",
             x=.01, ha="left", fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_18열_이동.png"), dpi=125); plt.close(fig)

# %% 3. 풀 전체 (ext150 · D+ · 우리 모드) : 분리도 vs 이동 초과
say("\n[3 풀 전체: 분리도 vs val 이동 초과]")
RP = pd.DataFrame([dict(열=c, **shift_test(POOL[c].values, POOLv[c].values, B=400)) for c in POOL.columns])
RP.to_csv(os.path.join(FIG, "풀_이동.csv"), index=False, encoding="utf-8-sig")
say(RP.판정.value_counts().to_string())
say("분리도 ≥ 0.6이면서 설명 가능(일반화 후보) 상위 15:"); say(RP[(RP.분리도 >= .6) & (RP.판정 == "설명 가능")].sort_values("분리도", ascending=False).head(15)[["열", "분리도", "방향", "val_AUC", "범위_p40", "초과"]].round(3).to_string(index=False))
fig, ax = plt.subplots(figsize=(11, 7))
cm = {"설명 가능": "#9aa3af", "고장방향 과잉": "#e08a2c", "반대방향(조건)": "#c62828"}
for f_, d in RP.groupby("판정"): ax.scatter(d.분리도, d.초과, s=14, color=cm[f_], alpha=.6, lw=0, label=f"풀 · {f_} ({len(d)})")
for _, r in R18.iterrows():
    ax.scatter(r.분리도, r.초과, s=60, color=cm[r.판정], edgecolor=INK, lw=1)
    ax.annotate(r.열.split("_")[0] if not r.열.startswith("RM") else r.열, (r.분리도, r.초과), xytext=(4, 3), textcoords="offset points", fontsize=8)
ax.axhline(0, color=GRID); ax.set_xlabel("학습 분리도 |AUC − 0.5| × 2 (라벨)"); ax.set_ylabel("val 이동 초과 = val AUC − 고장 22.5 % 섞였을 때 기대값\n(+ 고장 쪽으로 더 감, − 반대로 감)")
ax.set_title("오른쪽 · 0 근처 = 잘 가르고 val에서도 안정 (일반화 후보). 테두리 점 = 지금 ET 18열", loc="left", fontsize=10); ax.legend(fontsize=8, frameon=False); ax.grid(alpha=.25)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_분리도_vs_이동.png"), dpi=130); plt.close(fig)
say("\n완료")
