# %% 2-32 피처 점검 ① 충격 정의: H1_nimp (= log1p 충격 수) · H2_ratio21 (= 충격 높이 2위 ÷ 1위). 7칸 카드: 정의 · 분포 · 분리도/이동도 · 경계 파일 · 가설 · 중복 · 판정 후보
import os, sys, warnings
_d = os.path.abspath(os.path.dirname(__file__))
while not os.path.isdir(os.path.join(_d, "src")): _d = os.path.dirname(_d)
sys.path.insert(0, _d); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.signal import find_peaks, hilbert, sosfiltfilt
from sklearn.metrics import roc_auc_score
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from src.exp2util import NS
from src.impacts import highpass, _sos, _baseline, FS, SMOOTH_HZ, K_MIN, DIST_MS, EDGE_MS
HERE = os.path.dirname(os.path.abspath(__file__)); FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
_av = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = next((n for n in ["Malgun Gothic", "NanumGothic", "WenQuanYi Zen Hei"] if n in _av), "DejaVu Sans"); plt.rcParams["axes.unicode_minus"] = False
ns = NS(); y = ns.y; names = list(ns.names); nv = [n[-3:] for n in ns.nv]; nor = y == 0
T, Tv = ns.T.copy(), ns.Tv.copy(); T.index = names
allnames = names + ["val_" + n for n in nv]; X = np.vstack([ns.X, ns.Xv]); grp = np.array(["정상"] * 0 + ["정상" if v == 0 else "고장" for v in y] + ["val"] * len(nv))

# %% 파일마다 ratio = 포락선 ÷ 바탕 (src.impacts.detect_impacts와 같은 계산). 문턱 K를 바꿔 다시 셀 수 있게 한 번만 계산
RAT = []
for x in X:
    e = sosfiltfilt(_sos("lowpass", SMOOTH_HZ, FS), np.abs(hilbert(highpass(x)))); RAT.append(e / _baseline(e))
def peaks(r, K=K_MIN):
    pk = find_peaks(r, height=K, distance=max(1, int(round(DIST_MS * FS / 1000))))[0]; edge = int(round(EDGE_MS * FS / 1000)); return pk[(pk >= edge) & (pk < len(r) - edge)]
# 정의 검증: 위 계산이 모델에 쓰는 H1 · H2와 같은지
n0 = np.array([len(peaks(r)) for r in RAT]); H1 = np.log1p(n0)
hx = [np.sort(RAT[i][peaks(RAT[i])])[::-1] for i in range(len(RAT))]; H2 = np.array([h[1] / h[0] if len(h) >= 2 else 0.0 for h in hx])
ntr = len(names)
chk1 = np.abs(H1[:ntr] - T["H1_nimp"].values).max(); chk2 = np.abs(H2[:ntr] - T["H2_ratio21"].values).max()
chk1v = np.abs(H1[ntr:] - Tv["H1_nimp"].values).max(); chk2v = np.abs(H2[ntr:] - Tv["H2_ratio21"].values).max()

# %% 그림 1: 정의 — 예시 파일의 ratio 곡선 + 문턱 4.5 + 검출된 충격
ex = [("nor_025", "정상 (오경보)"), ("nor_006", "정상"), ("abn_010", "고장 (큰 타격 반복)"), ("abn_022", "고장 (미탐)"), ("val_004", "val 004 (J: 고장)"), ("val_025", "val 025 (경계)")]
ex = [(n, t) for n, t in ex if n in allnames]; q0 = [i for i in range(ntr) if y[i] == 0 and n0[i] == 0][:1]
if q0: ex.append((allnames[q0[0]], "정상 (충격 0개)"))
fig, axs = plt.subplots(len(ex), 1, figsize=(14, 2.1 * len(ex)), sharex=True)
for ax, (n, t) in zip(axs, ex):
    i = allnames.index(n); r = RAT[i]; pk = peaks(r); tt = np.arange(len(r)) / FS
    ax.plot(tt, r, lw=.6, color="#3b7dd8"); ax.axhline(K_MIN, color="k", ls="--", lw=.8); ax.plot(tt[pk], r[pk], "v", color="#e8743b", ms=6)
    ax.set_ylim(0, max(12, r.max() * 1.05)); ax.set_ylabel("포락선÷바탕", fontsize=8)
    ax.set_title(f"{n} — {t} · 충격 {len(pk)}개 · H1 {H1[i]:.2f} · H2 {H2[i]:.2f}", fontsize=9, loc="left")
axs[-1].set_xlabel("초"); fig.suptitle(f"H1·H2 정의: 1 kHz 고역통과 → |Hilbert| → 1 kHz 저역 → 100 ms 이동중앙값 바탕으로 나눈 값 ≥ {K_MIN} (점선)인 봉우리(▼, 간격 ≥ {DIST_MS} ms, 앞뒤 {EDGE_MS} ms 제외). H1 = log(1+개수), H2 = 2위 ÷ 1위 높이", x=.01, ha="left", fontsize=9.5)
fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(os.path.join(FIG, "1_정의_예시.png"), dpi=115); plt.close(fig)

# %% 그림 2: 분포
C = {"정상": "#3b7dd8", "고장": "#e8743b", "val": "#2ca02c"}
fig, axs = plt.subplots(1, 3, figsize=(17, 4.6)); mx = int(n0.max())
for g in C:
    m = grp == g; axs[0].hist(n0[m], bins=np.arange(0, min(mx, 45) + 2) - .5, alpha=.5, density=True, color=C[g], label=f"{g} (n={m.sum()}, 중앙 {np.median(n0[m]):.0f})")
    axs[1].hist(H2[m], bins=np.linspace(0, 1, 21), alpha=.5, density=True, color=C[g], label=g)
    axs[2].scatter(n0[m] + np.random.RandomState(0).uniform(-.25, .25, m.sum()), H2[m], s=22, color=C[g], alpha=.7, label=g, edgecolor="none")
axs[0].set_title("충격 개수 (원값)"); axs[0].set_xlabel("충격 수 n"); axs[0].legend(fontsize=8); axs[1].set_title("H2 = 2위 ÷ 1위 높이 (n<2 → 0)"); axs[1].set_xlabel("H2"); axs[2].set_title("n 대 H2"); axs[2].set_xlabel("충격 수 n"); axs[2].set_ylabel("H2"); axs[2].legend(fontsize=8)
for a in axs: a.grid(alpha=.2)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "2_분포.png"), dpi=115); plt.close(fig)

# %% 표: 분리도 · val 이동도 · 특수값 비율 · 경계 파일 백분위 · 중복 · 문턱 민감도
auc = lambda v, a, b: roc_auc_score(np.r_[np.zeros(a.sum()), np.ones(b.sum())], np.r_[v[a], v[b]])
isN, isA, isV = grp == "정상", grp == "고장", grp == "val"
rows = []
for nm, v in (("H1_nimp", H1), ("H2_ratio21", H2), ("충격 수 n(원값)", n0.astype(float))):
    af, av = auc(v, isN, isA), auc(v, isN, isV)
    rows.append(dict(피처=nm, AUC_고장=round(af, 2), 분리도=round(abs(af - .5) * 2, 2), AUC_val=round(av, 2), 이동도=round(abs(av - .5) * 2, 2), 정상중앙=round(np.median(v[isN]), 2), 고장중앙=round(np.median(v[isA]), 2), val중앙=round(np.median(v[isV]), 2)))
S = pd.DataFrame(rows)
sp = {g: dict(n0=float((n0[grp == g] == 0).mean()), n1=float((n0[grp == g] == 1).mean()), n_ge2=float((n0[grp == g] >= 2).mean())) for g in C}
H2sub = pd.Series({g: np.nan for g in C}); sub = n0 >= 2
af2 = roc_auc_score(np.r_[np.zeros((isN & sub).sum()), np.ones((isA & sub).sum())], np.r_[H2[isN & sub], H2[isA & sub]])
B = [("nor_025", "오경보"), ("nor_055", "오경보"), ("nor_059", "오경보"), ("nor_065", "오경보"), ("nor_116", "오경보"), ("abn_022", "미탐"), ("abn_005", "경계"), ("abn_002", "8.3k규칙"), ("abn_029", "8.3k규칙"), ("val_025", "val경계"), ("val_018", "val경계"), ("val_021", "val경계"), ("val_012", "val경계"), ("val_001", "val(8.3k만)"), ("val_027", "val경계")]
pct = lambda v, i: float((v[isN] <= v[i]).mean() * 100)
BT = pd.DataFrame([dict(파일=n, 구분=t, 충격수=int(n0[allnames.index(n)]), H1백분위=round(pct(H1, allnames.index(n))), H2=round(H2[allnames.index(n)], 2), H2백분위=round(pct(H2, allnames.index(n)))) for n, t in B if n in allnames])
# 상관 (학습 150, 스피어만)
others = ["H3_kurt_1-2k", "H5_kurt_4-8k", "H7_ring_1-2k", "H8_ring_2-4k", "H9_ring_4-8k", "H10_ring_8-12.5k", "H12_share_1-2k", "H13_share_2-4k", "H14_share_4-8k", "H15_share_8-12.5k", "H18_hfl", "L1_S"]
D = T[others].copy(); D["H1"] = H1[:ntr]; D["H2"] = H2[:ntr]
Cr = D.corr(method="spearman")[["H1", "H2"]].drop(["H1", "H2"]).round(2)
# 문턱 민감도
sens = []
for K in (3, 4.5, 6, 8, 12):
    nK = np.array([len(peaks(r, K)) for r in RAT]).astype(float)
    af, av = auc(nK, isN, isA), auc(nK, isN, isV)
    sens.append(dict(문턱K=K, 정상중앙=np.median(nK[isN]), 고장중앙=np.median(nK[isA]), val중앙=np.median(nK[isV]), 분리도=round(abs(af - .5) * 2, 2), 이동도=round(abs(av - .5) * 2, 2), 충격0개_정상비율=round(float((nK[isN] == 0).mean()), 2)))
SE = pd.DataFrame(sens)
fig, ax = plt.subplots(figsize=(7, 4.4)); ax.plot(SE.문턱K, SE.분리도, "o-", label="분리도 (정상 vs 고장)"); ax.plot(SE.문턱K, SE.이동도, "s-", label="val 이동도 (정상 vs val)"); ax.axvline(K_MIN, color="k", ls="--", lw=.8); ax.set_xlabel("충격 문턱 K (포락선÷바탕)"); ax.legend(); ax.grid(alpha=.2); ax.set_title("충격 수 n의 문턱 민감도 (점선 = 현재 4.5)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "3_문턱민감도.png"), dpi=115); plt.close(fig)
md = lambda df: "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n" + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in df.values)
txt = [f"정의 검증(모델 H1·H2와의 최대 차이): train H1 {chk1:.1e} · H2 {chk2:.1e} / val H1 {chk1v:.1e} · H2 {chk2v:.1e}\n", "분리도 · 이동도\n" + md(S), f"\nH2는 n≥2 파일만으로 보면 분리 AUC {af2:.2f} (n={int((isN & sub).sum())} 정상 · {int((isA & sub).sum())} 고장)",
       "\n충격 수 구성 (비율): " + str({g: {k: round(v, 2) for k, v in d.items()} for g, d in sp.items()}), "\n경계 파일의 정상 내 백분위\n" + md(BT), "\n학습 150 스피어만 상관(H1, H2 대 나머지 12개)\n" + md(Cr.reset_index()), "\n문턱 민감도\n" + md(SE)]
open(os.path.join(HERE, "결과.md"), "w", encoding="utf-8").write("\n".join(txt)); print("\n".join(txt))
