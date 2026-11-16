# %% [markdown]
# 4-3 팀원 피처 · ext150 중복 지도 — 피처를 줄이고 바꾸기 전에 그림으로 본다
# 질문 1. ext150(ET 입력 150개)은 실제로 몇 덩어리의 정보인가 (상관 군집 · VIF)
# 질문 2. 팀원 피처(11-16 1차 공유, 신효섭·권소린 보고서 정의)를 우리 데이터로 다시 계산하면 정상·고장을 가르는가
#         특히 우리 최종 모델(EXP-025)의 문제 파일(오경보 025·055·059·116, 미탐 022, 경계 005)에서 어떤가
# 질문 3. 팀원 피처는 ext150에 없는 새 정보인가 (ext150 열과의 최대 순위상관)
# - 라벨은 색칠·설명용 AUC에만 쓴다. 피처 정의는 팀원 보고서의 값을 그대로 옮겼다(대역 경계는 팀원이 정한 값 = 우리 기준 '사후')
# - AR(40) 백색화 계수는 정상 120개 전체로 맞춘다(폴드 밖, 설명용). 모델에 넣을 때는 폴드 안에서 다시 맞춰야 한다
# 실행: conda run -n qe-phm python "단계/4_피처/4-3_1116_팀원피처·중복지도/run.py"

# %% 설정
import os, sys, json
HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
ROOT = HERE
while not os.path.isdir(os.path.join(ROOT, "src")):
    ROOT = os.path.dirname(ROOT)
sys.path.insert(0, ROOT)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, find_peaks, stft, lfilter
from scipy.stats import kurtosis, spearmanr
from scipy.linalg import solve_toeplitz
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform
from sklearn.metrics import roc_auc_score
from src.exp import load_train, cached_table, feature_table_ext, vif
plt.rcParams["font.family"] = "Malgun Gothic"; plt.rcParams["axes.unicode_minus"] = False
FIG = os.path.join(HERE, "figs"); os.makedirs(FIG, exist_ok=True)
FS = 25000

X, y, names = load_train(); y = np.asarray(y); N = len(y); nor = y == 0
short = [os.path.splitext(os.path.basename(str(n)))[0] for n in names]
idx = {s: i for i, s in enumerate(short)}
FOCUS = {  # EXP-025 / EXP-030 기준 문제 파일
    "nor_025": "오경보", "nor_055": "오경보", "nor_059": "오경보", "nor_116": "오경보",
    "abn_022": "미탐", "abn_005": "경계",
    "abn_002": "8.3k 규칙", "abn_029": "8.3k 규칙", "abn_014": "유격 규칙", "abn_017": "유격 규칙",
}
FOCUS = {k: v for k, v in FOCUS.items() if k in idx}
COL = {"오경보": "#d62728", "미탐": "#9467bd", "경계": "#8c564b", "8.3k 규칙": "#2ca02c", "유격 규칙": "#17becf"}
log = []
def say(*a):
    s = " ".join(str(x) for x in a); print(s); log.append(s)

# %% 1. ext150 중복 지도
E = cached_table("ext", feature_table_ext, X)
E = E.loc[:, E.std() > 0]
R = E.rank().corr().values                                   # 순위상관 (스케일·이상값에 둔감)
D = 1 - np.abs(R); np.fill_diagonal(D, 0)
Z = linkage(squareform(D, checks=False), "average")
order = leaves_list(Z)
v = vif(E.values)
say(f"ext150 열 {E.shape[1]}개 · VIF 중앙값 {np.median(v):.0f} · VIF>10 {np.sum(v > 10)}개 · |r|>0.9 쌍 {(np.abs(np.triu(R, 1)) > 0.9).sum()}개")
for thr in (0.9, 0.8, 0.7):
    k = len(set(fcluster(Z, 1 - thr, "distance")))
    say(f"  |순위상관| ≥ {thr}로 묶으면 군집 {k}개")
fam = E.columns.str.split("_").str[0]
cl8 = fcluster(Z, 0.2, "distance")
fam_tab = pd.crosstab(pd.Series(cl8, name="군집"), pd.Series(fam.values, name="가족"))

fig, ax = plt.subplots(1, 2, figsize=(17, 8), gridspec_kw={"width_ratios": [3, 1.2]})
im = ax[0].imshow(np.abs(R)[np.ix_(order, order)], cmap="viridis", vmin=0, vmax=1)
fam_o = fam.values[order]
fams = list(dict.fromkeys(fam.values))
cmap = plt.get_cmap("tab20")
for j, f in enumerate(fam_o):
    ax[0].add_patch(plt.Rectangle((j - 0.5, -6), 1, 4, color=cmap(fams.index(f) % 20), clip_on=False))
ax[0].set_xlim(-0.5, len(order) - 0.5); ax[0].set_ylim(len(order) - 0.5, -6.5)
ax[0].set_title(f"ext150 |순위상관| (군집 순서로 정렬) — 위 색 띠 = 피처 가족\n"
                f"|r| ≥ 0.8로 묶으면 {len(set(cl8))}덩어리", loc="left")
ax[0].set_xticks([]); ax[0].set_yticks([])
plt.colorbar(im, ax=ax[0], fraction=0.03)
hand = [plt.Rectangle((0, 0), 1, 1, color=cmap(i % 20)) for i in range(len(fams))]
ax[0].legend(hand, fams, ncol=1, fontsize=8, loc="center left", bbox_to_anchor=(1.08, 0.5), title="가족")
vs = np.sort(v)[::-1]
ax[1].semilogy(np.arange(1, len(vs) + 1), vs, ".-")
ax[1].axhline(10, ls="--", c="gray"); ax[1].text(len(vs) * 0.5, 12, "VIF 10", color="gray")
ax[1].set_title(f"VIF (큰 순)\n중앙값 {np.median(v):.0f}, 10 넘는 열 {np.sum(v > 10)}/{len(v)}", loc="left")
ax[1].set_xlabel("열 순위"); ax[1].set_ylabel("VIF (log)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "1_ext150_중복지도.png"), dpi=130); plt.close(fig)

# %% 2. 팀원 피처 계산
Xc = X - X.mean(1, keepdims=True)
def filt(x, kind, f):
    return sosfiltfilt(butter(4, f, btype=kind, fs=FS, output="sos"), x, axis=-1)

XH = filt(Xc, "highpass", 200)                                # 신효섭 x_H
X1k = filt(Xc, "highpass", 1000)

# 2-1 신효섭: 포락선 충격 검출 + 감쇠정현파 사전(1–12 kHz 250 Hz × ζ 4개 = 180개)
SEG = int(0.008 * FS)                                         # 8 ms (그림 3과 같음)
t = np.arange(SEG) / FS
fgrid = np.arange(1000, 12001, 250); zgrid = [0.01, 0.02, 0.04, 0.08]
atoms = []                                                    # 위상 자유: sin·cos 짝을 직교화한 2차원 부분공간
for f in fgrid:
    for z in zgrid:
        w = 2 * np.pi * f; wd = w * np.sqrt(1 - z ** 2); a = np.exp(-z * w * t)
        Q, _ = np.linalg.qr(np.stack([a * np.sin(wd * t), a * np.cos(wd * t)], 1))
        atoms.append(Q)
atoms = np.stack(atoms)                                       # (180, SEG, 2)
afreq = np.repeat(fgrid, len(zgrid))
lp500 = butter(4, 500, btype="lowpass", fs=FS, output="sos")

def ring(i):
    env = sosfiltfilt(lp500, np.abs(hilbert(X1k[i])))
    pk, pr = find_peaks(env, height=6 * np.median(env), distance=int(0.005 * FS))
    pk = pk[pk + SEG <= X.shape[1]]
    if not len(pk):
        return pk, np.array([]), np.array([])
    S = np.stack([X1k[i, p:p + SEG] for p in pk]); S = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    fit = np.linalg.norm(np.einsum("ks,ast->kat", S, atoms), axis=2)   # (충격, 180)
    return pk, afreq[fit.argmax(1)], env[pk]

RING = [ring(i) for i in range(N)]
all_f = {0: [], 1: []}
rows = []
for i, (pk, fr, h) in enumerate(RING):
    all_f[int(y[i])] += list(fr)
    n = len(pk)
    o = dict(impact_n=np.log1p(n))
    if n:
        o.update(ring_2_5k=np.mean((fr >= 2000) & (fr < 5000)), ring_5_8k=np.mean((fr >= 5000) & (fr < 8000)),
                 ring_ge8k=np.mean(fr >= 8000), ring_f_iqr=np.subtract(*np.percentile(fr, [75, 25])),
                 ring_f_med=np.median(fr))
    rows.append(o)
T = pd.DataFrame(rows)

# 2-2 권소린: 큰 충격 20개의 직후 12 ms 대역 RMS 비, 고장 울림 대역(2.5–5 kHz) 첨도, 8–11k ÷ 5–8k
W12 = int(0.012 * FS); fr12 = np.fft.rfftfreq(W12, 1 / FS)
def band_rms(P, lo, hi):
    return np.sqrt(P[..., (fr12 >= lo) & (fr12 < hi)].sum(-1))
X10 = filt(Xc, "highpass", 10)
r25, r811, dom25 = [], [], []
for i, (pk, fr, h) in enumerate(RING):
    if not len(pk):
        r25.append(np.nan); r811.append(np.nan); dom25.append(np.nan); continue
    top = pk[np.argsort(h)[::-1][:20]]; top = top[top + W12 <= X.shape[1]]
    P = np.abs(np.fft.rfft(np.stack([X10[i, p:p + W12] for p in top]) * np.hanning(W12), axis=1)) ** 2
    b25, b58, b811 = band_rms(P, 2000, 5000), band_rms(P, 5000, 8000), band_rms(P, 8000, 11000)
    r25.append(np.log10(np.median(b25 / b58))); r811.append(np.log10(np.median(b811 / b58)))
    dom25.append(np.mean((b25 > b58) & (b25 > b811)))
T["so_ring25_over58"] = r25; T["so_ring811_over58"] = r811; T["so_ring_dom25"] = dom25
XB = filt(Xc, "bandpass", [2500, 5000])
T["so_bp25_5k_kurt"] = kurtosis(XB[:, 500:-500], axis=1, fisher=False)
T["so_bp25_5k_winkurt_max"] = [max(kurtosis(w, fisher=False) for w in np.array_split(XB[i, 500:-500], 10)) for i in range(N)]
B58 = filt(Xc, "bandpass", [5000, 8000]); B811 = filt(Xc, "bandpass", [8000, 11000])
T["so_811_over_58"] = np.log10(np.sqrt((B811 ** 2).mean(1)) / np.sqrt((B58 ** 2).mean(1)))

# 2-3 신효섭: 0.2초 구간 첨도 평균, 200 Hz–3 kHz 스펙트럼 첨도, AR(40) 백색화 잔차
T["seg_kurt_mean"] = [np.mean([kurtosis(w, fisher=False) for w in np.array_split(XH[i], 10)]) for i in range(N)]
f_s, _, Zs = stft(XH, fs=FS, nperseg=256, noverlap=192, axis=-1)
A = np.abs(Zs) ** 2                                           # (N, F, T)
SK = (A ** 2).mean(-1) / (A.mean(-1) ** 2 + 1e-30) - 2
band = (f_s >= 200) & (f_s <= 3000)
T["SK_200_3k"] = SK[:, band].max(1)
P_AR = 40; L08 = int(0.8 * FS)
ac = np.zeros(P_AR + 1)
for i in np.flatnonzero(nor):
    s = XH[i, :L08]; s = s - s.mean()
    full = np.correlate(s, s, "full")[L08 - 1:L08 + P_AR] / L08
    ac += full / full[0]
ac /= nor.sum()
a = solve_toeplitz(ac[:-1], ac[1:])
res = lfilter(np.r_[1, -a], [1], XH, axis=-1)[:, P_AR:]
T["AR_res_ratio"] = np.log10(res.var(1) / XH.var(1))
T["AR_res_kurt"] = kurtosis(res, axis=1, fisher=False)
T.index = short
T.to_csv(os.path.join(FIG, "teammate_features.csv"), encoding="utf-8-sig")
say(f"팀원 피처 {T.shape[1]}개 계산. 충격 0개 파일: 정상 {int(((T.impact_n == 0) & nor).sum())} · 고장 {int(((T.impact_n == 0) & ~nor).sum())}")

# %% 3. (11-16) 울림 주파수 재현 그림은 EDA 2-16(단계/2_EDA/유의미/2-16_1116_울림주파수/)으로 옮겼다

# %% 4. 팀원 피처: 정상·고장 분포 + 문제 파일 위치
def auc(c):
    m = ~np.isnan(c)
    a = roc_auc_score(y[m], c[m]); return max(a, 1 - a), a >= 0.5
cols = list(T.columns)
info = []
for c in cols:
    a, up = auc(T[c].values.astype(float))
    m = ~np.isnan(T[c].values.astype(float))
    rho = np.abs(np.array([spearmanr(T[c].values[m], E[e].values[m])[0] for e in E.columns]))
    j = np.nanargmax(rho)
    info.append(dict(feature=c, AUC=a, 고장쪽=("큼" if up else "작음"), ext최대상관=rho[j], 가장비슷한ext=E.columns[j]))
info = pd.DataFrame(info).set_index("feature")
ncol = 4; nrow = int(np.ceil(len(cols) / ncol))
fig, axs = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.3 * nrow)); axs = axs.ravel()
rng = np.random.default_rng(0)
for ax, c in zip(axs, cols):
    v = T[c].values.astype(float)
    for k, (lab, colr) in enumerate(((0, "#4472c4"), (1, "#ed7d31"))):
        m = (y == lab) & ~np.isnan(v)
        ax.scatter(k + rng.uniform(-0.18, 0.18, m.sum()), v[m], s=9, alpha=0.45, color=colr)
    lo, hi = np.nanpercentile(v[nor], [5, 95]); ax.axhspan(lo, hi, color="#4472c4", alpha=0.08)
    for f, kind in FOCUS.items():
        i = idx[f]
        if np.isnan(v[i]): continue
        ax.scatter(int(y[i]), v[i], s=55, facecolors="none", edgecolors=COL[kind], lw=1.6)
        ax.annotate(f[-3:], (int(y[i]) + 0.22, v[i]), fontsize=7, color=COL[kind], va="center")
    r = info.loc[c]
    ax.set_title(f"{c}\nAUC {r.AUC:.2f} (고장 {r.고장쪽}) · ext 최대 ρ {r.ext최대상관:.2f}", fontsize=9, loc="left")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["정상", "고장"]); ax.set_xlim(-0.5, 1.8)
for ax in axs[len(cols):]: ax.axis("off")
hand = [plt.Line2D([], [], marker="o", ls="", mfc="none", mec=c, label=k) for k, c in COL.items()]
fig.legend(handles=hand, loc="lower right", ncol=5, fontsize=9, title="EXP-025 문제 파일 (번호)")
fig.suptitle("팀원 피처를 우리 데이터로 계산 — 파란 띠 = 정상 5–95%, AUC는 설명용(사후)", x=0.01, ha="left")
fig.tight_layout(rect=(0, 0.04, 1, 0.97)); fig.savefig(os.path.join(FIG, "3_팀원피처_분포.png"), dpi=120); plt.close(fig)

# %% 5. 새 정보인가: ext150과의 최대 순위상관
o = info.sort_values("AUC", ascending=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 0.42 * len(o) + 1.5), sharey=True)
ax[0].barh(o.index, o.AUC, color="#888"); ax[0].axvline(0.5, c="k", lw=0.8); ax[0].set_xlim(0.45, 1)
ax[0].set_title("단변량 AUC (설명용)", loc="left")
c2 = ["#d62728" if r < 0.7 else "#ff7f0e" if r < 0.85 else "#7f7f7f" for r in o.ext최대상관]
ax[1].barh(o.index, o.ext최대상관, color=c2); ax[1].set_xlim(0, 1)
for k, (r, e) in enumerate(zip(o.ext최대상관, o.가장비슷한ext)):
    ax[1].text(min(r, 0.98) - 0.01, k, e, ha="right", va="center", fontsize=7, color="white")
ax[1].set_title("ext150 열과의 최대 |순위상관| (빨강 < 0.7 = 새 정보, 회색 ≥ 0.85 = 이미 있음)", loc="left")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "4_새정보_여부.png"), dpi=130); plt.close(fig)

# %% 6. 문제 파일 표 (정상 120개 중 백분위)
pct = pd.DataFrame({c: [100 * np.nanmean(T[c].values[nor] <= T[c].values[idx[f]]) if not np.isnan(T[c].values[idx[f]]) else np.nan
                        for f in FOCUS] for c in cols}, index=[f"{f} ({k})" for f, k in FOCUS.items()]).round(0)
pct.to_csv(os.path.join(FIG, "problem_file_percentile.csv"), encoding="utf-8-sig")
info.round(3).to_csv(os.path.join(FIG, "teammate_feature_info.csv"), encoding="utf-8-sig")
say("\n팀원 피처 요약 (AUC 큰 순)"); say(info.sort_values("AUC", ascending=False).round(3).to_string())
say("\n문제 파일의 정상 내 백분위 (고장 쪽이 '큼'이면 100에 가까울수록 고장처럼 보임)"); say(pct.T.to_string())
say("\next150 군집(|r|≥0.8) × 가족 표"); say(fam_tab.to_string())
with open(os.path.join(FIG, "summary.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(log))
